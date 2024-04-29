#!/usr/bin/env python3

import os, sys, time, cv2, pickle
import numpy as np
from tqdm import tqdm, tqdm_gui

from natsort import natsorted
from termcolor import colored
from typing import List, Optional

# Import ROS2 Libraries
import rclpy
from rclpy.node import Node

# Import Parent Folders
from pathlib import Path

import rclpy.time
sys.path.append(f'{str(Path(__file__).resolve().parents[1])}/utils')

# Import Mediapipe
from mediapipe.python.solutions import holistic, drawing_styles, drawing_utils
from mediapipe_gesture_recognition.msg import Keypoint, Hand, Pose, Face
from mediapipe_types import HolisticResults, Landmark

# Import Zero Padding Class
from utilities.zero_padding import ZeroPadding

class MediapipeDatasetProcess(Node):

    """ Dataset:

    1 Pickle File (.pkl) for each Gesture
    Each Pickle File contains a Number of Videos Representing the Gesture
    Each Video is Represented by a Sequence of 3D Keypoints (x,y,z,v) for each Frame of the Video

    Dataset Structure:

        - Array of Sequences (Videos): (Number of Sequences / Videos, Sequence Length, Number of Keypoints (Flattened Array of 3D Coordinates x,y,z,v))
        - Size: (N Video, N Frames, N Keypoints) -> (1000+, 85, 300) or (1000+, 85, 2212)

        Frames: 85 (Fixed) | Keypoints (300 or 2212):

        Right Hand: 21 * 4  = 84
        Left  Hand: 21 * 4  = 84
        Pose:       33 * 4  = 132
        Face:       478 * 4 = 1912

    """

    """ Mediapipe Holistic

        Returns:
        A NamedTuple with fields describing the landmarks on the most prominent person detected:
            1) "pose_landmarks" field that contains the pose landmarks.
            2) "pose_world_landmarks" field that contains the pose landmarks in real-world 3D coordinates that are in meters with the origin at the center between hips.
            3) "left_hand_landmarks" field that contains the left-hand landmarks.
            4) "right_hand_landmarks" field that contains the right-hand landmarks.
            5) "face_landmarks" field that contains the face landmarks.
            6) "segmentation_mask" field that contains the segmentation mask if "enable_segmentation" is set to true.

    """

    # Constants
    RIGHT_HAND, LEFT_HAND = True, False

    # Define Hand Landmark Names
    hand_landmarks_names = ['WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP',
                            'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP',
                            'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_MCP',
                            'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP', 'PINKY_MCP',
                            'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']

    # Define Pose Landmark Names
    pose_landmarks_names = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER',
                            'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
                            'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST',
                            'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB',
                            'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE',
                            'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']

    def __init__(self):

        # ROS Initialization
        super().__init__('mediapipe_dataset_processor_node')

        # Declare ROS2 Parameters
        self.declare_parameter('enable_right_hand', True)
        self.declare_parameter('enable_left_hand', True)
        self.declare_parameter('enable_pose', True)
        self.declare_parameter('enable_face', False)
        self.declare_parameter('debug', False)

        # Read Mediapipe Modules Parameters
        self.enable_right_hand = self.get_parameter('enable_right_hand').get_parameter_value().bool_value
        self.enable_left_hand  = self.get_parameter('enable_left_hand').get_parameter_value().bool_value
        self.enable_pose       = self.get_parameter('enable_pose').get_parameter_value().bool_value
        self.enable_face       = self.get_parameter('enable_face').get_parameter_value().bool_value
        self.debug             = self.get_parameter('debug').get_parameter_value().bool_value

        # Select Gesture File - Compute KeyPoints Number
        self.gesture_enabled_folder, self.keypoint_number = '', 0
        if self.enable_right_hand: self.gesture_enabled_folder += 'Right'; self.keypoint_number += 84
        if self.enable_left_hand:  self.gesture_enabled_folder += 'Left';  self.keypoint_number += 84
        if self.enable_pose:       self.gesture_enabled_folder += 'Pose';  self.keypoint_number += 132
        if self.enable_face:       self.gesture_enabled_folder += 'Face';  self.keypoint_number += 1912

        # Get Package Path - Get Dataset Folder
        self.package_path    = str(Path(__file__).resolve().parents[2])
        self.DATASET_PATH    = os.path.join(self.package_path, r'dataset/Gestures')
        self.gesture_path    = os.path.join(self.package_path, r'data/3D_Gestures', self.gesture_enabled_folder)
        self.checkpoint_file = os.path.join(self.gesture_path, 'Video Checkpoint.txt')

        # Create the Processed Gesture Data Folder
        os.makedirs(self.gesture_path, exist_ok=True)

        # Create Progress File if Not Exist
        if not os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, "w") as f: f.write(',')

        # Debug Print
        print(colored(f'\nFunctions Enabled:\n', 'yellow'))
        print(colored(f'  Right Hand: {self.enable_right_hand}',  'green' if self.enable_right_hand else 'red'))
        print(colored(f'  Left  Hand: {self.enable_left_hand}\n', 'green' if self.enable_left_hand  else 'red'))
        print(colored(f'  Skeleton:   {self.enable_pose}',        'green' if self.enable_pose else 'red'))
        print(colored(f'  Face Mesh:  {self.enable_face}\n',      'green' if self.enable_face else 'red'))

        # Initialize Mediapipe
        self.mp_drawing, self.mp_drawing_styles = drawing_utils, drawing_styles
        self.mp_holistic = holistic

        # Initialize Mediapipe Holistic
        self.holistic = self.mp_holistic.Holistic(refine_face_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        time.sleep(1)

    def newKeypoint(self, landmark:Landmark, number:int, name:str) -> Keypoint:

        ''' New Keypoint Creation Utility Function '''

        # Assign Keypoint Coordinates
        new_keypoint = Keypoint()
        new_keypoint.x = landmark.x
        new_keypoint.y = landmark.y
        new_keypoint.z = landmark.z
        new_keypoint.v = landmark.visibility

        # Assign Keypoint Number and Name
        new_keypoint.keypoint_number = number
        new_keypoint.keypoint_name = name

        return new_keypoint

    def processHand(self, RightLeft:bool, handResults:HolisticResults, image:cv2.typing.MatLike) -> Optional[Hand]:

        ''' Process Hand Keypoints '''

        # Drawing the Hand Landmarks
        self.mp_drawing.draw_landmarks(
        image,
        handResults.right_hand_landmarks if RightLeft else handResults.left_hand_landmarks,
        self.mp_holistic.HAND_CONNECTIONS,
        self.mp_drawing_styles.get_default_hand_landmarks_style(),
        self.mp_drawing_styles.get_default_hand_connections_style())

        # Create Hand Message
        hand_msg = Hand()
        hand_msg.header.stamp = self.get_clock().now().to_msg()
        hand_msg.header.frame_id = 'Hand Right Message' if RightLeft else 'Hand Left Message'
        hand_msg.right_or_left = hand_msg.RIGHT if RightLeft else hand_msg.LEFT

        if (((RightLeft == self.RIGHT_HAND) and (handResults.right_hand_landmarks))
         or ((RightLeft == self.LEFT_HAND)  and (handResults.left_hand_landmarks))):

            # Add Keypoints to Hand Message
            for i in range(len(handResults.right_hand_landmarks.landmark if RightLeft else handResults.left_hand_landmarks.landmark)):

                # Append Keypoint
                hand_msg.keypoints.append(self.newKeypoint(handResults.right_hand_landmarks.landmark[i] if RightLeft else handResults.left_hand_landmarks.landmark[i], i, self.hand_landmarks_names[i]))

            # Return Hand Keypoint Message
            return hand_msg

    def processPose(self, poseResults:HolisticResults, image:cv2.typing.MatLike) -> Optional[Pose]:

        ''' Process Pose Keypoints '''

        # Drawing the Pose Landmarks
        self.mp_drawing.draw_landmarks(
            image,
            poseResults.pose_landmarks,
            self.mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

        # Create Pose Message
        pose_msg = Pose()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'Pose Message'

        if poseResults.pose_landmarks:

            # Add Keypoints to Pose Message
            for i in range(len(poseResults.pose_landmarks.landmark)):

                # Append Keypoint
                pose_msg.keypoints.append(self.newKeypoint(poseResults.pose_landmarks.landmark[i], i, self.pose_landmarks_names[i]))

            # Return Pose Keypoint Message
            return pose_msg

    def processFace(self, faceResults:HolisticResults, image:cv2.typing.MatLike) -> Optional[Face]:

        ''' Process Face Keypoints '''

        # Drawing the Face Landmarks
        self.mp_drawing.draw_landmarks(
            image,
            faceResults.face_landmarks,
            self.mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())

        # Create Face Message
        face_msg = Face()
        face_msg.header.stamp = self.get_clock().now().to_msg()
        face_msg.header.frame_id = 'Face Message'

        if faceResults.face_landmarks:

            # Add Keypoints to Face Message
            for i in range(len(faceResults.face_landmarks.landmark)):

                # Assign Keypoint Coordinates
                new_keypoint = Keypoint()
                new_keypoint.x = faceResults.face_landmarks.landmark[i].x
                new_keypoint.y = faceResults.face_landmarks.landmark[i].y
                new_keypoint.z = faceResults.face_landmarks.landmark[i].z
                new_keypoint.v = 0

                # Assign Keypoint Number
                new_keypoint.keypoint_number = i

                # Assign Keypoint Name (468 Landmarks -> Names = FACE_KEYPOINT_1 ...)
                new_keypoint.keypoint_name = f'FACE_KEYPOINT_{i+1}'

                # Append Keypoint
                face_msg.keypoints.append(new_keypoint)

            # Return Face Message
            return face_msg

    def flattenKeypoints(self, pose_msg:Pose, left_msg:Hand, right_msg:Hand, face_msg:Face) -> np.ndarray:

        '''
        Flatten Incoming Messages or Create zeros Vector \n
        Concatenate each Output
        '''

        # Check if Messages are Available and Create Zeros Vectors if Not
        pose    = np.zeros(33 * 4)  if pose_msg  is None else np.array([[res.x, res.y, res.z, res.v] for res in pose_msg.keypoints]).flatten()
        left_h  = np.zeros(21 * 4)  if left_msg  is None else np.array([[res.x, res.y, res.z, res.v] for res in left_msg.keypoints]).flatten()
        right_h = np.zeros(21 * 4)  if right_msg is None else np.array([[res.x, res.y, res.z, res.v] for res in right_msg.keypoints]).flatten()
        face    = np.zeros(478 * 4) if face_msg  is None else np.array([[res.x, res.y, res.z, res.v] for res in face_msg.keypoints]).flatten()

        # Concatenate Data
        return np.concatenate([right_h, left_h, pose, face])

    def processResults(self, image:cv2.typing.MatLike) -> np.ndarray:

        ''' Process the Image to Obtain a Flattened Keypoint Sequence of the Frame '''

        # Instance the ROS Hand, Pose, Face Messages
        left_hand, right_hand, pose, face = Hand(), Hand(), Pose(), Face()

        # Process Left Hand Landmarks
        if self.enable_left_hand: left_hand = self.processHand(self.LEFT_HAND,  self.holistic_results, image)

        # Process Right Hand Landmarks
        if self.enable_right_hand: right_hand = self.processHand(self.RIGHT_HAND, self.holistic_results, image)

        # Process Pose Landmarks
        if self.enable_pose: pose = self.processPose(self.holistic_results, image)

        # Process Face Landmarks
        if self.enable_face: face = self.processFace(self.holistic_results, image)

        # Flatten All the Keypoints
        sequence = self.flattenKeypoints(pose, left_hand, right_hand, face)

        # Return the Flattened Keypoints Sequence
        return sequence

    def saveProcessedVideo(self, gesture:str, keypoints_sequence:np.ndarray):

        ''' Data Save Functions in a Common Gesture .pkl File'''

        # Gesture SaveFile
        gesture_savefile = os.path.join(f'{self.package_path}/data/3D_Gestures/{self.gesture_enabled_folder}', f'{gesture}.pkl')

        # Check if SaveFile Exist
        if os.path.exists(gesture_savefile):

            # Load the File
            load_file:List[np.ndarray] = pickle.load(open(gesture_savefile, 'rb'))

            # Append the New Keypoints Sequence
            load_file.append(keypoints_sequence)

            # Save the Updated File
            pickle.dump(load_file, open(gesture_savefile, 'wb'))

        else:

            # Save the New Keypoints Sequence
            pickle.dump([keypoints_sequence], open(gesture_savefile, 'wb'))

    def processDataset(self):

        ''' Read Videos from the Dataset Folder and Process Them with Mediapipe '''

        with open(self.checkpoint_file, "r") as f:

            lines = f.readlines()

            # Load the Last Gesture Name
            last_gesture = str(lines[0].split(",")[0])
            last_video = int(str(lines[0].split(",")[1])) if str(lines[0].split(",")[1]) != '' else 0

        if last_gesture == '': print(colored('\nStarting Dataset Processing\n', 'green'))
        else: print(colored('\nResuming Dataset Processing', 'green'), f' | Gesture: "{last_gesture}" | Video: {last_video}\n')

        try:

            # Loop Over Every Gesture Folder
            for folder in sorted(os.listdir(self.DATASET_PATH)):

                # Ignore Already Processed Gestures
                if folder >= last_gesture:

                    # TQDM Progress Bar
                    progress_bar = tqdm_gui(total=len(os.listdir(os.path.join(self.DATASET_PATH, folder))))

                    # Read Every Video in the Gesture Folder
                    for video in natsorted(os.listdir(os.path.join(self.DATASET_PATH, folder))):

                        # Update Progress Bar
                        progress_bar.set_description(f'Folder: {os.path.splitext(folder)[0]} | Video: {video}')
                        progress_bar.update(1)

                        # Ignore Already Processed Videos
                        if int(video.split(".")[0]) >= int(last_video):

                            # Get the Full Path of the Video for Each Gesture
                            video_path = os.path.join(self.DATASET_PATH, folder, video)

                            # Get the Gesture Name and the Video Number
                            self.gesture_name = os.path.splitext(folder)[0]
                            self.video_number = os.path.splitext(video)[0]

                            # Ignore Non-Video Files
                            if not video.endswith(('.mp4', '.avi', '.mov')):
                                continue

                            # Process the Video
                            video_sequence = np.array(self.processVideo(video_path))

                            # Save the Processed Video
                            self.saveProcessedVideo(self.gesture_name, video_sequence)

                        # Traceback - Update Checkpoint
                        with open(self.checkpoint_file, 'w') as f:
                            f.write(str(folder)+ "," + str(os.path.splitext(video)[0]))

                    # Print Finish of the Gesture Folder
                    progress_bar.close()
                    print(colored(f'\nGesture: "{self.gesture_name}"', 'green'), ' | All Video Processed and Saved\n')

                    # Reset Last Video (Otherwise also Next Gesture Folder starts from `last_video`)
                    last_video = 0

            # Print Finish of All Videos
            print('\nAll Video Processed and Saved\n')

            # Remove Video Checkpoint File
            os.remove(self.checkpoint_file)

            # Zero Padding - Create Zero-Padded Sequences
            print(colored('Zero Padding the Sequences\n', 'green'))
            ZeroPadding(self.gesture_path, self.keypoint_number, overwrite=True)
            print(colored('Zero Padding Completed\n', 'green'))

        # Ctrl+C -> Stop the Video Flow
        except KeyboardInterrupt:

            print("\n\nKeyboard Interrupt\n\n")

            with open(self.checkpoint_file, 'w') as f:
                f.write(str(folder)+ "," + str(os.path.splitext(video)[0]))

    def processVideo(self, video_path:str) -> List[np.ndarray]:

        ''' Function to Process a Video with Mediapipe '''

        # Open the Video
        self.cap = cv2.VideoCapture(video_path)

        video_sequence = []

        # Loop Through the Video Frames
        while self.cap.isOpened() and rclpy.ok():

            # Read the Frame
            ret, image = self.cap.read()

            # Check if the Frame is Available
            if not ret: break

            # To Improve Performance -> Process the Image as Not-Writeable
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Get Holistic Results from Mediapipe Holistic
            self.holistic_results = self.holistic.process(image)

            # To Draw the Annotations -> Set the Image Writable
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Process Mediapipe Results
            sequence = self.processResults(image)

            video_sequence.append(sequence)

            # Show and Flip the Image Horizontally for a Selfie-View Display
            cv2.imshow('MediaPipe Landmarks', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break

        # Close Video Cap
        self.cap.release()

        return video_sequence

if __name__ == '__main__':

    # ROS Initialization
    rclpy.init()

    # Create Mediapipe Dataset Process Class
    MediapipeProcess = MediapipeDatasetProcess()

    # Mediapipe Dataset Process Function
    MediapipeProcess.processDataset()
