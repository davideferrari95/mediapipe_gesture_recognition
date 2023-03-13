#!/usr/bin/env python3

import os, cv2, pickle
import rospy, rospkg 
import numpy as np
from termcolor import colored

import mediapipe
from mediapipe_gesture_recognition.msg import Keypoint, Hand, Pose, Face

class MediapipeDatasetProcess:

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
    rospy.init_node('mediapipe_dataset_processor_node', anonymous=True, disable_signals= True)

    # Get Package Path - Get Dataset Folder
    self.package_path = rospkg.RosPack().get_path('mediapipe_gesture_recognition')
    self.DATASET_PATH = os.path.join(self.package_path, 'dataset/Video')
    
    # Read Mediapipe Modules Parameters
    self.enable_right_hand = rospy.get_param('enable_right_hand', False)
    self.enable_left_hand  = rospy.get_param('enable_left_hand', False)
    self.enable_pose       = rospy.get_param('enable_pose', False)
    self.enable_face       = rospy.get_param('enable_face', False)

    # Select Gesture File
    self.gesture_enabled_folder = ''
    if self.enable_right_hand: self.gesture_enabled_folder += 'Right'
    if self.enable_left_hand:  self.gesture_enabled_folder += 'Left'
    if self.enable_pose:       self.gesture_enabled_folder += 'Pose'
    if self.enable_face:       self.gesture_enabled_folder += 'Face'

    # Debug Print
    print(colored(f'\nFunctions Enabled:\n', 'yellow'))
    print(colored(f'  Right Hand: {self.enable_right_hand}',  'green' if self.enable_right_hand else 'red'))
    print(colored(f'  Left  Hand: {self.enable_left_hand}\n', 'green' if self.enable_left_hand  else 'red'))
    print(colored(f'  Skeleton:   {self.enable_pose}',        'green' if self.enable_pose else 'red'))
    print(colored(f'  Face Mesh:  {self.enable_face}\n',      'green' if self.enable_face else 'red'))

    # Initialize Mediapipe:
    self.mp_drawing = mediapipe.solutions.drawing_utils
    self.mp_drawing_styles = mediapipe.solutions.drawing_styles
    self.mp_holistic = mediapipe.solutions.holistic

    # Initialize Mediapipe Holistic
    self.holistic = self.mp_holistic.Holistic(refine_face_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # TODO: Initialise the Walk
    self.gesture_name = ''
    self.video_number = ''

    #Initialize the load variable 
    self.last_gesture = ''
    self.last_video = ''
    self.progress_file = "/home/alberto/catkin_ws/src/mediapipe_gesture_recognition/useful_scripts/progress.txt"

    
    #All keypoint in all the frame in all the videos for each gesture   
    self.framekeypoints = []       #It will be [[n_videos],[n_frames],[n_keypoints]]

  def newKeypoint(self, landmark, number, name):

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

  def processHand(self, RightLeft, handResults, image):

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
    hand_msg.header.stamp = rospy.Time.now()
    hand_msg.header.frame_id = 'Hand Right Message' if RightLeft else 'Hand Left Message'
    hand_msg.right_or_left = hand_msg.RIGHT if RightLeft else hand_msg.LEFT

    if (((RightLeft == self.RIGHT_HAND) and (handResults.right_hand_landmarks))
     or ((RightLeft == self.LEFT_HAND)  and (handResults.left_hand_landmarks))):

      # Add Keypoints to Hand Message
      for i in range(len(handResults.right_hand_landmarks.landmark if RightLeft else handResults.left_hand_landmarks.landmark)):

        # Append Keypoint
        hand_msg.keypoints.append(self.newKeypoint(handResults.right_hand_landmarks.landmark[i] if RightLeft else handResults.left_hand_landmarks.landmark[i],
                                                   i, self.hand_landmarks_names[i]))

      # Return Hand Keypoint Message
      return hand_msg

  def processPose(self, poseResults, image):

    ''' Process Pose Keypoints '''

    # Drawing the Pose Landmarks
    self.mp_drawing.draw_landmarks(
      image,
      poseResults.pose_landmarks,
      self.mp_holistic.POSE_CONNECTIONS,
      landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

    # Create Pose Message
    pose_msg = Pose()
    pose_msg.header.stamp = rospy.Time.now()
    pose_msg.header.frame_id = 'Pose Message'

    if poseResults.pose_landmarks:

      # Add Keypoints to Pose Message
      for i in range(len(poseResults.pose_landmarks.landmark)):

        # Append Keypoint
        pose_msg.keypoints.append(self.newKeypoint(poseResults.pose_landmarks.landmark[i], i, self.pose_landmarks_names[i]))

      # Return Pose Keypoint Message
      return pose_msg

  def processFace(self, faceResults, image):

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
    face_msg.header.stamp = rospy.Time.now()
    face_msg.header.frame_id = 'Face Message'

    if faceResults.face_landmarks:

      # Add Keypoints to Face Message
      for i in range(len(faceResults.face_landmarks.landmark)):

        # Assign Keypoint Coordinates
        new_keypoint = Keypoint()
        new_keypoint.x = faceResults.face_landmarks.landmark[i].x
        new_keypoint.y = faceResults.face_landmarks.landmark[i].y
        new_keypoint.z = faceResults.face_landmarks.landmark[i].z

        # Assign Keypoint Number
        new_keypoint.keypoint_number = i

        # Assign Keypoint Name (468 Landmarks -> Names = FACE_KEYPOINT_1 ...)
        new_keypoint.keypoint_name = f'FACE_KEYPOINT_{i+1}'

        # Append Keypoint
        face_msg.keypoints.append(new_keypoint)

      # Return Face Message
      return face_msg

  def flattenKeypoints(self, pose_msg, left_msg, right_msg, face_msg):

    '''
    Flatten Incoming Messages or Create zeros Vector \n
    Concatenate each Output
    '''

    # Check if Messages are Available and Create Zeros Vectors if Not
    pose    = np.zeros(33 * 4)  if pose_msg  is None else np.array([[res.x, res.y, res.z, res.v] for res in pose_msg.keypoints]).flatten()
    left_h  = np.zeros(21 * 4)  if left_msg  is None else np.array([[res.x, res.y, res.z, res.v] for res in left_msg.keypoints]).flatten()
    right_h = np.zeros(21 * 4)  if right_msg is None else np.array([[res.x, res.y, res.z, res.v] for res in right_msg.keypoints]).flatten()
    face    = np.zeros(468 * 3) if face_msg  is None else np.array([[res.x, res.y, res.z, res.v] for res in face_msg.keypoints]).flatten()

    # Concatenate Data
    return np.concatenate([right_h, left_h, pose, face])

  def processResults(self, image):

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

  def savevideoFrame(self, gesture, keypoints_sequence):

    ''' Data Save Functions '''


    '''
    Loop to Save the Landmarks Coordinates for Each Frame of Each Video
    The Loop Continue Until the Service Response Keep Itself True with a 30FPS
    '''
    


    # Create a Gesture Folder 
    os.makedirs(os.path.join(f'{self.package_path}/database/3D_Gestures/{self.gesture_enabled_folder}/', gesture), exist_ok=True)
    
    gesture_file = os.path.join(f'{self.package_path}/database/3D_Gestures/{self.gesture_enabled_folder}/', gesture, str('load_file.pkl'))


    if os.path.exists(gesture_file):

      # Load the File
      load_file = pickle.load(open(gesture_file, 'rb'))

      # Append the New Keypoints Sequence
      load_file.append(keypoints_sequence)

      # Save the Updated File
      pickle.dump(load_file, open(gesture_file, 'wb'))
    else:
      
      # Save the New Keypoints Sequence
      pickle.dump([keypoints_sequence], open(gesture_file, 'wb'))
    


  def processDataset(self):

    with open(self.progress_file, "r") as f:
               
     lines = f.readlines()
     #Load the last gesture name
     last_gesture = str(lines[0].split(",")[0])
     last_video = str(lines[0].split(",")[1]) + ".avi"

    print('Starting Collection', "Gesture: ", last_gesture, "| Video:  ", last_video)
      
    try:
        #Read every gesture folder 
      for folder in sorted(os.listdir(self.DATASET_PATH)):

          #First check 
           if folder >= last_gesture:

            #Save the folder path
            folder_path = os.path.join(self.DATASET_PATH, folder)

            #Read every video in the gesture folder 
            for video in sorted(os.listdir(folder_path)):

                #Second check
                if video > last_video:

                  last_video = ""

                  # Get the Full Path of the Video for Each Gesture
                  video_path = os.path.join(folder_path, video)

                  # Get the Gesture Name and the Video Number
                  self.gesture_name = os.path.splitext(folder)[0]
                  self.video_number = os.path.splitext(video)[0]

                  #print("Folder: ", self.gesture_name, "| Video: ", self.video_number)
                  if not video.endswith(('.mp4', '.avi', '.mov')):
                    continue
                  
                  # Process the Video
                  video_sequence = self.process_video(video_path)

                  video_sequence = np.array(video_sequence)

                  #Zero-padding 
                  pad_amt = [(0, 40 - video_sequence.shape[0]), (0, 300 - video_sequence.shape[1])]
                  
                  padded_sequence = np.pad(video_sequence, pad_amt, 'constant', constant_values=0)

                  print(f'Sequences Shape: {np.array(padded_sequence).shape}')

                  # Save the Frame
                  self.savevideoFrame(self.gesture_name, padded_sequence)

                  # Print Finish of the Video
                  print(f'Video {video} of {folder} Processed')
                
                #Traceback
                with open(self.progress_file, 'w') as f:
                  f.write(str(folder)+ "," + str(os.path.splitext(video)[0]))
      
      # Print Finish of All Videos
      print('All Video Processed')      

    #If you press Ctrl+C stop the video flow
    except KeyboardInterrupt:
      
      print("\n\n Keyboard Interrupt \n\n")
      with open(self.progress_file, 'w') as f:
        f.write(str(folder)+ "," + str(os.path.splitext(video)[0]))

    
  # Definisci la funzione per elaborare un singolo video
  def process_video(self, video_path):
      
    # Open the Video
    self.cap = cv2.VideoCapture(video_path)

    video_sequence = []

    # Loop Through the Video Frames
    while self.cap.isOpened() and not rospy.is_shutdown():
               
      # Read the Frame
      ret, image = self.cap.read()

      # Check if the Frame is Available
      if not ret:
         print('Ignoring empty camera frame')
         break 
                
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

      # Show and Flip the Image Horizontally for a Selfie-View Display.
      cv2.imshow('MediaPipe Landmarks', cv2.flip(image, 1))
      if cv2.waitKey(5) & 0xFF == 27:
        break

    # Close Video Cap
    self.cap.release()

    return video_sequence


if __name__ == '__main__':

  # Create Mediapipe Dataset Process Class
  MediapipeProcess = MediapipeDatasetProcess()
  
  # Mediapipe Dataset Process Function
  MediapipeProcess.processDataset()


  


    
