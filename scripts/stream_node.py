#!/usr/bin/env python3

import rospy, cv2
import numpy as np
from termcolor import colored
from typing import Optional

# Import MediaPipe
from mediapipe.python.solutions import drawing_utils, drawing_styles, holistic, face_detection, objectron
from mediapipe_gesture_recognition.msg import Pose, Face, Keypoint, Hand

# Import Mediapipe Dataclasses
from utils.mediapipe_types import NormalizedLandmark, HolisticResults, DetectionResult, ObjectronResults

'''
To Obtain The Available Cameras:

    v4l2-ctl --list-devices

Intel(R) RealSense(TM) Depth Ca (usb-0000:00:14.0-2):
    /dev/video2
    /dev/video3
    /dev/video4 -> Black & White
    /dev/video5
    /dev/video6 -> RGB
    /dev/video7

VGA WebCam: VGA WebCam (usb-0000:00:14.0-5):
    /dev/video0 -> RGB
    /dev/video1
'''

class MediapipeStreaming:

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

    # Define Objectron Model Names
    available_objectron_models = ['Shoe', 'Chair', 'Cup', 'Camera']

    def __init__(self):

        # ROS Initialization
        rospy.init_node('mediapipe_stream_node', anonymous=True)
        self.ros_rate = rospy.Rate(30)

        # Read MediaPipe Modules Parameters (Available Objectron Models = ['Shoe', 'Chair', 'Cup', 'Camera'])
        self.enable_right_hand     = rospy.get_param('mediapipe_gesture_recognition/enable_right_hand', True)
        self.enable_left_hand      = rospy.get_param('mediapipe_gesture_recognition/enable_left_hand', True)
        self.enable_pose           = rospy.get_param('mediapipe_gesture_recognition/enable_pose', True)
        self.enable_face           = rospy.get_param('mediapipe_gesture_recognition/enable_face', False)
        self.enable_face_detection = rospy.get_param('mediapipe_gesture_recognition/enable_face_detection', False)
        self.enable_objectron      = rospy.get_param('mediapipe_gesture_recognition/enable_objectron', False)
        self.objectron_model       = rospy.get_param('mediapipe_gesture_recognition/objectron_model', 'Shoe')

        # Read Webcam Parameter -> Webcam / RealSense Input
        self.webcam    = rospy.get_param('mediapipe_gesture_recognition/webcam', 0)
        self.realsense = rospy.get_param('mediapipe_gesture_recognition/realsense', False)

        # Debug Print
        print(colored(f'\nFunctions Enabled:\n', 'yellow'))
        print(colored(f'  Right Hand: {self.enable_right_hand}',  'green' if self.enable_right_hand else 'red'))
        print(colored(f'  Left  Hand: {self.enable_left_hand}\n', 'green' if self.enable_left_hand  else 'red'))
        print(colored(f'  Skeleton:   {self.enable_pose}',        'green' if self.enable_pose else 'red'))
        print(colored(f'  Face Mesh:  {self.enable_face}\n',      'green' if self.enable_face else 'red'))
        print(colored(f'  Objectron:       {self.enable_objectron}',        'green' if self.enable_objectron      else 'red'))
        print(colored(f'  Face Detection:  {self.enable_face_detection}\n', 'green' if self.enable_face_detection else 'red'))

        # Check Objectron Model
        if not self.objectron_model in self.available_objectron_models:
            rospy.logerr('ERROR: Objectron Model Not Available | Shutting Down...')
            rospy.signal_shutdown('ERROR: Objectron Model Not Available')

        # MediaPipe Publishers
        self.hand_right_pub = rospy.Publisher('/mediapipe_gesture_recognition/right_hand', Hand, queue_size=1)
        self.hand_left_pub  = rospy.Publisher('/mediapipe_gesture_recognition/left_hand', Hand, queue_size=1)
        self.pose_pub       = rospy.Publisher('/mediapipe_gesture_recognition/pose', Pose, queue_size=1)
        self.face_pub       = rospy.Publisher('/mediapipe_gesture_recognition/face', Face, queue_size=1)

        # Initialize Webcam
        if not self.realsense:

            # Open Video Webcam
            self.cap = cv2.VideoCapture(self.webcam)

            # Check Webcam Availability
            if self.cap is None or not self.cap.isOpened():
                rospy.logerr(f'ERROR: Webcam {self.webcam} Not Available | Starting Default: 0')
                self.cap = cv2.VideoCapture(0)

        # Initialize Intel RealSense
        else: self.initRealSense()

        # Initialize MediaPipe Solutions (Holistic, Face Detection, Objectron)
        self.initMediaPipeSolutions()

    def initRealSense(self):

        # Import RealSense Library
        import pyrealsense2 as rs

        # Initialize Intel RealSense
        self.realsense_ctx = rs.context()
        self.connected_devices = []

        # Search for Connected Devices
        for i in range(len(self.realsense_ctx.devices)):
            detected_camera = self.realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
            self.connected_devices.append(detected_camera)

        # Using One Camera -> Select Last Device on the List
        self.device = self.connected_devices[-1]

        # Initialize Pipeline
        self.pipeline = rs.pipeline()
        self.config   = rs.config()

        # Enable Realsense Streams
        self.config.enable_device(self.device)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start Pipeline
        self.profile = self.pipeline.start(self.config)

        # Align Depth to Color
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        # Get Depth Scale
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()

        # Set Clipping Distance -> 2 Meters Depth -> Behind that the Camera See Nothing
        self.clipping_distance_in_meters = 2
        self.clipping_distance = self.clipping_distance_in_meters / self.depth_scale

    def initMediaPipeSolutions(self):

        """ Initialize MediaPipe Solutions """

        # Initialize MediaPipe:
        self.mp_drawing, self.mp_drawing_styles = drawing_utils, drawing_styles
        self.mp_holistic, self.mp_face_detection, self.mp_objectron = holistic, face_detection, objectron

        # Read Holistic Parameters
        static_image_mode        = rospy.get_param('/holistic/static_image_mode', False)
        model_complexity         = rospy.get_param('/holistic/model_complexity', 1)
        smooth_landmarks         = rospy.get_param('/holistic/smooth_landmarks', True)
        enable_segmentation      = rospy.get_param('/holistic/enable_segmentation', False)
        smooth_segmentation      = rospy.get_param('/holistic/smooth_segmentation', True)
        refine_face_landmarks    = rospy.get_param('/holistic/refine_face_landmarks', True)
        min_detection_confidence = rospy.get_param('/holistic/min_detection_confidence', 0.5)
        min_tracking_confidence  = rospy.get_param('/holistic/min_tracking_confidence', 0.5)

        # Read FaceMesh Parameters (0: Contours, 1: Tesselation)
        face_mesh_mode = rospy.get_param('mediapipe_gesture_recognition/face_mesh_mode', 0)
        self.face_mesh_connections = self.mp_holistic.FACEMESH_TESSELATION if face_mesh_mode == 1 else self.mp_holistic.FACEMESH_CONTOURS
        self.face_mesh_connection_drawing_spec = self.mp_drawing_styles.get_default_face_mesh_tesselation_style() if face_mesh_mode == 1 else self.mp_drawing_styles.get_default_face_mesh_contours_style()

        # Initialize MediaPipe Holistic
        if self.enable_right_hand or self.enable_left_hand or self.enable_pose or self.enable_face:
            self.holistic = self.mp_holistic.Holistic(static_image_mode, model_complexity, smooth_landmarks, enable_segmentation, smooth_segmentation,
                                                      refine_face_landmarks, min_detection_confidence, min_tracking_confidence)

        # Read Face Detection Parameters
        min_detection_confidence = rospy.get_param('/face_detection/min_detection_confidence', 0.5)
        model_selection          = rospy.get_param('/face_detection/model_selection', 0)

        # Initialize MediaPipe Face Detection
        if self.enable_face_detection:
            self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence, model_selection)

        # Read Objectron Parameters
        static_image_mode        = rospy.get_param('/objectron/static_image_mode', False)
        max_num_objects          = rospy.get_param('/objectron/max_num_objects', 5)
        min_detection_confidence = rospy.get_param('/objectron/min_detection_confidence', 0.5)
        min_tracking_confidence  = rospy.get_param('/objectron/min_tracking_confidence', 0.99)
        focal_length             = rospy.get_param('/objectron/focal_length', [1,1])
        principal_point          = rospy.get_param('/objectron/principal_point', [0,0])
        image_size               = rospy.get_param('/objectron/image_size', None)

        # Initialize MediaPipe Objectron
        if self.enable_objectron and self.objectron_model in ['Shoe', 'Chair', 'Cup', 'Camera']:
            self.objectron = self.mp_objectron.Objectron(static_image_mode, max_num_objects, min_detection_confidence, min_tracking_confidence,
                                                         self.objectron_model, focal_length, principal_point, None if image_size=='None' else image_size)

    def newKeypoint(self, landmark:NormalizedLandmark, number:int, name:str) -> Keypoint:

        """ New Keypoint Creation Function """

        # Assign Keypoint Coordinates
        new_keypoint = Keypoint()
        new_keypoint.x = landmark.x
        new_keypoint.y = landmark.y
        new_keypoint.z = landmark.z
        new_keypoint.v = landmark.visibility

        # RealSense Keypoint Additional Computation
        if self.realsense:

            # Compute Keypoint `x` and `y`
            x = int(new_keypoint.x * len(self.depth_image_flipped[0]))
            y = int(new_keypoint.y * len(self.depth_image_flipped))

            # Check Depth Limits
            if x >= len(self.depth_image_flipped[0]): x = len(self.depth_image_flipped[0]) - 1
            if y >= len(self.depth_image_flipped):    y = len(self.depth_image_flipped) - 1

            # Compute Keypoint Depth
            new_keypoint.depth = self.depth_image_flipped[y,x] * self.depth_scale

        # Assign Keypoint Number and Name
        new_keypoint.keypoint_number = number
        new_keypoint.keypoint_name = name

        return new_keypoint

    def processHand(self, RightLeft:bool, handResults:HolisticResults, image:cv2.typing.MatLike) -> Optional[Hand]:

        """ Process Hand Keypoints """

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

        # If Right / Left Hand Results Exists
        if (((RightLeft == self.RIGHT_HAND) and (handResults.right_hand_landmarks))
         or ((RightLeft == self.LEFT_HAND)  and (handResults.left_hand_landmarks))):

            # Add Keypoints to Hand Message
            for i in range(len(handResults.right_hand_landmarks.landmark if RightLeft else handResults.left_hand_landmarks.landmark)):

                # Append Keypoint
                hand_msg.keypoints.append(self.newKeypoint(handResults.right_hand_landmarks.landmark[i] if RightLeft else handResults.left_hand_landmarks.landmark[i], i+1, self.hand_landmarks_names[i]))

            # Publish Hand Keypoint Message
            self.hand_right_pub.publish(hand_msg) if RightLeft else self.hand_left_pub.publish(hand_msg)

    def processPose(self, poseResults:HolisticResults, image:cv2.typing.MatLike) -> Optional[Pose]:

        """ Process Pose Keypoints """

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

        # If Pose Results Results Exists
        if poseResults.pose_landmarks:

            # Add Keypoints to Pose Message
            for i in range(len(poseResults.pose_landmarks.landmark)):

                # Append Keypoint
                pose_msg.keypoints.append(self.newKeypoint(poseResults.pose_landmarks.landmark[i], i+1, self.pose_landmarks_names[i]))

            # Publish Pose Keypoint Message
            self.pose_pub.publish(pose_msg)

    def processFace(self, faceResults:HolisticResults, image:cv2.typing.MatLike) -> Optional[Face]:

        """ Process Face Keypoints """

        # Drawing the Face Landmarks
        self.mp_drawing.draw_landmarks(
            image,
            faceResults.face_landmarks,
            connections=self.face_mesh_connections,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.face_mesh_connection_drawing_spec)

        # Create Face Message
        face_msg = Face()
        face_msg.header.stamp = rospy.Time.now()
        face_msg.header.frame_id = 'Face Message'

        # If Face Results Results Exists
        if faceResults.face_landmarks:

            # Add Keypoints to Face Message
            for i in range(len(faceResults.face_landmarks.landmark)):

                # Assign Keypoint Coordinates
                new_keypoint = Keypoint()
                new_keypoint.x = faceResults.face_landmarks.landmark[i].x
                new_keypoint.y = faceResults.face_landmarks.landmark[i].y
                new_keypoint.z = faceResults.face_landmarks.landmark[i].z

                # Assign Keypoint Number
                new_keypoint.keypoint_number = i+1

                # Assign Keypoint Name (468 Landmarks -> Names = FACE_KEYPOINT_1 ...)
                new_keypoint.keypoint_name = f'FACE_KEYPOINT_{i+1}'

                # Append Keypoint
                face_msg.keypoints.append(new_keypoint)

            self.face_pub.publish(face_msg)

    def processFaceDetection(self, faceDetectionResults:DetectionResult, image:np.ndarray):

        """ Process Face Detection """

        # If Face Detection Results Results Exists
        if faceDetectionResults.detections:

            # Draw Face Detection
            for detection in faceDetectionResults.detections: self.mp_drawing.draw_detection(image, detection)

    def processObjectron(self, objectronResults:ObjectronResults, image:np.ndarray):

        """ Process Objectron """

        # If Objectron Results Results Exists
        if objectronResults.detected_objects:

            for detected_object in objectronResults.detected_objects:

                # Draw Landmarks
                self.mp_drawing.draw_landmarks(
                    image,
                    detected_object.landmarks_2d,
                    self.mp_objectron.BOX_CONNECTIONS)

                # Draw Axis
                self.mp_drawing.draw_axis(
                    image,
                    detected_object.rotation,
                    detected_object.translation)

    def processResults(self, image:np.ndarray):

        """ Process the Image with Enabled MediaPipe Solutions and Get MediaPipe Results """

        # Get Holistic, Face Detection, Objectron Results from MediaPipe
        holistic_results       = self.holistic.process(image)       if self.enable_right_hand or self.enable_left_hand or self.enable_pose or self.enable_face else None
        face_detection_results = self.face_detection.process(image) if self.enable_face_detection else None
        objectron_results      = self.objectron.process(image)      if self.enable_objectron else None

        # Process Left/Right Hand, Pose and Face Landmarks
        if self.enable_left_hand:  self.processHand(self.LEFT_HAND,  holistic_results, image)
        if self.enable_right_hand: self.processHand(self.RIGHT_HAND, holistic_results, image)
        if self.enable_pose: self.processPose(holistic_results, image)
        if self.enable_face: self.processFace(holistic_results, image)

        # Process Face Detection and Objectron
        if self.enable_face_detection: self.processFaceDetection(face_detection_results, image)
        if self.enable_objectron: self.processObjectron(objectron_results, image)

    def stream(self):

        """ Main Spinner Function """

        # Open Webcam or Realsense
        while (self.realsense or self.cap.isOpened()) and not rospy.is_shutdown():

            # Start Time
            start_time = rospy.Time.now()

            # RealSense Image Process
            if self.realsense:

                # Get Aligned Frames
                aligned_frames      = self.align.process(self.pipeline.wait_for_frames())
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame         = aligned_frames.get_color_frame()

                # Continue if Error in Aligned Frames
                if not aligned_depth_frame or not color_frame:
                    continue

                # Read Image
                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                self.depth_image_flipped = cv2.flip(depth_image, 1)
                color_image = np.asanyarray(color_frame.get_data())

                # Process Image
                self.depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                image = cv2.flip(color_image, 1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process Webcam Image
            else:

                # Read Webcam Image
                success, image = self.cap.read()

                if not success:
                    print('Ignoring Empty Camera Frame.')
                    # If loading a video, use 'break' instead of 'continue'
                    continue

                # To Improve Performance -> Process the Image as Not-Writeable
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Get and Process MediaPipe Results
            self.processResults(image)

            # Compute FPS
            fps = int(1 / (rospy.Time.now() - start_time).to_sec())

            if not self.realsense:

                # To Draw the Annotations -> Set the Image Writable
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Flip the image horizontally for a selfie-view display.
                image = cv2.flip(image, 1)

                # Draw FPS on Image
                image = cv2.putText(image, f"FPS: {fps}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,50,255), 1, cv2.LINE_AA)

            else:

                # Draw FPS on Image
                image = cv2.putText(image, f"FPS: {fps}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,50,255), 1, cv2.LINE_AA)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Show Image
            cv2.imshow('MediaPipe Landmarks', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

            # Sleep for the Remaining Cycle Time
            self.ros_rate.sleep()

if __name__ == '__main__':

    # Create MediaPipe Class
    MediapipeStream = MediapipeStreaming()

    # While ROS::OK
    while not rospy.is_shutdown():

        # MediaPipe Streaming Functions
        MediapipeStream.stream()

    # Close Webcam
    if not MediapipeStream.realsense: MediapipeStream.cap.release()
