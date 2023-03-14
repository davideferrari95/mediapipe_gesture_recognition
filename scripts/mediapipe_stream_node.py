#!/usr/bin/env python3

import rospy
import cv2
from termcolor import colored

import mediapipe
from mediapipe_gesture_recognition.msg import Pose, Face, Keypoint, Hand

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

    # Read Video Input (Webcam) Parameters
    self.webcam = rospy.get_param('webcam', 0)

    # Read Mediapipe Modules Parameters (Available Objectron Models = ['Shoe', 'Chair', 'Cup', 'Camera'])
    self.enable_right_hand      = rospy.get_param('enable_right_hand', False)
    self.enable_left_hand       = rospy.get_param('enable_left_hand', False)
    self.enable_pose            = rospy.get_param('enable_pose', False)
    self.enable_face            = rospy.get_param('enable_face', False)
    self.enable_face_detection  = rospy.get_param('enable_face_detection', False)
    self.enable_objectron       = rospy.get_param('enable_objectron', False)
    self.objectron_model        = rospy.get_param('objectron_model', 'Shoe')

    # Check Objectron Model
    if not self.objectron_model in self.available_objectron_models:
      rospy.logerr('ERROR: Objectron Model Not Available | Shutting Down...')
      rospy.signal_shutdown('ERROR: Objectron Model Not Available')

    # Debug Print
    print(colored(f'\nFunctions Enabled:\n', 'yellow'))
    print(colored(f'  Right Hand:     {self.enable_right_hand}',       'green' if self.enable_right_hand     else 'red'))
    print(colored(f'  Left  Hand:     {self.enable_left_hand}\n',      'green' if self.enable_left_hand      else 'red'))
    print(colored(f'  Skeleton:       {self.enable_pose}',             'green' if self.enable_pose           else 'red'))
    print(colored(f'  Face Mesh:      {self.enable_face}\n',           'green' if self.enable_face           else 'red'))
    print(colored(f'  Objectron:      {self.enable_objectron}',        'green' if self.enable_objectron      else 'red'))
    print(colored(f'  Face Detection: {self.enable_face_detection}\n', 'green' if self.enable_face_detection else 'red'))
    
    # Mediapipe Publishers
    self.hand_right_pub  = rospy.Publisher('/mediapipe_gesture_recognition/right_hand', Hand, queue_size=1)
    self.hand_left_pub   = rospy.Publisher('/mediapipe_gesture_recognition/left_hand', Hand, queue_size=1)
    self.pose_pub        = rospy.Publisher('/mediapipe_gesture_recognition/pose', Pose, queue_size=1)
    self.face_pub        = rospy.Publisher('/mediapipe_gesture_recognition/face', Face, queue_size=1)

    # Initialize Mediapipe
    self.mp_drawing         = mediapipe.solutions.drawing_utils
    self.mp_drawing_styles  = mediapipe.solutions.drawing_styles
    self.mp_holistic        = mediapipe.solutions.holistic
    self.mp_face_detection  = mediapipe.solutions.face_detection
    self.mp_objectron       = mediapipe.solutions.objectron

    # Initialize Mediapipe Solutions (Holistic, Face Detection, Objectron)
    self.initSolutions()

    # Open Video Webcam
    self.cap = cv2.VideoCapture(self.webcam)

    # Check Webcam Availability
    if self.cap is None or not self.cap.isOpened():
      rospy.logerr(f'ERROR: Webcam {self.webcam} Not Available | Starting Default: 0')
      self.cap = cv2.VideoCapture(0)

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

      # Publish Hand Keypoint Message
      self.hand_right_pub.publish(hand_msg) if RightLeft else self.hand_left_pub.publish(hand_msg)

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

      # Publish Pose Keypoint Message
      self.pose_pub.publish(pose_msg)    

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

      self.face_pub.publish(face_msg)

  def processFaceDetection(self, faceDetectionResults, image):

    ''' Process Face Detection '''

    if faceDetectionResults.detections:

      # Draw Face Detection
      for detection in faceDetectionResults.detections: self.mp_drawing.draw_detection(image, detection)

  def processObjectron(self, objectronResults, image):

    ''' Process Objectron '''

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

  def initSolutions(self):

    # Initialize Mediapipe Holistic
    if self.enable_right_hand or self.enable_left_hand or self.enable_pose or self.enable_face: 
      self.holistic = self.mp_holistic.Holistic(refine_face_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Initialize Mediapipe Face Detection
    elif self.enable_face_detection: 
      self.face_detection = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    # Initialize Mediapipe Objectron
    elif self.enable_objectron and self.objectron_model in ['Shoe', 'Chair', 'Cup', 'Camera']:
      self.objectron = self.mp_objectron.Objectron(static_image_mode=False, max_num_objects=5, min_detection_confidence=0.5,
                                              min_tracking_confidence=0.99, model_name=self.objectron_model) 

  def getResults(self, image):

      # Get Holistic Results from Mediapipe Holistic
      if self.enable_right_hand or self.enable_left_hand or self.enable_pose or self.enable_face: 
        self.holistic_results = self.holistic.process(image)

      # Get Face Detection Results from Mediapipe
      if self.enable_face_detection: self.face_detection_results = self.face_detection.process(image)

      # Get Objectron Results from Mediapipe
      if self.enable_objectron: self.objectron_results = self.objectron.process(image)

  def processResults(self, image):

    ''' Process the Image with Enabled Mediapipe Solutions '''

    # Process Left Hand Landmarks
    if self.enable_left_hand:  self.processHand(self.LEFT_HAND,  self.holistic_results, image)

    # Process Right Hand Landmarks
    if self.enable_right_hand: self.processHand(self.RIGHT_HAND, self.holistic_results, image)

    # Process Pose Landmarks
    if self.enable_pose: self.processPose(self.holistic_results, image)

    # Process Face Landmarks
    if self.enable_face: self.processFace(self.holistic_results, image)

    # Process Face Detection
    if self.enable_face_detection: self.processFaceDetection(self.face_detection_results, image)

    # Process Objectron
    if self.enable_objectron: self.processObjectron(self.objectron_results, image)

  def stream(self):

    # Open Webcam
    while self.cap.isOpened() and not rospy.is_shutdown():

      # Read Webcam Image
      success, image = self.cap.read()

      if not success:
        print('Ignoring empty camera frame.')
        continue

      # To Improve Performance -> Process the Image as Not-Writeable
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      # Get Mediapipe Result
      self.getResults(image)

      # To Draw the Annotations -> Set the Image Writable
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      # Process Mediapipe Results
      self.processResults(image)

      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('MediaPipe Landmarks', cv2.flip(image, 1))
      if cv2.waitKey(5) & 0xFF == 27:
        break

if __name__ == '__main__':

  # Create Mediapipe Class
  MediapipeStream = MediapipeStreaming()

  # While ROS::OK
  while not rospy.is_shutdown():

    # Mediapipe Streaming Functions
    MediapipeStream.stream()

    # Sleep for the Remaining Cycle Time
    MediapipeStream.ros_rate.sleep()

  # Close Webcam
  MediapipeStream.cap.release()
