#!/usr/bin/env python3

import rospy
import cv2
import mediapipe as mp
from mediapipe_gesture_recognition.msg import Pose, Face, Keypoint, Hand

class MediapipeStreaming:
  
  def __init__(self, webcam, enable_right_hand, enable_left_hand, enable_pose, enable_face):
    
    # Mediapipe Publishers
    self.hand_right_pub  = rospy.Publisher('/mediapipe_gesture_recognition/right_hand', Hand, queue_size=1)
    self.hand_left_pub   = rospy.Publisher('/mediapipe_gesture_recognition/left_hand', Hand, queue_size=1)
    self.pose_pub        = rospy.Publisher('/mediapipe_gesture_recognition/pose', Pose, queue_size=1)
    self.face_pub        = rospy.Publisher('/mediapipe_gesture_recognition/face', Face, queue_size=1)
    
    # Constants
    self.RIGHT_HAND = True
    self.LEFT_HAND = False

    # Video Input
    self.webcam = webcam

    # Boolean Parameters
    self.enable_right_hand = enable_right_hand
    self.enable_left_hand = enable_left_hand
    self.enable_pose = enable_pose
    self.enable_face = enable_face

    # Define Landmarks Names
    self.hand_landmarks_names = ['WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']
    self.pose_landmarks_names = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_TUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']

    # Initialize Mediapipe:
    self.mp_drawing = mp.solutions.drawing_utils
    self.mp_drawing_styles = mp.solutions.drawing_styles
    self.mp_hands = mp.solutions.hands
    self.mp_pose = mp.solutions.pose
    self.mp_face_mesh = mp.solutions.face_mesh
    self.mp_holistic = mp.solutions.holistic

    # Open Video Webcam
    self.cap = cv2.VideoCapture(self.webcam)
    
  def newKeypoint(self, landmark, number, name):
    
    # Assing Keypoint Coordinates
    new_keypoint = Keypoint()
    new_keypoint.x = landmark.x
    new_keypoint.y = landmark.y
    new_keypoint.z = landmark.z
    new_keypoint.v = landmark.visibility

    # Assing Keypoint Number and Name
    new_keypoint.keypoint_number = number
    new_keypoint.keypoint_name = name
    
    return new_keypoint
    
    
  def processHand(self, RightLeft, handResults, image):
        
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
        hand_msg.keypoints.append(self.newKeypoint(handResults.right_hand_landmarks.landmark[i] if RightLeft else handResults.left_hand_landmarks.landmark[i], i, self.hand_landmarks_names[i]))

      # Publish Hand Keypoint Message
      self.hand_right_pub.publish(hand_msg) if RightLeft else self.hand_left_pub.publish(hand_msg)

  def processPose(self, poseResults, image):
        
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

        # Assing Keypoint Coordinates
        new_keypoint = Keypoint()
        new_keypoint.x = faceResults.face_landmarks.landmark[i].x
        new_keypoint.y = faceResults.face_landmarks.landmark[i].y
        new_keypoint.z = faceResults.face_landmarks.landmark[i].z

        # Assing Keypoint Number
        new_keypoint.keypoint_number = i

        # Assing Keypoint Name (468 Landmarks -> Names = FACE_KEYPOINT_1 ...)
        new_keypoint.keypoint_name = f'FACE_KEYPOINT_{i+1}'

        # Append Keypoint
        face_msg.keypoints.append(new_keypoint)

      self.face_pub.publish(face_msg)
    
  def stream(self):
        
    # Open Mediapipe Holistic
    with self.mp_holistic.Holistic(refine_face_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
      
      # Open Webcam
      while self.cap.isOpened() and not rospy.is_shutdown():
        
        success, image = self.cap.read()
        
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue

        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          
        # Draw the annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.enable_right_hand or self.enable_left_hand:
          
          # Get Hand Results from Mediapipe Holistic
          hand_results = holistic.process(image)

          # Process Left/Right Hand Landmarks
          if self.enable_right_hand: self.processHand(self.RIGHT_HAND, hand_results, image)
          if self.enable_left_hand:  self.processHand(self.LEFT_HAND,  hand_results, image)

        if self.enable_pose:
          
          # Get Pose Results from Mediapipe Holistic
          pose_results = holistic.process(image)

          # Process Pose Landmakrs
          self.processPose(pose_results, image)
          
        if self.enable_face:
          
          # Get Face Results from Mediapipe Holistic
          face_results = holistic.process(image)

          # Process Pose Landmakrs
          self.processFace(face_results, image)

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Landmarks', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
          break

if __name__ == "__main__":
      
  # ROS Initialization
  rospy.init_node('mediapipe_stream_node', anonymous=True)
  ros_rate = rospy.Rate(100)
  
  # Read Webcam Parameters
  webcam_ = rospy.get_param('webcam', 0)

  # Read Mediapipe Modules Parameters
  enable_right_hand_ = rospy.get_param('enable_right_hand', False)
  enable_left_hand_ = rospy.get_param('enable_left_hand', False)
  enable_pose_ = rospy.get_param('enable_pose', False)
  enable_face_ = rospy.get_param('enable_face', False)
  
  # Create Mediapipe Class
  MediapipeStream = MediapipeStreaming(webcam_, enable_right_hand_, enable_left_hand_, enable_pose_, enable_face_)
  
  # While ROS::OK
  while not rospy.is_shutdown():
    
    # Mediapipe Streaming Functions
    MediapipeStream.stream()

    # Sleep for the Remaining Cycle Time
    ros_rate.sleep()

  # Close Webcam
  MediapipeStream.cap.release()
