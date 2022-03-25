#!/usr/bin/env python3

from mediapipe_gesture_recognition.scripts.Publisher import E_D_Webcam
import rospy
import mediapipe as mp
from mediapipe_gesture_recognition.msg import Hand, Pose, Face, Keypoint
import cv2

rospy.init_node('mediapipe_stream_node', anonymous=True)
rate = rospy.Rate(100) # 100hz 


# Mediapipe Publishers
hand_right_pub  = rospy.Publisher('/mediapipe/hand_right', Hand, queue_size=1)
hand_left_pub   = rospy.Publisher('/mediapipe/hand_left', Hand, queue_size=1)
pose_pub        = rospy.Publisher('/mediapipe/pose', Pose, queue_size=1)
face_pub        = rospy.Publisher('/mediapipe/face', Face, queue_size=1)

# Mediapipe Messages
hand_right_msg = Hand()
hand_left_msg = Hand()
pose_msg = Pose()
face_msg = Face()

# Read Webcam Parameters
webcam = rospy.get_param('webcam', 0)

# Read Mediapipe Modules Parameters
enable_holistic = rospy.get_param('enable_holistic', True)
enable_right_hand = rospy.get_param('enable_right_hand', False)
enable_left_hand = rospy.get_param('enable_left_hand', False)
enable_pose = rospy.get_param('enable_pose', False)
enable_face = rospy.get_param('enable_face', False)

# Define Constant Variables

# Define Landmarks Names
hand_landmarks_names = ['WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']
pose_landmarks_names = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_TUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']

# For webcam input:
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection

cap = cv2.VideoCapture(webcam)

# While loop mediapipe code that need to be run each cycle
#while not rospy.is_shutdown():

with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #if enable_right_hand:
    hand_results = hands.process(image)
    pose_results = pose.process(image)
    face_results = face_detection.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if hand_results.multi_hand_landmarks:
      for hand_landmarks in hand_results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        pose_results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if face_results.detections:
      for detection in face_results.detections:
        mp_drawing.draw_detection(image, detection)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()

# add keypoint to each message ordered

if hand_results.right_hand_landmarks:
  hand_right_msg.right_or_left = hand_right_msg.RIGHT

  # Run mediapipe right_hand detection (before and obtain the landmarks)
  for i in range(len(hand_results.right_hand_landmarks.landmark)):

    """
    landmark {
      x: 0.414364755153656
      y: 0.2742988169193268
      z: -0.5210666060447693
      visibility: 0.9999992847442627
    }
    landmark {
      x: 0.4284072518348694
      y: 0.2577638030052185
      z: -0.49063441157341003
      visibility: 0.9999985694885254
    } 
    """
    
    # Read keypoint
    new_keypoint = Keypoint()
    new_keypoint.x = hand_results.right_hand_landmarks.landmark[i].x
    new_keypoint.y = hand_results.right_hand_landmarks.landmark[i].y
    new_keypoint.z = hand_results.right_hand_landmarks.landmark[i].z
    new_keypoint.v = hand_results.right_hand_landmarks.landmark[i].visibility

    new_keypoint.keypoint_number = i
    new_keypoint.keypoint_name = hand_landmarks_names[i]

    # Append keypoint
    hand_right_msg.keypoints.append(new_keypoint)

  hand_right_pub.publish(hand_right_msg)


elif hand_results.left_hand_landmarks:
  hand_left_msg.right_or_left = hand_left_msg.LEFT

  # Run mediapipe left_hand detection
  for i in range(len(hand_results.left_hand_landmarks.landmark)):

    # Read keypoint
    new_keypoint = Keypoint()
    new_keypoint.x = hand_results.left_hand_landmarks.landmark[i].x
    new_keypoint.y = hand_results.left_hand_landmarks.landmark[i].y
    new_keypoint.z = hand_results.left_hand_landmarks.landmark[i].z
    new_keypoint.v = hand_results.left_hand_landmarks.landmark[i].visibility

    new_keypoint.keypoint_number = i
    new_keypoint.keypoint_name = hand_landmarks_names[i]

    # Append keypoint
    hand_left_msg.keypoints.append(new_keypoint)

  hand_left_pub.publish(hand_left_msg)

elif enable_pose:

  for i in range(6):
    # Read keypoint
    new_keypoint = Keypoint()
    new_keypoint.x = pose_results.pose_landmarks.landmark[i].x
    new_keypoint.y = pose_results.pose_landmarks.landmark[i].y
    new_keypoint.z = pose_results.pose_landmarks.landmark[i].z
    new_keypoint.v = pose_results.pose_landmarks.landmark[i].visibility

    new_keypoint.keypoint_number = i
    new_keypoint.keypoint_name = pose_landmarks_names[i]

    # Append keypoint
    pose_msg.keypoints.append(new_keypoint)

  pose_pub.publish(pose_msg)

elif enable_face:

        for i in range(6):
            # Read keypoint
            new_keypoint = Keypoint()
            new_keypoint.x = ...
            new_keypoint.y = ...
            new_keypoint.z = ...
            new_keypoint.v = ...
            new_keypoint.keypoint_number
            new_keypoint.keypoint_name
            # Append keypoint
            face_msg.keypoints.append(new_keypoint)

        face_pub.publish(face_msg) 
    
# Run mediapipe detection


# Sleep for the Remaining Cycle Time
rate.sleep()
