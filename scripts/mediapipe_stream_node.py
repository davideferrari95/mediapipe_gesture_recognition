#!/usr/bin/env python3

import rospy
import mediapipe as mp
from mediapipe_gesture_recognition.msg import Hand, Skeleton, Face, Keypoint
import cv2

rospy.init_node('mediapipe_stream_node', anonymous=True)
rate = rospy.Rate(100) # 100hz 

 
hand_right_pub = rospy.Publisher('/mediapipe/hand_right', Hand, queue_size=1)
hand_left_pub = rospy.Publisher('/mediapipe/hand_left', Hand, queue_size=1)
skeleton_pub = rospy.Publisher('/mediapipe/skeleton', Skeleton, queue_size=1)
face_pub = rospy.Publisher('/mediapipe/face', Face, queue_size=1)

var = Hand()
Hand.right_or_left = 1

hand_right_msg = Hand()
hand_left_msg = Hand()
skeleton_msg = Skeleton()
face_msg = Face()

# initialization
right_hand = False
left_hand = False
skeleton = False
face = False

while not rospy.is_shutdown():

    if right_hand:

        # Run mediapipe right_hand detection

        hand_right_msg.right_or_left = hand_right_msg.RIGHT

        for i in range(21):
            # Read keypoint
            new_keypoint = Keypoint()
            new_keypoint.x = ...
            new_keypoint.y = ...
            new_keypoint.z = ...
            new_keypoint.v = ...
            new_keypoint.keypoint_number
            new_keypoint.keypoint_name
            # Append keypoint
            hand_right_msg.keypoints.append(new_keypoint)

        hand_right_pub(hand_right_msg)

    elif left_hand:

            # Run mediapipe right_hand detection

            hand_left_msg.right_or_left = hand_left_msg.RIGHT

            for i in range(21):
                # Read keypoint
                new_keypoint = Keypoint()
                new_keypoint.x = ...
                new_keypoint.y = ...
                new_keypoint.z = ...
                new_keypoint.v = ...
                new_keypoint.keypoint_number
                new_keypoint.keypoint_name
                # Append keypoint
                hand_left_msg.keypoints.append(new_keypoint)

            hand_left_pub(hand_left_msg)

    elif skeleton:

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
                skeleton_msg.keypoints.append(new_keypoint)

            skeleton_pub(skeleton_msg)

    elif face:

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

            face_pub(face_msg) 
    
# Run mediapipe detection

# For webcam input:
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection

cap = cv2.VideoCapture(0)

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
    results = hands.process(image)
    results1 = pose.process(image)
    results2 = face_detection.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
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
        results1.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results2.detections:
      for detection in results2.detections:
        mp_drawing.draw_detection(image, detection)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()

# add keypoint to each message ordered
# publish each message

