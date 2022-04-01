#!/usr/bin/env python3

import rospy
import cv2
import mediapipe as mp
from mediapipe_gesture_recognition.msg import Hand, Pose, Face, Keypoint

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
mp_face_mesh = mp.solutions.face_mesh
mp_holistic = mp.solutions.holistic

# Video Webcam
cap = cv2.VideoCapture(webcam)


# While loop mediapipe code that need to be run each cycle
while not rospy.is_shutdown():

  with mp_holistic.Holistic(refine_face_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
      success, image = cap.read()
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

      if enable_right_hand or enable_left_hand:
        hand_results = holistic.process(image)

        #Drawing the right hand landmarks
        if enable_right_hand:
          mp_drawing.draw_landmarks(
              image,
              hand_results.right_hand_landmarks,
              mp_holistic.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())

          if hand_results.right_hand_landmarks:
            #add right_hand keypoint to ordered message
            hand_right_msg.right_or_left = hand_right_msg.RIGHT

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


        #Drawing the left hand landmarks
        if enable_left_hand:
          mp_drawing.draw_landmarks(
              image,
              hand_results.left_hand_landmarks,
              mp_holistic.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())

          #add left_hand keypoint to ordered message
          if hand_results.left_hand_landmarks:
            #add left_hand keypoint to ordered message
            hand_left_msg.right_or_left = hand_left_msg.RIGHT
  
            # Run mediapipe right_hand detection (before and obtain the landmarks)
            for i in range(len(hand_results.left_hand_landmarks.landmark)):
            
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
              new_keypoint.x = hand_results.left_hand_landmarks.landmark[i].x
              new_keypoint.y = hand_results.left_hand_landmarks.landmark[i].y
              new_keypoint.z = hand_results.left_hand_landmarks.landmark[i].z
              new_keypoint.v = hand_results.left_hand_landmarks.landmark[i].visibility
  
              new_keypoint.keypoint_number = i
              new_keypoint.keypoint_name = hand_landmarks_names[i]
  
              # Append keypoint
              hand_left_msg.keypoints.append(new_keypoint)
            
            hand_left_pub.publish(hand_left_msg)


      if enable_pose:
        pose_results = holistic.process(image)

        #Drawing the pose landmarks
        mp_drawing.draw_landmarks(
            image,
            pose_results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        if pose_results.pose_landmarks:

          #add pose keypoint to ordered message   
          for i in range(len(pose_results.pose_landmarks.landmark)): # BUG : TypeError: object of type 'NoneType' has no len()

            """
            x: 0.6068623065948486
            y: 0.6184841394424438
            z: -0.22558169066905975
            visibility: 0.9960732460021973
            , x: -0.21233102679252625
            y: 0.6388503909111023
            z: 0.1573469489812851
            visibility: 0.9742392301559448
            """

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


      if enable_face:
        face_results = holistic.process(image)

        #Drawing the face landmarks
        mp_drawing.draw_landmarks(
            image,
            face_results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

        if face_results.face_landmarks:
          #add face keypoint to ordered message
          for i in range(len(face_results.face_landmarks.landmark)): # BUG : TypeError: object of type 'NoneType' has no len()
            """
            landmark {
              x: 0.6572769284248352
              y: 0.44084274768829346
              z: -0.02765425108373165
            }
            landmark {
              x: 0.6293548941612244
              y: 0.4240654706954956
              z: -0.02765425108373165
            }
            """
  
            # Read keypoint
            new_keypoint = Keypoint()
            new_keypoint.x = face_results.face_landmarks.landmark[i].x
            new_keypoint.y = face_results.face_landmarks.landmark[i].y
            new_keypoint.z = face_results.face_landmarks.landmark[i].z
            # BUG: new_keypoint.v = pose_results.face_landmarks.landmark.visibility
            new_keypoint.keypoint_number = i
  
            # 468 Landmarks so custom names as FACE_KEYPOINT_1 ...
            new_keypoint.keypoint_name = f'FACE_KEYPOINT_{i}'
            # Append keypoint
            face_msg.keypoints.append(new_keypoint)
  
          face_pub.publish(face_msg) 


      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('MediaPipe Landmarks', cv2.flip(image, 1))
      if cv2.waitKey(5) & 0xFF == 27:
        break

  # Last while loop command - Sleep for the Remaining Cycle Time
  rate.sleep()

# END WHILE ROS:OK

# Close Webcam
cap.release()
