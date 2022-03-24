#!/usr/bin/env python3

import rospy
import mediapipe as mp
from mediapipe_gesture_recognition.msg import Hand, Skeleton, Face, Keypoint

rospy.init_node('mediapipe_stream_node', anonymous=True)
rate = rospy.Rate(100) # 100hz 

# Mediapipe Publishers
hand_right_pub  = rospy.Publisher('/mediapipe/hand_right', Hand, queue_size=1)
hand_left_pub   = rospy.Publisher('/mediapipe/hand_left', Hand, queue_size=1)
skeleton_pub    = rospy.Publisher('/mediapipe/skeleton', Skeleton, queue_size=1)
face_pub        = rospy.Publisher('/mediapipe/face', Face, queue_size=1)

# Mediapipe Messages
hand_right_msg = Hand()
hand_left_msg = Hand()
skeleton_msg = Skeleton()
face_msg = Face()

# Read Webcam Parameters
webcam = rospy.get_param('webcam', 0)

# Read Mediapipe Modules Parameters
enable_holistic = rospy.get_param('enable_holistic', True)
enable_right_hand = rospy.get_param('enable_right_hand', False)
enable_left_hand = rospy.get_param('enable_left_hand', False)
enable_pose = rospy.get_param('enable_pose', False)
enable_face = rospy.get_param('enable_face', False)

while not rospy.is_shutdown():

    if enable_right_hand:

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

    # Run mediapipe detection
    # add keypoint to each message ordered
    # publish each message

    # Sleep for the Remaining Cycle Time
    rate.sleep()