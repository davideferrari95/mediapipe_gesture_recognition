#!/usr/bin/env python3

import rospy
import mediapipe as mp
from mediapipe_gesture_recognition.msg import Hand, Skeleton, Face, Keypoint

rospy.init_node('mediapipe_stream_node', anonymous=True)
rate = rospy.Rate(100) # 100hz 
 
hand_right_pub = rospy.Publisher('/mediapipe/hand_right', Hand, queue_size=1)
hand_left_pub = rospy.Publisher('/mediapipe/hand_left', Hand, queue_size=1)
skeleton_pub = rospy.Publisher('/mediapipe/skeleton', Skeleton, queue_size=1)

hand_right_msg = Hand()
hand_left_msg = Hand()
skeleton_msg = Skeleton()

# initialization
right_hand = False



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

    # Run mediapipe detection
    # add keypoint to each message ordered
    # publish each message

