#!/usr/bin/env python3

import rospy
import mediapipe as mp
from mediapipe_gesture_recognition.msg import Hand, Skeleton, Face


rospy.init_node('mediapipe_stream_node', anonymous=True)
rate = rospy.Rate(100) # 100hz 

 
hand_right_pub = rospy.Publisher('/mediapipe/hand_right', Hand, queue_size=1)
hand_left_pub = rospy.Publisher('/mediapipe/hand_left', Hand, queue_size=1)
skeleton_pub = rospy.Publisher('/mediapipe/skeleton', Skeleton, queue_size=1)

hand_right_msg = Hand()
hand_left_msg = Hand()
skeleton_msg = Skeleton()

# Run mediapipe detection
# add keypoint to each message ordered
# publish each message
