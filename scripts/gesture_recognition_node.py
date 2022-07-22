#!/usr/bin/env python3

import rospy, rospkg
import pandas as pd 
import pickle 
import time

from mediapipe_gesture_recognition.msg import Pose, Face, Hand

# Get Package Path
package_path = rospkg.RosPack().get_path('mediapipe_gesture_recognition')

############################################################
#                    Callback Functions                    #
############################################################


def handRightCallback(data):
    global right_new_msg
    right_new_msg = data

def handLeftCallback(data):
    global left_new_msg 
    left_new_msg = data

def PoseCallback(data):
    global pose_new_msg 
    pose_new_msg = data

def FaceCallback(data):
    global face_new_msg
    face_new_msg = data


############################################################
#                     Countdown function                   #
############################################################


def countdown(num_of_secs):
    
    while (not rospy.is_shutdown() and num_of_secs!=0):
        m, s = divmod(num_of_secs, 60)
        min_sec_format = '{:02d}:{:02d}'.format(m, s)
        print(min_sec_format)
        time.sleep(1)
        num_of_secs -= 1


############################################################
#                       Main Spinner                       #
############################################################


def Recognition ():
    #Load the trained model for the detected landmarks
    with open(f'{package_path}/database/Gestures/{gesture_file}/trained_model.pkl', 'rb') as f:
        model = pickle.load(f) 

    #Create a list with the detected landmarks coordinates
    Landmarks = []
    while Landmarks == []:
        if (enable_right_hand_ == True and 'right_new_msg' in globals()):
            for i in range (len(right_new_msg.keypoints)):
                Landmarks.extend([right_new_msg.keypoints[i].x, right_new_msg.keypoints[i].y, right_new_msg.keypoints[i].z, right_new_msg.keypoints[i].v])

        if (enable_left_hand_ == True and 'left_new_msg' in globals()):
            for i in range (len(left_new_msg.keypoints)):
                Landmarks.extend([left_new_msg.keypoints[i].x, left_new_msg.keypoints[i].y, left_new_msg.keypoints[i].z, left_new_msg.keypoints[i].v])

        if (enable_pose_ == True and 'pose_new_msg' in globals()):
            for i in range (len(pose_new_msg.keypoints)):
                Landmarks.extend([pose_new_msg.keypoints[i].x, pose_new_msg.keypoints[i].y, pose_new_msg.keypoints[i].z, pose_new_msg.keypoints[i].v])

        if (enable_face_ == True and 'face_new_msg' in globals()):
            for i in range (len(face_new_msg.keypoints)):
                Landmarks.extend([face_new_msg.keypoints[i].x, face_new_msg.keypoints[i].y, face_new_msg.keypoints[i].z, face_new_msg.keypoints[i].v])

    
    #Prediction with the trained model
    X = pd.DataFrame([Landmarks])
    pose_recognition_class = model.predict(X)[0]
    pose_recognition_prob = model.predict_proba(X)[0]

    
    #Print the gesture recognised
    Prob=max(pose_recognition_prob)
    if (Prob>0.7):
        print(pose_recognition_class)


############################################################
#                    ROS Initialization                    #
############################################################


# ROS Initialization
rospy.init_node('mediapipe_gesture_recognition_training_node', anonymous=True) 
rate = rospy.Rate(30)

# Mediapipe Subscribers
rospy.Subscriber('/mediapipe_gesture_recognition/right_hand', Hand, handRightCallback)
rospy.Subscriber('/mediapipe_gesture_recognition/left_hand', Hand, handLeftCallback)
rospy.Subscriber('/mediapipe_gesture_recognition/pose', Pose, PoseCallback)
rospy.Subscriber('/mediapipe_gesture_recognition/face', Face, FaceCallback)

# Read Mediapipe Modules Parameters
enable_right_hand_ = rospy.get_param('enable_right_hand', False)
enable_left_hand_ = rospy.get_param('enable_left_hand', False)
enable_pose_ = rospy.get_param('enable_pose', False)
enable_face_ = rospy.get_param('enable_face', False)


############################################################
#                           Main                           #
############################################################


gesture_file = ''
if enable_right_hand_ ==True :
    gesture_file = gesture_file + "Right"
if enable_left_hand_==True:
    gesture_file = gesture_file + "Left"
if enable_pose_==True:
    gesture_file = gesture_file + "Pose"
if enable_face_==True:
    gesture_file = gesture_file + "Face"

while not rospy.is_shutdown():
        Recognition()