#!/usr/bin/env python3

import rospy, rospkg
import numpy as np
import pandas as pd, pickle

# Obtain name of files from a directory
from os import listdir
from os.path import join, isdir

# Import Mediapipe Messages
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
#                       Main Spinner                       #
############################################################


def extract_keypoints(pose_msg, face_msg, left_msg, right_msg):
    pose = np.array([[res.x, res.y, res.z, res.v] for res in pose_msg.keypoints]).flatten() if pose_new_msg else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in face_msg.keypoints]).flatten() if face_new_msg else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in left_msg.keypoints]).flatten() if left_new_msg else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in right_msg.keypoints]).flatten() if right_new_msg else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


def Recognition ():
    global sequence

    coordinates = []
    if (enable_pose_ == True and 'pose_new_msg' in globals()):
        pose = np.array([[res.x, res.y, res.z, res.v] for res in pose_new_msg.keypoints]).flatten() if pose_new_msg else np.zeros(33*4)
        coordinates.append(pose)
    
    if (enable_face_ == True and 'face_new_msg' in globals()):
        face = np.array([[res.x, res.y, res.z] for res in face_new_msg.keypoints]).flatten() if face_new_msg else np.zeros(468*3)
        coordinates.append(face)

    if (enable_left_hand_ == True and 'left_new_msg' in globals()):    
        lh = np.array([[res.x, res.y, res.z] for res in left_new_msg.keypoints]).flatten() if left_new_msg else np.zeros(21*3)
        coordinates.append(lh)

    if (enable_right_hand_ == True and 'right_new_msg' in globals()):
        rh = np.array([[res.x, res.y, res.z] for res in right_new_msg.keypoints]).flatten() if right_new_msg else np.zeros(21*3)
        coordinates.append(rh)

    keypoints = np.concatenate(coordinates)

    # Prediction logic
    #keypoints = extract_keypoints(pose_new_msg, face_new_msg, left_new_msg, right_new_msg)
    sequence.append(keypoints)        #Append the landmarks coordinates from the last frame to our sequence
    sequence = sequence[-30:]         #Permits to always analyse only the last 30 frames of the webcam to have a real time recognition without stops

    if len(sequence) == 30:
        
        # Obtain the probability of each gesture
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        #print (res)
        #print (np.argmax(res))
        
        #Print the name of the gesture recognised
        Prob = np.amax(res)
        if (Prob>0.7):    
            print(actions[np.argmax(res)])


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

#Save witch part of the body is being detected
gesture_file = ''
if enable_right_hand_ == True :
    gesture_file = gesture_file + "Right"
if enable_left_hand_== True:
    gesture_file = gesture_file + "Left"
if enable_pose_== True:
    gesture_file = gesture_file + "Pose"
if enable_face_ == True:
    gesture_file = gesture_file + "Face"

#Load the trained model for the detected landmarks
with open(f'/home/tanguy/tanguy_ws/src/mediapipe_gesture_recognition/database/3D_Gestures/{gesture_file}/trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

#Load the names of the actions saved
actions = np.array([f for f in listdir(f'{package_path}/database/3D_Gestures/{gesture_file}/') if isdir(join(f'{package_path}/database/3D_Gestures/{gesture_file}/', f))])
print(actions)

#We define the variable where we will store 30 consecutive frames to make our prediction
sequence = []

while not rospy.is_shutdown():
        Recognition()