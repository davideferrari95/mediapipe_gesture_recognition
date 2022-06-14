#!/usr/bin/env python3

import glob
import os
import rospy, rospkg
import time
import pandas as pd, pickle

# Import scikit functions
from sklearn.model_selection import train_test_split 
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Accuracy metrics 

# Obtain files from a directory
from os import listdir
from os.path import isfile, join, isdir

# Import Mediapipe Messages
from mediapipe_gesture_recognition.msg import Pose, Face, Hand

from scipy import stats
from turtle import right
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import weakref

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

def prob_viz(res, actions, input_frame):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame


def Recognition ():
    global sequence
    global predictions

    # 2. Prediction logic
    keypoints = extract_keypoints(pose_new_msg, face_new_msg, left_new_msg, right_new_msg)
    sequence.append(keypoints)
    sequence = sequence[-30:]

    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        print(actions[np.argmax(res)])
        predictions.append(np.argmax(res))
        print("no gesture")

'''
    #3. Viz logic
        if np.unique(predictions[-10:])[0]==np.argmax(res): 
            if res[np.argmax(res)] > threshold: 

                if len(sentence) > 0: 
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

        if len(sentence) > 5: 
            sentence = sentence[-5:]

        # Viz probabilities
        image = prob_viz(res, actions, image)
'''


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

#Load the trained model for the detected landmarks
with open(f'/home/tanguy/tanguy_ws/src/mediapipe_gesture_recognition/database/3D_Gestures/trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

actions = [f for f in listdir(f'{package_path}/database/3D_Gestures/') if isdir(join(f'{package_path}/database/3D_Gestures/', f))]
actions = np.array(actions)


# 1. New detection variables
sequence = []       #Collect 30 frames to make a prediction with this frames
sentence = []       #Concatenate the gestures detected
predictions = []
threshold = 0.5     #Confidence metric, if the prediction is sure at 50% it will print the gesture

while not rospy.is_shutdown():
        Recognition()