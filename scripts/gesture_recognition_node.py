#!/usr/bin/env python3

import rospy
import csv
import cv2 
import numpy as np
import pandas as pd 
import pickle 
import os
import time

from mediapipe_gesture_recognition.msg import Pose, Face, Hand
from mediapipe_stream_node import MediapipeStreaming 

# Mediapipe Subscribers Callback
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

#Define which gesture will be linked to which action for the robot
def Setup_of_robot_action():
    with open(f"/home/baheu/ws_sk_tracking/src/sk_tracking/TXT file/Position_Name/Position_Name.txt", "r") as file: 
        allText = file.read() 
        words = list(map(str, allText.split())) 
        print('\nFor the moment the robot can have 5 different information, GO, Turn left, Turn right, stop and back\nIn this next line of code you will define what action will do your position')
        print('\nTo set up a position you just have to write :\n1 for GO\n2 for LEFT\n3 for RIGHT\n4 for STOP\n5 for BACK')
    
    global liste
    nbr_pos=len(words)
    liste=["P1","P2","P3","P4","P5","P6","P7","P8","P9","P10"]
    for i in range(0,nbr_pos):   
        pos=input(f'\nFor which action your position {words[i]} will be used ? ')
        print(pos)
        if (pos=="1"):
            liste[0]=words[i]
        elif (pos=="2"):
            liste[1]=words[i]
        elif (pos=="3"):
            liste[2]=words[i]
        elif (pos=="4"):
            liste[3]=words[i]
        elif (pos=="5"):
            liste[4]=words[i]
        elif (pos=="6"):
           liste[5]=words[i]
        elif (pos=="7"):
           liste[6]=words[i]
        elif (pos=="8"):
           liste[4]=words[i]
        elif (pos=="9"):
           liste[8]=words[i]
        elif (pos=="10"):
           liste[9]=words[i]

def Justin_code():
    with open(f'/home/tanguy/tanguy_ws/src/mediapipe_gesture_recognition/PKL files/trained_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    webcam=0
    cap = cv2.VideoCapture(webcam)
    
    X = pd.DataFrame(['row'])             #row = row with all the coordinates detected in the stream
    pose_recognition_class = model.predict(X)[0]
    pose_recognition_prob = model.predict_proba(X)[0]
    print(pose_recognition_class, pose_recognition_prob)

    ret, frame = cap.read()
    # Recolor Feed
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False 

    Prob=max(pose_recognition_prob)
    if (Prob>0.7):
        print(pose_recognition_class)

    # Display Class
    cv2.putText(image, 'CLASS'
                , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, pose_recognition_class.split(' ')[0]
                , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Display Probability
    cv2.putText(image, 'PROB'
                , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    
    cv2.putText(image, str(round(pose_recognition_prob[np.argmax(pose_recognition_prob)],2))
                , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Here you can set up your robot driving parameters with the name of your class (like STOP, LEFT, RIGHT...) and the probability of precision
    Prob=max(pose_recognition_prob)
    if (pose_recognition_class==liste[0]) and (Prob>0.7):
        print('Gesture 1')
    elif (pose_recognition_class==liste[1]) and (Prob>0.7):
        print('Gesture 2')
    elif (pose_recognition_class==liste[2]) and (Prob>0.6):
        print('Gesture 3')
    elif (pose_recognition_class==liste[3]) and (Prob>0.7):
        print('Gesture 4')
    elif (pose_recognition_class==liste[4]) and (Prob>0.7):
        print('Gesture 5')

def countdown(num_of_secs):
    
    while (not rospy.is_shutdown() and num_of_secs!=0):
        m, s = divmod(num_of_secs, 60)
        min_sec_format = '{:02d}:{:02d}'.format(m, s)
        print(min_sec_format)
        time.sleep(1)
        num_of_secs -= 1

def Recognition ():
    
    with open(f'/home/tanguy/tanguy_ws/src/mediapipe_gesture_recognition/trained_models/trained_model.pkl', 'rb') as f:
        model = pickle.load(f) 

    Landmarks = []
    if (enable_right_hand == True):
        for i in range (len(right_new_msg.keypoints)):
            Landmarks.extend([right_new_msg.keypoints[i].x, right_new_msg.keypoints[i].y, right_new_msg.keypoints[i].z, right_new_msg.keypoints[i].v])

    if (enable_left_hand == True):
        for i in range (len(left_new_msg.keypoints)):
            Landmarks.extend([left_new_msg.keypoints[i].x, left_new_msg.keypoints[i].y, left_new_msg.keypoints[i].z, left_new_msg.keypoints[i].v])

    if (enable_pose == True):
        for i in range (len(pose_new_msg.keypoints)):
            Landmarks.extend([pose_new_msg.keypoints[i].x, pose_new_msg.keypoints[i].y, pose_new_msg.keypoints[i].z, pose_new_msg.keypoints[i].v])
    
    #if (enable_face = True):
        #for i in range (len(face_new_msg.keypoints)):
        #    Landmarks.extend([face_new_msg.keypoints[i].x, face_new_msg.keypoints[i].y, face_new_msg.keypoints[i].z, face_new_msg.keypoints[i].v])

    #print(Landmarks)
    

    X = pd.DataFrame([Landmarks])             #landmarks = list with all the coordinates detected in the stream
    pose_recognition_class = model.predict(X)[0]
    pose_recognition_prob = model.predict_proba(X)[0]
    #print(pose_recognition_class, pose_recognition_prob)


    Prob=max(pose_recognition_prob)
    if (Prob>0.7):
        Name = pose_recognition_class.rstrip(pose_recognition_class[-4])
        print(Name)
    else:
        print('No gesture recognized')

# ROS Initialization
rospy.init_node('mediapipe_gesture_recognition_training_node', anonymous=True) 
rate = rospy.Rate(30)

# Mediapipe Subscribers
rospy.Subscriber('/mediapipe_gesture_recognition/right_hand', Hand, handRightCallback)
rospy.Subscriber('/mediapipe_gesture_recognition/left_hand', Hand, handLeftCallback)
rospy.Subscriber('/mediapipe_gesture_recognition/pose', Pose, PoseCallback)
rospy.Subscriber('/mediapipe_gesture_recognition/face', Face, FaceCallback)

# Read Mediapipe Modules Parameters
enable_right_hand = rospy.get_param('enable_right_hand', False)
enable_left_hand = rospy.get_param('enable_left_hand', False)
enable_pose = rospy.get_param('enable_pose', False)
enable_face = rospy.get_param('enable_face', False)


print("\nRecognition starts in:")
countdown(5)
print("\nSTART\n")

while not rospy.is_shutdown():
    
    Recognition()