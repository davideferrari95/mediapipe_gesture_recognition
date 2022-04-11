#!/usr/bin/env python3

from turtle import right
import rospy
import csv
import cv2
import numpy as np
import pandas as pd 
import mediapipe as mp
from sklearn.model_selection import train_test_split 
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle 
import os

from mediapipe_gesture_recognition.msg import Pose, Face, Hand
from mediapipe_stream_node import MediapipeStreaming



# Mediapipe Subscribers Callback
def handRightCallback(right_hand): 
    print('-----------------------------------')
    print('Header', right_hand.header)  
    print('---')
    print('Right or Left', right_hand.right_or_left)
    print('---')
    print('Keypoints', right_hand.keypoints) #msg.keypoints[i]

    #Setup the positions
    nbr_pos=int(input("How many position do you want to setup? (min of 2) "))
    i=0
    name_position=[]
    while (i<nbr_pos):
        i+=1
        class_name=input("What's the name of your position ?")
        name_position.append(class_name)
        print("Press q when your position is setup") #BUG : Need to creat a line where we stop the code with the letter q

        for i in range (len(right_hand.keypoints)):
            with open(f'/home/tanguy/tanguy_ws/src/mediapipe_gesture_recognition/CSV files/{File_Name}.csv', mode='a', newline='') as f:         #Write the list into the CSV file
              csv.writer.writeheader()
              csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
              csv_writer.writerow({ 'x': right_hand.keypoints[i].x, 'y': right_hand.keypoints[i].y,
                                    'z': right_hand.keypoints[i].z, 'v': right_hand.keypoints[i].v,
                                    'keypoint_number': right_hand.keypoints[i].keypoint_number, 'keypoint_name': right_hand.keypoints[i].keypoint_name})   


    #Insert the landmarks coordinates in the csv file

    '''
    with open('names.csv', 'w', newline='') as csvfile:
    fieldnames = ['first_name', 'last_name']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerow({'first_name': 'Baked', 'last_name': 'Beans'})
    writer.writerow({'first_name': 'Lovely', 'last_name': 'Spam'})
    writer.writerow({'first_name': 'Wonderful', 'last_name': 'Spam'})
    '''

def handLeftCallback(left_hand):
    print('-----------------------------------')
    print('Header', left_hand.header)  
    print('---')
    print('Right or Left', left_hand.right_or_left)
    print('---')
    print('Keypoints', left_hand.keypoints) #msg.keypoints[i]

def PoseCallback(pose):
    print('-----------------------------------')
    print('Header', pose.header)  
    print('---')
    print('Right or Left', pose.right_or_left)
    print('---')
    print('Keypoints', pose.keypoints) #msg.keypoints[i]

def FaceCallback(face):
    print('-----------------------------------')
    print('Header', face.header)  
    print('---')
    print('Right or Left', face.right_or_left)
    print('---')
    print('Keypoints', face.keypoints) #msg.keypoints[i]

#Creation of the CSV file where the datas will be saved
def createfiles():
        # Firstly I create some txt files to save the parameters of this session to use it in the following nodes
        


        # Write the different solution used on a TXT file
        Solution=""
        if (enable_right_hand=="enable"):
            Solution= Solution + "Right"
        if (enable_left_hand=="enable"):
            Solution= Solution + " Left"
        if (enable_pose=="enable"):
            Solution= Solution + " Pose"
        Solution_txt=open(f'/home/tanguy/tanguy_ws/src/mediapipe_gesture_recognition/TXT file/Solution_Name/Solution_{File_Name}.txt','w')
        Solution_txt.write(Solution)
        Solution_txt.close()
        

        # In a second time I create the structure of my CSV file
        # Depending of the parameters we don't have the same numbers of landmarks so the structure of the CSV file will change
        # This will create   the first line of the CSV file with on the first column the name of the class and after the (x,y,z,v) coordinates of the first landmarks, second ...

        landmarks = ['class']
        for i in range (478+33+21+21+1): # BUG : The maximum value is 170
            landmarks += ['x{}'.format(i), 'y{}'.format(i), 'z{}'.format(i), 'v{}'.format(i), 'keypoint_number{}'.format(i), 'keypoint_name{}'.format(i)]
        with open(f'/home/tanguy/tanguy_ws/src/mediapipe_gesture_recognition/CSV files/{File_Name}.csv', mode='w', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting= csv.QUOTE_MINIMAL)
            csv_writer.writerow(landmarks)

#3/ TRAIN CUSTOM MODEL USING SCIKIT LEARN
def train_model():
    # 3.1/ READ IN COLLECTED DATA AND PROCESS

    df = pd.read_csv(f'/home/tanguy/tanguy_ws/src/mediapipe_gesture_recognition/CSV files/{Solution_Choice}.csv')       #read the coordinates on the CSV file 
    X = df.drop('class', axis=1)                                                                         # only show the features, like, only the coordinates not the class name
    y = df['class']                                                                                      # only show the target value witch is basically the class name
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)          #Take large random value with train and take small random value with test


    # 3.2/ TRAIN MACHINE LEARNING CLASSIFICATION MODEL

    pipelines = {                                                                                         #Create different pipelines, here you have 4 different machine learning model, later we will choose the best one
        'lr':make_pipeline(StandardScaler(), LogisticRegression()),
        'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
        'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
        'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
                }

    fit_models = {}                         
    for algo, pipeline in pipelines.items():
        model = pipeline.fit(X_train, y_train)
        fit_models[algo] = model                                                                          #this 4 lines is to run the automatic learning

    # 3.3/ EVALUATE AN SERIALIZE MODEL

    for algo, model in fit_models.items():
        yhat = model.predict(X_test)
        print(algo, accuracy_score(y_test, yhat))                                                            #These line is to predict and showed the precision of the 4 pipelines, to choose witch one is the preciser

    with open(f'/home/tanguy/tanguy_ws/src/mediapipe_gesture_recognition/PKL files/{Solution_Choice}.pkl', 'wb') as f:       #These two lines is build to export the best model "here it's rf" and save it in a files called pose_recognition.pkl
        pickle.dump(fit_models['rf'], f)


#Initialisation
rospy.init_node('mediapipe_streamgesture_recognition_training_node', anonymous=True) 
rate = rospy.Rate(100) 
hand_right_pub  = rospy.Publisher('/mediapipe_gesture_recognition/right_hand', Hand, queue_size=1) 
hand_left_pub   = rospy.Publisher('/mediapipe_gesture_recognition/left_hand', Hand, queue_size=1)
pose_pub        = rospy.Publisher('/mediapipe_gesture_recognition/pose', Pose, queue_size=1)
face_pub        = rospy.Publisher('/mediapipe_gesture_recognition/face', Face, queue_size=1)

# Mediapipe Subscribers
rospy.Subscriber('/mediapipe_gesture_recognition/right_hand', Hand, handRightCallback)

# Read Mediapipe Modules Parameters
enable_right_hand = rospy.get_param('enable_right_hand', False)
enable_left_hand = rospy.get_param('enable_left_hand', False)
enable_pose = rospy.get_param('enable_pose', False)
enable_face = rospy.get_param('enable_face', False)


#SETTINGS STEP

# While ROS OK
while not rospy.is_shutdown():
    # Write the project name on a TXT file
    File_Name=input("What is your project name ?")
    Project_name_txt=open('/home/tanguy/tanguy_ws/src/mediapipe_gesture_recognition/TXT file/projectname.txt','w')
    Project_name_txt.write(File_Name)
    Project_name_txt.close()
    
    Solution_Choice=input("\nWhat is the name of this training ?")

    createfiles()

    #train_model()

    handRightCallback()

    # Sleep for the Remaining Cycle Time
    rate.sleep() 