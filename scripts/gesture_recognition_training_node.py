#!/usr/bin/env python3

from asyncio.base_futures import _FINISHED
from re import I
from tkinter import Variable
from turtle import delay, right
from typing import Counter
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
import time
from mediapipe_gesture_recognition.msg import Pose, Face, Hand
from mediapipe_stream_node import MediapipeStreaming
import pyarrow.parquet as pq
import mysql.connector
from mysql.connector import Error


# Mediapipe Subscribers Callback
def handRightCallback(data): #Have to read the datas from the topic and copy the content of the datas to global variable 
                                    #and update that variable every time a message is received
                                    #use the callback only for read and use the global variable in the main loop
    #print('-----------------------------------')
    #print('Header', datas.header)  
    #print('---')
    #print('Right or Left', datas.right_or_left)
    #print('---')
    #print('Keypoints', datas.keypoints) #msg.keypoints[i]

    #BUG : save the incoming message in a global variable and when we are recording, save the message in the csv file
    global count  
    count = []
    for i in range(len(data.keypoints)):
        count.append(data.keypoints[i])

    global right_new_msg
    right_new_msg = []
    for i in range(len(data.keypoints)):
        right_new_msg.append(data.keypoints[i].keypoint_number)
        right_new_msg.append(data.keypoints[i].keypoint_name)
        right_new_msg.append(data.keypoints[i].x)
        right_new_msg.append(data.keypoints[i].y)
        right_new_msg.append(data.keypoints[i].z)
        right_new_msg.append(data.keypoints[i].v)


def handLeftCallback(data):
    global left_new_msg 
    left_new_msg = data 

def PoseCallback(data):
    global pose_new_msg 
    pose_new_msg = data

def FaceCallback(data):
    global face_new_msg 
    face_new_msg = data


#MySQL fonctions
def create_server_connection(host_name, user_name, user_password):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password
        )
        print("MySQL server connection successful")
    except Error as err:
        print(f"Error: '{err}'")

    return connection

def create_db_connection(host_name, user_name, user_password,db_name):  #exemple: connection = create_server_connection("localhost", "root", pw, "school")
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name
        )
        print("MySQL Database connection successful")
    except Error as err:
        print(f"Error: '{err}'")

    return connection

def create_database(connection, query):     #exemple:   create_database_query = "CREATE DATABASE school"
    cursor = connection.cursor()                       #create_database(connection, create_database_query)
    try:
        cursor.execute(query)
        print("Database created successfully")
    except Error as err:
        print(f"Error: '{err}'")

def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query successful")
    except Error as err:
        print(f"Error: '{err}'")

#Creation of the MySQL Database
def createMySQLdatabase():
    #SQL query to create the database:
    create_database_query = "CREATE DATABASE IF NOT EXISTS Gesture_recognition" 
    create_database(connection, create_database_query)  #Create database

#Creation of the MySQL table
def createMySQLtable():
    #SQL query to create the table:
    delete_table = "DROP TABLE Landmarks;"
    
    create_landmarks_table = """ 
        CREATE TABLE IF NOT EXISTS Landmarks ( 
        id int(10) NOT NULL AUTO_INCREMENT,
        keypoint_number0 int(6) DEFAULT NULL, 
        keypoint_name0 varchar(100) DEFAULT NULL, 
        x0 float(24) DEFAULT NULL, 
        y0 float(24) DEFAULT NULL, 
        z0 float(24) DEFAULT NULL, 
        v0 float(24) DEFAULT NULL, 
        PRIMARY KEY(id)
        ); """

    cursor = connection.cursor()
    cursor.execute(delete_table)
    cursor.execute(create_landmarks_table)
    connection.commit()
    print ("Table created successfully")
    #connection = create_db_connection("localhost", "root", '','Gesture_recognition') # Connect to the Database
    #execute_query(connection, create_landmarks_table)   #Create "Landmarks" table in the database
    
    for i in range (1, 21): #BUG : adapt the range to the exact number of landmarks
        add_columns = """
        ALTER TABLE Landmarks ADD COLUMN (
            keypoint_number%s int(6) DEFAULT NULL, 
            keypoint_name%s varchar(100) DEFAULT NULL, 
            x%s float(24) DEFAULT NULL, 
            y%s float(24) DEFAULT NULL, 
            z%s float(24) DEFAULT NULL, 
            v%s float(24) DEFAULT NULL 
            );"""
        
        #SQL query to add columns for all the landmarks coordinates
        cursor = connection.cursor()
        cursor.execute(add_columns, (i, i, i, i, i, i))
        connection.commit()
        print ("Table updated successfully")
        #execute_query(connection, add_columns)

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

        #exemple : "INSERT INTO favourite (number, info) VALUES (%s, %s)", (numbers, animals))


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

def countdown(num_of_secs):
    while (num_of_secs!=0):
        m, s = divmod(num_of_secs, 60)
        min_sec_format = '{:02d}:{:02d}'.format(m, s)
        print(min_sec_format)
        time.sleep(1)
        num_of_secs -= 1
        
    print('Countdown finished')

#Initialisation
rospy.init_node('mediapipe_streamgesture_recognition_training_node', anonymous=True) 
rate = rospy.Rate(100)

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


#SETTINGS STEP

# While ROS OK
while not rospy.is_shutdown(): # 2 parts : recording phase and then training phase
    #RECORDING PHASE
    
    # Write the project name on a TXT file
    File_Name=input("What is your project name ? ")
    Project_name_txt=open('/home/tanguy/tanguy_ws/src/mediapipe_gesture_recognition/TXT file/projectname.txt','w')
    Project_name_txt.write(File_Name)
    Project_name_txt.close()
    
    Solution_Choice=input("\nWhat is the name of this training ?")

 
    connection = create_server_connection("localhost", "root", '')

    #Setup positions
    nbr_pos=int(input("How many position do you want to setup? (min of 2) "))
    #n=0
    name_position=[] #List with the names of the differents positions
    createMySQLdatabase()
    
    for n in range(nbr_pos):
    #while (n<nbr_pos):

        if rospy.is_shutdown(): break

        #n+=1
        class_name=input("What's the name of your position ?")
        name_position.append(class_name)
        
        connection = create_db_connection("localhost", "root", '', 'Gesture_recognition')   #Connecton to the database "Gesture_recognition" 

        createMySQLtable()
        
        #Creation of a 5s counter
        print("The acquisition will start in 5s")
        countdown(5)

        #Recognise gesture for 30 seconds with another counter
        print("Start of the acquisition")
        
        start=rospy.Time.now()                             #Variable where we stock time
        while(not rospy.is_shutdown() and (rospy.Time.now()-start).to_sec()<5):
            
            for i in range (len(count)):
                pop_landmarks = """INSERT INTO Landmarks (keypoint_number%s, keypoint_name%s, x%s, y%s, z%s, v%s) 
                                    VALUES (%s, %s, %s, %s, %s, %s);"""
                cursor = connection.cursor()
                cursor.execute(pop_landmarks, (i, i, i, i, i, i, right_new_msg[i*6], right_new_msg[i*6+1], right_new_msg[i*6+2], right_new_msg[i*6+3], right_new_msg[i*6+4], right_new_msg[i*6+5]))
                connection.commit()
                print ("value inserted")

        print("End of the acquisition for this position")

    print('the',nbr_pos,'positions have been saved')
    #TRAINING PHASE : load database, 
    #train_model()

    # Sleep for the Remaining Cycle Time
    rate.sleep() 

    break




# 1 finish to setup the sql for each gesture
# try to record 3/4 gesture with only right hand
# train gesture recognition and try it with those gestures

# implement left hand, face, skeleton 
# setup different models: right + left or right + face or face + skeleton...

# record everythin: left, right, hend, skeleton
# train different models removing datas from temp sql