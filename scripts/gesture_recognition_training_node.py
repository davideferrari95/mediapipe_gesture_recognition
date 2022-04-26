#!/usr/bin/env python3

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
def handRightCallback(datas): #Have to read the datas from the topic and copy the content of the datas to global variable 
                                    #and update that variable every time a message is received
                                    #use the callback only for read and use the global variable in the main loop
    print('-----------------------------------')
    print('Header', datas.header)  
    print('---')
    print('Right or Left', datas.right_or_left)
    print('---')
    print('Keypoints', datas.keypoints) #msg.keypoints[i]

    #BUG : save the incoming message in a global variable and when we are recording, save the message in the csv file
    right_new_msg = datas
    

    ###for i in range (len(right_hand.keypoints)):
    ###    with open(f'/home/tanguy/tanguy_ws/src/mediapipe_gesture_recognition/CSV files/{File_Name}.csv', mode='a', newline='') as f:         #Write the list into the CSV file
    ###      csv.writer.writeheader()
    ###      csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    ###      csv_writer.writerow({ 'x{}'.format(i): right_hand.keypoints[i].x, 'y{}'.format(i): right_hand.keypoints[i].y,
    ###                            'z{}'.format(i): right_hand.keypoints[i].z, 'v{}'.format(i): right_hand.keypoints[i].v,
    ###                            'keypoint_number{}'.format(i): right_hand.keypoints[i].keypoint_number, 'keypoint_name{}'.format(i): right_hand.keypoints[i].keypoint_name })   


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

def handLeftCallback(datas):
    print('-----------------------------------')
    print('Header', datas.header)  

def PoseCallback(datas):
    print('-----------------------------------')
    print('Header', datas.header)  

def FaceCallback(datas):
    print('-----------------------------------')
    print('Header', datas.header)  


#MySQL fonctions
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
def createMySQLfiles():
    #SQL query to create the database:
    create_database_query = "CREATE DATABASE Gesture_recognition" 

    #SQL query to create the table:
    create_landmarks_table = """ 
        CREATE TABLE IF NOT EXISTS Landmarks ( 
        id int(10) NOT NULL,
        keypoint_number int(6) DEFAULT NULL, 
        keypoint_name varchar(100) DEFAULT NULL, 
        x float(24) DEFAULT NULL, 
        y float(24) DEFAULT NULL, 
        z float(24) DEFAULT NULL, 
        v float(24) DEFAULT NULL, 
        PRIMARY KEY(id), 
        ); """

    #SQL query to add columns for all the landmarks coordinates
    add_columns = """ALTER TABLE Landmarks ADD (
        keypoint_number int(6) DEFAULT NULL, 
        keypoint_name varchar(100) DEFAULT NULL, 
        x float(24) DEFAULT NULL, 
        y float(24) DEFAULT NULL, 
        z float(24) DEFAULT NULL, 
        v float(24) DEFAULT NULL, 
        );"""

    create_database(connection, create_database_query)  #Create database
    connection = create_db_connection("localhost", "root", '','Gesture_recognition') # Connect to the Database
    execute_query(connection, create_landmarks_table)   #Create "Landmarks" table in the database
    for i in range (468): #BUG : adapt the range to the exact number of landmarks
        execute_query(connection, add_columns)

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

def countdown(num_of_secs):
    while num_of_secs:
        m, s = divmod(num_of_secs, 60)
        min_sec_format = '{:02d}:{:02d}'.format(m, s)
        print(min_sec_format, end='/r')
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

    #Setup positions
    nbr_pos=int(input("How many position do you want to setup? (min of 2) "))
    i=0
    name_position=[]
    while (i<nbr_pos):
        i+=1
        class_name=input("What's the name of your position ?")
        name_position.append(class_name)
        
        #Creation of a 5s counter
        print("The acquisition will start in     5s")
        countdown(5)

        #Recognise gesture for 30 seconds with another counter
        print("Start of the acquisition")
        countdown(30)
        handRightCallback()
        createMySQLfiles()
        
        connection = create_db_connection("localhost", "root", '', 'Gesture_recognition')
 
        while():
            pop_landmarks = """ INSERT INTO Landmarks VALUES
            (datas.keypoint),
            """
            execute_query(connection, pop_landmarks)           



        print("End of the acquisition for this position")


    #TRAINING PHASE : load database, 
    train_model()

    

    # Sleep for the Remaining Cycle Time
    rate.sleep() 




