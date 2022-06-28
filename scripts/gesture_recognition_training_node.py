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
from os.path import isfile, join

# Import Mediapipe Messages
from mediapipe_gesture_recognition.msg import Pose, Face, Hand

# Get Package Path
package_path = rospkg.RosPack().get_path('mediapipe_gesture_recognition')

right_new_msg = Hand()
left_new_msg = Hand() 
pose_new_msg = Pose()
face_new_msg = Face()


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
#                Recording phase functions                 #
############################################################


def createDataFrame(msg):
    
    # Initialize Message Table
    table = []
    
    # Append Keypoint Messages
    for k in msg.keypoints: table.append([k.keypoint_number, k.keypoint_name, k.x, k.y, k.z, k.v])
    
    df = pd.DataFrame(table, columns=['Keypoint Number', 'Keypoint Name', 'x', 'y', 'z', 'visibility'])
    
    return df

def msg2dict(dict, right_hand, left_hand, pose, face):
    
    # New Message Number
    n = len(dict) + 1
        
    # Append New Data to Dict
    dict[f'Data {n}'] = {'Right Hand' : createDataFrame(right_hand), 
                         'Left Hand'  : createDataFrame(left_hand), 
                         'Pose'       : createDataFrame(pose), 
                         'Face'       : createDataFrame(face)}
    
    return dict

def saveGesture(data_dictionary, name):
    global gesture_file
    gesture_file = ''
    if enable_right_hand_ == True :
        gesture_file = gesture_file + "Right"
    if enable_left_hand_== True:
        gesture_file = gesture_file + "Left"
    if enable_pose_== True:
        gesture_file = gesture_file + "Pose"
    if enable_face_ == True:
        gesture_file = gesture_file + "Face"

    # Save Dict Object with Pickle
    with open(f'{package_path}/database/Gestures/{gesture_file}/{name}.pkl', 'wb') as savefile:
        pickle.dump(data_dictionary, savefile, protocol = pickle.HIGHEST_PROTOCOL)
        # print(data_dictionary)

    print('Gesture Saved')

def debugPrint(data_dictionary):
    
    # Print All Dictionary
    print(data_dictionary)
    
    # Process All Keys in a Dictionary
    for key in data_dictionary: 
        
        # Print All Data of a Message
        print(data_dictionary[key])
        
        # Print Each Dataframe
        print('\nRight Hand:\n', data_dictionary[key]['Right Hand'])
        print('\nLeft Hand:\n', data_dictionary[key]['Left Hand'])
        print('\nPose:\n', data_dictionary[key]['Pose'])
        print('\nFace:\n', data_dictionary[key]['Face'])
        
        # Access to Each Data in 'Right Hand' Dataframe
        print('\nRight Hand:\n', data_dictionary[key]['Right Hand'].loc[0])

def countdown(num_of_secs):
    
    while (not rospy.is_shutdown() and num_of_secs!=0):
        m, s = divmod(num_of_secs, 60)
        min_sec_format = '{:02d}:{:02d}'.format(m, s)
        print(min_sec_format)
        time.sleep(1)
        num_of_secs -= 1


############################################################
#                  Training phase functions                #
############################################################


def loadGesture(name):
        
    try:
            
        # Read Dict Object with Pickle
        with open(f'{package_path}/database/Gestures/{gesture_file}/{name}', 'rb') as inputfile:
            reading = pickle.load(inputfile)
            #print(reading)

            print(f'Gesture "{name}" Loaded')
            return reading
    
    except: 
        
        print(f'ERROR: Failed to Load Gesture "{name}"')
        return False

def adapt_dictionary(dictionary, name_position):

    list_landmarks = []     #'Right Hand', 'Left Hand', 'Pose' and/or 'Face'
    list_df=[]              #list with the related dfs (df_right_hand, df_left_hand, df_pose and/or df_face)
    if enable_right_hand_ ==True :
        list_landmarks.append('Right Hand')
    if enable_left_hand_==True:
        list_landmarks.append('Left Hand')
    if enable_pose_==True:
        list_landmarks.append('Pose')
    if enable_face_==True:
        list_landmarks.append('Face')
    
    for part in list_landmarks:
        print (part)
        df = pd.DataFrame()
        for i in range (len(dictionary['Data 1'][part])):
            df = pd.concat([df,dictionary['Data 1'][part].loc[i]], axis = 0)
        df=df.T

        #Add all the rows
        for key in dictionary:
            temporary_df = pd.DataFrame()
            for i in range (len(dictionary[key][part])):
                temporary_df = pd.concat([temporary_df,dictionary[key][part].loc[i]], axis = 0)
            temporary_df=temporary_df.T
            df = pd.concat([df, temporary_df], axis = 0)

        #Drop the unwanted datas
        df.drop(index = 0)
        del df['Keypoint Number']
        del df['Keypoint Name']
        #print(df)
        
        #Store the df in the correct variable
        if (part == 'Right Hand'):
            df_right_hand = df
            list_df.append(df_right_hand)
        if (part == 'Left Hand'):
            df_left_hand = df
            list_df.append(df_left_hand)
        if (part == 'Pose'):
            df_pose = df
            list_df.append(df_pose)
        if (part == 'Face'):
            df_face = df
            list_df.append(df_face)

    
    concat_df = pd.concat(list_df, axis =1)
    
    list_position=[]
    list_position = [name_position] * len((concat_df))
    list_position = pd.DataFrame (list_position)
    concat_df = list_position.join(concat_df, how='right')
  
    
    #list_position=[]
    #for i in range (len(concat_df)):
    #    list_position.append(name_position)
    #concat_df.insert(0, 'Position', list_position)
    #print (concat_df)
    
    return concat_df

def train_model():
    # Obtain the files for the positions saved  
    Saved_positions = [f for f in listdir(f'{package_path}/database/Gestures/{gesture_file}/') if isfile(join(f'{package_path}/database/Gestures/{gesture_file}/', f))]
    
    #Create a Dataframe with the values of all the positions saved
    list_df_fragments =[]
    for i in range(len(Saved_positions)):
        position_dictionary = loadGesture(Saved_positions[i])
        df = adapt_dictionary(position_dictionary, Saved_positions[i])
        list_df_fragments.append(df)
    df = pd.concat(list_df_fragments, axis =0)
    
    list_renamed_columns = ['Position']
    x = int((len(df.columns)-1)/4)
    for i in range(x):
        list_renamed_columns += ['x{}'.format(i), 'y{}'.format(i), 'z{}'.format(i), 'v{}'.format(i)]
    df.columns=[list_renamed_columns]
    print (df)
    
    # 3.1/ READ IN COLLECTED DATA AND PROCESS  
    X = df.drop(['Position'], axis = 1)                                                                     # only show the features, like, only the coordinates not the class name
    y = df['Position']                                                                                      # only show the target value witch is basically the class name
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)             #Take large random value with train and take small random value with test

    # 3.2/ TRAIN MACHINE LEARNING CLASSIFICATION MODEL

    pipelines = {                                                                                           #Create different pipelines, here you have 4 different machine learning model, later we will choose the best one
        'lr':make_pipeline(StandardScaler(), LogisticRegression()),
        'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
        'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
        'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
                }

    fit_models = {}                         
    for algo, pipeline in pipelines.items():
        model = pipeline.fit(X_train, y_train)
        fit_models[algo] = model                                                                            #this 4 lines is to run the automatic learning

    # 3.3/ EVALUATE AN SERIALIZE MODEL

    for algo, model in fit_models.items():
        yhat = model.predict(X_test)
        print(algo, accuracy_score(y_test, yhat))                                                           #These line is to predict and showed the precision of the 4 pipelines, to choose witch one is the preciser

    with open(f'{package_path}/database/Gestures/{gesture_file}/trained_model.pkl', 'wb') as savefile:                        #These two lines is build to export the best model "here it's rf" and save it in a files called trained_model.pkl
        pickle.dump(fit_models['rf'], savefile, protocol = pickle.HIGHEST_PROTOCOL)
        # print(data_dictionary)
    
    
    print('Model trained successfully')


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

# Read Training Parameters
recording_phase = rospy.get_param('recording', True)
training_phase = rospy.get_param('training', False)


############################################################
#                           Main                           #
############################################################


###     RECORDING PHASE     ###
while not rospy.is_shutdown() and recording_phase:

    # DataFrames Dictionary Initialization
    data = {} 
    
    # Gesture Labeling
    gesture_name = input("\nInsert Gesture Name: ")
    print(pose_new_msg)
    # Start Counter
    print("\nAcquisition Starts in:")
    countdown(5)
    print("\nSTART\n")    
    
    # Starting Time
    start = rospy.Time.now()         
    ACQUISITION_TIME = 10
    
    # Recognise gesture for "ACQUISITION_TIME" seconds with another counter
    while(not rospy.is_shutdown() and (rospy.Time.now() - start).to_sec() < ACQUISITION_TIME):

        # Add Incoming Messages to Dataframe
        data = msg2dict(data, right_new_msg, left_new_msg, pose_new_msg, face_new_msg)
        
        # Sleep for the Remaining Cycle Time (30 FPS)
        rate.sleep()
            
    print("End of The Acquisition, Saving...")
    #print(data)

    # Save Gesture Recorded Data
    saveGesture(data, gesture_name)
    
    if not input('\nStart Another Gesture Acquisition ? [Y/N]: ') in ['Y','y']:
        
        # ALLOW TRAINING PHASE
        if input('\nStart Model Training ? [Y/N]: ') in ['Y','y']: training_phase = True
        
        # STOP RECORDING PHASE
        recording_phase = False
        break
        


###     TRAINING PHASE     ###
if not rospy.is_shutdown() and training_phase:
    train_model
