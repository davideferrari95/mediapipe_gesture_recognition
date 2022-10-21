#!/usr/bin/env python3

#Global imports
import os
import rospy, rospkg
import time
import pandas as pd, pickle
import numpy as np

# Obtain files from a directory
from os import listdir
from os.path import join, isdir

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
#                 Recording phase functions                #
############################################################


def extract_keypoints(pose_msg, face_msg, left_msg, right_msg):
    pose = np.array([[res.x, res.y, res.z, res.v] for res in pose_msg.keypoints]).flatten() if pose_new_msg else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in face_msg.keypoints]).flatten() if face_new_msg else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in left_msg.keypoints]).flatten() if left_new_msg else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in right_msg.keypoints]).flatten() if right_new_msg else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

#Create a folder for each gesture, this folder contains folders for each videos we will record
def create_folders(gesture):
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(f'{package_path}/database/3D_Gestures/{gesture_file}/', gesture, str(sequence)))
        except:
            pass

#Create a loop to save the landmarks coordinates for each frame of the video (each video is made of the number of frames defined in the variable "sequence_length")
#Then loop to save all the videos for the gesture (the number of videos for training is defined by the variable "no_sequences")
def store_videos(gesture):
    # Loop through sequences aka videos
    for sequence in range(no_sequences):
        print("\nCollecting frames for {}  Video Number {}".format(gesture, sequence))
        print("Collection starting in 2 seconds")
        countdown(2)
        print("Starting collection")
        start = rospy.get_time() 
        
        # Loop through video length aka sequence length
        for frame_num in range(sequence_length):
            # Export keypoints values in the correct folder
            keypoints = extract_keypoints(pose_new_msg, face_new_msg, left_new_msg, right_new_msg)
            npy_path = os.path.join(f'{package_path}/database/3D_Gestures/{gesture_file}/', gesture, str(sequence), str(frame_num))
            np.save(npy_path, keypoints)
            time.sleep(1/30)            #To have 30 frames/s
        
        end= rospy.get_time() 
        acquisition_time = end-start
        print ("Lenght of the acquisition : ", acquisition_time)
        print("End collection")
        time.sleep(1)

#Create a countdown (the duration of the countdown will be defined by the variable "num_of_seconds")
def countdown(num_of_secs):
    
    while (not rospy.is_shutdown() and num_of_secs!=0):
        m, s = divmod(num_of_secs, 60)
        min_sec_format = '{:02d}:{:02d}'.format(m, s)
        print(min_sec_format)
        time.sleep(1)
        num_of_secs -= 1


############################################################
#                 Training phase functions                 #
############################################################


def train_3D_model():
    
    # Preprocess datas and create labels and features
    from sklearn.model_selection import train_test_split
    from keras.utils import to_categorical

    actions = [f for f in listdir(f'{package_path}/database/3D_Gestures/{gesture_file}/') if isdir(join(f'{package_path}/database/3D_Gestures/{gesture_file}/', f))]
    actions = np.array(actions)
    print (actions)
    
    label_map = {label:num for num, label in enumerate(actions)}
    
    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(f'{package_path}/database/3D_Gestures/{gesture_file}/', action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
    
    print(np.array(sequences).shape)
    print(np.array(labels).shape)

    X = np.array(sequences)
    print(X.shape)
    
    y = to_categorical(labels).astype(int)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    print(X_test.shape[2])
    print(y_test.shape)


    # Neural network with tensorflow
    from tensorflow import keras
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    from keras.callbacks import TensorBoard
    from keras.callbacks import EarlyStopping

    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    early_stopping = EarlyStopping(monitor = 'loss', patience = 15)

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,X_test.shape[2])))
    model.add(Dropout(0.5))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(Dropout(0.5))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(actions.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback, early_stopping])
    
    #Save the model in a file called trained_model.pkl
    with open(f'{package_path}/database/3D_Gestures/{gesture_file}/trained_model.pkl', 'wb') as savefile:
        pickle.dump(model, savefile, protocol = pickle.HIGHEST_PROTOCOL)


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

# Number of videos worth of data
no_sequences = 30

# Number of frames for each videos (30 frames/s) * lenght of the video in seconds
sequence_length = 30*1


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


###     RECORDING PHASE     ###

while not rospy.is_shutdown() and recording_phase:
    
    train_3D_model()

    # Gesture Labeling
    gesture_name = input("\nInsert Gesture Name: ")

    # Create folders for the videos
    create_folders(gesture_name)
    
    # Start Counter
    print("\nAcquisition Starts in:")
    countdown(5)
    
    #Tape and save videos
    store_videos(gesture_name)
    
    print('All the videos have been saved')
    
    if not input('\nStart Another Gesture Acquisition ? [Y/N]: ') in ['Y','y']:
        
        # ALLOW TRAINING PHASE
        if input('\nStart Model Training ? [Y/N]: ') in ['Y','y']: training_phase = True
        
        # STOP RECORDING PHASE
        recording_phase = False
        break
        

###     TRAINING PHASE     ###

if not rospy.is_shutdown() and training_phase:
    train_3D_model()
