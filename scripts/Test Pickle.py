#!/usr/bin/env python3

import rospy, rospkg
import os, time
import pandas as pd, pickle

# Import Mediapipe Messages
from mediapipe_gesture_recognition.msg import Pose, Face, Hand

# Get Package Path
package_path = rospkg.RosPack().get_path('mediapipe_gesture_recognition')

right_new_msg = Hand()
left_new_msg = Hand() 
pose_new_msg = Pose()
face_new_msg = Face()

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
    
    save_path = f'{package_path}/database/PKL'
    
    if not os.path.exists(save_path): os.makedirs(save_path) 
    
    # Save Dict Object with Pickle
    with open(f'{save_path}/{name}.pkl', 'wb') as savefile:
        pickle.dump(data_dictionary, savefile, protocol = pickle.HIGHEST_PROTOCOL)
        # print(data_dictionary)

    print('Gesture Saved')

def loadGesture(name):
        
    try:
            
        # Read Dict Object with Pickle
        with open(f'{package_path}/database/PKL/{name}.pkl', 'rb') as inputfile:
            reading = pickle.load(inputfile)
            # print(reading)

            print(f'Gesture "{name}" Loaded')
            return reading
    
    except: 
        
        print(f'ERROR: Failed to Load Gesture "{name}"')
        return False

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

# Read Training Parameters
recording_phase = rospy.get_param('recording', True)
training_phase = rospy.get_param('training', False)

# RECORDING PHASE
while not rospy.is_shutdown() and recording_phase:
    
    # DataFrames Dictionary Initialization
    data = {} 

    # Gesture Labeling
    gesture_name = input("\nInsert Gesture Name: ")

    # DEBUG: Increase Counter to 5
    # Start Counter
    print("\nAcquisition Starts in:")
    countdown(1)
    print("\nSTART\n")    
    
    # DEBUG: Increase Acquisition Time to 30
    # Starting Time
    start = rospy.Time.now()         
    ACQUISITION_TIME = 1/30
    
    # Recognise gesture for 30 seconds with another counter
    while(not rospy.is_shutdown() and (rospy.Time.now() - start).to_sec() < ACQUISITION_TIME):

        # Add Incoming Messages to Dataframe
        data = msg2dict(data, right_new_msg, left_new_msg, pose_new_msg, face_new_msg)
        
        rospy.loginfo_throttle(1, f'Recording...  {int(ACQUISITION_TIME - (rospy.Time.now() - start).to_sec())}')
        
        # Sleep for the Remaining Cycle Time (30 FPS)
        rate.sleep()
            
    print("\nEnd of The Acquisition, Saving...")
    
    # Save Gesture Recorded Data
    saveGesture(data, gesture_name)
    
    if not input('\nStart Another Gesture Acquisition ? [Y/N]: ') in ['Y','y']:
        
        # ALLOW TRAINING PHASE
        if input('\nStart Model Training ? [Y/N]: ') in ['Y','y']: training_phase = True
        
        # STOP RECORDING PHASE
        recording_phase = False
        break
        

# TRAINING PHASE 
if not rospy.is_shutdown() and training_phase:
    
    # train_model()
    ...
