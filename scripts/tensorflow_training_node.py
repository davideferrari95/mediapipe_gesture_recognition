#!/usr/bin/env python3

import os
import time
import rospy
import rospkg
import numpy as np

# Import Pickle for Saving
import pickle

# Tensorflow Neural Network
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
# SkLearn Preprocess Functions
from sklearn import model_selection as SKLearnModelSelection

# Import Mediapipe Messages
from mediapipe_gesture_recognition.msg import Pose, Face, Hand

# Import Utilities
from Utils import countdown


class GestureRecognitionTraining3D:

    ''' 3D Gesture Recognition Training Class '''

    def __init__(self):

        # ROS Initialization
        rospy.init_node('mediapipe_gesture_recognition_training_node', anonymous=True)
        self.rate = rospy.Rate(30)

        # Mediapipe Subscribers
        rospy.Subscriber('/mediapipe_gesture_recognition/right_hand', Hand, self.RightHandCallback)
        rospy.Subscriber('/mediapipe_gesture_recognition/left_hand', Hand, self.LeftHandCallback)
        rospy.Subscriber('/mediapipe_gesture_recognition/pose', Pose, self.PoseCallback)
        rospy.Subscriber('/mediapipe_gesture_recognition/face', Face, self.FaceCallback)

        # Read Mediapipe Modules Parameters
        self.enable_right_hand = rospy.get_param('enable_right_hand', False)
        self.enable_left_hand = rospy.get_param('enable_left_hand',  False)
        self.enable_pose = rospy.get_param('enable_pose', False)
        self.enable_face = rospy.get_param('enable_face', False)

        # Select Gesture File
        self.gesture_file = ''
        if self.enable_right_hand: self.gesture_file += 'Right'
        if self.enable_left_hand: self.gesture_file += 'Left'
        if self.enable_pose: self.gesture_file += 'Pose'
        if self.enable_face: self.gesture_file += 'Face'

        # Read Training Parameters
        self.recording_phase = rospy.get_param('recording', True)
        self.training_phase = rospy.get_param('training', False)

        # Get Package Path
        self.package_path = rospkg.RosPack().get_path('mediapipe_gesture_recognition')

        # Number of videos worth of data
        self.no_sequences = 10

        # Number of frames for each videos (30 frames/s) * length of the video in seconds
        self.sequence_length = 30*1

    # Callback Functions
    def RightHandCallback(self, data): self.right_new_msg: Hand() = data
    def LeftHandCallback(self, data):  self.left_new_msg:  Hand() = data
    def PoseCallback(self, data):      self.pose_new_msg:  Pose() = data
    def FaceCallback(self, data):      self.face_new_msg:  Face() = data

    # A simple countdown function
    def countdown(num_of_secs):

        print("\nAcquisition Starts in:")

        # Wait Until 0 Seconds Remaining
        while (not rospy.is_shutdown() and num_of_secs != 0):

            m, s = divmod(num_of_secs, 60)
            min_sec_format = '{:02d}:{:02d}'.format(m, s)
            print(min_sec_format)
            rospy.sleep(1)
            num_of_secs -= 1

            print("\nSTART\n")


############################################################
#                 Recording Phase Functions                #
############################################################

    # Keypoint Flattening Function (here the missing soruces  pose_msg, face_msg, , right_msg)

    def flattenKeypoints(self, left_msg, right_msg, pose_msg):
        ''' 
        Flatten Incoming Messages of Create zeros Vector \n
        Concatenate each Output
        '''

        # Flatten Incoming Messages of Create zeros Vector
        pose = np.array([[res.x, res.y, res.z, res.v] for res in pose_msg.keypoints]).flatten() if self.pose_new_msg else np.zeros(33*4)
        #face    = np.array([[res.x, res.y, res.z, res.v  for res in face_msg.keypoints]).flatten()  if self.face_new_msg  else np.zeros(468*3)
        left_h = np.array([[res.x, res.y, res.z, res.v] for res in left_msg.keypoints]).flatten() if self.left_new_msg else np.zeros(21*4)
        right_h = np.array([[res.x, res.y, res.z, res.v] for res in right_msg.keypoints]).flatten() if self.right_new_msg else np.zeros(21*4)

        # TODO: Delete new_messages

        print(pose.shape)
        # print(face.shape)
        print(left_h.shape, right_h.shape)

        # Concatenate Data
        # pose, face, left_h, right_h
        return np.concatenate([right_h, left_h, pose])

    # Create Folders for Each Gesture
    def createFolders(self, gesture):
        ''' 
        Create a Folder for Each Gesture \n
        This Folder Contains a Folder for Each Recorded Video
        '''

        for sequence in range(self.no_sequences):
            try:
                os.makedirs(os.path.join(f'{self.package_path}/database/3D_Gestures/{self.gesture_file}/', gesture, str(sequence)))
            except:
                pass

    # Record Videos Function
    def recordVideos(self, gesture):
        '''
        Loop to Save the Landmarks Coordinates for each Frame of the Video
        Each Video contains the Number of Frames defined in "sequence_length" (30FPS)
        Then Loop to Save All the Videos for the Recorded Gesture
        Number of Videos for Training is defined in "no_sequences"
        '''

        # Loop through Sequences (Videos)
        for sequence in range(self.no_sequences):

            print(f'\nCollecting Frames for "{gesture}" | Video Number: {sequence}')
            print('Collection Starting in 2 Seconds')
            countdown(2)
            print('Starting Collection')
            start = rospy.get_time()

            # Loop through Video Length (Sequence Length)
            for frame_num in range(self.sequence_length):

                # Flatten and Concatenate Keypoints (Insert your gesture source here ) (here the missing sources self.pose_new_msg, self.face_new_msg, self.right_new_msg)
                keypoints = self.flattenKeypoints(self.left_new_msg, self.right_new_msg, self.pose_new_msg)

                # Export Keypoints Values in the Correct Folder
                npy_path = os.path.join(f'{self.package_path}/database/3D_Gestures/{self.gesture_file}/', gesture, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Sleep for 30 FPS
                # TODO: Adjust To have 30 frames/s
                time.sleep(1/30)

            print(f'Length of Acquisition : {rospy.get_time() - start}')
            print('End Collection')
            time.sleep(1)

        print('All Videos Saved')

    def Record(self):
        ''' Recording Phase '''

        while not rospy.is_shutdown():

            # Gesture Labeling
            gesture_name = input('\nInsert Gesture Name: ')

            # Create Folders for Each Video
            self.createFolders(gesture_name)

            # Start Counter
            print('\nAcquisition Starts in:')

            countdown(5)

            # Record and Save Videos
            self.recordVideos(gesture_name)

            # Ask for Another Recognition Phase
            if not input('\nStart Another Gesture Acquisition ? [Y/N]: ') in ['Y', 'y']:

                # Ask for Training Phase
                if input('\nStart Model Training ? [Y/N]: ') in ['Y', 'y']:
                    self.training_phase = True

                # STOP Recording Phase
                self.recording_phase = False
                break


############################################################
#                 Training Phase Functions                 #
############################################################


    def createModel(self, input_shape, output_shape, optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy']):

        # Create Model as a Sequential NN
        model = Sequential()

        # Add LSTM (Long Short-Term Memory) Layers
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape))
        model.add(Dropout(0.5))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(Dropout(0.5))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dropout(0.5))

        # Add Dense (FullyConnected) Layers
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(output_shape, activation='softmax'))

        # Add Compile Parameters
        model.compile(optimizer, loss, metrics)

        return model

    def modelCallbacks(self):

        log_dir = os.path.join('Logs')
        tb_callback = TensorBoard(log_dir=log_dir)

        early_stopping = EarlyStopping(monitor='loss', patience=30)

        return [tb_callback, early_stopping]

    def processGestures(self, gestures):

        # Transform to Numpy Array
        gestures = np.array(gestures)
        print("gesto:", gestures)

        # Map Gesture Label
        label_map = {label: num for num, label in enumerate(gestures)}

        # Create Sequence and Label Vectors
        sequences, labels = [], []

        # Loop Over Gestures
        for gesture in gestures:

            # Loop Over Video Sequence
            for sequence in range(self.no_sequences):

                frame = []

                # Loop Over Frames
                for frame_num in range(self.sequence_length):

                    # Load Frame
                    frame.append(np.load(os.path.join(f'{self.package_path}/database/3D_Gestures/{self.gesture_file}/',
                                                      gesture, str(sequence), '{}.npy'.format(frame_num))))

                # Append Gesture Sequence and Label
                sequences.append(frame)
                labels.append(label_map[gesture])

                # Info Print
                print(f'Sequences Shape: {np.array(sequences).shape}\n')
                print(f'Labels Shape: {np.array(labels).shape} \n')

        return sequences, labels

    def TrainNetwork(self):

        # Load Gesture List
        gestures = [f for f in os.listdir(f'{self.package_path}/database/3D_Gestures/{self.gesture_file}/')
                    if os.path.isdir(os.path.join(f'{self.package_path}/database/3D_Gestures/{self.gesture_file}/', f))]

        gestures = np.array(gestures)  # correzione

        # Process Gestures
        sequences, labels = self.processGestures(gestures)

        # Create Input Array
        X = np.array(sequences)
        print(f'X Shape: {X.shape}')

        # Convert Labels to Integer Categories
        Y = np_utils.to_categorical(labels).astype(int)

        #print(X.shape), print(Y.shape)
        #print(type(X)), print(type(Y))
        #print(X), print(Y)

        # Split Dataset
        X_train, X_test, Y_train, Y_test = SKLearnModelSelection.train_test_split(X, Y, test_size=0.1)
        print(f'X_Test Shape: {X_test.shape[2]} | Y_Test Shape: {Y_test.shape}')

        # print(X_train.shape)
        # print(X_test.shape)
        # print(Y_train.shape)
        # print(Y_test.shape)

        # print(30,X_test.shape[2])
        # print(gestures.shape[0])

        # Create NN Model
        model = self.createModel(input_shape=(30, X_test.shape[2]), output_shape=gestures.shape[0],
                                 optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        # Train Model
        model.fit(X_train, Y_train, epochs=100,
                  callbacks=self.modelCallbacks())

        # Save Model as 'trained_model.pkl'
        with open(f'{self.package_path}/database/3D_Gestures/{self.gesture_file}/trained_model.pkl', 'wb') as save_file:
            pickle.dump(model, save_file, protocol=pickle.HIGHEST_PROTOCOL)


############################################################
#                           Main                           #
############################################################

if __name__ == '__main__':

    # Instantiate Gesture Recognition Training Class
    GRT = GestureRecognitionTraining3D()

    print("Record Phase = ", GRT.recording_phase)
    print("\nTraining Phase = ", GRT.training_phase)

    # Recording Phase
    if GRT.recording_phase == True and GRT.training_phase == False:

        print("\nSTART RECORDING PHASE\n")

        # Add new databse gesture
        GRT.Record()

        print("\nEND RECORDING PHASE\n")

        print("Record Phase = ", GRT.recording_phase)
        print("\nTraining Phase = ", GRT.training_phase)

    if GRT.recording_phase == False and GRT.training_phase == True and not rospy.is_shutdown():

        print("\nSTART TRAINING PHASE\n")

        # Train Network
        GRT.TrainNetwork()

        print("\n END RECORDING PHASE\n")

        # Terminate Training Phase
        GRT.training_phase = False

        print("Record Phase = ", GRT.recording_phase)
        print("\nTraining Phase = ", GRT.training_phase, "\n")

    # Shutdown
    else:
        rospy.shutdown()
