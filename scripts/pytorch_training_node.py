#!/usr/bin/env python3

import os, time
import rospy, rospkg
import numpy as np


# Pytorch Neural Network
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import Dataset, DataLoader

# Useful Function
from keras.utils import np_utils
from sklearn import model_selection as SKLearnModelSelection

# Import Mediapipe Messages
from mediapipe_gesture_recognition.msg import Pose, Face, Hand

# Import Utilities
from scripts.utils.utils import countdown

class GestureRecognitionTraining3D:
    
    ''' 3D Gesture Recognition Training Class '''

    def __init__(self):
        
        # ROS Initialization
        rospy.init_node('mediapipe_gesture_recognition_training_node', anonymous=True) 
        self.rate = rospy.Rate(30)
        
        # Mediapipe Subscribers
        rospy.Subscriber('/mediapipe_gesture_recognition/right_hand', Hand, self.RightHandCallback)
        rospy.Subscriber('/mediapipe_gesture_recognition/left_hand',  Hand, self.LeftHandCallback)
        rospy.Subscriber('/mediapipe_gesture_recognition/pose',       Pose, self.PoseCallback)
        rospy.Subscriber('/mediapipe_gesture_recognition/face',       Face, self.FaceCallback)

        # Gesture Subscriber
        rospy.Subscriber('/mediapipe_gesture_recognition/gesture',    str,  self.GestureCallback)

        # Read Mediapipe Modules Parameters
        self.enable_right_hand = rospy.get_param('enable_right_hand', False)
        self.enable_left_hand  = rospy.get_param('enable_left_hand',  False)
        self.enable_pose = rospy.get_param('enable_pose', False)
        self.enable_face = rospy.get_param('enable_face', False)
        
        # Select Gesture File
        self.gesture_file = ''
        if self.enable_right_hand: self.gesture_file += 'Right'
        if self.enable_left_hand:  self.gesture_file += 'Left'
        if self.enable_pose:       self.gesture_file += 'Pose'
        if self.enable_face:       self.gesture_file += 'Face'
        
        # Read Training Parameters
        self.recording_phase = rospy.get_param('recording', True)
        self.training_phase  = rospy.get_param('training', False)

        # Get Package Path
        self.package_path = rospkg.RosPack().get_path('mediapipe_gesture_recognition')
        
        # Number of videos worth of data
        self.no_sequences = 10

        # Number of frames for each videos (30 frames/s) * length of the video in seconds
        self.sequence_length = 30*1

        #Split Dataset
        self.train_percent = 0.6
        self.val_percent = 0.20
        self.test_percent = 0.20

    # Callback Functions
    def RightHandCallback(self, data): self.right_new_msg: Hand() = data
    def LeftHandCallback(self, data):  self.left_new_msg:  Hand() = data
    def PoseCallback(self, data):      self.pose_new_msg:  Pose() = data
    def FaceCallback(self, data):      self.face_new_msg:  Face() = data
    def GestureCallback(self, data):   self.gesture:       str()  = data     #labels for traininf itself 

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
        pose    = np.array([[res.x, res.y, res.z, res.v]        for res in pose_msg.keypoints]).flatten()  if self.pose_new_msg  else np.zeros(33*4)
        #face    = np.array([[res.x, res.y, res.z, res.v]        for res in face_msg.keypoints]).flatten()  if self.face_new_msg  else np.zeros(468*3)
        left_h  = np.array([[res.x, res.y, res.z, res.v]        for res in left_msg.keypoints]).flatten()  if self.left_new_msg  else np.zeros(21*4)
        right_h = np.array([[res.x, res.y, res.z, res.v]        for res in right_msg.keypoints]).flatten() if self.right_new_msg else np.zeros(21*4)
        
        # TODO: Delete new_messages
        
        print(pose.shape)
        #print(face.shape)
        print(left_h.shape, right_h.shape)

        # Concatenate Data
        # pose, face, left_h, right_h
        return np.concatenate([right_h, left_h, pose]) #pose, face, left_h, right_h

    # Create Folders for Each Gesture
    def createFolders(self, gesture):
        
        ''' 
        Create a Folder for Each Gesture \n
        This Folder Contains a Folder for Each Recorded Video
        '''
        
        for sequence in range(self.no_sequences):
            try: os.makedirs(os.path.join(f'{self.package_path}/database/3D_Gestures/{self.gesture_file}/', gesture, str(sequence)))
            except: pass

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
            #print('Collection Starting in 2 Seconds')
            #countdown(2)
            print('Starting Collection')
            start = rospy.get_time() 
            
            # Loop through Video Length (Sequence Length)
            for frame_num in range(self.sequence_length):
                
                # Flatten and Concatenate Keypoints (Insert your gesture source here ) (here the missing sources self.pose_new_msg, self.face_new_msg, self.right_new_msg)
                keypoints = self.flattenKeypoints(self.left_new_msg, self.right_new_msg, self.pose_new_msg)
                
                #Export Keypoints Values in the Correct Folder
                npy_path = os.path.join(f'{self.package_path}/database/3D_Gestures/{self.gesture_file}/', gesture, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)
                
                # Sleep for 30 FPS
                # TODO: Adjust To have 30 frames/s
                time.sleep(1/30)
            
            print (f'Length of Acquisition : {rospy.get_time() - start}')
            print('End Collection')
            #time.sleep(1)
        
        print('All Videos Saved')

    def Record(self):
        
        ''' Recording Phase '''
        
        while not rospy.is_shutdown():

            # Gesture Labeling
            #gesture_name = input('\nInsert Gesture Name: ')

            self.createFolders(self.gesture)

            # Create Folders for Each Video
            #self.createFolders(gesture_name)
            
            # Start Counter
            print('\nAcquisition Starts in:')

            countdown(5)
            
            # Record and Save Videos
            #self.recordVideos(gesture_name)

            self.recordVideos(self.gesture)
                    
            # Ask for Another Recognition Phase
            if not input('\nStart Another Gesture Acquisition ? [Y/N]: ') in ['Y','y']:
                
                # Ask for Training Phase
                if input('\nStart Model Training ? [Y/N]: ') in ['Y','y']: self.training_phase = True
                
                # STOP Recording Phase
                self.recording_phase = False
                break
    
    def processGestures(self, gestures):
        
        # Transform to Numpy Array
        gestures = np.array(gestures)
        print("gesto:", gestures)
        
        # Map Gesture Label
        label_map = {label:num for num, label in enumerate(gestures)}
        
        # Create Sequence and Label Vectors
        sequences, labels = [], []

        # Loop Over Gestures
        for gesture in gestures:
            
            # Loop Over Video Sequence
            for sequence in range(int(self.no_sequences)):

                frame = []

                # Loop Over Frames
                for frame_num in range(int(self.sequence_length)):
                    
                    # Load Frame
                    frame.append(np.load(os.path.join(f'{self.package_path}/database/3D_Gestures/{self.gesture_file}/',
                                          gesture, str(sequence), '{}.npy'.format(frame_num))))
                
                # Append Gesture Sequence and Label
                sequences.append(frame)
                labels.append(label_map[gesture])

                # Info Print
                print(f'Sequences Shape: {np.array(sequences).shape}')
                print(f'Labels Shape: {np.array(labels).shape}')
        
        return sequences, labels

    def trainPhase(self):
        
        # Load Gesture List
        gestures = [f for f in os.listdir(f'{self.package_path}/database/3D_Gestures/{self.gesture_file}/') \
                if os.path.isdir(os.path.join(f'{self.package_path}/database/3D_Gestures/{self.gesture_file}/', f))]
        
        
        gestures = np.array(gestures)   #correzione 

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

        #Y = torch.nn.functional.one_hot(labels).astype(int)

        
        #Split Dataset 
        self.X_train, self.X_rim, self.Y_train, self.Y_rim = SKLearnModelSelection.train_test_split(X, Y, test_size=0.2)
        self.X_val, self.X_test, self.Y_val, self.Y_test = SKLearnModelSelection.train_test_split(self.X_rim,self.Y_rim, test_size=0.5)

        #print(self.X_train.shape)
        #print(self.X_test.shape)
        #print(self.X_val.shape)
        #print(self.Y_train.shape)
        #print(self.Y_test.shape)
        #print(self.Y_val.shape)
        #print(30,self.X_test.shape[2])
        #print(gestures.shape[0])

        self.X_train = torch.from_numpy(self.X_train).float()
        self.X_test = torch.from_numpy(self.X_test).float()
        self.X_val = torch.from_numpy(self.X_val).float()
        self.Y_train = torch.from_numpy(self.Y_train).float()
        self.Y_test = torch.from_numpy(self.Y_test).float()
        self.Y_val = torch.from_numpy(self.Y_val).float()

        
        model = NeuralNetwork((30,self.X_test.shape[2]), gestures.shape[0], self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test)
        trainer = Trainer(
                   auto_lr_find=True, 
                   max_epochs = 25, 
                   fast_dev_run = False, 
                   log_every_n_steps=1, 
                   callbacks=[EarlyStopping(monitor="val_loss", stopping_threshold = 0.5)]
                                 )    
        trainer.fit(model)
        
        with open(f'{self.package_path}/database/3D_Gestures/{self.gesture_file}/trained_model.pth', 'wb') as FILE:
            torch.save(model, FILE)

        print('Model Saved Correctly')
       

class CustomDataset(Dataset):
    def __init__(self, X, y, transform = None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

############################################################
#             Neural Newtork by Pytorch lightining         #
############################################################

class NeuralNetwork(pl.LightningModule):
    def __init__(self, input_shape,output_shape, x_train, y_train, x_val, y_val, x_test, y_test,transform = None):
        super(NeuralNetwork, self).__init__()

        #Make the six layers
        self.lstm1 = nn.LSTM(input_size=input_shape[-1], hidden_size=64, num_layers=1, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True, bidirectional=False)
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True, bidirectional=False)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_shape)

        #Get the training parameters
        self.transform = transform
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

       

    def forward(self, x):

        x, _ = self.lstm1(x)
        x = F.relu(x)
        #x = F.dropout(x, p=0.5)
        
        x, _ = self.lstm2(x)
        x = F.relu(x)
        #x = F.dropout(x, p=0.5)
        
        x, _ = self.lstm3(x)
        x = F.relu(x)
        #x = F.dropout(x, p=0.5)
        
        x = x[:, -1, :] # Select only the last output from the LSTM
        
        x = self.fc1(x)
        x = F.relu(x)
       # x = F.dropout(x, p=0.5)
        
        x = self.fc2(x)
        x = F.relu(x)
        #x = F.dropout(x, p=0.5)
        
        x = self.fc3(x)
        return x
    

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr = 0.4)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        self.log("train_loss", loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        self.log("val_loss", loss)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        self.log('Test loss', loss)
        return {'test_loss': loss}
    
    def train_dataloader(self):

        trainData = CustomDataset(self.x_train, self.y_train)
        print("\nTraind Data size: ", len(trainData))
        return DataLoader(trainData, batch_size=64, num_workers = 12, shuffle=True)

    def val_dataloader(self):

        valData = CustomDataset(self.x_val, self.y_val)
        print("\nVal Data size: ", len(valData))
        return DataLoader(valData, batch_size=64, num_workers = 12, shuffle=False)

    def test_dataloader(self):

        testData = CustomDataset(self.x_test, self.y_test)
        print("\nTest Data size: ", len(testData))
        return DataLoader(testData, batch_size=64, num_workers = 12, shuffle=False)
        
############################################################
#                           Main                           #
############################################################

if __name__ == '__main__':
    
    # Instantiate Gesture Recognition Training Class
    GRT = GestureRecognitionTraining3D()

    print("Record Phase = ",GRT.recording_phase)
    print("\nTraining Phase = ", GRT.training_phase)

    # Recording Phase
    if  GRT.recording_phase == True and GRT.training_phase == False: 
        
        print("\nSTART RECORDING PHASE\n")

        #Add new databse gesture 
        GRT.Record()    

        print("\nEND RECORDING PHASE\n")  

        print("Record Phase = ",GRT.recording_phase)
        print("\nTraining Phase = ", GRT.training_phase)
    

    
    if GRT.recording_phase == False and GRT.training_phase == True and not rospy.is_shutdown(): 
        
        print("\nSTART TRAINING PHASE\n")

        # Train Network
        GRT.trainPhase()

        print("\n END RECORDING PHASE\n")

        # Terminate Training Phase
        GRT.training_phase = False

        print("Record Phase = ",GRT.recording_phase)
        print("\nTraining Phase = ", GRT.training_phase, "\n")
    