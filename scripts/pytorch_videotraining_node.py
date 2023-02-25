#!/usr/bin/env python3

import os
import time
import csv
import rospy
import rospkg
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

from mediapipe_gesture_recognition.srv import VideoSequence, VideoSequenceResponse

# Import Utilities
from moviepy.video.io.VideoFileClip import VideoFileClip


class GestureRecognitionTraining3D:

    ''' 3D Gesture Recognition Training Class '''

    def __init__(self):

        #Set which configuration we want to train (you must configurate these by yourself)
        self.enable_right_hand = True
        self.enable_left_hand = True
        self.enable_pose = True
        self.enable_face = False

        # Select Gesture File
        self.gesture_file = ''
        if self.enable_right_hand: self.gesture_file += 'Right'
        if self.enable_left_hand:  self.gesture_file += 'Left'
        if self.enable_pose:       self.gesture_file += 'Pose'
        if self.enable_face:       self.gesture_file += 'Face'

        # Get Package Path
        self.package_path = rospkg.RosPack().get_path('mediapipe_gesture_recognition')
        
        # Split Dataset
        self.train_percent = 0.9
        self.val_percent = 0.05
        self.test_percent = 0.05
      
        


############################################################
#                 Train Phase Functions                #
############################################################
    def processGestures(self, gestures):  #Chedi a davide se va bene 
        
        gestures = np.array(gestures)

        # Map Gesture Label
        label_map = {label:num for num, label in enumerate(gestures)}
        
        
        sequences, labels = [], []

        # Loop Over Gestures
        for gesture in gestures:
            
            # Loop Over Video Sequence
            for sequence in os.listdir(os.path.join(f'{self.package_path}/database/3D_Gestures/{self.gesture_file}/', str(gesture))):

                frame = []
                
                npy_path = os.path.join(f'{self.package_path}/database/3D_Gestures/{self.gesture_file}/', gesture, str(sequence))
                totframe = len(os.listdir(npy_path))

                

                # Loop Over Frames
                for frame_num in range(totframe-27, totframe): #Take the last 30Frames 
                    
                    # Load Frame
                    frame.append(np.load(os.path.join(npy_path, str(frame_num)+".npy")))

                #print("Gesture:", gesture, "Video:", os.path.basename(npy_path), "with frame:", np.shape(frame))
            
                sequences.append(frame)       
                # Append Gesture Sequence and Label
                labels.append(label_map[gesture])
            
                # Info Print
                #print(f'Sequences Shape: {np.array(sequences).shape}')
                #print(f'Labels Shape: {np.array(labels).shape}')
            
        return sequences, labels

    def trainPhase(self):
        
        # Load Gesture List
        gestures = [f for f in os.listdir(f'{self.package_path}/database/3D_Gestures/{self.gesture_file}/') \
                if os.path.isdir(os.path.join(f'{self.package_path}/database/3D_Gestures/{self.gesture_file}/', f))]
        
        # Convert into np array 
        gestures = np.array(gestures)   #correzione 
       
        # print("Type of gestures: ", gestures)

        #Process Gestures
        sequences, labels = self.processGestures(gestures)

        # Create Input Array
        X = np.array(sequences)
        print(f'X Shape: {X.shape}')
        
        # Convert Labels to Integer Categories
        Y = np_utils.to_categorical(labels).astype(int)
        
        # Split Dataset 
        self.X_train, self.X_rim, self.Y_train, self.Y_rim = SKLearnModelSelection.train_test_split(X, Y, test_size=0.1)
        self.X_val, self.X_test, self.Y_val, self.Y_test = SKLearnModelSelection.train_test_split(self.X_rim,self.Y_rim, test_size=0.5)

        print(self.X_train.shape)
        print(self.X_test.shape)
        print(self.X_val.shape)
        print(self.Y_train.shape)
        print(self.Y_test.shape)
        print(self.Y_val.shape)
        print(30,self.X_test.shape[2])
        print(gestures.shape[0])

        # Convert to Torch Tensors
        self.X_train = torch.from_numpy(self.X_train).float()
        self.X_test = torch.from_numpy(self.X_test).float()
        self.X_val = torch.from_numpy(self.X_val).float()
        self.Y_train = torch.from_numpy(self.Y_train).float()
        self.Y_test = torch.from_numpy(self.Y_test).float()
        self.Y_val = torch.from_numpy(self.Y_val).float()

      

        # Create Model
        model = NeuralNetwork((30,self.X_test.shape[-1]), gestures.shape[0], self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test)
        
        # Create Trainer 
        trainer = Trainer(
                   #auto_lr_find=True, 
                   max_epochs = 250, 
                   fast_dev_run = False, 
                   log_every_n_steps=1, 
                   #callbacks=[EarlyStopping(monitor="val_loss", stopping_threshold = 0.1)]
                                 )    
        
        # Train Model
        trainer.fit(model)
        
        # Save Model
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

        # Make the six layers
        self.lstm1 = nn.LSTM(input_size=input_shape[-1], hidden_size=64, num_layers=1, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True, bidirectional=False)
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True, bidirectional=False)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_shape)

        # Get the training parameters
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
        # x = F.dropout(x, p=0.5)
        
        x, _ = self.lstm2(x)
        x = F.relu(x)
        # x = F.dropout(x, p=0.5)
        
        x, _ = self.lstm3(x)
        x = F.relu(x)
        # x = F.dropout(x, p=0.5)
        
        x = x[:, -1, :] # Select only the last output from the LSTM
        
        x = self.fc1(x)
        x = F.relu(x)
       # x = F.dropout(x, p=0.5)
        
        x = self.fc2(x)
        x = F.relu(x)
        # x = F.dropout(x, p=0.5)
        
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
    

    #Instantiate Gesture Recognition Training Class
    GRT = GestureRecognitionTraining3D()
    

    print("\nSTART TRAINING PHASE\n")  

    # Train Network
    GRT.trainPhase()  

    print("\n END RECORDING PHASE\n")  

     
    