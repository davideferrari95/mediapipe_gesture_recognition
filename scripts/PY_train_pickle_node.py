#!/usr/bin/env python3

import os, pickle
# import time
# import csv
# import rospy
import rospkg
import numpy as np
import time


# Pytorch Neural Network
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy
from torchmetrics.functional import accuracy
import torchmetrics

# import torch.nn.utils.rnn as rnn_utils


# Useful Function
from keras.utils import np_utils
from sklearn import model_selection as SKLearnModelSelection


# tensorboard --logdir lightning_logs/

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

        # Get Model Path
        self.model_path = os.path.join(self.package_path, 'model')

        # Database bath
        self.database_path = os.path.join(f'{self.package_path}/database/3D_Gestures/{self.gesture_file}/Padded_file/')
        
        # Split Dataset
        self.train_percent = 0.9
        self.val_percent = 0.05
        self.test_percent = 0.05
        
      
############################################################
#                 Train Phase Functions                    #
############################################################

    def processGestures(self, gestures): 

        gestures = np.array(gestures)

        # Map Gesture Label
        label_map = {label.split(".")[0]:num for num, label in enumerate(gestures)}
        
        video_sequence, labels = [], []

        # Loop Over Gestures
        for gesture in sorted(gestures):

            load_file_path = os.path.join(self.database_path, f'{gesture}')

            # Load File
            with open(load_file_path, 'rb') as f:

                #Get gesture sequence from pkl file 
                sequence = pickle.load(f)

                #Loop over the sequence
                for array in sequence:

                    # Get Label
                    for i in range (np.array(array, dtype= object).shape[0]):

                        #Get the gesture name deleting the extenction name
                        gesture_name = os.path.splitext(gesture)[0]

                        labels.append(label_map[gesture_name])

                    print(f'Upgrade Label Shape: {np.array(labels).shape}')
                    
                    #Concatenate the padded sequence with the previous one
                    if np.ndim(np.array(video_sequence)) == 1:

                        video_sequence = array
         
                    else:
                        video_sequence = np.concatenate((video_sequence, array), axis = 0)
                    
                print("I'm processing |", gesture, "And the array now is", video_sequence.shape, "at time:", time.time())

        # Info Print
        print(f'Sequences Shape: {np.array(video_sequence).shape}')
        print(f'Labels Shape: {np.array(labels).shape}')

        return video_sequence, labels

    def trainPhase(self):
        
        # Load Gesture List
        gestures = [f for f in os.listdir(self.database_path)]
        
        #Process Gestures
        sequences, labels = self.processGestures(gestures)

        # Convert to Numpy Array
        gestures = np.array(gestures)   

        # Create Input Array
        X = np.array(sequences)

        print(f'X Shape: {X.shape}')
        
        # Convert Labels to Integer Categories
        Y = np_utils.to_categorical(labels).astype(int)
        
        # Split Dataset 
        self.X_train, self.X_rim, self.Y_train, self.Y_rim = SKLearnModelSelection.train_test_split(X, Y, test_size=0.1)
        self.X_val, self.X_test, self.Y_val, self.Y_test = SKLearnModelSelection.train_test_split(self.X_rim,self.Y_rim, test_size=0.5)
        
        # Print Dataset split Shape
        print("\nX training: ", self.X_train.shape)
        print("Y training: ", self.Y_train.shape)
        print("X validation: ", self.X_val.shape)
        print("Y validation: ", self.Y_val.shape)
        print("X test: ", self.X_test.shape)
        print("Y test: ", self.Y_test.shape)
        
        #Print Neural Network Input and Output Shape
        print("\nNN input: ", 85, self.X_test.shape[2])
        print("NN output: ", gestures.shape[0], "\n")

        # Convert to Torch Tensors
        self.X_train = torch.from_numpy(self.X_train).float()
        self.X_test = torch.from_numpy(self.X_test).float()
        self.X_val = torch.from_numpy(self.X_val).float()
        self.Y_train = torch.from_numpy(self.Y_train).float()
        self.Y_test = torch.from_numpy(self.Y_test).float()
        self.Y_val = torch.from_numpy(self.Y_val).float()

      
        # Create Model
        model = NeuralNetwork((85, self.X_test.shape[2]), gestures.shape[0], self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        
        # Create Trainer 
        trainer = Trainer(
                   #accelerator= "gpu",
                   auto_lr_find=True, 
                   max_epochs = 2000, 
                   fast_dev_run = False, 
                   log_every_n_steps=1, 
                   callbacks=[EarlyStopping(monitor="train_loss", patience = 150, mode = "min", min_delta = 0.01 )]
                                 )    
        
        # Train Model
        trainer.fit(model)
        
        # Save Model
        with open(f'{self.model_path}/py_trained_model.pth', 'wb') as FILE:
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

    def __init__(self, input_shape, output_shape, x_train, y_train, x_val, y_val, x_test, y_test,transform = None):
        super(NeuralNetwork, self).__init__()

        self.lstm1 = nn.LSTM(input_size= input_shape[-1], hidden_size=64, num_layers=1, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True, bidirectional=False)
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True, bidirectional=False)

        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, output_shape)

        # Get the training parameters
        self.transform = transform
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

        # self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes = output_shape)
        
    def forward(self, x):

        x, _ = self.lstm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)
        
        x, _ = self.lstm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)
        
        x, _ = self.lstm3(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)

        x = x[:, -1, :] # Select only the last output from the LSTM
        
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)
        
        x = self.fc2(x)

        return x
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 0.0005) #Poi vedi Adam, e vedi il lr che non è vero che si trova da solo 
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        self.log("train_loss", loss)
        
        # acc = self.accuracy(y_pred, y)
        # self.log("train_acc", acc, prog_bar= True, on_step=True, on_epoch=False)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        self.log("val_loss", loss)
        
        # acc = self.accuracy(y_pred, y)
        # self.log("val_acc", acc, prog_bar= True, on_step=False, on_epoch=True)

        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        self.log('Test loss', loss)

        # acc = self.accuracy(y_pred, y)
        # self.log("test_acc", acc, prog_bar= True, on_step=False, on_epoch=True)

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

    print("\n END TRAINING PHASE\n") 