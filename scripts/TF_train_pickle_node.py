#!/usr/bin/env python3


#Base libraries
import os, time, csv, rospy, rospkg, numpy as np, pickle

#Tensorflow libraries 
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.utils import np_utils

# Useful Function libreries
from keras.utils import np_utils
from sklearn import model_selection as SKLearnModelSelection



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
        self.database_path = os.path.join(f'{self.package_path}/database/3D_Gestures/{self.gesture_file}/Padded_file')
        
        # Split Dataset
        self.train_percent = 0.9
        self.test_percent = 0.1
      
        


############################################################
#                 Train Phase Functions                #
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
        X_train, X_test, Y_train, Y_test = SKLearnModelSelection.train_test_split(X, Y, test_size=self.test_percent, random_state=42)
        print(f'X_Test Shape: {X_test.shape[2]} | Y_Test Shape: {Y_test.shape}')

        # Create NN Model
        model = self.createModel(input_shape=(85, X_test.shape[2]), output_shape=gestures.shape[0],
                                 optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        # Train Model
        model.fit(X_train, Y_train, epochs=250,callbacks=self.modelCallbacks())

        # Save Model as 'trained_model.pkl'
        with open(f'{self.model_path}/TF_trained_model.pkl', 'wb') as save_file:
            pickle.dump(model, save_file, protocol=pickle.HIGHEST_PROTOCOL)


        print('Model TF Saved Correctly')
############################################################
#             Neural Newtork by Pytorch lightining         #
############################################################

    def createModel(self, input_shape, output_shape, optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy']):

        # Create Model as a Sequential NN
        model = Sequential()

        # Add LSTM (Long Short-Term Memory) Layers
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(Dropout(0.5))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(Dropout(0.5))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dropout(0.5))

        # Add Dense (FullyConnected) Layers
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation="relu"))
        # model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(output_shape, activation='softmax'))

        # Add Compile Parameters
        model.compile(optimizer, loss, metrics)

        return model

    def modelCallbacks(self):

        log_dir = os.path.join('Logs')
        tb_callback = TensorBoard(log_dir=log_dir)

        early_stopping = EarlyStopping(monitor='loss', patience=30)

        return [tb_callback, early_stopping]
        
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

     
    