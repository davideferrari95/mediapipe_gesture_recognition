#!/usr/bin/env python3

import rospy, rospkg
import pandas as pd

# Import Pickle for Saving
import pickle

# Import Sklearn Functions
from sklearn.model_selection import train_test_split 
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Import Files Check Functions
from os import listdir
from os.path import isfile, join

# Import Mediapipe Messages
from mediapipe_gesture_recognition.msg import Pose, Face, Hand

# Import Utilities
from scripts.utils.utils import countdown

class GestureRecognitionTraining2D:

    ''' 2D Gesture Recognition Training Class '''

    def __init__(self):

        # ROS Initialization
        rospy.init_node('mediapipe_gesture_recognition_training_node', anonymous=True) 
        self.rate = rospy.Rate(30)

        # Mediapipe Subscribers
        rospy.Subscriber('/mediapipe_gesture_recognition/right_hand', Hand, self.RightHandCallback)
        rospy.Subscriber('/mediapipe_gesture_recognition/left_hand',  Hand, self.LeftHandCallback)
        rospy.Subscriber('/mediapipe_gesture_recognition/pose',       Pose, self.PoseCallback)
        rospy.Subscriber('/mediapipe_gesture_recognition/face',       Face, self.FaceCallback)

        # Read Mediapipe Modules Parameters
        self.enable_right_hand = rospy.get_param('enable_right_hand', False)
        self.enable_left_hand  = rospy.get_param('enable_left_hand',  False)
        self.enable_pose = rospy.get_param('enable_pose', False)
        self.enable_face = rospy.get_param('enable_face', False)
        
        # Read Training Parameters
        self.recording_phase = rospy.get_param('recording', True)
        self.training_phase  = rospy.get_param('training', False)

        # Get Package Path
        self.package_path = rospkg.RosPack().get_path('mediapipe_gesture_recognition')

    # Callback Functions
    def RightHandCallback(self, data): self.right_new_msg: Hand() = data
    def LeftHandCallback(self, data):  self.left_new_msg:  Hand() = data
    def PoseCallback(self, data):      self.pose_new_msg:  Pose() = data
    def FaceCallback(self, data):      self.face_new_msg:  Face() = data

############################################################
#                Recording Phase Functions                 #
############################################################

    # Message To Dictionary Conversion
    def msg2dict(self, dict, right_hand, left_hand, pose, face):

        # Append Keypoint Messages in a DataFrame
        def createDataFrame(msg):  return pd.DataFrame([[k.keypoint_number, k.keypoint_name, k.x, k.y, k.z, k.v] for k in msg.keypoints], 
                                                         columns=['Keypoint Number', 'Keypoint Name', 'x', 'y', 'z', 'visibility'])

        # Append New Data to Dict
        dict[f'Data {len(dict) + 1}'] = {'Right Hand' : createDataFrame(right_hand), 
                                         'Left Hand'  : createDataFrame(left_hand), 
                                         'Pose'       : createDataFrame(pose), 
                                         'Face'       : createDataFrame(face)
        }

        return dict

    # Save Recorded Gesture
    def saveGesture(self, data_dictionary, name):

        # Select Gesture File
        self.gesture_file = ''
        if self.enable_right_hand: self.gesture_file += 'Right'
        if self.enable_left_hand:  self.gesture_file += 'Left'
        if self.enable_pose:       self.gesture_file += 'Pose'
        if self.enable_face:       self.gesture_file += 'Face'

        # Save Dict Object with Pickle
        with open(f'{self.package_path}/database/2D_Gestures/{self.gesture_file}/{name}.pkl', 'wb') as save_file:
            pickle.dump(data_dictionary, save_file, protocol = pickle.HIGHEST_PROTOCOL)
            # print(data_dictionary)

        print('Gesture Saved')

    def debugPrint(self, data_dictionary):

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

    # Record Function
    def Record(self):

        # DataFrames Dictionary Initialization
        data = {} 

        # Gesture Labeling
        gesture_name = input('\nInsert Gesture Name: ')

        # Counter
        countdown(5)

        # Starting Time
        start = rospy.Time.now()
        ACQUISITION_TIME = 10

        # Recognize Gesture for "ACQUISITION_TIME" Seconds
        while(not rospy.is_shutdown() and (rospy.Time.now() - start).to_sec() < ACQUISITION_TIME):

            # Add Incoming Messages to Dataframe
            data = self.msg2dict(data, self.right_new_msg, self.left_new_msg, self.pose_new_msg, self.face_new_msg)

            # Sleep for the Remaining Cycle Time (30 FPS)
            self.rate.sleep()

        print('End of The Acquisition, Saving...')

        # Save Gesture Recorded Data
        self.saveGesture(data, gesture_name)

        if not input('\nStart Another Gesture Acquisition ? [Y/N]: ') in ['Y','y']:

            # Ask for Training Phase
            if input('\nStart Model Training ? [Y/N]: ') in ['Y','y']: self.training_phase = True

            # Stop Recording Phase
            self.recording_phase = False

############################################################
#                  Training Phase Functions                #
############################################################

    def loadGesture(self, name):

        try:

            # Read Dict Object with Pickle
            with open(f'{self.package_path}/database/2D_Gestures/{self.gesture_file}/{name}', 'rb') as input_file:
                reading = pickle.load(input_file)
                print(f'Gesture "{name}" Loaded')
                return reading

        except:

            # Loading Error
            print(f'ERROR: Failed to Load Gesture "{name}"')
            return False

    def adaptDictionary(self, dictionary, name_position):

        # 'Right Hand', 'Left Hand', 'Pose', 'Face'
        list_landmarks = []

        # List with the Related dfs (df_right_hand, df_left_hand, df_pose, df_face) 
        list_df=[]

        # Append Only Enabled Landmarks
        if self.enable_right_hand: list_landmarks.append('Right Hand')
        if self.enable_left_hand:  list_landmarks.append('Left Hand')
        if self.enable_pose:       list_landmarks.append('Pose')
        if self.enable_face:       list_landmarks.append('Face')

        for part in list_landmarks:

            print (part)

            # Initialize DataFrame
            df = pd.DataFrame()

            # Process The First Data
            for i in range (len(dictionary[1][part])):
                df = pd.concat([df, dictionary[1][part].loc[i]], axis = 0)

            # Update Dictionary
            df = df.T

            # Add All the Other Rows
            for key in dictionary:

                # Initialize Temporary DataFrame
                temporary_df = pd.DataFrame()

                # Process All The Rows
                for i in range (len(dictionary[key][part])):
                    temporary_df = pd.concat([temporary_df, dictionary[key][part].loc[i]], axis = 0)

                # Update Dictionary
                temporary_df = temporary_df.T
                df = pd.concat([df, temporary_df], axis = 0)

            # Drop the Unwanted Data
            df.drop(index = 0)
            del df['Keypoint Number']
            del df['Keypoint Name']

            # Store the df in the List
            if part in ['Right Hand', 'Left Hand', 'Pose', 'Face']: list_df.append(df)

        # Concatenate the List
        concat_df = pd.concat(list_df, axis =1)

        # Create the Name List of The Gesture
        list_position = [name_position] * len((concat_df))
        list_position = pd.DataFrame (list_position)

        # Concatenate the Name of The Gesture
        return list_position.join(concat_df, how='right')

    def createTrainingDataframe(self):

        # Load the Files for the Saved Positions  
        saved_positions = [f for f in listdir(f'{self.package_path}/database/2D_Gestures/{self.gesture_file}/') 
                           if isfile(join(f'{self.package_path}/database/2D_Gestures/{self.gesture_file}/', f))]

        # Initialize DataFrame
        list_df_fragments =[]

        # Create a Dataframe with the Values of All the Saved Positions
        for i in range(len(saved_positions)):

            # Load the Position Dictionary
            position_dictionary = self.loadGesture(saved_positions[i])

            # Adapt the Dictionary
            df = self.adaptDictionary(position_dictionary, saved_positions[i])
            list_df_fragments.append(df)

        # Concatenate the Dictionary
        df = pd.concat(list_df_fragments, axis =0)

        # Create First Line of New DataFrame [Position, x1, y1, z1, v1]
        list_renamed_columns = ['Position']
        for i in range(int((len(df.columns)-1)/4)): list_renamed_columns += ['x{}'.format(i), 'y{}'.format(i), 'z{}'.format(i), 'v{}'.format(i)]

        # Add to DataFrame
        df.columns=[list_renamed_columns]
        print (df)

        # Take Only the Feature Variables [Only the Coordinates]  
        variables = df.drop(['Position'], axis = 1)

        # Get the Label to Match [Class Name]
        label = df['Position'] 

        # Split the Data adding Random Value in Training
        X_train, X_test, Y_train, Y_test = train_test_split(variables, label, test_size=0.3, random_state=1234)

        return X_train, X_test, Y_train, Y_test

    def Train(self):

        # Create the DataFrame for Training
        X_train, X_test, Y_train, Y_test = self.createTrainingDataframe()

        # Different Pipelines (4 Different Machine Learning Models)
        pipelines = {
            'lr':make_pipeline(StandardScaler(), LogisticRegression()),
            'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
            'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
            'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
        }

        # Run the Automatic Learning
        fit_models = {}
        for algo, pipeline in pipelines.items():
            model = pipeline.fit(X_train, Y_train)
            fit_models[algo] = model 

        # Evaluate and Serialize the Model
        for algo, model in fit_models.items():
            y_hat = model.predict(X_test)
            print(algo, accuracy_score(Y_test, y_hat))

        # Export the Best Model and Save It as "trained_model.pkl"
        with open(f'{self.package_path}/database/2D_Gestures/{self.gesture_file}/trained_model.pkl', 'wb') as save_file:
            pickle.dump(fit_models['rf'], save_file, protocol = pickle.HIGHEST_PROTOCOL)
            # print(data_dictionary)

        print('Model Trained Successfully')

############################################################
#                           Main                           #
############################################################

if __name__ == '__main__':

    # Instantiate Gesture Recognition Training Class
    GRT = GestureRecognitionTraining2D()

    while not rospy.is_shutdown():

        # Recording Phase
        if  (GRT.recording_phase): GRT.Record()

        # Training Phase
        if (GRT.training_phase): GRT.Train()

        # Shutdown ROS
        else: rospy.shutdown()
