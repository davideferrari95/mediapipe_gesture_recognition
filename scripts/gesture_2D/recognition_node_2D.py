#!/usr/bin/env python3

import rospy, rospkg
import pandas as pd 
import pickle, warnings

# Ignore Pickle Warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Import Mediapipe Messages
from mediapipe_gesture_recognition.msg import Pose, Face, Hand

class GestureRecognition2D:

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

        # Read Gesture Recognition Precision Probability Parameter
        self.recognition_precision_probability = rospy.get_param('recognition_precision_probability', 0.8)

        # Get Package Path
        package_path = rospkg.RosPack().get_path('mediapipe_gesture_recognition')

        # Choose Gesture File
        gesture_file = ''
        if self.enable_right_hand: gesture_file += 'Right'
        if self.enable_left_hand:  gesture_file += 'Left'
        if self.enable_pose:       gesture_file += 'Pose'
        if self.enable_face:       gesture_file += 'Face'

        # Load the Trained Model for the Detected Landmarks
        with open(f'{package_path}/database/2D_Gestures/{gesture_file}/trained_model.pkl', 'rb') as f:
            self.model = pickle.load(f) 

    # Callback Functions
    def RightHandCallback(self, data): self.right_new_msg = data
    def LeftHandCallback(self, data):  self.left_new_msg  = data
    def PoseCallback(self, data):      self.pose_new_msg  = data
    def FaceCallback(self, data):      self.face_new_msg  = data

    # Gesture Recognition Function
    def Recognition(self):

        # Create a List with the Detected Landmarks Coordinates
        Landmarks = []

        # Loop Until Landmarks Found
        while Landmarks == []:

            # Exit if ROS Shutdown
            if rospy.is_shutdown(): exit()

            # Check [Right Hand, Left Hand, Pose, Face] Landmarks
            Landmarks = self.process_landmarks(self.enable_right_hand, 'right_new_msg', Landmarks)
            Landmarks = self.process_landmarks(self.enable_left_hand,  'left_new_msg',  Landmarks)
            Landmarks = self.process_landmarks(self.enable_pose, 'pose_new_msg', Landmarks)
            Landmarks = self.process_landmarks(self.enable_face, 'face_new_msg', Landmarks)

        # Prediction with the Trained Model
        X = pd.DataFrame([Landmarks])
        pose_recognition_name = self.model.predict(X)[0]
        pose_recognition_prob = self.model.predict_proba(X)[0]

        # Print the Recognized Gesture 
        Prob = max(pose_recognition_prob)
        if (Prob > self.recognition_precision_probability): print(pose_recognition_name)

    # Process Landmark Messages Function
    def process_landmarks(self, enable, message_name, Landmarks):

        # Check Landmarks Existence
        if (enable == True and hasattr(self, message_name)):

            # Get Message Variable Name
            message = getattr(self, message_name)

            # Process All Landmarks
            for i in range (len(message.keypoints)):

                # Extend Landmark Vector -> Saving New Keypoints
                Landmarks.extend([message.keypoints[i].x, message.keypoints[i].y, 
                                  message.keypoints[i].z, message.keypoints[i].v])

            # Delete Class Attribute -> Delete Message
            delattr(self, message_name)

        return Landmarks

if __name__ == '__main__':

    # Instantiate Gesture Recognition Class
    GR = GestureRecognition2D()

    while not rospy.is_shutdown():

        # Main Recognition Function
        GR.Recognition()

        # Sleep Rate Time
        GR.rate.sleep()
