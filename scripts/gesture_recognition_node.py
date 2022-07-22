#!/usr/bin/env python3

import rospy, rospkg
import pandas as pd 
import pickle, warnings

# Ignore Pickle Warnings
warnings.filterwarnings("ignore", category=UserWarning)

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
        if self.enable_right_hand == True: gesture_file += "Right"
        if self.enable_left_hand  == True: gesture_file += "Left"
        if self.enable_pose       == True: gesture_file += "Pose"
        if self.enable_face       == True: gesture_file += "Face"
        
        # Load the Trained Model for the Detected Landmarks
        with open(f'{package_path}/database/Gestures/{gesture_file}/trained_model.pkl', 'rb') as f:
            self.model = pickle.load(f) 

    # Callback Functions
    def RightHandCallback(self, data): self.right_new_msg = data
    def LeftHandCallback(self, data):  self.left_new_msg  = data
    def PoseCallback(self, data):      self.pose_new_msg  = data
    def FaceCallback(self, data):      self.face_new_msg  = data

    def Recognition(self):
        
        # Create a List with the Detected Landmarks Coordinates
        Landmarks = []
        
        # Loop Until Landmarks Found
        while Landmarks == [] and not rospy.is_shutdown():
            
            # Check Right Hand Landmarks
            if (self.enable_right_hand == True and hasattr(self, 'right_new_msg')):
                
                # Process All Landmarks
                for i in range (len(self.right_new_msg.keypoints)):
                    
                    Landmarks.extend([self.right_new_msg.keypoints[i].x, self.right_new_msg.keypoints[i].y, 
                                      self.right_new_msg.keypoints[i].z, self.right_new_msg.keypoints[i].v])

            # Check Left Hand Landmarks
            if (self.enable_left_hand == True and hasattr(self, 'left_new_msg')):
                
                # Process All Landmarks
                for i in range (len(self.left_new_msg.keypoints)):
                    
                    Landmarks.extend([self.left_new_msg.keypoints[i].x, self.left_new_msg.keypoints[i].y, 
                                      self.left_new_msg.keypoints[i].z, self.left_new_msg.keypoints[i].v])

            # Check Pose Landmarks
            if (self.enable_pose == True and hasattr(self, 'pose_new_msg')):
                
                # Process All Landmarks
                for i in range (len(self.pose_new_msg.keypoints)):
                
                    Landmarks.extend([self.pose_new_msg.keypoints[i].x, self.pose_new_msg.keypoints[i].y, 
                                      self.pose_new_msg.keypoints[i].z, self.pose_new_msg.keypoints[i].v])

            # Check Face Landmarks
            if (self.enable_face == True and hasattr(self, 'face_new_msg')):
                
                # Process All Landmarks
                for i in range (len(self.face_new_msg.keypoints)):

                    Landmarks.extend([self.face_new_msg.keypoints[i].x, self.face_new_msg.keypoints[i].y, 
                                      self.face_new_msg.keypoints[i].z, self.face_new_msg.keypoints[i].v])
        
        # Prediction with the Trained Model
        X = pd.DataFrame([Landmarks])
        pose_recognition_name = self.model.predict(X)[0]
        pose_recognition_prob = self.model.predict_proba(X)[0]

        # Print the Recognised Gesture 
        Prob = max(pose_recognition_prob)
        if (Prob > self.recognition_precision_probability): print(pose_recognition_name)


############################################################
#                           Main                           #
############################################################

if __name__ == "__main__":
    
    # Instantiate Gesture Recognition Class
    Recognition = GestureRecognition2D()
    
    while not rospy.is_shutdown():
        
        # Main Recognition Function
        Recognition.Recognition()
        
        # Sleep Rate Time
        Recognition.rate.sleep()
