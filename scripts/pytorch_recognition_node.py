#!/usr/bin/env python3

#Useful libreries
import rospy, rospkg, numpy as np, pickle, warnings, os
from pytorch_videotraining_node import NeuralNetwork, CustomDataset, GestureRecognitionTraining3D 
from scripts.utils.utils import countdown

#Pytorch libreries
import torch

# Ignore Pickle Warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Import Mediapipe Messages
from mediapipe_gesture_recognition.msg import Pose, Face, Hand


class GestureRecognition3D:

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
        self.recognition_precision_probability = rospy.get_param('recognition_precision_probability', 80)

        # Number of Consecutive Frames Needed to Make our Prediction
        self.sequence = []
 
        # Get Package Path
        package_path = rospkg.RosPack().get_path('mediapipe_gesture_recognition')
        
        # Choose Gesture File
        gesture_file = ''
        if self.enable_right_hand: gesture_file += 'Right'
        if self.enable_left_hand:  gesture_file += 'Left'
        if self.enable_pose:       gesture_file += 'Pose'
        if self.enable_face:       gesture_file += 'Face'
        
        print("We are in :", gesture_file, "Configuration")

        # Load the Trained Model for the Detected Landmarks
        with open(f'{package_path}/database/3D_Gestures/{gesture_file}/trained_model.pth', 'rb') as FILE:

            self.model = torch.load(FILE)
            self.model.eval()
            
        # Load the Names of the Saved Actions
        self.actions = np.array([f for f in os.listdir(f'{package_path}/database/3D_Gestures/{gesture_file}/') if os.path.isdir(os.path.join(f'{package_path}/database/3D_Gestures/{gesture_file}/', f))])
       
        print("We have this types of gestures: ", self.actions)


    # Callback Functions
    def RightHandCallback(self, data): self.right_new_msg = data
    def LeftHandCallback(self, data):  self.left_new_msg  = data
    def PoseCallback(self, data):      self.pose_new_msg  = data
    def FaceCallback(self, data):      self.face_new_msg  = data

     # Process Landmark Messages Function
    def process_landmarks(self, enable, message_name, Landmarks):
        
        # Check Landmarks Existence
        if (enable == True and hasattr(self, message_name)):
            
            # Get Message Variable Name
            message = getattr(self, message_name)
            
            # Extend Landmark Vector -> Saving New Keypoints 
            #Landmarks.append(np.array([[value.x, value.y, value.z, value.v] for value in message.keypoints]).flatten() if message else np.zeros(33*4))
            Landmarks.append(np.zeros(468 * 3) if message is None else np.array([[res.x, res.y, res.z, res.v] for res in message.keypoints]).flatten())

        return Landmarks

    
    # Gesture Recognition Function
    def Recognition(self):
        
        with torch.no_grad():

            # Coordinate Vector
            Landmarks = []
            
            # Check [Right Hand, Left Hand, Pose, Face] Landmarks
            Landmarks = self.process_landmarks(self.enable_right_hand, 'right_new_msg', Landmarks)
            Landmarks = self.process_landmarks(self.enable_left_hand,  'left_new_msg',  Landmarks)
            Landmarks = self.process_landmarks(self.enable_pose, 'pose_new_msg', Landmarks)
            Landmarks = self.process_landmarks(self.enable_face, 'face_new_msg', Landmarks)
            
            # Concatenate Landmarks Vectors
            keypoints = np.concatenate(Landmarks)
    
            # Append the Landmarks Coordinates from the Last Frame to our Sequence
            self.sequence.append(keypoints)
    
            # Analyze Only the Last 30 Frames
            self.sequence = self.sequence[-30:]
            
            if len(self.sequence) == 30:
                
                # Obtain the Probability of Each Gesture
                output = self.model(torch.Tensor(self.sequence).view(1, 30, -1))
                
                # Get the Probability of the Most Probable Gesture
                prob = torch.softmax(output, dim=1)[0] * 100
                
                #Print the probabilities
                print(prob, ": This is the probability Tensor")
                
                # Get the Index of the Highest Probability
                index = int(prob.argmax(dim = 0))
    
                # Print the Name of the Gesture Recognized
                if (prob[index] > self.recognition_precision_probability):
                    
                    Recognised_gesture = self.actions[index]
                    prob_recognised = float(prob[index])
                    
                    print("Recignised Gesture:", Recognised_gesture, "With this probability:", prob_recognised)
    
    
   
############################################################
#                           Main                           #
############################################################


if __name__ == '__main__':
    
    #Time to prepare yourself
    countdown(3)

    # Instantiate Gesture Recognition Class
    GR = GestureRecognition3D()
    
    while not rospy.is_shutdown():
        
        # Main Recognition Function
        GR.Recognition()
        
        # Sleep Rate Time
        GR.rate.sleep()
