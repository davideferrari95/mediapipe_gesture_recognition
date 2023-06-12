#!/usr/bin/env python3


#Useful libreries
import rospy, rospkg, numpy as np, pickle, warnings, os, matplotlib.pyplot as plt
from  matplotlib.animation import FuncAnimation
from PY_train_pickle_node import NeuralNetwork, CustomDataset, GestureRecognitionTraining3D 
from Utils import countdown
from std_msgs.msg import Int32MultiArray

#Pytorch libreries
import torch

# Ignore Pickle Warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Import Mediapipe Messages
from mediapipe_gesture_recognition.msg import Pose, Face, Hand
from termcolor import colored




class GestureRecognition3D:

    def __init__(self):
        
        # ROS Initialization
        rospy.init_node('mediapipe_gesture_recognition_training_node', anonymous=True)
        self.rate = rospy.Rate(20)
        
        # Mediapipe Subscribers
        rospy.Subscriber('/mediapipe_gesture_recognition/right_hand', Hand, self.RightHandCallback)
        rospy.Subscriber('/mediapipe_gesture_recognition/left_hand',  Hand, self.LeftHandCallback)
        rospy.Subscriber('/mediapipe_gesture_recognition/pose',       Pose, self.PoseCallback)
        rospy.Subscriber('/mediapipe_gesture_recognition/face',       Face, self.FaceCallback)

        #Fusion Publisher 
        self.pub = rospy.Publisher('gesture', Int32MultiArray, queue_size=1000)

        
        # Read Mediapipe Modules Parameters
        self.enable_right_hand = rospy.get_param('enable_right_hand', False)
        self.enable_left_hand  = rospy.get_param('enable_left_hand',  False)
        self.enable_pose = rospy.get_param('enable_pose', False)
        self.enable_face = rospy.get_param('enable_face', False)
 
        # Read Gesture Recognition Precision Probability Parameter
        self.recognition_precision_probability = rospy.get_param('recognition_precision_probability', 0.8)

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
        with open(f'{package_path}/model/py_trained_model.pth', 'rb') as FILE:

            self.model = torch.load(FILE)
            self.model.eval()
            
        # Load the Names of the Saved Actions
        self.actions = np.array([os.path.splitext(f)[0] for f in os.listdir(f'{package_path}/database/3D_Gestures/{gesture_file}/Padded_file')])
       
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
            Landmarks.append(np.array([[value.x, value.y, value.z, value.v] for value in message.keypoints]).flatten() if message else np.zeros(33*4))
            #Landmarks.append(np.zeros(468 * 3) if message is None else np.array([[res.x, res.y, res.z, res.v] for res in message.keypoints]).flatten())

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
            self.sequence = self.sequence[-85:]
            
            if len(self.sequence) == 85:
                
                # Obtain the Probability of Each Gesture
                output = self.model(torch.Tensor(self.sequence).view(1, 85, -1))
                
                # Get the Probability of the Most Probable Gesture
                prob = torch.softmax(output, dim=1)[0] 
                
                
                # Get the Index of the Highest Probability
                index = int(prob.argmax(dim = 0))
    
                # Print the Name of the Gesture Recognized
                if (prob[index] > self.recognition_precision_probability):
                    
                    Recognised_gesture = self.actions[index]
                    #prob_recognised = float(prob[index])
                    msg = Int32MultiArray()

                    
                    if Recognised_gesture == "Point at": 

                        print("Gesture Point at Recognised")
                        msg.data = [1]
                        self.pub.publish(msg)

                    if Recognised_gesture == "Stop":

                        print("Stop Recognised")
                        msg.data = [2]
                        self.pub.publish(msg)

                    if Recognised_gesture == "Pause":

                        print("Pause Recognised")
                        msg.data = [5]
                        self.pub.publish(msg)


                #Print del riconoscimento e colorazione a seconda del Ricoscimento
                print("\n\n\n\n\n\n\n\n\n")
                print("{:<30} | {:<10}".format("Type of gesture", "Probability\n"))

                for i in range(len(self.actions)):

                    color = None

                    if prob.numpy()[i] < 0.45:
                        color = 'red'
                    elif prob.numpy()[i] <0.8:
                        color = 'yellow'
                    else:
                        color = 'green'
                    colored_prob = colored("{:<.1f}%".format(prob.numpy()[i]*100), color)
                    print("{:<30} | {:<}".format(self.actions[i], colored_prob))

                print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
                # rospy.sleep(0.15)



                

    

    

   
############################################################
#                           Main                           #
############################################################


if __name__ == '__main__':
    
    #Time to prepare yourself
    countdown(2)

    # Instantiate Gesture Recognition Class
    GR = GestureRecognition3D()
    
    while not rospy.is_shutdown():
        
        # Main Recognition Function
        GR.Recognition()
        
        # Sleep Rate Time
        GR.rate.sleep()
