#!/usr/bin/env python3

import rospy, rospkg, os
import numpy as np
from typing import List, Union
from termcolor import colored

# Import Neural Network and Model
from matplotlib.animation import FuncAnimation
from PY_train_pickle_node import NeuralNetwork, CustomDataset, GestureRecognitionTraining3D
import torch

# Import ROS Messages
from std_msgs.msg import Int32MultiArray
from mediapipe_gesture_recognition.msg import Pose, Face, Hand, Keypoint

class GestureRecognition3D:

  # Number of Consecutive Frames Needed to Make our Prediction
  sequence = []

  # Available Gesture Dictionary
  available_gestures = {}

  def __init__(self):

    # ROS Initialization
    rospy.init_node('mediapipe_gesture_recognition_node', anonymous=True)
    # TODO: why 20 FPS ?
    self.rate = rospy.Rate(20)

    # Initialize Keypoint Messages
    self.initKeypointMessages()

    # Mediapipe Subscribers
    rospy.Subscriber('/mediapipe_gesture_recognition/right_hand', Hand, self.RightHandCallback)
    rospy.Subscriber('/mediapipe_gesture_recognition/left_hand',  Hand, self.LeftHandCallback)
    rospy.Subscriber('/mediapipe_gesture_recognition/pose',       Pose, self.PoseCallback)
    rospy.Subscriber('/mediapipe_gesture_recognition/face',       Face, self.FaceCallback)

    # Fusion Publisher
    self.fusion_pub = rospy.Publisher('gesture', Int32MultiArray, queue_size=1000)

    # Read Mediapipe Modules Parameters
    self.enable_right_hand = rospy.get_param('mediapipe_gesture_recognition/enable_right_hand', False)
    self.enable_left_hand  = rospy.get_param('mediapipe_gesture_recognition/enable_left_hand',  False)
    self.enable_pose = rospy.get_param('mediapipe_gesture_recognition/enable_pose', False)
    self.enable_face = rospy.get_param('mediapipe_gesture_recognition/enable_face', False)

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
    print(colored(f'\n\nLoading: {gesture_file} Configuration', 'yellow'))

    try:

      # Load the Trained Model for the Detected Landmarks
      FILE = open(f'{package_path}/model/{gesture_file}/model.pth', 'rb')

      self.model = torch.load(FILE)
      self.model.eval()

    # ERROR: Model Not Available
    except FileNotFoundError: print(colored(f'ERROR: Model {gesture_file} Not Available\n\n', 'red')); exit(0)

    # Load the Names of the Saved Actions
    self.actions = np.array([os.path.splitext(f)[0] for f in os.listdir(f'{package_path}/database/3D_Gestures/{gesture_file}/Gestures')])
    for index, action in enumerate(np.sort(self.actions)): self.available_gestures[str(action)] = index
    print(colored(f'Available Gestures: {self.available_gestures}\n\n', 'green'))

  def initKeypointMessages(self):

    """ Initialize Keypoint Messages """

    # Constants
    RIGHT_HAND, LEFT_HAND = True, False

    # Define Hand Landmark Names
    hand_landmarks_names = ['WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP',
                            'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP',
                            'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_MCP',
                            'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP', 'PINKY_MCP',
                            'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']

    # Define Pose Landmark Names
    pose_landmarks_names = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER',
                            'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
                            'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST',
                            'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB',
                            'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE',
                            'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']

    face_landmarks = 478

    # Init Keypoint Messages
    self.right_new_msg, self.left_new_msg, self.pose_new_msg, self.face_new_msg = Hand(), Hand(), Pose(), Face()
    self.right_new_msg.right_or_left, self.left_new_msg.right_or_left = RIGHT_HAND, LEFT_HAND
    self.right_new_msg.keypoints = self.left_new_msg.keypoints = [Keypoint() for _ in range(len(hand_landmarks_names))]
    self.pose_new_msg.keypoints = [Keypoint() for _ in range(len(pose_landmarks_names))]
    self.face_new_msg.keypoints = [Keypoint() for _ in range(face_landmarks)]

    # Hand Keypoint Messages
    for index, keypoint in enumerate(self.right_new_msg.keypoints): keypoint.keypoint_number, keypoint.keypoint_name = index + 1, hand_landmarks_names[index]
    for index, keypoint in enumerate(self.left_new_msg.keypoints):  keypoint.keypoint_number, keypoint.keypoint_name = index + 1, hand_landmarks_names[index]
    for index, keypoint in enumerate(self.pose_new_msg.keypoints):  keypoint.keypoint_number, keypoint.keypoint_name = index + 1, pose_landmarks_names[index]
    for index, keypoint in enumerate(self.face_new_msg.keypoints):  keypoint.keypoint_number, keypoint.keypoint_name = index + 1, f'FACE_KEYPOINT_{index + 1}'

  # Callback Functions
  def RightHandCallback(self, data:Hand): self.right_new_msg = data
  def LeftHandCallback(self,  data:Hand): self.left_new_msg  = data
  def PoseCallback(self, data:Pose):      self.pose_new_msg  = data
  def FaceCallback(self, data:Face):      self.face_new_msg  = data

  # Process Landmark Messages Function
  def process_landmarks(self, enable:bool, message_name:str, Landmarks:List[np.ndarray]):

    """ Process Landmark Messages """

    # Check Landmarks Existence
    if (enable == True and hasattr(self, message_name)):

      # Get Message Variable Name
      message: Union[Hand, Pose, Face] = getattr(self, message_name)

      # Extend Landmark Vector -> Saving New Keypoints
      Landmarks.append(np.array([[value.x, value.y, value.z, value.v] for value in message.keypoints]).flatten() if message else np.zeros(33*4))
      #Landmarks.append(np.zeros(468 * 3) if message is None else np.array([[res.x, res.y, res.z, res.v] for res in message.keypoints]).flatten())

      # Clean Message
      # for value in message.keypoints: value.x, value.y, value.z, value.v = 0.0, 0.0, 0.0, 0.0

    return Landmarks

  # Gesture Recognition Function
  def Recognition(self):

    """ Gesture Recognition Function """

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

          # Get Recognized Gesture from the Gesture List
          recognized_gesture = self.actions[index]
          # print(f'Gesture Recognized: "{recognized_gesture}"')

          # TODO: Fix Fusion -> Redo Training with Changed Gestures
          # Publish ROS Message
          msg = Int32MultiArray()
          msg.data = [self.available_gestures[recognized_gesture]]
          self.fusion_pub.publish(msg)

          # TODO: Better Gesture Print
          print("\n\n\n\n\n\n\n\n\n")
          print("{:<30} | {:<10}".format('Type of Gesture', 'Probability\n'))

          for i in range(len(self.actions)):

            # Print Colored Gesture
            color = 'red' if prob.numpy()[i] < 0.45 else 'yellow' if prob.numpy()[i] <0.8 else 'green'
            print("{:<30} | {:<}".format(self.actions[i], colored("{:<.1f}%".format(prob.numpy()[i]*100), color)))

          print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

if __name__ == '__main__':

  # Instantiate Gesture Recognition Class
  GR = GestureRecognition3D()

  while not rospy.is_shutdown():

    # Main Recognition Function
    GR.Recognition()

    # Sleep Rate Time
    GR.rate.sleep()
