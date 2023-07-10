#!/usr/bin/env python3

import rospy, rospkg, os
import torch, numpy as np
from tqdm import tqdm
from typing import List, Union
from termcolor import colored

# Import Neural Network and Model
from training_node import NeuralClassifier
from utils.utils import DEVICE, load_parameters

# Import ROS Messages
from std_msgs.msg import Int32MultiArray
from mediapipe_gesture_recognition.msg import Pose, Face, Hand, Keypoint

class GestureRecognition3D:

  # Available Gesture Dictionary
  available_gestures = {}

  # ROS Gesture Message
  gesture_msg = Int32MultiArray()

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
    self.fusion_pub = rospy.Publisher('gesture', Int32MultiArray, queue_size=1)

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

      # Read Network Parameters from YAML File
      parameters = load_parameters(f'{package_path}/model/{gesture_file}/model_parameters.yaml')
      parameters['input_size']  = torch.Size(tuple(i for i in parameters['input_size']))
      parameters['output_size'] = torch.Size(tuple(i for i in parameters['output_size']))
      # print(colored(f'Parameters: {parameters}\n\n', 'green'))

      # Create the Sequence Tensor of the Right Shape
      self.sequence = torch.zeros(parameters['input_size'], dtype=torch.float32, device=DEVICE)
      # print(colored(f'Sequence Shape: {self.sequence.shape}\n\n', 'green'))

      # Load the Trained Model
      self.model = NeuralClassifier(*parameters.values()).to(DEVICE)
      self.model.load_state_dict(torch.load(FILE))
      self.model.eval()

    # ERROR: Model Not Available
    except FileNotFoundError: print(colored(f'ERROR: Model {gesture_file} Not Available\n\n', 'red')); exit(0)

    # Load the Names of the Saved Actions
    self.actions = np.sort(np.array([os.path.splitext(f)[0] for f in os.listdir(f'{package_path}/database/{gesture_file}/Gestures')]))
    for index, action in enumerate(self.actions): self.available_gestures[str(action)] = index
    print(colored(f'Available Gestures: {self.available_gestures}\n\n', 'green'))

    # Clear Terminal
    os.system('clear')

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
  def process_landmarks(self, enable:bool, message_name:str) -> Union[torch.Tensor, None]:

    """ Process Landmark Messages """

    # Check Landmarks Existence
    if (enable == True and hasattr(self, message_name)):

      # Get Message Variable Name
      message: Union[Hand, Pose, Face] = getattr(self, message_name)

      # Create Landmark Tensor -> Saving New Keypoints
      landmarks = torch.tensor([[value.x, value.y, value.z, value.v] for value in message.keypoints], device=DEVICE).flatten()

      # Clean Received Message
      # for value in message.keypoints: value.x, value.y, value.z, value.v = 0.0, 0.0, 0.0, 0.0

      return landmarks

    return None

  def print_probabilities(self, prob:torch.Tensor):

    # TODO: Better Gesture Print
    tqdm.write("{:<30} | {:<10}".format('Type of Gesture', 'Probability\n'))

    for i in range(len(self.actions)):

      # Print Colored Gesture
      color = 'red' if prob[i] < 0.45 else 'yellow' if prob[i] <0.8 else 'green'
      tqdm.write("{:<30} | {:<}".format(self.actions[i], colored("{:<.1f}%".format(prob[i]*100), color)))

    print("\n\n\n\n\n\n\n\n\n\n")

  # Gesture Recognition Function
  def Recognition(self):

    """ Gesture Recognition Function """

    while not rospy.is_shutdown():

      # Check [Right Hand, Left Hand, Pose, Face] Landmarks -> Create a Tensor List
      landmarks_list = [self.process_landmarks(enable, name) for enable, name in
                        zip([self.enable_right_hand, self.enable_left_hand, self.enable_pose, self.enable_face],
                            ['right_new_msg', 'left_new_msg', 'pose_new_msg', 'face_new_msg'])]

      # Remove None Values from the List, Concatenate the Tensors and Append to our Sequence
      keypoints = torch.cat([t for t in landmarks_list if t is not None], dim=0).unsqueeze(0)
      assert keypoints.shape[1] == self.sequence.shape[1], f'ERROR: Wrong Keypoints Shape: {keypoints.shape[1]} instead of {self.sequence.shape[1]}'
      self.sequence = torch.cat((self.sequence[1:], keypoints), dim=0)

      # Obtain the Probability of Each Gesture
      prob:torch.Tensor = self.model(self.sequence.unsqueeze(0))[0]

      # Get the Index of the Highest Probability
      index = int(prob.argmax(dim = 0))

      # Send Gesture Message if Probability is Higher than the Threshold
      if (prob[index] > self.recognition_precision_probability):

        # TODO: Fix Fusion -> Redo Training with Changed Gestures
        # Publish ROS Message with the Name of the Gesture Recognized
        self.gesture_msg.data = [self.available_gestures[self.actions[index]]]
        self.fusion_pub.publish(self.gesture_msg)

      # Print Gesture Probability
      self.print_probabilities(prob)

      # Sleep Rate Time
      self.rate.sleep()

if __name__ == '__main__':

  # Instantiate Gesture Recognition Class
  GR = GestureRecognition3D()

  # Main Recognition Function
  GR.Recognition()
