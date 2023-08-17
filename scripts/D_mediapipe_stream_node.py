#!/usr/bin/env python3
#!/home/alberto/miniconda3/envs/myclone/bin python3


import rospy
import cv2
import mediapipe as mp
from mediapipe_gesture_recognition.msg import Pose, Face, Keypoint, Hand
import pyrealsense2 as rs
import numpy as np
import datetime as dt


'''
To Obtain The Available Cameras: 

  v4l2-ctl --list-devices

Intel(R) RealSense(TM) Depth Ca (usb-0000:00:14.0-2):
	/dev/video2
	/dev/video3
	/dev/video4 -> Black & White
	/dev/video5
	/dev/video6 -> RGB
	/dev/video7

VGA WebCam: VGA WebCam (usb-0000:00:14.0-5):
	/dev/video0 -> RGB
	/dev/video1
'''

class MediapipeStreaming:
  
  def __init__(self, webcam, enable_right_hand = False, enable_left_hand = False, \
                             enable_pose = False, enable_face = False, enable_face_detection = False, \
                             enable_objectron = False, objectron_model = 'Shoe'):
    
    # Mediapipe Publishers
    self.hand_right_pub  = rospy.Publisher('/mediapipe_gesture_recognition/right_hand', Hand, queue_size=1)
    self.hand_left_pub   = rospy.Publisher('/mediapipe_gesture_recognition/left_hand', Hand, queue_size=1)
    self.pose_pub        = rospy.Publisher('/mediapipe_gesture_recognition/pose', Pose, queue_size=1)
    self.face_pub        = rospy.Publisher('/mediapipe_gesture_recognition/face', Face, queue_size=1)
    
    # Constants
    self.RIGHT_HAND = True
    self.LEFT_HAND = False

    # Define Hand Landmark Names
    self.hand_landmarks_names = ['WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP', 
                                 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP', 
                                 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_MCP', 
                                 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP', 'PINKY_MCP', 
                                 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']
    
    # Define Pose Landmark Names
    self.pose_landmarks_names = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 
                                 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 
                                 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 
                                 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 
                                 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 
                                 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
    
    # Define Objectron Model Names
    self.available_objectron_models = ['Shoe', 'Chair', 'Cup', 'Camera']
    
    # Video Input
    # self.webcam = 0

    # Boolean Parameters
    self.enable_right_hand = enable_right_hand
    self.enable_left_hand = enable_left_hand
    self.enable_pose = enable_pose
    self.enable_face = enable_face
    self.enable_face_detection = enable_face_detection
    self.enable_objectron = enable_objectron
    self.objectron_model = objectron_model      
      
    if not self.objectron_model in self.available_objectron_models:
      rospy.logerr('ERROR: Objectron Model Not Available | Shutting Down...')
      rospy.signal_shutdown('ERROR: Objectron Model Not Available')

    # Initialize Mediapipe:
    self.mp_drawing = mp.solutions.drawing_utils
    self.mp_drawing_styles = mp.solutions.drawing_styles
    self.mp_holistic = mp.solutions.holistic
    self.mp_face_detection = mp.solutions.face_detection
    self.mp_objectron = mp.solutions.objectron

    #Initialize realSense
    self.realsense_ctx = rs.context()
    self.connected_devices = []

    for i in range(len(self.realsense_ctx.devices)):
      detected_camera = self.realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)

      self.connected_devices.append(detected_camera)
    
    self.device = self.connected_devices[-1] # In this example we are only using one camera
    # print(self.connected_devices)
    self.pipeline = rs.pipeline()
    self.config = rs.config()

    # ====== Enable Streams ======
    self. config.enable_device(self.device)
    self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # ====== Start Pipeline ======
    self.profile = self.pipeline.start(self.config)

    # ====== Align Depth to Color ======

    self.align_to = rs.stream.color
    self.align = rs.align(self.align_to)

    # ====== Get depth Scale ======
    self.depth_sensor = self.profile.get_device().first_depth_sensor()
    self.depth_scale = self.depth_sensor.get_depth_scale()
    # print(f"\tDepth Scale for Camera SN {device} is: {depth_scale}")

    # ====== Set clipping distance ====== #Oltre due metri di profondità, non vede più nulla (distanza di clipping)
    self.clipping_distance_in_meters = 2
    self.clipping_distance = self.clipping_distance_in_meters / self.depth_scale
    
    # Open Video Webcam
    # self.cap = cv2.VideoCapture(self.webcam)

    # # Check Webcam Availability
    # if self.cap is None or not self.cap.isOpened():
    #   rospy.logerr(f'ERROR: Webcam {self.webcam} Not Available | Starting Default: 0')
    #   self.cap = cv2.VideoCapture(0)
   
  
############################################################
#                Keypoint Utility Functions                #
############################################################
  
  def newKeypoint(self, landmark, number, name):
    
    # Assign Keypoint Coordinates
    new_keypoint = Keypoint()
    new_keypoint.x = landmark.x
    new_keypoint.y = landmark.y
    new_keypoint.z = landmark.z
    new_keypoint.v = landmark.visibility
    
    x = int(new_keypoint.x*len(self.depth_image_flipped[0]))
    y = int(new_keypoint.y*len(self.depth_image_flipped))

    if x >= len(self.depth_image_flipped[0]):
      x = len(self.depth_image_flipped[0]) - 1

    if y >= len(self.depth_image_flipped):
      y = len(self.depth_image_flipped) - 1

    new_keypoint.depth = self.depth_image_flipped[y, x] * self.depth_scale


    # Assign Keypoint Number and Name
    new_keypoint.keypoint_number = number
    new_keypoint.keypoint_name = name

    # if name == "LEFT_SHOULDER":

    #   print(new_keypoint)
  
    return new_keypoint


############################################################
#                    Holistic Functions                    #
############################################################

  def processHand(self, RightLeft, handResults, image):
        
    # Drawing the Hand Landmarks
    self.mp_drawing.draw_landmarks(
        image,
        handResults.right_hand_landmarks if RightLeft else handResults.left_hand_landmarks,
        self.mp_holistic.HAND_CONNECTIONS,
        self.mp_drawing_styles.get_default_hand_landmarks_style(),
        self.mp_drawing_styles.get_default_hand_connections_style())

    # Create Hand Message
    hand_msg = Hand()
    hand_msg.header.stamp = rospy.Time.now()
    hand_msg.header.frame_id = 'Hand Right Message' if RightLeft else 'Hand Left Message'
    hand_msg.right_or_left = hand_msg.RIGHT if RightLeft else hand_msg.LEFT
    
    if (((RightLeft == self.RIGHT_HAND) and (handResults.right_hand_landmarks)) 
     or ((RightLeft == self.LEFT_HAND)  and (handResults.left_hand_landmarks))):

      # Add Keypoints to Hand Message
      for i in range(len(handResults.right_hand_landmarks.landmark if RightLeft else handResults.left_hand_landmarks.landmark)):
      
        # Append Keypoint
        hand_msg.keypoints.append(self.newKeypoint(handResults.right_hand_landmarks.landmark[i] if RightLeft else handResults.left_hand_landmarks.landmark[i], 
                                                   i, self.hand_landmarks_names[i]))
        
      
      
      #print(hand_msg)       Trying to indentify the right amount of keypoints number 

      # Publish Hand Keypoint Message
      self.hand_right_pub.publish(hand_msg) if RightLeft else self.hand_left_pub.publish(hand_msg)

  def processPose(self, poseResults, image):
        
    # Drawing the Pose Landmarks
    self.mp_drawing.draw_landmarks(
        image,
        poseResults.pose_landmarks,
        self.mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

    # Create Pose Message
    pose_msg = Pose()
    pose_msg.header.stamp = rospy.Time.now()
    pose_msg.header.frame_id = 'Pose Message'

    if poseResults.pose_landmarks:
    
      # Add Keypoints to Pose Message
      for i in range(len(poseResults.pose_landmarks.landmark)):

        # Append Keypoint
        pose_msg.keypoints.append(self.newKeypoint(poseResults.pose_landmarks.landmark[i], i, self.pose_landmarks_names[i]))
        
      #print(pose_msg)  Trying to indentify the right amount of keypoints number 

      # Publish Pose Keypoint Message
      self.pose_pub.publish(pose_msg)    

  def processFace(self, faceResults, image):
      
    # Drawing the Face Landmarks
    self.mp_drawing.draw_landmarks(
        image,
        faceResults.face_landmarks,
        self.mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())

    # Create Face Message
    face_msg = Face()
    face_msg.header.stamp = rospy.Time.now()
    face_msg.header.frame_id = 'Face Message'

    if faceResults.face_landmarks:
    
      # Add Keypoints to Face Message
      for i in range(len(faceResults.face_landmarks.landmark)):

        # Assign Keypoint Coordinates
        new_keypoint = Keypoint()
        new_keypoint.x = faceResults.face_landmarks.landmark[i].x
        new_keypoint.y = faceResults.face_landmarks.landmark[i].y
        new_keypoint.z = faceResults.face_landmarks.landmark[i].z

        # Assign Keypoint Number
        new_keypoint.keypoint_number = i

        # Assign Keypoint Name (468 Landmarks -> Names = FACE_KEYPOINT_1 ...)
        new_keypoint.keypoint_name = f'FACE_KEYPOINT_{i+1}'

        # Append Keypoint
        face_msg.keypoints.append(new_keypoint)
      
      #print(face_msg)                Trying to indentify the right amount of keypoints number 
      self.face_pub.publish(face_msg)
      
      
############################################################
#                 Face Detection Functions                 #
############################################################
  
  def processFaceDetection(self, faceDetectionResults, image):
  
    if faceDetectionResults.detections:
      
      # Draw Face Detection
      for detection in faceDetectionResults.detections: self.mp_drawing.draw_detection(image, detection)
  
  
############################################################
#                   Objectron Functions                    #
############################################################
  
  def processObjectron(self, objectronResults, image):
        
    if objectronResults.detected_objects:
        
      for detected_object in objectronResults.detected_objects:
            
        # Draw Landmarks
        self.mp_drawing.draw_landmarks(
          image,
          detected_object.landmarks_2d,
          self.mp_objectron.BOX_CONNECTIONS)

        # Draw Axis
        self.mp_drawing.draw_axis(
          image,
          detected_object.rotation,
          detected_object.translation)
        
        
############################################################
#                    Process Functions                     #
############################################################

  def initSolutions(self):
        
    # Initialize Mediapipe Holistic
    if self.enable_right_hand or self.enable_left_hand or self.enable_pose or self.enable_face: 
      self.holistic = self.mp_holistic.Holistic(refine_face_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
      
    # Initialize Mediapipe Face Detection
    elif self.enable_face_detection: 
      self.face_detection = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    
    # Initialize Mediapipe Objectron
    elif self.enable_objectron and self.objectron_model in ['Shoe', 'Chair', 'Cup', 'Camera']:
      self.objectron = self.mp_objectron.Objectron(static_image_mode=False, max_num_objects=5, min_detection_confidence=0.4,
                                              min_tracking_confidence=0.99, model_name=self.objectron_model, model_complexity = 3) 

  def getResults(self, image):
        
      # Get Holistic Results from Mediapipe Holistic
      if self.enable_right_hand or self.enable_left_hand or self.enable_pose or self.enable_face: 
        self.holistic_results = self.holistic.process(image)
        
      # Get Face Detection Results from Mediapipe
      if self.enable_face_detection: self.face_detection_results = self.face_detection.process(image)

      # Get Objectron Results from Mediapipe
      if self.enable_objectron: self.objectron_results = self.objectron.process(image)
  
  def processResults(self, image):
        
    # Process Left Hand Landmarks
    if self.enable_left_hand:  self.processHand(self.LEFT_HAND,  self.holistic_results, image)
      
    # Process Right Hand Landmarks
    if self.enable_right_hand: self.processHand(self.RIGHT_HAND, self.holistic_results, image)

    # Process Pose Landmarks
    if self.enable_pose: self.processPose(self.holistic_results, image)
      
    # Process Face Landmarks
    if self.enable_face: self.processFace(self.holistic_results, image)
      
    # Process Face Detection
    if self.enable_face_detection: self.processFaceDetection(self.face_detection_results, image)

    # Process Objectron
    if self.enable_objectron: self.processObjectron(self.objectron_results, image)
      
############################################################
#                       Main Spinner                       #
############################################################
  
  def stream(self):
      
    # Initialize Mediapipe Solutions (Holistic, Face Detection, Objectron)
    self.initSolutions()
    
    # Open Webcam
    while not rospy.is_shutdown():

      start_time = dt.datetime.today().timestamp()

      
      # Get and align frames
      self.frames = self.pipeline.wait_for_frames()
      aligned_frames = self.align.process(self.frames)
      self.aligned_depth_frame = aligned_frames.get_depth_frame()
      self.color_frame = aligned_frames.get_color_frame()


      if not self.aligned_depth_frame or not self.color_frame:
        continue

      # Process images
      depth_image = np.asanyarray(self.aligned_depth_frame.get_data())
      self.depth_image_flipped = cv2.flip(depth_image,1)
      color_image = np.asanyarray(self.color_frame.get_data())

      self.depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
      images = cv2.flip(color_image,1)

      images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
      
      # Get Mediapipe Result
      self.getResults(images)
      
      # Process Mediapipe Results
      self.processResults(images)

      # Display FPS
      time_diff = dt.datetime.today().timestamp() - start_time
      fps = int(1 / time_diff)
      org = (20, 100)
      color = (0,50,255)
      thickness = 1
      self.color_images_rgb = cv2.putText(images, f"FPS: {fps}", org, cv2.FONT_HERSHEY_SIMPLEX, .5, color, thickness, cv2.LINE_AA)

      self.color_images_rgb = cv2.cvtColor(self.color_images_rgb, cv2.COLOR_BGR2RGB)
      
      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('MediaPipe Landmarks', self.color_images_rgb)
      if cv2.waitKey(5) & 0xFF == 27:
        break


############################################################
#                    ROS Initialization                    #
############################################################

def initROS(name, rate):
      
  # ROS Initialization
  rospy.init_node(name, anonymous=True)
  global ros_rate; ros_rate = rospy.Rate(rate)
  
  # Read Webcam Parameters
  global webcam_; webcam_ = rospy.get_param('webcam', 0)

  # Read Mediapipe Modules Parameters (Available Objectron Models = ['Shoe', 'Chair', 'Cup', 'Camera'])
  global enable_right_hand_;      enable_right_hand_      = rospy.get_param('enable_right_hand', False)
  global enable_left_hand_;       enable_left_hand_       = rospy.get_param('enable_left_hand', False)
  global enable_pose_;            enable_pose_            = rospy.get_param('enable_pose', False)
  global enable_face_;            enable_face_            = rospy.get_param('enable_face', False)
  global enable_face_detection_;  enable_face_detection_  = rospy.get_param('enable_face_detection', False)
  global enable_objectron_;       enable_objectron_       = rospy.get_param('enable_objectron', False)
  global objectron_model_;        objectron_model_        = rospy.get_param('objectron_model', 'Shoe')
  
  # Debug Print
  from termcolor import colored
  print(colored(f'\nFunctions Enabled:\n', 'yellow'))
  print(colored(f'  Right Hand: {enable_right_hand_}',  'green' if enable_right_hand_ else 'red'))
  print(colored(f'  Left  Hand: {enable_left_hand_}\n', 'green' if enable_left_hand_  else 'red'))
  print(colored(f'  Skeleton:   {enable_pose_}',        'green' if enable_pose_ else 'red'))
  print(colored(f'  Face Mesh:  {enable_face_}\n',      'green' if enable_face_ else 'red'))
  print(colored(f'  Objectron:       {enable_objectron_}',        'green' if enable_objectron_      else 'red'))
  print(colored(f'  Face Detection:  {enable_face_detection_}\n', 'green' if enable_face_detection_ else 'red'))
  
############################################################
#                           Main                           #
############################################################
  
if __name__ == '__main__':
  
  initROS('mediapipe_stream_node', 30)
      
  # Create Mediapipe Class
  if   enable_objectron_:      MediapipeStream = MediapipeStreaming(webcam_, enable_objectron = True, objectron_model = objectron_model_)
  elif enable_face_detection_: MediapipeStream = MediapipeStreaming(webcam_, enable_face_detection = True)
  else:                        MediapipeStream = MediapipeStreaming(webcam_, enable_right_hand_, enable_left_hand_, enable_pose_, enable_face_)
  
  # While ROS::OK
  while not rospy.is_shutdown():
    
    # Mediapipe Streaming Functions
    MediapipeStream.stream()

    # Sleep for the Remaining Cycle Time
    ros_rate.sleep()

  # Close Webcam
  MediapipeStream.cap.release()
