# MediaPipe Gesture Recognition

3D Gesture Recognition with Google MediaPipe in ROS1 - Noetic

## Dependencies

- ROS Noetic
- MediaPipe

## Installation

- Clone the GitHub Repository in the ROS Workspace:

      git clone -b master https://github.com/davideferrari95/mediapipe_gesture_recognition

- Install the Required Dependencies:

      rosdep install --from-paths src --ignore-src -r -y

- Install the Python Dependencies:

      pip install -r requirements.txt

## Dataset Creation

### Record Gesture Videos Dataset

- Run `video_recorder.py`:

      python ../scripts/dataset_utilities video_recorder.py

  - Edit the `GESTURES` List to Add New Selectable Gestures.
  - Edit `video_duration`, `pause`, `video_format` to Change Recording Parameters.

### Convert Video Dataset in Keypoints using MediaPipe API

- Launch `process_dataset_node.py`:

      roslaunch mediapipe_gesture_recognition process_dataset_node.launch

  - `enable_right_hand`, `enable_left_hand`, `enable_pose`, `enable_face` to Enable/Disable Keypoints.
  - Apply `zero pre-padding` to Pad Keypoints to the Same Length.

## Train Neural Network

- Run `training_node.py`:

      python ../scripts/training_node.py

  - Edit `config/config.yaml` to Change Training Parameters.

## Start Gesture Recognition

- Launch `stream_node.py`:

      roslaunch mediapipe_gesture_recognition stream_node.launch

  - `enable_right_hand`, `enable_left_hand`, `enable_pose`, `enable_face` to Enable/Disable Keypoints.
  - `enable_face_detection` to Enable/Disable Face Detection.
  - `face_mesh_mode` to Change Face Mesh Mode.
  - `enable_objectron` to Enable/Disable Objectron.
  - `objectron_model` to Change Objectron Model.
  - `webcam` to Change Webcam Source.
  - `realsense` to Enable/Disable Intel RealSense.

- Launch `recognition_node.py`:

      roslaunch mediapipe_gesture_recognition recognition_node.launch

  - `recognition_precision_probability` to Change Recognition Precision Probability.
  - `realsense` to Enable/Disable Intel RealSense.
