# mediapipe_gesture_recognition

3D Gesture Recognition with Google MediaPipe

## Dataset Creation

### Convert Frames into Videos

- Run `dataset_converter.py`:

        rosrun mediapipe_gesture_recognition Pro_converter.py

  - `root_path` = your Gesture_frames folder
  - `video_with_labels_path` = your video folder
  - `data_file` = your .csv label total file path

### Get Keypoints using MediaPipe API

- Launch `process_dataset_node.py`:

        roslaunch mediapipe_gesture_recognition process_dataset_node.launch

## Train Neural Network

- Run `training_node.py`:

        python ../training_node.py

## Start Gesture Recognition

- Launch `stream_node.py`:

        roslaunch mediapipe_gesture_recognition stream_node.launch realsense:=True

- Launch `recognition_node.py`:

        roslaunch mediapipe_gesture_recognition recognition_node.launch
