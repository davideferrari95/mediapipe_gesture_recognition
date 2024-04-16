# mediapipe_gesture_recognition

Gesture Recognition with Google MediaPipe

## Dataset Creation

### Convert Frames into Videos

- Run `dataset_converter.py`:

    &nbsp;

        rosrun mediapipe_gesture_recognition Pro_converter.py

  - `root_path` = your Gesture_frames folder
  - `video_with_labels_path` = your video folder
  - `data_file` = your .csv label total file path

### Get Keypoints using MediaPipe API

- Launch `process_dataset_node.py`:

    &nbsp;

        roslaunch mediapipe_gesture_recognition process_dataset_node.launch

### Train Neural Network

- Run `training_node.py`:

    &nbsp;

        python ../training_node.py

### Start Gesture Recognition

- Launch `recognition_node.py`:

    &nbsp;

        roslaunch mediapipe_gesture_recognition recognition_node.launch
