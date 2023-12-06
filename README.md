# Mediapipe Gesture Recognition

Gesture Recognition with Google MediaPipe

## Setup and Usage

### Convert Frames into Videos

- run mediapipe_gesture_recognition/useful_scripts/Pro_converter.py setting:

        root_path = your Gesture_frames folder
        video_with_labels_path = your video folder 
        data_file = your csv label total file path 

In your terminal:

    ros2 run mediapipe_gesture_recognition Pro_converter.py

### Get all Keypoints using MediaPipe API

- launch the video launch file with:

        ros2 launch mediapipe_gesture_recognition video_node_launch.py


### Train Neural Network

- Run PyTorch Training Node:

        ros2 run mediapipe_gesture_recognition pytorch_videotraining_node.py

### Use Trained Model to Recognize Gestures in Real Time

- Run PyTorch model:

        ros2 run mediapipe_gesture_recognition pytorch_recognition_node.py
