# mediapipe_gesture_recognition
Gesture Recognition with Google MediaPipe

# Step to follow to use Mediapipe Gesture Recognition

## Step 1: 

### Convert your frames into videos

```
run mediapipe_gesture_recognition/useful_scripts/Pro_converter.py setting:

- root_path = your Gesture_frames folder
- video_with_labels_path = your video folder 
- data_file = your csv label total file path 

In your terminal:

rosrun mediapipe_gesture_recognition Pro_converter.py

```

## Step 2: 

### See all the video and get all the keypoints using mediapipe API
```

launch the video launch file with:

roslaunch mediapipe_gesture_recognition video_node.launch

```

## Step 3: 


### Train your Neural Network:

```
If you want TensorFlow NN, run mediapipe_gesture_recognition/scripts/tensorflow_videotraining_node.py, in your terminal:

rosrun mediapipe_gesture_recognition tensorflow_videotraining_node.py

If you want Pytorch NN, run mediapipe_gesture_recognition/scripts/pytorch_videotraining_node.py, in your terminal:

rosrun mediapipe_gesture_recognition pytorch_videotraining_node.py

```

## Step 4: 

### Use your trained model to recognise your gesture: 

```
If you want to use Tensorflow model, run mediapipe_gesture_recognition/scripts/tensorflow_recognition_node.py, in your terminal:

rosrun mediapipe_gesture_recognition tensorflow_recognition_node.py

If you want to use Pytorch model, run mediapipe_gesture_recognition/scripts/pytorch_recognition_node.py, in your terminal:

rosrun mediapipe_gesture_recognition pytorch_recognition_node.py

```