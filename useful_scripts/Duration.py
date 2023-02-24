import os
from moviepy.video.io.VideoFileClip import VideoFileClip

 
gesture = "Stop Sign"

#Determine video folder
folder_directory = "/home/alberto/catkin_ws/src/mediapipe_gesture_recognition/dataset/Video_with_labels"

#Determine gesture folder
directory = os.path.join(folder_directory, gesture)
#Video path
video_files = [f for f in os.listdir(directory) if f.endswith(('.mp4', '.avi', '.mov'))]

#Initialise total duration
total_duration = 0

#Loop through video files
for filename in video_files:
    video_path = os.path.join(directory, filename)
    clip = VideoFileClip(video_path)
    total_duration += clip.duration

    no_sequence = int(total_duration) +1   #The next integer number

print(f'Total second for {gesture}:', total_duration, 'seconds, with',no_sequence, 'sequences')
