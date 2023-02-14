import os
#import mediapipe as mp
#import cv2
#import rospy
import pandas as pd


video_folder = "/home/alberto/catkin_ws/src/mediapipe_gesture_recognition/dataset/Video_restanti/"

video_files = [
    f for f in os.listdir(video_folder) 
    if f.endswith('.mp4')
    ]

data = pd.read_csv("/home/alberto/catkin_ws/src/mediapipe_gesture_recognition/dataset/Labels/Total.csv")

i = 1

for video_file in video_files:

    #cap = cv2.VideoCapture(os.path.join(video_folder, video_file))

    video_name, _ = (os.path.splitext(video_file))
    try:
        video_number = int(video_name)
    except ValueError:
        print("Conversione da stringa a int fallita per", video_name)


    riga_label = data[data.iloc[:, 0] == video_number]
    label = riga_label.iloc[:,1]


    if   label.empty:
         print(i,"Sto passando il video:", video_number)
    else:
        print(i,"Sto passando il video:", video_number, "con comando:", label.iloc[0])
        i = i+1

    #cap.release()