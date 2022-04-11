#!/usr/bin/env python3

import rospy
import csv
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle 
import os

from mediapipe_gesture_recognition.msg import Pose, Face, Hand
from mediapipe_stream_node import MediapipeStreaming 


#Define which gesture will be linked to which action for the robot
def Setup_of_robot_action():
    with open(f"/home/baheu/ws_sk_tracking/src/sk_tracking/TXT file/Position_Name/Position_Name_{Solution_Choice}.txt", "r") as file: 
        allText = file.read() 
        words = list(map(str, allText.split())) 
        print('\nFor the moment the robot can have 5 different information, GO, Turn left, Turn right, stop and back\nIn this next line of code you will define what action will do your position')
        print('\nTo set up a position you just have to write :\n1 for GO\n2 for LEFT\n3 for RIGHT\n4 for STOP\n5 for BACK')
    
    global liste
    nbr_pos=len(words)
    liste=["P1","P2","P3","P4","P5","P6","P7","P8","P9","P10"]
    for i in range(0,nbr_pos):   
        pos=input(f'\nFor which action your position {words[i]} will be used ? ')
        print(pos)
        if (pos=="1"):
            liste[0]=words[i]
        elif (pos=="2"):
            liste[1]=words[i]
        elif (pos=="3"):
            liste[2]=words[i]
        elif (pos=="4"):
            liste[3]=words[i]
        elif (pos=="5"):
            liste[4]=words[i]
        elif (pos=="6"):
           liste[5]=words[i]
        elif (pos=="7"):
           liste[6]=words[i]
        elif (pos=="8"):
           liste[4]=words[i]
        elif (pos=="9"):
           liste[8]=words[i]
        elif (pos=="10"):
           liste[9]=words[i]






print("\nDo you want to build a new solution model or to use a previous one ? ")
Build_or_not=input("\nWrite NEW for a new solution and PREVIOUS for a previous solution :  " )
webcam=0
choice_cam=input("\nWrite YES if you want to use an external camera, otherwise write NO :" )
if (choice_cam=="YES" or choice_cam=="yes" or choice_cam=="Yes"or choice_cam=="y"or choice_cam=="Y"):
    webcam=2
if (Build_or_not=="NEW" or Build_or_not=="New" or Build_or_not=="new"):

# LOOP TO CREATE A NEW MACHINE LEARNING MODEL
    Choice_Okay=False
    while (Choice_Okay == False):
        print("\nThe following name are your previous position saving, type the name before the point to use it !")
        dirPath = r"/home/tanguy/tanguy_ws/src/mediapipe_gesture_recognition/CSV files"                                                               #Take all files name stored in that path                                    
        result = [f for f in os.listdir(dirPath) if os.path.isfile(os.path.join(dirPath, f))]                                           #Take all files name stored and create a list 
        print(result)
        Solution_Choice=input("\nWhich want do you want to use ? (Type the exact name before the point) :  ")
        M_Solution=open(f'/home/tanguy/tanguy_ws/src/mediapipe_gesture_recognition/TXT file/Solution_Name/Solution_{Solution_Choice}.txt','r')        #Read the Solution TXT file with the name of your project 
        Mediapipe_Solution=M_Solution.read()                                                                                            #Same
        Pos_Name=open(f'/home/tanguy/tanguy_ws/src/mediapipe_gesture_recognition/TXT file/Position_Name/Position_Name_{Solution_Choice}.txt','r')     #Read the Postion name TXT file with the name of your project
        Position_Name=Pos_Name.read()                                                                                                   #Same
        print(f'\nYou choose {Solution_Choice} and you set up these following position : {Position_Name}  \n This position will use the following solution : {Mediapipe_Solution} ')

        # BUG : Setup_of_robot_action()

        Continue=input("\nIf you made a mistake during the setup please write NO, otherwise write YES :  ")
        if (Continue=="YES") or (Continue=="y") or (Continue== "Yes") or (Continue=="yes"):
            Choice_Okay=True

    train_model()




#LOOP TO USE A PREVIOUS MACHINE LEARNING MODEL
else :
    Choice_Okay2=False
    while (Choice_Okay2 == False):
        print("\nThe following name are your previous position saving, type the name before the point to use it !")
        dirPath = r"/home/tanguy/tanguy_ws/src/mediapipe_gesture_recognition/PKL files"
        result = [f for f in os.listdir(dirPath) if os.path.isfile(os.path.join(dirPath, f))]
        print(result)
        Solution_Choice=input("\nWhich want do you want to use ? (Type the exact name before the point) :  ")
        M_Solution=open(f'/home/tanguy/tanguy_ws/src/mediapipe_gesture_recognition/TXT file/Solution_Name/Solution_{Solution_Choice}.txt','r')
        Mediapipe_Solution=M_Solution.read()
        Pos_Name=open(f'/home/tanguy/tanguy_ws/src/mediapipe_gesture_recognition/TXT file/Position_Name/Position_Name_{Solution_Choice}.txt','r')
        Position_Name=Pos_Name.read()
        print(f'\nYou choose {Solution_Choice} and you set up these following position : {Position_Name}  \nThis position will use the following solution : {Mediapipe_Solution} ')

        #BUG : Setup_of_robot_action()

        Continue=input("\nIf you made a mistake during the setup please write NO, otherwise write YES :  ")
        if (Continue=="YES") or (Continue=="y") or (Continue== "Yes") or (Continue=="yes"):
            Choice_Okay2=True
