#!/usr/bin/env python3
# license removed for brevity
import rospy

import mediapipe as mp
import cv2

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
from turtle import back, backward, forward,right,left
from Robot_Class import RobotControl 

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe solutions
rc=RobotControl()

def detection():
    
    with open(f'/home/baheu/ws_sk_tracking/src/sk_tracking/PKL files/{Solution_Choice}.pkl', 'rb') as f:
        model = pickle.load(f)

    cap = cv2.VideoCapture(webcam)

    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            rc.__init__()
            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False        

            # Make Detections
            results = holistic.process(image)
            
            # Recolor image back to BGR for rendering
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


            if (Mediapipe_Solution=="Right Left Pose"): 
                 # 2. Right hand
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                    )
                 # 3. Left Hand
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                        )
                # 4. Pose Detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )
                # Export coordinates
                try:
                    
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())


                    #Extract Right Hand landmarks
                    right_hand = results.right_hand_landmarks.landmark
                    rh_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right_hand]).flatten())

                    #Extract Left Hand landmarks
                    left_hand = results.left_hand_landmarks.landmark
                    lh_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left_hand]).flatten())

                    # Concate rows
                    row = rh_row+lh_row+pose_row

                    X = pd.DataFrame([row])
                    pose_recognition_class = model.predict(X)[0]
                    pose_recognition_prob = model.predict_proba(X)[0]
                    print(pose_recognition_class, pose_recognition_prob)

                    # Grab ear coords
                    coords = tuple(np.multiply(
                                    np.array(
                                        (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                                , [640,480]).astype(int))

                    cv2.rectangle(image, 
                                (coords[0], coords[1]+5), 
                                (coords[0]+len(pose_recognition_class)*20, coords[1]-30), 
                                (245, 117, 16), -1)
                    cv2.putText(image, pose_recognition_class, coords, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Get status box
                    cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)

                    # Display Class
                    cv2.putText(image, 'CLASS'
                                , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, pose_recognition_class.split(' ')[0]
                                , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Display Probability
                    cv2.putText(image, 'PROB'
                                , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    
                    
                    cv2.putText(image, str(round(pose_recognition_prob[np.argmax(pose_recognition_prob)],2))
                                , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    

                    # Here you can set up your robot driving parameters with the name of your class (like STOP, LEFT, RIGHT...) and the probability of precision
                    Prob=max(pose_recognition_prob)
                    if (pose_recognition_class==liste[0]) and (Prob>0.7):
                       rc.move_straight()
                    elif (pose_recognition_class==liste[1]) and (Prob>0.7):
                       rc.turn_left()
                    elif (pose_recognition_class==liste[2]) and (Prob>0.6):
                       rc.turn_right()
                    elif (pose_recognition_class==liste[3]) and (Prob>0.7):
                       rc.stop()
                    elif (pose_recognition_class==liste[4]) and (Prob>0.7):
                       rc.move_back()
                    
                    



                except:
                 pass
            if (Mediapipe_Solution=="Right Left"): 
                   # 2. Right hand
                   mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                       mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                       mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                       )
                   # 3. Left Hand
                   mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                           mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                           mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                           )

                   # Export coordinates
                   try:

                       #Extract Right Hand landmarks
                       right_hand = results.right_hand_landmarks.landmark
                       rh_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right_hand]).flatten())

                       #Extract Left Hand landmarks
                       left_hand = results.left_hand_landmarks.landmark
                       lh_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left_hand]).flatten())

                       # Concate rows
                       row = rh_row+lh_row

                       X = pd.DataFrame([row])
                       pose_recognition_class = model.predict(X)[0]
                       pose_recognition_prob = model.predict_proba(X)[0]
                       print(pose_recognition_class, pose_recognition_prob)

                       # Grab ear coords
                       coords = tuple(np.multiply(
                                       np.array(
                                           (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                           results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                                   , [640,480]).astype(int))

                       cv2.rectangle(image, 
                                   (coords[0], coords[1]+5), 
                                   (coords[0]+len(pose_recognition_class)*20, coords[1]-30), 
                                   (245, 117, 16), -1)
                       cv2.putText(image, pose_recognition_class, coords, 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                       # Get status box
                       cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)

                       # Display Class
                       cv2.putText(image, 'CLASS'
                                   , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                       cv2.putText(image, pose_recognition_class.split(' ')[0]
                                   , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                       # Display Probability
                       cv2.putText(image, 'PROB'
                                   , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                       cv2.putText(image, str(round(pose_recognition_prob[np.argmax(pose_recognition_prob)],2))
                                   , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                       # Here you can set up your robot driving parameters with the name of your class (like STOP, LEFT, RIGHT...) and the probability of precision
                       Prob=max(pose_recognition_prob)
                       if (pose_recognition_class==liste[0]) and (Prob>0.7):
                          rc.move_straight()
                       elif (pose_recognition_class==liste[1]) and (Prob>0.7):
                          rc.turn_left()
                       elif (pose_recognition_class==liste[2]) and (Prob>0.6):
                          rc.turn_right()
                       elif (pose_recognition_class==liste[3]) and (Prob>0.7):
                          rc.stop()
                       elif (pose_recognition_class==liste[4]) and (Prob>0.7):
                          rc.move_back()
                    
                   except:
                    pass
            if (Mediapipe_Solution=="Right Pose"):
                # 2. Right hand
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                    )

                # 4. Pose Detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )
                # Export coordinates
                try:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())


                    #Extract Right Hand landmarks
                    right_hand = results.right_hand_landmarks.landmark
                    rh_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right_hand]).flatten())

                    # Concate rows
                    row = rh_row+pose_row

                    X = pd.DataFrame([row])
                    pose_recognition_class = model.predict(X)[0]
                    pose_recognition_prob = model.predict_proba(X)[0]
                    print(pose_recognition_class, pose_recognition_prob)

                    # Grab ear coords
                    coords = tuple(np.multiply(
                                    np.array(
                                        (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                                , [640,480]).astype(int))

                    cv2.rectangle(image, 
                                (coords[0], coords[1]+5), 
                                (coords[0]+len(pose_recognition_class)*20, coords[1]-30), 
                                (245, 117, 16), -1)
                    cv2.putText(image, pose_recognition_class, coords, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Get status box
                    cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)

                    # Display Class
                    cv2.putText(image, 'CLASS'
                                , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, pose_recognition_class.split(' ')[0]
                                , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Display Probability
                    cv2.putText(image, 'PROB'
                                , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(pose_recognition_prob[np.argmax(pose_recognition_prob)],2))
                                , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    # Here you can set up your robot driving parameters with the name of your class (like STOP, LEFT, RIGHT...) and the probability of precision
                    Prob=max(pose_recognition_prob)
                    if (pose_recognition_class==liste[0]) and (Prob>0.7):
                       rc.move_straight()
                    elif (pose_recognition_class==liste[1]) and (Prob>0.7):
                       rc.turn_left()
                    elif (pose_recognition_class==liste[2]) and (Prob>0.6):
                       rc.turn_right()
                    elif (pose_recognition_class==liste[3]) and (Prob>0.7):
                       rc.stop()
                    elif (pose_recognition_class==liste[4]) and (Prob>0.7):
                       rc.move_back()
                except:
                 pass
            if (Mediapipe_Solution=="Right"):
                # 2. Right hand
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                    )

                # Export coordinates
                try:


                    #Extract Right Hand landmarks
                    right_hand = results.right_hand_landmarks.landmark
                    rh_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right_hand]).flatten())


                    # Concate rows
                    row = rh_row

                    X = pd.DataFrame([row])
                    pose_recognition_class = model.predict(X)[0]
                    pose_recognition_prob = model.predict_proba(X)[0]
                    print(pose_recognition_class, pose_recognition_prob)

                    # Grab ear coords
                    coords = tuple(np.multiply(
                                    np.array(
                                        (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                                , [640,480]).astype(int))

                    cv2.rectangle(image, 
                                (coords[0], coords[1]+5), 
                                (coords[0]+len(pose_recognition_class)*20, coords[1]-30), 
                                (245, 117, 16), -1)
                    cv2.putText(image, pose_recognition_class, coords, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Get status box
                    cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)

                    # Display Class
                    cv2.putText(image, 'CLASS'
                                , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, pose_recognition_class.split(' ')[0]
                                , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Display Probability
                    cv2.putText(image, 'PROB'
                                , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(pose_recognition_prob[np.argmax(pose_recognition_prob)],2))
                                , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    # Here you can set up your robot driving parameters with the name of your class (like STOP, LEFT, RIGHT...) and the probability of precision
                    Prob=max(pose_recognition_prob)
                    if (pose_recognition_class==liste[0]) and (Prob>0.7):
                       rc.move_straight()
                    elif (pose_recognition_class==liste[1]) and (Prob>0.7):
                       rc.turn_left()
                    elif (pose_recognition_class==liste[2]) and (Prob>0.6):
                       rc.turn_right()
                    elif (pose_recognition_class==liste[3]) and (Prob>0.7):
                       rc.stop()
                    elif (pose_recognition_class==liste[4]) and (Prob>0.7):
                       rc.move_back()
                except:
                 pass
            if (Mediapipe_Solution==" Left Pose"):

                 # 3. Left Hand
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                        )
                # 4. Pose Detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )
                # Export coordinates
                try:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                    #Extract Left Hand landmarks
                    left_hand = results.left_hand_landmarks.landmark
                    lh_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left_hand]).flatten())

                    # Concate rows
                    row = lh_row+pose_row

                    X = pd.DataFrame([row])
                    pose_recognition_class = model.predict(X)[0]
                    pose_recognition_prob = model.predict_proba(X)[0]
                    print(pose_recognition_class, pose_recognition_prob)

                    # Grab ear coords
                    coords = tuple(np.multiply(
                                    np.array(
                                        (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                                , [640,480]).astype(int))

                    cv2.rectangle(image, 
                                (coords[0], coords[1]+5), 
                                (coords[0]+len(pose_recognition_class)*20, coords[1]-30), 
                                (245, 117, 16), -1)
                    cv2.putText(image, pose_recognition_class, coords, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Get status box
                    cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)

                    # Display Class
                    cv2.putText(image, 'CLASS'
                                , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, pose_recognition_class.split(' ')[0]
                                , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Display Probability
                    cv2.putText(image, 'PROB'
                                , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(pose_recognition_prob[np.argmax(pose_recognition_prob)],2))
                                , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    # Here you can set up your robot driving parameters with the name of your class (like STOP, LEFT, RIGHT...) and the probability of precision
                    Prob=max(pose_recognition_prob)
                    if (pose_recognition_class==liste[0]) and (Prob>0.7):
                       rc.move_straight()
                    elif (pose_recognition_class==liste[1]) and (Prob>0.7):
                       rc.turn_left()
                    elif (pose_recognition_class==liste[2]) and (Prob>0.6):
                       rc.turn_right()
                    elif (pose_recognition_class==liste[3]) and (Prob>0.7):
                       rc.stop()
                    elif (pose_recognition_class==liste[4]) and (Prob>0.7):
                       rc.move_back()
                except:
                 pass
            if (Mediapipe_Solution==" Left"):

                 # 3. Left Hand
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                        )

                # Export coordinates
                try:

                    #Extract Left Hand landmarks
                    left_hand = results.left_hand_landmarks.landmark
                    lh_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left_hand]).flatten())

                    # Concate rows
                    row = lh_row

                    X = pd.DataFrame([row])
                    pose_recognition_class = model.predict(X)[0]
                    pose_recognition_prob = model.predict_proba(X)[0]
                    print(pose_recognition_class, pose_recognition_prob)

                    # Grab ear coords
                    coords = tuple(np.multiply(
                                    np.array(
                                        (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                                , [640,480]).astype(int))

                    cv2.rectangle(image, 
                                (coords[0], coords[1]+5), 
                                (coords[0]+len(pose_recognition_class)*20, coords[1]-30), 
                                (245, 117, 16), -1)
                    cv2.putText(image, pose_recognition_class, coords, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Get status box
                    cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)

                    # Display Class
                    cv2.putText(image, 'CLASS'
                                , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, pose_recognition_class.split(' ')[0]
                                , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Display Probability
                    cv2.putText(image, 'PROB'
                                , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(pose_recognition_prob[np.argmax(pose_recognition_prob)],2))
                                , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    # Here you can set up your robot driving parameters with the name of your class (like STOP, LEFT, RIGHT...) and the probability of precision
                    Prob=max(pose_recognition_prob)
                    if (pose_recognition_class==liste[0]) and (Prob>0.7):
                       rc.move_straight()
                    elif (pose_recognition_class==liste[1]) and (Prob>0.7):
                       rc.turn_left()
                    elif (pose_recognition_class==liste[2]) and (Prob>0.6):
                       rc.turn_right()
                    elif (pose_recognition_class==liste[3]) and (Prob>0.7):
                       rc.stop()
                    elif (pose_recognition_class==liste[4]) and (Prob>0.7):
                       rc.move_back()
                except:
                 pass
            if (Mediapipe_Solution==" Pose"):

                # 4. Pose Detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )
                # Export coordinates
                try:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())


                    # Concate rows
                    row = pose_row

                    X = pd.DataFrame([row])
                    pose_recognition_class = model.predict(X)[0]
                    pose_recognition_prob = model.predict_proba(X)[0]
                    print(pose_recognition_class, pose_recognition_prob)

                    # Grab ear coords
                    coords = tuple(np.multiply(
                                    np.array(
                                        (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                                , [640,480]).astype(int))

                    cv2.rectangle(image, 
                                (coords[0], coords[1]+5), 
                                (coords[0]+len(pose_recognition_class)*20, coords[1]-30), 
                                (245, 117, 16), -1)
                    cv2.putText(image, pose_recognition_class, coords, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Get status box
                    cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)

                    # Display Class
                    cv2.putText(image, 'CLASS'
                                , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, pose_recognition_class.split(' ')[0]
                                , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Display Probability
                    cv2.putText(image, 'PROB'
                                , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(pose_recognition_prob[np.argmax(pose_recognition_prob)],2))
                                , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    # Here you can set up your robot driving parameters with the name of your class (like STOP, LEFT, RIGHT...) and the probability of precision
                    Prob=max(pose_recognition_prob)
                    if (pose_recognition_class==liste[0]) and (Prob>0.7):
                       rc.move_straight()
                    elif (pose_recognition_class==liste[1]) and (Prob>0.7):
                       rc.turn_left()
                    elif (pose_recognition_class==liste[2]) and (Prob>0.6):
                       rc.turn_right()
                    elif (pose_recognition_class==liste[3]) and (Prob>0.7):
                       rc.stop()
                    elif (pose_recognition_class==liste[4]) and (Prob>0.7):
                       rc.move_back()
                except:
                 pass


            cv2.imshow('Raw Webcam Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()


def train_model():
    #3/ TRAIN CUSTOM MODEL USING SCIKIT LEARN
    # 3.1/ READ IN COLLECTED DATA AND PROCESS

    df = pd.read_csv(f'/home/baheu/ws_sk_tracking/src/sk_tracking/CSV files/{Solution_Choice}.csv')       #read the coordinates on the CSV file 
    X = df.drop('class', axis=1)                                                                         # only show the features, like, only the coordinates not the class name
    y = df['class']                                                                                      # only show the target value witch is basically the class name
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)          #Take large random value with train and take small random value with test


    # 3.2/ TRAIN MACHINE LEARNING CLASSIFICATION MODEL

    pipelines = {                                                                                         #Create different pipelines, here you have 4 different machine learning model, later we will choose the best one
        'lr':make_pipeline(StandardScaler(), LogisticRegression()),
        'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
        'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
        'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
                }

    fit_models = {}                         
    for algo, pipeline in pipelines.items():
        model = pipeline.fit(X_train, y_train)
        fit_models[algo] = model                                                                          #this 4 lines is to run the automatic learning

    # 3.3/ EVALUATE AN SERIALIZE MODEL

    for algo, model in fit_models.items():
        yhat = model.predict(X_test)
        print(algo, accuracy_score(y_test, yhat))                                                            #These line is to predict and showed the precision of the 4 pipelines, to choose witch one is the preciser

    with open(f'/home/baheu/ws_sk_tracking/src/sk_tracking/PKL files/{Solution_Choice}.pkl', 'wb') as f:       #These two lines is build to export the best model "here it's rf" and save it in a files called pose_recognition.pkl
        pickle.dump(fit_models['rf'], f)

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
        
    

#SETTINGS STEP

print("\nDo you want to build a new solution model or to use a previous one ? ")
Build_or_not=input("\nWrite NEW for a new solution and PREVIOUS for a previous solution :  " )
webcam=0
choice_cam=input("\nWrite YES if you want to use an external camera, otherwise write NO" )
if (choice_cam=="YES" or choice_cam=="yes" or choice_cam=="Yes"or choice_cam=="y"or choice_cam=="Y"):
    webcam=2
if (Build_or_not=="NEW" or Build_or_not=="New" or Build_or_not=="new"):


# LOOP TO CREATE A NEW MACHINE LEARNING MODEL
    
    Choice_Okay=False
    while (Choice_Okay == False):
        print("\nThe following name are your previous position saving, type the name before the point to use it !")
        dirPath = r"/home/baheu/ws_sk_tracking/src/sk_tracking/CSV files"                                                               #Take all files name stored in that path                                    
        result = [f for f in os.listdir(dirPath) if os.path.isfile(os.path.join(dirPath, f))]                                           #Take all files name stored and create a list 
        print(result)
        Solution_Choice=input("\nWhich want do you want to use ? (Type the exact name before the point) :  ")
        M_Solution=open(f'/home/baheu/ws_sk_tracking/src/sk_tracking/TXT file/Solution_Name/Solution_{Solution_Choice}.txt','r')        #Read the Solution TXT file with the name of your project 
        Mediapipe_Solution=M_Solution.read()                                                                                            #Same
        Pos_Name=open(f'/home/baheu/ws_sk_tracking/src/sk_tracking/TXT file/Position_Name/Position_Name_{Solution_Choice}.txt','r')     #Read the Postion name TXT file with the name of your project
        Position_Name=Pos_Name.read()                                                                                                   #Same
        print(f'\nYou choose {Solution_Choice} and you set up these following position : {Position_Name}  \n This position will use the following solution : {Mediapipe_Solution} ')

        Setup_of_robot_action()
       
        Continue=input("\nIf you made a mistake during the setup please write NO, otherwise write YES :  ")
        if (Continue=="YES") or (Continue=="y") or (Continue== "Yes") or (Continue=="yes"):
            Choice_Okay=True

    train_model()

    detection()




#LOOP TO USE A PREVIOUS MACHINE LEARNING MODEL
else :
    Choice_Okay2=False
    while (Choice_Okay2 == False):
        print("\nThe following name are your previous position saving, type the name before the point to use it !")
        dirPath = r"/home/baheu/ws_sk_tracking/src/sk_tracking/PKL files"
        result = [f for f in os.listdir(dirPath) if os.path.isfile(os.path.join(dirPath, f))]
        print(result)
        Solution_Choice=input("\nWhich want do you want to use ? (Type the exact name before the point) :  ")
        M_Solution=open(f'/home/baheu/ws_sk_tracking/src/sk_tracking/TXT file/Solution_Name/Solution_{Solution_Choice}.txt','r')
        Mediapipe_Solution=M_Solution.read()
        Pos_Name=open(f'/home/baheu/ws_sk_tracking/src/sk_tracking/TXT file/Position_Name/Position_Name_{Solution_Choice}.txt','r')
        Position_Name=Pos_Name.read()
        print(f'\nYou choose {Solution_Choice} and you set up these following position : {Position_Name}  \nThis position will use the following solution : {Mediapipe_Solution} ')

        Setup_of_robot_action()

        Continue=input("\nIf you made a mistake during the setup please write NO, otherwise write YES :  ")
        if (Continue=="YES") or (Continue=="y") or (Continue== "Yes") or (Continue=="yes"):
            Choice_Okay2=True
    
    detection()