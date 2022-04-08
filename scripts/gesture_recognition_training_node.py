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



# Mediapipe Subscribers Callbacke
def handRightCallback(msg):
    print('-----------------------------------')
    print('Header', msg.header)
    print('---')
    print('Right or Left', msg.right_or_left)
    print('---')
    print('Keypoints', msg.keypoints) #msg.keypoints[i]

    Fn=open('/home/tanguy/tanguy_ws/src/mediapipe_gesture_recognition/TXT file/projectname.txt','r')  #Here I open the TXT file "projectname" and read it
    File_Name=Fn.read()     #I store the reading value in a variable to use it in my callback function to call the good csv file

    #Insert the landmarks coordinates in the csv file

    '''
    with open('names.csv', 'w', newline='') as csvfile:
    fieldnames = ['first_name', 'last_name']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerow({'first_name': 'Baked', 'last_name': 'Beans'})
    writer.writerow({'first_name': 'Lovely', 'last_name': 'Spam'})
    writer.writerow({'first_name': 'Wonderful', 'last_name': 'Spam'})
    '''

    for i in range (len(msg.keypoints)):
        with open(f'/home/tanguy/tanguy_ws/src/mediapipe_gesture_recognition/CSV files/{File_Name}.csv', mode='a', newline='') as f:         #Write the list into the CSV file
          csv.writer.writeheader()
          csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
          csv_writer.writerow({ 'keypoint.x': msg.keypoints[i].x, 'keypoint.y': msg.keypoints[i].y, 'keypoint.z': msg.keypoints[i].z,
                                'keypoint.v': msg.keypoints[i].v, 'keypoint number': msg.keypoints[i].keypoint_number, 'keypoint name': msg.keypoints[i].keypoint_name})   



#Initialisation
rospy.init_node('mediapipe_streamgesture_recognition_training_node', anonymous=True) 
rate = rospy.Rate(100) 
hand_right_pub  = rospy.Publisher('/mediapipe_gesture_recognition/right_hand', Hand, queue_size=1) 
hand_left_pub   = rospy.Publisher('/mediapipe_gesture_recognition/left_hand', Hand, queue_size=1)
pose_pub        = rospy.Publisher('/mediapipe_gesture_recognition/pose', Pose, queue_size=1)
face_pub        = rospy.Publisher('/mediapipe_gesture_recognition/face', Face, queue_size=1)

# Mediapipe Subscribers
rospy.Subscriber('/mediapipe_gesture_recognition/right_hand', Hand, handRightCallback)

# Read Mediapipe Modules Parameters
enable_right_hand = rospy.get_param('enable_right_hand', False)
enable_left_hand = rospy.get_param('enable_left_hand', False)
enable_pose = rospy.get_param('enable_pose', False)
enable_face = rospy.get_param('enable_face', False)




# While ROS OK
#while not rospy.is_shutdown():
#    ...




#######################################################################################################################################################




#Creation of the CSV file where the datas will be saved
def createfiles():
        # Firstly I create some txt files to save the parameters of this session to use it in the following nodes
        
        # Write the project name on a TXT file
        File_Name=input("What is your project name ?")
        Project_name_txt=open('/home/tanguy/tanguy_ws/src/mediapipe_gesture_recognition/TXT file/projectname.txt','w')
        Project_name_txt.write(File_Name)
        Project_name_txt.close()

        # Write the different solution used on a TXT file
        Solution=""
        if (enable_right_hand=="enable"):
            Solution= Solution + "Right"
        if (enable_left_hand=="enable"):
            Solution= Solution + " Left"
        if (enable_pose=="enable"):
            Solution= Solution + " Pose"
        Solution_txt=open(f'/home/tanguy/tanguy_ws/src/mediapipe_gesture_recognition/TXT file/Solution_Name/Solution_{File_Name}.txt','w')
        Solution_txt.write(Solution)
        Solution_txt.close()
        

        # In a second time I create the structure of my CSV file
        # Depending of the parameters we don't have the same numbers of landmarks so the structure of the CSV file will change
        # This will create   the first line of the CSV file with on the first column the name of the class and after the (x,y,z,v) coordinates of the first landmarks, second ...

        landmarks = ['class']
        for val in range (468+33+21+21+1): # BUG : Maybe we can use one value for the range to simplify the creation of the CSV file
            landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]   
        with open(f'/home/tanguy/tanguy_ws/src/mediapipe_gesture_recognition/CSV files/{File_Name}.csv', mode='w', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting= csv.QUOTE_MINIMAL)
            csv_writer.writerow(landmarks)

#3/ TRAIN CUSTOM MODEL USING SCIKIT LEARN
def train_model():
    # 3.1/ READ IN COLLECTED DATA AND PROCESS

    df = pd.read_csv(f'/home/tanguy/tanguy_ws/src/mediapipe_gesture_recognition/CSV files/{Solution_Choice}.csv')       #read the coordinates on the CSV file 
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

    with open(f'/home/tanguy/tanguy_ws/src/mediapipe_gesture_recognition/PKL files/{Solution_Choice}.pkl', 'wb') as f:       #These two lines is build to export the best model "here it's rf" and save it in a files called pose_recognition.pkl
        pickle.dump(fit_models['rf'], f)



#SETTINGS STEP

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

    MediapipeStreaming.stream 




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
    
    MediapipeStreaming.stream  