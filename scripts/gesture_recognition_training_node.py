#!/usr/bin/env python3

import rospy
import csv
from mediapipe_gesture_recognition.msg import Pose, Face, Hand
#from mediapipe_stream_node import


rospy.init_node('mediapipe_streamgesture_recognition_training_node', anonymous=True)
rate = rospy.Rate(100) 

hand_left_pub   = rospy.Publisher('/mediapipe_gesture_recognition/left_hand', Hand, queue_size=1)
pose_pub        = rospy.Publisher('/mediapipe_gesture_recognition/pose', Pose, queue_size=1)
face_pub        = rospy.Publisher('/mediapipe_gesture_recognition/face', Face, queue_size=1)

# Mediapipe Subscribers Callbacke
def handRightCallback(msg):
    print('-----------------------------------')
    print('Header', msg.header)
    print('---')
    print('Right or Left', msg.right_or_left)
    print('---')
    print('Keypoints', msg.keypoints) #msg.keypoints[i]

# Mediapipe Subscribers
rospy.Subscriber('/mediapipe_gesture_recognition/right_hand', Hand, handRightCallback)

# Read Mediapipe Modules Parameters
enable_right_hand = rospy.get_param('enable_right_hand', False)
enable_left_hand = rospy.get_param('enable_left_hand', False)
enable_pose = rospy.get_param('enable_pose', False)
enable_face = rospy.get_param('enable_face', False)


# While ROS OK
while not rospy.is_shutdown():
    ...




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
        # This will create the first line of the CSV file with on the first column the name of the class and after the (x,y,z,v) coordinates of the first landmarks, second ...

        landmarks = ['class']
        for val in range len(landmark): # BUG : Maybe we can use one value for the range to simplify the creation of the CSV file
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

    with open(f'/home/baheu/ws_sk_tracking/src/sk_tracking/PKL files/{Solution_Choice}.pkl', 'wb') as f:       #These two lines is build to export the best model "here it's rf" and save it in a files called pose_recognition.pkl
        pickle.dump(fit_models['rf'], f)