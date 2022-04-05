#!/usr/bin/env python3

import rospy
from mediapipe_gesture_recognition.msg import Pose, Face, Hand

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





def createfiles():
        # Firstly I create some txt files to save the parameters of this session to use it in the following nodes
        
        # Write the project name on a TXT file
        File_Name=input("What is your project name ?")
        Project_name_txt=open('/home/baheu/ws_sk_tracking/src/sk_tracking/TXT file/projectname.txt','w')    
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
        Solution_txt=open(f'/home/baheu/ws_sk_tracking/src/sk_tracking/TXT file/Solution_Name/Solution_{File_Name}.txt','w')
        Solution_txt.write(Solution)
        Solution_txt.close()
        

        # In a second time I create the structure of my CSV file
        # Depending of the parameters we don't have the same numbers of landmarks so the structure of the CSV file will change
        # This will create the first line of the CSV file with on the first column the name of the class and after the (x,y,z,v) coordinates of the first landmarks, second ...

        if (enable_right_hand== "enable" and  enable_left_hand=="enable" and enable_pose=="enable"):    
            landmarks = ['class']
            for val in range(1, 75+1): # BUG : Maybe we can use one value for the range to simplify the creation of the CSV file
                landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]   

            with open(f'/home/tanguy/tanguy_ws/src/mediapipe_gesture_recognition/CSV files/{File_Name}.csv', mode='w', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(landmarks)