#!/usr/bin/env python3

# 0/ IMPORT THE DEPENDENCIES

import mediapipe as mp
import cv2
import csv
import numpy as np
from sk_tracking.msg import Holistic
import rospy
import numpy as np

#Define some variable with the intial values setted up on the launch file
E_D_Webcam = rospy.get_param("/webcam")              
E_D_Right_Hand = rospy.get_param("/enable_right_hand")
E_D_Left_Hand = rospy.get_param("/enable_left_hand")
E_D_Pose = rospy.get_param("/enable_pose")


def talker():
  
 
 pub_h = rospy.Publisher('H_Topic', Holistic, queue_size=10) # Create my publisher


 rospy.init_node('talker', anonymous=True) #Initialize my node
 rate = rospy.Rate(10) # 10hz 
 
 
        
 mp_drawing = mp.solutions.drawing_utils # Drawing helpers
 mp_holistic = mp.solutions.holistic # Mediapipe olutions
        
        
 nbr_pos=int(input("How many position do you want to setup? (min of 2) "))
 i=0
 name_position=[]
 while (i<nbr_pos):
             i+=1
             class_name=input("What's the name of your position ?")
             name_position.append(class_name)
             print("Press q want your position is setup")
             cap = cv2.VideoCapture(E_D_Webcam)
             
             with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:  # Initiate holistic model
                while cap.isOpened():
                    try:
                     ret, frame = cap.read()
                    # Recolor Feed
                     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                     image.flags.writeable = False        
                    # Make Detections
                     results = holistic.process(image)
                    # Recolor image back to BGR for rendering
                     image.flags.writeable = True   
                     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # All the different loop of extracting and publishing the landmarks, the choosen loop will depend with which parameters you will use
                     if (E_D_Right_Hand == "enable" and  E_D_Left_Hand=="enable" and E_D_Pose=="enable" ):  #This loop will use 3 mediapipes solutions (Right hand, Left Hand and Pose)
                         # Drawing of right hand points on the screen
                         mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                                              mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                                              )
                         #Drawing of the left hand points on the screen
                         mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                                              mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                                              mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                                              )
                         # Drawing of the Pose points on the screen 
                         mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                                              )
                         msg_h = Holistic()  # Set a variable as the type of my msg file called Holistic
                         if results.right_hand_landmarks :                                  
                            
                            Right_hand_results = results.right_hand_landmarks.landmark  #Set up a variable with the value of all my right hand landmarks
                            right_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in Right_hand_results]).flatten())   #Create a list with all the right hand landmarks stored in this order : x,y,z,visibility

                         if results.left_hand_landmarks:
                 
                            Left_hand_results= results.left_hand_landmarks.landmark  #Same for left hand
                            left_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in Left_hand_results]).flatten()) #Same
                       
                         if results.pose_landmarks:

                            Pose_results= results.pose_landmarks.landmark                         
                            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in Pose_results]).flatten())   
 
                         
                        
                         land=right_hand_row+left_hand_row+pose_row     #Create a unique list with the value of the landmarks of the right hand, left hand and the pose
                         msg_h.Name=class_name  #Insert my class name into the Holistic message structure                                           
                         msg_h.H_Key=land       #Insert my list of landmarks into the Holistic message structure                     
                         pub_h.publish(msg_h)   #Publish the messages 

                 
                     if (E_D_Right_Hand == "enable" and  E_D_Left_Hand=="enable" and E_D_Pose=="disable" ): # With all the following loop that will be the same structure, the only difference is the parameters
                         mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                                              mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                                              )

                         mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                                              mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                                              mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                                              )


                         msg_h = Holistic()
                         if results.right_hand_landmarks :                                  
                            
                            Right_hand_results = results.right_hand_landmarks.landmark
                            right_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in Right_hand_results]).flatten())

                         if results.left_hand_landmarks:
                 
                            Left_hand_results= results.left_hand_landmarks.landmark  
                            left_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in Left_hand_results]).flatten())
                       

                         
                         
                         land=right_hand_row+left_hand_row
                         msg_h.Name=class_name                                              
                         msg_h.H_Key=land                        
                         pub_h.publish(msg_h)
                            


                     if (E_D_Right_Hand == "enable" and  E_D_Left_Hand=="disable" and E_D_Pose=="enable" ):
                         mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                                              mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                                              )

                         mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                                              )

                         msg_h = Holistic()
                         if results.right_hand_landmarks :                                  
                            
                            Right_hand_results = results.right_hand_landmarks.landmark
                            right_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in Right_hand_results]).flatten())

                         if results.pose_landmarks:

                            Pose_results= results.pose_landmarks.landmark                          
                            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in Pose_results]).flatten())
 
                         
                        
                         land=right_hand_row+pose_row
                         msg_h.Name=class_name                                              
                         msg_h.H_Key=land                        
                         pub_h.publish(msg_h)


                     if (E_D_Right_Hand == "enable" and  E_D_Left_Hand=="disable" and E_D_Pose=="disable"):

                         mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                                              mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                                              )
                         msg_h = Holistic()
                         if results.right_hand_landmarks :                                  
                            
                            Right_hand_results = results.right_hand_landmarks.landmark
                            right_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in Right_hand_results]).flatten())

                         
                         land=right_hand_row
                         msg_h.Name=class_name                                              
                         msg_h.H_Key=land                        
                         pub_h.publish(msg_h)
                 
                     if (E_D_Right_Hand == "disable" and  E_D_Left_Hand=="enable" and E_D_Pose=="enable" ):
                         mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                                              mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                                              )
                         mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                                              )
                         msg_h = Holistic()
                         if results.left_hand_landmarks:
                 
                            Left_hand_results= results.left_hand_landmarks.landmark  
                            left_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in Left_hand_results]).flatten())
                         if results.pose_landmarks:

                            Pose_results= results.pose_landmarks.landmark                          
                            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in Pose_results]).flatten())
 

                         
                         land=left_hand_row + pose_row
                         msg_h.Name=class_name                                              
                         msg_h.H_Key=land                        
                         pub_h.publish(msg_h)

                     if (E_D_Right_Hand == "disable" and  E_D_Left_Hand=="enable" and E_D_Pose=="disable"):

                         mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                     mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                     )

                         msg_h = Holistic()
                         if results.left_hand_landmarks :                                  
                            
                            Left_hand_results = results.left_hand_landmarks.landmark
                            left_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in Left_hand_results]).flatten())

                        
                         land=left_hand_row
                         msg_h.Name=class_name                                              
                         msg_h.H_Key=land                        
                         pub_h.publish(msg_h)

                 
                     if (E_D_Right_Hand == "disable" and  E_D_Left_Hand=="disable" and E_D_Pose=="enable" ):

                         mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                                              )

                         msg_h = Holistic()
                         if results.pose_landmarks :                                  
                            
                            Pose_results = results.left_hand_landmarks.landmark
                            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in Pose_results]).flatten())

                         
                         land=pose_row
                         msg_h.Name=class_name                                              
                         msg_h.H_Key=land                        
                         pub_h.publish(msg_h)
                                 

                     if (E_D_Right_Hand == "disable" and  E_D_Left_Hand=="disable" and E_D_Pose=="disable" ):
                         print("It's better if you don't disable all of the solutions")


                    except:
                         pass
                                 
                    cv2.imshow('Raw Webcam Feed', image)    #Display the image with a screen name
                    if cv2.waitKey(10) & 0xFF == ord('q'):  #Can close the screen ont type the letter q
                        break
                        
             cap.release()  #Closing the screen
             cv2.destroyAllWindows()
 

 if (nbr_pos == 2):
     print(f"You have setup {nbr_pos} position : {name_position[0]} , {name_position[1]} ")
     name=str(name_position[0]+" "+name_position[1])
 elif (nbr_pos == 3):
     print(f"You have setup {nbr_pos} position : {name_position[0]} , {name_position[1]} , {name_position[2]} ")
     name=str(name_position[0]+" "+name_position[1]+" "+name_position[2])
 elif (nbr_pos == 4):
     print(f"You have setup {nbr_pos} position : {name_position[0]} , {name_position[1]} , {name_position[2]} , {name_position[3]} ")
     name=str(name_position[0]+" "+name_position[1]+" "+name_position[2]+" "+name_position[3])
 elif (nbr_pos == 5):
     print(f"You have setup {nbr_pos} position : {name_position[0]} , {name_position[1]} , {name_position[2]} , {name_position[3]} , {name_position[4]} ")
     name=str(name_position[0]+" "+name_position[1]+" "+name_position[2]+" "+name_position[3]+" "+name_position[4])

 name_project=open(f'/home/baheu/ws_sk_tracking/src/sk_tracking/TXT file/projectname.txt','r')
 Name_Project=name_project.read()
 name_pos=open(f'/home/baheu/ws_sk_tracking/src/sk_tracking/TXT file/Position_Name/Position_Name_{Name_Project}.txt','w')
 name_pos.write(name)
 name_pos.close()

 rate.sleep()
             
  

def param():
    #Set up some global variables to be used in different function of the node
    global E_D_Webcam
    global E_D_Right_Hand
    global E_D_Left_Hand
    global E_D_Pose


    #Print the value of the default parameters
    print("The default parameters are the following one : ")
    print("Webcam :",E_D_Webcam)
    print("Right Hand :",E_D_Right_Hand)
    print("Left Hand :",E_D_Left_Hand)
    print("Pose :",E_D_Pose)
    print("If you want to use your webcam, 0 is the good value, if you want to use an external camera write webcam after")
    #Create a loop if the user want to change some of the parameters
    choice_okay="False"
    choice=input("Do you want to change something in the setup ?")
    if (choice=="yes" or choice=="Yes" or choice=="YES" or choice=="y"):
            while(choice_okay!="GO" ):
                choice_param=input("Please write the name of the parameters that you want to disable ")
                if (choice_param=="Webcam" or choice_param=="webcam"):
                    E_D_Webcam=2

                elif (choice_param=="Right Hand" or choice_param=="right hand" or choice_param=="Right hand"):
                  
                    E_D_Right_Hand="disable"
                    print(E_D_Right_Hand)

                elif (choice_param=="Left_Hand" or choice_param=="left hand" or choice_param=="Left Hand" or choice_param=="Left hand"):
                    E_D_Left_Hand="disable"

                elif (choice_param=="Pose" or choice_param=="pose"):
                    E_D_Pose="disable"


                else :
                    print("You don't write it corectly, please retry")

                choice_okay=input("Are you done with the setup, if it's okay please write : GO , if it's not write anything ")


        
def createfiles():
        # Firstly I create some txt files to save the parameters of this session to use it in the following nodes
        
        # Write the project name on a TXT file
        File_Name=input("What is your project name ?")
        Project_name_txt=open('/home/baheu/ws_sk_tracking/src/sk_tracking/TXT file/projectname.txt','w')    
        Project_name_txt.write(File_Name)
        Project_name_txt.close()

        # Write the different solution used on a TXT file
        Solution=""
        if (E_D_Right_Hand=="enable"):
            Solution="Right"
        if (E_D_Left_Hand=="enable"):
            Solution= Solution + " Left"
        if (E_D_Pose=="enable"):
            Solution= Solution + " Pose"
        Solution_txt=open(f'/home/baheu/ws_sk_tracking/src/sk_tracking/TXT file/Solution_Name/Solution_{File_Name}.txt','w')
        Solution_txt.write(Solution)
        Solution_txt.close()
        

        # In a second time I create the structure of my CSV file
        # Depending of the parameters we don't have the same numbers of landmarks so the structure of the CSV file will change
        # This will create the first line of the CSV file with on the first column the name of the class and after the (x,y,z,v) coordinates of the first landmarks, second ...

        if (E_D_Right_Hand == "enable" and  E_D_Left_Hand=="enable" and E_D_Pose=="enable"):    
            landmarks = ['class']
            for val in range(1, 75+1):
                landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]   

            with open(f'/home/baheu/ws_sk_tracking/src/sk_tracking/CSV files/{File_Name}.csv', mode='w', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(landmarks)

        if (E_D_Right_Hand == "enable" and  E_D_Left_Hand=="enable" and E_D_Pose=="disable") :  # Same for each loop, just the number of colums will change
            landmarks = ['class']
            for val in range(1, 42+1):
                landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

            with open(f'/home/baheu/ws_sk_tracking/src/sk_tracking/CSV files/{File_Name}.csv', mode='w', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(landmarks)

        if (E_D_Right_Hand == "enable" and  E_D_Left_Hand=="disable" and E_D_Pose=="enable") or (E_D_Right_Hand == "disable" and  E_D_Left_Hand=="enable" and E_D_Pose=="enable"):
            landmarks = ['class']
            for val in range(1, 54+1):
                landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

            with open(f'/home/baheu/ws_sk_tracking/src/sk_tracking/CSV files/{File_Name}.csv', mode='w', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(landmarks)

        if (E_D_Right_Hand == "enable" and  E_D_Left_Hand=="disable" and E_D_Pose=="disable" ) or (E_D_Right_Hand == "disable" and  E_D_Left_Hand=="enable" and E_D_Pose=="disable" ) :
            landmarks = ['class']
            for val in range(1, 21+1):
                landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

            with open(f'/home/baheu/ws_sk_tracking/src/sk_tracking/CSV files/{File_Name}.csv', mode='w', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(landmarks)

        if (E_D_Right_Hand == "disable" and  E_D_Left_Hand=="disable" and E_D_Pose=="enable" ) :
            landmarks = ['class']
            for val in range(1, 33+1):
                landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

            with open(f'/home/baheu/ws_sk_tracking/src/sk_tracking/CSV files/{File_Name}.csv', mode='w', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(landmarks)
        


if __name__ == '__main__':
    try:
            
       
        param()   

        createfiles()

        talker()
        
    except rospy.ROSInterruptException:
        pass

