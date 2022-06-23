#!/usr/bin/env python3
import rospy
from sk_tracking.msg import Holistic
import csv

def H_callback(data):
     
     var = []       
     rh=list(data.H_Key)        #Take the value stored in the custom message with the key "H_Key" which means my list with all my landmarks
     name=data.Name             #Take the value of the name of the position
     var.append(name)           #Add the name to the list 
     var.extend(rh)             #Add the landmarks to the list after the name
    
     
     with open(f'/home/baheu/ws_sk_tracking/src/sk_tracking/CSV files/{File_Name}.csv', mode='a', newline='') as f:         #Write the list into the CSV file
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(var)
   
def listener():

    rospy.init_node('listener', anonymous=True)     #Here I initialize my node 
    rospy.Subscriber("H_Topic", Holistic, H_callback)   #I subscribe to the topic "H_Topic", the same that I publishing datas on the publisher node

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    Isitgood=input("Write GO when you have set the project name : ")        #I use this to really start the program after the creation of the txt file called "projectname" in the publisher node
    if Isitgood=="go" or "GO" or "Go":
        Fn=open('/home/baheu/ws_sk_tracking/src/sk_tracking/TXT file/projectname.txt','r')  #Here I open the TXT file "projectname" and read it
        File_Name=Fn.read()     #I store the reading value in a variable to use it in my callback function to call the good csv file

        listener()
