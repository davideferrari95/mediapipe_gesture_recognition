#!/usr/bin/env python3

import rospy
#from std_msgs.msg import String



class culo:

    def __init__(self):

        rospy.init_node('mediapipe_gesture_recognition_training_node', anonymous=True)
        #rospy.Subscriber("/gesture_type", String, self.GestureCallback)

        #self.gesturearrived = False
        self.labels = [
        "Swiping Left",
        "Swiping Right",
        "Swiping Down",
        "Swiping Up",
        "Pushing Hand Away",
        "Pulling Hand In",
        "Sliding Two Fingers Left",
        "Sliding Two Fingers Right",
        "Sliding Two Fingers Down",
        "Sliding Two Fingers Up",
        "Pushing Two Fingers Away",
        "Pulling Two Fingers In",
        "Rolling Hand Forward",
        "Rolling Hand Backward",
        "Turning Hand Clockwise",
        "Turning Hand Counterclockwise",
        "Zooming In With Full Hand",
        "Zooming Out With Full Hand",
        "Zooming In With Two Fingers",
        "Zooming Out With Two Fingers",
        "Thumb Up",
        "Thumb Down",
        "Shaking Hand",
        "Stop Sign",
        "Drumming Fingers",
        "No gesture",
        "Doing other things"
        ]
        self.gesturelabeled = False

    # def GestureCallback(self,data):

    #     with open("/home/alberto/catkin_ws/src/mediapipe_gesture_recognition/scripts/gesture.txt", "w") as f:
    #         f.write(data.data)

    #     self.gesturearrived = True
    
    def printaInput(self):

        #if self.gesturearrived ==True:
            with open("/home/alberto/catkin_ws/src/mediapipe_gesture_recognition/scripts/gesture.txt", "r") as f:
                self.gesture = f.readline()
                #self.gesturearrived = False

                if self.gesture in self.labels:


                    print(self.gesture)


if __name__ == '__main__':

    c = culo()

    while not rospy.is_shutdown():
            
        c.printaInput() 