#!/usr/bin/env python3

import rospy, numpy as np
from typing import List, Tuple
from std_msgs.msg import Int32MultiArray
from mediapipe_gesture_recognition.msg import Hand, Pose

# Define Left and Right Hands
LEFT, RIGHT = 0,1

# Areas Coordinates Definition
AREA_1 = np.array([-1, -0.5, -2])
AREA_2 = np.array([3, 0.5, -2])

class CoordinatePointer:

    def __init__(self):

        # Node Initialization
        rospy.init_node('coordinate_pointer', anonymous=True)

        # Subscribers Initialization
        rospy.Subscriber('/mediapipe_gesture_recognition/right_hand', Hand, self.RightHandCallback)
        rospy.Subscriber('/mediapipe_gesture_recognition/left_hand', Hand, self.LeftHandCallback)
        rospy.Subscriber('/mediapipe_gesture_recognition/pose', Pose, self.PoseCallback)

        # Publisher Initialization
        self.area_pub = rospy.Publisher('area', Int32MultiArray, queue_size=1)

        # TODO: publish all the items and areas on fusion node

        # Variables Initialization
        self.right_msg, self.left_msg, self.pose_msg = Hand(), Hand(), Pose()
        self.area_threshold, self.item_threshold = 1.8, 0.2
        self.z = -2

    # Callback Functions
    def RightHandCallback(self, data:Hand): self.right_msg: Hand = data
    def LeftHandCallback(self, data:Hand):  self.left_msg:  Hand = data
    def PoseCallback(self, data:Pose):      self.pose_msg:  Pose = data

    def find_line_equation(self, p1:List[float], p2:List[float]) -> Tuple[float]:

        """ Find the Equation of the Line Passing Through p1 and p2 """

        # Unpack the Coordinates        
        x1, y1, z1 = p1
        x2, y2, z2 = p2

        # Calculate the Differences
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1

        # Equation of the Line: z = m * sqrt((x2 - x1)^2 + (y2 - y1)^2) + q
        m = dz / ((dx ** 2 + dy ** 2 + dz ** 2) ** 0.5)
        q = z1 - m * ((dx ** 2 + dy ** 2) ** 0.5)

        return m, q

    def find_point_on_line(self, p1:List[float], p2:List[float]) -> List[float]:

        """ Find the Point on the Line Passing Through p1 and p2 with the Given z Coordinate """

        # Find the Equation of the Line
        m, q = self.find_line_equation(p1, p2)

        # Unpack the Coordinates
        x1, y1, _ = p1

        # Calculate the x and y Coordinates
        x = ((self.z - q) / m) * ((p2[0] - p1[0]) / ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)**0.5) + x1
        y = ((self.z - q) / m) * ((p2[1] - p1[1]) / ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)**0.5) + y1

        # Point Definition p3 = [x, y, z]
        return [x, y, self.z]

    def create_3DLine(self, hand:bool):

        """ Create the 3D Line """

        if not self.pose_msg == None:

            # Coordinate Definition
            x1 = float(self.pose_msg.keypoints[15].x)     if hand == RIGHT else float(self.pose_msg.keypoints[16].x)
            y1 = float(self.pose_msg.keypoints[15].y)     if hand == RIGHT else float(self.pose_msg.keypoints[16].y)
            z1 = float(self.pose_msg.keypoints[15].depth) if hand == RIGHT else float(self.pose_msg.keypoints[16].depth)
            x2 = float(self.pose_msg.keypoints[13].x)     if hand == RIGHT else float(self.pose_msg.keypoints[14].x)
            y2 = float(self.pose_msg.keypoints[13].y)     if hand == RIGHT else float(self.pose_msg.keypoints[14].y)
            z2 = float(self.pose_msg.keypoints[13].depth) if hand == RIGHT else float(self.pose_msg.keypoints[14].depth)

            # Point Definition
            p1 = np.array([x1, y1, z1])
            p2 = np.array([x2, y2, z2])

            # Find the 3rd Point
            p3 = self.find_point_on_line(p1, p2)

            # Compute the Distance from the Areas
            distance_from_area_1 = np.linalg.norm(np.cross(p2-p1, p1-AREA_1)) / np.linalg.norm(p2-p1)
            distance_from_area_2 = np.linalg.norm(np.cross(p2-p1, p2-AREA_2)) / np.linalg.norm(p2-p1)

            print('\n\n\n\n\n')
            print(f'Distance Area 1 - {"Right" if hand == RIGHT else "Left"}: {distance_from_area_1}')
            print(f'Distance Area 2 - {"Right" if hand == RIGHT else "Left"}: {distance_from_area_2}')
            print(f'Point 3 Coordinates - {"Right" if hand == RIGHT else "Left"}: {p3}')
            print("\n\n\n\n\n")

            # Initialize the ROS Message
            area_msg = Int32MultiArray()

            # TODO: Print if the Point Belongs to the Neighborhood of the Line
            if distance_from_area_1 <= self.area_threshold:
                print(f'You are Pointing Area 1 with the {"Right" if hand == RIGHT else "Left"} Hand')
                area_msg.data = [1]
                self.area_pub.publish(area_msg)
                return area_msg.data

            if distance_from_area_2 <= self.area_threshold:
                print(f'You are Pointing Area 2 with the {"Right" if hand == RIGHT else "Left"} Hand')
                area_msg.data = [2]
                self.area_pub.publish(area_msg)
                return area_msg.data

if __name__ == '__main__':

    CP = CoordinatePointer()

    while not rospy.is_shutdown():

        rospy.sleep(0.25)

        # Print and Publish the Detected Areas
        print(CP.create_3DLine(LEFT), CP.create_3DLine(RIGHT))
