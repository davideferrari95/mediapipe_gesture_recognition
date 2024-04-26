#!/usr/bin/env python3

import rospy, csv, numpy as np
from typing import List, Tuple
from mediapipe_gesture_recognition.msg import Hand, Pose

class CoordinateRecorder:

    """ Coordinate Recorder Class for Point-At Gesture """

    def __init__(self):

        # Node Initialization
        rospy.init_node('item_register', anonymous=True)

        # Subscribers Initialization
        rospy.Subscriber('/mediapipe_gesture_recognition/right_hand', Hand, self.RightHandCallback)
        rospy.Subscriber('/mediapipe_gesture_recognition/left_hand', Hand, self.LeftHandCallback)
        rospy.Subscriber('/mediapipe_gesture_recognition/pose', Pose, self.PoseCallback)

        # Variables Initialization
        self.right_msg, self.left_msg, self.pose_msg  = Hand(), Hand(), Pose()
        self.z = -1

        # Input Request
        self.name_label = input('Insert the name of the object/area you want to register the coordinates: ')

    # Callback Functions
    def RightHandCallback(self, data:Hand): self.right_msg: Hand = data
    def LeftHandCallback(self, data:Hand):  self.left_msg:  Hand = data
    def PoseCallback(self, data:Pose):      self.pose_msg:  Pose = data

    def countdown(self, num_of_secs:int):

        """ Countdown Function """

        print('\nAcquisition Starts in:')

        # Wait Until 0 Seconds Remaining
        while (not rospy.is_shutdown() and num_of_secs != 0):

            # Print the Remaining Time
            print('{:02d}:{:02d}'.format(*divmod(num_of_secs, 60)))

            # Wait 1 Second
            rospy.sleep(1)
            num_of_secs -= 1

        print('\nSTART\n')

    def get_coordinates(self) -> List[float]:

        """ Get the Coordinates of the Point-At Gesture """

        # Coordinate Definition
        x1 = float(self.pose_msg.keypoints[15].x)
        y1 = float(self.pose_msg.keypoints[15].y)
        z1 = float(self.pose_msg.keypoints[15].depth)
        x2 = float(self.pose_msg.keypoints[13].x)
        y2 = float(self.pose_msg.keypoints[13].y)
        z2 = float(self.pose_msg.keypoints[13].depth)

        # Point Definition
        p1 = np.array([x1, y1, z1])
        p2 = np.array([x2, y2, z2])

        # Find the 3rd Point
        p3 = self.find_point_on_line(p1, p2)

        return p3

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

    def save_coordinates(self, label:str, p3:List[float]):

        """ Save the Coordinates of the Point-At Gesture """

        # Initialize
        coordinates = []
        item_found = False

        # Read the CSV File
        with open('item_coordinates.csv', 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')

            # Loop Over the Rows
            for row in reader:

                # Check if the Label is Already in the CSV File
                if row[0] == label:

                    # Update the Coordinates
                    row[1:4] = [str(p3[0]), str(p3[1]), str(p3[2])]
                    coordinates.append(row)
                    print("The {} are updated in x:{}, y:{}, z:{}".format(label, row[1], row[2], row[3]))

                    item_found = True

                # Append the Row
                else: coordinates.append(row)

            if not item_found:

                # Append the New Row
                new_row = [label, str(p3[0]), str(p3[1]), str(p3[2])]
                coordinates.append(new_row)
                print("The {} are now in x:{}, y:{}, z:{}".format(label, new_row[1], new_row[2], new_row[3]))

        # Write the CSV File
        with open('item_coordinates.csv', 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')

            # Loop Over the Rows -> Write the Rows
            for row in coordinates: writer.writerow(row)

if __name__ == '__main__':

    # Coordinate Recorder Initialization
    CR = CoordinateRecorder()

    # Countdown
    CR.countdown(3)

    # Take and Save Coordinates
    CR.save_coordinates(CR.name_label, CR.get_coordinates)
