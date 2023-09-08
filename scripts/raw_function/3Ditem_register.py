#!/usr/bin/env python3

import rospy, numpy as np
from mediapipe_gesture_recognition.msg import Hand, Pose
# from Utils import Start_countdown
import csv

item_coordinates_file = r'/home/davide/ROS/ICRA_2023/src/Mediapipe/mediapipe_gesture_recognition - ICRA/scripts/raw_function/item_coordinates.csv'
# Countdown Function
def countdown(num_of_secs):

  import rospy

  print("\nAcquisition Starts in:")

  # Wait Until 0 Seconds Remaining
  while (not rospy.is_shutdown() and num_of_secs != 0):
    m, s = divmod(num_of_secs, 60)
    min_sec_format = '{:02d}:{:02d}'.format(m, s)
    print(min_sec_format)
    rospy.sleep(1)
    num_of_secs -= 1

  print("\nSTART\n")

class ItemRegister:

    def __init__(self):

        # Inizializzazione del nodo
        rospy.init_node('pointer_subscriber', anonymous=True)

        # Inizializzazione dei topic
        rospy.Subscriber("/mediapipe_gesture_recognition/right_hand", Hand, self.RightHandCallback)
        rospy.Subscriber("/mediapipe_gesture_recognition/left_hand", Hand, self.LeftHandCallback)
        rospy.Subscriber("/mediapipe_gesture_recognition/pose", Pose, self.PoseCallback)

        self.right_msg = None
        self.left_msg  = None
        self.pose_msg  = None

        self.z = -1

    # Callback Functions
    def RightHandCallback(self, data): self.right_msg: Hand() = data
    def LeftHandCallback(self, data):  self.left_msg:  Hand() = data
    def PoseCallback(self, data):      self.pose_msg:  Pose() = data

    def takeCoordinates(self):

        # Definizione delle coordinate

        x1 = float(self.pose_msg.keypoints[15].x)
        y1 = float(self.pose_msg.keypoints[15].y)
        z1 = float(self.pose_msg.keypoints[15].depth)
        x2 = float(self.pose_msg.keypoints[13].x)
        y2 = float(self.pose_msg.keypoints[13].y)
        z2 = float(self.pose_msg.keypoints[13].depth)

        # Definizione dei punti
        p1 = np.array([x1, y1, z1])
        p2 = np.array([x2, y2, z2])

        # Find the point
        p3 = self.find_point_on_line(p1, p2)

        return p3

    def find_line_equation(self, p1, p2):

        # Trova l'equazione della retta che passa per i punti p1 e p2
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1
        m = dz / ((dx ** 2 + dy ** 2 + dz ** 2) ** 0.5)
        q = z1 - m * ((dx ** 2 + dy ** 2) ** 0.5)

        return m, q

    def find_point_on_line(self, p1, p2):

        #Trova il punto sulla retta che passa per i punti p1 e p2 con la coordinata z data

        m, q = self.find_line_equation(p1, p2)

        x1, y1, z1 = p1

        x = ((self.z - q) / m) * ((p2[0] - p1[0]) / ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)**0.5) + x1
        y = ((self.z - q) / m) * ((p2[1] - p1[1]) / ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)**0.5) + y1

        p3 = [x, y, self.z]

        return p3

    def saveCoordinates(self, label, p3):

        coordinates = []
        item_found = False

        with open(item_coordinates_file, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')

            for row in reader:

                if row[0] == label:

                    row[1:4] = [str(p3[0]), str(p3[1]), str(p3[2])]
                    coordinates.append(row)
                    print("The {} are updated in x:{}, y:{}, z:{}".format(label, row[1], row[2], row[3]))

                    item_found = True
                else:
                    coordinates.append(row)

            if not item_found:

                new_row = [label, str(p3[0]), str(p3[1]), str(p3[2])]
                coordinates.append(new_row)
                print("The {} are now in x:{}, y:{}, z:{}".format(label, new_row[1], new_row[2], new_row[3]))

        with open(item_coordinates_file, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')

            for row in coordinates:
                writer.writerow(row)

if __name__ == '__main__':

    R = ItemRegister()

    name_label = input("Inserisci il nome dell'oggetto/area di cui vuoi registrare le coordinate: ")

    # Start_countdown(8)
    # rospy.sleep(5)
    countdown(5)

    item_coordinates = R.takeCoordinates()

    R.saveCoordinates(name_label, item_coordinates)
