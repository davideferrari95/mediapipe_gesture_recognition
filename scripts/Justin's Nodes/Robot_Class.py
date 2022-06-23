#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist



class RobotControl():

    def __init__(self):
        rospy.init_node('robot_control_node', anonymous=True)
        self.vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.summit_vel_publisher = rospy.Publisher('/summit_xl_control/cmd_vel', Twist, queue_size=1)
        
        self.cmd = Twist()
       
        self.ctrl_c = False
        self.rate = rospy.Rate(10)
     

    def publish_once_in_cmd_vel(self):
        """
        This is because publishing in topics sometimes fails the first time you publish.
        In continuos publishing systems there is no big deal but in systems that publish only
        once it IS very important.
        """
        while not self.ctrl_c:
            connections = self.vel_publisher.get_num_connections()
            summit_connections = self.summit_vel_publisher.get_num_connections()
            if connections > 0 or summit_connections > 0:
                self.vel_publisher.publish(self.cmd)
                self.summit_vel_publisher.publish(self.cmd)
                #rospy.loginfo("Cmd Published")
                break
            else:
                self.rate.sleep()


    def move_straight(self):
        
        # Initilize velocities
        self.cmd.linear.x = 0.1
        self.cmd.linear.y = 0
        self.cmd.linear.z = 0
        self.cmd.angular.x = 0
        self.cmd.angular.y = 0
        self.cmd.angular.z = 0

        # Publish the velocity
        self.publish_once_in_cmd_vel()

    def move_back(self):

        # Initilize velocities
        self.cmd.linear.x = -0.1
        self.cmd.linear.y = 0
        self.cmd.linear.z = 0
        self.cmd.angular.x = 0
        self.cmd.angular.y = 0
        self.cmd.angular.z = 0

        # Publish the velocity
        self.publish_once_in_cmd_vel()

    def turn_left(self):
        self.cmd.linear.x = 0
        self.cmd.linear.y = 0
        self.cmd.linear.z = 0
        self.cmd.angular.x = 0
        self.cmd.angular.y = 0
        self.cmd.angular.z = 0.2

        # Publish the velocity
        self.publish_once_in_cmd_vel()

    def turn_right(self):
        self.cmd.linear.x = 0
        self.cmd.linear.y = 0
        self.cmd.linear.z = 0
        self.cmd.angular.x = 0
        self.cmd.angular.y = 0
        self.cmd.angular.z = -0.2

        # Publish the velocity
        self.publish_once_in_cmd_vel()
    
    def stop(self):
        self.cmd.linear.x = 0
        self.cmd.linear.y = 0
        self.cmd.linear.z = 0
        self.cmd.angular.x = 0
        self.cmd.angular.y = 0
        self.cmd.angular.z = 0

        # Publish the velocity
        self.publish_once_in_cmd_vel()



