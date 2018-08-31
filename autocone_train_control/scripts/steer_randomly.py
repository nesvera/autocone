#!/usr/bin/env python

import rospy

from ackermann_msgs.msg import AckermannDrive

import random

max_step = 0.1

class Drive:

    def __init__(self):

        self.ackermann_cmd = AckermannDrive()
        self.ackermann_cmd.speed = 0.1
        self.ackermann_cmd.steering_angle = 0.0
    
        self.ackermann_pub = rospy.Publisher('/ackermann_cmd', AckermannDrive, queue_size=1)

        self.rate = rospy.Rate(10)

    def update(self):

        while not rospy.is_shutdown():

            angle_inc_step = random.uniform(-1.0, 1.0)

            self.ackermann_cmd.steering_angle  += angle_inc_step
            #self.ackermann_cmd.steering_angle = random.randint(-100, 100)/100.0
            #self.ackermann_cmd.steering_angle = random.uniform(-1.0, 1.0)

            if self.ackermann_cmd.steering_angle > 1.0:
                self.ackermann_cmd.steering_angle = 1.0
            
            elif self.ackermann_cmd.steering_angle < -1.0:
                self.ackermann_cmd.steering_angle = -1.0

            self.ackermann_pub.publish(self.ackermann_cmd)

            self.rate.sleep()


if __name__ == "__main__":

    rospy.init_node('steer_randomly', anonymous=True)

    d = Drive()

    while True:
        d.update()