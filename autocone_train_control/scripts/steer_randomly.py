#!/usr/bin/env python

import rospy

from ackermann_msgs.msg import AckermannDrive
from rosgraph_msgs.msg import Clock

from gazebo_msgs.msg import ContactsState

import time
import random

max_step = 0.1

class Drive:

    def __init__(self):

        self.ackermann_cmd = AckermannDrive()
        self.ackermann_cmd.speed = 1
        self.ackermann_cmd.steering_angle = 0.0
    
        self.ackermann_pub = rospy.Publisher('/ackermann_cmd', AckermannDrive, queue_size=1)
        self.time_sub = rospy.Subscriber('/clock', Clock, self._clock_callback)
        self.bumper_sub = rospy.Subscriber('/bumper_sensor', ContactsState, self._bumper_callback)

        self.drive_cur_time = 0
        self.drive_init_time = 0

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

            self.ackermann_cmd.steering_angle = random.uniform(-0.5, 0.5)
            #self.ackermann_cmd.steering_angle = random.gauss(0, 0.3)

            self.ackermann_pub.publish(self.ackermann_cmd)

            self.rate.sleep()

    def _clock_callback(self, data):
        self.drive_cur_time = data.clock.secs*1000 + (data.clock.nsecs/1000000.0)

    # callback listening for collision
    def _bumper_callback(self, data):
        
        # Check if hit something
        states = data.states

        if len(states) > 0:
            print(self.drive_cur_time - self.drive_init_time)
            self.drive_init_time = self.drive_cur_time

            time.sleep(0.5)



if __name__ == "__main__":

    rospy.init_node('steer_randomly', anonymous=True)

    d = Drive()

    while True:
        d.update()