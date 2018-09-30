#!/usr/bin/env python

import rospy
from ackermann_msgs.msg import AckermannDrive
from gazebo_msgs.msg import ContactsState

import pygame

class Drive:

    def __init__(self):
        self.ackermann_cmd = AckermannDrive()
        self.ackermann_cmd.speed = 1
        self.ackermann_cmd.steering_angle = 0.0

        self.ackermann_pub = rospy.Publisher('/ackermann_cmd', AckermannDrive, queue_size=1)

        self.rate = rospy.Rate(30)
        self.screen = pygame.display.set_mode((1, 1))
        

    def start(self):
        pygame.init()
        while not rospy.is_shutdown():
            print(self.ackermann_cmd)

            pygame.event.get()

            keys_pressed = pygame.key.get_pressed()
            if keys_pressed[pygame.K_LEFT]:
                self.ackermann_cmd.steering_angle = 1.0
            elif keys_pressed[pygame.K_RIGHT]:
                self.ackermann_cmd.steering_angle = -1.0

            self.ackermann_pub.publish(self.ackermann_cmd)
            self.ackermann_cmd.steering_angle = 0.0

            self.rate.sleep()
                
if __name__ == "__main__":
    rospy.init_node('steer_arrows', anonymous=True)

    d = Drive()

    # Collect events until released
    while True:
        d.start()
        