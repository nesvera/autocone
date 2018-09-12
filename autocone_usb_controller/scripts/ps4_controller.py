#!/usr/bin/env python

import rospy
import pygame

from ackermann_msgs.msg import AckermannDrive

class PS4Controller:
    def __init__(self):
        rospy.init_node("ps4_controller")
        
        print(pygame.__version__)
        pygame.joystick.init() # Initialize pygame joysticks module
        n_joysticks = pygame.joystick.get_count() # Get number of connected joysticks
        print("Found {} joysticks. I'm gonna use the first one.".format(n_joysticks))
        self.joy = pygame.joystick.Joystick(0) # Select the first joystick
        self.joy.init() # Initialize joystick
        
    def run(self, hz):
        cmd_publisher = rospy.Publisher("/ackermann_cmd", AckermannDrive, queue_size=1)
        rate = rospy.Rate(hz) # 30 Hz
        screen = pygame.display.set_mode((1, 1))

        # Mapping for PS4 Controller
        # Axis 0 - Left Stick X
        # Axis 1 - Left Stick Y
        # Axis 2 - Left "Trigger"
        # Axis 3 - Right Stick X
        # Axis 4 - Right Stick Y
        # Axis 5 - Right "Trigger"

        while not rospy.is_shutdown():
            # This has to be called at the start of each loop in order for pygame
            # to update the axis status
            pygame.event.get() 

            # Create a new "empty" message
            msg = AckermannDrive()
            print(msg.steering_angle)
            print(msg.speed)

            msg.steering_angle = -self.joy.get_axis(0)            

            l_t = self.joy.get_axis(2)
            r_t = self.joy.get_axis(5)
            msg.speed = 2.0 * ((r_t + 1.0) - (l_t  + 1.0))

            print("steering: {} speed: {}".format(msg.steering_angle, msg.speed))
            
            cmd_publisher.publish(msg)

            rate.sleep()

if __name__ == "__main__":
    ps4_controller = PS4Controller()
    ps4_controller.run(30)
