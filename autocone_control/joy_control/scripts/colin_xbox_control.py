#!/usr/bin/env python

import rospy

from sensor_msgs.msg import Joy
from ackermann_msgs.msg import AckermannDrive


import sys
import argparse

fixed_speed = False
fixed_speed_value = 0
max_speed = 80
controller = 'xbox_360'

ackermann_cmd = AckermannDrive()
ackermann_pub = None

new_data = False

rate = None


def joy_callback(data):
    global new_data 

    axes = data.axes
    buttons = data.buttons

    left_trigger = data.axes[2]
    right_trigger = data.axes[5]
    left_x_stick = data.axes[0]

    forward = (-right_trigger+1)/2.
    backward = (-left_trigger+1)/2.

    speed = forward
    steering = left_x_stick

    ackermann_cmd.speed = speed
    ackermann_cmd.steering_angle = steering

    new_data = True


def routine():
    global new_data

    while not rospy.is_shutdown():

        if fixed_speed:
            ackermann_cmd.speed = fixed_speed_value

        ackermann_pub.publish(ackermann_cmd)

        rate.sleep()
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fixed_speed', action='store', dest='fix_speed',
                        default=0, required=False,
                        help="Value of fixed velocity.")
    parser.add_argument('-m ', '--max_speed', action='store', dest='max_speed',
                        default=0, required=False,
                        help="Limit of speed.")
    parser.add_argument('-c', '--controller', action='store', dest='controller',
                        default='ps4', required=False,
                        help="Controller: ps4, xbox or other")

    arguments = parser.parse_args(rospy.myargv()[1:])

    # controller type
    controller = arguments.controller
    
    # car running on fixed speed
    fixed_speed_value = float(arguments.fix_speed)
    max_speed = float(arguments.max_speed)

    if fixed_speed_value > 0:
        fixed_speed = True

    rospy.init_node('joy_sim_control', anonymous=True)

    # Subscribe to the topic that contains the controller keys
    rospy.Subscriber('/joy', Joy, joy_callback)    

    rate = rospy.Rate(30)    

    ackermann_pub = rospy.Publisher('/ackermann_cmd', AckermannDrive, queue_size=1)

    routine()

