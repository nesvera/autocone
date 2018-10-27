#!/usr/bin/env python

import rospy

from sensor_msgs.msg import Joy
from ackermann_msgs.msg import AckermannDrive

import sys
import argparse
from sklearn.cluster import DBSCAN
from sklearn import linear_model

fixed_speed = False
fixed_speed_value = 0
max_speed = 0
max_steering = 100

ackermann_cmd = AckermannDrive()
ackermann_pub = None

import cv2
import numpy as np
import time
import collections
import time

		
def joy_callback(data):
    global new_data 

    axes = data.axes
    buttons = data.buttons

    left_trigger = data.axes[2]
    right_trigger = data.axes[5]
    left_x_stick = data.axes[0]

    forward = (-right_trigger+1)/2.
    backward = (-left_trigger+1)/2.

    speed = forward*max_speed
    #steering = left_x_stick*max_steering

    ackermann_cmd.speed = float(speed)

    new_data = True

def routine():
    global teste
    
    steer_list = collections.deque(maxlen=10)

    while not rospy.is_shutdown():

        #chama funcao do catani
        #ackermann_cmd.steering_angle = float(steering)

        erro = teste.find_error()
	#print "recebeu"
        #print("erro = "+str(erro))

        steer = 100*(-erro/30.)
        steer_list.append(steer)
	
        avg_steer = 0
        for i in steer_list:
            avg_steer += i 

        avg_steer = avg_steer / 10.

        ackermann_cmd.steering_angle = int(steer)
	
        ackermann_pub.publish(ackermann_cmd)

        rate.sleep()
                

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fixed_speed', action='store', dest='fix_speed',
                        default=0, required=False,
                        help="Value of fixed velocity.")
    parser.add_argument('-m ', '--max_speed', action='store', dest='max_speed',
                        default=50, required=False,
                        help="Limit of speed.")

    arguments = parser.parse_args(rospy.myargv()[1:])
    
    # car running on fixed speed
    fixed_speed_value = float(arguments.fix_speed)
    max_speed = float(arguments.max_speed)

    if fixed_speed_value > 0:
        fixed_speed = True

    rospy.init_node('line_follower', anonymous=True)

    rate = rospy.Rate(30)    

    rospy.Subscriber('/joy', Joy, joy_callback) 
    ackermann_pub = rospy.Publisher('/ackermann_cmd', AckermannDrive, queue_size=1)
    teste = vision()

    routine()
