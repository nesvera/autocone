#!/usr/bin/env python

import rospy

from sensor_msgs.msg import Joy
from ackermann_msgs.msg import AckermannDrive

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import sys
import argparse
from sklearn.cluster import DBSCAN
from sklearn import linear_model
import os
import pickle
import time
import signal

sys.path.append("../../")
from autocone_colin_utils.vision import Vision
from autocone_colin_utils.camera_calibrator import Parameters
from autocone_colin_utils.pid import PID

import collections
import cv2

DEBUG = True

fixed_speed = False
fixed_speed_value = 0
max_speed = 0
max_steering = 100

ackermann_cmd = AckermannDrive()
ackermann_pub = None

steer_list = collections.deque(maxlen=5)
frame_median = collections.deque(maxlen=5)

autonomous_mode = False

def joy_callback(data):
    global autonomous_mode

    axes = data.axes
    buttons = data.buttons

    left_trigger = data.axes[2]
    right_trigger = data.axes[5]
    left_x_stick = data.axes[0]

    forward = (-right_trigger+1)/2.
    backward = (-left_trigger+1)/2.

    speed = forward*max_speed
    ackermann_cmd.speed = float(speed)

    # start button
    if buttons[7] == 1:
        autonomous_mode = True

    # back button
    if buttons[6] == 1:
        autonomous_mode = False

def routine():
    global frame

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    start_time = time.time()
    while True:
        
        _, frame = cap.read()
        
        # Image processing
        center_offset, upper_point, angle_to_upper = vision.find_error(frame)

        steer = pid.update(0, angle_to_upper)
        
        # limit of steering
        if steer > 100:
            steer = 100
        elif steer < -100:
            steer = -100

        # Set commands to the car
        ackermann_cmd.steering_angle = int(steer)
        
        speed_angle = abs(angle_to_upper)
        if speed_angle > 15:
            #ackermann_cmd.speed = float(50)
            print("devagar carai")
        else:
            #ackermann_cmd.speed = float(60)
            print("acelera") 

        # Send command to the car
        if autonomous_mode == True:
            ackermann_pub.publish(ackermann_cmd)
            print("ta andando")

        # FPS
        frame_median.append(time.time()-start_time)
        start_time = time.time()
        fps = 0
        for i in frame_median:
            fps += i
        fps /= 5.
        fps =  1/fps

        print(int(fps), int(center_offset), int(upper_point), int(angle_to_upper), int(steer))

        #if DEBUG:
        draw()

def draw():
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

def exit_handler(signal, frame):
	exit(0)

if __name__ == "__main__":
    global parameters

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fixed_speed', action='store', dest='fix_speed',
                        default=0, required=False,
                        help="Value of fixed velocity.")
    parser.add_argument('-m ', '--max_speed', action='store', dest='max_speed',
                        default=80, required=False,
                        help="Limit of speed.")
    parser.add_argument('-p', '--parameters', action='store', dest='parameters',
                        required=True, help="Path to the parameters of the camera")
    parser.add_argument('-d', '--debug', action='store', dest='debug',
                        required=False, help="Enable debug mode")

    arguments = parser.parse_args(rospy.myargv()[1:])
    
    # car running on fixed speed
    fixed_speed_value = float(arguments.fix_speed)
    max_speed = float(arguments.max_speed)
    parameters_path = arguments.parameters
    DEBUG = arguments.debug

    # Load old parameters if file exist
    if os.path.exists(parameters_path):
        filehandler = open(parameters_path, 'r')
        parameters_dict = pickle.load(filehandler)
        filehandler.close()	

        vision = Vision()
        vision.set_parameters(parameters_dict)

    else:
        print("Problems with parameters file!")
        exit(0)

    # Steering control
    pid = PID(3, 0.1, 0, 50)

    rospy.init_node('line_follower', anonymous=True)
    rate = rospy.Rate(30)    

    # Subscribe to controller topic
    rospy.Subscriber('/joy', Joy, joy_callback)

    # Subscribe to camera topic
    #rospy.Subscriber('/usb_cam/image_raw', Image, image_callback)
    bridge = CvBridge()

    # Subscribe to publish on the car topic
    ackermann_pub = rospy.Publisher('/ackermann_cmd', AckermannDrive, queue_size=30)

    # set function to save when exit de script
    signal.signal(signal.SIGINT, exit_handler)

    routine()
