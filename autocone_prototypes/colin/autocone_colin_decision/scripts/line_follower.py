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

sys.path.append("../../")
from autocone_colin_utils.vision import Vision
from autocone_colin_utils.camera_calibrator import Parameters
from autocone_colin_utils.pid import PID

fixed_speed = False
fixed_speed_value = 0
max_speed = 0
max_steering = 100

ackermann_cmd = AckermannDrive()
ackermann_pub = None

def joy_callback(data):
    axes = data.axes
    buttons = data.buttons

    left_trigger = data.axes[2]
    right_trigger = data.axes[5]
    left_x_stick = data.axes[0]

    forward = (-right_trigger+1)/2.
    backward = (-left_trigger+1)/2.

    speed = forward*max_speed
    ackermann_cmd.speed = float(speed)

def image_callback(data):
    
    # Convert image from ROS format to cv2
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        #print(e)
        pass

    
    # Image processing
    
    error = float(vision.find_error(cv_image))
    error = (error/90.)*0.14
    error = pid.update(0, error)


    if error > 0.14:
        error = 0.14
    elif error < -0.14:
        error = -0.14

    print(error)

    ackermann_cmd.steering_angle = float(error)

    # Send command to the car
    ackermann_cmd.speed = float(3)
    ackermann_pub.publish(ackermann_cmd)


if __name__ == "__main__":
    global parameters

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fixed_speed', action='store', dest='fix_speed',
                        default=0, required=False,
                        help="Value of fixed velocity.")
    parser.add_argument('-m ', '--max_speed', action='store', dest='max_speed',
                        default=50, required=False,
                        help="Limit of speed.")
    parser.add_argument('-p', '-parameters', action='store', dest='parameters',
                        required=True, help="Path to the parameters of the camera")

    arguments = parser.parse_args(rospy.myargv()[1:])
    
    # car running on fixed speed
    fixed_speed_value = float(arguments.fix_speed)
    max_speed = float(arguments.max_speed)
    parameters_path = arguments.parameters

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
    pid = PID(2.5, 0.05, 0, 1.4)

    rospy.init_node('line_follower', anonymous=True)
    rate = rospy.Rate(30)    

    # Subscribe to controller topic
    rospy.Subscriber('/joy', Joy, joy_callback)

    # Subscribe to camera topic
    rospy.Subscriber('/usb_cam/image_raw', Image, image_callback)
    bridge = CvBridge()

    # Subscribe to publish on the car topic
    ackermann_pub = rospy.Publisher('/ackermann_cmd', AckermannDrive, queue_size=1)
    
    rospy.spin()
