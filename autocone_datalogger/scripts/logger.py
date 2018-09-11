#!/usr/bin/env python
import rospy
import rospkg

from gazebo_msgs.msg import (
    ContactsState,
)

from sensor_msgs.msg import (
    Image,
)

from ackermann_msgs.msg import AckermannDrive

from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np

# Instantiate CvBridge
bridge = CvBridge()

DEBUG = False
DEBUG_PLOT = True

class Datalogger:

    def __init__(self):
        rospy.init_node("logger", anonymous=True)

        self.rate = rospy.Rate(30)

        rospy.Subscriber("/camera/image_raw", Image, self._image_calback, queue_size=1) 
        rospy.Subscriber("/bumper_sensor", ContactsState, self._bumper_callback, queue_size=1)
        rospy.Subscriber('/ackermann_cmd', AckermannDrive, self._car_control_callback, queue_size=1)

        self.image_width = 1280
        self.image_height = 960

        self.header = None
        self.camera_image = np.zeros([self.image_width, self.image_height, 3])
        self.collision = 0
        self.controller = None

    def _image_calback(self, data):
        cv2_img = None

        try:
            #Convert ROS image to Opencv
            cv2_img = bridge.imgmsg_to_cv2(data, "bgr8")

        except CvBridgeError, e:
            print(e)

        else:
            #cv2.imshow('image', cv2_img)
            #cv2.waitKey(1)
            
            self.header = data.header.stamp
            self.camera_image = cv2_img

    def _bumper_callback(self, data):
        
        # Check if hit something
        states = data.states
        
        if len(states) > 0:
            #print("bateu")
            self.collision = 1

        else:
            self.collision = 0

    def _car_control_callback(self, data):
        self.controller = data

    def routine(self):

        while True:
            time = self.header
            cv2.imwrite('/home/nesvera/Documents/train_pic/'+str(time)+'.jpg', self.camera_image)

            self.rate.sleep()

    

if __name__ == '__main__':
    logger = Datalogger()
    logger.routine()