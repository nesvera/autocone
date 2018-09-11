#!/usr/bin/env python
import rospy
import rospkg

from gazebo_msgs.msg import (
    ContactsState,
)

from sensor_msgs.msg import (
    Image,
)

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

        rospy.Subscriber("/camera/image_raw", Image, self._image_calback) 
        rospy.Subscriber("/bumper_sensor", ContactsState, self._bumper_callback, queue_size=1)
        #rospy.Subscriber("/controller")

        rospy.spin()

    def _image_calback(self, data):
        cv2_img = None

        try:
            #Convert ROS image to Opencv
            cv2_img = bridge.imgmsg_to_cv2(data, "bgr8")

        except CvBridgeError, e:
            print(e)

        else:
            cv2.imshow('image', cv2_img)
            cv2.waitKey(0)

    def _bumper_callback(self, data):
        
        # Check if hit something
        states = data.states
        
        if len(states) > 0:
            print("bateu")

    def _car_state_callback(self, data):
        pass

if __name__ == '__main__':
    logger = Datalogger()