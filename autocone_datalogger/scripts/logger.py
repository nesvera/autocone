#!/usr/bin/env python
import rospy
import rospkg
import getpass

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
import datetime

# Instantiate CvBridge
bridge = CvBridge()

DEBUG = False
DEBUG_PLOT = True

class Datalogger:

    def __init__(self):
        rospy.init_node("logger", anonymous=True)

        self.rate = rospy.Rate(30)

        # get simulation name
        self.sim_name = rospy.get_param('sim_name')
        username = getpass.getuser()
        self.dataset_folder = '/home/'+ username + '/Documents/autocone_dataset/'
        self.dataset_image_folder = self.dataset_folder + "/" + self.sim_name + "/"
        self.dataset_text_file = self.dataset_image_folder + self.sim_name + ".txt"

        rospy.Subscriber("/camera/image_raw", Image, self._image_calback, queue_size=1) 
        rospy.Subscriber("/bumper_sensor", ContactsState, self._bumper_callback, queue_size=1)
        rospy.Subscriber('/ackermann_cmd', AckermannDrive, self._car_control_callback, queue_size=1)

        self.image_width = 1280
        self.image_height = 960

        self.new_data = False
        self.header = None
        self.camera_image = np.zeros([self.image_width, self.image_height, 3])
        self.collision = 0
        self.steering = 0
        self.speed = 0

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

            self.new_data = True

    def _bumper_callback(self, data):
        
        # Check if hit something
        states = data.states
        
        if len(states) > 0:
            self.collision = 1

        else:
            self.collision = 0

    def _car_control_callback(self, data):
        self.steering = data.steering_angle
        self.speed = data.speed

    def routine(self):

        while not rospy.is_shutdown():

            if self.new_data == True:
                # year-month-day-hour-minute-seconds-microseconds
                filename = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')

                resized_image = cv2.resize(self.camera_image, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite((self.dataset_image_folder + str(filename) + '.jpg'), resized_image)

                output_msg = filename + ";" + str(self.speed) + ";" + str(self.steering) + ";" + str(self.collision) + ";*"
                print(output_msg)

                self.new_data = False

            self.rate.sleep()

    

if __name__ == '__main__':
    logger = Datalogger()
    logger.routine()