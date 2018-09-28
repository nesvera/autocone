#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2

import keras
from keras.models import load_model
from keras.optimizers import Adam
from keras import backend as K

import rospy
from ackermann_msgs.msg import AckermannDrive
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()

class Predict:

    def __init__(self):
        self.camera_image = 0
        self.speed = 0
        self.steering = 0
        self.model = self.load_trained_model('/home/catkin_ws/src/autocone/autocone_utils/cnn/saved_models/my_model2.h5')
        
        rospy.Subscriber("/camera/image_raw/binary", Image, self._image_calback, queue_size=1) 
        rospy.Subscriber('/ackermann_cmd', AckermannDrive, self._car_control_callback, queue_size=1)

        self.width = 384
        self.height = 288

        self.ackermann_cmd = AckermannDrive()
        self.ackermann_cmd.speed = 1
        self.ackermann_cmd.steering_angle = 0.0

        self.ackermann_pub = rospy.Publisher('/ackermann_cmd', AckermannDrive, queue_size=1)

        self.rate = rospy.Rate(30)

    def _image_calback(self, data):
        cv2_img = None
        data.encoding = "mono8"

        try:
            #Convert ROS image to Opencv
            cv2_img = bridge.imgmsg_to_cv2(data, "mono8")   # binary
        except CvBridgeError, e:
            print(e)
        else:
            self.camera_image = cv2_img
            self.predict(self.camera_image)

    def _car_control_callback(self, data):
        self.steering = data.steering_angle
        self.speed = data.speed

    def load_trained_model(self, model_path):
        model = load_model(model_path)
        learning_rate = 0.01
        model.compile(loss='mean_squared_error', 
                    optimizer=Adam(lr=learning_rate), 
                    metrics=['accuracy'])
        return model

    def resize_and_reshape(self, img):
        resized = cv.resize(img,(0.3*self.width, 0.3*self.height), interpolation = cv.INTER_CUBIC)
        img = np.asarray(img)
        shape = img.shape

        if K.image_data_format() == 'channels_first': # channels, rows, cols
            img = img.reshape(img.shape[0], 1, shape[1], shape[2])
            input_shape = (1, shape[1], shape[2])
        else:
            img = img.reshape(img.shape[0], shape[1], shape[2], 1) # rows, cols, channels
            input_shape = (shape[1], shape[2], 1)

        return img

    def predict(self):
        cv2_img = bridge.imgmsg_to_cv2(data, "mono8")   # binary
        img = self.resize_and_reshape(cv2_img)
        result = self.model.predict(img)
        self.ackermann_cmd.speed = result[0]
        self.ackermann_cmd.steering_angle = result[1]
        print(result)
        self.rate.sleep()

if __name__ == '__main__':
    p = Predict()

