#!/usr/bin/env python

import rospy
import cv2
import numpy as np
print(cv2.__version__)
import os
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image

LOWER_COLOR = np.array([0, 150, 150])
UPPER_COLOR = np.array([15, 255, 255])

# Gets image from the gazebo simulation camera and apply opencv transformations
# to extract the traffic cones information
class SimVision:
    def __init__(self):
        rospy.init_node("sim_vision")

        # Used to conver ros image messages to opencv images
        self.cv_bridge = CvBridge()

        cam_sub = rospy.Subscriber("/camera/image_raw", Image, self.cvProcess)

        self.pub = rospy.Publisher('/camera/image_raw/binary', Image, queue_size=10)

    def cvProcess(self, img_data):
        cv_img = self.cv_bridge.imgmsg_to_cv2(img_data, "bgr8")

        mask = cv2.inRange(cv_img, LOWER_COLOR, UPPER_COLOR)

        try:
            self.pub.publish(self.cv_bridge.cv2_to_imgmsg(mask, encoding="passthrough"))
        except CvBridgeError as e:
            print(e)

        #cv2.imshow("Mask", mask)
        #cv2.imshow("Img", cv_img)
        #cv2.waitKey(1)

if __name__ == "__main__":
    sim_vision = SimVision()
    rospy.spin()
