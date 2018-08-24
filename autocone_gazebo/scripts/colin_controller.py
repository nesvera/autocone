#!/usr/bin/env python

import rospy

from std_msgs.msg import Float64

class ColinController():

    def __init__(self):
        
        rospy.init_node("colin_controller")

        pub1 = rospy.Publisher('/autocone_colin/rear_right_axle_ctl/command', Float64, queue_size=1)
        pub2 = rospy.Publisher('/autocone_colin/rear_left_axle_ctl/command', Float64, queue_size=1)

        rate = rospy.Rate(100) #100 Hz

        while not rospy.is_shutdown():

            pub1.publish(2)
            pub2.publish(2)


    

# main
if __name__ == "__main__":

    controller = ColinController()


