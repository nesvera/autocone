#!/usr/bin/env python
import rospy
import rospkg

import math

from std_msgs.msg import String
from gazebo_msgs.srv import *

from gazebo_msgs.msg import *
from std_srvs.srv import Empty
from geometry_msgs.msg import Point, Pose, Quaternion

def spawn_cone():

    block_pose=Pose(Point(x=0.6725, y=0.1265, z=0.7825), Quaternion(0, 0, 0, 0))
    block_reference_frame="world"

    # get model path
    model_path = rospkg.RosPack().get_path('autocone_description') + "/urdf/models/mini_cone"
    model = model_path + "/model.sdf"

    a = 1
    x = 0
    y = 0
    theta = 0

    count = 0
    while True:

        try:
            spawn = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)

            rospy.wait_for_service("gazebo/spawn_sdf_model")

            initial_pose = geometry_msgs.msg.Pose()
            initial_pose.position.x = x
            initial_pose.position.y = y
            initial_pose.position.z = 0

            f = open(model,'r')
            sdff = f.read()

            print spawn("cube"+str(count), sdff, "default", initial_pose, "world")
            count += 1

        except rospy.ServiceException, e:
            print ("Deu ruim: %s"%e)

        
        a += 0.01
        theta += 0.1
        x = a*math.cos(theta)
        y = a*math.sin(theta)
        

        

    


if __name__ == '__main__':
    
    rospy.init_node('spawner', anonymous=True)

    while True:
        spawn_cone()