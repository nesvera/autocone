#!/usr/bin/env python
import rospy
import rospkg

from gazebo_msgs.srv import (
    SpawnModel, 
    GetWorldProperties, 
    DeleteModel,
    SetModelState,
)

from gazebo_msgs.msg import (
    ModelState,
)

from std_msgs.msg import String
from std_srvs.srv import Empty

from geometry_msgs.msg import Pose

import math
import time

class TrainControl:

    def __init__(self):
        
        # Path to the models
        self.cone_model_path = rospkg.RosPack().get_path('autocone_description') + "/urdf/models/mini_cone/model.sdf"
        self.cone_file = None

        # Open and store models to spawn
        try:
            f = open(self.cone_model_path, 'r')
            self.cone_file = f.read()

        except:
            print("Could not read file: " + self.cone_model_path)

        # Models identifiers
        self.cone_count = 0

        self.spawn_srv = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel, persistent=True)
        self.world_properties_srv = rospy.ServiceProxy("gazebo/get_world_properties", GetWorldProperties, persistent=True)
        self.delete_srv = rospy.ServiceProxy("gazebo/delete_model", DeleteModel, persistent=True)
        self.pause_physics_srv = rospy.ServiceProxy("gazebo/pause_physics", Empty, persistent=True)
        self.unpause_physics_srv = rospy.ServiceProxy("gazebo/unpause_physics", Empty, persistent=True)
        self.set_model_state_srv = rospy.ServiceProxy("gazebo/set_model_state", SetModelState, persistent=True)

    def spawn_cone(self, posX, posY):
        
        try:            
            cone_pose = Pose()
            cone_pose.position.x = posX
            cone_pose.position.y = posY
            cone_pose.position.z = 0

            model_name = "cone_" + str(self.cone_count)
            self.spawn_srv(model_name, self.cone_file, "default", cone_pose, "world")
            
            self.cone_count += 1

        except rospy.ServiceException, e:
            print("Error: " + e)

    def reset_world(self):

        # Service to return models spawned on gazebo world
        rospy.wait_for_service("gazebo/get_world_properties")
        
        model_names = None

        # Read models
        try:
            resp_properties = self.world_properties_srv()
            model_names = resp_properties.model_names

        except rospy.ServiceException, e:
            print("Error: " + e)

        # Remove cones and car
        try:
            for name in model_names:

                # Dont remove ground_plane
                if "ground_plane" in name or "ackermann_vehicle" in name:
                    print("achou")
                    continue

                print(name)

                # Call delete
                resp_delete = self.delete_srv(name)

                if resp_delete.success == False:
                    print("Error trying to delete cone")

                time.sleep(1) 

        except rospy.ServiceException, e:
            print("Error: " + e)

    def move_model(self, model_name, posX, posY):
        model = ModelState()
        model.pose.position.x = posX
        model.pose.position.y = posY
        model.pose.position.z = 0
        model.model_name = model_name

        # Remove cones and car
        try:
            rospy.wait_for_service("gazebo/set_model_state")
            self.set_model_state_srv(model)

        except rospy.ServiceException, e:
            print("Error: " + e)

    def test_move(self):

        x = 0
        y = 0

        # Service to return models spawned on gazebo world
        rospy.wait_for_service("gazebo/get_world_properties")
        
        model_names = None

        # Read models
        try:
            resp_properties = self.world_properties_srv()
            model_names = resp_properties.model_names

        except rospy.ServiceException, e:
            print("Error: " + e)

        # Remove cones and car
        try:
            for name in model_names:

                # Dont remove ground_plane
                if "ground_plane" in name or "ackermann_vehicle" in name:
                    print("achou")
                    continue

                print(name)               

                self.move_model(name, x, y)
                x += 1

                time.sleep(0.1) 

        except rospy.ServiceException, e:
            print("Error: " + e)


    def pause_physics(self):
        self.pause_physics_srv()

    def unpause_physics(self):
        self.unpause_physics_srv()

    



if __name__ == '__main__':

    rospy.init_node('train_control', anonymous=True)

    b = TrainControl()

    b.pause_physics()

    a = 1
    x = 0
    y = 0
    theta = 0

    for i in range(10):
        a += 0.01
        theta += 0.1
        x = a*math.cos(theta)
        y = a*math.sin(theta)

        b.spawn_cone(x, y)

    b.unpause_physics()

    raw_input()

    b.pause_physics()
    b.test_move()
    b.unpause_physics()

    raw_input()

    b.pause_physics()
    b.reset_world()
    b.unpause_physics()
