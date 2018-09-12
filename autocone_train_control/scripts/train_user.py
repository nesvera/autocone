#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import rospy
import rospkg

import pickle

from gazebo_msgs.srv import (
    SpawnModel, 
    GetWorldProperties, 
    DeleteModel,
    SetModelState,
    GetModelState,
)

from gazebo_msgs.msg import (
    ModelState,
    ContactsState,
    ContactState,
)

from std_msgs.msg import String
from std_srvs.srv import Empty

from geometry_msgs.msg import Pose

from tf.transformations import quaternion_from_euler

import random
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

import math
import time
import sys

DEBUG = False
DEBUG_PLOT = True

class GazeboInterface:

    def __init__(self):
        
        # Services
        self.spawn_srv = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel, persistent=True)
        self.world_properties_srv = rospy.ServiceProxy("gazebo/get_world_properties", GetWorldProperties, persistent=True)
        self.delete_srv = rospy.ServiceProxy("gazebo/delete_model", DeleteModel, persistent=True)
        self.pause_physics_srv = rospy.ServiceProxy("gazebo/pause_physics", Empty, persistent=True)
        self.unpause_physics_srv = rospy.ServiceProxy("gazebo/unpause_physics", Empty, persistent=True)
        self.set_model_state_srv = rospy.ServiceProxy("gazebo/set_model_state", SetModelState, persistent=True)
        self.get_model_state_srv = rospy.ServiceProxy("gazebo/get_model_state", GetModelState, persistent=True)

    def spawn_model(self, model_file, model_name,  model_pose):
        
        try:            
            self.spawn_srv(model_name, model_file, "default", model_pose, "world")

        except rospy.ServiceException, e:
            print("Error: " + e)

    def get_world_properties(self):

        # Service to return models spawned on gazebo world
        rospy.wait_for_service("gazebo/get_world_properties")
        
        model_names = None

        # Read models
        try:
            resp_properties = self.world_properties_srv()
            model_names = resp_properties.model_names

        except rospy.ServiceException, e:
            print("Error: " + e)

        return model_names

    def move_model(self, model_name, model_pose):

        model = ModelState()
        model.pose = model_pose
        model.model_name = model_name

        rospy.wait_for_service("gazebo/set_model_state")

        # move
        try:
            self.set_model_state_srv(model)

        except rospy.ServiceException, e:
            print("Error: " + e)

    def get_model_position(self, model_name):
        rospy.wait_for_service("gazebo/get_model_state")

        # Get model position
        try:
            model_position = self.get_model_state_srv(model_name=model_name).pose.position          

        except rospy.ServiceException, e:
            print("Error: " + e)

        np_position = np.array([model_position.x, model_position.y])

        return np_position

    def pause_physics(self):
        self.pause_physics_srv()

    def unpause_physics(self):
        self.unpause_physics_srv()

class TrainControl:

    def __init__(self):

        self.enable_drive_flag = False

        self.vehicle_name = "ackermann_vehicle"

        self.qnt_cones = 500

        self.qnt_tracks = 1                         # number of tracks to train
        self.qnt_runs = 1000                          # number of reset of the car on a track

        # Cone model
        self.cone_model_path = rospkg.RosPack().get_path('autocone_description') + "/urdf/models/mini_cone/model.sdf"
        self.cone_file = None

        # Open and store models to spawn
        try:
            f = open(self.cone_model_path, 'r')
            self.cone_file = f.read()

        except:
            print("Could not read file: " + self.cone_model_path)

        # Models identifiers
        self.cone_name_list = list()
        
        self.path_points = []
        self.cur_restart_point = 0

        # Position 
        self.point_list = list()
        self.point_respawn_index = 1                     # index 

        # Class to interact with gazebo
        self.gazebo_interface = GazeboInterface()

        # init node
        rospy.init_node('train_control', anonymous=True)

        # Subscribers
        rospy.Subscriber("/bumper_sensor", ContactsState, self._bumper_callback, queue_size=1)

    def restart(self):

        quat = quaternion_from_euler(0, 0, math.pi/4.0)
       
        model_pose = Pose()
        model_pose.position.x = 0
        model_pose.position.y = 0
        model_pose.position.z = 0
        model_pose.orientation.x = quat[0]
        model_pose.orientation.y = quat[1]
        model_pose.orientation.z = quat[2]
        model_pose.orientation.w = quat[3]

    
        self.gazebo_interface.move_model(self.vehicle_name, model_pose)

    # callback listening for collision
    def _bumper_callback(self, data):
        
        # Check if hit something
        states = data.states

        if len(states) > 0:
            self.gazebo_interface.pause_physics()
            self.collision = True  
            self.enable_drive_flag = False    
            time.sleep(1)
            #print("bateeeeeu")

    def load_track(self):

        with open('/home/nesvera/catkin_ws/src/autocone/autocone_train_control/tracks/track.pickle', 'rb') as fp:
            self.track_points = pickle.load(fp)

        # Place/Move cones
        for i, pos in enumerate(self.track_points):

            progress_bar(i, len(self.track_points), prefix='Progress', length=50)

            cone_pose = Pose()
            cone_pose.position.x = pos[0]
            cone_pose.position.y = pos[1]
            cone_pose.position.z = 0

            cone_name = "cone_" + str(i)

            self.gazebo_interface.spawn_model(self.cone_file, cone_name, cone_pose)  

    def routine(self):

        # Pause simulation during startup
        print("Simulation Paused!")
        self.gazebo_interface.pause_physics()

        # spawn a lot of cones
        print("Loading track ...")
        self.load_track()
        
        while not rospy.is_shutdown():

            # enable car drive
            self.gazebo_interface.unpause_physics()
            self.enable_drive_flag = True

            # wait until it crash into a cone
            while self.enable_drive_flag == True:
                pass

            # reset car position
            self.gazebo_interface.pause_physics()
            self.restart()

            time.sleep(0.1)


# Print iterations progress
def progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """

    total -= 1

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total: 
        print("")

if __name__ == '__main__':

    print("Starting gazebo ...")

    control = TrainControl()
    control.routine()

