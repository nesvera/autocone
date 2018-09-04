#!/usr/bin/env python
import rospy
import rospkg
import random
import numpy as np
from scipy import interpolate

def random_points(length,npoints):
    size = length/2
    
    end = 0
    while end==0:
        X = [0]
        Y = [0]
        
        i = 0
        k = 0
        while i<length-1:
            a = X[-1]+random.randint(-1,1)
            b = Y[-1]+random.randint(-1,1)
            
            f = 0
            for j in range(len(X)-1):
                if ((X[j]-a)**2+(Y[j]-b)**2)**(1/2)<=1:
                    f = 1
            
            if len(X)>1:
                if ((X[-2]-a)**2+(Y[-2]-b)**2)**(1/2)<=2:
                    f = 1
            
            if f == 0 and abs(a)<size and abs(b)<size:
                X.append(a)
                Y.append(b)
                i=i+1
                k = 0
                
            k=k+1
            
            if k > 10:
                del X[-1]
                del Y[-1]
                i = i-1
            
      
            if i>=length*0.7:
                if X[1]*-1==X[-1] and Y[1]*-1==Y[-1]:
                    X.append(0)
                    Y.append(0)
                    if ((X[-3])**2+(Y[-3])**2)**(1/2)>2:
                        end = 1
                        i = length
        
    
    
#    print("Length = ",len(X))
    
    X.append(X[1])
    Y.append(Y[1])
    #X.append(X[2])
    #Y.append(Y[2])
    
    x = np.array(X)
    y = np.array(Y)
    tck, u = interpolate.splprep([x, y], s=0)
    unew = np.arange(0, 1, 1/npoints)
    out = interpolate.splev(unew, tck)
    
    X4 = []
    Y4 = []
    med_X=[]
    med_Y=[]
    for i in range(len(out[0])):
        if X[1]>0:
            if Y[1]>0:
                if out[0][i]>=0 and out[0][i]<=1 and out[1][i]>=0 and out[1][i]<=1:
                    med_X.append(out[0][i])
                    med_Y.append(out[1][i])
                else:
                    X4.append(out[0][i])
                    Y4.append(out[1][i])
                    
                
            else:
                if out[0][i]>=0 and out[0][i]<=1 and out[1][i]<=0 and out[1][i]>=-1:
                    med_X.append(out[0][i])
                    med_Y.append(out[1][i])
                else:
                    X4.append(out[0][i])
                    Y4.append(out[1][i])
                   
        else:
            if Y[1]>0:
                if out[0][i]>=-1 and out[0][i]<=0 and out[1][i]>=0 and out[1][i]<=1:
                    med_X.append(out[0][i])
                    med_Y.append(out[1][i])
                else:
                    X4.append(out[0][i])
                    Y4.append(out[1][i])
                    
            else:
                if out[0][i]>=-1 and out[0][i]<=0 and out[1][i]<=0 and out[1][i]>=-1:
                    med_X.append(out[0][i])
                    med_Y.append(out[1][i])
                else:
                    X4.append(out[0][i])
                    Y4.append(out[1][i])
                    
    X1 = []
    Y1 = []
    X2 = []
    Y2 = []
    i = 0
    while i < 1000:
        if ((med_X[i]-med_X[i+1])**2+(med_Y[i]-med_Y[i+1])**2)**(1/2)<0.5:
            X1.append(med_X[i])
            Y1.append(med_Y[i])
        else:
            i = i + 1
            for j in range(i,len(med_X)):
                X2.append(med_X[j])
                Y2.append(med_Y[j])
            i = 1000
        i = i + 1
        
    
    
    X3 = []
    Y3 = []
    for i in range(len(X1)):
        X3.append((X1[i]+X2[i])/2)
        Y3.append((Y1[i]+Y2[i])/2)
    
    
    del(X4[0])
    del(Y4[0])
    
    for i in range(len(X3)):
        X4.append(X3[i])
        Y4.append(Y3[i])
    
    X4.append(X4[0])
    Y4.append(Y4[0])
    
        
    points = []
    for i in range(len(X4)):
        points.append(np.array([X4[i],Y4[i]]))
    
    return points
    
    
    


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

import math
import time
import numpy as np

class TrainControl:

    def __init__(self):

        self.enable_drive_flag = False

        self.vehicle_name = "ackermann_vehicle"

        self.qnt_tracks = 1                         # number of tracks to train
        self.qnt_runs = 100                          # number of reset of the car on a track

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
        self.cone_list = list()

        # Position 
        self.point_list = list()
        self.point_respawn_index = 1                     # index 

        self.track_width = 1.0                          # distance between two cones
        self.up_vector = np.array([0, 0, 1])            # vector pointing up

        # init node
        rospy.init_node('train_control', anonymous=True)

        # Subscribers
        rospy.Subscriber("/bumper_sensor", ContactsState, self._bumper_callback, queue_size=1)

        # Services
        self.spawn_srv = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel, persistent=True)
        self.world_properties_srv = rospy.ServiceProxy("gazebo/get_world_properties", GetWorldProperties, persistent=True)
        self.delete_srv = rospy.ServiceProxy("gazebo/delete_model", DeleteModel, persistent=True)
        self.pause_physics_srv = rospy.ServiceProxy("gazebo/pause_physics", Empty, persistent=True)
        self.unpause_physics_srv = rospy.ServiceProxy("gazebo/unpause_physics", Empty, persistent=True)
        self.set_model_state_srv = rospy.ServiceProxy("gazebo/set_model_state", SetModelState, persistent=True)
        self.get_model_state_srv = rospy.ServiceProxy("gazebo/get_model_state", GetModelState, persistent=True)

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

        except rospy.ServiceException, e:
            print("Error: " + e)

    def move_model(self, model_name, posX, posY, yaw):
        model = ModelState()
        model.pose.position.x = posX
        model.pose.position.y = posY
        model.pose.position.z = 0
        model.pose.orientation.x = 0
        model.pose.orientation.y = 0
        model.pose.orientation.z = yaw
        model.model_name = model_name

        rospy.wait_for_service("gazebo/set_model_state")

        while True:

            # Try to move
            try:
                self.set_model_state_srv(model)

            except rospy.ServiceException, e:
                print("Error: " + e)

            # Check if it has moved
            prop_position = self.get_model_position(model_name=model_name)
            if prop_position[0] == posX and prop_position[1] == posY:
                break   

    def get_model_position(self, model_name):
        
        # Get cones position
        try:
            rospy.wait_for_service("gazebo/get_model_state")
            cone_position = self.get_model_state_srv(model_name=model_name).pose.position          

        except rospy.ServiceException, e:
            print("Error: " + e)

        np_cone_position = np.array([cone_position.x, cone_position.y])
        return np_cone_position

    def test_move(self):

        self.pause_physics_srv()

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

        try:
            for name in model_names:

                # Dont remove ground_plane
                if "ground_plane" in name or "ackermann_vehicle" in name:
                    print("achou")
                    continue

                print(name)               

                self.move_model(name, x, y, 0)
                #self.get_model_position(name)


                x += 1

                if x % 10 == 0:
                    y += 1
                    x = 0

                #time.sleep(0.1) 

        except rospy.ServiceException, e:
            print("Error: " + e)

        self.unpause_physics_srv()


    def pause_physics(self):
        self.pause_physics_srv()

    def unpause_physics(self):
        self.unpause_physics_srv()

    def generate_track(self):

        # Generate track
        self.track_points = random_points(30,500)

        # Place cones
        for i in range(len(self.track_points)):

            # get two points
            if i < len(self.track_points)-1:
                p1 = np.array([self.track_points[i][0], self.track_points[i][1], 0])
                p2 = np.array([self.track_points[i+1][0], self.track_points[i+1][1], 0])

            else:
                p1 = np.array([self.track_points[i][0], self.track_points[i][1], 0])
                p2 = np.array([self.track_points[0][0], self.track_points[0][1], 0])

            # calculate cross product
            forward_vector = p2-p1 
            forward_vector = forward_vector / np.linalg.norm(forward_vector)
            right_vector = np.cross(forward_vector, self.up_vector)
            
            cone1 = p1 + (self.track_width/2.0)*right_vector
            cone2 = p1 - (self.track_width/2.0)*right_vector

            # change cone position
            self.spawn_cone(cone1[0], cone1[1])
            self.spawn_cone(cone2[0], cone2[1])

            print(i/len(self.track_points))

            #print(str(cone1[0]) + "," + str(cone1[1]) + "  -  " + str(cone2[0]) + "," + str(cone2[1]))
            #time.sleep(0.1)


    # callback listening for collision
    def _bumper_callback(self, data):
        
        '''
        # Check if hit a cone
        states = data.states[0]
        collision_name = states.collision1_name
        
        if "cone" in collision_name:
            self.collision = True
            print("bateeeeeu")
        '''

        # Check if hit something
        states = data.states

        if len(states) > 0:
            self.pause_physics()
            self.collision = True  
            self.enable_drive_flag = False    
            print("bateeeeeu")

            time.sleep(0.5)


    def routine(self):

        self.pause_physics()

        # spawn a lot of cones
        #self.spawn_many_cones()
        self.generate_track()
        
        for track in range(self.qnt_tracks):

            # generate a track

            for run in range(self.qnt_runs):

                # enable car drive
                self.unpause_physics()
                self.enable_drive_flag = True

                # wait until it crash into a cone
                while self.enable_drive_flag == True:
                    pass

                # reset car position
                self.pause_physics()
                self.move_model(self.vehicle_name, 0, 0, 0)


if __name__ == '__main__':

    control = TrainControl()
    control.routine()

'''
    b.pause_physics()
    #b.generate_track()
    b.test_move()
    b.unpause_physics()

    raw_input()

    exit(1)

    

    exit(1)

    b.pause_physics()

    a = 1
    x = 0
    y = 0
    theta = 0

    for i in range(0):
        a += 0.01
        theta += 0.1
        x = a*math.cos(theta)
        y = a*math.sin(theta)

        b.spawn_cone(x, y)

    b.unpause_physics()

    print("press enter")

    raw_input()

    b.pause_physics()
    b.test_move()
    b.unpause_physics()

    raw_input()

    b.pause_physics()
    b.reset_world()
    b.unpause_physics()
'''
