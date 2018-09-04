#!/usr/bin/env python

import rospy

from ackermann_msgs.msg import AckermannDrive

from gazebo_msgs.msg import ContactsState
from rosgraph_msgs.msg import Clock

import random
import time
import numpy as np

max_step = 0.1

class Drive:

    def __init__(self):

        self.ackermann_cmd = AckermannDrive()
        self.ackermann_cmd.speed = 0.4
        self.ackermann_cmd.steering_angle = 0.0
    
        self.collision = False

        self.ackermann_pub = rospy.Publisher('/ackermann_cmd', AckermannDrive, queue_size=1)
        self.bumper_sub = rospy.Subscriber('/bumper_sensor', ContactsState, self._bumper_callback, queue_size=1)
        self.time_sub = rospy.Subscriber('/clock', Clock, self._clock_callback)

        self.drive_cur_time = 0
        self.drive_init_time = 0
        self.drive_time = 0

        self.population = np.zeros((4, 100), dtype=np.float)
        self.fitness = np.zeros((4, 2), dtype=np.float)
        
        self.individuo_counter = 0
        self.step_counter = 0

        for i in range(self.population.shape[0]):
            for j in range(self.population.shape[1]):
                self.population[i][j] = random.uniform(-1.0, 1.0)

        for i in range(self.fitness.shape[0]):
            self.fitness[i][0] = i
            self.fitness[i][1] = 0


        self.rate = rospy.Rate(10)

    def update(self):

        while not rospy.is_shutdown():

            # while car no colide, 
            while self.collision == False:
                print(self.individuo_counter, self.step_counter)

                # get an action of step n
                self.ackermann_cmd.steering_angle = self.population[self.individuo_counter][self.step_counter]
                
                # increment step_counter
                self.step_counter += 1
                if self.step_counter == self.population.shape[1]:
                    self.step_counter = 0

                # publish
                self.ackermann_pub.publish(self.ackermann_cmd)

                self.rate.sleep()

            self.step_counter = 0
            self.collision = False

            # calculate fitness
            self.fitness[self.individuo_counter][1] = self.drive_time

            # get another individuo
            self.individuo_counter += 1

            # if the population is over, execute generic algorithm
            if self.individuo_counter == self.population.shape[0]:
                
                arr = self.fitness[self.fitness[:,1].argsort()]
                best = arr[-2][0]

                while True:

                    self.step_counter = 0
                    # while car no colide, 
                    while self.collision == False:

                        # get an action of step n
                        self.ackermann_cmd.steering_angle = self.population[int(best)][self.step_counter]
                        
                        # increment step_counter
                        self.step_counter += 1
                        if self.step_counter == self.population.shape[1]:
                            self.step_counter = 0

                        # publish
                        self.ackermann_pub.publish(self.ackermann_cmd)

                        self.rate.sleep()

                    self.collision = False



    def _clock_callback(self, data):
        self.drive_cur_time = data.clock.secs*1000 + (data.clock.nsecs/1000000.0)

    # callback listening for collision
    def _bumper_callback(self, data):
        
        # Check if hit something
        states = data.states

        if len(states) > 0:
            self.collision = True
            self.drive_time = self.drive_cur_time - self.drive_init_time
            print(self.drive_time)
            self.drive_init_time = self.drive_cur_time


            time.sleep(0.5)


if __name__ == "__main__":

    rospy.init_node('steer_randomly', anonymous=True)

    d = Drive()

    while True:
        d.update()