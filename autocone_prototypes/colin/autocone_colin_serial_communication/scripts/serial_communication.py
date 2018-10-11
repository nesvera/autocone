#!/usr/bin/env python

'''
    This node implements the communication between the computer and
    the controller board.

    -> Open serial communication with the controller board
    -> Get ackermann data and send to the controller board
    -> Read controller board data from the car
'''

import rospy
from std_msgs.msg import Int32
from ackermann_msgs.msg import AckermannDrive

import serial
from serial import SerialException
import time

# Jetson to arduino variables
throtle = 0
steering = 0
jetsonStop = False

throtleMin = -100
throtleMax = 100
steeringMin = -100
steeringMax = 100

# Arduino to Jetson variables
aproxSpeed = 0                              # speed from hall effect sensor
hallCounter = 0                             # number of pulses from hall effect sensor
accelX = 0
accelY = 0
accelZ = 0
gyroRoll = 0
gyroPitch = 0
gyroYaw = 0
radioStop = False 

# Callback to sendo data to arduino
def Move(data):

    throtle = float(data.speed)
    steering  = float(data.steering_angle)

    #print(throtle, steering)

    # build message package
    msg = "&" + str(int(throtle)) + ";" + str(int(steering)) + ";" + str(int(jetsonStop)) + ";*"
    #msg = "20;*"
    #print(msg)

    # Send to arduino
    serialComm.write(bytes(msg))

    #print(msg)


# Loop that reads messages of arduino
def ReadSerial():

    while True:
        receiveMsg = serialComm.readline()
        print(receiveMsg)
        pass


if __name__ == '__main__':
    global serialComm

    # Configure serial communication
    serialComm = serial.Serial()
    serialComm.port = '/dev/ttyACM0'
    serialComm.baudrate = 1000000
    serialComm.timeout = 0.5                                      # timeout in seconds

    try:
        serialComm.open()

    except SerialException:
        print("Failed to open serial communication!")

    if serialComm.is_open == False:
        print("Serial port is closed")

    time.sleep(5)

    print("Communication Initialized!")

    # Calibrate arduino
    msg = "&30;*"

    #raw_input()
    #print("um")
    #serialComm.write(bytes(msg))
    #raw_input()
    #print("dois")
    #serialComm.write(bytes(msg))
    #raw_input()


    # Initialize the node
    rospy.init_node('arduino_comm', anonymous=True)

    rospy.Subscriber('/ackermann_cmd', AckermannDrive, Move)

    # Loop to read serial data
    ReadSerial()    
