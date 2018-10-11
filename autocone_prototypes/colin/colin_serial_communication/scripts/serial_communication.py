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

# Jetson to arduino variables
throtle = 0
steering = 0
jetsonStop = True

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

    throtle = int(data.data)

    # build message package
    msg = str(throtle) + ";" + str(steering) + ";" + str(int(jetsonStop)) + ";*"
    #print(msg)

    # Send to arduino
    serialComm.write(msg)


# Loop that reads messages of arduino
def ReadSerial():

    while True:

        receiveMsg = serialComm.readline()
        print(receiveMsg)


if __name__ == '__main__':
    global serialComm

    # Configure serial communication
    serialComm = serial.Serial()
    serialComm.port = '/dev/ttyACM0'
    serialComm.baudrate = 1000000
    serialComm.timeout = 1.5                                      # timeout in seconds

    try:
        serialComm.open()

    except SerialException:
        print("Failed to open serial communication!")

    if serialComm.is_open == False:
        print("Serial port is closed")

    # Calibrate arduino

    # Initialize the node
    rospy.init_node('arduino_comm', anonymous=True)

    rospy.Subscriber("/arduino/cmd_drive", Int32, Move)

    # Loop to read serial data
    ReadSerial()    
