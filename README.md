# autocone

Tested on:

    Ubuntu 16.04 LTS
    ROS Kinetic

ROS dependencies needed:

    sudo apt-get ros-kinetic-joint-state-controller
    sudo apt-get ros-kinetic-effort-controllers
    sudo apt-get ros-kinetic-gazebo-ros-control
    sudo apt-get install ros-kinetic-joy

    catkin_make install

How to use

    roslaunch autocone_gazebo empty_colin.launch
    rosrun autocone_train_control train_control.py
    rosrun rqt_image_view rqt_image_view
    rosrun rqt_publisher rqt_publisher

    roslaunch autocone_train_control random_multiple_track.launch
    roslaunch autocone_train_control human_one_track.launch controller:="ps4" fixed_speed:=1

Colin on Jetson

    sudo apt-get install ros-kinetic-joy

    Install Microsoft xbox360 controller (http://kyubot.tistory.com/92)
    sudo apt-get install jstest-gtk joystick xboxdrv
    sudo xboxdrv --silent
    Create a file: /etc/rc.local
    File contents:
    #!/bin/sh -e
    xboxdrv --silent
    exit 0
    Save the file and make it executable with this command:
    sudo chmod +x /etc/rc.local

    sudo apt-get install ros-kinetic-ackermann*