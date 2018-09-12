# autocone

Tested on:

    Ubuntu 16.04 LTS
    ROS Kinetic

ROS dependencies needed:

    sudo apt-get ros-kinetic-joint-state-controller
    sudo apt-get ros-kinetic-effort-controllers
    sudo apt-get ros-kinetic-gazebo-ros-control

    catkin_make install

How to use

    roslaunch autocone_gazebo empty_colin.launch
    rosrun autocone_train_control train_control.py
    rosrun rqt_image_view rqt_image_view
    rosrun rqt_publisher rqt_publisher
