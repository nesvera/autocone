# autocone

Tested on:

    Ubuntu 16.04 LTS
    ROS Kinetic

ROS dependencies needed:

    sudo apt-get install ros-kinetic-joint-state-controller
    sudo apt-get install ros-kinetic-effort-controllers
    sudo apt-get install ros-kinetic-gazebo-ros-control
    sudo apt-get install ros-kinetic-joy


    catkin_make install

How to use

    roslaunch autocone_gazebo empty_colin.launch
    rosrun autocone_train_control train_control.py
    rosrun rqt_image_view rqt_image_view
    rosrun rqt_publisher rqt_publisher
