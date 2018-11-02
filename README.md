# autocone

Tested on:

    Ubuntu 16.04 LTS
    ROS Kinetic

ROS dependencies needed:

    sudo apt-get install ros-kinetic-joint-state-controller
    sudo apt-get install ros-kinetic-effort-controllers
    sudo apt-get install ros-kinetic-gazebo-ros-control
    sudo apt-get install ros-kinetic-joy
    sudo apt-get install ros-kinetic-ackermann-*
    sudo apt-get install ros-kinetic-std-msgs 

    catkin_make install

How to use

    roslaunch autocone_gazebo empty_colin.launch
    rosrun autocone_train_control train_control.py
    rosrun rqt_image_view rqt_image_view
    rosrun rqt_publisher rqt_publisher

    roslaunch autocone_train_control random_multiple_track.launch
    roslaunch autocone_train_control human_one_track.launch controller:="ps4" fixed_speed:=1
