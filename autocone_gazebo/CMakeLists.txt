cmake_minimum_required(VERSION 2.8.3)
project(autocone_gazebo)

find_package(catkin REQUIRED)
catkin_python_setup()

catkin_package()

install(PROGRAMS
  scripts/colin_controller.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

