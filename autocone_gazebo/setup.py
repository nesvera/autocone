from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
  scripts=['scripts/colin_controller.py'],
  requires=['std_msgs', 'rospy']
)

setup(**d)