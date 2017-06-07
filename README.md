# OpenPose-ROS
ROS wrapper for [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

## Installation
### Preliminaries
1. OpenPose, see [here](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md)
1. __To avoid _Segmentation fault_ compile OpenPose with OpenCV 3__

### OpenPose-ROS
1. In a catkin workspace do the following steps ([how to create a catkin workspace](http://wiki.ros.org/catkin/Tutorials/create_a_workspace))
1. ```catkin_make```

## Running
1. ```source catkin_ws/devel/setup.bash
1. ```roscore```
1. ```rosrun openpose-ros openpose-ros-node```

## Tested
* Ubuntu 16.04 
* ROS Kinetic
* CUDA 8.0
* cuDNN 6.0
* __OpenCV 3.2__