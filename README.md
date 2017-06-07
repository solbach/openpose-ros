# openpose-ros
ROS wrapper for [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

## Tested
* Ubuntu 16.04 
* ROS Kinetic
* CUDA 8.0
* cuDNN 6.0
* __OpenCV 3.2__

## Installation

1. OpenPose, see [here](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md)
2. In a catkin workspace do the following steps ([how to create a catkin workspace](http://wiki.ros.org/catkin/Tutorials/create_a_workspace))
3. ```catkin_make```

## Running
1. ```source catkin_ws/devel/setup.bash
2. ```roscore```
3. ```rosrun openpose-ros openpose-ros-node```
