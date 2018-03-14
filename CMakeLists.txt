cmake_minimum_required(VERSION 2.8.3)
project(openpose_ros)

## Add support for C++11, supported in ROS Kinetic and newer
add_definitions(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  cv_bridge
  image_transport
  gflags
  glog_catkin
  message_generation
)

add_message_files(
        FILES
        Bodypart.msg
        PersonDetection.msg
)

add_service_files( DIRECTORY srv
        FILES
        Person.srv
        )

generate_messages(DEPENDENCIES std_msgs sensor_msgs)

find_package(CUDA REQUIRED)
find_package(OpenCV 3.1 REQUIRED)

set(CAFFE_DIR /usr/local)
set(OPENPOSE_DIR /usr/local)

set(CAFFE_INCLUDE_DIRS ${CAFFE_DIR}/include)
set(CAFFE_LINK_LIBRARY ${CAFFE_DIR}/lib/libcaffe.so)

set(OPENPOSE_INCLUDE_DIRS ${OPENPOSE_DIR}/include)
set(OPENPOSE_LINK_LIBRARY ${OPENPOSE_DIR}/lib/libopenpose.so)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES openpose_ros
   CATKIN_DEPENDS roscpp rospy message_runtime std_msgs sensor_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories( include )
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${CAFFE_INCLUDE_DIRS}
  ${OPENPOSE_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}_node src/openpose_ros_node.cpp)
add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${CAFFE_LINK_LIBRARY}
  ${OPENPOSE_LINK_LIBRARY}
)

#MESSAGE(OpenCV_VERSION: ${OpenCV_VERSION})
#MESSAGE(OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS})

#get_cmake_property(_variableNames VARIABLES)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()
