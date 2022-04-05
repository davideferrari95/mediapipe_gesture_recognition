cmake_minimum_required(VERSION 3.0.2)
project(mediapipe_gesture_recognition)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  message_generation
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
)

################################################
## Declare ROS messages, services and actions ##
################################################

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  Pose.msg  
  Keypoint.msg
  Hand.msg
  Face.msg
)

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
    std_msgs
    ${PROJECT_NAME}
)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################

catkin_package(
  INCLUDE_DIRS
    # include
  CATKIN_DEPENDS
    roscpp
    rospy
    std_msgs
    message_runtime
  # LIBRARIES
  # DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
  # include
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/sk_tracking.cpp
# )

## Declare a C++ executable
# add_executable(${PROJECT_NAME}_node src/sk_tracking_node.cpp)
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES})

#############
## Install ##
#############

install(PROGRAMS 
  scripts/mediapipe_stream_node.py
  scripts/gesture_recognition_node.py
  scripts/gesture_recognition_training_node.py
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)
