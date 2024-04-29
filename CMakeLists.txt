cmake_minimum_required(VERSION 3.5)
project(mediapipe_gesture_recognition)

# Default to C99
if(NOT CMAKE_C_STANDARD)
    set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find Dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

# Find Messages Dependencies
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# Find Messages Generation Dependencies
find_package(rosidl_default_generators REQUIRED)

# Find Other Dependencies
find_package(image_transport REQUIRED)

# Set Cusom Messages
set(msg_files
    "msg/Pose.msg"  
    "msg/Keypoint.msg"
    "msg/Hand.msg"
    "msg/Face.msg"
    "msg/Frame.msg"
)

# Generate Messages, Services and Actions
rosidl_generate_interfaces(${PROJECT_NAME}
    ${msg_files}
    DEPENDENCIES std_msgs
)

# Include C++ Directories
include_directories(
    include
    ${${PROJECT_NAME}_INCLUDE_DIRS}
)

# Install Python Modules
ament_python_install_package(scripts)

# Install Python Scripts
install(PROGRAMS

    # Stream, Recognition, and Training Nodes
    scripts/stream_node.py
    scripts/training_node.py
    scripts/recognition_node.py

    # Dataset Utilities
    scripts/dataset_utilities/process_dataset_node.py
    scripts/dataset_utilities/jester_dataset/dataset_converter.py

    # Point-At Utilities
    # scripts/point_at_utilities/point_area_node.py
    # scripts/point_at_utilities/area_coordinates_recorder.py

    DESTINATION lib/${PROJECT_NAME}
)

# Install Launch Files
install(DIRECTORY
    launch
    DESTINATION share/${PROJECT_NAME}/
)

# Install Config Files
install(DIRECTORY
    config
    DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    #set(ament_cmake_copyright_FOUND TRUE)
    #set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
endif()

ament_package()
