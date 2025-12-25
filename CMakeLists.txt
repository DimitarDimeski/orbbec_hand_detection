cmake_minimum_required(VERSION 3.8)
project(orbbec_hand_detection)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/HandDetection.msg"
  "msg/TouchEvent.msg"
  "msg/TouchDetection.msg"
  DEPENDENCIES std_msgs
)

# Install Python package (this includes the generated message modules)
ament_python_install_package(${PROJECT_NAME})

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# Install rviz files
install(DIRECTORY rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

