cmake_minimum_required(VERSION 3.8)
project(orbbec_hand_detection)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# Generate messages (this is the only thing CMake needs to do)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/HandDetection.msg"
  "msg/TouchEvent.msg"
  "msg/TouchDetection.msg"
  DEPENDENCIES std_msgs
)

ament_package()

