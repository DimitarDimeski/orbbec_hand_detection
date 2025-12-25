cmake_minimum_required(VERSION 3.8)
project(orbbec_hand_detection)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/HandDetection.msg"
  "msg/TouchEvent.msg"
  "msg/TouchDetection.msg"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

install(
  DIRECTORY orbbec_hand_detection
  DESTINATION lib/python3/site-packages
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# Install rviz files
install(DIRECTORY rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

