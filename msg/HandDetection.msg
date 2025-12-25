# Hand detection message containing pointer finger coordinates
# Published by MediaPipe detection node

std_msgs/Header header

# List of detected pointer finger tip coordinates in image space
float32[] x_coordinates  # X coordinates in pixels
float32[] y_coordinates  # Y coordinates in pixels

