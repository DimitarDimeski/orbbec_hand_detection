#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import os


class ArucoDisplayNode(Node):
    def __init__(self):
        super().__init__('aruco_display_node')
        self.get_logger().info('Starting ArUco display node...')

        # Parameters
        self.declare_parameter('width', 1920)
        self.declare_parameter('height', 1080)
        self.declare_parameter('marker_size', 400)
        self.declare_parameter('dictionary_id', cv2.aruco.DICT_4X4_50)
        self.declare_parameter('output_yaml', 'calibration.yaml')

        # Get parameter values
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.marker_size = self.get_parameter('marker_size').value
        self.dictionary_id = self.get_parameter('dictionary_id').value
        self.output_yaml = self.get_parameter('output_yaml').value


        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.dictionary_id)

        # Create and show image and inverted image in case of black background
        self.image, self.image_inverted = self.create_image_with_aruco_markers()
        self.window_name = "Fullscreen ArUco"

        # Display image in fullscreen
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(self.window_name, self.image)

        # Keep track if latest image has been inverted
        self.latest_inverted = False

        # Create a timer to periodically check shutdown
        self.timer = self.create_timer(0.5, self.update_window)
        self.get_logger().info("ArUco markers displayed. Node will stay open until shutdown.")


    def create_image_with_aruco_markers(self):
        image = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

        marker_positions = [
            (0, 0),
            (self.width - self.marker_size, 0),
            (0, self.height - self.marker_size),
            (self.width - self.marker_size, self.height - self.marker_size)
        ]

        for i, (x, y) in enumerate(marker_positions):
            marker = cv2.aruco.generateImageMarker(self.aruco_dict, i, self.marker_size)
            image[y:y+self.marker_size, x:x+self.marker_size] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

        # Invert the image in case of black background
        inverted_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inverted_image = cv2.bitwise_not(inverted_image)
        inverted_image = cv2.cvtColor(inverted_image, cv2.COLOR_GRAY2BGR)
        
        return image, inverted_image

    def update_window(self):

        # If latest hasn't been inverted, invert the next
        if not self.latest_inverted:
            image_to_show = self.image_inverted
            
        # If latest has been inverted, display the regular one
        if self.latest_inverted:
            image_to_show = self.image

        self.latest_inverted = not self.latest_inverted

        # Shutdown if calibration file was created (calibration is done)
        if os.path.isfile(self.output_yaml):
            self.get_logger().info("Calibration complete. Shutting down aruco dispay node.")
            rclpy.shutdown()
            
        # This keeps the window responsive (for X11/Wayland event loops)
        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            self.get_logger().info("Window closed manually. Shutting down node.")
            rclpy.shutdown()
        else:
            cv2.imshow(self.window_name, image_to_show)
            cv2.waitKey(1)

    def destroy_node(self):
        # Clean up OpenCV window on shutdown
        cv2.destroyAllWindows()
        self.get_logger().info("ArUco display node shutting down.")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDisplayNode()
    rclpy.spin(node)
    node.destroy_node()


if __name__ == '__main__':
    main()

