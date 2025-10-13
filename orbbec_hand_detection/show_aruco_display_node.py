#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np


class ArucoDisplayNode(Node):
    def __init__(self):
        super().__init__('aruco_display_node')
        self.get_logger().info('Starting ArUco display node...')

        # Parameters (can be set from launch if desired)
        self.declare_parameter('width', 1920)
        self.declare_parameter('height', 1080)
        self.declare_parameter('marker_size', 200)
        self.declare_parameter('dictionary_id', cv2.aruco.DICT_4X4_50)

        # Get parameter values
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.marker_size = self.get_parameter('marker_size').value
        self.dictionary_id = self.get_parameter('dictionary_id').value

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.dictionary_id)

        # Create and show image
        image = self.create_image_with_aruco_markers()
        self.display_image(image)

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

        return image

    def display_image(self, image):
        window_name = "Fullscreen ArUco"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(window_name, image)
        self.get_logger().info("ArUco markers displayed. Press any key to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.get_logger().info("Window closed, shutting down node.")
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDisplayNode()
    # We don't spin since the node terminates after displaying
    # rclpy.spin(node)  # Not needed
    # Just keep it alive until shutdown (cv2.waitKey handles blocking)
    # Node shuts down itself after closing the window


if __name__ == '__main__':
    main()
