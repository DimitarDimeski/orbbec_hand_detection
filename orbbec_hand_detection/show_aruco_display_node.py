#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np


class ArucoDisplayNode(Node):
    def __init__(self):
        super().__init__('aruco_display_node')
        self.get_logger().info('Starting ArUco display node...')

        # Parameters
        self.declare_parameter('width', 1920)
        self.declare_parameter('height', 1080)
        self.declare_parameter('marker_size', 400)
        self.declare_parameter('dictionary_id', cv2.aruco.DICT_4X4_50)

        # Get parameter values
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.marker_size = self.get_parameter('marker_size').value
        self.dictionary_id = self.get_parameter('dictionary_id').value

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.dictionary_id)

        # Create and show image
        self.image = self.create_image_with_aruco_markers()
        self.window_name = "Fullscreen ArUco"

        # Display image in fullscreen
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(self.window_name, self.image)

        # Create a timer to periodically check shutdown
        self.timer = self.create_timer(0.1, self.update_window)
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

        return image

    def update_window(self):
        # This keeps the window responsive (for X11/Wayland event loops)
        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            self.get_logger().info("Window closed manually. Shutting down node.")
            rclpy.shutdown()
        else:
            cv2.imshow(self.window_name, self.image)
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
    rclpy.shutdown()


if __name__ == '__main__':
    main()
