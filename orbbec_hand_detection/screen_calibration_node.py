#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import yaml
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo

class ScreenPlaneCalibrator(Node):
    def __init__(self):
        super().__init__('screen_plane_calibrator')
        self.bridge = CvBridge()

        # Parameters
        self.declare_parameter('rgb_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('depth_info_topic', '/camera/depth/camera_info')
        self.declare_parameter('output_yaml', 'screen_plane.yaml')

        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.depth_info_topic = self.get_parameter('depth_info_topic').value
        self.output_yaml = self.get_parameter('output_yaml').value

        # Subscriptions
        self.rgb_sub = self.create_subscription(Image, self.rgb_topic, self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.depth_info_sub = self.create_subscription(CameraInfo, self.depth_info_topic, self.depth_info_callback, 10)

        self.depth_intrinsics_received = False
        self.depth_image = None
        self.depth_K = None

        # ArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

        self.get_logger().info('Screen Plane Calibrator Node Started.')

    def depth_info_callback(self, msg: CameraInfo):
        if not self.depth_intrinsics_received:
            K = msg.k
            self.depth_K = {
                'fx': K[0],
                'fy': K[4],
                'cx': K[2],
                'cy': K[5]
            }
            self.depth_intrinsics_received = True
            self.get_logger().info(f"Depth intrinsics received: {self.depth_K}")
            self.destroy_subscription(self.depth_info_sub)

    def depth_callback(self, msg: Image):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def rgb_callback(self, msg: Image):
        if self.depth_image is None or not self.depth_intrinsics_received:
            return

        rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is None or len(ids) < 4:
            self.get_logger().warn("Not all 4 ArUco markers detected.")
            return

        centers_3d = []
        for i, corner in enumerate(corners):
            c = corner[0]
            cx = int(np.mean(c[:, 0]))
            cy = int(np.mean(c[:, 1]))
            z = float(self.depth_image[cy, cx]) / 1000.0  # assuming mm to meters
            if z <= 0:
                self.get_logger().warn(f"Invalid depth for marker {i}.")
                return

            x = (cx - self.depth_K['cx']) * z / self.depth_K['fx']
            y = (cy - self.depth_K['cy']) * z / self.depth_K['fy']
            centers_3d.append(np.array([x, y, z]))

        if len(centers_3d) < 4:
            self.get_logger().warn("Less than 4 valid 3D points found.")
            return

        # Define the plane from three points (top-left, top-right, bottom-left)
        A, B, C, D = self.define_plane_from_points(centers_3d[0], centers_3d[1], centers_3d[2])

        # Save to YAML
        self.save_to_yaml(A, B, C, D, rgb.shape, self.depth_image.shape, self.depth_K)
        self.get_logger().info(f"Plane parameters saved to {self.output_yaml}")

        # Stop after calibration
        self.destroy_subscription(self.rgb_sub)
        self.destroy_subscription(self.depth_sub)
        self.get_logger().info("Calibration complete. Node will shut down in 2 seconds...")
        self.create_timer(2.0, lambda: rclpy.shutdown())
        

    def define_plane_from_points(self, p1, p2, p3):
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        A, B, C = normal
        D = -np.dot(normal, p1)
        return (A, B, C, D)

    def save_to_yaml(self, A, B, C, D, rgb_shape, depth_shape, K):
        data = {
            'plane': {'A': float(A), 'B': float(B), 'C': float(C), 'D': float(D)},
            'rgb_resolution': {'width': rgb_shape[1], 'height': rgb_shape[0]},
            'depth_resolution': {'width': depth_shape[1], 'height': depth_shape[0]},
            'depth_intrinsics': K
        }

        with open(self.output_yaml, 'w') as f:
            yaml.dump(data, f)

        self.get_logger().info(f"Calibration data written to {self.output_yaml}")

def main(args=None):
    rclpy.init(args=args)
    node = ScreenPlaneCalibrator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
