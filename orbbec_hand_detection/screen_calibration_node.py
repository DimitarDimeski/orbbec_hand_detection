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
        self.declare_parameter('num_samples_avg', 10)
        self.declare_parameter('output_yaml', 'calibration.yml')
        self.declare_parameter('rotate_image', False)

        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.depth_info_topic = self.get_parameter('depth_info_topic').value
        self.output_yaml = self.get_parameter('output_yaml').value
        self.num_samples_avg = self.get_parameter('num_samples_avg').value
        self.rotate_image = self.get_parameter('rotate_image').value

        # Subscriptions
        self.rgb_sub = self.create_subscription(Image, self.rgb_topic, self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.depth_info_sub = self.create_subscription(CameraInfo, self.depth_info_topic, self.depth_info_callback, 10)

        self.depth_intrinsics_received = False
        self.depth_image = None
        self.depth_K = None
        self.num_samples = 0
        self.filtered_depth = []
        self.corners = None
        self.screen_corners = None

        # ArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # Boolean to invert the image half of the time
        self.latest_inverted = False

        self.get_logger().info('Screen Plane Calibrator Node Started.')

    def depth_info_callback(self, msg: CameraInfo):
        if not self.depth_intrinsics_received:
            K = msg.k
            self.depth_K = {
                'fx': float(K[0]),
                'fy': float(K[4]),
                'cx': float(K[2]),
                'cy': float(K[5])
            }
            self.depth_intrinsics_received = True
            self.get_logger().info(f"Depth intrinsics received: {self.depth_K}")
            self.destroy_subscription(self.depth_info_sub)

    def depth_callback(self, msg: Image):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        
        # Rotate image if not inline
        if self.rotate_image:
            self.depth_image = cv2.rotate(self.depth_image, cv2.ROTATE_180)
		

    def rgb_callback(self, msg: Image):
        if self.depth_image is None or not self.depth_intrinsics_received:
            return

        rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Rotate image if not inline
        if self.rotate_image:
            rgb = cv2.rotate(rgb, cv2.ROTATE_180)

        # Only perform the code in this block once, after that skip to avoid unnecessary computation
        if self.num_samples == 0:
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            
            # Invert the image if previous hasn't been inverted
            if not self.latest_inverted:
                gray = cv2.bitwise_not(gray)

            self.latest_inverted = not self.latest_inverted
            
            self.corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

            if ids is not None:
                self.get_logger().info(f"Detected {len(ids)} markers.")
                
            if ids is None or len(ids) < 4:
                self.get_logger().warn("Not all 4 ArUco markers detected.")
                return

            # Flatten IDs for easier handling
            ids = ids.flatten()
            
            # Dictionary to hold marker corners
            marker_dict = {id_: c.reshape((4, 2)) for id_, c in zip(ids, self.corners)}

            # Compute outermost edges of screen
            all_points = np.concatenate(list(marker_dict.values()), axis=0)
            x_min, y_min = np.min(all_points, axis=0)
            x_max, y_max = np.max(all_points, axis=0)

            # Define corners in consistent order (TL, TR, BL, BR)
            x1, y1 = int(x_min), int(y_min)     # Top-left
            x2, y2 = int(x_max), int(y_min)     # Top-right
            x3, y3 = int(x_min), int(y_max)     # Bottom-left
            x4, y4 = int(x_max), int(y_max)     # Bottom-right

            self.screen_corners = [[x1, y1],
                              [x2, y2],
                              [x3, y3],
                              [x4, y4]]

        # If number of needed samples is reached average the depth at the three points and write to file
        if self.num_samples == self.num_samples_avg:
            self.filtered_depth = self.filtered_depth / self.num_samples_avg

            self.get_logger().info(f"Filtered depth: {self.filtered_depth}")

            # Define the plane from three points (top-left, top-right, bottom-left)
            A, B, C, D = self.define_plane_from_points(self.filtered_depth[0], self.filtered_depth[1], self.filtered_depth[2])

            # Save to YAML
            self.save_to_yaml(A, B, C, D, rgb.shape, self.depth_image.shape, self.depth_K, self.screen_corners)
            self.get_logger().info(f"Calibration parameters saved to {self.output_yaml}")

            # Stop after calibration
            self.destroy_subscription(self.rgb_sub)
            self.destroy_subscription(self.depth_sub)
            self.get_logger().info("Calibration complete. Node will shut down in 2 seconds...")
            self.create_timer(2.0, lambda: rclpy.shutdown())

        # Gather depth information at aruco marker points
        centers_3d = []
        for i, corner in enumerate(self.corners):
            c = corner[0]
            cx = int(np.mean(c[:, 0]))
            cy = int(np.mean(c[:, 1]))
            z = float(self.depth_image[cy, cx])
            if z <= 0:
                self.get_logger().warn(f"Invalid depth for marker {i}.")
                return

            x = (cx - self.depth_K['cx']) * z / self.depth_K['fx']
            y = (cy - self.depth_K['cy']) * z / self.depth_K['fy']
            centers_3d.append(np.array([x, y, z]))

        if len(centers_3d) < 4:
            self.get_logger().warn("Less than 4 valid 3D points found.")
            return

        if self.num_samples == 0:
            self.filtered_depth = np.array(centers_3d)
        else:
            self.filtered_depth += np.array(centers_3d)

        self.get_logger().info(f"Depth at corners: {centers_3d}")

        self.num_samples += 1


    def define_plane_from_points(self, p1, p2, p3):
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        A, B, C = normal
        D = -np.dot(normal, p1)
        return (A, B, C, D)

    def save_to_yaml(self, A, B, C, D, rgb_shape, depth_shape, K, screen_corners):
        data = {
            'plane': {'A': float(A), 'B': float(B), 'C': float(C), 'D': float(D)},
            'rgb_resolution': {'width': rgb_shape[1], 'height': rgb_shape[0]},
            'depth_resolution': {'width': depth_shape[1], 'height': depth_shape[0]},
            'depth_K': K,
            'screen': {
                'x1': screen_corners[0][0], 'y1': screen_corners[0][1],
                'x2': screen_corners[1][0], 'y2': screen_corners[1][1],
                'x3': screen_corners[2][0], 'y3': screen_corners[2][1],
                'x4': screen_corners[3][0], 'y4': screen_corners[3][1],
            }
        }

        with open(self.output_yaml, 'w') as f:
            yaml.dump(data, f)

        self.get_logger().info(f"Calibration data written to {self.output_yaml}")

def main(args=None):
    rclpy.init(args=args)
    node = ScreenPlaneCalibrator()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()

