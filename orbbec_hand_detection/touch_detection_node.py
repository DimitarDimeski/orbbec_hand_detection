#!/usr/bin/env python3
"""
Touch Detection Node

This node synchronizes depth images and hand detections, then computes
touch detections using vectorized operations for efficiency.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml
import os
import logging
from message_filters import Subscriber, ApproximateTimeSynchronizer

from orbbec_hand_detection_msgs.msg import HandDetection, TouchDetection, TouchEvent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TouchDetectionNode(Node):
    """Node that detects touches by combining hand detections with depth data."""

    def __init__(self):
        super().__init__('touch_detection_node')

        # Declare ROS parameters
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('detection_topic', '/hand_detections')
        self.declare_parameter('touch_detections_topic', '/touch_detections')
        self.declare_parameter('calib_yaml_path', '')
        self.declare_parameter('screen_width', 1920)
        self.declare_parameter('screen_height', 1080)
        self.declare_parameter('depth_threshold', 0.01)
        self.declare_parameter('depth_threshold_lower', 0.5)
        self.declare_parameter('depth_threshold_upper', 0.5)
        self.declare_parameter('top_offset', 0)
        self.declare_parameter('bottom_offset', 0)
        self.declare_parameter('left_offset', 0)
        self.declare_parameter('right_offset', 0)
        self.declare_parameter('rotate_image', False)

        # Read parameter values
        self.depth_topic = self.get_parameter('depth_topic').value
        self.detection_topic = self.get_parameter('detection_topic').value
        self.touch_detections_topic = self.get_parameter('touch_detections_topic').value
        calib_path = self.get_parameter('calib_yaml_path').get_parameter_value().string_value
        self.screen_width = self.get_parameter('screen_width').value
        self.screen_height = self.get_parameter('screen_height').value
        self.depth_threshold = self.get_parameter('depth_threshold').value
        self.depth_threshold_lower = self.get_parameter('depth_threshold_lower').value
        self.depth_threshold_upper = self.get_parameter('depth_threshold_upper').value
        self.top_offset = self.get_parameter('top_offset').value
        self.bottom_offset = self.get_parameter('bottom_offset').value
        self.left_offset = self.get_parameter('left_offset').value
        self.right_offset = self.get_parameter('right_offset').value
        self.rotate_image = self.get_parameter('rotate_image').value

        if not calib_path or not os.path.exists(calib_path):
            raise FileNotFoundError(f"Calibration YAML not found or invalid path: {calib_path}")

        # Load calibration data
        with open(calib_path, 'r') as f:
            calib_data = yaml.safe_load(f)
        self.get_logger().info(f"Loaded calibration data from {calib_path}")

        # Plane parameters
        self.A = calib_data['plane']['A']
        self.B = calib_data['plane']['B']
        self.C = calib_data['plane']['C']
        self.D = calib_data['plane']['D']

        # Screen corners
        self.x1 = calib_data['screen']['x1']
        self.y1 = calib_data['screen']['y1']
        self.x2 = calib_data['screen']['x2']
        self.y2 = calib_data['screen']['y2']
        self.x3 = calib_data['screen']['x3']
        self.y3 = calib_data['screen']['y3']
        self.x4 = calib_data['screen']['x4']
        self.y4 = calib_data['screen']['y4']

        # Parameters for the detection surface in the image
        self.surface_width = self.x2 - self.x1
        self.surface_height = self.y4 - self.y1

        # Depth intrinsics
        self.depth_K = calib_data['depth_K']
        self.fx = self.depth_K['fx']
        self.fy = self.depth_K['fy']
        self.cx = self.depth_K['cx']
        self.cy = self.depth_K['cy']

        # RGB image dimensions
        self.image_width = calib_data['rgb_resolution']['width']
        self.image_height = calib_data['rgb_resolution']['height']

        # Precompute plane denominator for efficiency
        self.plane_denominator = np.sqrt(self.A**2 + self.B**2 + self.C**2)

        # Touch ID counter
        self.touch_id_counter = 0

        # Subscribers
        self.depth_sub = Subscriber(self, Image, self.depth_topic)
        self.detection_sub = Subscriber(self, HandDetection, self.detection_topic)

        # Approximate time synchronizer
        self.ts = ApproximateTimeSynchronizer(
            [self.depth_sub, self.detection_sub],
            queue_size=10,
            slop=0.01
        )
        self.ts.registerCallback(self.synchronized_callback)

        # Publisher
        self.touch_pub = self.create_publisher(
            TouchDetection,
            self.touch_detections_topic,
            10
        )

        self.bridge = CvBridge()

        logger.info(f"RGB Resolution: {self.image_width} x {self.image_height}")
        logger.info(f"Loaded plane: A={self.A}, B={self.B}, C={self.C}, D={self.D}")
        logger.info(f"Screen borders: ({self.x1},{self.y1}),({self.x2},{self.y2}),"
                   f"({self.x3},{self.y3}),({self.x4},{self.y4})")

    def depth_to_point_vectorized(self, u, v, depths):
        """
        Convert depth map pixels and their depth values into 3D space (vectorized).
        
        Args:
            u: X coordinates (array)
            v: Y coordinates (array)
            depths: Depth values (array)
        
        Returns:
            Array of 3D points [N x 3]
        """
        z = depths
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return np.column_stack([x, y, z])

    def distance_from_plane_vectorized(self, points):
        """
        Calculate perpendicular distance from 3D points to the plane (vectorized).
        
        Args:
            points: Array of 3D points [N x 3]
        
        Returns:
            Array of distances
        """
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        numerator = np.abs(self.A * x + self.B * y + self.C * z + self.D)
        return numerator / self.plane_denominator

    def detect_touch_vectorized(self, finger_points, threshold=0.01):
        """
        Check if finger points are close to the plane (vectorized).
        
        Args:
            finger_points: Array of 3D points [N x 3]
            threshold: Distance threshold in meters
        
        Returns:
            Tuple of (is_touch array, distances array)
        """
        distances = self.distance_from_plane_vectorized(finger_points)
        lower_bound = threshold - self.depth_threshold_lower
        upper_bound = threshold + self.depth_threshold_upper
        is_touch = (distances > lower_bound) & (distances < upper_bound)
        return is_touch, distances

    def get_depth_patch_median(self, depth_image, x, y, patch_size=5):
        """
        Get median depth value from a patch around the given coordinates.
        
        Args:
            depth_image: Depth image array
            x: X coordinate (int)
            y: Y coordinate (int)
            patch_size: Size of patch (default 5x5)
        
        Returns:
            Median depth value
        """
        half_size = patch_size // 2
        y_min = max(0, int(y) - half_size)
        y_max = min(depth_image.shape[0], int(y) + half_size + 1)
        x_min = max(0, int(x) - half_size)
        x_max = min(depth_image.shape[1], int(x) + half_size + 1)
        
        patch = depth_image[y_min:y_max, x_min:x_max]
        valid_patch = patch[~np.isnan(patch) & (patch > 0)]
        
        if len(valid_patch) > 0:
            return np.median(valid_patch)
        else:
            return np.nan

    def map_point_to_screen_resolution(self, point, bbox_width, bbox_height, target_width=1920, target_height=1080):
        """Map point from image space to screen resolution."""
        x, y = point
        scale_x = target_width / bbox_width
        scale_y = target_height / bbox_height
        return int(x * scale_x), int(y * scale_y)

    def synchronized_callback(self, depth_msg, detection_msg):
        """Process synchronized depth and detection messages."""
        try:
            # Convert depth image
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')

            # Rotate if needed
            if self.rotate_image:
                depth_image = cv2.rotate(depth_image, cv2.ROTATE_180)

            # Get detection coordinates
            x_coords = np.array(detection_msg.x_coordinates)
            y_coords = np.array(detection_msg.y_coordinates)

            if len(x_coords) == 0:
                # No detections, publish empty message
                touch_msg = TouchDetection()
                touch_msg.header = detection_msg.header
                touch_msg.touches = []
                self.touch_pub.publish(touch_msg)
                return

            # Filter detections within screen bounds
            valid_mask = (
                (x_coords >= self.x1) & (x_coords <= self.x2) &
                (y_coords >= self.y1) & (y_coords <= self.y4)
            )

            if not np.any(valid_mask):
                # No valid detections, publish empty message
                touch_msg = TouchDetection()
                touch_msg.header = detection_msg.header
                touch_msg.touches = []
                self.touch_pub.publish(touch_msg)
                return

            # Filter to valid coordinates
            valid_x = x_coords[valid_mask]
            valid_y = y_coords[valid_mask]

            # Get depth values for each detection (vectorized where possible)
            depths = np.array([
                self.get_depth_patch_median(depth_image, x, y)
                for x, y in zip(valid_x, valid_y)
            ])

            # Filter out invalid depths
            valid_depth_mask = ~np.isnan(depths) & (depths > 0)
            if not np.any(valid_depth_mask):
                touch_msg = TouchDetection()
                touch_msg.header = detection_msg.header
                touch_msg.touches = []
                self.touch_pub.publish(touch_msg)
                return

            valid_x = valid_x[valid_depth_mask]
            valid_y = valid_y[valid_depth_mask]
            depths = depths[valid_depth_mask]

            # Convert to 3D points (vectorized)
            finger_points = self.depth_to_point_vectorized(valid_x, valid_y, depths)

            # Detect touches (vectorized)
            is_touch, distances = self.detect_touch_vectorized(
                finger_points,
                threshold=self.depth_threshold
            )

            # Create touch events
            touch_events = []
            for i, (x, y, depth, dist, touch) in enumerate(zip(
                valid_x, valid_y, depths, distances, is_touch
            )):
                # Map to screen coordinates
                point = (int(x - self.x1), int(y - self.y1))
                screen_width_offsetted = self.screen_width - self.left_offset - self.right_offset
                screen_height_offsetted = self.screen_height - self.top_offset - self.bottom_offset

                scaled_x, scaled_y = self.map_point_to_screen_resolution(
                    point, self.surface_width, self.surface_height,
                    screen_width_offsetted, screen_height_offsetted
                )

                scaled_x = scaled_x + self.left_offset
                scaled_y = scaled_y + self.top_offset

                # Create touch event
                touch_event = TouchEvent()
                touch_event.touch_id = self.touch_id_counter
                self.touch_id_counter += 1
                touch_event.x = float(scaled_x)
                touch_event.y = float(scaled_y)
                touch_event.depth = float(depth)
                touch_event.distance = float(dist)
                touch_event.is_touch = bool(touch)

                touch_events.append(touch_event)

            # Create and publish touch detection message
            touch_msg = TouchDetection()
            touch_msg.header = detection_msg.header  # Preserve original timestamp
            touch_msg.touches = touch_events

            self.touch_pub.publish(touch_msg)

            if len(touch_events) > 0:
                logger.debug(f"Published {len(touch_events)} touch detection(s)")

        except Exception as e:
            logger.error(f"Error processing synchronized messages: {e}", exc_info=True)


def main(args=None):
    rclpy.init(args=args)
    node = TouchDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

