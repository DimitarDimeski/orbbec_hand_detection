#!/usr/bin/env python3
"""
MediaPipe Detection Node

This node subscribes to RGB images, processes them with MediaPipe to detect hands,
and publishes pointer finger tip coordinates in image space.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import mediapipe as mp
import logging

from orbbec_hand_detection_msgs.msg import HandDetection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MediaPipeDetectionNode(Node):
    """Node that detects hands using MediaPipe and publishes pointer finger coordinates."""

    def __init__(self):
        super().__init__('mediapipe_detection_node')

        # Declare ROS parameters
        self.declare_parameter('rgb_topic', '/camera/color/image_raw')
        self.declare_parameter('detection_topic', '/hand_detections')
        self.declare_parameter('min_detection_confidence', 0.5)
        self.declare_parameter('min_tracking_confidence', 0.5)
        self.declare_parameter('static_image_mode', False)
        self.declare_parameter('max_num_hands', 10)
        self.declare_parameter('rotate_image', False)
        self.declare_parameter('model_complexity', 1)
        self.declare_parameter('use_depth', False)
        self.declare_parameter('use_grayscale', False)
        self.declare_parameter('adjust_contrast_brightness', False)
        self.declare_parameter('contrast', 0)
        self.declare_parameter('brightness', 0)
        self.declare_parameter('debug', False)

        # Read parameter values
        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.detection_topic = self.get_parameter('detection_topic').value
        self.min_detection_conf = self.get_parameter('min_detection_confidence').value
        self.min_tracking_conf = self.get_parameter('min_tracking_confidence').value
        self.static_image_mode = self.get_parameter('static_image_mode').value
        self.max_num_hands = self.get_parameter('max_num_hands').value
        self.rotate_image = self.get_parameter('rotate_image').value
        self.model_complexity = self.get_parameter('model_complexity').value
        self.use_depth = self.get_parameter('use_depth').value
        self.use_grayscale = self.get_parameter('use_grayscale').value
        self.adjust_contrast_brightness = self.get_parameter('adjust_contrast_brightness').value
        self.contrast = self.get_parameter('contrast').value
        self.brightness = self.get_parameter('brightness').value
        self.debug = self.get_parameter('debug').value

        # Subscriber
        self.rgb_sub = self.create_subscription(
            Image,
            self.rgb_topic,
            self.image_callback,
            10
        )

        # Publisher
        self.detection_pub = self.create_publisher(
            HandDetection,
            self.detection_topic,
            10
        )

        # Optional debug publisher for images with landmarks
        self.image_pub = None
        if self.debug:
            self.image_pub = self.create_publisher(
                Image,
                '/image_landmarked',
                10
            )
            logger.info("Debug mode enabled: Publishing images with landmarks to /image_landmarked")

        self.bridge = CvBridge()

        # MediaPipe setup
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=self.static_image_mode,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_conf,
            min_tracking_confidence=self.min_tracking_conf,
            model_complexity=self.model_complexity
        )

        logger.info("MediaPipe Detection Node initialized")

    def image_callback(self, rgb_msg):
        """Process RGB image and publish hand detections."""
        try:
            # Convert ROS image to OpenCV
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')

            # Get image dimensions (after potential rotation)
            image_height, image_width = rgb_image.shape[:2]

            # Rotate image if needed
            if self.rotate_image:
                rgb_image = cv2.rotate(rgb_image, cv2.ROTATE_180)
                # Update dimensions after rotation (they stay the same, but for clarity)
                image_height, image_width = rgb_image.shape[:2]

            # Prepare image for MediaPipe
            rgb_image_for_mediapipe = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

            # Apply preprocessing if needed
            if self.use_depth:
                # This would require depth image, but we only have RGB here
                # Skip this preprocessing in this node
                pass

            if self.adjust_contrast_brightness:
                rgb_image_for_mediapipe = cv2.convertScaleAbs(
                    rgb_image_for_mediapipe,
                    alpha=self.contrast,
                    beta=self.brightness
                )

            if self.use_grayscale:
                gray = cv2.cvtColor(rgb_image_for_mediapipe, cv2.COLOR_RGB2GRAY)
                rgb_image_for_mediapipe = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

            # Process with MediaPipe
            results = self.hands.process(rgb_image_for_mediapipe)

            # Create detection message
            detection_msg = HandDetection()
            detection_msg.header = rgb_msg.header  # Preserve original timestamp

            # Extract pointer finger tip coordinates
            x_coords = []
            y_coords = []

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get index finger tip coordinates
                    index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                    x_coords.append(index_tip.x * image_width)
                    y_coords.append(index_tip.y * image_height)

            detection_msg.x_coordinates = x_coords
            detection_msg.y_coordinates = y_coords

            # Publish detection message
            self.detection_pub.publish(detection_msg)

            # Debug: Draw landmarks and publish image
            if self.debug and self.image_pub is not None:
                # Draw hand landmarks on the original BGR image
                annotated_image = rgb_image.copy()
                # Convert to RGB for MediaPipe drawing utilities
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand connections (MediaPipe drawing utils work with RGB)
                        mp.solutions.drawing_utils.draw_landmarks(
                            annotated_image_rgb,
                            hand_landmarks,
                            mp.solutions.hands.HAND_CONNECTIONS,
                            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                            mp.solutions.drawing_styles.get_default_hand_connections_style()
                        )
                        
                        # Draw pointer finger tip with a circle
                        index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                        tip_x = int(index_tip.x * image_width)
                        tip_y = int(index_tip.y * image_height)
                        # Draw circles in RGB (green = (0, 255, 0) in RGB)
                        cv2.circle(annotated_image_rgb, (tip_x, tip_y), 10, (0, 255, 0), -1)
                        cv2.circle(annotated_image_rgb, (tip_x, tip_y), 15, (0, 255, 0), 2)
                
                # Convert back to BGR for publishing
                annotated_image = cv2.cvtColor(annotated_image_rgb, cv2.COLOR_RGB2BGR)
                
                # Convert back to ROS Image message and publish
                try:
                    image_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
                    image_msg.header = rgb_msg.header  # Preserve original timestamp
                    self.image_pub.publish(image_msg)
                except Exception as e:
                    logger.error(f"Error publishing debug image: {e}")

            if len(x_coords) > 0:
                logger.debug(f"Published {len(x_coords)} hand detection(s)")

        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)


def main(args=None):
    rclpy.init(args=args)
    node = MediaPipeDetectionNode()

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

