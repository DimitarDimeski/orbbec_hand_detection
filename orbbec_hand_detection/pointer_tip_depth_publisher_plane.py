#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import json
import yaml
import random
import time
import sys
import asyncio
import numpy as np
import logging
from datetime import datetime
import threading

from message_filters import Subscriber, ApproximateTimeSynchronizer

import mediapipe as mp

from cierra_event_bus import (
    CierraEventBus, Priority, AIInferenceEvents, 
    CustomEventPatterns, CustomEventType, InteractionEvents,
    TouchEventBuilder, GestureEventBuilder
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PointerTipDepthPlanePublisher(Node):
    def __init__(self, loop):
        super().__init__('pointer_tip_depth_plane')

        self.async_loop = loop  # Store asyncio event loop


        # Declare ROS parameters
        self.declare_parameter('calib_yaml_path', '')
        self.declare_parameter('nats_url', 'nats://localhost:4223')
        self.declare_parameter('screen_width', 1920)
        self.declare_parameter('screen_height', 1080)
        self.declare_parameter('depth_threshold', 0.01)
        self.declare_parameter('depth_threshold_lower', 0.5)
        self.declare_parameter('depth_threshold_upper', 0.5) 
        self.declare_parameter('min_detection_confidence', 0.5)
        self.declare_parameter('min_tracking_confidence', 0.5)
        self.declare_parameter('top_offset', 0)
        self.declare_parameter('bottom_offset', 0)
        self.declare_parameter('left_offset', 0)
        self.declare_parameter('right_offset', 0)
        self.declare_parameter('contrast', 1.5)
        self.declare_parameter('brightness', 20)
        self.declare_parameter('rotate_image', False)
        self.declare_parameter('use_depth', False)
        self.declare_parameter('use_grayscale', False)
        self.declare_parameter('adjust_contrast_brightness', False)
        self.declare_parameter('static_image_mode', False)
        self.declare_parameter('max_num_hands', 10)

        # Read parameter values
        calib_path = self.get_parameter('calib_yaml_path').value
        self.nats_url = self.get_parameter('nats_url').value
        self.screen_width = self.get_parameter('screen_width').value
        self.screen_height = self.get_parameter('screen_height').value
        self.depth_threshold = self.get_parameter('depth_threshold').value
        self.depth_threshold_lower = self.get_parameter('depth_threshold_lower').value
        self.depth_threshold_upper = self.get_parameter('depth_threshold_upper').value
        self.min_detection_conf = self.get_parameter('min_detection_confidence').value
        self.min_tracking_conf = self.get_parameter('min_tracking_confidence').value
        self.top_offset = self.get_parameter('top_offset').value
        self.bottom_offset = self.get_parameter('bottom_offset').value
        self.left_offset = self.get_parameter('left_offset').value
        self.right_offset = self.get_parameter('right_offset').value
        self.contrast = self.get_parameter('contrast').value
        self.brightness = self.get_parameter('brightness').value
        self.rotate_image = self.get_parameter('rotate_image').value
        self.use_depth = self.get_parameter('use_depth').value
        self.use_grayscale = self.get_parameter('use_grayscale').value
        self.adjust_contrast_brightness = self.get_parameter('adjust_contrast_brightness').value
        self.static_image_mode = self.get_parameter('static_image_mode').value
        self.max_num_hands = self.get_parameter('max_num_hands').value
        
        if not calib_path or not os.path.exists(calib_path):
            raise FileNotFoundError(f"Calibration YAML not found or invalid path: {calib_path}")
        
        # --- Load calibration data ---
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
        
        # Get your 4 screen corner points from calib_data
        pts = np.array([
        [calib_data['screen']['x1'], calib_data['screen']['y1']],
        [calib_data['screen']['x2'], calib_data['screen']['y2']],
        [calib_data['screen']['x3'], calib_data['screen']['y3']],
        [calib_data['screen']['x4'], calib_data['screen']['y4']]
        ], dtype=np.int32)
        
        # Define destination points in normalized image frame
        pts_dst = np.array([
   	    [0, 0],   # top-left
        [1, 0],   # top-right
        [1, 1],   # bottom-left
        [0, 1]    # bottom-right
        ], dtype=np.float32)
	
        # Compute homography
        self.transform_image_to_screen, _ = cv2.findHomography(pts, pts_dst)
	
        # Reshape to match contour format (n,1,2)
        self.contour = pts.reshape((-1, 1, 2))

        # Parameters for the detection surface in the image
        self.surface_width = self.x2 - self.x1
        self.surface_height = self.y4 - self.y1

        # Depth intrinsics
        self.depth_K = calib_data['depth_K']

        # RGB image dimensions
        self.image_width = calib_data['rgb_resolution']['width']
        self.image_height = calib_data['rgb_resolution']['height']

        
        # Subscribers
        self.rgb_sub = Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/camera/depth/image_raw')

        # Approximate sync
        self.ts = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.image_callback)

        self.publisher_ = self.create_publisher(Image, '/image_landmarked', 10)
        self.bridge = CvBridge()

        # Mediapipe setup
        self.hands = mp.solutions.hands.Hands(
            min_detection_confidence=self.min_detection_conf,
            min_tracking_confidence=self.min_tracking_conf
        )


        self.touch_id_counter = 0

        logger.info(f"NATS URL: {self.nats_url}")
        logger.info(f"RGB Resolution: {self.image_width} x {self.image_height} x 3")
        logger.info(f"Loaded plane: A={self.A}, B={self.B}, C={self.C}, D={self.D}")
        logger.info(f"Screen borders: ({self.x1},{self.y1}),({self.x2},{self.y2}),({self.x3},{self.y3}),({self.x4},{self.y4})")


    async def connect_to_bus(self):
        try:
            logger.info("Connecting to bus... ")
            self.bus = CierraEventBus()
            await self.bus.connect()
            logger.info("Successfully connected to NATS server.")
        except Exception as e:
            logger.error(f"Failed to connect to NATS server: {e}")

    def generate_touch_id(self) -> str:
        return f"{self.touch_id_counter}"

    def map_point_to_screen_resolution(self, point, target_width=1920, target_height=1080):
      	"""
        Scale point to match screen resolution
      	"""
      	x, y = point
      	return int(x * target_width), int(y * target_height)

    def depth_to_point(self, u, v, depth, fx, fy, cx, cy):
        """
        Convert depth map pixel (u, v) and its depth value into 3D space (x, y, z).
        """
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.array([x, y, z])
        
    def distance_from_plane(self, point, A, B, C, D):
        """
        Calculate perpendicular distance from a 3D point to the plane.
        """
        x, y, z = point
        numerator = abs(A * x + B * y + C * z + D)
        denominator = np.sqrt(A**2 + B**2 + C**2)
        return numerator / denominator
    
    
    def detect_touch(self, finger_point, plane, threshold=0.01):
        """
        Check if the finger point is close to the plane.
        """
        A, B, C, D = plane
        dist = self.distance_from_plane(finger_point, A, B, C, D)
        
        is_touch = False
        
        if dist > (threshold - self.depth_threshold_lower) and dist < (threshold + self.depth_threshold_upper):
            is_touch = True 
        
        return is_touch, dist

    def image_callback(self, rgb_msg, depth_msg):


        # Access intrinsics if needed
        fx = self.depth_K['fx']
        fy = self.depth_K['fy']
        cx = self.depth_K['cx']
        cy = self.depth_K['cy']
            
        rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        
        # Rotate images 180 degress if not inline
        if self.rotate_image:
            rgb_image = cv2.rotate(rgb_image, cv2.ROTATE_180)
            depth_image = cv2.rotate(depth_image, cv2.ROTATE_180)
            
        rgb_image_for_mediapipe = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            
        if self.use_depth:
            depth_8u = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_8u = np.uint8(depth_8u)
        
            rgb_image_for_mediapipe = cv2.cvtColor(depth_8u, cv2.COLOR_GRAY2RGB)
            
        if self.adjust_contrast_brightness:
            rgb_image_for_mediapipe = cv2.convertScaleAbs(rgb_image_for_mediapipe, alpha=self.contrast, beta=self.brightness)
        
        if self.use_grayscale:
            gray = cv2.cvtColor(rgb_image_for_mediapipe, cv2.COLOR_RGB2GRAY)
            rgb_image_for_mediapipe = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            
        results = self.hands.process(rgb_image_for_mediapipe)

        plane = (self.A, self.B, self.C, self.D)
	
	# Visualize borders of screen
	# Red top edge
        cv2.line(rgb_image, (self.x1, self.y1), (self.x2 , self.y2), (0, 0, 255), 3)
        # Green other edges
        cv2.line(rgb_image, (self.x2, self.y2), (self.x3 , self.y3), (0, 255, 0), 3)
        cv2.line(rgb_image, (self.x3, self.y3), (self.x4 , self.y4), (0, 255, 0), 3)
        cv2.line(rgb_image, (self.x4, self.y4), (self.x1 , self.y1), (0, 255, 0), 3)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_tip_x = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x * self.image_width)
                index_tip_y = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y * self.image_height)
                
                mp.solutions.drawing_utils.draw_landmarks(rgb_image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                depth = 0
		
		        # Check if point is inside the screen area
                point_inside_screen = cv2.pointPolygonTest(self.countour, (index_tip_x, index_tip_y), measureDist=False)
		
                if point_inside_screen >= 0:
                
                    # Get a patch of depth values around finger tip in order to reduce noise 
                    finger_patch = depth_image[index_tip_y-2:index_tip_y+3, index_tip_x-2:index_tip_x+3]
                    
                    finger_depth = np.median(finger_patch)

                    finger_point = self.depth_to_point(index_tip_x, index_tip_y, finger_depth, fx, fy, cx, cy)
                    is_touch, distance = self.detect_touch(finger_point, plane, threshold=self.depth_threshold)
                    
                    logger.info(f'Index finger tip coordinates: ({index_tip_x}, {index_tip_y}), Depth: {finger_depth:.3f} m, Distance: {distance}, Finger Point: {finger_point} Touch: {is_touch}')
                    cv2.putText(rgb_image, f'{finger_depth:.2f}mm, Screen Distance: {distance}', (index_tip_x, index_tip_y), font, font_scale, (0, 255, 0), thickness)

                    if is_touch :

                        cv2.putText(rgb_image, 'TOUCH', (0, 0), font, 2, (0, 255, 0), 4)
                        
                        point = np.array([[[index_tip_x, index_tip_y]]], dtype=np.float32)
                        
                        
                        screen_point_norm = cv2.perspectiveTransform(point, self.transform_image_to_screen)[0][0]
			
			            # Account for offset in screen cropping
                        screen_width_offsetted = self.screen_width - self.left_offset - self.right_offset
                        screen_height_offsetted = self.screen_height - self.top_offset - self.bottom_offset    
                                          
                        screen_point_x, screen_point_y = self.map_point_to_screen_resolution(screen_point_norm, screen_width_offsetted, screen_height_offsetted)
                        
                        screen_point_x = screen_point_x + self.left_offset
                        screen_point_y = screen_point_y + self.top_offset

                        self.touch_id = self.generate_touch_id()
                        self.touch_id_counter += 1

                        event_data = {
                            'touchId': self.touch_id,
                            'x': screen_point_x,
                            'y': screen_point_y,
                            'pressure': 0.8,
                            'target': 'smart_table_surface',
                            'userId': 'test_user',
                            'sessionId': f'test_session_{int(time.time() * 1000)}'
                        }

                        message = {
                            'id': f"{self.touch_id_counter}",
                            'key': 'interaction.touch.down',
                            'data': event_data,
                            'publisher': None,
                            'priority': Priority.MEDIUM,
                            'created': datetime.now().isoformat()
                        }

                        payload = json.dumps(message).encode()

                        if hasattr(self, 'bus') and self.bus and self.bus.nats_client:
                            # Use asyncio to schedule coroutine correctly
                            asyncio.run_coroutine_threadsafe(
                            self.bus.nats_client.publish('interaction.touch.down', payload),
                            self.async_loop
                            )
                            logger.info(f"👇 TouchDown: {self.touch_id} at ({screen_point_x}, {screen_point_y})")
                        
        msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding='bgr8')
        self.publisher_.publish(msg)
        self.get_logger().info('Published image frame')

    def destroy_node(self):
        if hasattr(self, 'bus'):
            # Properly schedule async disconnect
            asyncio.run_coroutine_threadsafe(self.bus.disconnect(), self.async_loop)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    # Create a new asyncio event loop and run it in a background thread
    asyncio_loop = asyncio.new_event_loop()

    def start_asyncio_loop(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=start_asyncio_loop, args=(asyncio_loop,), daemon=True)
    thread.start()

    node = PointerTipDepthPlanePublisher(loop=asyncio_loop)

    # Schedule the async connect_to_bus coroutine
    asyncio.run_coroutine_threadsafe(node.connect_to_bus(), asyncio_loop)

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():  # Prevent double shutdown
            rclpy.shutdown()
        asyncio_loop.call_soon_threadsafe(asyncio_loop.stop)


if __name__ == '__main__':
    main()

