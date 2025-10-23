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


class PointerTipDepthPublisher(Node):
    def __init__(self, loop):
        super().__init__('pointer_tip_depth')

        self.async_loop = loop  # Store asyncio event loop

        # Declare ROS parameters
        self.declare_parameter('calib_yaml_path', '')
        self.declare_parameter('nats_url', 'nats://localhost:4222')
        self.declare_parameter('screen_width', 1920)
        self.declare_parameter('screen_height', 1080)
        self.declare_parameter('depth_threshold', 0.01) # meters
        self.declare_parameter('min_detection_confidence', 0.5)
        self.declare_parameter('min_tracking_confidence', 0.5)

        # Read parameter values
        calib_path = self.get_parameter('calib_yaml_path').get_parameter_value().string_value
        self.nats_url = self.get_parameter('nats_url').value
        self.screen_width = self.get_parameter('screen_width').value
        self.screen_height = self.get_parameter('screen_height').value
        self.depth_threshold = self.get_parameter('depth_threshold').value
        self.min_detection_conf = self.get_parameter('min_detection_confidence').get_parameter_value().double_value
        self.min_tracking_conf = self.get_parameter('min_tracking_confidence').get_parameter_value().double_value
        
        if not calib_path or not os.path.exists(calib_path):
            raise FileNotFoundError(f"Calibration YAML not found or invalid path: {calib_path}")
        
        # --- Load calibration data ---
        with open(calib_path, 'r') as f:
            calib_data = yaml.safe_load(f)
        self.get_logger().info(f"Loaded calibration data from {calib_path}")
        
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

    def map_point_to_screen_resolution(self, point, bbox_width, bbox_height, target_width=1920, target_height=1080):
        x, y = point
        scale_x = target_width / bbox_width
        scale_y = target_height / bbox_height
        return int(x * scale_x), int(y * scale_y)


    def image_callback(self, rgb_msg, depth_msg):
    
        # Access intrinsics if needed
        fx = self.depth_K['fx']
        fy = self.depth_K['fy']
        cx = self.depth_K['cx']
        cy = self.depth_K['cy']
            
        rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        
        rgb_image_for_mediapipe = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image_for_mediapipe)

        cv2.rectangle(rgb_image, (self.x1, self.y1), (self.x4, self.y4), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2


        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_tip_x = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x * self.image_width)
                index_tip_y = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y * self.image_height)

                depth = 0

                if self.x1 <= index_tip_x <= self.x2 and self.y1 <= index_tip_y <= self.y4:
                    finger_depth = depth_image[index_tip_y, index_tip_x]

                    mp.solutions.drawing_utils.draw_landmarks(rgb_image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                    logger.info(f'Index finger tip coordinates: ({index_tip_x}, {index_tip_y}), Depth: {finger_depth:.3f} m')
                    cv2.putText(rgb_image, f'{finger_depth:.2f}mm, Screen Distance: {distance}', (index_tip_x, index_tip_y), font, font_scale, (0, 255, 0), thickness)

                    mp.solutions.drawing_utils.draw_landmarks(rgb_image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                    if  finger_depth >= self.depth_threshold:
                        cv2.putText(rgb_image, 'TOUCH', (0, 0), font, 2, (0, 255, 0), 4)

                        point = (int(index_tip_x - self.x1), int(index_tip_y - self.y1))
                        scaled_x, scaled_y = self.map_point_to_screen_resolution(point, self.table_width, self.table_height, self.screen_width, self.screen_height)

                        self.touch_id = self.generate_touch_id()
                        self.touch_id_counter += 1

                        event_data = {
                            'touchId': self.touch_id,
                            'x': scaled_x,
                            'y': scaled_y,
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
                            logger.info(f"👇 TouchDown: {self.touch_id} at ({scaled_x}, {scaled_y})")
                        
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

    node = PointerTipDepthPublisher(loop=asyncio_loop)

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
