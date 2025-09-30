import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import json
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

        # Subscribers
        self.rgb_sub = Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/camera/depth/image_raw')

        self.publisher_ = self.create_publisher(Image, '/image_landmarked', 10)

        self.ts = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.image_callback)
        
        self.bridge = CvBridge()
        self.hands = mp.solutions.hands.Hands(
            min_detection_confidence=0.2,
            min_tracking_confidence=0.2)

        # Configuration
        self.nats_url = os.getenv('NATS_URL', 'nats://localhost:4222')
        self.table_width = int(os.getenv('WIDTH', 640))
        self.table_height = int(os.getenv('HEIGHT', 480))
        self.depth_threshold = int(os.getenv('DEPTH', 880))
        self.screen_width = int(os.getenv('SCREEN_WIDTH', 1920))
        self.screen_height = int(os.getenv('SCREEN_HEIGHT', 1080))
        
        self.touch_id = None
        self.touch_id_counter = 0
        self.touch_down = InteractionEvents.touch_down()

        logger.info(f"NATS URL: {self.nats_url}")
        logger.info(f"Width x Height x Depth:  {self.table_width} x {self.table_height} x {self.depth_threshold}")
        logger.info(f"Screen Resolution: {self.screen_width} x {self.screen_height}")

        self.events_received = []

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

    def interaction_handler(self, data, event):
        self.events_received.append({
            "type": event.get_key(),
            "data": data,
            "priority": event.get_priority()
        })
        logger.info(f"🎯 {event.get_key()}: {data}")

    def image_callback(self, rgb_msg, depth_msg):
        rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        rgb_image_for_mediapipe = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image_for_mediapipe)

        image_height, image_width, _ = rgb_image.shape
        logger.info(f"RGB Image Shape:  {rgb_image.shape}")

        if self.table_width <= image_width and self.table_height <= image_height:
            x1 = int((image_width - self.table_width) / 2)
            y1 = int((image_height - self.table_height) / 2)
            x2 = x1 + self.table_width
            y2 = y1 + self.table_height
        else:
            x1, y1 = 0, 0
            x2, y2 = image_width, image_height

        cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        logger.info(f"Depth At Center Of Table: {depth_image[int(self.table_height / 2), int(self.table_width / 2)]}")

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_tip_x = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)
                index_tip_y = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)

                depth = 0

                if 0 <= index_tip_x < depth_image.shape[1] and 0 <= index_tip_y < depth_image.shape[0]:
                    depth = depth_image[index_tip_y, index_tip_x]
                    logger.info(f'Index finger tip coordinates: ({index_tip_x}, {index_tip_y}), Depth: {depth:.3f} m')
                    cv2.putText(rgb_image, f'{depth:.2f}mm', (index_tip_x, index_tip_y), font, font_scale, (0, 255, 0), thickness)

                    mp.solutions.drawing_utils.draw_landmarks(rgb_image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                    if x1 <= index_tip_x <= x2 and y1 <= index_tip_y <= y2 and depth >= self.depth_threshold:
                        cv2.putText(rgb_image, 'TOUCH', (0, 0), font, 2, (0, 255, 0), 4)

                        point = (int(index_tip_x - x1), int(index_tip_y - y1))
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
