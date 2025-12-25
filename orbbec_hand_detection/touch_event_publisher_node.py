#!/usr/bin/env python3
"""
Touch Event Publisher Node

This node subscribes to touch detection messages and publishes each
touch event to the NATS event bus.
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import asyncio
import threading
import json
import time
import logging
from datetime import datetime

from orbbec_hand_detection.msg import TouchDetection

from cierra_event_bus import (
    CierraEventBus, Priority
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TouchEventPublisherNode(Node):
    """Node that publishes touch events to the NATS event bus."""

    def __init__(self, loop):
        super().__init__('touch_event_publisher_node')

        self.async_loop = loop  # Store asyncio event loop

        # Declare ROS parameters
        self.declare_parameter('nats_url', 'nats://localhost:4223')
        self.declare_parameter('touch_detections_topic', '/touch_detections')

        # Read parameter values
        self.nats_url = self.get_parameter('nats_url').value
        self.touch_detections_topic = self.get_parameter('touch_detections_topic').value

        # Subscriber
        self.touch_sub = self.create_subscription(
            TouchDetection,
            self.touch_detections_topic,
            self.touch_callback,
            10
        )

        logger.info(f"NATS URL: {self.nats_url}")

    async def connect_to_bus(self):
        """Connect to NATS event bus."""
        try:
            logger.info("Connecting to bus...")
            self.bus = CierraEventBus(self.nats_url)
            await self.bus.connect()
            logger.info("Successfully connected to NATS server.")
        except Exception as e:
            logger.error(f"Failed to connect to NATS server: {e}")

    def touch_callback(self, touch_msg):
        """Process touch detection message and publish events to bus."""
        if not hasattr(self, 'bus') or not self.bus or not self.bus.nats_client:
            logger.warning("Event bus not connected, skipping touch events")
            return

        try:
            # Process each touch event
            for touch_event in touch_msg.touches:
                # Only publish if it's a valid touch
                if touch_event.is_touch:
                    event_data = {
                        'touchId': str(touch_event.touch_id),
                        'x': int(touch_event.x),
                        'y': int(touch_event.y),
                        'pressure': 0.8,  # Could be derived from depth/distance
                        'target': 'smart_table_surface',
                        'userId': 'test_user',
                        'sessionId': f'test_session_{int(time.time() * 1000)}'
                    }

                    message = {
                        'id': str(touch_event.touch_id),
                        'key': 'interaction.touch.down',
                        'data': event_data,
                        'publisher': None,
                        'priority': Priority.MEDIUM,
                        'created': datetime.now().isoformat()
                    }

                    payload = json.dumps(message).encode()

                    # Use asyncio to schedule coroutine correctly
                    asyncio.run_coroutine_threadsafe(
                        self.bus.nats_client.publish('interaction.touch.down', payload),
                        self.async_loop
                    )

                    logger.info(f"👇 TouchDown: {touch_event.touch_id} at ({touch_event.x}, {touch_event.y})")

        except Exception as e:
            logger.error(f"Error publishing touch event: {e}", exc_info=True)

    def destroy_node(self):
        """Clean up node resources."""
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

    node = TouchEventPublisherNode(loop=asyncio_loop)

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

