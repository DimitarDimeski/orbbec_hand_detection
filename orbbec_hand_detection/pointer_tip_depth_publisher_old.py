import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from message_filters import Subscriber, ApproximateTimeSynchronizer

import mediapipe as mp

class PointerTipDepthPublisher(Node):
    def __init__(self):
        super().__init__('pointer_tip_depth')
        self.rgb_sub = Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/camera/depth/image_raw')

        self.publisher_ = self.create_publisher(Image, '/image_landmarked', 10)

        self.ts = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.image_callback)
        
        self.bridge = CvBridge()

        self.hands = mp.solutions.hands.Hands(
            min_detection_confidence=0.2,
            min_tracking_confidence=0.2)


    def image_callback(self, rgb_msg, depth_msg):
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        
            rgb_image_for_mediapipe = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_image_for_mediapipe)
        
            image_height, image_width, _ = rgb_image.shape
        
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (0, 255, 0)
            thickness = 2
        
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    index_tip_x = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)
                    index_tip_y = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
                    
                    if 0 <= index_tip_x < depth_image.shape[1] and 0 <= index_tip_y < depth_image.shape[0]:
                        depth = depth_image[index_tip_y, index_tip_x]
                        print(f'Index finger tip coordinates: ({index_tip_x}, {index_tip_y}), Depth: {depth:.3f} m')
        
                        cv2.putText(rgb_image, f'{depth:.2f}mm', (index_tip_x, index_tip_y), font, font_scale, color, thickness, cv2.LINE_AA)
        
                    mp.solutions.drawing_utils.draw_landmarks(
                        rgb_image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        
            msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding='bgr8')
            self.publisher_.publish(msg)
            self.get_logger().info('Published image frame')

    def destroy_node(self):
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PointerTipDepthPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

