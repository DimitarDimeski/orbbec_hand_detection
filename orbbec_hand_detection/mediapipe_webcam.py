import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

import mediapipe as mp


class WebcamImagePublisher(Node):
    def __init__(self):
        super().__init__('mediapipe_webcam')
        self.publisher_ = self.create_publisher(Image, '/image_raw', 10)
        self.timer = self.create_timer(0.03, self.timer_callback)  # 10 Hz
        self.cap = cv2.VideoCapture(0)  # Use 0 for default webcam
        self.bridge = CvBridge()

        self.hands = mp.solutions.hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        if not self.cap.isOpened():
            self.get_logger().error("Could not open webcam.")
            rclpy.shutdown()

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("Failed to capture frame")
            return

        # Convert BGR to RGB (optional, depending on consumer node)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(frame)
        
        image_height, image_width, _ = frame.shape
        
        if results.multi_hand_landmarks:
              for hand_landmarks in results.multi_hand_landmarks:
                  print(
                            f'Index finger tip coordinates: (',
                            f'{hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                            f'{hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                        )

                  mp.solutions.drawing_utils.draw_landmarks(
                            frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        # Convert to ROS2 Image message and publish
        
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher_.publish(msg)
        self.get_logger().info('Published image frame')

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WebcamImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
