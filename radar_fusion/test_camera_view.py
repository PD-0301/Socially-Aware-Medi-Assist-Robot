#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ArducamPublisher(Node):
    def __init__(self):
        super().__init__('arducam_publisher')
        self.publisher = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()

        # Open the correct camera device (Arducam)
        self.cap = cv2.VideoCapture(2)  # /dev/video2
        if not self.cap.isOpened():
            self.get_logger().error("❌ Failed to open Arducam at /dev/video2")
            exit(1)

        # Timer to grab and publish frames at ~10 Hz
        self.timer = self.create_timer(0.1, self.publish_frame)

    def publish_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("⚠️ Failed to read frame from Arducam.")
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher.publish(msg)
        self.get_logger().info('📤 Published frame from Arducam')

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ArducamPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
