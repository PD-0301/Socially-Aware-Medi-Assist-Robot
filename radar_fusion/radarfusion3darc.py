import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Header
from sensor_msgs.msg import PointCloud2, PointField, Image
from geometry_msgs.msg import Twist
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
import serial
import struct
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np

class RadarObstacleNode(Node):
    def __init__(self):
        super().__init__('radar_obstacle_node')

        # === Radar Serial Config ===
        self.radar = serial.Serial('/dev/ttyACM1', baudrate=921600, timeout=0.5)
        self.MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'

        # === Publishers ===
        self.obstacle_pub = self.create_publisher(String, 'radar/obstacle_status', 10)
        self.pc_publisher = self.create_publisher(PointCloud2, 'radar/points', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # === Camera Subscription ===
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.current_frame = None

        # === Detection Settings ===
        self.min_detection_radius = 0.0
        self.max_detection_radius = 8.0  # 10 meters

        # === Plot Setup ===
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.scatter = self.ax.scatter([], [], c='red')
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(0, 12)
        self.ax.set_title("📡 Live Radar View")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.plot(0, 0, marker='o', color='black', markersize=6)
        self.ax.text(0, 0.5, "Radar", ha='center')

        # === Main Timer Callback ===
        self.timer = self.create_timer(0.1, self.timer_callback)

    def image_callback(self, msg):
        try:
            self.current_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.imshow("Camera Feed", self.current_frame)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Camera conversion failed: {e}")

    def timer_callback(self):
        data = self.radar.read(4096)
        idx = data.find(self.MAGIC_WORD)
        radar_points = []

        if idx != -1 and len(data) > idx + 48:
            pkt_len = struct.unpack('<I', data[idx+12:idx+16])[0]
            if len(data) >= idx + pkt_len:
                num_obj = struct.unpack('<H', data[idx+28:idx+30])[0]
                obj_start = idx + 48
                for i in range(num_obj):
                    try:
                        x, y, z, v = struct.unpack('<ffff', data[obj_start+i*16:obj_start+(i+1)*16])
                        radar_points.append((x, y, z, v))
                    except:
                        continue

        # Filter radar points
        filtered_points = []
        detected = False
        for x, y, z, v in radar_points:
            distance = math.sqrt(x**2 + y**2)
            if self.min_detection_radius <= distance <= self.max_detection_radius:
                filtered_points.append((x, y, z, v))
                detected = True
                self.get_logger().info(f"🛑 Obstacle at ({x:.2f}, {y:.2f}) - Distance: {distance:.2f} m, Velocity: {v:.2f}")

        # Obstacle status
        msg = String()
        msg.data = "🚦 Obstacle Detected" if detected else "✅ Path Clear"
        self.obstacle_pub.publish(msg)

        # Motion control
        twist = Twist()
        if detected:
            twist.linear.x = 0.0
            twist.angular.z = 0.5
        else:
            twist.linear.x = 0.2
            twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

        # Publish PointCloud2
        self.publish_pointcloud(filtered_points)

        # === Plot Radar Points ===
        self.update_plot(filtered_points)

    def update_plot(self, radar_points):
        if radar_points:
            xy = np.array([(x, y) for (x, y, z, v) in radar_points])
            self.scatter.set_offsets(xy)
        else:
            self.scatter.set_offsets(np.empty((0, 2)))
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def publish_pointcloud(self, radar_points):
        if not radar_points:
            return
        header = self.get_clock().now().to_msg()
        msg_header = Header(stamp=header, frame_id='map')
        fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        radar_array = [(x, y, z, v) for (x, y, z, v) in radar_points]
        cloud_msg = pc2.create_cloud(msg_header, fields, radar_array)
        self.pc_publisher.publish(cloud_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RadarObstacleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    main()
