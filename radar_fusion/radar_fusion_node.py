import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import serial
import struct
import math

class RadarObstacleNode(Node):
    def __init__(self):
        super().__init__('radar_obstacle_node')

        # Radar serial config
        self.radar = serial.Serial('/dev/ttyACM1', baudrate=921600, timeout=0.5)
        self.MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'

        # ROS publishers
        self.obstacle_pub = self.create_publisher(String, 'radar/obstacle_status', 10)
        self.pc_publisher = self.create_publisher(PointCloud2, 'radar/points', 10)

        # Detection config
        self.min_detection_radius = 0.0  # meters
        self.max_detection_radius = 10.0  # meters (15 cm range)

        # Timer
        self.timer = self.create_timer(0.1, self.timer_callback)

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

        # Filter for XY plane and within range 0 cm to 15 cm
        filtered_points = []
        detected = False
        for x, y, z, v in radar_points:
            distance = math.sqrt(x**2 + y**2)
            if self.min_detection_radius <= distance <= self.max_detection_radius:
                filtered_points.append((x, y, z, v))
                detected = True
                self.get_logger().info(f"\U0001f6d1 Obstacle at ({x:.2f}, {y:.2f}) - Distance: {distance:.2f} m, Velocity: {v:.2f}")

        msg = String()
        msg.data = "\U0001f6d8 Obstacle Detected" if detected else "\u2705 Path Clear"
        self.obstacle_pub.publish(msg)

        # Publish point cloud for RViz
        self.publish_pointcloud(filtered_points)

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
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
