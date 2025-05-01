import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import serial
import struct

class RadarOnlyNode(Node):
    def __init__(self):
        super().__init__('radar_only_node')

        # Publisher for radar points
        self.pc_publisher = self.create_publisher(PointCloud2, 'radar/points', 10)

        # Radar serial config
        self.radar = serial.Serial('/dev/ttyACM1', baudrate=921600, timeout=0.5)
        self.MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'

        # Timer
        self.timer = self.create_timer(0.1, self.timer_callback)

    def publish_pointcloud(self, radar_points):
        if not radar_points:
            return

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'radar_link'

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]

        radar_xyz = [(x, y, 0.0) for (x, y) in radar_points]
        cloud_msg = pc2.create_cloud(header, fields, radar_xyz)
        self.pc_publisher.publish(cloud_msg)
        print(f"[DEBUG] Published {len(radar_xyz)} radar points")

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
                        radar_points.append((x, y))
                    except:
                        continue

        self.publish_pointcloud(radar_points)

def main(args=None):
    rclpy.init(args=args)
    node = RadarOnlyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
