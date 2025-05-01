from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='radar_fusion',
            executable='radar_fusion_node',
            name='radar_fusion_node',
            output='screen',
        )
    ])
