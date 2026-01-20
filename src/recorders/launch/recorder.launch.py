from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='recorders',
            executable='vel_recorder',
            name='vel_recorder',
            output='screen'
        ),
        Node(
            package='recorders',
            executable='lidar_recorder',
            name='lidar_recorder',
            output='screen'
        ),
    ])