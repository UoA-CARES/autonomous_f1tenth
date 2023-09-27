from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    main = Node(
        package='controllers',
        executable='rl_policy',
        output='screen',
        parameters=[{'car_name': ''}],
    )

    return LaunchDescription([main])