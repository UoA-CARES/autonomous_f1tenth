import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory

def generate_launch_description():

    config_path = os.path.join(
        get_package_share_directory('controllers'),
        'rl_policy.yaml'
    )
    
    main = Node(
        package='controllers',
        executable='rl_policy',
        output='screen',
        name='rl_policy',
        parameters=[
            config_path
        ],
    )
    
    vel_recorder = Node(
        package='recorders',
        executable='vel_recorder',
        name='vel_recorder',
        output='screen',
        parameters=[{'onSim': False}]
    )
    
    lidar_recorder = Node(
        package='recorders',
        executable='lidar_recorder',
        name='lidar_recorder',
        output='screen'
    )

    return LaunchDescription([main, vel_recorder, lidar_recorder])
