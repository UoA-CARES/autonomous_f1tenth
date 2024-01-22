import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory
import yaml

def generate_launch_description():

    config_path = os.path.join(
        get_package_share_directory('controllers'),
        'rl_policy.yaml'
    )

    #config = yaml.load(open(config_path), Loader=yaml.Loader)
    
    main = Node(
        package='controllers',
        executable='rl_policy',
        output='screen',
        parameters=[
            config_path
        ],
    )

    return LaunchDescription([main])
