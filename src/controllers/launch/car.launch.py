import os
from ament_index_python import get_package_share_directory
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
from launch.substitutions import TextSubstitution
import yaml


def generate_launch_description():
    pkg_controllers = get_package_share_directory('controllers')
    config_path = os.path.join(pkg_controllers, 'car.yaml')
    config = yaml.load(open(config_path), Loader=yaml.Loader)
    alg = config['car']['ros__parameters']['algorithm']
    tracking = config['car']['ros__parameters']['tracking']

    if tracking:
        alg = Node(
            package='controllers',
            executable='track',
            output='screen',
            parameters=[{'car_name': TextSubstitution(text=str(config['car']['ros__parameters'].get('car_name', 'f1tenth')))},
                        {'alg': TextSubstitution(text=str(alg))}, {'isCar': True}],
        )
    elif (f'{alg}' != 'rl'):
        alg = Node(
            package='controllers',
            executable=f'{alg}_policy',
            output='screen',
            parameters=[{'car_name': TextSubstitution(text=str(config['car']['ros__parameters'].get('car_name', 'f1tenth')))}],
        )
    else:
        alg = IncludeLaunchDescription(
            launch_description_source = PythonLaunchDescriptionSource(os.path.join(pkg_controllers, f'{alg}.launch.py')),
            launch_arguments={
                'car_name': TextSubstitution(text=str(config['car']['ros__parameters'].get('car_name', 'f1tenth'))),
            }.items()
        )  
    return LaunchDescription([alg])