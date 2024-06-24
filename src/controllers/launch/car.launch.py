import os
from ament_index_python import get_package_share_directory
from launch_ros.actions import Node, SetParameter
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.substitutions import TextSubstitution
import yaml


alg_launch = {
    'ftg': 'ftg',
    'rl': 'rl',
    'random': 'random',
}

def generate_launch_description():
    pkg_f1tenth_description = get_package_share_directory('f1tenth_description')
    pkg_controllers = get_package_share_directory('controllers')

    config_path = os.path.join(
        pkg_controllers,
        'car.yaml'
    )

    config = yaml.load(open(config_path), Loader=yaml.Loader)
    alg = config['car']['ros__parameters']['algorithm']


    if (f'{alg}' != 'rl'):
        alg = Node(
            package='controllers',
            executable=f'{alg}_policy',
            output='screen',
            parameters=[{'car_name': TextSubstitution(text=str(config['car']['ros__parameters']['car_name']) if 'car_name' in config['car']['ros__parameters'] else 'f1tenth')}],
        )
    #algorithm = 0
    else:
        alg = IncludeLaunchDescription(
            launch_description_source = PythonLaunchDescriptionSource(
                os.path.join(pkg_controllers, f'{alg_launch[alg]}.launch.py')),
            launch_arguments={
                'car_name': TextSubstitution(text=str(config['car']['ros__parameters']['car_name']) if 'car_name' in config['car']['ros__parameters'] else 'f1tenth'),
            }.items()
        )
    

    

    return LaunchDescription([
        #TODO: Find a way to remove this
        alg,
        #algorithm
])