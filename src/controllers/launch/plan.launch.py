import os
from ament_index_python import get_package_share_directory
from launch_ros.actions import Node, SetParameter
from launch import LaunchDescription
from launch.substitutions import TextSubstitution
import yaml

def generate_launch_description():
    pkg_controllers = get_package_share_directory('controllers')
    config_path = os.path.join(pkg_controllers, 'plan.yaml')
    config = yaml.load(open(config_path), Loader=yaml.Loader)
    alg = config['plan']['ros__parameters']['algorithm']
    map = config['plan']['ros__parameters']['map']
    yaml_path = config['plan']['ros__parameters']['yaml_path']

    alg = Node(
            package='controllers',
            executable='planner',
            output='screen',
            parameters=[{'alg': TextSubstitution(text=str(alg))}, 
                {'map': TextSubstitution(text=str(map))}, 
                {'yaml_path': TextSubstitution(text=str(yaml_path))}]
    )
    return LaunchDescription([SetParameter(name='use_sim_time', value=True), alg])

