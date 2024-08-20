import os
from ament_index_python import get_package_share_directory
import launch_ros
from launch_ros.actions import Node, SetParameter
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.substitutions import TextSubstitution
import yaml

env_launch = {
    'CarGoal': 'cargoal',
    'CarWall': 'carwall',
    'CarBlock': 'carblock',
    'CarTrack': 'cartrack',
    'CarBeat': 'carbeat',
}

alg_launch = {
    'ftg': 'ftg',
    'rl': 'rl',
    'random': 'random',
    'turn_drive': 'turn_drive',
    'mpc': 'mpc',
}

def generate_launch_description():

    
    alg = Node(
            package='controllers',
            executable='planner',
            output='screen'
        )


    


    return LaunchDescription([
        SetParameter(name='use_sim_time', value=True),
        alg,
])

