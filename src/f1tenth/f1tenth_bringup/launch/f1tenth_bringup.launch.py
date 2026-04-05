import os

from launch_ros.substitutions import FindPackageShare
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, TextSubstitution

def opaque_hardware_launch_config():
    return []

def opaque_simulation_launch_config():
    return []


def generate_launch_description():
    '''
    inputs:
        hardware_mode : if true loads hardware if false spawns in sim
        name: name of f1tenth car
        config: overides defult config (this contains the topic names as well)
        sim_would: if hardware false then it will spawn in set sim_world
    '''

    return LaunchDescription([
    ])
