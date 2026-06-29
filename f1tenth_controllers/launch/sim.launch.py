import os

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    pkg_bringup = get_package_share_directory("f1tenth_bringup")

    delegated = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_bringup, "sim.launch.py"))
    )
    return LaunchDescription([delegated])
