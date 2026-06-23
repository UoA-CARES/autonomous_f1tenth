import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup_share = get_package_share_directory("f1tenth_bringup")
    default_joy_config = os.path.join(bringup_share, "config", "joy.yaml")

    joy_config_arg = DeclareLaunchArgument(
        "joy_config",
        default_value=default_joy_config,
        description="Joy and joy_teleop parameter file.",
    )

    joy_node = Node(
        package="joy",
        executable="joy_node",
        name="joy",
        output="screen",
        parameters=[LaunchConfiguration("joy_config")],
    )

    joy_teleop_node = Node(
        package="joy_teleop",
        executable="joy_teleop",
        name="joy_teleop",
        output="screen",
        parameters=[LaunchConfiguration("joy_config")],
    )

    return LaunchDescription(
        [
            joy_config_arg,
            joy_node,
            joy_teleop_node,
        ]
    )
