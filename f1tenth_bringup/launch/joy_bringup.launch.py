import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup_share = get_package_share_directory("f1tenth_bringup")
    default_joy_config = os.path.join(bringup_share, "config", "joy.yaml")

    joy_topic = LaunchConfiguration("joy_topic")
    teleop_topic = LaunchConfiguration("teleop_topic")

    joy_config_arg = DeclareLaunchArgument(
        "joy_config",
        default_value=default_joy_config,
        description="Joy and joy_teleop parameter file.",
    )
    ros_domain_id_arg = DeclareLaunchArgument(
        "ros_domain_id",
        default_value=EnvironmentVariable("ROS_DOMAIN_ID", default_value="0"),
        description="ROS domain ID used by joystick nodes.",
    )
    rmw_implementation_arg = DeclareLaunchArgument(
        "rmw_implementation",
        default_value="rmw_fastrtps_cpp",
        description="ROS middleware implementation used by joystick nodes.",
    )
    ros_localhost_only_arg = DeclareLaunchArgument(
        "ros_localhost_only",
        default_value="0",
        description="Set ROS_LOCALHOST_ONLY for joystick nodes; use 0 for multi-machine control.",
    )
    ros_discovery_server_arg = DeclareLaunchArgument(
        "ros_discovery_server",
        default_value=EnvironmentVariable("ROS_DISCOVERY_SERVER", default_value=""),
        description="Fast DDS discovery server address, for example 192.168.1.10:11811.",
    )
    joy_topic_arg = DeclareLaunchArgument(
        "joy_topic",
        default_value="joy",
        description="Topic published by joy_node and consumed by joy_teleop.",
    )
    teleop_topic_arg = DeclareLaunchArgument(
        "teleop_topic",
        default_value="teleop",
        description="Ackermann topic published by joy_teleop.",
    )

    joy_node = Node(
        package="joy",
        executable="joy_node",
        name="joy",
        output="screen",
        emulate_tty=True,
        parameters=[LaunchConfiguration("joy_config")],
        remappings=[("joy", joy_topic)],
        respawn=True,
        respawn_delay=2.0,
    )

    joy_teleop_node = Node(
        package="joy_teleop",
        executable="joy_teleop",
        name="joy_teleop",
        output="screen",
        emulate_tty=True,
        parameters=[LaunchConfiguration("joy_config")],
        remappings=[
            ("joy", joy_topic),
            ("teleop", teleop_topic),
        ],
        respawn=True,
        respawn_delay=2.0,
    )

    return LaunchDescription(
        [
            joy_config_arg,
            ros_domain_id_arg,
            rmw_implementation_arg,
            ros_localhost_only_arg,
            ros_discovery_server_arg,
            joy_topic_arg,
            teleop_topic_arg,
            SetEnvironmentVariable(
                "ROS_DOMAIN_ID",
                LaunchConfiguration("ros_domain_id"),
            ),
            SetEnvironmentVariable(
                "ROS_LOCALHOST_ONLY",
                LaunchConfiguration("ros_localhost_only"),
            ),
            SetEnvironmentVariable(
                "RMW_IMPLEMENTATION",
                LaunchConfiguration("rmw_implementation"),
            ),
            SetEnvironmentVariable(
                "ROS_DISCOVERY_SERVER",
                LaunchConfiguration("ros_discovery_server"),
            ),
            joy_node,
            joy_teleop_node,
        ]
    )
