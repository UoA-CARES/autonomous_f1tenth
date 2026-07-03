import os

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _set_optional_discovery_server(context):
    discovery_server = LaunchConfiguration("ros_discovery_server").perform(context)
    if not discovery_server:
        return []
    return [SetEnvironmentVariable("ROS_DISCOVERY_SERVER", discovery_server)]


def _create_controller_launch(context):
    pkg_controllers = get_package_share_directory("f1tenth_controllers")

    algorithm = LaunchConfiguration("algorithm").perform(context)
    tracking = LaunchConfiguration("tracking").perform(context).lower() == "true"
    car_name = LaunchConfiguration("car_name").perform(context)

    if tracking:
        return [
            Node(
                package="f1tenth_controllers",
                executable="track",
                output="screen",
                parameters=[
                    {"car_name": car_name},
                    {"alg": algorithm},
                    {"isCar": True},
                ],
            )
        ]

    if algorithm != "rl":
        return [
            Node(
                package="f1tenth_controllers",
                executable=f"{algorithm}_policy",
                output="screen",
                parameters=[
                    {"car_name": car_name},
                    {"drive_topic": f"/{car_name}/rl_drive"},
                    {"min_velocity": LaunchConfiguration("ftg_min_velocity")},
                    {"max_velocity": LaunchConfiguration("ftg_max_velocity")},
                ],
            ),
            Node(
                package="f1tenth_controllers",
                executable="rl_deadman",
                output="screen",
                name="rl_deadman",
                parameters=[
                    {
                        "car_name": car_name,
                        "deadman_topic": LaunchConfiguration("deadman_topic"),
                        "deadman_timeout_sec": LaunchConfiguration(
                            "deadman_timeout_sec"
                        ),
                        "command_timeout_sec": LaunchConfiguration(
                            "command_timeout_sec"
                        ),
                    }
                ],
            ),
        ]

    return [
        IncludeLaunchDescription(
            launch_description_source=PythonLaunchDescriptionSource(
                os.path.join(pkg_controllers, "rl.launch.py")
            ),
            launch_arguments={
                "car_name": car_name,
                "rmw_implementation": LaunchConfiguration("rmw_implementation"),
                "ros_domain_id": LaunchConfiguration("ros_domain_id"),
                "ros_discovery_server": LaunchConfiguration("ros_discovery_server"),
                "deadman_topic": LaunchConfiguration("deadman_topic"),
                "cares_python_path": LaunchConfiguration("cares_python_path"),
                "deadman_timeout_sec": LaunchConfiguration("deadman_timeout_sec"),
                "command_timeout_sec": LaunchConfiguration("command_timeout_sec"),
            }.items(),
        )
    ]


def generate_launch_description():

    algorithm_arg = DeclareLaunchArgument("algorithm", default_value="ftg")
    tracking_arg = DeclareLaunchArgument("tracking", default_value="False")
    car_name_arg = DeclareLaunchArgument("car_name", default_value="f1tenth")
    rmw_implementation_arg = DeclareLaunchArgument(
        "rmw_implementation",
        default_value="rmw_fastrtps_cpp",
        description="ROS middleware implementation used by the real-car controller.",
    )
    ros_domain_id_arg = DeclareLaunchArgument(
        "ros_domain_id",
        default_value="0",
        description="ROS domain used by the real-car controller.",
    )
    ros_discovery_server_arg = DeclareLaunchArgument(
        "ros_discovery_server",
        default_value="",
        description="Optional Fast DDS discovery server, for example 172.22.1.87:11811.",
    )
    cares_python_path_arg = DeclareLaunchArgument(
        "cares_python_path",
        default_value=".",
        description="Path containing the cares_reinforcement_learning package to add to PYTHONPATH for algorithm:=rl.",
    )
    ftg_min_velocity_arg = DeclareLaunchArgument(
        "ftg_min_velocity",
        default_value="0.3",
        description="Minimum FTG speed in m/s for algorithm:=ftg.",
    )
    ftg_max_velocity_arg = DeclareLaunchArgument(
        "ftg_max_velocity",
        default_value="1.0",
        description="Maximum FTG speed in m/s for algorithm:=ftg.",
    )
    deadman_topic_arg = DeclareLaunchArgument(
        "deadman_topic",
        default_value="/rl_deadman",
    )
    deadman_timeout_arg = DeclareLaunchArgument(
        "deadman_timeout_sec",
        default_value="0.25",
    )
    command_timeout_arg = DeclareLaunchArgument(
        "command_timeout_sec",
        default_value="0.25",
    )

    return LaunchDescription(
        [
            algorithm_arg,
            tracking_arg,
            car_name_arg,
            rmw_implementation_arg,
            ros_domain_id_arg,
            ros_discovery_server_arg,
            cares_python_path_arg,
            ftg_min_velocity_arg,
            ftg_max_velocity_arg,
            deadman_topic_arg,
            deadman_timeout_arg,
            command_timeout_arg,
            SetEnvironmentVariable(
                "RMW_IMPLEMENTATION",
                LaunchConfiguration("rmw_implementation"),
            ),
            SetEnvironmentVariable(
                "ROS_DOMAIN_ID",
                LaunchConfiguration("ros_domain_id"),
            ),
            OpaqueFunction(function=_set_optional_discovery_server),
            OpaqueFunction(function=_create_controller_launch),
        ]
    )
