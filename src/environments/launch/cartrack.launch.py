import os

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch(context):
    pkg_ros_gz_sim = get_package_share_directory("ros_gz_sim")
    pkg_environments = get_package_share_directory("environments")
    pkg_f1tenth_bringup = get_package_share_directory("f1tenth_bringup")

    track = LaunchConfiguration("track").perform(context)
    car_name = LaunchConfiguration("car_name").perform(context)

    gz_sim = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={
            "gz_args": f"-s -r {pkg_environments}/worlds/{track}.sdf",
        }.items(),
    )

    f1tenth = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            os.path.join(pkg_f1tenth_bringup, "simulation_bringup.launch.py")
        ),
        launch_arguments={"name": car_name, "world": "empty"}.items(),
    )

    return [gz_sim, f1tenth]


def generate_launch_description():

    track_arg = DeclareLaunchArgument("track", default_value="track_1")

    car_name = DeclareLaunchArgument("car_name", default_value="f1tenth")

    service_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        arguments=[
            "/world/empty/control@ros_gz_interfaces/srv/ControlWorld",
            "/world/empty/create@ros_gz_interfaces/srv/SpawnEntity",
            "/world/empty/remove@ros_gz_interfaces/srv/DeleteEntity",
            "/world/empty/set_pose@ros_gz_interfaces/srv/SetEntityPose",
            "/world/empty/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock",
        ],
        remappings=[
            ("/world/empty/clock", "/clock"),
        ],
    )

    stepping_service = Node(
        package="environments",
        executable="SteppingService",
        output="screen",
        emulate_tty=True,
    )

    reset = Node(
        package="environments",
        executable="F1TenthReset",
        parameters=[{"env_name": "car_track"}],
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription(
        [
            track_arg,
            OpaqueFunction(function=launch),
            service_bridge,
            reset,
            stepping_service,
            car_name,
        ]
    )
