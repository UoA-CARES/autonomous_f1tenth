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
                parameters=[{"car_name": car_name}],
            )
        ]

    return [
        IncludeLaunchDescription(
            launch_description_source=PythonLaunchDescriptionSource(
                os.path.join(pkg_controllers, "rl.launch.py")
            ),
            launch_arguments={
                "car_name": car_name,
            }.items(),
        )
    ]


def generate_launch_description():

    algorithm_arg = DeclareLaunchArgument("algorithm", default_value="ftg")
    tracking_arg = DeclareLaunchArgument("tracking", default_value="False")
    car_name_arg = DeclareLaunchArgument("car_name", default_value="f1tenth")

    return LaunchDescription(
        [
            algorithm_arg,
            tracking_arg,
            car_name_arg,
            OpaqueFunction(function=_create_controller_launch),
        ]
    )
