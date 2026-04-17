import os

import yaml
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
from launch_ros.actions import SetParameter

env_launch = {
    "CarTrack": "cartrack",
    "CarRace": "carrace",
}


def _create_environment_launch(context):
    pkg_bringup = get_package_share_directory("f1tenth_bringup")

    environment_name = LaunchConfiguration("environment").perform(context)
    track = LaunchConfiguration("track").perform(context)
    car_name = "f1tenth"  # Hard-coded car name
    num_opponents = LaunchConfiguration("num_opponents").perform(context)

    launch_arguments = {
        "track": track,
        "car_name": car_name,
    }
    if environment_name == "CarRace":
        launch_arguments["num_opponents"] = num_opponents

    return [
        IncludeLaunchDescription(
            launch_description_source=PythonLaunchDescriptionSource(
                os.path.join(pkg_bringup, f"{env_launch[environment_name]}.launch.py")
            ),
            launch_arguments=launch_arguments.items(),
        )
    ]


def generate_launch_description():
    pkg_f1tenth_description = get_package_share_directory("f1tenth_description")
    pkg_bringup = get_package_share_directory("f1tenth_bringup")

    config_path = os.path.join(pkg_bringup, "config", "environment_config.yaml")

    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    config_params = config["launch"]

    environment_arg = DeclareLaunchArgument("environment")
    track_arg = DeclareLaunchArgument(
        "track",
        default_value=str(config_params.get("track", "multi_track")),
    )
    num_opponents_arg = DeclareLaunchArgument(
        "num_opponents",
        default_value=str(config_params.get("num_opponents", 1)),
    )

    return LaunchDescription(
        [
            environment_arg,
            track_arg,
            num_opponents_arg,
            SetEnvironmentVariable(
                name="GZ_SIM_RESOURCE_PATH", value=pkg_f1tenth_description[:-19]
            ),
            SetParameter(name="use_sim_time", value=True),
            OpaqueFunction(function=_create_environment_launch),
        ]
    )
