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
    pkg_environments = get_package_share_directory("environments")

    environment_name = LaunchConfiguration("environment").perform(context)
    track = LaunchConfiguration("track").perform(context)
    car_name = LaunchConfiguration("car_name").perform(context)
    opponent_car_name = LaunchConfiguration("opponent_car_name").perform(context)

    launch_arguments = {
        "track": track,
        "car_name": car_name,
    }
    if environment_name == "CarRace":
        launch_arguments["opponent_car_name"] = opponent_car_name

    return [
        IncludeLaunchDescription(
            launch_description_source=PythonLaunchDescriptionSource(
                os.path.join(
                    pkg_environments, f"{env_launch[environment_name]}.launch.py"
                )
            ),
            launch_arguments=launch_arguments.items(),
        )
    ]


def generate_launch_description():
    pkg_f1tenth_description = get_package_share_directory("f1tenth_description")

    config_path = os.path.join(
        get_package_share_directory("environments"), "config", "train.yaml"
    )

    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    config_params = config["train"]["ros__parameters"]

    environment_arg = DeclareLaunchArgument("environment")
    track_arg = DeclareLaunchArgument(
        "track",
        default_value=str(config_params.get("track", "multi_track")),
    )
    car_name_arg = DeclareLaunchArgument(
        "car_name",
        default_value=str(config_params.get("car_name", "f1tenth")),
    )
    opponent_car_name_arg = DeclareLaunchArgument(
        "opponent_car_name",
        default_value=str(config_params.get("opponent_car_name", "f2tenth")),
    )

    return LaunchDescription(
        [
            environment_arg,
            track_arg,
            car_name_arg,
            opponent_car_name_arg,
            SetEnvironmentVariable(
                name="GZ_SIM_RESOURCE_PATH", value=pkg_f1tenth_description[:-19]
            ),
            SetParameter(name="use_sim_time", value=True),
            OpaqueFunction(function=_create_environment_launch),
        ]
    )
