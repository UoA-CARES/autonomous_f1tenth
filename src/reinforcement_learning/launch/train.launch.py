import os

import yaml
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import TextSubstitution
from launch_ros.actions import SetParameter

env_launch = {
    "CarTrack": "cartrack",
    "CarRace": "carrace",
}


def generate_launch_description():
    pkg_f1tenth_description = get_package_share_directory("f1tenth_description")
    pkg_environments = get_package_share_directory("environments")

    config_path = os.path.join(
        get_package_share_directory("reinforcement_learning"), "train.yaml"
    )

    config = yaml.load(open(config_path), Loader=yaml.Loader)
    env = config["train"]["ros__parameters"]["environment"]

    environment = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            os.path.join(pkg_environments, f"{env_launch[env]}.launch.py")
        ),
        launch_arguments={
            "track": TextSubstitution(
                text=str(config["train"]["ros__parameters"]["track"])
            ),
            "car_name": TextSubstitution(
                text=str(config["train"]["ros__parameters"]["car_name"])
            ),
            "car_one": TextSubstitution(
                text=(
                    str(config["train"]["ros__parameters"]["car_name"])
                    if "car_name" in config["train"]["ros__parameters"]
                    else "f1tenth"
                )
            ),
            "car_two": TextSubstitution(
                text=(
                    str(config["train"]["ros__parameters"]["ftg_car_name"])
                    if "ftg_car_name" in config["train"]["ros__parameters"]
                    else "f2tenth"
                )
            ),
        }.items(),  # TODO: this doesn't do anything
    )

    return LaunchDescription(
        [
            SetEnvironmentVariable(
                name="GZ_SIM_RESOURCE_PATH", value=pkg_f1tenth_description[:-19]
            ),
            SetParameter(name="use_sim_time", value=True),
            environment,
        ]
    )
