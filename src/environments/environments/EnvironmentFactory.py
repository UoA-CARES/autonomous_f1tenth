import os

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory

from environments.CarRaceEnvironment import CarRaceEnvironment
from environments.CarTrackEnvironment import CarTrackEnvironment


class EnvironmentFactory:
    def __init__(self):
        rclpy.init()

    def create(self, task: str, config: dict):
        # TODO remove hard code to this path, make it more flexible
        config_path = os.path.join(
            get_package_share_directory("environments"),
            "config",
            "train.yaml",
        )
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)["train"]["ros__parameters"]

        # Basic Single Agent Environment
        if task == "CarTrack":
            return CarTrackEnvironment(
                car_name="f1tenth",
                max_steps=config["max_steps"],
                track=config["track"],
                observation_mode=config["observation_mode"],
            )
        # Basic Single Agent vs FTG Environment
        elif task == "CarRace":
            return CarRaceEnvironment(
                car_name="f1tenth",
                max_steps=config["max_steps"],
                track=config["track"],
                observation_mode=config["observation_mode"],
            )

        raise ValueError(f"EnvironmentFactory: Environment not found {task}")
