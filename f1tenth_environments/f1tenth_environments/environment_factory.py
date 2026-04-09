import rclpy

from .car_race_environment import CarRaceEnvironment
from .car_track_environment import CarTrackEnvironment


class EnvironmentFactory:
    def __init__(self):
        rclpy.init()

    def create(self, task: str, config: dict | None = None):
        config = config or {}
        train_config = config.get("train", {}).get("ros__parameters", config)
        action_config = config.get("actions", {})

        car_name = train_config.get("car_name", "f1tenth")
        max_steps = train_config.get("max_steps", 3000)
        track = train_config.get("track", "multi_track")
        observation_mode = train_config.get("observation_mode", "lidar_only")
        max_speed = action_config.get("max_speed", 5.0)
        max_turn = action_config.get("max_turn", 0.434)
        min_speed = action_config.get("min_speed", 0.5)
        min_turn = action_config.get("min_turn", -0.434)

        # Basic Single Agent Environment
        if task == "CarTrack":
            return CarTrackEnvironment(
                car_name=car_name,
                max_steps=max_steps,
                track=track,
                observation_mode=observation_mode,
                max_speed=max_speed,
                max_turn=max_turn,
                min_speed=min_speed,
                min_turn=min_turn,
            )
        # Basic Single Agent vs FTG Environment
        elif task == "CarRace":
            return CarRaceEnvironment(
                car_name=car_name,
                max_steps=max_steps,
                track=track,
                observation_mode=observation_mode,
                max_speed=max_speed,
                max_turn=max_turn,
                min_speed=min_speed,
                min_turn=min_turn,
            )

        raise ValueError(f"EnvironmentFactory: Environment not found {task}")
