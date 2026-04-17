import rclpy

from .car_race_environment import CarRaceEnvironment
from .car_track_environment import CarTrackEnvironment


class EnvironmentFactory:
    def __init__(self):
        rclpy.init()

    def create(self, task: str, config: dict | None = None):
        config = config or {}

        # Basic Envrionment Setup
        max_steps = config.get("max_steps", 1000)
        step_sleep_time_ms = config.get("step_sleep_time_ms", 100)

        # Which Track will be used for training and evaluation.
        track = config.get("track", "multi_track")
        train_eval_split = config.get("train_eval_split", 0.5)

        # State Builder Configurations
        odom_mode = config.get("odom_mode", "velocity_only")
        lidar_mode = config.get("lidar_mode", "processed")
        lidar_state_size = config.get("lidar_state_size", 9)

        max_speed = config.get("max_speed", 5.0)
        min_speed = config.get("min_speed", 0.5)
        max_turn = config.get("max_turn", 0.434)

        # Reward Function Configurations
        wall_proximity_reward_weight = config.get("wall_proximity_reward_weight", 0.7)
        turn_reward_weight = config.get("turn_reward_weight", 0.3)
        collision_penalty = config.get("collision_penalty", 1.0)
        # Environment-specific Configurations
        collision_range_m = config.get("collision_range_m", 0.2)
        goal_reach_radius_m = config.get("goal_reach_radius_m", 0.1)

        stall_progress_threshold_m = config.get("stall_progress_threshold_m", 0.02)
        stall_limit_steps = config.get("stall_limit_steps", 5)

        if task == "CarTrack":
            return CarTrackEnvironment(
                lidar_state_size=lidar_state_size,
                goal_reach_radius_m=goal_reach_radius_m,
                max_steps=max_steps,
                collision_range_m=collision_range_m,
                step_sleep_time_ms=step_sleep_time_ms,
                track=track,
                odom_mode=odom_mode,
                lidar_mode=lidar_mode,
                max_speed=max_speed,
                max_turn=max_turn,
                min_speed=min_speed,
                train_eval_split=train_eval_split,
                wall_proximity_reward_weight=wall_proximity_reward_weight,
                turn_reward_weight=turn_reward_weight,
                stall_progress_threshold_m=stall_progress_threshold_m,
                stall_limit_steps=stall_limit_steps,
                collision_penalty=collision_penalty,
            )
        elif task == "CarRace":
            return CarRaceEnvironment(
                lidar_state_size=lidar_state_size,
                goal_reach_radius_m=goal_reach_radius_m,
                max_steps=max_steps,
                collision_range_m=collision_range_m,
                step_sleep_time_ms=step_sleep_time_ms,
                track=track,
                odom_mode=odom_mode,
                lidar_mode=lidar_mode,
                max_speed=max_speed,
                max_turn=max_turn,
                min_speed=min_speed,
                train_eval_split=train_eval_split,
                wall_proximity_reward_weight=wall_proximity_reward_weight,
                turn_reward_weight=turn_reward_weight,
                stall_progress_threshold_m=stall_progress_threshold_m,
                stall_limit_steps=stall_limit_steps,
                collision_penalty=collision_penalty,
            )

        raise ValueError(f"EnvironmentFactory: Environment not found {task}")
