from .f1tenth_environment import F1tenthEnvironment
from .state_builder import LidarMode, OdomMode


class CarTrackEnvironment(F1tenthEnvironment):
    """Single-car track driving environment."""

    def __init__(
        self,
        lidar_state_size: int,
        goal_reach_radius_m: float,
        max_steps: int,
        collision_range_m: float,
        step_sleep_time_ms: float,
        track: str,
        odom_mode: OdomMode,
        lidar_mode: LidarMode,
        max_speed: float,
        max_turn: float,
        min_speed: float,
        train_eval_split: float,
        wall_proximity_reward_weight: float,
        turn_reward_weight: float,
        stall_progress_threshold_m: float,
        stall_limit_steps: int,
        collision_penalty: float,
    ) -> None:
        super().__init__(
            env_name="car_track",
            lidar_state_size=lidar_state_size,
            goal_reach_radius_m=goal_reach_radius_m,
            max_steps=max_steps,
            collision_range_m=collision_range_m,
            step_sleep_time_ms=step_sleep_time_ms,
            track=track,
            odom_mode=odom_mode,
            lidar_mode=lidar_mode,
            train_eval_split=train_eval_split,
            max_speed=max_speed,
            min_speed=min_speed,
            max_turn=max_turn,
            wall_proximity_reward_weight=wall_proximity_reward_weight,
            turn_reward_weight=turn_reward_weight,
            stall_progress_threshold_m=stall_progress_threshold_m,
            stall_limit_steps=stall_limit_steps,
            collision_penalty=collision_penalty,
        )
        super().__init__(
            env_name="car_track",
            goal_reach_radius_m=goal_reach_radius_m,
            max_steps=max_steps,
            collision_range_m=collision_range_m,
            step_sleep_time_ms=step_sleep_time_ms,
            lidar_state_size=lidar_state_size,
            track=track,
            odom_mode=odom_mode,
            lidar_mode=lidar_mode,
            train_eval_split=train_eval_split,
            max_speed=max_speed,
            min_speed=min_speed,
            max_turn=max_turn,
            wall_proximity_reward_weight=wall_proximity_reward_weight,
            turn_reward_weight=turn_reward_weight,
            stall_progress_threshold_m=stall_progress_threshold_m,
            stall_limit_steps=stall_limit_steps,
            collision_penalty=collision_penalty,
        )
