from .f1tenth_environment import F1tenthEnvironment
from .state_builder import LidarMode, OdomMode


class CarTrackEnvironment(F1tenthEnvironment):
    """Single-car track driving environment.

    The agent follows a track using lidar and odometry observations.
    """

    def __init__(
        self,
        car_name: str,
        reward_range: float = 3,
        max_steps: int = 1000,
        collision_range_m: float = 0.2,
        step_sleep_time_ms: float = 100,
        track: str = "track_01",
        odom_mode: OdomMode = "velocity_only",
        lidar_mode: LidarMode = "processed",
        max_speed: float = 5.0,
        max_turn: float = 0.434,
        min_speed: float = 0.5,
    ) -> None:
        super().__init__(
            env_name="car_track",
            car_name=car_name,
            reward_range=reward_range,
            max_steps=max_steps,
            collision_range_m=collision_range_m,
            step_sleep_time_ms=step_sleep_time_ms,
            lidar_state_size=9,
            track=track,
            odom_mode=odom_mode,
            lidar_mode=lidar_mode,
            max_speed=max_speed,
            max_turn=max_turn,
            min_speed=min_speed,
        )
