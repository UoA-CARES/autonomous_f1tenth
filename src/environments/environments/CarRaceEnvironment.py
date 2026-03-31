from environments.F1tenthEnvironment import F1tenthEnvironment
from environments.observation_types import ObservationMode


class CarRaceEnvironment(F1tenthEnvironment):
    """
    CarRace environment.

    This environment behaves like `F1tenthEnvironment`, but also resets a second,
    externally controlled car whenever the episode is reset.
    """

    def __init__(
        self,
        car_name: str,
        reward_range: float = 0.5,
        max_steps: int = 3000,
        collision_range_m: float = 0.2,
        step_sleep_time_ms: float = 100,
        track: str = "track_01",
        observation_mode: ObservationMode = "lidar_only",
        opponent_car_name: str = "f2tenth",
    ):
        super().__init__(
            env_name="car_race",
            car_name=car_name,
            reward_range=reward_range,
            max_steps=max_steps,
            collision_range_m=collision_range_m,
            step_sleep_time_ms=step_sleep_time_ms,
            lidar_observation_size=10,
            track=track,
            observation_mode=observation_mode,
        )

        self.opponent_car_name = opponent_car_name
        self.get_logger().info("Environment Setup Complete")

    def _get_opponent_spawn_pose(
        self, primary_spawn_index: int
    ) -> tuple[float, float, float]:
        if self.is_eval and len(self.current_waypoints) > 16:
            opponent_x, opponent_y, opponent_yaw, _ = self.current_waypoints[16]
            return opponent_x, opponent_y, opponent_yaw

        opponent_index = (primary_spawn_index + 2) % len(self.current_waypoints)
        opponent_x, opponent_y, opponent_yaw, _ = self.current_waypoints[opponent_index]
        return opponent_x, opponent_y, opponent_yaw

    def _reset_positions(self):
        super()._reset_positions()

        opponent_x, opponent_y, opponent_yaw = self._get_opponent_spawn_pose(
            self.spawn_index
        )

        self._call_reset_service(
            car_x=opponent_x,
            car_y=opponent_y,
            car_yaw=opponent_yaw,
            goal_x=self.goal_position[0],
            goal_y=self.goal_position[1],
            car_name=self.opponent_car_name,
        )
