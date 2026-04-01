import re

from environments.F1tenthEnvironment import F1tenthEnvironment
from environments.observation_types import ObservationMode


class CarRaceEnvironment(F1tenthEnvironment):
    """
    CarRace environment.

    This environment behaves like `F1tenthEnvironment`, but also resets one or more
    externally controlled FTG cars whenever the episode is reset.
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

        self.get_logger().info("Environment Setup Complete")

    def _get_opponent_spawn_pose(
        self, primary_spawn_index: int, opponent_order: int
    ) -> tuple[float, float, float]:
        if self.is_eval and len(self.current_waypoints) > 0:
            eval_index = (16 + opponent_order) % len(self.current_waypoints)
            opponent_x, opponent_y, opponent_yaw, _ = self.current_waypoints[eval_index]
            return opponent_x, opponent_y, opponent_yaw

        opponent_index = (primary_spawn_index + 2 + opponent_order) % len(
            self.current_waypoints
        )
        opponent_x, opponent_y, opponent_yaw, _ = self.current_waypoints[opponent_index]
        return opponent_x, opponent_y, opponent_yaw

    def _discover_opponent_car_names(self) -> list[str]:
        """
        Discover FTG opponents from active ROS topics.

        Expected naming convention: f{n}tenth where ego is f1tenth and opponents are
        f2tenth, f3tenth, ...
        """
        discovered_names: set[str] = set()
        name_pattern = re.compile(r"^f(\d+)tenth$")

        for topic_name, _ in self.get_topic_names_and_types():
            topic_parts = topic_name.strip("/").split("/")
            if len(topic_parts) == 0:
                continue

            car_name = topic_parts[0]
            match = name_pattern.match(car_name)
            if match is None:
                continue

            car_index = int(match.group(1))
            if car_name == self.car_name or car_index <= 1:
                continue

            discovered_names.add(car_name)

        def _car_sort_key(name: str) -> int:
            match = name_pattern.match(name)
            return int(match.group(1)) if match else 10_000

        return sorted(discovered_names, key=_car_sort_key)

    def _reset_positions(self):
        super()._reset_positions()

        opponent_car_names = self._discover_opponent_car_names()
        self.get_logger().info(
            f"Resetting {len(opponent_car_names)} opponent car(s): {opponent_car_names}"
        )

        for opponent_order, opponent_car_name in enumerate(opponent_car_names):
            opponent_x, opponent_y, opponent_yaw = self._get_opponent_spawn_pose(
                self.spawn_index,
                opponent_order,
            )

            self._call_reset_service(
                car_x=opponent_x,
                car_y=opponent_y,
                car_yaw=opponent_yaw,
                goal_x=self.goal_position[0],
                goal_y=self.goal_position[1],
                car_name=opponent_car_name,
            )
