import re

from .f1tenth_environment import F1tenthEnvironment
from .observation_types import ObservationMode


class CarRaceEnvironment(F1tenthEnvironment):
    """Track driving environment with opponent car resets.

    Extends the base environment by discovering opponent cars and repositioning
    them when an episode resets.
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
        max_speed: float = 5.0,
        max_turn: float = 0.434,
        min_speed: float = 0.5,
        min_turn: float = -0.434,
    ) -> None:
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
            max_speed=max_speed,
            max_turn=max_turn,
            min_speed=min_speed,
            min_turn=min_turn,
        )

    def _get_opponent_spawn_pose(
        self, primary_spawn_index: int, opponent_order: int
    ) -> tuple[float, float, float]:
        """Return the waypoint-based spawn pose for one opponent."""
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
        """Find opponent cars from active ROS topic namespaces."""
        discovered_names: set[str] = set()
        name_pattern = re.compile(r"^f(\d+)tenth$")

        for topic_name, _ in self.get_topic_names_and_types():
            topic_root = topic_name.strip("/").split("/", 1)[0]
            if not topic_root:
                continue

            car_name = topic_root
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

    def _reset_positions(self) -> None:
        """Reset the ego car first, then reposition all discovered opponents."""
        super()._reset_positions()

        opponent_car_names = self._discover_opponent_car_names()

        for opponent_order, opponent_car_name in enumerate(opponent_car_names):
            opponent_x, opponent_y, opponent_yaw = self._get_opponent_spawn_pose(
                self.spawn_index,
                opponent_order,
            )

            self._set_reset_poses(
                car_x=opponent_x,
                car_y=opponent_y,
                car_yaw=opponent_yaw,
                goal_x=self.goal_position[0],
                goal_y=self.goal_position[1],
                car_name=opponent_car_name,
                update_goal=False,
            )
