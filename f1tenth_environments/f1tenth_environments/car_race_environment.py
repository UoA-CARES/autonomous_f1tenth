import re

from .f1tenth_environment import F1tenthEnvironment
from .state_builder import LidarMode, OdomMode


class CarRaceEnvironment(F1tenthEnvironment):
    """Track driving environment with opponent car resets.

    Extends the base environment by discovering opponent cars and repositioning
    them when an episode resets.
    """

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
            env_name="car_race",
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
            env_name="car_race",
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

            self._set_model_pose(
                model_name=opponent_car_name,
                x=float(opponent_x),
                y=float(opponent_y),
                z=0.0,
                yaw=float(opponent_yaw),
            )
