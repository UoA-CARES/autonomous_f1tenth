import math
import os
import random
from abc import ABC
from typing import Literal

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import LaserScan
from std_srvs.srv import SetBool

from environment_interfaces.srv import Reset

from . import util
from .observation_types import Observation, ObservationMode, OdomState
from .waypoints import waypoints


class F1tenthEnvironment(Node, ABC):

    def __init__(
        self,
        env_name: str,
        car_name: str,
        reward_range: float = 0.5,
        max_steps: int = 3000,
        collision_range_m: float = 0.2,
        step_sleep_time_ms: float = 100,
        lidar_observation_size: int = 10,
        track: str = "track_01",
        observation_mode: ObservationMode = "lidar_only",
        train_eval_split: float = 0.5,
    ):
        super().__init__(f"{env_name}_environment")

        if lidar_observation_size < 1:
            raise ValueError("Make sure number of lidar points is more than 0")

        self.car_name = car_name
        self.goal_reach_radius = reward_range
        self.max_steps = max_steps
        self.collision_range_m = collision_range_m
        self.step_sleep_time_ms = step_sleep_time_ms
        self.lidar_observation_size = lidar_observation_size
        self.track_train_eval_split = train_eval_split

        self.observation_mode = observation_mode
        match observation_mode:
            case "lidar_only":
                odom_observation_size = 2
            case "no_position":
                odom_observation_size = 6
            case "full_state":
                odom_observation_size = 10
            case _:
                raise ValueError(f"Unsupported observation_mode: {observation_mode}")
        self.observation_size = odom_observation_size + self.lidar_observation_size

        self.action_num = 2

        self.tracks = self._load_tracks(track)
        self.track_models = util.get_track_math_defs(self.tracks)
        self.track_names = list(self.tracks.keys())

        self.current_track = self.track_names[0]
        self.current_waypoints = self.tracks[self.current_track]
        self.current_track_model = self.track_models[self.current_track]

        # Backward-compatible aliases for existing subclasses
        self.all_track_waypoints = self.tracks
        self.all_track_models = self.track_models

        self.eval_track_begin_idx: int = int(
            len(self.track_names) * self.track_train_eval_split
        )
        self.eval_track_idx = 0

        self.lidar_reduction_mode: Literal["avg", "raw"] = "avg"

        config_path = os.path.join(
            get_package_share_directory("environments"),
            "config",
            "config.yaml",
        )
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        self.max_actions = np.asarray(
            [config["actions"]["max_speed"], config["actions"]["max_turn"]]
        )
        self.min_actions = np.asarray(
            [config["actions"]["min_speed"], config["actions"]["min_turn"]]
        )

        self.cmd_vel_pub = self.create_publisher(Twist, f"/{self.car_name}/cmd_vel", 1)

        sub_depth = 3
        qos = QoSProfile(depth=sub_depth)
        self.odom_sub = Subscriber(
            self,
            Odometry,
            f"/{self.car_name}/odometry",
            qos_profile=qos,
        )

        self.lidar_sub = Subscriber(
            self,
            LaserScan,
            f"/{self.car_name}/scan",
            qos_profile=qos,
        )

        self.processed_publisher = self.create_publisher(
            LaserScan, f"/{self.car_name}/processed_scan", 1
        )

        self.message_filter = ApproximateTimeSynchronizer(
            [self.odom_sub, self.lidar_sub],
            sub_depth,
            0.1,
        )
        self.message_filter.registerCallback(self._message_filter_callback)

        # Reset Client -----------------------------------------------
        self.reset_client = self.create_client(Reset, f"{env_name}_reset")
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("reset service not available, waiting again...")

        # Stepping Client ---------------------------------------------
        self.stepping_client = self.create_client(SetBool, "stepping_service")
        while not self.stepping_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("stepping service not available, waiting again...")

        self.reward_modifiers: list[tuple[Literal["turn", "wall_proximity"], float]] = [
            ("turn", 0.3),
            ("wall_proximity", 0.7),
        ]

        self.latest_data: tuple[Odometry, LaserScan] | None = None
        self.current_observation: Observation | None = None

        self.step_counter = 0
        self.goals_reached = 0

        self.previous_closest_spline_t: float | None = None
        self.is_eval = False
        self.spawn_index = 0

        self.goal_position: tuple[float, float] = (0.0, 0.0)

        self.progress_not_met_cnt = 0

    def _load_tracks(self, track_name: str) -> dict:
        if "multi_track" in track_name or track_name == "staged_tracks":
            _, all_track_waypoints = util.get_all_goals_and_waypoints_in_multi_tracks(
                track_name
            )
            return all_track_waypoints

        track_key = track_name
        return {track_key: waypoints[track_key]}

    def _get_track_split_keys(self) -> tuple[list[str], list[str]]:
        split_idx = min(self.eval_track_begin_idx, len(self.track_names))
        split_idx = max(0, split_idx)

        train_keys = self.track_names[:split_idx]
        eval_keys = self.track_names[split_idx:]

        if len(train_keys) == 0:
            train_keys = self.track_names
        if len(eval_keys) == 0:
            eval_keys = self.track_names

        return train_keys, eval_keys

    def _select_track_name(self) -> str:
        train_keys, eval_keys = self._get_track_split_keys()

        if self.is_eval:
            selected = eval_keys[self.eval_track_idx % len(eval_keys)]
            self.eval_track_idx = (self.eval_track_idx + 1) % len(eval_keys)
            return selected

        return random.choice(train_keys)

    def _reset_positions(self):
        self.current_track = self._select_track_name()
        self.current_waypoints = self.tracks[self.current_track]
        self.current_track_model = self.track_models[self.current_track]

        if self.is_eval:
            car_x, car_y, car_yaw, index = self.current_waypoints[10]
        else:
            car_x, car_y, car_yaw, index = random.choice(self.current_waypoints)

        self.spawn_index = index
        goal_x, goal_y, _, _ = self.current_waypoints[
            (
                self.spawn_index + 1
                if self.spawn_index + 1 < len(self.current_waypoints)
                else 0
            )
        ]

        self.goal_position = (goal_x, goal_y)
        self._call_reset_service(
            car_x=car_x,
            car_y=car_y,
            car_yaw=car_yaw,
            goal_x=goal_x,
            goal_y=goal_y,
            car_name=self.car_name,
        )

    def _reset(self) -> tuple[np.ndarray, dict]:
        self._reset_positions()

        self._set_simulation_paused(paused=False)
        state, observation, _ = self._get_observation()
        self.current_observation = observation
        self._set_simulation_paused(paused=True)

        self.previous_closest_spline_t = (
            self.current_track_model.get_closest_point_on_spline(
                [observation.odom.x, observation.odom.y],
                t_only=True,
            )
        )

        return state, {}

    def reset(self, training: bool = True) -> np.ndarray:
        self.step_counter = 0
        self.goals_reached = 0

        self.progress_not_met_cnt = 0

        self.is_eval = not training

        self._set_velocity(0, 0)

        state, _ = self._reset()
        return state

    def _message_filter_callback(self, odom: Odometry, lidar: LaserScan) -> None:
        self.latest_data = (odom, lidar)

    def _get_data(self, timeout: float = 5.0) -> tuple[Odometry, LaserScan]:
        # Drain anything stale
        self.latest_data = None
        end_time = self.get_clock().now().nanoseconds + int(timeout * 1e9)

        while self.get_clock().now().nanoseconds < end_time:
            rclpy.spin_once(self, timeout_sec=0.01)
            if self.latest_data is not None:
                return self.latest_data

        raise TimeoutError("No synced data received")

    def _sleep(self, duration_ms: float) -> None:
        """
        Sleep while still processing incoming messages, to allow for callbacks to run.

        Critical that this uses self.get_clock() for timekeeping, to ensure it works properly with simulated time.
        """
        end_time = self.get_clock().now().nanoseconds + int(duration_ms * 1e6)

        while self.get_clock().now().nanoseconds < end_time:
            rclpy.spin_once(self, timeout_sec=0.01)

    def _is_terminated(self, observation: Observation, ranges: list[float]) -> bool:
        quaternion = observation.odom.quaternion_wxyz()
        return util.has_collided(
            ranges, self.collision_range_m
        ) or util.has_flipped_over(quaternion)

    def _is_truncated(self) -> bool:
        return self.progress_not_met_cnt >= 5 or self.step_counter >= self.max_steps

    def _process_lidar_observation(
        self, lidar_msg: LaserScan
    ) -> tuple[list[float], LaserScan]:
        match self.lidar_reduction_mode:
            case "avg":
                processed_lidar_range = util.avg_lidar(
                    lidar_msg, self.lidar_observation_size
                )
                visualization_scan = util.create_lidar_msg(
                    lidar_msg, self.lidar_observation_size, processed_lidar_range
                )
            case "raw":
                processed_lidar_range = np.array(lidar_msg.ranges.tolist())
                processed_lidar_range = np.nan_to_num(
                    processed_lidar_range, posinf=-5, nan=-1, neginf=-5
                ).tolist()
                visualization_scan = util.create_lidar_msg(
                    lidar_msg, len(processed_lidar_range), processed_lidar_range
                )

        return processed_lidar_range, visualization_scan

    def _get_observation(self) -> tuple[np.ndarray, Observation, list[float]]:
        odom_msg, lidar_msg = self._get_data()

        processed_lidar_range, visualization_scan = self._process_lidar_observation(
            lidar_msg
        )

        self.processed_publisher.publish(visualization_scan)

        observation = Observation(
            odom=OdomState.from_odometry(odom_msg),
            lidar=np.asarray(processed_lidar_range, dtype=np.float32),
        )
        state = observation.to_policy_array(self.observation_mode)

        return state, observation, lidar_msg.ranges.tolist()

    def _compute_reward(
        self,
        current_observation: Observation,
        next_observation: Observation,
        raw_lidar_range: list[float],
        step_progress: float,
    ) -> tuple[float, dict]:
        reward = 0
        reward_info = {}

        base_reward, base_reward_info = self._calculate_progressive_reward(
            next_observation,
            raw_lidar_range,
            step_progress,
        )
        reward += base_reward
        reward_info.update(base_reward_info)

        for modifier_type, weight in self.reward_modifiers:
            match modifier_type:
                case "wall_proximity":
                    dist_to_wall = min(raw_lidar_range)
                    close_to_wall_penalize_factor = 1 / (
                        1 + np.exp(50 * (dist_to_wall - 0.3))
                    )
                    reward -= reward * close_to_wall_penalize_factor * weight
                    reward_info.update({"dist_to_wall": ["avg", dist_to_wall]})
                case "turn":
                    angular_vel_diff = abs(
                        current_observation.odom.angular_velocity
                        - next_observation.odom.angular_velocity
                    )
                    turning_penalty_factor = 1 - (
                        1 / (1 + np.exp(15 * (angular_vel_diff - 0.5)))
                    )
                    reward -= reward * turning_penalty_factor * weight

        return reward, reward_info

    def _calculate_progressive_reward(
        self,
        next_observation: Observation,
        raw_lidar_range: list[float],
        step_progress: float,
    ) -> tuple[float, dict]:
        reward = 0
        distance_to_goal = math.dist(
            self.goal_position,
            [next_observation.odom.x, next_observation.odom.y],
        )

        if step_progress < 0.02:
            self.progress_not_met_cnt += 1
        else:
            self.progress_not_met_cnt = 0

        reward += step_progress

        if distance_to_goal < self.goal_reach_radius:
            self.goals_reached += 1
            new_x, new_y, _, _ = self.current_waypoints[
                (self.spawn_index + self.goals_reached) % len(self.current_waypoints)
            ]
            self.goal_position = (new_x, new_y)
            self._update_goal_service(new_x, new_y)

        if self.progress_not_met_cnt >= 5:
            reward -= 2

        quaternion = next_observation.odom.quaternion_wxyz()
        if util.has_collided(
            raw_lidar_range, self.collision_range_m
        ) or util.has_flipped_over(quaternion):
            reward -= 2.5

        return reward, {}

    def _clamp_step_progress(self, step_progress: float, linear_speed: float) -> float:
        """
        Clamp spline progress to a physically plausible per-step travel distance.
        Max distance = speed (m/s) * step_duration (s), where
        step_duration = step_sleep_time_ms / 1000. A 1 cm floor handles near-zero
        speed. Sign is preserved so backward motion is represented correctly.
        """
        step_duration_s = self.step_sleep_time_ms / 1000.0
        max_progress = max(abs(linear_speed) * step_duration_s, 0.01)
        return float(np.clip(step_progress, -max_progress, max_progress))

    def _step(self) -> tuple[np.ndarray, float, bool, bool, dict]:

        next_state, next_observation, raw_lidar_range = self._get_observation()
        self._set_simulation_paused(paused=True)

        if self.current_observation is None:
            raise RuntimeError(
                "Current observation is not initialized - call reset first"
            )

        if self.previous_closest_spline_t is None:
            self.previous_closest_spline_t = (
                self.current_track_model.get_closest_point_on_spline(
                    [self.current_observation.odom.x, self.current_observation.odom.y],
                    t_only=True,
                )
            )

        current_closest_spline_t = self.current_track_model.get_closest_point_on_spline(
            [next_observation.odom.x, next_observation.odom.y], t_only=True
        )

        step_progress = self.current_track_model.get_distance_along_track_parametric(
            self.previous_closest_spline_t,
            current_closest_spline_t,
            approximate=True,
        )
        step_progress = self._clamp_step_progress(
            step_progress, next_observation.odom.linear_velocity
        )

        self.previous_closest_spline_t = current_closest_spline_t

        reward, reward_info = self._compute_reward(
            self.current_observation,
            next_observation,
            raw_lidar_range,
            step_progress,
        )
        terminated = self._is_terminated(next_observation, raw_lidar_range)
        truncated = self._is_truncated()

        info = {
            "linear_velocity": ["avg", next_observation.odom.linear_velocity],
            "angular_velocity_diff": [
                "avg",
                abs(
                    next_observation.odom.angular_velocity
                    - self.current_observation.odom.angular_velocity
                ),
            ],
            "traveled distance": ["sum", step_progress],
        }
        info.update(reward_info)

        self.current_observation = next_observation

        return next_state, reward, terminated, truncated, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.step_counter += 1

        clipped_action = np.clip(
            np.asarray(action, dtype=np.float32), self.min_actions, self.max_actions
        )
        lin_vel, steering_angle = clipped_action
        self._set_simulation_paused(paused=False)

        self._set_velocity(lin_vel, steering_angle)

        self._sleep(self.step_sleep_time_ms)

        next_state, reward, terminated, truncated, info = self._step()

        self._set_simulation_paused(paused=True)

        return next_state, reward, terminated, truncated, info

    def _set_velocity(
        self, lin_vel: float, steering_angle: float, wheelbase: float = 0.325
    ):
        angular = util.ackermann_to_twist(steering_angle, lin_vel, wheelbase)
        velocity_msg = Twist()
        velocity_msg.angular.z = float(angular)
        velocity_msg.linear.x = float(lin_vel)
        self.cmd_vel_pub.publish(velocity_msg)

    def _set_simulation_paused(self, paused: bool):
        request = SetBool.Request()
        request.data = paused
        future = self.stepping_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def _call_reset_service(
        self,
        car_x: float,
        car_y: float,
        car_yaw: float,
        goal_x: float,
        goal_y: float,
        car_name: str,
    ):
        request = Reset.Request()
        request.car_name = car_name
        request.gx = float(goal_x)
        request.gy = float(goal_y)
        request.cx = float(car_x)
        request.cy = float(car_y)
        request.cyaw = float(car_yaw)
        request.flag = "car_and_goal"

        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        return future.result()

    def _update_goal_service(self, x: float, y: float):
        request = Reset.Request()
        request.gx = x
        request.gy = y
        request.flag = "goal_only"
        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def set_seed(self, seed):
        pass
