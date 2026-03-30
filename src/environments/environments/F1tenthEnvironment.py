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
from .observation_types import Observation, OdomState
from .waypoints import waypoints


class F1tenthEnvironment(Node, ABC):

    def __init__(
        self,
        env_name: str,
        car_name: str,
        reward_range: float = 0.5,
        max_steps: int = 3000,
        collision_range: float = 0.2,
        step_sleep_time: float = 0.5,
        lidar_observation_size: int = 10,
        track: str = "track_1",
        observation_mode: str = "lidar_only",
        train_eval_split: float = 0.5,
    ):
        super().__init__(f"{env_name}_environment")

        if lidar_observation_size < 1:
            raise ValueError("Make sure number of lidar points is more than 0")

        #####################################################################################################################
        # Init params ----------------------------------------------
        self.name = car_name
        self.reward_range = reward_range
        self.max_steps = max_steps
        self.collision_range = collision_range
        self.step_sleep_time = step_sleep_time
        self.lidar_observation_size = lidar_observation_size
        self.track = track
        self.track_train_eval_split = train_eval_split

        #####################################################################################################################
        # Observation params ---------------------------------------------
        self.observation_mode = observation_mode
        match observation_mode:
            case "lidar_only":
                odom_observation_size = 2
            case "no_position":
                odom_observation_size = 6
            case _:
                odom_observation_size = 10
        self.observation_size = odom_observation_size + self.lidar_observation_size

        self.action_num = 2

        #####################################################################################################################
        # Track params -----------------------------------------
        self.tracks = self._load_tracks(self.track)
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

        #####################################################################################################################
        # Vehicle params -------------------------------------------
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

        #####################################################################################################################
        # Pub/Sub ----------------------------------------------------
        self.cmd_vel_pub = self.create_publisher(Twist, f"/{self.name}/cmd_vel", 1)

        sub_depth = 3
        qos = QoSProfile(depth=sub_depth)
        self.odom_sub = Subscriber(
            self,
            Odometry,
            f"/{self.name}/odometry",
            qos_profile=qos,
        )

        self.lidar_sub = Subscriber(
            self,
            LaserScan,
            f"/{self.name}/scan",
            qos_profile=qos,
        )

        self.processed_publisher = self.create_publisher(
            LaserScan, f"/{self.name}/processed_scan", 1
        )

        self.message_filter = ApproximateTimeSynchronizer(
            [self.odom_sub, self.lidar_sub],
            sub_depth,
            0.1,
        )
        self.message_filter.registerCallback(self.message_filter_callback)

        # Reset Client -----------------------------------------------
        self.reset_client = self.create_client(Reset, f"{env_name}_reset")
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("reset service not available, waiting again...")

        # Stepping Client ---------------------------------------------
        self.stepping_client = self.create_client(SetBool, "stepping_service")
        while not self.stepping_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("stepping service not available, waiting again...")

        #####################################################################################################################
        # Reward configuration -----------------------------------------
        self.base_reward_function: Literal["progressive"] = "progressive"

        self.reward_modifiers: list[tuple[Literal["turn", "wall_proximity"], float]] = [
            ("turn", 0.3),
            ("wall_proximity", 0.7),
        ]

        #####################################################################################################################
        # Initialise loop vars ---------------------------------------------
        self.latest_data: tuple[Odometry, LaserScan] | None = None
        self.current_observation: Observation | None = None

        self.step_counter = 0
        self.step_progress = 0
        self.goals_reached = 0

        self.previous_closest_point = None
        self.is_eval = False
        self.spawn_index = 0

        self.goal_position = [0, 0]

        self.progress_not_met_cnt = 0
        self.steps_since_last_goal = 0

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

    def _reset(self, _training: bool) -> tuple[np.ndarray, dict]:
        self.current_track = self._select_track_name()
        self.current_waypoints = self.tracks[self.current_track]
        self.current_track_model = self.track_models[self.current_track]

        if self.is_eval:
            car_x, car_y, car_yaw, index = self.current_waypoints[10]
        else:
            car_x, car_y, car_yaw, index = random.choice(self.current_waypoints)

        self.spawn_index = index
        x, y, _, _ = self.current_waypoints[
            (
                self.spawn_index + 1
                if self.spawn_index + 1 < len(self.current_waypoints)
                else 0
            )
        ]

        # point toward next goal
        self.goal_position = [x, y]
        self.call_reset_service(
            car_x=car_x,
            car_y=car_y,
            car_yaw=car_yaw,
            goal_x=x,
            goal_y=y,
            car_name=self.name,
        )

        self.call_step(pause=False)
        state, observation, _ = self._get_observation()
        self.current_observation = observation
        self.call_step(pause=True)

        self.previous_closest_point = (
            self.current_track_model.get_closest_point_on_spline(
                [observation.odom.x, observation.odom.y],
                t_only=True,
            )
        )

        info = {}
        return state, info

    def reset(self, training: bool = True) -> np.ndarray:
        self.step_counter = 0
        self.step_progress = 0
        self.goals_reached = 0

        self.steps_since_last_goal = 0
        self.progress_not_met_cnt = 0

        self.is_eval = not training

        self.set_velocity(0, 0)

        state, _ = self._reset(training)
        return state

    def message_filter_callback(self, odom: Odometry, lidar: LaserScan) -> None:
        self.latest_data = (odom, lidar)

    def get_data(self, timeout: float = 5.0) -> tuple[Odometry, LaserScan]:
        # Drain anything stale
        self.latest_data = None
        end_time = self.get_clock().now().nanoseconds + int(timeout * 1e9)

        while self.get_clock().now().nanoseconds < end_time:
            rclpy.spin_once(self, timeout_sec=0.01)
            if self.latest_data is not None:
                return self.latest_data

        raise TimeoutError("No synced data received")

    def sleep(self, duration: float) -> None:
        end_time = self.get_clock().now().nanoseconds + int(duration * 1e9)

        while self.get_clock().now().nanoseconds < end_time:
            rclpy.spin_once(self, timeout_sec=0.01)

    def is_terminated(self, observation: Observation, ranges: list[float]):
        quaternion = observation.odom.quaternion_wxyz()
        return util.has_collided(ranges, self.collision_range) or util.has_flipped_over(
            quaternion
        )

    def is_truncated(self):
        return self.progress_not_met_cnt >= 5 or self.step_counter >= self.max_steps

    def _get_observation(self) -> tuple[np.ndarray, Observation, list[float]]:
        odom_msg, lidar_msg = self.get_data()

        match self.lidar_reduction_mode:
            case "avg":
                processed_lidar_range = util.avg_lidar(
                    lidar_msg, self.lidar_observation_size
                )
                visualized_range = processed_lidar_range
                scan = util.create_lidar_msg(
                    lidar_msg, self.lidar_observation_size, visualized_range
                )
            case "raw":
                processed_lidar_range = np.array(lidar_msg.ranges.tolist())
                processed_lidar_range = np.nan_to_num(
                    processed_lidar_range, posinf=-5, nan=-1, neginf=-5
                ).tolist()
                visualized_range = processed_lidar_range
                scan = util.create_lidar_msg(
                    lidar_msg, self.lidar_observation_size, visualized_range
                )

        self.processed_publisher.publish(scan)

        observation = Observation(
            odom=OdomState.from_odometry(odom_msg),
            lidar=np.asarray(processed_lidar_range, dtype=np.float32),
        )
        state = observation.to_policy_array(self.observation_mode)

        return state, observation, lidar_msg.ranges.tolist()

    def compute_reward(
        self,
        current_observation: Observation,
        next_observation: Observation,
        raw_lidar_range: list[float],
    ) -> tuple[float, dict]:
        reward = 0
        reward_info = {}

        base_reward, base_reward_info = self.calculate_progressive_reward(
            current_observation, next_observation, raw_lidar_range
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

    def calculate_progressive_reward(
        self,
        current_observation: Observation,
        next_observation: Observation,
        raw_range: list[float],
    ):
        reward = 0
        goal_position = self.goal_position
        current_distance = math.dist(
            goal_position,
            [next_observation.odom.x, next_observation.odom.y],
        )

        if self.step_progress < 0.02:
            self.progress_not_met_cnt += 1
        else:
            self.progress_not_met_cnt = 0

        reward += self.step_progress
        self.steps_since_last_goal += 1

        if current_distance < self.reward_range:
            self.goals_reached += 1
            new_x, new_y, _, _ = self.current_waypoints[
                (self.spawn_index + self.goals_reached) % len(self.current_waypoints)
            ]
            self.goal_position = [new_x, new_y]
            self.update_goal_service(new_x, new_y)
            self.steps_since_last_goal = 0

        if self.progress_not_met_cnt >= 5:
            reward -= 2

        quaternion = next_observation.odom.quaternion_wxyz()
        if util.has_collided(raw_range, self.collision_range) or util.has_flipped_over(
            quaternion
        ):
            reward -= 2.5

        info = {}
        return reward, info

    def _clamp_step_progress(self, step_progress: float, linear_speed: float) -> float:
        """
        Clamp spline progress to a physically plausible per-step travel distance.
        Max distance = speed (m/s) * step_length (s). A 1 cm floor handles near-zero
        speed. Sign is preserved so backward motion is represented correctly.
        """
        max_progress = max(abs(linear_speed) * self.step_sleep_time, 0.01)
        return float(np.clip(step_progress, -max_progress, max_progress))

    def _step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:

        next_state, next_observation, raw_lidar_range = self._get_observation()
        self.call_step(pause=True)

        if self.current_observation is None:
            raise RuntimeError(
                "Current observation is not initialized - call reset first"
            )

        if self.previous_closest_point is None:
            self.previous_closest_point = (
                self.current_track_model.get_closest_point_on_spline(
                    [self.current_observation.odom.x, self.current_observation.odom.y],
                    t_only=True,
                )
            )

        current_closest_point = self.current_track_model.get_closest_point_on_spline(
            [next_observation.odom.x, next_observation.odom.y], t_only=True
        )

        self.step_progress = (
            self.current_track_model.get_distance_along_track_parametric(
                self.previous_closest_point, current_closest_point, approximate=True
            )
        )
        self.step_progress = self._clamp_step_progress(
            self.step_progress, next_observation.odom.linear_velocity
        )

        self.previous_closest_point = current_closest_point

        reward, reward_info = self.compute_reward(
            self.current_observation, next_observation, raw_lidar_range
        )
        terminated = self.is_terminated(next_observation, raw_lidar_range)
        truncated = self.is_truncated()

        info = {
            "linear_velocity": ["avg", next_observation.odom.linear_velocity],
            "angular_velocity_diff": [
                "avg",
                abs(
                    next_observation.odom.angular_velocity
                    - self.current_observation.odom.angular_velocity
                ),
            ],
            "traveled distance": ["sum", self.step_progress],
        }
        info.update(reward_info)

        self.current_observation = next_observation

        return next_state, reward, terminated, truncated, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.step_counter += 1

        lin_vel, steering_angle = action
        self.call_step(pause=False)

        self.set_velocity(lin_vel, steering_angle)

        self.sleep(self.step_sleep_time)

        next_state, reward, terminated, truncated, info = self._step(action)

        self.call_step(pause=True)

        return next_state, reward, terminated, truncated, info

    def set_velocity(
        self, lin_vel: float, steering_angle: float, wheelbase: float = 0.325
    ):
        angular = util.ackermann_to_twist(steering_angle, lin_vel, wheelbase)
        velocity_msg = Twist()
        velocity_msg.angular.z = float(angular)
        velocity_msg.linear.x = float(lin_vel)
        self.cmd_vel_pub.publish(velocity_msg)

    def call_step(self, pause: bool):
        request = SetBool.Request()
        request.data = pause
        future = self.stepping_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def call_reset_service(
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

    def update_goal_service(self, x: float, y: float):
        request = Reset.Request()
        request.gx = x
        request.gy = y
        request.flag = "goal_only"
        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def set_seed(self, seed):
        pass
