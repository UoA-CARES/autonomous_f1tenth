import math
import os
import random
from abc import ABC, abstractmethod
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
from .util_track_progress import TrackMathDef
from .waypoints import waypoints


class F1tenthEnvironment(Node, ABC):

    def __init__(
        self,
        env_name: str,
        car_name: str,
        reward_range: float = 0.5,
        max_steps: int = 3000,
        collision_range: float = 0.2,
        step_length: float = 0.5,
        lidar_points: int = 10,
        track: str = "track_1",
        observation_mode: str = "lidar_only",
    ):
        super().__init__(f"{env_name}_environment")

        if lidar_points < 1:
            raise ValueError("Make sure number of lidar points is more than 0")

        #####################################################################################################################
        # Init params ----------------------------------------------
        self.name = car_name
        self.reward_range = reward_range
        self.max_steps = max_steps
        self.collision_range = collision_range
        self.step_length = step_length
        self.lidar_points = lidar_points
        self.track = track

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
        self.observation_size = odom_observation_size + self.lidar_points

        self.action_num = 2

        #####################################################################################################################
        # Track params -----------------------------------------
        self.is_multi_track = (
            "multi_track" in self.track or self.track == "staged_tracks"
        )
        if self.is_multi_track:
            _, self.all_track_waypoints = (
                util.get_all_goals_and_waypoints_in_multi_tracks(self.track)
            )
            self.all_track_models = util.get_track_math_defs(self.all_track_waypoints)
            self.curr_track = list(self.all_track_waypoints.keys())[0]
            self.curr_waypoints = self.all_track_waypoints[self.curr_track]
            self.curr_track_model = self.all_track_models[self.curr_track]
        else:
            if "test_track" in self.track:
                track_key = self.track[:-4]
            else:
                track_key = self.track

            self.curr_waypoints = waypoints[track_key]  # from waypoints.py
            self.curr_track_model = TrackMathDef(np.array(self.curr_waypoints)[:, :2])

        if track == "narrow_multi_track":
            self.multi_track_train_eval_split = 12 / 15
        else:
            self.multi_track_train_eval_split = 0.5

        if self.is_multi_track:
            self.eval_track_begin_idx = int(
                len(self.all_track_waypoints) * self.multi_track_train_eval_split
            )
            self.eval_track_idx = 0

        #####################################################################################################################
        # Vehicle params -------------------------------------------
        self.lidar_processing: Literal["avg", "raw"] = "avg"

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
        self._latest_data: tuple[Odometry, LaserScan] | None = None
        self.current_state: np.ndarray | None = None

        self.step_counter = 0
        self.step_progress = 0
        self.goals_reached = 0

        self.prev_closest_point = None
        self.is_eval = False
        self.spawn_index = 0

        self.goal_position = [0, 0]

        self.progress_not_met_cnt = 0
        self.steps_since_last_goal = 0

    def _reset(self, training: bool) -> tuple[np.ndarray, dict]:

        if self.is_multi_track:
            if (
                self.eval_track_begin_idx is not None
                and self.eval_track_begin_idx >= len(self.all_track_waypoints)
            ):
                if self.is_eval:
                    all_track_keys = list(self.all_track_waypoints.keys())
                    self.current_track = all_track_keys[self.eval_track_idx]
                    self.eval_track_idx += 1
                    self.eval_track_idx = self.eval_track_idx % len(all_track_keys)
                else:
                    self.current_track = random.choice(
                        list(self.all_track_waypoints.keys())
                    )
            else:
                if self.is_eval:
                    eval_track_key_list = list(self.all_track_waypoints.keys())[
                        self.eval_track_begin_idx :
                    ]
                    self.current_track = eval_track_key_list[self.eval_track_idx]
                    self.eval_track_idx += 1
                    self.eval_track_idx = self.eval_track_idx % len(eval_track_key_list)
                else:
                    self.current_track = random.choice(
                        list(self.all_track_waypoints.keys())[
                            : self.eval_track_begin_idx
                        ]
                    )
            self.curr_waypoints = self.all_track_waypoints[self.current_track]

        if self.is_eval:
            car_x, car_y, car_yaw, index = self.curr_waypoints[10]
        else:
            car_x, car_y, car_yaw, index = random.choice(self.curr_waypoints)

        self.spawn_index = index
        x, y, _, _ = self.curr_waypoints[
            (
                self.spawn_index + 1
                if self.spawn_index + 1 < len(self.curr_waypoints)
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
        state, full_state, _ = self._get_observation()
        self.current_state = full_state
        self.call_step(pause=True)

        if self.is_multi_track:
            self.curr_track_model = self.all_track_models[self.current_track]

        self.prev_closest_point = self.curr_track_model.get_closest_point_on_spline(
            full_state[:2], t_only=True
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

        self.current_state, _ = self._reset(training)
        return self.current_state

    def message_filter_callback(self, odom: Odometry, lidar: LaserScan) -> None:
        self._latest_data = (odom, lidar)

    def get_data(self, timeout: float = 5.0) -> tuple[Odometry, LaserScan]:
        # Drain anything stale
        self._latest_data = None
        end_time = self.get_clock().now().nanoseconds + int(timeout * 1e9)

        while self.get_clock().now().nanoseconds < end_time:
            rclpy.spin_once(self, timeout_sec=0.01)
            if self._latest_data is not None:
                return self._latest_data

        raise TimeoutError("No synced data received")

    def sleep(self, duration: float) -> None:
        end_time = self.get_clock().now().nanoseconds + int(duration * 1e9)

        while self.get_clock().now().nanoseconds < end_time:
            rclpy.spin_once(self, timeout_sec=0.01)

    def is_terminated(self, state, ranges):
        return util.has_collided(ranges, self.collision_range) or util.has_flipped_over(
            state[2:6]
        )

    def is_truncated(self):
        return self.progress_not_met_cnt >= 5 or self.step_counter >= self.max_steps

    def _get_observation(self):
        odom, lidar = self.get_data()
        odom = util.process_odom(odom)
        num_points = self.lidar_points
        state = []

        match (self.observation_mode):
            case "no_position":
                state += odom[2:]
            case "lidar_only":
                state += odom[-2:]
            case _:
                state += odom
        match self.lidar_processing:
            case "avg":
                processed_lidar_range = util.avg_lidar(lidar, num_points)
                visualized_range = processed_lidar_range
                scan = util.create_lidar_msg(lidar, num_points, visualized_range)
            case "raw":
                processed_lidar_range = np.array(lidar.ranges.tolist())
                processed_lidar_range = np.nan_to_num(
                    processed_lidar_range, posinf=-5, nan=-1, neginf=-5
                ).tolist()
                visualized_range = processed_lidar_range
                scan = util.create_lidar_msg(lidar, num_points, visualized_range)

        self.processed_publisher.publish(scan)

        full_state = odom + processed_lidar_range

        state += processed_lidar_range
        state = np.asarray(state)

        return state, full_state, lidar.ranges

    def compute_reward(
        self, state: np.ndarray, next_state: np.ndarray, raw_lidar_range: LaserScan
    ) -> tuple[float, dict]:
        reward = 0
        reward_info = {}

        base_reward, base_reward_info = self.calculate_progressive_reward(
            state, next_state, raw_lidar_range
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
                    print(
                        f"--- Wall proximity penalty factor: {weight} * {close_to_wall_penalize_factor}"
                    )
                case "turn":
                    # steering_angle1 = twist_to_ackermann(state[7], state[6], L=0.325)
                    angular_vel_diff = abs(state[7] - next_state[7])
                    turning_penalty_factor = 1 - (
                        1 / (1 + np.exp(15 * (angular_vel_diff - 0.5)))
                    )
                    reward -= reward * turning_penalty_factor * weight
                    print(
                        f"--- Turning penalty factor: {weight} * {turning_penalty_factor}"
                    )
        return reward, reward_info

    def calculate_progressive_reward(
        self, state: np.ndarray, next_state: np.ndarray, raw_range: LaserScan
    ):
        reward = 0
        goal_position = self.goal_position
        current_distance = math.dist(goal_position, next_state[:2])

        if self.step_progress < 0.02:
            self.progress_not_met_cnt += 1
        else:
            self.progress_not_met_cnt = 0

        reward += self.step_progress
        self.steps_since_last_goal += 1

        if current_distance < self.reward_range:
            self.goals_reached += 1
            new_x, new_y, _, _ = self.curr_waypoints[
                (self.spawn_index + self.goals_reached) % len(self.curr_waypoints)
            ]
            self.goal_position = [new_x, new_y]
            self.update_goal_service(new_x, new_y)
            self.steps_since_last_goal = 0

        if self.progress_not_met_cnt >= 5:
            reward -= 2

        if util.has_collided(raw_range, self.collision_range) or util.has_flipped_over(
            next_state[2:6]
        ):
            reward -= 2.5

        info = {}
        return reward, info

    def _step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:

        next_state, full_next_state, raw_lidar_range = self._get_observation()
        self.call_step(pause=True)

        if not self.prev_closest_point:
            self.prev_closest_point = self.curr_track_model.get_closest_point_on_spline(
                self.current_state[:2], t_only=True
            )

        t2 = self.curr_track_model.get_closest_point_on_spline(
            full_next_state[:2], t_only=True
        )
        self.step_progress = self.curr_track_model.get_distance_along_track_parametric(
            self.prev_closest_point, t2, approximate=True
        )

        self.prev_closest_point = t2

        if abs(self.step_progress) > (full_next_state[6] / 10 * 3):
            self.step_progress = full_next_state[6] / 10 * 0.8

        reward, reward_info = self.compute_reward(
            self.current_state, full_next_state, raw_lidar_range
        )
        terminated = self.is_terminated(full_next_state, raw_lidar_range)
        truncated = self.is_truncated()

        info = {
            "linear_velocity": ["avg", full_next_state[6]],
            "angular_velocity_diff": [
                "avg",
                abs(full_next_state[7] - self.current_state[7]),
            ],
            "traveled distance": ["sum", self.step_progress],
        }
        info.update(reward_info)

        self.current_state = full_next_state

        return next_state, reward, terminated, truncated, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.step_counter += 1

        lin_vel, steering_angle = action
        self.call_step(pause=False)

        self.set_velocity(lin_vel, steering_angle)

        self.sleep(self.step_length)

        next_state, reward, terminated, truncated, info = self._step(action)

        self.call_step(pause=True)

        self.current_state = next_state

        info = {}
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
