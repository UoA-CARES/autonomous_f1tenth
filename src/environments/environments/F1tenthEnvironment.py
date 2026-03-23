import os
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
        env_name,
        car_name,
        reward_range=0.5,
        max_steps=3000,
        collision_range=0.2,
        step_length=0.5,
        lidar_points=10,
        track="track_1",
        observation_mode="lidar_only",
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
        # Network params ---------------------------------------------
        # configure odom observation size:
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
        # Environment params -----------------------------------------
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

        #####################################################################################################################
        # Message filter ---------------------------------------------
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
        # Initialise loop vars ---------------------------------------------
        self._latest_data: tuple[Odometry, LaserScan] | None = None
        self.current_state: np.ndarray | None = None

        self.step_counter = 0
        self.step_progress = 0
        self.goals_reached = 0

        self.prev_closest_point = None
        self.is_eval = False
        self.spawn_index = 0

    @abstractmethod
    def _reset(self, training: bool) -> tuple[np.ndarray, dict]: ...

    def reset(self, training: bool = True) -> np.ndarray:
        self.step_counter = 0
        self.step_progress = 0
        self.goals_reached = 0

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

    @abstractmethod
    def _step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]: ...

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
