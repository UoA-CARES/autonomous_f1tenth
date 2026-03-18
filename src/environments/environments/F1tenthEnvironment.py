import queue
import time
from pathlib import Path
from typing import Literal

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Twist
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy import Future
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_srvs.srv import SetBool

from environment_interfaces.srv import Reset

from .util import (
    ackermann_to_twist,
    get_all_goals_and_waypoints_in_multi_tracks,
    get_track_math_defs,
)
from .util_track_progress import TrackMathDef
from .waypoints import waypoints


class F1tenthEnvironment(Node):

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
        config_path="/home/anyone/autonomous_f1tenth/src/environments/config/config.yaml",
    ):
        super().__init__(env_name + "_environment")

        if lidar_points < 1:
            raise ValueError("Make sure number of lidar points is more than 0")

        #####################################################################################################################
        # Init params ----------------------------------------------
        self.NAME = car_name
        self.REWARD_RANGE = reward_range
        self.MAX_STEPS = max_steps
        self.COLLISION_RANGE = collision_range
        self.STEP_LENGTH = step_length
        self.LIDAR_POINTS = lidar_points
        self.TRACK = track
        self.ODOM_OBSERVATION_MODE = observation_mode

        #####################################################################################################################
        # Network params ---------------------------------------------
        # configure odom observation size:
        match observation_mode:
            case "lidar_only":
                odom_observation_size = 2
            case "no_position":
                odom_observation_size = 6
            case _:
                odom_observation_size = 10
        self.OBSERVATION_SIZE = odom_observation_size + self.LIDAR_POINTS

        self.ACTION_NUM = 2

        #####################################################################################################################
        # Environment params -----------------------------------------
        self.is_multi_track = (
            "multi_track" in self.TRACK or self.TRACK == "staged_tracks"
        )
        if self.is_multi_track:
            _, self.all_track_waypoints = get_all_goals_and_waypoints_in_multi_tracks(
                self.TRACK
            )
            self.ALL_TRACK_MODELS = get_track_math_defs(self.all_track_waypoints)
            self.CURR_TRACK = list(self.all_track_waypoints.keys())[0]
            self.CURR_WAYPOINTS = self.all_track_waypoints[self.CURR_TRACK]
            self.CURR_TRACK_MODEL = self.ALL_TRACK_MODELS[self.CURR_TRACK]
        else:
            if "test_track" in self.TRACK:
                track_key = self.TRACK[0:-4]
            else:
                track_key = self.TRACK
            self.CURR_WAYPOINTS = waypoints[track_key]  # from waypoints.py
            self.CURR_TRACK_MODEL = TrackMathDef(np.array(self.CURR_WAYPOINTS)[:, :2])

        #####################################################################################################################
        # Vehicle params -------------------------------------------
        self.LIDAR_PROCESSING: Literal["avg", "raw"] = "avg"

        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        self.MAX_ACTIONS = np.asarray(
            [config["actions"]["max_speed"], config["actions"]["max_turn"]]
        )
        self.MIN_ACTIONS = np.asarray(
            [config["actions"]["min_speed"], config["actions"]["min_turn"]]
        )

        #####################################################################################################################
        # Pub/Sub ----------------------------------------------------
        self.CMD_VEL_PUB = self.create_publisher(Twist, f"/{self.NAME}/cmd_vel", 1)

        self.ODOM_SUB = Subscriber(
            self,
            Odometry,
            f"/{self.NAME}/odometry",
        )

        self.LIDAR_SUB = Subscriber(
            self,
            LaserScan,
            f"/{self.NAME}/scan",
        )

        self.PROCESSED_PUBLISHER = self.create_publisher(
            LaserScan, f"/{self.NAME}/processed_scan", 1
        )

        #####################################################################################################################
        # Message filter ---------------------------------------------
        self.MESSAGE_FILTER = ApproximateTimeSynchronizer(
            [self.ODOM_SUB, self.LIDAR_SUB],
            1,
            0.1,
        )
        self.MESSAGE_FILTER.registerCallback(self.message_filter_callback)

        # Reset Client -----------------------------------------------
        self.RESET_CLIENT = self.create_client(Reset, env_name + "_reset")
        while not self.RESET_CLIENT.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("reset service not available, waiting again...")

        # Stepping Client ---------------------------------------------
        self.STEPPING_CLIENT = self.create_client(SetBool, "stepping_service")
        while not self.STEPPING_CLIENT.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("stepping service not available, waiting again...")

        #####################################################################################################################
        # Initialise vars ---------------------------------------------

        # Loop vars
        self._data_queue: queue.Queue[tuple[Odometry, LaserScan]] = queue.Queue(
            maxsize=1
        )
        self.STEP_COUNTER = 0
        self.STEP_PROGRESS = 0
        self.GOALS_REACHED = 0
        self.CURR_STATE = None
        self.PREV_CLOSEST_POINT = None
        self.IS_EVAL = False
        self.SPAWN_INDEX = 0

        # Futures
        self.LAST_STATE = Future()
        self.ODOM_OBSERVATION_FUTURE = Future()

        #####################################################################################################################

    def reset(self):
        raise NotImplementedError("reset() not implemented")

    def message_filter_callback(self, odom: Odometry, lidar: LaserScan) -> None:
        self._data_queue.put((odom, lidar))

    def get_data(self, timeout: float = 5.0) -> tuple[Odometry, LaserScan]:
        # Drain anything that arrived before we asked - queue should be at most size 1,
        # so this is just to ensure we don't get stale data after reset or similar
        try:
            self._data_queue.get_nowait()
        except queue.Empty:
            pass
        # Block until a fresh pair arrives after this point
        try:
            return self._data_queue.get(timeout=timeout)
        except queue.Empty as e:
            raise TimeoutError("No synced data received from the car") from e

    def sleep(self, duration: float):
        end_time = self.get_clock().now().nanoseconds + int(duration * 1e9)

        while self.get_clock().now().nanoseconds < end_time:
            rclpy.spin_once(self, timeout_sec=0.01)

    def step(self, action):
        self.STEP_COUNTER += 1

        state = self.get_observation()

        lin_vel, steering_angle = action
        self.set_velocity(lin_vel, steering_angle)

        self.call_step(pause=False)
        self.sleep(self.STEP_LENGTH)
        next_state = self.get_observation()
        self.call_step(pause=True)

        reward = self.compute_reward(state, next_state)
        terminated = self.is_terminated(next_state)
        truncated = self.STEP_COUNTER >= self.MAX_STEPS

        info = {}
        return next_state, reward, terminated, truncated, info

    def get_observation(self):
        raise NotImplementedError("get_observation() not implemented")

    def compute_reward(self, state, next_state):
        raise NotImplementedError("compute_reward() not implemented")

    def is_terminated(self, state):
        raise NotImplementedError("is_terminated() not implemented")

    def set_velocity(self, lin_vel: float, steering_angle: float, L: float = 0.325):
        angular = ackermann_to_twist(steering_angle, lin_vel, L)
        velocity_msg = Twist()
        velocity_msg.angular.z = float(angular)
        velocity_msg.linear.x = float(lin_vel)
        self.CMD_VEL_PUB.publish(velocity_msg)

    def call_step(self, pause: bool):
        request = SetBool.Request()
        request.data = pause
        future = self.STEPPING_CLIENT.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def call_reset_service(
        self,
        car_x: float,
        car_y: float,
        car_Y: float,
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
        request.cyaw = float(car_Y)
        request.flag = "car_and_goal"

        future = self.RESET_CLIENT.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        return future.result()

    def update_goal_service(self, x: float, y: float):
        request = Reset.Request()
        request.gx = x
        request.gy = y
        request.flag = "goal_only"
        future = self.RESET_CLIENT.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def set_seed(self, seed):
        pass
