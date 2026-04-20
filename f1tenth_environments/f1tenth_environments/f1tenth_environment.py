import math
import random
import re
from abc import ABC

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile
from ros_gz_interfaces.srv import ControlWorld, SetEntityPose
from sensor_msgs.msg import LaserScan

from . import geometry_utils, lidar_processor, msg_utils, track_utils, waypoints
from .state_builder import LidarMode, OdomMode, StateBuilder, StateData


class F1tenthEnvironment(Node, ABC):
    """Base ROS2 node for F1Tenth RL environments.

    Manages the step/reset loop, observation collection, and reward computation.
    Subclass to implement task-specific behaviour.
    """

    def __init__(
        self,
        env_name: str,
        goal_reach_radius_m: float,
        max_steps: int,
        collision_range_m: float,
        step_sleep_time_ms: float,
        lidar_state_size: int,
        track: str,
        odom_mode: OdomMode,
        lidar_mode: LidarMode,
        train_eval_split: float,
        max_speed: float,
        min_speed: float,
        max_turn: float,
        wall_proximity_reward_weight: float,
        turn_reward_weight: float,
        stall_progress_threshold_m: float,
        stall_limit_steps: int,
        collision_penalty: float,
    ):
        """
        Initialize the F1Tenth RL environment node.

        Args:
            env_name: Name of the environment instance.
            goal_reach_radius_m: Radius (meters) for goal completion.
            max_steps: Maximum steps per episode.
            collision_range_m: Lidar collision threshold (meters).
            step_sleep_time_ms: Step duration in milliseconds.
            lidar_state_size: Number of lidar points in state.
            track: Track name or multi-track specifier.
            odom_mode: Odometry mode for state builder.
            lidar_mode: Lidar mode for state builder.
            train_eval_split: Fraction of tracks for training.
            max_speed: Maximum allowed speed.
            min_speed: Minimum allowed speed.
            max_turn: Maximum allowed steering angle (radians).
            wall_proximity_reward_weight: Weight for wall proximity in reward.
            turn_reward_weight: Weight for turn smoothness in reward.
            stall_progress_threshold_m: Progress threshold to count as stall (meters).
            stall_limit_steps: Number of consecutive stall steps before truncation.
            collision_penalty: Fixed penalty subtracted from reward on collision.
        """
        super().__init__(f"{env_name}_environment")

        self.car_name = "f1tenth"
        self.goal_reach_radius_m = goal_reach_radius_m
        self.max_steps = max_steps
        self.collision_range_m = collision_range_m
        self.step_sleep_time_ms = step_sleep_time_ms
        self.train_eval_split = train_eval_split

        self.wall_proximity_reward_weight = wall_proximity_reward_weight
        self.turn_reward_weight = turn_reward_weight
        self.stall_progress_threshold_m = stall_progress_threshold_m
        self.stall_limit_steps = stall_limit_steps
        self.collision_penalty = collision_penalty

        self.wheelbase_m = 0.325

        # Setup Tracks and Waypoints for tracking car progress
        self.tracks = self._load_tracks(track)
        self.track_progress_models = track_utils.get_track_progress_models(self.tracks)
        self.track_names = list(self.tracks.keys())

        self.current_track = self.track_names[0]
        self.current_waypoints = self.tracks[self.current_track]
        self.current_track_model = self.track_progress_models[self.current_track]

        self.eval_tracks_start_idx: int = int(
            len(self.track_names) * self.train_eval_split
        )
        self.eval_track_idx = 0

        # Setup Publishers and Subscribers
        self.latest_data: tuple[Odometry, LaserScan] | None = None
        self.previous_state_data: StateData | None = None

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

        self.state_scan_pub = self.create_publisher(
            LaserScan, f"/{self.car_name}/state_scan", 1
        )

        sync_slop_sec = 0.1
        self.message_filter = ApproximateTimeSynchronizer(
            [self.odom_sub, self.lidar_sub],
            sub_depth,
            sync_slop_sec,
        )
        self.message_filter.registerCallback(self._message_filter_callback)

        self.world_control_client = self.create_client(
            ControlWorld, "world/empty/control"
        )
        while not self.world_control_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "world control service not available, waiting again..."
            )

        self.set_pose_client = self.create_client(SetEntityPose, "world/empty/set_pose")
        while not self.set_pose_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("set_pose service not available, waiting again...")

        # Setup Environment Parameters
        if lidar_mode == "raw":
            # Ensure we have received at least one message to get the correct lidar size
            _, lidar_data = self._get_data()
            lidar_state_size = len(lidar_data.ranges)

        if lidar_state_size < 1:
            raise ValueError("Make sure number of lidar points is more than 0")

        max_turn = abs(float(max_turn))
        self.max_actions = np.asarray([max_speed, max_turn], dtype=np.float32)
        self.min_actions = np.asarray([min_speed, -max_turn], dtype=np.float32)

        self.state_builder = StateBuilder(
            odom_mode=odom_mode,
            lidar_mode=lidar_mode,
            lidar_state_size=lidar_state_size,
            min_speed=min_speed,
            max_speed=max_speed,
            max_turn=max_turn,
            wheelbase_m=self.wheelbase_m,
        )

        self.observation_size = self.state_builder.policy_state_size
        self.action_num = 2  # linear and angular velocity

        # Reward Weights as precalculated factors to modify the base reward based on progress,
        # to encourage desirable behaviour
        self.wall_proximity_reward_weight = 0.7
        self.turn_reward_weight = 0.3

        # Loop Parameters
        self.step_counter = 0
        self.goals_reached = 0

        self.is_eval = False
        self.spawn_index = 0

        self.goal_position: tuple[float, float] = (0.0, 0.0)

        self.stall_counter = 0

        self.opponent_car_names = self._discover_opponent_car_names()

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

    def _load_tracks(self, track_name: str) -> dict:
        if "multi_track" in track_name or track_name == "staged_tracks":
            _, all_track_waypoints = (
                track_utils.get_all_goals_and_waypoints_in_multi_tracks(track_name)
            )
            return all_track_waypoints

        return {track_name: waypoints.waypoints[track_name]}

    def _get_track_split_keys(self) -> tuple[list[str], list[str]]:
        split_idx = min(self.eval_tracks_start_idx, len(self.track_names))
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

    def _reset_positions(self) -> None:
        self.current_track = self._select_track_name()
        self.current_waypoints = self.tracks[self.current_track]
        self.current_track_model = self.track_progress_models[self.current_track]

        if self.is_eval:
            eval_spawn_waypoint_idx = 10
            car_x, car_y, car_yaw, index = self.current_waypoints[
                eval_spawn_waypoint_idx
            ]
        else:
            car_x, car_y, car_yaw, index = random.choice(self.current_waypoints)

        self.spawn_index = index
        goal_x, goal_y, _, _ = self.current_waypoints[
            (self.spawn_index + 1) % len(self.current_waypoints)
        ]

        self.goal_position = (goal_x, goal_y)
        self._set_model_pose(
            model_name=self.car_name,
            x=float(car_x),
            y=float(car_y),
            z=0.0,
            yaw=float(car_yaw),
        )

        for opponent_order, opponent_car_name in enumerate(self.opponent_car_names):
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

    def _reset(self) -> np.ndarray:
        self._reset_positions()

        self._set_simulation_paused(paused=False)
        state_data = self._build_state_data()
        self.previous_state_data = state_data
        self._set_simulation_paused(paused=True)

        return state_data.state

    def reset(self, training: bool = True) -> np.ndarray:
        self.step_counter = 0
        self.goals_reached = 0

        self.stall_counter = 0

        self.is_eval = not training

        self._set_velocity(0, 0)

        state = self._reset()
        return state

    def _message_filter_callback(self, odom: Odometry, lidar: LaserScan) -> None:
        self.latest_data = (odom, lidar)

    def _get_data(self, timeout: float = 5.0) -> tuple[Odometry, LaserScan]:
        # Drain anything stale
        self.latest_data = None
        end_time = self.get_clock().now().nanoseconds + int(timeout * 1e9)
        spin_timeout_sec = 0.01

        while self.get_clock().now().nanoseconds < end_time:
            rclpy.spin_once(self, timeout_sec=spin_timeout_sec)
            if self.latest_data is not None:
                return self.latest_data

        raise TimeoutError("No synced data received")

    def _sleep(self, duration_ms: float) -> None:
        """
        Sleep while still processing incoming messages, to allow for callbacks to run.

        Critical that this uses self.get_clock() for timekeeping, to ensure it works properly with simulated time.
        """
        end_time = self.get_clock().now().nanoseconds + int(duration_ms * 1e6)
        spin_timeout_sec = 0.01

        while self.get_clock().now().nanoseconds < end_time:
            rclpy.spin_once(self, timeout_sec=spin_timeout_sec)

    def _is_terminated(self, state_data: StateData) -> bool:
        return self._has_collision_or_flip(state_data)

    def _has_collision_or_flip(self, state_data: StateData) -> bool:
        quaternion = state_data.quaternion_wxyz()
        return lidar_processor.has_collided(
            state_data.lidar_sanitised_data, self.collision_range_m
        ) or geometry_utils.has_flipped_over(quaternion)

    def _is_truncated(self) -> bool:
        return (
            self.stall_counter >= self.stall_limit_steps
            or self.step_counter >= self.max_steps
        )

    def _build_state_data(self) -> StateData:
        odom_msg, lidar_msg = self._get_data()
        state_data = self.state_builder.build_state(odom_msg, lidar_msg)
        self.state_scan_pub.publish(state_data.lidar_state_scan)
        return state_data

    def _update_goal_progress(self, next_state_data: StateData) -> None:
        next_x, next_y = next_state_data.position_xy()
        distance_to_goal = math.dist(self.goal_position, [next_x, next_y])
        if distance_to_goal < self.goal_reach_radius_m:
            self._advance_goal()

    def _advance_goal(self) -> None:
        """
        Move the target goal to the next waypoint on the track.
        """
        self.goals_reached += 1
        new_x, new_y, _, _ = self.current_waypoints[
            (self.spawn_index + self.goals_reached) % len(self.current_waypoints)
        ]
        self.goal_position = (new_x, new_y)

    def _calculate_progress_reward(
        self,
        step_progress: float,
    ) -> float:
        """
        Normalize step progress to [0, 1] for reward calculation.

        Args:
            step_progress: Track progress in meters.
        Returns:
            Normalized progress reward in [0, 1].
        """
        if step_progress < self.stall_progress_threshold_m:
            self.stall_counter += 1
        else:
            self.stall_counter = 0

        # Estimated based on the max speed and step duration,
        # to give a reward of 1.0 for making maximum possible progress in a step,
        # and scale down linearly from there.
        max_progress_per_step_m = float(self.max_actions[0]) * (
            self.step_sleep_time_ms / 1000.0
        )

        if max_progress_per_step_m <= 0.0:
            return 0.0

        normalized_progress = step_progress / max_progress_per_step_m
        return float(np.clip(normalized_progress, 0.0, 1.0))

    def _compute_reward(
        self,
        previous_state_data: StateData,
        current_state_data: StateData,
    ) -> tuple[float, dict]:
        """
        Compute reward and info for the transition from previous_state_data to current_state_data.

        The main reward is a normalized progress value in [0, 1], scaled by wall and turn discounts,
        and penalized by a fixed collision penalty. The typical reward range is [0, 1] for normal steps,
        with a minimum of -1.0 if a collision occurs (reward - collision_penalty).

        Args:
            previous_state_data: State before action.
            current_state_data: State after action.
        Returns:
            reward: Scalar reward value (normalized, usually in [0, 1], can be as low as -1.0 on collision).
            info: Dict of reward components and diagnostics.
        """
        track_progress = self._compute_step_progress(
            previous_state_data=previous_state_data,
            current_state_data=current_state_data,
        )

        progress_reward = self._calculate_progress_reward(track_progress)

        prev_angular_velocity = previous_state_data.angular_velocity()
        curr_angular_velocity = current_state_data.angular_velocity()

        dist_to_wall = float(np.min(current_state_data.lidar_sanitised_data))
        angular_velocity_change = abs(prev_angular_velocity - curr_angular_velocity)

        wall_threshold_m = 0.3
        wall_k = 50.0
        wall_factor = 1.0 / (1.0 + np.exp(wall_k * (dist_to_wall - wall_threshold_m)))

        turn_threshold_rads = 0.5
        turn_k = 15.0
        turn_factor = 1.0 / (
            1.0 + np.exp(-turn_k * (angular_velocity_change - turn_threshold_rads))
        )

        wall_discount = 1.0 - (wall_factor * self.wall_proximity_reward_weight)
        turn_discount = 1.0 - (turn_factor * self.turn_reward_weight)

        reward = progress_reward * wall_discount * turn_discount

        collision = self._has_collision_or_flip(current_state_data)
        if collision:
            reward -= self.collision_penalty

        reward_info = {
            "track_progress": track_progress,
            "progress_reward": progress_reward,
            "dist_to_wall": dist_to_wall,
            "angular_velocity_change": angular_velocity_change,
            "wall_factor": wall_factor,
            "turn_factor": turn_factor,
            "wall_discount": wall_discount,
            "turn_discount": turn_discount,
            "collision": float(collision),
        }

        return reward, reward_info

    def _clamp_step_progress(self, step_progress: float, linear_speed: float) -> float:
        """
        Clamp spline progress to a physically plausible per-step travel distance.
        Max distance = speed (m/s) * step_duration (s), where
        step_duration = step_sleep_time_ms / 1000. A 1 cm floor handles near-zero
        speed. Sign is preserved so backward motion is represented correctly.
        """
        step_duration_s = self.step_sleep_time_ms / 1000.0
        minimum_progress_floor_m = 0.01
        max_progress = max(
            abs(linear_speed) * step_duration_s, minimum_progress_floor_m
        )
        return float(np.clip(step_progress, -max_progress, max_progress))

    def _compute_step_progress(
        self,
        previous_state_data: StateData,
        current_state_data: StateData,
    ) -> float:
        """
        Compute progress along the track between previous and current state.

        Args:
            previous_state_data: State before action.
            current_state_data: State after action.
        Returns:
            step_progress: Track progress in meters.
        """
        prev_x, prev_y = previous_state_data.position_xy()
        prev_spline_t = self.current_track_model.world_coord_to_spline_coord(
            np.asarray([prev_x, prev_y], dtype=np.float64)
        )

        curr_x, curr_y = current_state_data.position_xy()
        curr_spline_t = self.current_track_model.world_coord_to_spline_coord(
            np.asarray([curr_x, curr_y], dtype=np.float64)
        )

        step_progress = self.current_track_model.linear_distance_between_spline_coords(
            prev_spline_t,
            curr_spline_t,
        )
        step_progress = self._clamp_step_progress(
            step_progress, current_state_data.linear_velocity()
        )
        return step_progress

    def _transition(self) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Perform a transition: step the simulation, compute reward, and update state.
        Returns:
            observation: Current state after action.
            reward: Reward for the transition.
            terminated: True if episode ended by termination condition.
            truncated: True if episode ended by truncation (timeout/stall).
            info: Additional diagnostic info dict.
        """
        current_state_data = self._build_state_data()
        self._set_simulation_paused(paused=True)

        reward, reward_info = self._compute_reward(
            previous_state_data=self.previous_state_data,
            current_state_data=current_state_data,
        )
        self._update_goal_progress(current_state_data)

        terminated = self._is_terminated(current_state_data)
        truncated = self._is_truncated()

        info = {"linear_velocity": current_state_data.linear_velocity()}
        info.update(reward_info)

        self.previous_state_data = current_state_data

        return current_state_data.state, reward, terminated, truncated, info

    def _set_velocity(self, lin_vel: float, steering_angle: float) -> None:
        angular = geometry_utils.ackermann_to_twist(
            steering_angle, lin_vel, self.wheelbase_m
        )
        velocity_msg = Twist()
        velocity_msg.angular.z = float(angular)
        velocity_msg.linear.x = float(lin_vel)
        self.cmd_vel_pub.publish(velocity_msg)

    def _set_simulation_paused(self, paused: bool):
        request = ControlWorld.Request()
        request.world_control.pause = paused
        future = self.world_control_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Apply an action and advance the simulation by one step.

        Args:
            action: [linear_velocity, steering_angle] array.
        Returns:
            observation: Next state after action.
            reward: Reward for the transition.
            terminated: True if episode ended by termination condition.
            truncated: True if episode ended by truncation (timeout/stall).
            info: Additional diagnostic info dict.
        """
        self.step_counter += 1

        if self.previous_state_data is None:
            raise RuntimeError(
                "Previous state data is not initialized - call reset first"
            )

        clipped_action = np.clip(
            np.asarray(action, dtype=np.float32), self.min_actions, self.max_actions
        )
        lin_vel, steering_angle = clipped_action
        self._set_simulation_paused(paused=False)

        self._set_velocity(lin_vel, steering_angle)

        self._sleep(self.step_sleep_time_ms)

        next_state, reward, terminated, truncated, info = self._transition()

        return next_state, reward, terminated, truncated, info

    def _set_model_pose(
        self,
        model_name: str,
        x: float,
        y: float,
        z: float,
        yaw: float = 0.0,
    ):
        request = msg_utils.build_set_model_pose_request(
            model_name=model_name,
            x=x,
            y=y,
            z=z,
            yaw=yaw,
        )
        future = self.set_pose_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def set_seed(self, seed: int) -> None:
        # just a place holder for external code that expects an environment to have a set_seed method
        pass
