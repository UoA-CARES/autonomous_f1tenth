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
        command_latency_ms: float = 0.0,
    ):
        """
        Initialize the F1Tenth RL environment node.

        Args:
            env_name: Name of the environment instance.
            goal_reach_radius_m: Radius (meters) for goal completion.
            max_steps: Maximum steps per episode.
            collision_range_m: Lidar collision threshold (meters).
            step_sleep_time_ms: Step duration in milliseconds.
            command_latency_ms: Delay before each new command reaches the simulated car.
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
        self.command_latency_ms = float(command_latency_ms)
        if not 0.0 <= self.command_latency_ms <= self.step_sleep_time_ms:
            raise ValueError(
                "command_latency_ms must be between 0 and step_sleep_time_ms"
            )
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
        self.is_eval = False
        self.spawn_index = 0
        self.stall_counter = 0
        self.total_linear_velocity = 0.0

        self.opponent_car_names = self._discover_opponent_car_names()
        self.latest_opponent_odometries: dict[str, Odometry | None] = {
            name: None for name in self.opponent_car_names
        }
        self.opponent_odom_subs = {
            name: self.create_subscription(
                Odometry,
                f"/{name}/odometry",
                self._make_opponent_odom_callback(name),
                qos,
            )
            for name in self.opponent_car_names
        }
        self.previous_race_positions: dict[str, float] = {}
        self.race_origin_spline_coord = 0.0
        self.race_origin_track_distance = 0.0
        self.overtake_rear_margin_m = 0.25
        self.overtake_front_margin_m = 0.25
        self.overtake_eligible_opponents: set[str] = set()
        self.overtaken_opponents: set[str] = set()
        self.overtakes_per_episode = 0
        self.pole_position_steps = 0

    def _discover_opponent_car_names(self) -> list[str]:
        """Find opponent cars from active ROS topic namespaces."""
        discovered_names: set[str] = set()
        name_pattern = re.compile(r"^opponent_(\d+)$")

        for topic_name, _ in self.get_topic_names_and_types():
            topic_root = topic_name.strip("/").split("/", 1)[0]
            if not topic_root:
                continue

            car_name = topic_root
            match = name_pattern.match(car_name)
            if match is None:
                continue

            car_index = int(match.group(1))
            if car_name == self.car_name:
                continue

            discovered_names.add(car_name)
            # print(f"Discovered opponent car: {car_name} from topic {topic_name}")

        def _car_sort_key(name: str) -> int:
            match = name_pattern.match(name)
            return int(match.group(1)) if match else 10_000

        return sorted(discovered_names, key=_car_sort_key)

    def _make_opponent_odom_callback(self, opponent_car_name: str):
        def _callback(odom: Odometry) -> None:
            self.latest_opponent_odometries[opponent_car_name] = odom

        return _callback

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

        opponent_index = (primary_spawn_index + 5 + (opponent_order * 2)) % len(
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

        self.latest_data = None
        self._next_build_clear_existing = False
        self._set_simulation_paused(paused=False)
        state_data = self._build_state_data()
        self.previous_state_data = state_data
        self.race_origin_track_distance = self._track_distance_from_position(
            state_data.position_xy()
        )
        self._set_simulation_paused(paused=True)

        return state_data.state

    def reset(self, training: bool = True) -> np.ndarray:
        self.step_counter = 0
        self.total_linear_velocity = 0.0
        self.overtakes_per_episode = 0
        self.overtake_eligible_opponents = set()
        self.overtaken_opponents = set()
        self.pole_position_steps = 0
        self.previous_race_positions = {}
        self.latest_opponent_odometries = {
            name: None for name in self.opponent_car_names
        }

        self.stall_counter = 0

        self.is_eval = not training

        self._set_velocity(0, 0)

        state = self._reset()
        if self.previous_state_data is not None:
            self.previous_race_positions = self._get_race_positions(
                self.previous_state_data
            )
            self._reset_overtake_tracking(self.previous_race_positions)
        return state

    def _message_filter_callback(self, odom: Odometry, lidar: LaserScan) -> None:
        self.latest_data = (odom, lidar)

    def _get_data(
        self, timeout: float = 5.0, clear_existing: bool = True
    ) -> tuple[Odometry, LaserScan]:
        # Drain anything stale
        if clear_existing:
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

    def _build_state_data(self, clear_existing: bool | None = None) -> StateData:
        if clear_existing is None:
            clear_existing = getattr(self, "_next_build_clear_existing", True)
            self._next_build_clear_existing = True
        odom_msg, lidar_msg = self._get_data(clear_existing=clear_existing)
        state_data = self.state_builder.build_state(odom_msg, lidar_msg)
        self.state_scan_pub.publish(state_data.lidar_state_scan)
        return state_data

    def _spline_coord_from_position(self, position_xy: tuple[float, float]) -> float:
        return self.current_track_model.world_coord_to_spline_coord(
            np.asarray(position_xy, dtype=np.float64)
        )

    def _track_distance_from_position(self, position_xy: tuple[float, float]) -> float:
        return self.current_track_model.track_distance_from_world_coord(
            np.asarray(position_xy, dtype=np.float64)
        )

    def _race_progress_from_position(self, position_xy: tuple[float, float]) -> float:
        track_distance = self._track_distance_from_position(position_xy)
        return self.current_track_model.forward_distance_between_track_distances(
            self.race_origin_track_distance,
            track_distance,
        )

    def _unwrap_race_positions(
        self,
        previous_positions: dict[str, float],
        current_wrapped_positions: dict[str, float],
    ) -> dict[str, float]:
        lap_length = self.current_track_model.waypoint_lap_length
        unwrapped_positions = {}
        for name, current_wrapped in current_wrapped_positions.items():
            previous_position = previous_positions.get(name)
            if previous_position is None:
                unwrapped_positions[name] = current_wrapped
                continue

            previous_wrapped = previous_position % lap_length
            delta = self.current_track_model.signed_delta_between_track_distances(
                previous_wrapped,
                current_wrapped,
            )
            unwrapped_positions[name] = previous_position + delta
        return unwrapped_positions

    def _get_agent_race_position(self, state_data: StateData) -> float:
        return self._race_progress_from_position(state_data.position_xy())

    def _get_opponent_race_position(self, opponent_car_name: str) -> float | None:
        odom = self.latest_opponent_odometries.get(opponent_car_name)
        if odom is None:
            return None

        return self._race_progress_from_position(
            (odom.pose.pose.position.x, odom.pose.pose.position.y)
        )

    def _get_race_positions(self, state_data: StateData) -> dict[str, float]:
        positions = {self.car_name: self._get_agent_race_position(state_data)}
        for opponent_car_name in self.opponent_car_names:
            opponent_position = self._get_opponent_race_position(opponent_car_name)
            if opponent_position is not None:
                positions[opponent_car_name] = opponent_position
        return positions

    def _reset_overtake_tracking(self, race_positions: dict[str, float]) -> None:
        self.overtaken_opponents = set()
        self.overtake_eligible_opponents = set(self.opponent_car_names)

        agent_position = race_positions.get(self.car_name)
        if agent_position is None:
            return

        for opponent_car_name in self.opponent_car_names:
            opponent_position = race_positions.get(opponent_car_name)
            if opponent_position is None:
                continue
            if agent_position > opponent_position - self.overtake_rear_margin_m:
                self.overtake_eligible_opponents.discard(opponent_car_name)

    def _count_new_agent_overtakes(
        self,
        previous_positions: dict[str, float],
        current_positions: dict[str, float],
    ) -> int:
        current_agent_position = current_positions.get(self.car_name)
        if current_agent_position is None:
            return 0

        new_overtakes = 0
        for opponent_car_name in self.opponent_car_names:
            current_opponent_position = current_positions.get(opponent_car_name)
            if current_opponent_position is None:
                continue
            if opponent_car_name in self.overtaken_opponents:
                continue

            current_gap = current_agent_position - current_opponent_position
            if current_gap <= -self.overtake_rear_margin_m:
                self.overtake_eligible_opponents.add(opponent_car_name)
                continue

            if (
                opponent_car_name in self.overtake_eligible_opponents
                and current_gap >= self.overtake_front_margin_m
            ):
                self.overtaken_opponents.add(opponent_car_name)
                self.overtake_eligible_opponents.discard(opponent_car_name)
                new_overtakes += 1

        return new_overtakes

    def _is_agent_in_pole_position(self, race_positions: dict[str, float]) -> bool:
        agent_position = race_positions.get(self.car_name)
        if agent_position is None:
            return False

        opponent_positions = [
            position
            for name, position in race_positions.items()
            if name != self.car_name
        ]
        return bool(opponent_positions) and all(
            agent_position > position for position in opponent_positions
        )

    def _get_opponent_distance_info(
        self, race_positions: dict[str, float]
    ) -> dict[str, float]:
        agent_position = race_positions.get(self.car_name)
        if agent_position is None:
            return {}

        return {
            opponent_car_name: race_positions[opponent_car_name] - agent_position
            for opponent_car_name in self.opponent_car_names
            if opponent_car_name in race_positions
        }

    def _get_world_positions(
        self, current_state_data: StateData
    ) -> dict[str, tuple[float, float]]:
        """World-frame XY of every car, from the odometry already subscribed to.

        Fed into `info` so a training/eval consumer can plot the cars' racing
        lines (and stitch a video from them) without re-deriving position from
        track progress, which is track-relative rather than world-frame.
        """
        positions = {
            self.car_name: tuple(
                float(value) for value in current_state_data.position_xy()
            )
        }
        for opponent_car_name in self.opponent_car_names:
            odom = self.latest_opponent_odometries.get(opponent_car_name)
            if odom is not None:
                positions[opponent_car_name] = (
                    float(odom.pose.pose.position.x),
                    float(odom.pose.pose.position.y),
                )
        return positions

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
        curr_x, curr_y = current_state_data.position_xy()

        step_progress = self.current_track_model.signed_delta_between_world_coords(
            np.asarray([prev_x, prev_y], dtype=np.float64),
            np.asarray([curr_x, curr_y], dtype=np.float64),
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
        wrapped_race_positions = self._get_race_positions(current_state_data)
        race_positions = self._unwrap_race_positions(
            self.previous_race_positions,
            wrapped_race_positions,
        )
        self.overtakes_per_episode += self._count_new_agent_overtakes(
            self.previous_race_positions,
            race_positions,
        )
        if self._is_agent_in_pole_position(race_positions):
            self.pole_position_steps += 1

        terminated = self._is_terminated(current_state_data)
        truncated = self._is_truncated()
        self.total_linear_velocity += current_state_data.linear_velocity()
        avg_linear_velocity = (
            self.total_linear_velocity / self.step_counter
            if self.step_counter > 0
            else 0.0
        )
        distance_to_opponents = self._get_opponent_distance_info(race_positions)
        episode_done = terminated or truncated
        episode_overtakes = self.overtakes_per_episode
        agent_track_position = race_positions.get(self.car_name, 0.0)
        world_positions = self._get_world_positions(current_state_data)

        info = {
            "linear_velocity": current_state_data.linear_velocity(),
            "avg_linear_velocity": avg_linear_velocity,
            "average_linear_velocity_per_episode": avg_linear_velocity,
            "distance_to_opponents": distance_to_opponents,
            "overtakes_per_episode": (
                episode_overtakes if episode_done else 0
            ),
            "time_in_pole_position": self.pole_position_steps,
            "agent_track_position": agent_track_position,
            "agent_track_position_m": agent_track_position,
            "position_xy": world_positions[self.car_name],
            "positions_xy": world_positions,
        }
        for opponent_car_name, opponent_distance in distance_to_opponents.items():
            info[f"distance_to_{opponent_car_name}"] = opponent_distance
            opponent_track_position = race_positions[opponent_car_name]
            info[f"{opponent_car_name}_track_position"] = opponent_track_position
            info[f"{opponent_car_name}_track_position_m"] = opponent_track_position
        for opponent_car_name, opponent_xy in world_positions.items():
            if opponent_car_name == self.car_name:
                continue
            info[f"{opponent_car_name}_position_xy"] = opponent_xy
        info.update(reward_info)

        self.previous_state_data = current_state_data
        self.previous_race_positions = race_positions

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
        self._wait_for_evaluation_service(
            future,
            operation="world control",
        )
        return future.result()

    def _wait_for_evaluation_service(self, future, *, operation: str) -> None:
        timeout_s = getattr(
            self,
            "evaluation_service_timeout_s",
            None,
        )
        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=timeout_s,
        )
        if timeout_s is not None and not future.done():
            future.cancel()
            raise TimeoutError(
                f"Gazebo {operation} service did not respond within "
                f"{timeout_s:.1f} wall seconds"
            )

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
        self.latest_data = None
        self._next_build_clear_existing = False
        self._set_simulation_paused(paused=False)
        if self.command_latency_ms > 0.0:
            self._sleep(self.command_latency_ms)
        self._set_velocity(lin_vel, steering_angle)

        remaining_step_ms = self.step_sleep_time_ms - self.command_latency_ms
        if remaining_step_ms > 0.0:
            self._sleep(remaining_step_ms)

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
        self._wait_for_evaluation_service(
            future,
            operation=f"set pose for {model_name}",
        )
        return future.result()

    def set_seed(self, seed: int) -> None:
        # just a place holder for external code that expects an environment to have a set_seed method
        pass
