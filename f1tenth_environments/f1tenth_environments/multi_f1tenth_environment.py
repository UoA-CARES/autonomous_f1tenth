import math
import os
import random
import re
import time
from .f1tenth_environment import F1tenthEnvironment
import rclpy

from pettingzoo import ParallelEnv
from geometry_msgs.msg import Twist
from gymnasium import spaces
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from ros_gz_interfaces.srv import ControlWorld, SetEntityPose
from sensor_msgs.msg import LaserScan

import numpy as np

from . import geometry_utils, lidar_processor, msg_utils, track_utils, waypoints
from .state_builder import LidarMode, OdomMode, StateBuilder, StateData

class MultiF1TenthEnvironment(F1tenthEnvironment, ParallelEnv, Node):
    """Multi-agent F1Tenth environment using PettingZoo API."""

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
        position_speed_multiplier: float,
        command_latency_ms: float = 0.0,
    ):
        # Intialize ROS2 node
        Node.__init__(self, f"{env_name}_multi_environment")

        # Initialize initial parameters
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
        self.position_speed_multiplier = position_speed_multiplier

        self.wall_proximity_reward_weight = wall_proximity_reward_weight
        self.turn_reward_weight = turn_reward_weight
        self.stall_progress_threshold_m = stall_progress_threshold_m
        self.stall_limit_steps = stall_limit_steps
        self.collision_penalty = collision_penalty

        configured_num_opponents = self._get_configured_num_opponents()
        self.opponent_car_names = self._discover_opponent_car_names(
            configured_num_opponents
        )
        self.wheelbase_m = 0.325
        
        # Setup Tracks and Waypoints for tracking car progress
        self.tracks = self._load_tracks(track)
        self.track_progress_models = track_utils.get_track_progress_models(self.tracks)
        self.track_names = list(self.tracks.keys())
        self.fixed_track_name = self._get_fixed_track_name(track)
        self.eval_tracks_start_idx: int = int(
            len(self.track_names) * self.train_eval_split
        )
        self.eval_track_idx = 0

        self.current_track = self.fixed_track_name or self.track_names[0]
        self.current_waypoints = self.tracks[self.current_track]
        self.current_track_model = self.track_progress_models[self.current_track]

        qos = QoSProfile(depth=3)

        self.state_scan_pub = self.create_publisher(
            LaserScan, f"/{self.car_name}/state_scan", 1
        )

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

        # Reward Weights as precalculated factors to modify the base reward based on progress,
        # to encourage desirable behaviour
        self.wall_proximity_reward_weight = 0.7
        self.turn_reward_weight = 0.3

        # Loop Parameters
        self.step_counter = 0

        self.is_eval = True
        self.spawn_index = 0

        # Multi agents specific stuff
        self.agents = [self.car_name] + self.opponent_car_names

        self.spawn_indices = {agent: 0 for agent in self.agents}
        self.action_num = {agent: 2 for agent in self.agents}
        
        self.stall_counters = {agent: 0 for agent in self.agents}
        self.latest_data = {agent: None for agent in self.agents}
        self.agent_goals = {agent: (0.0, 0.0) for agent in self.agents}
        self.previous_state_data = {agent: None for agent in self.agents}
        self.goals_reached = {agent: 0 for agent in self.agents}
        self.overtake_counts = {agent: 0 for agent in self.agents}
        self.total_linear_velocity = {agent: 0.0 for agent in self.agents}
        self.pole_position_steps = {agent: 0 for agent in self.agents}
        self.previous_race_positions: dict[str, float] = {}
        self.race_origin_track_distance = 0.0
        self.cmd_vel_pubs = {}
        self.last_reset_seed: int | None = None
        self.last_command_sim_time_s: float | None = None
        self.last_observation_sim_time_s: float | None = None

        self.odom_subs = {}
        self.lidar_subs = {}
        self.latest_odoms = {agent: None for agent in self.agents}
        self.latest_lidars = {agent: None for agent in self.agents}

        for agent in self.agents:
            self.odom_subs[agent] = self.create_subscription(
                Odometry,
                f"/{agent}/odometry",
                self._make_odom_callback(agent),
                qos,
            )
            self.lidar_subs[agent] = self.create_subscription(
                LaserScan,
                f"/{agent}/scan",
                self._make_lidar_callback(agent),
                qos_profile_sensor_data,
            )
            self.cmd_vel_pubs[agent] = self.create_publisher(Twist, f"/{agent}/cmd_vel", 1)

        if lidar_mode == "raw":
            _, lidar_data = self._get_data(self.car_name)
            lidar_state_size = len(lidar_data.ranges)
    
    
    def _get_configured_num_opponents(self) -> int | None:
        env_value = os.environ.get("F1TENTH_NUM_OPPONENTS")
        default_value = -1
        if env_value is not None:
            try:
                default_value = int(env_value)
            except ValueError:
                self.get_logger().warning(
                    f"Ignoring invalid F1TENTH_NUM_OPPONENTS={env_value!r}"
                )

        self.declare_parameter("num_opponents", default_value)
        param_value = int(self.get_parameter("num_opponents").value)
        return param_value if param_value >= 0 else None

    def _get_fixed_track_name(self, configured_track: str) -> str | None:
        requested_track = os.environ.get("F1TENTH_ACTIVE_TRACK")
        if not requested_track:
            return None

        if requested_track in self.tracks:
            self.get_logger().info(
                f"MARL reset fixed to track {requested_track!r} from {configured_track!r}."
            )
            return requested_track

        self.get_logger().warning(
            f"Ignoring unknown F1TENTH_ACTIVE_TRACK={requested_track!r}; "
            f"available tracks: {list(self.tracks.keys())}"
        )
        return None

    def _select_track_name(self) -> str:
        if self.fixed_track_name is not None:
            return self.fixed_track_name

        if self.is_eval:
            selected = self.track_names[self.eval_track_idx % len(self.track_names)]
            self.eval_track_idx = (self.eval_track_idx + 1) % len(self.track_names)
            return selected

        return random.choice(self.track_names)

    def _discover_opponent_car_names(
        self,
        expected_count: int | None = None,
        timeout_sec: float = 5.0,
        stable_sec: float = 2.0,
    ) -> list[str]:
        if expected_count is not None:
            return [f"opponent_{index}" for index in range(1, expected_count + 1)]

        discovered_names: set[str] = set()
        stable_since = time.monotonic()
        deadline = time.monotonic() + timeout_sec

        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            current_names = self._discover_opponent_car_names_once()
            if current_names != discovered_names:
                discovered_names = current_names
                stable_since = time.monotonic()
            if discovered_names and time.monotonic() - stable_since >= stable_sec:
                break

        def _car_sort_key(name: str) -> int:
            match = re.match(r"^opponent_(\d+)$", name)
            return int(match.group(1)) if match else 10_000

        return sorted(discovered_names, key=_car_sort_key)

    def _discover_opponent_car_names_once(self) -> set[str]:
        discovered_names: set[str] = set()
        name_pattern = re.compile(r"^opponent_(\d+)$")

        for topic_name, _ in self.get_topic_names_and_types():
            topic_root = topic_name.strip("/").split("/", 1)[0]
            if name_pattern.match(topic_root):
                discovered_names.add(topic_root)

        return discovered_names

    def _make_odom_callback(self, agent: str):
        def _callback(odom: Odometry) -> None:
            self.latest_odoms[agent] = odom
            self._update_latest_data(agent)
        return _callback

    def _make_lidar_callback(self, agent: str):
        def _callback(lidar: LaserScan) -> None:
            self.latest_lidars[agent] = lidar
            self._update_latest_data(agent)
        return _callback

    def _update_latest_data(self, agent: str) -> None:
        odom = self.latest_odoms[agent]
        lidar = self.latest_lidars[agent]
        if odom is not None and lidar is not None:
            self.latest_data[agent] = (odom, lidar)

    def _update_goal_progress(self, agent: str, state_data: StateData) -> None:
        next_x, next_y = state_data.position_xy()
        if (
            math.dist(self.agent_goals[agent], [next_x, next_y])
            < self.goal_reach_radius_m
        ):
            self.goals_reached[agent] += 1
            new_x, new_y, _, _ = self.current_waypoints[
                (self.spawn_indices[agent] + self.goals_reached[agent] + 1)
                % len(self.current_waypoints)
            ]
            self.agent_goals[agent] = (new_x, new_y)

    def _get_opponent_spawn_index(
        self, primary_spawn_index: int, opponent_order: int
    ) -> int:
        waypoint_count = len(self.current_waypoints)
        if waypoint_count == 0:
            return 0

        start_gap = max(1, int(os.environ.get("F1TENTH_OPPONENT_START_GAP", "8")))
        gap = max(1, int(os.environ.get("F1TENTH_OPPONENT_GAP", "4")))
        return (primary_spawn_index + start_gap + opponent_order * gap) % waypoint_count

    def _get_opponent_spawn_pose(
        self, primary_spawn_index: int, opponent_order: int
    ) -> tuple[float, float, float]:
        """Return a MARL opponent spawn pose just ahead of the agent car."""
        if not self.current_waypoints:
            return 0.0, 0.0, 0.0

        opponent_index = self._get_opponent_spawn_index(
            primary_spawn_index, opponent_order
        )
        opponent_x, opponent_y, opponent_yaw, _ = self.current_waypoints[opponent_index]
        return opponent_x, opponent_y, opponent_yaw

    def _get_agent_race_position(self, agent: str, state_data: StateData) -> float:
        track_distance = self.current_track_model.track_distance_from_world_coord(
            np.asarray(state_data.position_xy(), dtype=np.float64)
        )
        return self.current_track_model.forward_distance_between_track_distances(
            self.race_origin_track_distance,
            track_distance,
        )

    def _get_race_positions(
        self,
        all_state_data: dict[str, StateData],
    ) -> dict[str, float]:
        return {
            agent: self._get_agent_race_position(agent, state_data)
            for agent, state_data in all_state_data.items()
            if agent in self.agents
        }

    def _count_new_agent_overtakes(
        self,
        agent: str,
        previous_positions: dict[str, float],
        current_positions: dict[str, float],
    ) -> int:
        previous_agent_position = previous_positions.get(agent)
        current_agent_position = current_positions.get(agent)
        if previous_agent_position is None or current_agent_position is None:
            return 0

        new_overtakes = 0
        for other_agent in self.agents:
            if other_agent == agent:
                continue

            previous_opponent_position = previous_positions.get(other_agent)
            current_opponent_position = current_positions.get(other_agent)
            if (
                previous_opponent_position is None
                or current_opponent_position is None
            ):
                continue

            was_not_ahead = previous_agent_position <= previous_opponent_position
            is_ahead = current_agent_position > current_opponent_position
            if was_not_ahead and is_ahead:
                new_overtakes += 1

        return new_overtakes

    def _reset_positions(self, options: dict | None = None) -> None:
        options = options or {}
        requested_track = options.get("track_name")
        if requested_track is not None and requested_track not in self.tracks:
            raise ValueError(
                f"Unknown evaluation track {requested_track!r}; "
                f"available tracks: {list(self.tracks)}"
            )

        self.current_track = requested_track or self._select_track_name()
        self.current_waypoints = self.tracks[self.current_track]
        self.current_track_model = self.track_progress_models[self.current_track]

        spawn_poses = options.get("spawn_poses")
        if spawn_poses is not None:
            expected_agents = set(self.agents)
            supplied_agents = set(spawn_poses)
            if supplied_agents != expected_agents:
                raise ValueError(
                    "Evaluation spawn_poses must match environment agents; "
                    f"expected={sorted(expected_agents)}, "
                    f"supplied={sorted(supplied_agents)}"
                )

            self.spawn_indices = {}
            for agent in self.agents:
                pose = spawn_poses[agent]
                missing_fields = {
                    field for field in ("x", "y", "yaw") if field not in pose
                }
                if missing_fields:
                    raise ValueError(
                        f"Evaluation pose for {agent!r} is missing "
                        f"{sorted(missing_fields)}"
                    )
                self.spawn_indices[agent] = int(
                    pose.get("waypoint_index", 0)
                )
                self._set_model_pose(
                    model_name=agent,
                    x=float(pose["x"]),
                    y=float(pose["y"]),
                    z=float(pose.get("z", 0.0)),
                    yaw=float(pose["yaw"]),
                )
            self.spawn_index = self.spawn_indices[self.car_name]
            return

        # Spawn main car
        if self.is_eval:
            index = min(10, len(self.current_waypoints) - 1)
        else:
            index = random.randrange(len(self.current_waypoints))
        car_x, car_y, car_yaw, _ = self.current_waypoints[index]

        self.spawn_index = index  # keep for base class compatibility
        self.spawn_indices = {self.car_name: index}

        self._set_model_pose(
            model_name=self.car_name,
            x=float(car_x),
            y=float(car_y),
            z=0.0,
            yaw=float(car_yaw),
        )

        # Spawn opponents
        for opponent_order, opponent_car_name in enumerate(self.opponent_car_names):
            opponent_index = self._get_opponent_spawn_index(
                self.spawn_index, opponent_order
            )
            opponent_x, opponent_y, opponent_yaw = self._get_opponent_spawn_pose(
                self.spawn_index, opponent_order
            )
            self.spawn_indices[opponent_car_name] = opponent_index

            self._set_model_pose(
                model_name=opponent_car_name,
                x=float(opponent_x),
                y=float(opponent_y),
                z=0.0,
                yaw=float(opponent_yaw),
            )

    def reset(self, seed=None, options=None) -> dict:
        reset_options = dict(options or {})
        self.step_counter = 0
        self.is_eval = bool(reset_options.get("evaluation", False))
        self.last_reset_seed = seed
        self.last_command_sim_time_s = None
        self.last_observation_sim_time_s = None
        self._last_track_distances = {agent: 0.0 for agent in self.agents}

        self._set_simulation_paused(paused=True)

        for agent in self.agents:
            self.stall_counters[agent] = 0
            self.goals_reached[agent] = 0
            self.total_linear_velocity[agent] = 0.0
            self.pole_position_steps[agent] = 0
        self._stop_all_agents()

        self._reset_positions(options=reset_options)
        self._stop_all_agents()
        self._clear_all_data()
        self._set_simulation_paused(paused=False)

        # Collect data for ALL agents in one go while sim is running
        all_state_data = self._build_all_state_data(clear_existing=False)

        self._set_simulation_paused(paused=True)
        self.last_observation_sim_time_s = self._sim_time_seconds()

        # Initialise each agent's goal to the next waypoint from their spawn position
        for agent in self.agents:
            x, y = all_state_data[agent].position_xy()
            nearest_idx = min(
                range(len(self.current_waypoints)),
                key=lambda i: math.dist((x, y), (self.current_waypoints[i][0], self.current_waypoints[i][1]))
            )
            self.spawn_indices[agent] = nearest_idx
            wx, wy, _, _ = self.current_waypoints[
                (nearest_idx + 1) % len(self.current_waypoints)
            ]
            self.agent_goals[agent] = (wx, wy)

        obs = {}
        infos = {}
        for agent in self.agents:
            self.overtake_counts[agent] = 0
            self.previous_state_data[agent] = all_state_data[agent]
            obs[agent] = all_state_data[agent].state
            infos[agent] = {}

        self.race_origin_track_distance = (
            self.current_track_model.track_distance_from_world_coord(
                np.asarray(
                    all_state_data[self.car_name].position_xy(),
                    dtype=np.float64,
                )
            )
        )
        self.previous_race_positions = self._get_race_positions(all_state_data)
        self._last_track_distances = dict(self.previous_race_positions)

        return obs, infos

    def _get_track_distance(self, agent: str) -> float:
        if self.latest_data[agent] is None:
            return float(self.goals_reached[agent])
        
        odom, _ = self.latest_data[agent]
        x = odom.pose.pose.position.x
        y = odom.pose.pose.position.y
        goal_x, goal_y = self.agent_goals[agent]

        total_waypoints = len(self.current_waypoints)
        laps_completed = self.goals_reached[agent] // total_waypoints
        waypoints_this_lap = self.goals_reached[agent] % total_waypoints

        dist_to_next_goal = math.dist((x, y), (goal_x, goal_y))
        max_dist = self.goal_reach_radius_m * 2.0
        fraction = float(np.clip(1.0 - (dist_to_next_goal / max_dist), 0.0, 1.0))

        return float(laps_completed * total_waypoints + waypoints_this_lap + fraction)
    
    def _get_track_distance_from_state(self, agent: str, state_data: StateData) -> float:
        x, y = state_data.position_xy()
        goal_x, goal_y = self.agent_goals[agent]

        total_waypoints = len(self.current_waypoints)
        laps_completed = self.goals_reached[agent] // total_waypoints
        waypoints_this_lap = self.goals_reached[agent] % total_waypoints

        dist_to_next_goal = math.dist((x, y), (goal_x, goal_y))
        
        current_wp_idx = (self.spawn_indices[agent] + self.goals_reached[agent]) % total_waypoints
        next_wp_idx = (current_wp_idx + 1) % total_waypoints
        prev_x, prev_y, _, _ = self.current_waypoints[current_wp_idx]
        next_x, next_y, _, _ = self.current_waypoints[next_wp_idx]
        waypoint_spacing = math.dist((prev_x, prev_y), (next_x, next_y))
        max_dist = max(waypoint_spacing, 0.1)

        fraction = float(np.clip(1.0 - (dist_to_next_goal / max_dist), 0.0, 1.0))

        return float(laps_completed * total_waypoints + waypoints_this_lap + fraction)

    def _get_opponent_distance_info(
        self,
        agent: str,
        race_positions: dict[str, float],
    ) -> dict[str, float]:
        agent_position = race_positions.get(agent)
        if agent_position is None:
            return {}

        return {
            other_agent: race_positions[other_agent] - agent_position
            for other_agent in self.agents
            if other_agent != agent and other_agent in race_positions
        }

    def _is_agent_in_pole_position(
        self,
        agent: str,
        race_positions: dict[str, float],
    ) -> bool:
        agent_position = race_positions.get(agent)
        if agent_position is None:
            return False

        other_positions = [
            position
            for other_agent, position in race_positions.items()
            if other_agent != agent
        ]
        return bool(other_positions) and all(
            agent_position > position for position in other_positions
        )

    def _build_agent_metric_info(
        self,
        agent: str,
        current_state: StateData,
        race_positions: dict[str, float],
        terminated: bool,
        truncated: bool,
    ) -> dict:
        self.total_linear_velocity[agent] += current_state.linear_velocity()
        avg_linear_velocity = (
            self.total_linear_velocity[agent] / self.step_counter
            if self.step_counter > 0
            else 0.0
        )
        distance_to_opponents = self._get_opponent_distance_info(
            agent, race_positions
        )
        if self._is_agent_in_pole_position(agent, race_positions):
            self.pole_position_steps[agent] += 1
        episode_done = terminated or truncated
        live_overtakes = self.overtake_counts[agent]

        info = {
            "linear_velocity": current_state.linear_velocity(),
            "avg_linear_velocity": avg_linear_velocity,
            "average_linear_velocity_per_episode": avg_linear_velocity,
            "distance_to_opponents": distance_to_opponents,
            "overtakes_per_episode": (
                live_overtakes if episode_done else 0
            ),
            "number_of_overtakes_per_episode": (
                live_overtakes if episode_done else 0
            ),
            "time_in_pole_position": self.pole_position_steps[agent],
            "agent_track_position": race_positions.get(agent, 0.0),
            "overtakes": live_overtakes,
            "agent_overtakes": live_overtakes,
            "overtakes_live": live_overtakes,
            "goals_reached": self.goals_reached[agent],
        }
        for other_agent, opponent_distance in distance_to_opponents.items():
            info[f"distance_to_{other_agent}"] = opponent_distance
            info[f"{other_agent}_track_position"] = race_positions[other_agent]

        return info

    def step(self, actions: dict) -> tuple[dict, dict, dict, dict, dict]:
        self.step_counter += 1
        self._clear_all_data()
        self._set_simulation_paused(paused=False)
        if self.command_latency_ms > 0.0:
            self._sleep(self.command_latency_ms)

        self.last_command_sim_time_s = self._sim_time_seconds()
        for agent, action in actions.items():
            agent_max_speed = (
                self.max_actions[0]
                if agent == self.car_name
                else self.max_actions[0] * self.position_speed_multiplier
            )

            agent_max_actions = np.array(
                [agent_max_speed, self.max_actions[1]],
                dtype=np.float32,
            )
            clipped = np.clip(
                np.asarray(action, dtype=np.float32),
                self.min_actions,
                agent_max_actions,
            )
            lin_vel, steering = clipped
            angular = geometry_utils.ackermann_to_twist(steering, lin_vel, self.wheelbase_m)
            msg = Twist()
            msg.linear.x = float(lin_vel)
            msg.angular.z = float(angular)
            self.cmd_vel_pubs[agent].publish(msg)

        remaining_step_ms = self.step_sleep_time_ms - self.command_latency_ms
        if remaining_step_ms > 0.0:
            self._sleep(remaining_step_ms)
        all_state_data = self._build_all_state_data(clear_existing=False)
        self._set_simulation_paused(paused=True)
        self.last_observation_sim_time_s = self._sim_time_seconds()

        obs, rewards, terminateds, truncateds, infos = {}, {}, {}, {}, {}

        for agent in self.agents:
            current_state = all_state_data[agent]

            reward, info = self._compute_reward(
                self.previous_state_data[agent],
                current_state,
                agent,
            )

            self._update_goal_progress(agent, current_state)
            terminateds[agent] = self._is_terminated(current_state)
            truncateds[agent] = self._is_truncated(agent)
            obs[agent] = current_state.state
            rewards[agent] = reward
            infos[agent] = info
            self.previous_state_data[agent] = current_state

        wrapped_race_positions = self._get_race_positions(all_state_data)
        race_positions = self._unwrap_race_positions(
            self.previous_race_positions,
            wrapped_race_positions,
        )
        for agent in self.agents:
            self.overtake_counts[agent] += self._count_new_agent_overtakes(
                agent,
                self.previous_race_positions,
                race_positions,
            )

        if any(terminateds.values()):
            terminateds = {agent: True for agent in self.agents}

        for agent in self.agents:
            infos[agent].update(
                self._build_agent_metric_info(
                    agent,
                    all_state_data[agent],
                    race_positions,
                    terminateds[agent],
                    truncateds[agent],
                )
            )

        self.previous_race_positions = race_positions
        self._last_track_distances = dict(race_positions)

        return obs, rewards, terminateds, truncateds, infos

    def _sim_time_seconds(self) -> float:
        """Return the active ROS/Gazebo clock without changing simulation state."""
        return self.get_clock().now().nanoseconds / 1e9
    
    # override for multi agents
    def _get_required_data_topic_names(self) -> list[str]:
        topic_names = []
        for agent in self.agents:
            topic_names.extend((f"/{agent}/odometry", f"/{agent}/scan"))
        return topic_names

    def _get_visible_data_topic_names(self) -> set[str]:
        return {
            topic_name
            for topic_name, _ in self.get_topic_names_and_types()
            if topic_name in self._get_required_data_topic_names()
        }

    def _format_data_timeout_message(self, missing: list[str]) -> str:
        required_topics = self._get_required_data_topic_names()
        visible_topics = sorted(self._get_visible_data_topic_names())
        absent_topics = sorted(set(required_topics) - set(visible_topics))

        message_parts = [
            f"No fresh odometry/lidar received for agents: {missing}.",
            f"Required topics: {required_topics}.",
            f"Visible required topics: {visible_topics}.",
        ]
        if absent_topics:
            message_parts.append(f"Absent required topics: {absent_topics}.")
        message_parts.append(
            "Check that the Gazebo launch num_opponents matches "
            "F1TENTH_NUM_OPPONENTS and that only one world named 'empty' is running."
        )
        return " ".join(message_parts)

    def _stop_all_agents(self) -> None:
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        for agent in self.agents:
            self.cmd_vel_pubs[agent].publish(msg)

    def stop_agent(self, agent: str) -> None:
        """Publish one zero command for an evaluation car that has DNFed."""
        if agent not in self.cmd_vel_pubs:
            raise ValueError(f"Unknown agent {agent!r}")
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_vel_pubs[agent].publish(msg)

    def _clear_all_data(self) -> None:
        for agent in self.agents:
            self.latest_data[agent] = None
            self.latest_odoms[agent] = None
            self.latest_lidars[agent] = None

    def _get_all_data(self, timeout: float = 5.0, clear_existing: bool = True) -> dict:
        """Spin until fresh odometry and lidar have arrived for every agent."""
        if clear_existing:
            self._clear_all_data()

        end_time = time.monotonic() + timeout
        while time.monotonic() < end_time:
            rclpy.spin_once(self, timeout_sec=0.01)
            if all(self.latest_data[agent] is not None for agent in self.agents):
                return dict(self.latest_data)

        missing = [a for a in self.agents if self.latest_data[a] is None]
        raise TimeoutError(self._format_data_timeout_message(missing))

    def _build_all_state_data(self, clear_existing: bool = True) -> dict[str, StateData]:
        """Build state for all agents from a single shared spin."""
        data = self._get_all_data(clear_existing=clear_existing)
        return {
            agent: self.state_builder.build_state(odom, lidar)
            for agent, (odom, lidar) in data.items()
        }
    
    def _is_truncated(self, agent: str) -> bool:
        return (
            self.stall_counters[agent] >= self.stall_limit_steps
            or self.step_counter >= self.max_steps
        )

    def _calculate_progress_reward(self, step_progress: float, agent: str, effective_max_speed: float) -> float:
        if step_progress < self.stall_progress_threshold_m:
            self.stall_counters[agent] += 1
        else:
            self.stall_counters[agent] = 0

        max_progress_per_step_m = float(effective_max_speed) * (
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
        agent: str,
    ) -> tuple[float, dict]:
        track_progress = self._compute_step_progress(previous_state_data, current_state_data)
        effective_max_speed = (
            self.max_actions[0]
            if agent == self.car_name
            else self.max_actions[0] * self.position_speed_multiplier
        )
        progress_reward = self._calculate_progress_reward(track_progress, agent, effective_max_speed)

        prev_angular_velocity = previous_state_data.angular_velocity()
        curr_angular_velocity = current_state_data.angular_velocity()
        dist_to_wall = float(np.min(current_state_data.lidar_sanitised_data))
        angular_velocity_change = abs(prev_angular_velocity - curr_angular_velocity)

        wall_threshold_m = 0.3
        wall_k = 50.0
        wall_factor = 1.0 / (1.0 + np.exp(wall_k * (dist_to_wall - wall_threshold_m)))

        turn_threshold_rads = 0.5
        turn_k = 15.0
        turn_factor = 1.0 / (1.0 + np.exp(-turn_k * (angular_velocity_change - turn_threshold_rads)))

        wall_discount = 1.0 - (wall_factor * self.wall_proximity_reward_weight)
        turn_discount = 1.0 - (turn_factor * self.turn_reward_weight)

        reward = progress_reward * wall_discount * turn_discount

        collision = self._has_collision_or_flip(current_state_data)
        if collision:
            reward -= self.collision_penalty

        return reward, {
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
    
    @property
    def observation_size(self):
        per_agent_obs_size = int(self.state_builder.policy_state_size)
        num_agents = len(self.agents)
        teams: dict[str, list[str]] = {}
        for agent in self.agents:
            team = agent.rsplit("_", 1)[0]
            teams.setdefault(team, []).append(agent)
        
        return {
            "obs": {agent: per_agent_obs_size for agent in self.agents},
            "state": per_agent_obs_size * num_agents,
            "num_agents": num_agents,
            "teams": teams,
        }
    
    @property
    def observation_spaces(self) -> dict:
        return {
            agent: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.observation_size["obs"][agent],),
                dtype=np.float32
            ) for agent in self.agents
        }

    @property
    def action_spaces(self) -> dict:
        return {
            agent: spaces.Box(
                low=self.min_actions,
                high=self.max_actions,
                dtype=np.float32
            ) for agent in self.agents
        }
