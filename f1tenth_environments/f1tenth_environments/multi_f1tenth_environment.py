import re
import math
from .f1tenth_environment import F1tenthEnvironment
import rclpy

from pettingzoo import ParallelEnv
from geometry_msgs.msg import Twist
from gymnasium import spaces
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile
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
        collision_penalty: float
    ):
        # Intialize ROS2 node
        Node.__init__(self, f"{env_name}_multi_environment")

        # Initialize initial parameters
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

        self.opponent_car_names = self._discover_opponent_car_names()
        self.wheelbase_m = 0.325
        
        # Setup Tracks and Waypoints for tracking car progress
        self.tracks = self._load_tracks(track)
        self.track_progress_models = track_utils.get_track_progress_models(self.tracks)
        self.track_names = list(self.tracks.keys())
        self.eval_tracks_start_idx: int = int(
            len(self.track_names) * self.train_eval_split
        )
        self.eval_track_idx = 0

        self.current_track = self.track_names[0]
        self.current_waypoints = self.tracks[self.current_track]
        self.current_track_model = self.track_progress_models[self.current_track]

        sub_depth = 3
        qos = QoSProfile(depth=sub_depth)

        self.state_scan_pub = self.create_publisher(
            LaserScan, f"/{self.car_name}/state_scan", 1
        )

        sync_slop_sec = 0.1

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
        self.agents = [self.car_name] +  self.opponent_car_names
        self.action_num = {agent: 2 for agent in self.agents}
        
        self.stall_counters = {agent: 0 for agent in self.agents}
        self.latest_data = {agent: None for agent in self.agents}
        self.agent_goals = {agent: (0.0, 0.0) for agent in self.agents}
        self.previous_state_data = {agent: None for agent in self.agents}
        self.goals_reached = {agent: 0 for agent in self.agents}
        self.overtake_counts = {agent: 0 for agent in self.agents}
        self.previous_track_positions = {agent: 0 for agent in self.agents}
        self.message_filters = {}
        self.cmd_vel_pubs = {}

        self.odom_subs = {}
        self.lidar_subs = {}

        for agent in self.agents:
            odom_sub = Subscriber(self, Odometry, f"/{agent}/odometry", qos_profile=qos)
            lidar_sub = Subscriber(self, LaserScan, f"/{agent}/scan", qos_profile=qos)
            self.odom_subs[agent] = odom_sub
            self.lidar_subs[agent] = lidar_sub
            self.cmd_vel_pubs[agent] = self.create_publisher(Twist, f"/{agent}/cmd_vel", 1)

            sync = ApproximateTimeSynchronizer([odom_sub, lidar_sub], sub_depth, sync_slop_sec)
            sync.registerCallback(self._make_agent_callback(agent))
            self.message_filters[agent] = sync

        if lidar_mode == "raw":
            _, lidar_data = self._get_data(self.car_name)
            lidar_state_size = len(lidar_data.ranges)
    
    
    def _make_agent_callback(self, agent: str):
        def _callback(odom: Odometry, lidar: LaserScan) -> None:
            self.latest_data[agent] = (odom, lidar)
        return _callback

    def _update_goal_progress(self, agent: str, state_data: StateData) -> None:
        next_x, next_y = state_data.position_xy()
        if math.dist(self.agent_goals[agent], [next_x, next_y]) < self.goal_reach_radius_m:
            self.goals_reached[agent] += 1
            new_x, new_y, _, _ = self.current_waypoints[
                (self.spawn_index + self.goals_reached[agent]) % len(self.current_waypoints)
            ]
            self.agent_goals[agent] = (new_x, new_y)

    def _check_overtakes(self) -> None:
        for agent in self.agents:
            for other in self.agents:
                if agent == other:
                    continue
                # Agent has overtaken other if it was behind before and is ahead now
                was_behind = self.previous_track_positions[agent] <= self.previous_track_positions[other]
                is_ahead = self.goals_reached[agent] > self.goals_reached[other]
                if was_behind and is_ahead:
                    self.overtake_counts[agent] += 1

    def _snapshot_track_positions(self) -> None:
        for agent in self.agents:
            self.previous_track_positions[agent] = self.goals_reached[agent]
    
    def _message_filter_callback(self, *msgs) -> None:
        for i, agent in enumerate(self.agents):
            odom = msgs[i * 2]
            lidar = msgs[i * 2 + 1]
            self.latest_data[agent] = (odom, lidar)

    def reset(self, seed=None, options=None) -> dict:
        self.step_counter = 0
        self.is_eval = False

        for agent in self.agents:
            self.stall_counters[agent] = 0
            msg = Twist()
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            self.cmd_vel_pubs[agent].publish(msg)

        self._reset_positions()
        self._set_simulation_paused(paused=False)

        # Collect data for ALL agents in one go while sim is running
        all_state_data = self._build_all_state_data()

        self._set_simulation_paused(paused=True)

        obs = {}
        infos = {}
        for agent in self.agents:
            self.overtake_counts[agent] = 0
            self.previous_track_positions[agent] = 0
            self.previous_state_data[agent] = all_state_data[agent]
            obs[agent] = all_state_data[agent].state
            infos[agent] = {}

        return obs, infos

    def step(self, actions: dict) -> tuple[dict, dict, dict, dict, dict]:
        self.step_counter += 1
        self._set_simulation_paused(paused=False)

        for agent, action in actions.items():
            clipped = np.clip(np.asarray(action, dtype=np.float32), self.min_actions, self.max_actions)
            lin_vel, steering = clipped
            angular = geometry_utils.ackermann_to_twist(steering, lin_vel, self.wheelbase_m)
            msg = Twist()
            msg.linear.x = float(lin_vel)
            msg.angular.z = float(angular)
            self.cmd_vel_pubs[agent].publish(msg)

        self._sleep(self.step_sleep_time_ms)

        # Collect data for ALL agents before pausing — sim must be running for messages to arrive
        all_state_data = self._build_all_state_data()

        self._set_simulation_paused(paused=True)

        self._snapshot_track_positions()

        obs, rewards, terminateds, truncateds, infos = {}, {}, {}, {}, {}

        for agent in self.agents:
            current_state = all_state_data[agent]
            reward, info = self._compute_reward(self.previous_state_data[agent], current_state, agent)
            self._update_goal_progress(agent, current_state)

            terminateds[agent] = self._is_terminated(current_state)
            truncateds[agent] = self._is_truncated(agent)
            obs[agent] = current_state.state
            rewards[agent] = reward
            infos[agent] = info
            self.previous_state_data[agent] = current_state
        
        self._check_overtakes()
        for agent in self.agents:
            infos[agent]["overtakes"] = self.overtake_counts[agent]
            infos[agent]["goals_reached"] = self.goals_reached[agent]
        
        if any(terminateds.values()):
            terminateds = {agent: True for agent in self.agents}

        return obs, rewards, terminateds, truncateds, infos
    
    # override for multi agents
    def _get_all_data(self, timeout: float = 5.0) -> dict:
        """Spin until fresh data has arrived for every agent, in a single spin loop."""
        for agent in self.agents:
            self.latest_data[agent] = None

        end_time = self.get_clock().now().nanoseconds + int(timeout * 1e9)
        while self.get_clock().now().nanoseconds < end_time:
            rclpy.spin_once(self, timeout_sec=0.01)
            if all(self.latest_data[agent] is not None for agent in self.agents):
                return dict(self.latest_data)

        missing = [a for a in self.agents if self.latest_data[a] is None]
        raise TimeoutError(f"No synced data received for agents: {missing}")

    def _build_all_state_data(self) -> dict[str, StateData]:
        """Build state for all agents from a single shared spin."""
        data = self._get_all_data()
        return {
            agent: self.state_builder.build_state(odom, lidar)
            for agent, (odom, lidar) in data.items()
        }
    
    def _is_truncated(self, agent: str) -> bool:
        return (
            self.stall_counters[agent] >= self.stall_limit_steps
            or self.step_counter >= self.max_steps
        )

    def _calculate_progress_reward(self, step_progress: float, agent: str) -> float:
        if step_progress < self.stall_progress_threshold_m:
            self.stall_counters[agent] += 1
        else:
            self.stall_counters[agent] = 0

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
        agent: str,
    ) -> tuple[float, dict]:
        track_progress = self._compute_step_progress(previous_state_data, current_state_data)
        progress_reward = self._calculate_progress_reward(track_progress, agent)

        # rest is identical to base class
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
        
        return {
            "obs": {agent: per_agent_obs_size for agent in self.agents},
            "state": per_agent_obs_size * num_agents,  # Combined state of all agents
            "num_agents": num_agents,
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
