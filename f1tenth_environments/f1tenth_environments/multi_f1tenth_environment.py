import re
from f1tenth_environments.f1tenth_environments.f1tenth_environment import F1tenthEnvironment
import rclpy

from pettingzoo import ParallelEnv
from geometry_msgs.msg import Twist
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

        self.current_track = self.track_names[0]
        self.current_waypoints = self.tracks[self.current_track]
        self.current_track_model = self.track_progress_models[self.current_track]

        sub_depth = 3
        qos = QoSProfile(depth=sub_depth)

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

        if lidar_mode == "raw":
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

        # Multi agents specific stuff
        self.agents = self._discover_opponent_car_names()
        
        self.stall_counters = {agent: 0 for agent in self.agents}
        self.agent_goals = {agent: (0.0, 0.0) for agent in self.agents}
        self.latest_data = {agent: None for agent in self.agents}
        self.message_filters = {}
        self.cmd_vel_pubs = {}

        all_subs = []
        self.odom_subs = {}
        self.lidar_subs = {}

        for agent in self.agents:
            odom_sub = Subscriber(self, Odometry, f"/{agent}/odometry", qos_profile=qos)
            lidar_sub = Subscriber(self, LaserScan, f"/{agent}/scan", qos_profile=qos)
            self.odom_subs[agent] = odom_sub
            self.lidar_subs[agent] = lidar_sub
            all_subs.append(odom_sub)
            all_subs.append(lidar_sub)
            self.cmd_vel_pubs[agent] = self.create_publisher(Twist, f"/{agent}/cmd_vel", 1)

        self.message_filter = ApproximateTimeSynchronizer(all_subs, sub_depth, sync_slop_sec)
        self.message_filter.registerCallback(self._message_filter_callback)
    
    def _message_filter_callback(self, *msgs) -> None:
        for i, agent in self.agents:
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
        obs = {}
        for agent in self.agents:
            state_data = self._build_state_data(agent)
            self.previous_state_data[agent] = state_data
            obs[agent] = state_data.state
        self._set_simulation_paused(paused=True)

        return obs

    def step(self, actions: dict) -> tuple[dict, dict, dict, dict, dict]:
        self.step_counter += 1
        self._set_simulation_paused(paused=False)

        for agent, action in actions.items():
            clipped = np.clip(action, self.min_actions, self.max_actions)
            lin_vel, steering = clipped
            angular = geometry_utils.ackermann_to_twist(steering, lin_vel, self.wheelbase_m)
            msg = Twist()
            msg.linear.x = float(lin_vel)
            msg.angular.z = float(angular)
            self.cmd_vel_pubs[agent].publish(msg)

        self._sleep(self.step_sleep_time_ms)

        obs, rewards, terminateds, truncateds, infos = {}, {}, {}, {}, {}
        self._set_simulation_paused(paused=True)

        for agent in self.agents:
            current_state = self._build_state_data(agent)
            reward, info = self._compute_reward(self.previous_state_data[agent], current_state)
            self._update_goal_progress(agent, current_state)

            terminateds[agent] = self._is_terminated(current_state)
            truncateds[agent] = self._is_truncated(agent)
            obs[agent] = current_state.state
            rewards[agent] = reward
            infos[agent] = info
            self.previous_state_data[agent] = current_state

        return obs, rewards, terminateds, truncateds, infos
    
    @property
    def observation_spaces(self) -> dict:
        return {
            agent: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.observation_size,),
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
