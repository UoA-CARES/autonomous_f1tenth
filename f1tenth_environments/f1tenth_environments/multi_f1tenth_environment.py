import re

from pettingzoo import ParallelEnv
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

class MultiF1TenthEnvironment(ParallelEnv, Node):
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

        # Multi agents specific stuff
        self.agents = self._discover_opponent_car_names()
        self.agent_states
        self.stall_counters
        self.agent_goals
    
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
