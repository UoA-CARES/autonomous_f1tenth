import random

import numpy as np
import rclpy
from rclpy import Future
from sensor_msgs.msg import LaserScan

from environments.ParentCarEnvironment import ParentCarEnvironment


class CarWallEnvironment(ParentCarEnvironment):
    """
    CarWall Reinforcement Learning Environment:

        Task:
            Here the agent learns to drive the f1tenth car to a goal position

        Observation:
            It's position (x, y), orientation (w, x, y, z), lidar points (approx. ~600 rays) and the goal's position (x, y)

        Action:
            It's linear and angular velocity
        
        Reward:
            It's progress toward the goal plus,
            100+ if it reaches the goal plus,
            -50 if it collides with the wall

        Termination Conditions:
            When the agent is within REWARD_RANGE units or,
            When the agent is within COLLISION_RANGE units
        
        Truncation Condition:
            When the number of steps surpasses MAX_STEPS
    """

    def __init__(self, car_name, reward_range=0.2, max_steps=50, collision_range=0.2, step_length=0.5):
        super().__init__('car_wall', car_name, reward_range, max_steps, collision_range, step_length)

    def reset(self):
        self.step_counter = 0

        self.set_velocity(0, 0)

        # TODO: Remove Hard coded-ness of 10x10
        self.goal_position = self.generate_goal()

        while not self.timer_future.done():
            rclpy.spin_once(self)

        self.timer_future = Future()

        self.call_reset_service()

        observation = self.get_observation()

        info = {}

        return observation, info

    def generate_goal(self, inner_bound=3, outer_bound=5):
        inner_bound = float(inner_bound)
        outer_bound = float(outer_bound)

        x_pos = random.uniform(-outer_bound, outer_bound)
        x_pos = x_pos + inner_bound if x_pos >= 0 else x_pos - inner_bound
        y_pos = random.uniform(-outer_bound, outer_bound)
        y_pos = y_pos + inner_bound if y_pos >= 0 else y_pos - inner_bound

        return [x_pos, y_pos]

    def get_observation(self):

        # Get Position and Orientation of F1tenth
        odom, lidar = self.get_data()
        odom = self.process_odom(odom)
        ranges, _ = self.process_lidar(lidar)

        reduced_range = self.avg_reduce_lidar(lidar)

        # Get Goal Position
        return odom + reduced_range + self.goal_position

    def avg_reduce_lidar(self, lidar: LaserScan):
        ranges = lidar.ranges
        ranges = np.nan_to_num(ranges, posinf=float(-1), neginf=float(-1))
        ranges = list(ranges)

        reduced_range = []

        for i in range(10):
            avg = sum(ranges[i * 64: i * 64 + 64]) / 64
            reduced_range.append(avg)

        return reduced_range
