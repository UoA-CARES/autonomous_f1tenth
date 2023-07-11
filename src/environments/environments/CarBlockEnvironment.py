import math
import random

import numpy as np
import rclpy
from rclpy import Future
from sensor_msgs.msg import LaserScan

from environments.F1tenthEnvironment import F1tenthEnvironment
from environments.CarWallEnvironment import CarWallEnvironment

from .util import reduce_lidar, process_odom, generate_position
from .termination import reached_goal, has_collided, has_flipped_over

from environment_interfaces.srv import Reset

class CarBlockEnvironment(F1tenthEnvironment):
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
        super().__init__('car_block', car_name, max_steps, step_length)

        self.OBSERVATION_SIZE = 8 + 10 + 2 # odom + lidar + goal_position
        self.COLLISION_RANGE = collision_range
        self.REWARD_RANGE = reward_range

        self.goal_position = [10, 10]

        self.get_logger().info('Environment Setup Complete')

    def reset(self):
        self.step_counter = 0

        self.set_velocity(0, 0)

        self.goal_position = generate_position(11, 13)

        self.sleep()

        self.timer_future = Future()

        new_x, new_y = self.goal_position
        self.call_reset_service(new_x, new_y)

        observation = self.get_observation()

        info = {}

        return observation, info

    def is_terminated(self, state):
        return \
            reached_goal(state[:2], state[-2:],self.REWARD_RANGE) \
            or has_collided(state[8:-2], self.COLLISION_RANGE) \
            or has_flipped_over(state[2:6])

    def call_reset_service(self, goal_x, goal_y):
        req = Reset.Request()
        
        req.gx = goal_x
        req.gy = goal_y
        
        future = self.reset_client.call_async(req)
        rclpy.spin_until_future_complete(future=future, node=self)

    def get_observation(self):

        # Get Position and Orientation of F1tenth
        odom, lidar = self.get_data()
        odom = process_odom(odom)

        reduced_range = reduce_lidar(lidar)

        # Get Goal Position
        return odom + reduced_range + self.goal_position

    def compute_reward(self, state, next_state):

        goal_position = state[-2:]

        old_distance = math.dist(goal_position, state[:2])
        current_distance = math.dist(goal_position, next_state[:2])

        delta_distance = old_distance - current_distance

        reward = 10 * (delta_distance / old_distance)

        if current_distance < self.REWARD_RANGE:
            reward += 100

        if has_collided(next_state[8:-2], self.COLLISION_RANGE) or has_flipped_over(next_state[2:6]):
            reward -= 25  # TODO: find optimal value for this

        return reward

    
