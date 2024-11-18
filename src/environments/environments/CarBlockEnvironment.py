import math
import random

import numpy as np
import rclpy
from rclpy import Future

from environments.F1tenthEnvironment import F1tenthEnvironment
 
from .util import reduce_lidar, process_odom, generate_position
from .termination import reached_goal, has_collided, has_flipped_over

from environment_interfaces.srv import Reset

class CarBlockEnvironment(F1tenthEnvironment):
    """
    CarBlock Reinforcement Learning Environment:

        Task:
            Agent learns to navigate to a goal position while avoiding obstacles that are dynamically placed at the start of each episode

        Observation:
            Car Position (x, y)
            Car Orientation (x, y, z, w)
            Car Velocity
            Car Angular Velocity
            Lidar Data

        Action:
            Its linear and angular velocity (Twist)
        
        Reward:
            Its progress towards the goal * 10
            +100 if it reaches the goal
            -25 if it collides with an obstacle or flips over

        Termination Conditions:
            When the agent collides with a wall or reaches the goal
        
        Truncation Condition:
            When the number of steps surpasses MAX_STEPS
    """

    def __init__(self, config):
        
        # extract parameters from config dictionary
        car_name = config['car_name']
        reward_range = config.get('reward_range', 0.2)
        max_steps = config.get('max_steps', 50)
        collision_range = config.get('collision_range', 0.2)
        step_length = config.get('step_length', 0.5)

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

        self.call_step(pause=False)
        observation = self.get_observation()
        self.call_step(pause=True)
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
        req.car_name = self.NAME
        
        future = self.reset_client.call_async(req)
        rclpy.spin_until_future_complete(future=future, node=self)

    def get_observation(self):

        # Get Position and Orientation of F1tenth
        odom, lidar = self.get_data()
        odom = process_odom(odom)

        reduced_range = reduce_lidar(lidar, 10)

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
    
    def parse_observation(self, observation):
        string = f'CarBlock Observation\n'
        string += f'Position: {observation[:2]}\n'
        string += f'Orientation: {observation[2:6]}\n'
        string += f'Car Velocity: {observation[6]}\n'
        string += f'Car Angular Velocity: {observation[7]}\n'
        string += f'Lidar: {observation[8:-2]}\n'
        string += f'Goal Position: {observation[-2:]}\n'

        return string

    
