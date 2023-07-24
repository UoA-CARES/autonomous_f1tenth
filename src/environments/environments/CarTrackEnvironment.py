import math

import numpy as np
import rclpy
from rclpy import Future
from sensor_msgs.msg import LaserScan

from environment_interfaces.srv import Reset
from environments.F1tenthEnvironment import F1tenthEnvironment
from .termination import has_collided, has_flipped_over
from .util import process_odom, reduce_lidar
from .track_reset import track_info

class CarTrackEnvironment(F1tenthEnvironment):
    """
    CarTrack Reinforcement Learning Environment:

        Task:
            Here the agent learns to drive the f1tenth car to a goal position

        Observation:
            It's position (x, y), orientation (w, x, y, z), lidar points (approx. ~600 rays) and the goal's position (x, y)

        Action:
            It's linear and angular velocity
        
        Reward:
            It's progress toward the goal plus,
            50+ if it reaches the goal plus,
            -25 if it collides with the wall

        Termination Conditions:
            When the agent is within REWARD_RANGE units or,
            When the agent is within COLLISION_RANGE units
        
        Truncation Condition:
            When the number of steps surpasses MAX_STEPS
    """

    def __init__(self, 
                 car_name, 
                 reward_range=1, 
                 max_steps=50, 
                 collision_range=0.2, 
                 step_length=0.5, 
                 track='track_1',
                 observation_mode='full'
                 ):
        super().__init__('car_track', car_name, max_steps, step_length)

        # Environment Details ----------------------------------------
        self.MAX_STEPS_PER_GOAL = max_steps
        
        match observation_mode:
            case 'no_position':
                self.OBSERVATION_SIZE = 6 + 10
            case 'lidar_only':
                self.OBSERVATION_SIZE = 10
            case _:
                self.OBSERVATION_SIZE = 8 + 10

        self.COLLISION_RANGE = collision_range
        self.REWARD_RANGE = reward_range

        self.observation_mode = observation_mode

        # Reset Client -----------------------------------------------
        self.goal_number = 0
        self.all_goals = track_info[track]['goals']

        self.car_reset_positions = track_info[track]['reset']

        self.get_logger().info('Environment Setup Complete')

    def reset(self):
        self.step_counter = 0

        self.set_velocity(0, 0)

        # TODO: Remove Hard coded-ness of 10x10
        self.goal_number = 0
        self.goal_position = self.generate_goal(self.goal_number)

        while not self.timer_future.done():
            rclpy.spin_once(self)

        self.timer_future = Future()

        self.call_reset_service()

        observation, _ = self.get_observation()

        info = {}

        return observation, info

    def step(self, action):
        self.step_counter += 1

        _, full_state = self.get_observation()

        lin_vel, ang_vel = action
        self.set_velocity(lin_vel, ang_vel)

        while not self.timer_future.done():
            rclpy.spin_once(self)

        self.timer_future = Future()

        next_state, full_next_state = self.get_observation()
        reward = self.compute_reward(full_state, full_next_state)
        terminated = self.is_terminated(full_next_state)
        truncated = self.step_counter >= self.MAX_STEPS
        info = {}

        return next_state, reward, terminated, truncated, info

    def is_terminated(self, state):
        return has_collided(state[8:], self.COLLISION_RANGE) \
            or has_flipped_over(state[2:6])

    def generate_goal(self, number):
        print("Goal", number, "spawned")
        return self.all_goals[number % len(self.all_goals)]

    def call_reset_service(self):
        """
        Reset the car and goal position
        """

        x, y = self.goal_position

        request = Reset.Request()
        request.gx = x
        request.gy = y
        request.cx = self.car_reset_positions['x']
        request.cy = self.car_reset_positions['y']
        request.cyaw = self.car_reset_positions['yaw']
        request.flag = "car_and_goal"

        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        return future.result()

    def update_goal_service(self, number):
        """
        Reset the goal position
        """

        x, y = self.generate_goal(number)
        self.goal_position = [x, y]

        request = Reset.Request()
        request.gx = x
        request.gy = y
        request.flag = "goal_only"

        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        return future.result()

    def get_observation(self):

        # Get Position and Orientation of F1tenth
        odom, lidar = self.get_data()
        odom = process_odom(odom)

        reduced_range = reduce_lidar(lidar)

        # Get Goal Position
        
        match (self.observation_mode):
            case 'no_position':
                state = odom[2:] + reduced_range
            case 'lidar_only':
                state = odom[-2:] + reduced_range 
            case _:
                state = odom + reduced_range

        full_state = odom + reduced_range

        return state, full_state

    def compute_reward(self, state, next_state):

        # TESTING ONLY

        # if self.goal_number < len(self.all_goals) - 1:
        #     self.goal_number += 1
        # else:
        #     self.goal_number = 0

        # self.update_goal_service(self.goal_number)
        # ==============================================================

        reward = 0

        goal_position = self.goal_position

        prev_distance = math.dist(goal_position, state[:2])
        current_distance = math.dist(goal_position, next_state[:2])
        
        reward += prev_distance - current_distance
        
        if current_distance < self.REWARD_RANGE:
            reward += 50
            self.goal_number += 1
            self.step_counter = 0
            self.update_goal_service(self.goal_number)

        if has_collided(next_state[8:], self.COLLISION_RANGE) or has_flipped_over(next_state[2:6]):
            reward -= 25  # TODO: find optimal value for this

        return reward
