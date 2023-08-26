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
from .goal_positions import goal_positions
from .waypoints import waypoints, Waypoint

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
                 reward_range=0.5, 
                 max_steps=500, 
                 collision_range=0.2, 
                 step_length=0.5, 
                 track='track_1',
                 observation_mode='full',
                 ):
        super().__init__('car_track', car_name, max_steps, step_length)

        # Environment Details ----------------------------------------
        self.MAX_STEPS_PER_GOAL = max_steps
        
        match observation_mode:
            case 'no_position':
                self.OBSERVATION_SIZE = 6 + 10
            case 'lidar_only':
                self.OBSERVATION_SIZE = 10 + 2
            case _:
                self.OBSERVATION_SIZE = 8 + 10

        self.COLLISION_RANGE = collision_range
        self.REWARD_RANGE = reward_range

        self.observation_mode = observation_mode

        # Reset Client -----------------------------------------------
        self.all_goals = goal_positions[track]

        self.goals_reached = 0
        self.start_goal_index = 0
        self.car_waypoints = waypoints[track]

        self.steps_since_last_goal = 0

        self.get_logger().info('Environment Setup Complete')

    def reset(self):
        self.step_counter = 0

        self.set_velocity(0, 0)

        # TODO: Remove Hard coded-ness of 10x10
        
        # Reset Information
        # Generate a random reset position from the waypoints
        car_x, car_y, car_yaw, index = self.car_waypoints[np.random.randint(low=0, high=len(self.car_waypoints))]
        
        self.start_goal_index = index
        self.goal_position = self.all_goals[self.start_goal_index]

        self.steps_since_last_goal = 0
        self.goals_reached = 0

        goal_x, goal_y = self.goal_position

        while not self.timer_future.done():
            rclpy.spin_once(self)

        self.timer_future = Future()

        self.call_reset_service(car_x=car_x, car_y=car_y, car_Y=car_yaw, goal_x=goal_x, goal_y=goal_y)

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
            or has_flipped_over(state[2:6]) or \
            self.goals_reached == len(self.all_goals) or \
            self.steps_since_last_goal >= 10

    def call_reset_service(self, car_x, car_y, car_Y, goal_x, goal_y):
        """
        Reset the car and goal position
        """

        request = Reset.Request()
        request.gx = float(goal_x)
        request.gy = float(goal_y)
        request.cx = float(car_x)
        request.cy = float(car_y)
        request.cyaw = float(car_Y)
        request.flag = "car_and_goal"

        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        return future.result()

    def update_goal_service(self, x, y):
        """
        Reset the goal position
        """

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

        reward = 0

        goal_position = self.goal_position


        prev_distance = math.dist(goal_position, state[:2])
        current_distance = math.dist(goal_position, next_state[:2])
        
        # reward += 10 * (prev_distance - current_distance) / prev_distance 
        

        self.steps_since_last_goal += 1

        if current_distance < self.REWARD_RANGE:
            print(f'Goal #{self.goals_reached} Reached')
            reward += 2
            self.goals_reached += 1

            # Updating Goal Position
            new_x, new_y = self.all_goals[(self.start_goal_index + self.goals_reached) % len(self.all_goals)]
            self.goal_position = [new_x, new_y]

            self.update_goal_service(new_x, new_y)

            self.steps_since_last_goal = 0
        
        if self.steps_since_last_goal >= 10:
            reward -= 10
        
        if self.goals_reached >= len(self.all_goals):
            reward += 100
        
        if has_collided(next_state[8:], self.COLLISION_RANGE) or has_flipped_over(next_state[2:6]):
            reward -= 25  # TODO: find optimal value for this

        return reward
