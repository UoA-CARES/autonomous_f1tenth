import time
import math
import numpy as np
import random

import rclpy
from rclpy.node import Node
from rclpy import Future

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger
from environment_interfaces.srv import Reset


class CarGoalEnvironment(Node):
    """
    CarGoal Reinforcement Learning Environment:

        Task:
            Here the agent learns to drive the f1tenth car to a goal position

        Observation:
            It's position (x, y), orientation (w, x, y, z) and the goal's position (x, y)

        Action:
            It's linear and angular velocity
        
        Reward:
            It's progress toward the goal plus,
            100+ if it reaches the goal plus,

        Termination Conditions:
            When the agent is within REWARD_RANGE units
        
        Truncation Condition:
            When the number of steps surpasses MAX_STEPS
    """

    def __init__(self, car_name, reward_range=1, max_steps=50, step_length=0.5):
        super().__init__('car_goal_environment')
        
        # Environment Details ----------------------------------------
        self.NAME = car_name
        self.REWARD_RANGE = reward_range
        self.MAX_STEPS = max_steps
        self.STEP_LENGTH = step_length

        self.step_counter = 0

        # Pub/Sub ----------------------------------------------------
        self.cmd_vel_pub = self.create_publisher(
                Twist,
                f'/{self.NAME}/cmd_vel',
                10
            )

        self.odom_sub = self.create_subscription(
            Odometry,
            f'/{self.NAME}/odometry',
            self.odom_callback,
            10
            )
        
        self.odom_future = Future()

        # Reset Client -----------------------------------------------
        self.reset_client = self.create_client(
            Reset,
            'car_wall_reset'
        )

        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset service not available, waiting again...')

        time.sleep(2)

        # TODO: generate goal
        self.goal_position = [0, 0] # x and y

        
    def reset(self):
        self.step_counter = 0

        # Call reset Service
        self.set_velocity(0, 0)
        
        self.goal_position = self.generate_goal()

        time.sleep(self.STEP_LENGTH)

        self.call_reset_service()

        time.sleep(self.STEP_LENGTH)
        
        observation = self.get_observation()
        
        info = {}

        return observation, info

    def generate_goal(self, inner_bound=3, outer_bound=7):
        inner_bound = float(inner_bound)
        outer_bound = float(outer_bound)

        x_pos = random.uniform(-outer_bound, outer_bound)
        x_pos = x_pos + inner_bound if x_pos >= 0 else x_pos - inner_bound
        y_pos = random.uniform(-outer_bound, outer_bound)
        y_pos = y_pos + inner_bound if y_pos >= 0 else y_pos - inner_bound

        return [x_pos, y_pos]


    def step(self, action):
        self.step_counter += 1

        state = self.get_observation()

        lin_vel, ang_vel = action
        self.set_velocity(lin_vel, ang_vel)

        time.sleep(self.STEP_LENGTH)
        
        next_state = self.get_observation()
        reward = self.compute_reward(state, next_state)
        terminated = self.is_terminated(next_state)
        truncated = self.step_counter >= self.MAX_STEPS
        info = {}

        return next_state, reward, terminated, truncated, info

    def get_observation(self):

        # Get Position and Orientation of F1tenth
        odom = self.get_odom()
        odom = self.process_odom(odom)

        # Get Goal Position
        return odom + self.goal_position

    def call_reset_service(self):
        x, y = self.goal_position

        request = Reset.Request()
        request.x = x 
        request.y = y

        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        # print(f'Reset Response Recieved: {future.result()}')
        return future.result()
    
    def is_terminated(self, observation):
        current_distance = math.dist(observation[-2:], observation[:2])
        return current_distance <= self.REWARD_RANGE
    
    def compute_reward(self, state, next_state):

        goal_position = state[-2:]

        old_distance = math.dist(goal_position, state[:2])
        current_distance = math.dist(goal_position, next_state[:2])

        delta_distance = old_distance - current_distance

        reward = -0.25
        reward += next_state[6] / 5

        if current_distance < self.REWARD_RANGE:
            reward += 100

        reward += delta_distance

        return reward
    
    def odom_callback(self, odom):
        """
        Callback for listening to Odometry on topic
        """
        self.odom_future.set_result(odom)

    def get_odom(self):
        rclpy.spin_until_future_complete(self, self.odom_future)
        future = self.odom_future
        self.odom_future = Future()
        return future.result()
    
    def process_odom(self, odom: Odometry):
        pose = odom.pose.pose
        position = pose.position
        orientation = pose.orientation

        twist = odom.twist.twist
        lin_vel = twist.linear
        ang_vel = twist.angular

        return [position.x, position.y, orientation.w, orientation.x, orientation.y, orientation.z, lin_vel.x, ang_vel.z]

    def set_velocity(self, linear: float, angular: float):
        """
        Publish Twist messages to f1tenth cmd_vel topic
        """
        velocity_msg = Twist()
        velocity_msg.angular.z = float(angular)
        velocity_msg.linear.x = float(linear)

        self.cmd_vel_pub.publish(velocity_msg)

    

        