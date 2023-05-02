from time import sleep

import rclpy
from rclpy.node import Node
from rclpy import Future
from rclpy.subscription import Subscription
from message_filters import Subscriber, ApproximateTimeSynchronizer

from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from rclpy.executors import SingleThreadedExecutor
from threading import Thread
from std_srvs.srv import Trigger

import random
import numpy as np
import time
import subprocess
import math

class CarGoalEnvironment(Node):

    def __init__(self, car_name, reward_range=1, max_steps=15, collision_range=0.5, step_length=0.5):
        super().__init__('car_goal_environment')
        
        # Env Details ------------------------------------------------
        self.NAME = car_name
        self.REWARD_RANGE = reward_range
        self.MAX_STEPS = max_steps
        self.COLLISION_RANGE = collision_range
        self.STEP_LENGTH = step_length

        self.step_counter = 0
        
        self.pose = None
        

        # Pub/Sub ----------------------------------------------------
        self.cmd_vel_pub = self.create_publisher(
                Twist,
                f'/model/{self.NAME}/cmd_vel',
                10
            )

        self.pose_sub = self.create_subscription(
            Odometry,
            f'/model/{self.NAME}/odometry',
            self.pose_callback,
            10
            )
        
        self.pose_future = Future()

        # Reset Client -----------------------------------------------
        self.reset_client = self.create_client(
            Trigger,
            'car_goal_reset'
        )

        # while not self.reset_client.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('reset service not available, waiting again...')

        time.sleep(2)

        # Generate Goal
        self.goal_position = [0, 0] # x and y

        time.sleep(5)
        

    def pose_callback(self, pose):
        self.pose_future.set_result(pose)

    def set_velocity(self, linear: float, angular: float):
        velocity_msg = Twist()
        velocity_msg.angular.z = float(angular)
        velocity_msg.linear.x = float(linear)

        self.cmd_vel_pub.publish(velocity_msg)

    def get_odom(self):
        rclpy.spin_until_future_complete(self, self.pose_future)
        future = self.pose_future
        self.pose_future = Future()
        return future.result()
    
    def process_odom(self, odom: Odometry):

        pose = odom.pose.pose
        position = pose.position
        orientation = pose.orientation

        twist = odom.twist.twist
        lin_vel = twist.linear
        ang_vel = twist.angular

        return [position.x, position.y, orientation.w, orientation.x, orientation.y, orientation.z, lin_vel.x, ang_vel.z]

    def get_obs(self):

        # Get Position and Orientation of F1tenth
        odom = self.get_odom()
        odom = self.process_odom(odom)

        # Get Goal Position
        return odom + self.goal_position

    def reset(self):
        self.step_counter = 0

        # Call reset Service

        time.sleep(self.STEP_LENGTH)
        
        observation = self.get_obs()
        
        info = {}

        return observation, info

    def step(self, action):
        self.step_counter += 1

        state = self.get_obs()

        lin_vel, ang_vel = action
        self.set_velocity(lin_vel, ang_vel)

        time.sleep(self.STEP_LENGTH)
        
        next_state = self.get_obs()
        reward = self.compute_reward(state, next_state)
        terminated = self.is_terminated(next_state)
        truncated = self.step_counter >= self.MAX_STEPS
        info = {}

        return next_state, reward, terminated, truncated, info

    def is_terminated(self, observation):
        current_distance = math.dist(observation[-2:], observation[:2])
        return current_distance <= self.REWARD_RANGE
    
    def compute_reward(self, state, next_state):

        goal_position = state[-2:]

        old_distance = math.dist(goal_position, state[:2])
        current_distance = math.dist(goal_position, next_state[:2])

        delta_distance = old_distance - current_distance

        reward = 0

        if current_distance < self.REWARD_RANGE:
            reward += 100

        reward += delta_distance * 10

        return reward

    

        