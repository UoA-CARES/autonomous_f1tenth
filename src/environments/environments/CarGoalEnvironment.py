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

class CarGoalEnvironment(Node):

    def __init__(self, car_name, reward_range=1, max_steps=15, collision_range=0.5, step_length=0.5):
        super().__init__('car_goal_environment')
        
        # Env Details ------------------------------------------------
        self.name = car_name
        self.reward_range = reward_range
        self.max_steps = max_steps
        self.collision_range = collision_range
        self.step_length = step_length

        self.step_counter = 0
        
        self.pose = None
        

        # Pub/Sub ----------------------------------------------------
        self.cmd_vel_pub = self.create_publisher(
                Twist,
                f'/model/{self.name}/cmd_vel',
                10
            )

        self.pose_sub = self.create_subscription(
            Odometry,
            f'/model/{self.name}/odometry',
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
        """
        Transforms the raw odometer data into a more digestible format. Here, we only use the following:
            Position:
                x and y
            Quaternion:
                z and w
            Velocity:
                linear and angular
        :param odom: Raw odometer data
        :return: the processed odometer data
        """
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
        
        # Call reset Service

        time.sleep(self.step_length)
        
        observation = self.get_obs()
        
        info = {}

        return observation, info

        

        