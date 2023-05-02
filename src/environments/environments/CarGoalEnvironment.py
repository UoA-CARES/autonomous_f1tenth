from time import sleep

import rclpy
from rclpy.node import Node
from rclpy.subscription import Subscription
from message_filters import Subscriber, ApproximateTimeSynchronizer

from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from rclpy.executors import SingleThreadedExecutor
from threading import Thread
from std_srvs.srv import Trigger

import random
import time
import subprocess

class CarGoalEnvironment(Node):

    def __init__(self, car_name, reward_range=1, max_steps=15, collision_range=0.5, step_length=0.5):
        super().__init__('car_goal_environment')
        
        # Env Details
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
            Pose,
            f'/model/{self.name}/pose',
            self.pose_callback,
            10
            )
        
        self.reset_client = self.create_client(
            Trigger,
            'car_goal_reset'
        )

        # while not self.reset_client.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('reset service not available, waiting again...')

    def pose_callback(self, pose):
        self.get_logger().info('pose logged')
        self.pose = pose

    def set_velocity(self, linear: float, angular: float):
        velocity_msg = Twist()
        velocity_msg.angular.z = float(angular)
        velocity_msg.linear.x = float(linear)

        self.cmd_vel_pub.publish(velocity_msg)

    def reset(self):
        
        time.sleep(self.step_length)

        rclpy.spin_once(self)

        self.get_logger().info(f'{self.pose}')

        

        