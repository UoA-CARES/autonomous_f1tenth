import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy import Future
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from environment_interfaces.srv import Reset
from .util import ackermann_to_twist


class F1tenthEnvironment(Node):
    '''
    Repository Parent Environment:
        
        The responsibilities of this class is the following:
            - handle the topic subscriptions/publishers
            - fetching of car data (raw)
            - define the interface for environments to implement
    '''
    def __init__(self, env_name, car_name, max_steps, step_length):
        super().__init__(env_name + '_environment')

        # Environment Details ----------------------------------------
        self.NAME = car_name
        self.MAX_STEPS = max_steps
        self.STEP_LENGTH = step_length

        self.MAX_ACTIONS = np.asarray([0.5, 0.85])
        self.MIN_ACTIONS = np.asarray([0, -0.85])
 
        self.ACTION_NUM = 2

        self.step_counter = 0

        # Pub/Sub ----------------------------------------------------
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            f'/{self.NAME}/cmd_vel',
            10
        )

        self.odom_sub = Subscriber(
            self,
            Odometry,
            f'/{self.NAME}/odometry',
        )

        self.lidar_sub = Subscriber(
            self,
            LaserScan,
            f'/{self.NAME}/scan',
        )

        self.message_filter = ApproximateTimeSynchronizer(
            [self.odom_sub, self.lidar_sub],
            10,
            0.1,
        )

        self.message_filter.registerCallback(self.message_filter_callback)

        self.observation_future = Future()

        # Reset Client -----------------------------------------------
        self.reset_client = self.create_client(
            Reset,
            env_name + '_reset'
        )

        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset service not available, waiting again...')

        self.timer = self.create_timer(step_length, self.timer_cb)
        self.timer_future = Future()

    def reset(self):
        raise NotImplementedError('reset() not implemented')

    def step(self, action):
        self.step_counter += 1

        state = self.get_observation()

        lin_vel, ang_vel = action
        self.set_velocity(lin_vel, ang_vel)

        while not self.timer_future.done():
            rclpy.spin_once(self)

        self.timer_future = Future()

        next_state = self.get_observation()
        reward = self.compute_reward(state, next_state)
        terminated = self.is_terminated(next_state)
        truncated = self.step_counter >= self.MAX_STEPS
        info = {}

        return next_state, reward, terminated, truncated, info

    def get_observation(self):
        raise NotImplementedError('get_observation() not implemented')

    def compute_reward(self, state, next_state):
        raise NotImplementedError('compute_reward() not implemented')

    def is_terminated(self, state):
        raise NotImplementedError('is_terminated() not implemented')

    def message_filter_callback(self, odom: Odometry, lidar: LaserScan):
        self.observation_future.set_result({'odom': odom, 'lidar': lidar})

    def get_data(self):
        rclpy.spin_until_future_complete(self, self.observation_future)
        future = self.observation_future
        self.observation_future = Future()
        data = future.result()
        return data['odom'], data['lidar']

    def set_velocity(self, linear, angle):
        """
        Publish Twist messages to f1tenth cmd_vel topic
        """
        L = 0.25
        velocity_msg = Twist()
        angular = ackermann_to_twist(angle, linear, L)
        velocity_msg.angular.z = float(angular)
        velocity_msg.linear.x = float(linear)

        self.cmd_vel_pub.publish(velocity_msg)

    def sleep(self):
        while not self.timer_future.done():
            rclpy.spin_once(self)
    
    def timer_cb(self):
        self.timer_future.set_result(True)
