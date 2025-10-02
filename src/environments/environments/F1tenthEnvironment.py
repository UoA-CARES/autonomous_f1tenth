import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy import Future
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import SetBool
from environment_interfaces.srv import Reset
from .util import ackermann_to_twist
import yaml


class F1tenthEnvironment(Node):
    '''
    Repository Parent Environment:
        
        The responsibilities of this class is the following:
            - handle the topic subscriptions/publishers
            - fetching of car data (raw)
            - define the interface for environments to implement
    '''
    def __init__(self,
                 env_name,
                 car_name,
                 max_steps,
                 step_length,
                 lidar_points = 10,
                 config_path='/home/anyone/autonomous_f1tenth/src/environments/config/config.yaml',
                 ):
        super().__init__(env_name + '_environment')

        if lidar_points < 1:
            raise Exception("Make sure number of lidar points is more than 0")
        

        # Environment Details ----------------------------------------
                
        # Load configuration from YAML file
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        self.NAME = car_name
        self.MAX_STEPS = max_steps
        self.STEP_LENGTH = step_length
        self.LIDAR_POINTS = lidar_points

        self.MAX_ACTIONS = np.asarray([config['actions']['max_speed'], config['actions']['max_turn']])
        self.MIN_ACTIONS = np.asarray([config['actions']['min_speed'], config['actions']['min_turn']])
 
        self.ACTION_NUM = 2

        self.step_counter = 0

        # Pub/Sub ----------------------------------------------------
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            f'/{self.NAME}/cmd_vel',
            1
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

        self.processed_publisher = self.create_publisher(
            LaserScan,
            f'/{self.NAME}/processed_scan',
            1
        )

        self.message_filter = ApproximateTimeSynchronizer(
            [self.odom_sub, self.lidar_sub],
            1,
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


        # Stepping Client ---------------------------------------------

        self.stepping_client = self.create_client(
            SetBool,
            'stepping_service'
        )

        while not self.stepping_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('stepping service not available, waiting again...')

        self.timer = self.create_timer(step_length, self.timer_cb)
        self.timer_future = Future()
        self.LAST_STATE = Future()

    def reset(self):
        raise NotImplementedError('reset() not implemented')

    def step(self, action):
        self.step_counter += 1
        self.call_step(pause=False)

        state = self.get_observation()
        
        lin_vel, steering_angle = action
        self.set_velocity(lin_vel, steering_angle)

        while not self.timer_future.done():
            rclpy.spin_once(self)

        self.timer_future = Future()
        
        

        next_state = self.get_observation()
        self.call_step(pause=True)

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

    def get_data(self) -> tuple[Odometry,LaserScan]:
        rclpy.spin_until_future_complete(self, self.observation_future, timeout_sec=0.5)
        if (self.observation_future.result()) == None:
            future = self.LAST_STATE
            self.get_logger().info("Using previous observation")
        else:
            future = self.observation_future
            self.LAST_STATE = future
        self.observation_future = Future()
        data = future.result()
        return data['odom'], data['lidar']

    def set_velocity(self, lin_vel, steering_angle, L=0.325):
        """
        Publish Twist Message. In place since simulator takes angular velocity commands but policies should produce ackermann steering angle.
        Takes linear velocity and steering ANGLE, NOT angular velocity.
        """
        angular = ackermann_to_twist(steering_angle, lin_vel, L)
        velocity_msg = Twist()
        velocity_msg.angular.z = float(angular)
        velocity_msg.linear.x = float(lin_vel)

        self.cmd_vel_pub.publish(velocity_msg)

    def sleep(self):
        while not self.timer_future.done():
            rclpy.spin_once(self)
        self.timer_future = Future()
    
    def call_step(self, pause):
        request = SetBool.Request()
        request.data = pause

        future = self.stepping_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        return future.result()

    def timer_cb(self):
        self.timer_future.set_result(True)

    def increment_stage(self):
        raise NotImplementedError('Staged training is not implemented')