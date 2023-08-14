import math

import numpy as np
import rclpy
from rclpy import Future
from sensor_msgs.msg import LaserScan
from launch_ros.actions import Node

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy import Future
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from environment_interfaces.srv import CarBeatReset
from environments.F1tenthEnvironment import F1tenthEnvironment
from .termination import has_collided, has_flipped_over
from .util import process_odom, reduce_lidar
from .track_reset import track_info

class CarBeatEnvironment(Node):

    def __init__(self, car_one_name, car_two_name, reward_range=1, max_steps=50, collision_range=0.2, step_length=0.5, track='track_1'):
        super().__init__('car_beat_environment')

        # Environment Details ----------------------------------------
        self.NAME = car_one_name
        self.OTHER_CAR_NAME = car_two_name
        self.MAX_STEPS = max_steps
        self.STEP_LENGTH = step_length
        self.MAX_ACTIONS = np.asarray([3, 3.14])
        self.MIN_ACTIONS = np.asarray([0, -3.14])
        self.MAX_STEPS_PER_GOAL = max_steps

        # TODO: Update this
        self.OBSERVATION_SIZE = 6 + 10  # Car position + Lidar rays
        self.COLLISION_RANGE = collision_range
        self.REWARD_RANGE = reward_range
        self.ACTION_NUM = 2

        self.step_counter = 0

        # Goal/Track Info -----------------------------------------------
        self.goal_number = 0
        self.ftg_goal_number = 1

        self.all_goals = track_info[track]['goals']

        self.car_reset_positions = track_info[track]['reset']

        # Pub/Sub ----------------------------------------------------
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            f'/{self.NAME}/cmd_vel',
            10
        )
        
        self.reset_pub = self.create_publisher(
            Empty,
            f'/reset',
            10
        )

        self.odom_sub_one = Subscriber(
            self,
            Odometry,
            f'/{self.NAME}/odometry',
        )

        self.lidar_sub_one = Subscriber(
            self,
            LaserScan,
            f'/{self.NAME}/scan',
        )

        self.odom_sub_two = Subscriber(
            self,
            Odometry,
            f'/{self.OTHER_CAR_NAME}/odometry',
        )

        self.lidar_sub_two = Subscriber(
            self,
            LaserScan,
            f'/{self.OTHER_CAR_NAME}/scan',
        )

        self.message_filter = ApproximateTimeSynchronizer(
            [self.odom_sub_one, self.lidar_sub_one, self.odom_sub_two, self.lidar_sub_two],
            10,
            0.1,
        )

        self.message_filter.registerCallback(self.message_filter_callback)

        self.observation_future = Future()

        # Reset Client -----------------------------------------------
        self.reset_client = self.create_client(
            CarBeatReset,
            'car_beat_reset'
        )

        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset service not available, waiting again...')

        self.timer = self.create_timer(step_length, self.timer_cb)
        self.timer_future = Future()

    def reset(self):

        self.step_counter = 0

        self.set_velocity(0, 0)

        # TODO: Remove Hard coded-ness of 10x10
        self.goal_number = 0
        self.ftg_goal_number = 1

        self.goal_position = self.generate_goal(self.goal_number)

        while not self.timer_future.done():
            rclpy.spin_once(self)

        self.timer_future = Future()

        self.call_reset_service()

        state, _ = self.get_observation()

        info = {}

        return state, info

    def step(self, action):
        self.step_counter += 1

        _, full_state = self.get_observation()

        lin_vel, ang_vel = action
        self.set_velocity(lin_vel, ang_vel)

        while not self.timer_future.done():
            rclpy.spin_once(self)

        self.timer_future = Future()

        next_state, full_next_state  = self.get_observation()
        reward = self.compute_reward(full_state, full_next_state)
        terminated = self.is_terminated(full_next_state)
        truncated = self.step_counter >= self.MAX_STEPS
        info = {}

        return next_state, reward, terminated, truncated, info

    def message_filter_callback(self, odom_one: Odometry, lidar_one: LaserScan, odom_two: Odometry, lidar_two: LaserScan):
        self.observation_future.set_result({'odom_one': odom_one, 'lidar_one': lidar_one, 'odom_two': odom_two, 'lidar_two': lidar_two})

    def get_data(self):
        rclpy.spin_until_future_complete(self, self.observation_future)
        future = self.observation_future
        self.observation_future = Future()
        data = future.result()
        return data['odom_one'], data['lidar_one'], data['odom_two'], data['lidar_two'] 

    def set_velocity(self, linear, angular):
        """
        Publish Twist messages to f1tenth cmd_vel topic
        """
        velocity_msg = Twist()
        velocity_msg.angular.z = float(angular)
        velocity_msg.linear.x = float(linear)

        self.cmd_vel_pub.publish(velocity_msg)

    def sleep(self):
        while not self.timer_future.done():
            rclpy.spin_once(self)
    
    def timer_cb(self):
        self.timer_future.set_result(True)

    def is_terminated(self, state):
        return has_collided(state[8:19], self.COLLISION_RANGE) \
            or has_flipped_over(state[2:6]) \
            or self.goal_number > self.ftg_goal_number

    def generate_goal(self, number):
        print("Goal", number, "spawned")
        return self.all_goals[number % len(self.all_goals)]

    def call_reset_service(self):
        """
        Reset the car and goal position
        """

        x, y = self.goal_position

        request = CarBeatReset.Request()
        
        request.gx = x
        request.gy = y

        request.cx_one = self.car_reset_positions['x']
        request.cy_one = self.car_reset_positions['y']
        request.cyaw_one = self.car_reset_positions['yaw']

        request.cx_two = self.all_goals[0][0]
        request.cy_two = self.all_goals[0][1]

        # TODO: Fix this
        request.cyaw_two = self.car_reset_positions['yaw']

        request.flag = "car_and_goal"

        # Publish to reset Topic to reset other nodes
        empty_msg = Empty()
        self.reset_pub.publish(empty_msg)

        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        return future.result()

    def update_goal_service(self, number):
        """
        Reset the goal position
        """

        x, y = self.generate_goal(number)
        self.goal_position = [x, y]

        request = CarBeatReset.Request()
        request.gx = x
        request.gy = y
        request.flag = "goal_only"

        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        return future.result()

    def get_observation(self):

        # Get Position and Orientation of F1tenth
        odom_one, lidar_one, odom_two, lidar_two = self.get_data()

        odom_one = process_odom(odom_one)
        odom_two = process_odom(odom_two)

        lidar_one = reduce_lidar(lidar_one)
        lidar_two = reduce_lidar(lidar_two)

        # No x and y position
        state = odom_one[2:] + lidar_one
        full_state = odom_one + lidar_one + odom_two + lidar_two + self.goal_position

        return state, full_state
 
    def compute_reward(self, state, next_state):

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

        ftg_current_distance = math.dist(self.all_goals[self.ftg_goal_number], next_state[18:20])
        
        # Keeping track of FTG car goal number
        if ftg_current_distance < self.REWARD_RANGE:
            self.ftg_goal_number += 1
        
        # If RL car has overtaken FTG car
        if self.goal_number > self.ftg_goal_number:
            reward  += 200

        if has_collided(next_state[8:19], self.COLLISION_RANGE) or has_flipped_over(next_state[2:6]):
            reward -= 25  # TODO: find optimal value for this

        prev_car_one_pos = state[:2]
        prev_car_two_pos = state[18:20]
        
        curr_car_one_pos = next_state[:2]
        curr_car_two_pos = next_state[18:20]
        
        prev_progress = math.dist(prev_car_one_pos, prev_car_two_pos)
        curr_progress = math.dist(curr_car_one_pos, curr_car_two_pos)
        
        reward += (prev_progress - curr_progress) / prev_progress
        
        return reward
