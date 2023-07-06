import math

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from message_filters import Subscriber, ApproximateTimeSynchronizer
from nav_msgs.msg import Odometry
from rclpy import Future
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from environment_interfaces.srv import Reset


class ParentCarEnvironment(Node):
    def __init__(self, env_name, car_name, reward_range, max_steps, collision_range, step_length):
        super().__init__(env_name + '_environment')

        # Environment Details ----------------------------------------
        self.NAME = car_name
        self.REWARD_RANGE = reward_range
        self.MAX_STEPS = max_steps
        self.COLLISION_RANGE = collision_range
        self.STEP_LENGTH = step_length

        self.MAX_ACTIONS = np.asarray([3, 3.14])
        self.MIN_ACTIONS = np.asarray([-0.5, -3.14])

        self.OBSERVATION_SIZE = 8 + 10 + 2  # Car position + Lidar rays + goal position
        self.ACTION_NUM = 2

        self.step_counter = 0

        self.check_goal = True

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
        # TODO: Map the lidar to a dynamic topic => of the form /model/<name>/lidar
        self.lidar_sub = Subscriber(
            self,
            LaserScan,
            f'/lidar',
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

        self.goal_position = [10, 10]  # x and y

        self.timer = self.create_timer(step_length, self.timer_cb)
        self.timer_future = Future()

    def timer_cb(self):
        self.timer_future.set_result(True)

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
        print(self.step_counter, self.MAX_STEPS)
        truncated = self.step_counter >= self.MAX_STEPS
        info = {}

        return next_state, reward, terminated, truncated, info

    def call_reset_service(self):
        x, y = self.goal_position
        # TODO: Change x and y to gx and gy
        request = Reset.Request()
        request.gx = x
        request.gy = y

        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        return future.result()

    def get_observation(self):
        raise NotImplementedError('get_observation() not implemented')

    def is_terminated(self, observation):
        """
        Observation (ranges all inclusive):
            0 to 8 => odom
            -1 to -2 => goal x, y
            9 to -3 => lidar
        """

        collided_wall = self.has_collided(observation)
        flipped_over = self.has_flipped_over(observation)

        if collided_wall:
            print("Collided with wall")
        if flipped_over:
            print("Flipped over")

        if self.check_goal:
            current_distance = math.dist(observation[-2:], observation[:2])
            reached_goal = current_distance <= self.REWARD_RANGE

            if reached_goal:
                print("Reached goal")

            return reached_goal or collided_wall or flipped_over
        else:
            return collided_wall or flipped_over

    def has_collided(self, observation):
        lidar_ranges = observation[9:-2]
        return any(0 < ray < self.COLLISION_RANGE for ray in lidar_ranges)

    def has_flipped_over(self, observation):
        w, x, y, z = observation[2:6]
        return abs(x) > 0.5 or abs(y) > 0.5

    def compute_reward(self, state, next_state):
        raise NotImplementedError('compute_reward() not implemented')

    def message_filter_callback(self, odom: Odometry, lidar: LaserScan):
        self.observation_future.set_result({'odom': odom, 'lidar': lidar})

    def get_data(self):
        rclpy.spin_until_future_complete(self, self.observation_future)
        future = self.observation_future
        self.observation_future = Future()
        data = future.result()
        return data['odom'], data['lidar']

    def process_odom(self, odom: Odometry):
        pose = odom.pose.pose
        position = pose.position
        orientation = pose.orientation

        twist = odom.twist.twist
        lin_vel = twist.linear
        ang_vel = twist.angular

        return [position.x, position.y, orientation.w, orientation.x, orientation.y, orientation.z, lin_vel.x,
                ang_vel.z]

    def process_lidar(self, lidar: LaserScan):
        ranges = lidar.ranges
        ranges = np.nan_to_num(ranges, posinf=float(-1), neginf=float(-1))
        ranges = list(ranges)

        intensities = list(lidar.intensities)
        return ranges, intensities

    def set_velocity(self, linear, angular):
        """
        Publish Twist messages to f1tenth cmd_vel topic
        """
        velocity_msg = Twist()
        velocity_msg.angular.z = float(angular)
        velocity_msg.linear.x = float(linear)

        self.cmd_vel_pub.publish(velocity_msg)
