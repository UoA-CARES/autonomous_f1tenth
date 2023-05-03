import time
import math

import rclpy
from rclpy.node import Node
from rclpy import Future

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger



class CarGoalEnvironment(Node):
    """
    CarGoal Reinforcement Learning Environment:

        Task:
            Here the agent learns to drive the f1tenth car to a goal position

        Observation:
            It's position (x, y), orientation (w, x, y, z) and the goal's position (x, y)

        Action:
            It's linear and angular velocity
        
        Termination Conditions:
            When the agent is within REWARD_RANGE units
        
        Truncation Condition:
            When the number of steps surpasses MAX_STEPS
    """

    def __init__(self, car_name, reward_range=1, max_steps=15, collision_range=0.5, step_length=0.5):
        super().__init__('car_goal_environment')
        
        # Environment Details ----------------------------------------
        self.NAME = car_name
        self.REWARD_RANGE = reward_range
        self.MAX_STEPS = max_steps
        self.COLLISION_RANGE = collision_range
        self.STEP_LENGTH = step_length

        self.step_counter = 0

        # Pub/Sub ----------------------------------------------------
        self.cmd_vel_pub = self.create_publisher(
                Twist,
                f'/model/{self.NAME}/cmd_vel',
                10
            )

        self.odom_sub = self.create_subscription(
            Odometry,
            f'/model/{self.NAME}/odometry',
            self.odom_callback,
            10
            )
        
        self.odom_future = Future()

        # Reset Client -----------------------------------------------
        self.reset_client = self.create_client(
            Trigger,
            'car_goal_reset'
        )

        # while not self.reset_client.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('reset service not available, waiting again...')

        time.sleep(2)

        # TODO: generate goal
        self.goal_position = [0, 0] # x and y

        time.sleep(5)
        
    def reset(self):
        self.step_counter = 0

        # Call reset Service

        time.sleep(self.STEP_LENGTH)
        
        observation = self.get_observation()
        
        info = {}

        return observation, info

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

    

        