import math
import rclpy
from rclpy import Future
from .util import process_odom, generate_position
from .termination import reached_goal
from environments.F1tenthEnvironment import F1tenthEnvironment
from environment_interfaces.srv import Reset

class CarGoalEnvironment(F1tenthEnvironment):
    """
    CarGoal Reinforcement Learning Environment:

        Task:
            Here the agent learns to drive the f1tenth car to a goal position.
            This happens in an open area.

        Observation:
            Car Position (x, y)
            Car Orientation (x, y, z, w)
            Car Velocity
            Car Angular Velocity
            Goal Position (x, y)

        Action:
            Its linear and angular velocity
        
        Reward:
            Its progress toward the goal plus,
            +100 if it reaches the goal plus

        Termination Conditions:
            When the agent is within REWARD_RANGE units of the goal
        
        Truncation Condition:
            When the number of steps surpasses MAX_STEPS
    """

    def __init__(self, car_name, reward_range=0.2, max_steps=50, step_length=0.5):
        super().__init__('car_goal', car_name, max_steps, step_length)
        
        self.OBSERVATION_SIZE = 8 + 2 # odom + goal_position
        self.REWARD_RANGE = reward_range
        
        self.goal_position = [10, 10]
        
        self.get_logger().info('Environment Setup Complete')

    def reset(self):
        self.step_counter = 0

        self.set_velocity(0, 0)

        self.goal_position = generate_position(5, 10)

        self.sleep()
        
        self.timer_future = Future()

        new_x, new_y = self.goal_position
        
        self.call_reset_service(new_x, new_y)

        self.call_step(pause=False)
        observation = self.get_observation()
        self.call_step(pause=True)

        info = {}

        return observation, info

    def is_terminated(self, state):
        return reached_goal(state[:2], state[-2:],self.REWARD_RANGE)
    
    def call_reset_service(self, goal_x, goal_y):
        req = Reset.Request()
        
        req.gx = goal_x
        req.gy = goal_y
        req.car_name = self.NAME
        
        future = self.reset_client.call_async(req)
        rclpy.spin_until_future_complete(future=future, node=self)
        
    def get_observation(self):
        odom, _, _ = self.get_data()
        odom = process_odom(odom)

        return odom + self.goal_position

    def compute_reward(self, state, next_state):

        goal_position = state[-2:]

        old_distance = math.dist(goal_position, state[:2])
        current_distance = math.dist(goal_position, next_state[:2])

        delta_distance = old_distance - current_distance

        reward = -0.25

        if current_distance < self.REWARD_RANGE:
            reward += 100

        reward += delta_distance

        return reward
    
    def parse_observation(self, observation):
        
        string = f'CarGoal Observation\n'
        string += f'Position: {observation[:2]}\n'
        string += f'Orientation: {observation[2:6]}\n'
        string += f'Car Velocity: {observation[6]}\n'
        string += f'Car Angular Velocity: {observation[7]}\n'
        string += f'Goal Position: {observation[-2:]}\n'

        return string
    
    
        