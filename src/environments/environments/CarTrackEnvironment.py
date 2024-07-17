import math
import rclpy
import numpy as np
from rclpy import Future
import random
from environment_interfaces.srv import Reset
from environments.F1tenthEnvironment import F1tenthEnvironment
from .termination import has_collided, has_flipped_over
from .util import process_odom, avg_lidar, create_lidar_msg, get_all_goals_and_waypoints_in_multi_tracks, ackermann_to_twist
from .goal_positions import goal_positions
from .waypoints import waypoints
from std_srvs.srv import SetBool

class CarTrackEnvironment(F1tenthEnvironment):

    """
    CarTrack Reinforcement Learning Environment:

        Task:
            Agent learns to drive a track

        Observation:
            full:
                Car Position (x, y)
                Car Orientation (x, y, z, w)
                Car Velocity
                Car Angular Velocity
                Lidar Data
            no_position:
                Car Orientation (x, y, z, w)
                Car Velocity
                Car Angular Velocity
                Lidar Data
            lidar_only:
                Car Velocity
                Car Angular Velocity
                Lidar Data

        Action:
            It's linear and angular velocity (Twist)
        
        Reward:
            +2 if it comes within REWARD_RANGE units of a goal
            -25 if it collides with a wall

        Termination Conditions:
            When the agent collides with a wall or the Follow The Gap car
        
        Truncation Condition:
            When the number of steps surpasses MAX_GOALS
    """

    def __init__(self, 
                 car_name, 
                 reward_range=0.5, 
                 max_steps=500, 
                 collision_range=0.2, 
                 step_length=0.5, 
                 track='track_1',
                 observation_mode='lidar_only',
                 max_goals=500,
                 ):
        super().__init__('car_track', car_name, max_steps, step_length)

        # Environment Details ----------------------------------------
        self.MAX_STEPS_PER_GOAL = max_steps
        self.MAX_GOALS = max_goals

        match observation_mode:
            case 'no_position':
                self.OBSERVATION_SIZE = 6 + 10
            case 'lidar_only':
                self.OBSERVATION_SIZE = 10 + 2
            case _:
                self.OBSERVATION_SIZE = 8 + 10

        self.COLLISION_RANGE = collision_range
        self.REWARD_RANGE = reward_range

        self.observation_mode = observation_mode
        self.track = track

        # Reset Client -----------------------------------------------

        self.goals_reached = 0
        self.start_goal_index = 0
        self.steps_since_last_goal = 0

        if 'multi_track' not in track:
            self.all_goals = goal_positions[track]
            self.car_waypoints = waypoints[track]
        else:
            self.all_car_goals, self.all_car_waypoints = get_all_goals_and_waypoints_in_multi_tracks(track)
            self.current_track = list(self.all_car_goals.keys())[0]

            self.all_goals = self.all_car_goals[self.current_track]
            self.car_waypoints = self.all_car_waypoints[self.current_track]

        self.get_logger().info('Environment Setup Complete')

    def reset(self):
        self.step_counter = 0
        self.steps_since_last_goal = 0
        self.goals_reached = 0

        self.set_velocity(0, 0)
        
        if 'multi_track' in self.track:
            self.current_track = random.choice(list(self.all_car_goals.keys()))
            self.all_goals = self.all_car_goals[self.current_track]
            self.car_waypoints = self.all_car_waypoints[self.current_track]

        # New random starting point for car
        car_x, car_y, car_yaw, index = random.choice(self.car_waypoints)
        
        # Update goal pointer to reflect starting position
        self.start_goal_index = index
        self.goal_position = self.all_goals[self.start_goal_index]

        goal_x, goal_y = self.goal_position
        self.call_reset_service(car_x=car_x, car_y=car_y, car_Y=car_yaw, goal_x=goal_x, goal_y=goal_y, car_name=self.NAME)

        self.call_step(pause=False)
        observation, _ = self.get_observation()
        self.call_step(pause=True)

        info = {}

        return observation, info

    def step(self, action):
        self.step_counter += 1

        self.call_step(pause=False)
        _, full_state = self.get_observation()

        lin_vel, steering_angle = action
        
        L = 0.315
        angular = ackermann_to_twist(steering_angle, lin_vel, L)

        self.set_velocity(lin_vel, angular)

        self.sleep()
        
        next_state, full_next_state = self.get_observation()
        self.call_step(pause=True)
        
        reward = self.compute_reward(full_state, full_next_state)
        terminated = self.is_terminated(full_next_state)
        truncated = self.steps_since_last_goal >= 10
        info = {}

        return next_state, reward, terminated, truncated, info

    def is_terminated(self, state):
        return has_collided(state[8:], self.COLLISION_RANGE) \
            or has_flipped_over(state[2:6]) or \
            self.goals_reached >= self.MAX_GOALS

    def get_observation(self):

        # Get Position and Orientation of F1tenth
        odom, lidar = self.get_data()
        odom = process_odom(odom)

        num_points = self.LIDAR_POINTS

        reduced_range = avg_lidar(lidar, num_points)
        
        match (self.observation_mode):
            case 'no_position':
                state = odom[2:] + reduced_range
            case 'lidar_only':
                state = odom[-2:] + reduced_range 
            case _:
                state = odom + reduced_range

        
        scan = create_lidar_msg(lidar, num_points, reduced_range)

        self.processed_publisher.publish(scan)

        full_state = odom + reduced_range

        return state, full_state

    def compute_reward(self, state, next_state):

        reward = 0

        goal_position = self.goal_position

        current_distance = math.dist(goal_position, next_state[:2])
        previous_distance = math.dist(goal_position, state[:2])

        reward += previous_distance - current_distance

        self.steps_since_last_goal += 1

        if current_distance < self.REWARD_RANGE:
            print(f'Goal #{self.goals_reached} Reached')
            reward += 2
            self.goals_reached += 1

            # Updating Goal Position
            new_x, new_y = self.all_goals[(self.start_goal_index + self.goals_reached) % len(self.all_goals)]
            self.goal_position = [new_x, new_y]

            self.update_goal_service(new_x, new_y)

            self.steps_since_last_goal = 0
        
        if self.steps_since_last_goal >= 10:
            reward -= 10

        if has_collided(next_state[8:], self.COLLISION_RANGE) or has_flipped_over(next_state[2:6]):
            reward -= 25

        return reward


    # Utility Functions --------------------------------------------

    def call_reset_service(self, car_x, car_y, car_Y, goal_x, goal_y, car_name):
        """
        Reset the car and goal position
        """

        request = Reset.Request()
        request.car_name = car_name
        request.gx = float(goal_x)
        request.gy = float(goal_y)
        request.cx = float(car_x)
        request.cy = float(car_y)
        request.cyaw = float(car_Y)
        request.flag = "car_and_goal"

        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        return future.result()

    def update_goal_service(self, x, y):
        """
        Reset the goal position
        """

        request = Reset.Request()
        request.gx = x
        request.gy = y
        request.flag = "goal_only"

        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        return future.result()
    
    def sleep(self):
        while not self.timer_future.done():
            rclpy.spin_once(self)

        self.timer_future = Future()
    
    def parse_observation(self, observation):
        
        string = f'CarTrack Observation\n'

        match (self.observation_mode):
            case 'no_position':
                string += f'Orientation: {observation[:4]}\n'
                string += f'Car Velocity: {observation[4]}\n'
                string += f'Car Angular Velocity: {observation[5]}\n'
                string += f'Lidar: {observation[6:]}\n'
            case 'lidar_only':
                string += f'Car Velocity: {observation[0]}\n'
                string += f'Car Angular Velocity: {observation[1]}\n'
                string += f'Lidar: {observation[2:]}\n'
            case _:
                string += f'Position: {observation[:2]}\n'
                string += f'Orientation: {observation[2:6]}\n'
                string += f'Car Velocity: {observation[6]}\n'
                string += f'Car Angular Velocity: {observation[7]}\n'
                string += f'Lidar: {observation[8:]}\n'

        return string
