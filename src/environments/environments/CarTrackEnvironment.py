import math
import rclpy
from rclpy import Future
import random
from environment_interfaces.srv import Reset
from environments.F1tenthEnvironment import F1tenthEnvironment
from .termination import has_collided, has_flipped_over
from .util import process_odom, reduce_lidar, get_all_goals_and_waypoints_in_multi_tracks
from .goal_positions import goal_positions
from .waypoints import waypoints

class CarTrackEnvironment(F1tenthEnvironment):

    def __init__(self, 
                 car_name, 
                 reward_range=0.5, 
                 max_steps=500, 
                 collision_range=0.2, 
                 step_length=0.5, 
                 track='track_1',
                 observation_mode='lidar_only',
                 max_goals=500
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

        self.sleep()

        goal_x, goal_y = self.goal_position
        self.call_reset_service(car_x=car_x, car_y=car_y, car_Y=car_yaw, goal_x=goal_x, goal_y=goal_y)

        observation, _ = self.get_observation()

        info = {}

        return observation, info

    def step(self, action):
        self.step_counter += 1

        _, full_state = self.get_observation()

        lin_vel, ang_vel = action
        self.set_velocity(lin_vel, ang_vel)

        self.sleep()

        next_state, full_next_state = self.get_observation()

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

        reduced_range = reduce_lidar(lidar)

        # Get Goal Position
        
        match (self.observation_mode):
            case 'no_position':
                state = odom[2:] + reduced_range
            case 'lidar_only':
                state = odom[-2:] + reduced_range 
            case _:
                state = odom + reduced_range

        full_state = odom + reduced_range

        return state, full_state

    def compute_reward(self, state, next_state):

        reward = 0

        goal_position = self.goal_position

        current_distance = math.dist(goal_position, next_state[:2])

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

    def call_reset_service(self, car_x, car_y, car_Y, goal_x, goal_y):
        """
        Reset the car and goal position
        """

        request = Reset.Request()
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
