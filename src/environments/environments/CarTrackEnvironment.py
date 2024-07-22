import math
import rclpy
import numpy as np
from rclpy import Future
import random
from environment_interfaces.srv import Reset
from environments.F1tenthEnvironment import F1tenthEnvironment
from .termination import has_collided, has_flipped_over
from .util import get_track_math_defs, process_ae_lidar, process_odom, avg_lidar, create_lidar_msg, get_all_goals_and_waypoints_in_multi_tracks, ackermann_to_twist, reconstruct_ae_latent
from .util_track_progress import TrackMathDef
from .goal_positions import goal_positions
from .waypoints import waypoints
from std_srvs.srv import SetBool
from typing import Literal, List, Optional
import torch
from datetime import datetime

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

        

        #####################################################################################################################
        # CHANGE SETTINGS HERE, might be specific to environment, therefore not moved to config file (for now at least).

        # Reward configuration
        self.BASE_REWARD_FUNCTION:Literal["goal_hitting", "progressive"] = 'progressive'
        self.EXTRA_REWARD_TERMS:List[Literal['penalize_turn']] = []

        # Observation configuration
        self.LIDAR_PROCESSING:Literal["avg","pretrained_ae", "raw"] = 'avg'
        self.EXTRA_OBSERVATIONS:List[Literal['prev_ang_vel']] = []

        #optional stuff
        pretrained_ae_path = "/home/anyone/autonomous_f1tenth/lidar_ae_ftg_rand.pt" #"/ws/lidar_ae_ftg_rand.pt"

        #####################################################################################################################

        # Environment Details ----------------------------------------
        self.MAX_STEPS_PER_GOAL = max_steps
        self.MAX_GOALS = max_goals

        # configure odom observation size:
        match observation_mode:
            case 'lidar_only':
                odom_observation_size = 2
            case 'no_position':
                odom_observation_size = 6
            case _:
                odom_observation_size = 10

        # configure overall observation size
        self.OBSERVATION_SIZE = odom_observation_size + self.LIDAR_POINTS+ self.get_extra_observation_size()


        self.COLLISION_RANGE = collision_range
        self.REWARD_RANGE = reward_range

        self.odom_observation_mode = observation_mode
        self.track = track

        # observation method specific setup
        if 'prev_ang_vel' in self.EXTRA_OBSERVATIONS:
            self.prev_ang_vel = 0

        if self.LIDAR_PROCESSING == 'pretrained_ae':
            from .autoencoders.lidar_autoencoder import LidarConvAE
            self.ae_lidar_model = LidarConvAE()
            self.ae_lidar_model.load_state_dict(torch.load(pretrained_ae_path))
            self.ae_lidar_model.eval()

        # reward function specific setup:
        if self.BASE_REWARD_FUNCTION == 'progressive':
            self.prev_t = None
            self.progress_not_met_cnt = 0
            self.last_step_progress = 0

            self.all_track_models = None
            self.track_model = None
        
        


        # Reset Client -----------------------------------------------

        self.goals_reached = 0
        self.start_goal_index = 0
        self.steps_since_last_goal = 0
        self.full_current_state = None

        if 'multi_track' not in track:
            self.all_goals = goal_positions[track]
            self.car_waypoints = waypoints[track]

            if self.BASE_REWARD_FUNCTION == 'progressive':
                self.track_model = TrackMathDef(np.array(self.car_waypoints)[:,:2])
        else:
            self.all_car_goals, self.all_car_waypoints = get_all_goals_and_waypoints_in_multi_tracks(track)
            self.current_track = list(self.all_car_goals.keys())[0]

            self.all_goals = self.all_car_goals[self.current_track]
            self.car_waypoints = self.all_car_waypoints[self.current_track]

            if self.BASE_REWARD_FUNCTION == 'progressive':
                self.all_track_models = get_track_math_defs(self.all_car_waypoints)
                self.track_model = self.all_track_models[self.current_track]

        self.get_logger().info('Environment Setup Complete')



#    ____ _        _    ____ ____    _____ _   _ _   _  ____ _____ ___ ___  _   _ ____  
#   / ___| |      / \  / ___/ ___|  |  ___| | | | \ | |/ ___|_   _|_ _/ _ \| \ | / ___| 
#  | |   | |     / _ \ \___ \___ \  | |_  | | | |  \| | |     | |  | | | | |  \| \___ \ 
#  | |___| |___ / ___ \ ___) |__) | |  _| | |_| | |\  | |___  | |  | | |_| | |\  |___) |
#   \____|_____/_/   \_\____/____/  |_|    \___/|_| \_|\____| |_| |___\___/|_| \_|____/ 
                                                                                      

    def get_extra_observation_size(self):
        if self.EXTRA_OBSERVATIONS:
            total = 0
            for obs in self.EXTRA_OBSERVATIONS:
                match obs:
                    case 'prev_ang_vel':
                        total += 1
                    case _:
                        print("Unknown extra observation.")
            return total
        else:
            return 0


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

        # Get initial observation
        self.call_step(pause=False)
        state, full_state , _ = self.get_observation()
        self.full_current_state = full_state
        self.call_step(pause=True)

        info = {}

         # observation method specific resets
        if 'prev_ang_vel' in self.EXTRA_OBSERVATIONS:
            self.prev_ang_vel = 0

        # reward function specific resets
        if self.BASE_REWARD_FUNCTION == 'progressive':
            self.prev_t = self.track_model.get_closest_point_on_spline(full_state[:2], t_only=True)
            self.progress_not_met_cnt = 0

        return state, info

    def step(self, action):
        self.step_counter += 1
        
        # get current state
        full_state = self.full_current_state

        # unpause simulation
        self.call_step(pause=False)

        # take action and wait
        lin_vel, steering_angle = action
        self.set_velocity(lin_vel, steering_angle)

        self.sleep()
        
        # record new state
        next_state, full_next_state, raw_lidar_range = self.get_observation()
        self.call_step(pause=True)

        # set new step as 'current state' for next step
        self.full_current_state = full_next_state
        
        # calculate reward & end conditions
        reward = self.compute_reward(full_state, full_next_state, raw_lidar_range)
        terminated = self.is_terminated(full_next_state, raw_lidar_range)
        truncated = self.is_truncated()

        info = {}

        return next_state, reward, terminated, truncated, info

    def is_terminated(self, state, ranges):
        return has_collided(ranges, self.COLLISION_RANGE) \
            or has_flipped_over(state[2:6])

    def is_truncated(self):
        match self.BASE_REWARD_FUNCTION:
            case 'goal_hitting':
                return self.steps_since_last_goal >= 20 or \
                self.goals_reached >= self.MAX_GOALS or \
                self.step_counter >= self.MAX_STEPS
            case 'progressive':
                return self.progress_not_met_cnt >= 3 or \
                self.goals_reached >= self.MAX_GOALS or \
                self.step_counter >= self.MAX_STEPS

    def get_observation(self):

        # Get Position and Orientation of F1tenth
        odom, lidar = self.get_data()
        odom = process_odom(odom)
        
        num_points = self.LIDAR_POINTS
        
        # init state
        state = []
        
        # Add odom data
        match (self.odom_observation_mode):
            case 'no_position':
                state += odom[2:]
            case 'lidar_only':
                state += odom[-2:] 
            case _:
                state += odom 
        
        # Add lidar data:
        match self.LIDAR_PROCESSING:
            case 'pretrained_ae':
                processed_lidar_range = process_ae_lidar(lidar, self.ae_lidar_model, is_latent_only=True)
                visualized_range = reconstruct_ae_latent(lidar, self.ae_lidar_model, processed_lidar_range)
                #TODO: get rid of hard coded lidar points num
                scan = create_lidar_msg(lidar, 682, visualized_range)
            case 'avg':
                processed_lidar_range = avg_lidar(lidar, num_points)
                visualized_range = processed_lidar_range
                scan = create_lidar_msg(lidar, num_points, visualized_range)
            case 'raw':
                processed_lidar_range = np.array(lidar.ranges.tolist())
                processed_lidar_range = np.nan_to_num(processed_lidar_range, posinf=-5, nan=-1, neginf=-5).tolist()  
                visualized_range = processed_lidar_range
                scan = create_lidar_msg(lidar, num_points, visualized_range)
        
        self.processed_publisher.publish(scan)

        state += processed_lidar_range
        
        full_state = odom + processed_lidar_range

        return state, full_state, lidar.ranges

    def compute_reward(self, state, next_state, raw_lidar_range):
        reward = 0

        # calculate base reward
        match self.BASE_REWARD_FUNCTION:
            case 'goal_hitting':
                reward += self.calculate_goal_hitting_reward(state, next_state, raw_lidar_range)
            case 'progressive':
                reward += self.calculate_progressive_reward(state, next_state, raw_lidar_range)
        
        # calulate extra reward terms
        for term in self.EXTRA_REWARD_TERMS:
            match term:
                case 'penalize_turn':
                    turn_penalty = abs(state[7] - next_state[7])*0.12
                    reward -= turn_penalty

        return reward
    
    ##########################################################################################
    ########################## Reward Calculation ############################################
    ##########################################################################################
    def calculate_goal_hitting_reward(self, state, next_state, raw_range):
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
        
        if self.steps_since_last_goal >= 20:
            reward -= 10

        if has_collided(raw_range, self.COLLISION_RANGE) or has_flipped_over(next_state[2:6]):
            reward -= 25

        return reward
    
    def calculate_progressive_reward(self, state, next_state, raw_range):
        reward = 0

        goal_position = self.goal_position

        current_distance = math.dist(goal_position, next_state[:2])

        if not self.prev_t:
            self.prev_t = self.track_model.get_closest_point_on_spline(state[:2], t_only=True)

        t2 = self.track_model.get_closest_point_on_spline(next_state[:2], t_only=True)
        
        step_progress = self.track_model.get_distance_along_track_parametric(self.prev_t, t2)
        self.prev_t = t2
        
        # keep track of non moving steps
        if step_progress < 0.02:
            self.progress_not_met_cnt += 1
        else:
            self.progress_not_met_cnt = 0


        #guard against random error from progress estimate
        if abs(step_progress) > 1:
            reward += 0.01
        else:
            reward += step_progress

        print(f"Step progress: {step_progress}")
       
        self.steps_since_last_goal += 1

        if current_distance < self.REWARD_RANGE:
            print(f'Goal #{self.goals_reached} Reached')
            # reward += 2
            self.goals_reached += 1

            # Updating Goal Position
            new_x, new_y = self.all_goals[(self.start_goal_index + self.goals_reached) % len(self.all_goals)]
            self.goal_position = [new_x, new_y]

            self.update_goal_service(new_x, new_y)

            self.steps_since_last_goal = 0

        if self.progress_not_met_cnt >= 3:
            reward -= 2

        if has_collided(raw_range, self.COLLISION_RANGE) or has_flipped_over(next_state[2:6]):
            reward -= 2.5

        return reward

    ##########################################################################################
    ########################## Utility Functions #############################################
    ##########################################################################################

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

        match (self.odom_observation_mode):
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
