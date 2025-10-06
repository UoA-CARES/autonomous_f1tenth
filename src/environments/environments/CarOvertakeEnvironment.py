import math
import rclpy
import numpy as np
from rclpy import Future
import random
from environment_interfaces.srv import Reset
from environments.F1tenthEnvironment import F1tenthEnvironment
from .util import has_collided, has_flipped_over
from .util import get_track_math_defs, process_ae_lidar, process_odom, avg_lidar, create_lidar_msg, get_all_goals_and_waypoints_in_multi_tracks, ackermann_to_twist, reconstruct_ae_latent
from .util_track_progress import TrackMathDef
from .waypoints import waypoints
from std_srvs.srv import SetBool
from typing import Literal, List, Optional, Tuple
import torch
from datetime import datetime
import yaml

class CarOvertakeEnvironment(F1tenthEnvironment):

    """
    CarOvertake Reinforcement Learning Environment:

        Task:
            Agent learns to drive a track with multiple cars on it. The agent must overtake the cars as it reaches them.

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
            When the agent collides with a wall or any Follow The Gap car
        
        Truncation Condition:
            Reaching max_steps
    """

    def __init__(self, 
                 car_name, 
                 reward_range=0.5, 
                 max_steps=3000, 
                 collision_range=0.2, 
                 step_length=0.5, 
                 track='track_1',
                 observation_mode='lidar_only',
                 config_path='/home/anyone/autonomous_f1tenth/src/environments/config/config.yaml',
                 ):
        
        max_steps = 200
        super().__init__('car_overtake', car_name, reward_range, max_steps, collision_range, step_length, lidar_points = 10, track, observation_mode)

        

        #####################################################################################################################
        # CHANGE SETTINGS HERE, might be specific to environment, therefore not moved to config file (for now at least).
            
        # Reward configuration
        self.BASE_REWARD_FUNCTION:Literal["goal_hitting", "progressive"] = 'progressive'
        self.EXTRA_REWARD_TERMS:List[Literal['penalize_turn']] = []
        self.REWARD_MODIFIERS:List[Tuple[Literal['turn','wall_proximity'],float]] = [('turn', 0.3), ('wall_proximity', 0.7)] # [ (penalize_turn", 0.3), (penalize_wall_proximity, 0.7) ]

        # Observation configuration
        self.EXTRA_OBSERVATIONS:List[Literal['prev_ang_vel']] = []

        # Evaluation settings
        self.MULTI_TRACK_TRAIN_EVAL_SPLIT=0.5 

        #optional stuff
        pretrained_ae_path = "/home/anyone/autonomous_f1tenth/lidar_ae_ftg_rand.pt" #"/ws/lidar_ae_ftg_rand.pt"

        #####################################################################################################################

        # Environment Details ----------------------------------------

        # initialize track progress utilities
        self.PREV_T = None

        if self.LIDAR_PROCESSING == 'pretrained_ae':
            from .autoencoders.lidar_autoencoder import LidarConvAE
            self.AE_LIDAR_MODEL = LidarConvAE()
            self.AE_LIDAR_MODEL.load_state_dict(torch.load(pretrained_ae_path))
            self.AE_LIDAR_MODEL.eval()

        # reward function specific setup:
        if self.BASE_REWARD_FUNCTION == 'progressive':
            self.PROGRESS_NOT_MET_CNT = 0


        # Reset Client -----------------------------------------------

        self.START_WAYPOINT_INDEX = 0
        self.STEPS_SINCE_LAST_GOAL = 0
        self.FULL_CURRENT_STATE = None

        if not self.IS_MULTI_TRACK:
            if "test_track" in track:
                track_key = track[0:-4] # "test_track_xx_xxx" -> "test_track_xx", here due to test_track's different width variants having the same waypoints.
            else:
                track_key = track

            self.TRACK_WAYPOINTS = waypoints[track_key]
            self.CURR_TRACK_MODEL = TrackMathDef(np.array(self.TRACK_WAYPOINTS)[:,:2])
            
        else:
            _, self.all_track_waypoints = get_all_goals_and_waypoints_in_multi_tracks(track)
            self.CURRENT_TRACK_KEY = list(self.all_track_waypoints.keys())[0]

            # set current track waypoints
            self.TRACK_WAYPOINTS = self.all_track_waypoints[self.CURRENT_TRACK_KEY]

            # set track models
            self.ALL_TRACK_MODELS = get_track_math_defs(self.all_track_waypoints)
            self.CURR_TRACK_MODEL = self.ALL_TRACK_MODELS[self.CURRENT_TRACK_KEY]

        if self.IS_MULTI_TRACK:
            # define from which track in the track lists to be used for eval only
            self.EVAL_TRACK_BEGIN_IDX = int(len(self.all_track_waypoints)*self.MULTI_TRACK_TRAIN_EVAL_SPLIT)
            # idx used to loop through eval tracks sequentially
            self.EVAL_TRACK_IDX = 0

        self.get_logger().info('Environment Setup Complete')



#    ____ _        _    ____ ____    _____ _   _ _   _  ____ _____ ___ ___  _   _ ____  
#   / ___| |      / \  / ___/ ___|  |  ___| | | | \ | |/ ___|_   _|_ _/ _ \| \ | / ___| 
#  | |   | |     / _ \ \___ \___ \  | |_  | | | |  \| | |     | |  | | | | |  \| \___ \ 
#  | |___| |___ / ___ \ ___) |__) | |  _| | |_| | |\  | |___  | |  | | |_| | |\  |___) |
#   \____|_____/_/   \_\____/____/  |_|    \___/|_| \_|\____| |_| |___\___/|_| \_|____/ 
                                                                                      

    def get_extra_observation_size(self):
        total = 0
        for obs in self.EXTRA_OBSERVATIONS:
            match obs:
                case 'prev_ang_vel':
                    total += 1
                case _:
                    print("Unknown extra observation.")
        return total
    
    def randomize_yaw(self, yaw, percentage=0.5):
        factor = 1 + random.uniform(-percentage, percentage)
        return yaw + factor
    



    def reset(self):
        self.STEP_COUNTER = 0
        self.STEPS_SINCE_LAST_GOAL = 0
        self.GOALS_REACHED = 0

        self.set_velocity(0, 0)
        
        if self.IS_MULTI_TRACK:
            # Evaluating: loop through eval tracks sequentially
            if self.IS_EVAL:
                eval_track_key_list = list(self.all_track_waypoints.keys())[self.EVAL_TRACK_BEGIN_IDX:]
                self.CURRENT_TRACK_KEY = eval_track_key_list[self.EVAL_TRACK_IDX]
                self.EVAL_TRACK_IDX += 1
                self.EVAL_TRACK_IDX = self.EVAL_TRACK_IDX % len(eval_track_key_list)

            # Training: choose a random track that is not used for evaluation
            else:
                self.CURRENT_TRACK_KEY = random.choice(list(self.all_track_waypoints.keys())[:self.EVAL_TRACK_BEGIN_IDX])
            
            self.TRACK_WAYPOINTS = self.all_track_waypoints[self.CURRENT_TRACK_KEY]

        # start at beginning of track when evaluating
        # if self.is_evaluating:
        #     car_x, car_y, car_yaw, index = self.track_waypoints[10]
        #     car_2_x, car_2_y, car_2_yaw, _ = self.track_waypoints[16]
        #     car_3_x, car_3_y, car_3_yaw, _ = self.track_waypoints[21]
        # # start the car randomly along the track
        # else:
        car_x, car_y, car_yaw, index = random.choice(self.TRACK_WAYPOINTS)
        car_yaw = self.randomize_yaw(car_yaw, 0.25)

        car_2_offset = random.randint(8, 16)  
        car_2_index = (index + car_2_offset) % len(self.TRACK_WAYPOINTS)
        car_2_x, car_2_y, car_2_yaw, _ = self.TRACK_WAYPOINTS[car_2_index]
        car_2_yaw = self.randomize_yaw(car_2_yaw, 0.25)

        car_3_offset = random.randint(20, 40)  
        car_3_index = (index + car_3_offset) % len(self.TRACK_WAYPOINTS)
        car_3_x, car_3_y, car_3_yaw, _ = self.TRACK_WAYPOINTS[car_3_index]
        car_3_yaw = self.randomize_yaw(car_3_yaw, 0.25)

        # Update goal pointer to reflect starting position
        self.START_WAYPOINT_INDEX = index
        x,y,_,_ = self.TRACK_WAYPOINTS[self.START_WAYPOINT_INDEX+1 if self.START_WAYPOINT_INDEX+1 < len(self.TRACK_WAYPOINTS) else 0]# point toward next goal
        self.goal_position = [x,y]

        self.call_reset_service(car_x=car_x, car_y=car_y, car_Y=car_yaw, goal_x=x, goal_y=y, car_name=self.NAME)
        self.call_reset_service(car_x=car_2_x, car_y=car_2_y, car_Y=car_2_yaw, goal_x=x, goal_y=y, car_name='f1tenth_2')
        self.call_reset_service(car_x=car_3_x, car_y=car_3_y, car_Y=car_3_yaw, goal_x=x, goal_y=y, car_name='f1tenth_3')

        # Get initial observation
        self.call_step(pause=False)
        state, full_state , _ = self.get_observation()
        self.FULL_CURRENT_STATE = full_state
        self.call_step(pause=True)

        info = {}

        # get track progress related info
        # set new track model if its multi track
        if self.IS_MULTI_TRACK:
            self.CURR_TRACK_MODEL = self.ALL_TRACK_MODELS[self.CURRENT_TRACK_KEY]
        self.PREV_T = self.CURR_TRACK_MODEL.get_closest_point_on_spline(full_state[:2], t_only=True)

        # reward function specific resets
        if self.BASE_REWARD_FUNCTION == 'progressive':
            self.PROGRESS_NOT_MET_CNT = 0

        return state, info
    
    def start_eval(self):
        self.EVAL_TRACK_IDX = 0
        self.IS_EVAL = True

    def stop_eval(self):
        self.IS_EVAL = False

    def step(self, action):
        self.STEP_COUNTER += 1
        
        # get current state
        full_state = self.FULL_CURRENT_STATE

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
        self.FULL_CURRENT_STATE = full_next_state
        
        # calculate progress along track
        if not self.PREV_T:
            self.PREV_T = self.CURR_TRACK_MODEL.get_closest_point_on_spline(full_state[:2], t_only=True)

        t2 = self.CURR_TRACK_MODEL.get_closest_point_on_spline(full_next_state[:2], t_only=True)
        self.STEP_PROGRESS = self.CURR_TRACK_MODEL.get_distance_along_track_parametric(self.PREV_T, t2, approximate=True)
        self.CENTRE_LINE_OFFSET = self.CURR_TRACK_MODEL.get_distance_to_spline_point(t2, full_next_state[:2])

        self.PREV_T = t2

        # guard against random error from progress estimate. See get_closest_point_on_spline, suspect differential evo have something to do with this.
        if abs(self.STEP_PROGRESS) > (full_next_state[6]/10*3): # traveled distance should not too different from lin vel * step time
            self.STEP_PROGRESS = full_next_state[6]/10*0.8 # reasonable estimation fo traveleled distance based on current lin vel but only 80% of it just in case its exploited by agent

        # calculate reward & end conditions
        reward, reward_info = self.compute_reward(full_state, full_next_state, raw_lidar_range)
        terminated = self.is_terminated(full_next_state, raw_lidar_range)
        truncated = self.is_truncated()

        # additional information that might be logged: based on RESULT observation
        info = {
            'linear_velocity':["avg", full_next_state[6]],
            'angular_velocity_diff':["avg", abs(full_next_state[7] - full_state[7])],
            'traveled distance': ['sum', self.STEP_PROGRESS]
        }
        info.update(reward_info)

        if self.IS_EVAL and (terminated or truncated):
            self.EVAL_TRACK_IDX

        return next_state, reward, terminated, truncated, info

    def is_terminated(self, state, ranges):
        return has_collided(ranges, self.COLLISION_RANGE) \
            or has_flipped_over(state[2:6])

    def is_truncated(self):

        match self.BASE_REWARD_FUNCTION:

            case 'goal_hitting':
                return self.STEPS_SINCE_LAST_GOAL >= 20 or \
                self.STEP_COUNTER >= self.MAX_STEPS
            case 'progressive':
                return self.PROGRESS_NOT_MET_CNT >= 5 or \
                self.STEP_COUNTER >= self.MAX_STEPS
            case _:
                raise Exception("Unknown truncate condition for reward function.")


    def get_observation(self):

        # Get Position and Orientation of F1tenth
        odom, lidar = self.get_data()
        odom = process_odom(odom)
        
        num_points = self.LIDAR_POINTS
        
        # init state
        state = []
        
        # Add odom data
        match (self.ODOM_OBSERVATION_MODE):
            case 'no_position':
                state += odom[2:]
            case 'lidar_only':
                state += odom[-2:] 
            case _:
                state += odom 
        
        # Add lidar data:
        match self.LIDAR_PROCESSING:
            case 'pretrained_ae':
                processed_lidar_range = process_ae_lidar(lidar, self.AE_LIDAR_MODEL, is_latent_only=True)
                visualized_range = reconstruct_ae_latent(lidar, self.AE_LIDAR_MODEL, processed_lidar_range)
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
        
        self.PROCESSED_PUBLISHER.publish(scan)

        state += processed_lidar_range

        # Add extra observation:
        for extra_observation in self.EXTRA_OBSERVATIONS:
            match extra_observation:
                case 'prev_ang_vel':
                    if self.FULL_CURRENT_STATE:
                        state += [self.FULL_CURRENT_STATE[7]]
                    else:
                        state += [state[7]]

        
        full_state = odom + processed_lidar_range

        return state, full_state, lidar.ranges

    def compute_reward(self, state, next_state, raw_lidar_range):
        '''Compute reward based on FULL states: odom + lidar + extra'''
        reward = 0
        reward_info = {}

        # calculate base reward
        match self.BASE_REWARD_FUNCTION:
            case 'goal_hitting':
                base_reward, base_reward_info = self.calculate_goal_hitting_reward(state, next_state, raw_lidar_range)
                reward += base_reward
                reward_info.update(base_reward_info)
            case 'progressive':
                base_reward, base_reward_info = self.calculate_progressive_reward(state, next_state, raw_lidar_range)
                reward += base_reward
                reward_info.update(base_reward_info)
            
            case _:
                raise Exception("Unknown reward function. Check environment.")

        # calulate extra reward terms
        for term in self.EXTRA_REWARD_TERMS:
            match term:
                case 'penalize_turn':
                    turn_penalty = abs(state[7] - next_state[7])*0.12
                    reward -= turn_penalty
                    reward_info.update({"turn_penalty":("avg",turn_penalty)})
        
        # calculate reward modifiers:
        for modifier_type, weight in self.REWARD_MODIFIERS:
            match modifier_type:
                case 'wall_proximity':
                    dist_to_wall = min(raw_lidar_range)
                    close_to_wall_penalize_factor = 1 / (1 + np.exp(35 * (dist_to_wall - 0.5))) #y=\frac{1}{1+e^{35\left(x-0.5\right)}}
                    reward -= reward * close_to_wall_penalize_factor * weight
                    reward_info.update({"dist_to_wall":["avg",dist_to_wall]})
                    print(f"--- Wall proximity penalty factor: {weight} * {close_to_wall_penalize_factor}")   
                case 'turn':
                    angular_vel_diff = abs(state[7] - next_state[7])
                    turning_penalty_factor = 1 - (1 / (1 + np.exp(15 * (angular_vel_diff - 0.3)))) #y=1-\frac{1}{1+e^{15\left(x-0.3\right)}}
                    reward -= reward * turning_penalty_factor * weight
                    print(f"--- Turning penalty factor: {weight} * {turning_penalty_factor}")  

        return reward, reward_info
    
    ##########################################################################################
    ########################## Reward Calculation ############################################
    ##########################################################################################
    def calculate_goal_hitting_reward(self, state, next_state, raw_range):
        reward = 0

        goal_position = self.goal_position

        current_distance = math.dist(goal_position, next_state[:2])
        previous_distance = math.dist(goal_position, state[:2])

        reward += previous_distance - current_distance

        self.STEPS_SINCE_LAST_GOAL += 1

        if current_distance < self.REWARD_RANGE:
            print(f'Goal #{self.GOALS_REACHED} Reached')
            reward += 2
            self.GOALS_REACHED += 1

            # Updating Goal Position
            new_x, new_y, _, _ = self.TRACK_WAYPOINTS[(self.START_WAYPOINT_INDEX + self.GOALS_REACHED) % len(self.TRACK_WAYPOINTS)]
            self.goal_position = [new_x, new_y]

            self.update_goal_service(new_x, new_y)

            self.STEPS_SINCE_LAST_GOAL = 0
        
        if self.STEPS_SINCE_LAST_GOAL >= 20:
            reward -= 10

        if has_collided(raw_range, self.COLLISION_RANGE) or has_flipped_over(next_state[2:6]):
            reward -= 25

        info = {}

        return reward, info
    
    def calculate_progressive_reward(self, state, next_state, raw_range):
        reward = 0

        goal_position = self.goal_position

        current_distance = math.dist(goal_position, next_state[:2])
        
        # keep track of non moving steps
        if self.STEP_PROGRESS < 0.02:
            self.PROGRESS_NOT_MET_CNT += 1
        else:
            self.PROGRESS_NOT_MET_CNT = 0

        reward += self.STEP_PROGRESS

        print(f"Step progress: {self.STEP_PROGRESS}")
       
        self.STEPS_SINCE_LAST_GOAL += 1

        if current_distance < self.REWARD_RANGE:
            print(f'Goal #{self.GOALS_REACHED} Reached')
            # reward += 2
            self.GOALS_REACHED += 1

            # Updating Goal Position
            new_x, new_y, _, _ = self.TRACK_WAYPOINTS[(self.START_WAYPOINT_INDEX + self.GOALS_REACHED) % len(self.TRACK_WAYPOINTS)]
            self.goal_position = [new_x, new_y]

            self.update_goal_service(new_x, new_y)

            self.STEPS_SINCE_LAST_GOAL = 0

        if self.PROGRESS_NOT_MET_CNT >= 5:
            reward -= 2

        if has_collided(raw_range, self.COLLISION_RANGE) or has_flipped_over(next_state[2:6]):
            reward -= 2.5

        info = {}

        return reward, info