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
from .waypoints import waypoints
from std_srvs.srv import SetBool
from typing import Literal, List, Optional, Tuple
import torch
from datetime import datetime
import random
from collections import deque
from itertools import chain

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
                 ):
        super().__init__('car_track', car_name, max_steps, step_length)

        

        #####################################################################################################################
        # CHANGE SETTINGS HERE, might be specific to environment, therefore not moved to config file (for now at least).

        # Reward configuration
        self.BASE_REWARD_FUNCTION:Literal["goal_hitting", "progressive"] = 'progressive'
        self.EXTRA_REWARD_TERMS:List[Literal['penalize_turn']] = []
        self.REWARD_MODIFIERS:List[Tuple[Literal['turn','wall_proximity'],float]] = [('turn', 0.3), ('wall_proximity', 0.7)] # [ (penalize_turn", 0.3), (penalize_wall_proximity, 0.7) ]

        # Observation configuration
        self.LIDAR_PROCESSING:Literal["avg","pretrained_ae", "raw"] = 'avg'
        self.LIDAR_POINTS = 10 #10, 683
        self.LIDAR_OBS_STACK_SIZE = 3
        
        # TD3AE and SACAE config
        self.IS_AUTO_ENCODER_ALG = False # Here since observation needs to be different: AE alg has dict states
        self.INFO_VECTOR_LENGTH = 2
        self.EXTRA_OBSERVATIONS:List[Literal['prev_ang_vel']] = []
        
        # Evaluation settings
        self.EVAL_TRACK_BEGIN_IDX = 30 # multi_track_01: 20, multi_track_02: 26, multi_track_03: 30
        self.MAX_STEPS_EVAL = 1000
        # self.MAX_EVAL_STEPS_PER_TRACK = 400 # 400 * 6 = 2400 eval steps per xxxxxx steps trained 
        # self.MAX_STEP_PER_EVAL_EPISODE = 20
        # self.eval_episode_step_counter = 0

        # Steering noise addition: to simulate steering command not 100% accurate in real life, sampled uniformly between += noise amp
        self.STEERING_NOISE_AMP = 0.02 #0.02
        

        # Respawning balancing setting: respawn car on track with least steps trained trained on it.
        self.IS_BALANCING_RESET = True
        

        #optional stuff
        pretrained_ae_path = "/home/anyone/autonomous_f1tenth/lidar_ae_ftg_rand.pt" #"/ws/lidar_ae_ftg_rand.pt"

        # Speed and turn limit
        self.MAX_ACTIONS = np.asarray([3, 0.434])
        self.MIN_ACTIONS = np.asarray([0, -0.434])

        #####################################################################################################################

        # Environment Details ----------------------------------------
        self.MAX_STEPS_PER_GOAL = max_steps

        # configure odom observation size:
        match observation_mode:
            case 'lidar_only':
                odom_observation_size = 2
            case 'no_position':
                odom_observation_size = 6
            case _:
                odom_observation_size = 10

        # configure overall observation size
        # for auto encoder based algorithm, observation size only refer to each image input
        if self.IS_AUTO_ENCODER_ALG:
            self.OBSERVATION_SIZE = (self.LIDAR_OBS_STACK_SIZE, self.LIDAR_POINTS)
        else:
            self.OBSERVATION_SIZE = odom_observation_size + (self.LIDAR_POINTS * self.LIDAR_OBS_STACK_SIZE) + self.get_extra_observation_size()

        self.COLLISION_RANGE = collision_range
        self.REWARD_RANGE = reward_range

        self.odom_observation_mode = observation_mode
        self.track = track
        self.is_multi_track = 'multi_track' in track

        # initialize track progress utilities
        self.prev_t = None
        self.all_track_models = None
        self.track_model = None
        self.step_progress = 0

        if self.LIDAR_PROCESSING == 'pretrained_ae':
            from .autoencoders.lidar_autoencoder import LidarConvAE
            self.ae_lidar_model = LidarConvAE()
            self.ae_lidar_model.load_state_dict(torch.load(pretrained_ae_path))
            self.ae_lidar_model.eval()

        # reward function specific setup:
        if self.BASE_REWARD_FUNCTION == 'progressive':
            self.progress_not_met_cnt = 0


        # Reset Client -----------------------------------------------

        self.goals_reached = 0
        self.start_waypoint_index = 0
        self.steps_since_last_goal = 0
        self.full_current_state = None
        
        # simulate steering command not 100% true to actual steer
        if self.STEERING_NOISE_AMP != 0:
            self.episode_steering_skew = random.uniform(-self.STEERING_NOISE_AMP, self.STEERING_NOISE_AMP)

        # using multiple observations
        if self.LIDAR_OBS_STACK_SIZE > 1:
            self.lidar_obs_stack = deque([], maxlen=self.LIDAR_OBS_STACK_SIZE)

        ######## NOT ON MULTI TRACK ########
        if not self.is_multi_track:
            if "test_track" in track:
                track_key = track[0:-4] # "test_track_xx_xxx" -> "test_track_xx", here due to test_track's different width variants having the same waypoints.
            else:
                track_key = track

            self.track_waypoints = waypoints[track_key]
            self.track_model = TrackMathDef(np.array(self.track_waypoints)[:,:2])
        
        ######## ON MULTI TRACK ########
        else:
            _, self.all_track_waypoints = get_all_goals_and_waypoints_in_multi_tracks(track)
            all_track_keys = list(self.all_track_waypoints.keys())
            self.current_track_key = all_track_keys[0]

            if self.IS_BALANCING_RESET:
                self.training_counter_per_track = {track_key:0 for track_key in all_track_keys[:self.EVAL_TRACK_BEGIN_IDX]}

            # set current track waypoints
            self.track_waypoints = self.all_track_waypoints[self.current_track_key]

            # set track models
            self.all_track_models = get_track_math_defs(self.all_track_waypoints)
            self.track_model = self.all_track_models[self.current_track_key]


        # Evaluation related setup ---------------------------------------------------
        self.is_evaluating = False

        if self.is_multi_track:
            # idx used to loop through eval tracks sequentially
            self.eval_track_idx = 0

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



    def reset(self):
        self.step_counter = 0
        self.steps_since_last_goal = 0
        self.goals_reached = 0


        self.set_velocity(0, 0)
        print("---- RESET CALLED")

        # adjust episode steering skew (simulate actual car)
        if self.STEERING_NOISE_AMP != 0:
            self.episode_steering_skew = random.uniform(-self.STEERING_NOISE_AMP, self.STEERING_NOISE_AMP)
        
        if self.is_multi_track:
            # Evaluating: loop through eval tracks sequentially
            if self.is_evaluating:
                eval_track_key_list = list(self.all_track_waypoints.keys())[self.EVAL_TRACK_BEGIN_IDX:]
                self.current_track_key = eval_track_key_list[self.eval_track_idx]

                self.eval_track_idx += 1
                self.eval_track_idx = self.eval_track_idx % len(eval_track_key_list)

            # Training: choose a track that is not used for evaluation
            else:
                if self.IS_BALANCING_RESET:
                    # choose track with least steps trained on it
                    self.current_track_key = min(self.training_counter_per_track, key=self.training_counter_per_track.get)
                else:
                    self.current_track_key = random.choice(list(self.all_track_waypoints.keys())[:self.EVAL_TRACK_BEGIN_IDX])
            
            self.track_waypoints = self.all_track_waypoints[self.current_track_key]

        # start at beginning of track when evaluating
        if self.is_evaluating:
            car_x, car_y, car_yaw, index = self.track_waypoints[3] # 3 in case track_model being weird with start of spline or something, probably just me being schizo
        # start the car randomly along the track
        else:
            car_x, car_y, car_yaw, index = random.choice(self.track_waypoints)
        
        # Update goal pointer to reflect starting position
        self.start_waypoint_index = index
        x,y,_,_ = self.track_waypoints[self.start_waypoint_index+1 if self.start_waypoint_index+1 < len(self.track_waypoints) else 0]# point toward next goal
        self.goal_position = [x,y]

        print(f"New track: {self.current_track_key}")
        # print(f"LOC: ({car_x},{car_y},{car_yaw})")

        self.call_reset_service(car_x=car_x, car_y=car_y, car_Y=car_yaw, goal_x=x, goal_y=y, car_name=self.NAME)


        # Get initial observation
        self.call_step(pause=False)
        state, full_state , _ = self.get_observation()
        self.full_current_state = full_state
        self.call_step(pause=True)

        info = {}

        # get track progress related info
        # set new track model if its multi track
        if self.is_multi_track:
            self.track_model = self.all_track_models[self.current_track_key]
        self.prev_t = self.track_model.get_closest_point_on_spline(full_state[:2], t_only=True)

        # reward function specific resets
        if self.BASE_REWARD_FUNCTION == 'progressive':
            self.progress_not_met_cnt = 0

        return state, info
    
    def start_eval(self):
        self.eval_track_idx = 0
        self.is_evaluating = True

        # eval_track_key_list = list(self.all_track_waypoints.keys())[self.EVAL_TRACK_BEGIN_IDX:]
        # self.eval_counter_per_track = {track_key:0 for track_key in eval_track_key_list}

    
    def stop_eval(self):
        self.is_evaluating = False

    def step(self, action):
        # update counters
        self.step_counter += 1

        if self.IS_BALANCING_RESET:
            # Add to training counter if training
            if not self.is_evaluating:
                self.training_counter_per_track[self.current_track_key] += 1
            # # Add to eval counter if evaluating
            # else:
            #     self.eval_counter_per_track[self.current_track_key] += 1
        
        # get current state
        full_state = self.full_current_state

        # unpause simulation
        self.call_step(pause=False)

        # take action and wait
        lin_vel, steering_angle = action
            # take in steering noise
        if self.STEERING_NOISE_AMP != 0:
            steering_angle = steering_angle + self.episode_steering_skew

        self.set_velocity(lin_vel, steering_angle)

        self.sleep()
        
        # record new state
        next_state, full_next_state, raw_lidar_range = self.get_observation()
        self.call_step(pause=True)

        # set new step as 'current state' for next step
        self.full_current_state = full_next_state
        
        # calculate progress along track
        if not self.prev_t:
            self.prev_t = self.track_model.get_closest_point_on_spline(full_state[:2], t_only=True)

        t2 = self.track_model.get_closest_point_on_spline(full_next_state[:2], t_only=True)
        self.step_progress = self.track_model.get_distance_along_track_parametric(self.prev_t, t2, approximate=True)
        self.center_line_offset = self.track_model.get_distance_to_spline_point(t2, full_next_state[:2])

        self.prev_t = t2

        # guard against random error from progress estimate. See get_closest_point_on_spline, suspect differential evo have something to do with this.
        if abs(self.step_progress) > (full_next_state[6]/10*3): # traveled distance should not too different from lin vel * step time
            self.step_progress = full_next_state[6]/10*0.8 # reasonable estimation fo traveleled distance based on current lin vel but only 80% of it just in case its exploited by agent

        # calculate reward & end conditions
        reward, reward_info = self.compute_reward(full_state, full_next_state, raw_lidar_range)
        terminated = self.is_terminated(full_next_state, raw_lidar_range)
        truncated = self.is_truncated()

        # additional information that might be logged: based on RESULT observation
        info = {
            'linear_velocity':["avg", full_next_state[6]],
            'angular_velocity_diff':["avg", abs(full_next_state[7] - full_state[7])],
            'traveled distance': ['sum', self.step_progress]
        }
        info.update(reward_info)

        if self.is_evaluating and (terminated or truncated):
            self.eval_track_idx

        print(f"Action: {lin_vel} | {steering_angle}")

        return next_state, reward, terminated, truncated, info

    def is_terminated(self, state, ranges):
        if has_collided(ranges, self.COLLISION_RANGE) \
            or has_flipped_over(state[2:6]):
            print(f"TERMINATED: wall {has_collided(ranges, self.COLLISION_RANGE)} | flip {has_flipped_over(state[2:6])}")
        return has_collided(ranges, self.COLLISION_RANGE) \
            or has_flipped_over(state[2:6])

    def is_truncated(self):

        match self.BASE_REWARD_FUNCTION:

            case 'goal_hitting':
                return self.steps_since_last_goal >= 20 or \
                self.step_counter >= self.MAX_STEPS
            
            case 'progressive':
                if self.progress_not_met_cnt >= 5:
                    print(f"TRUNCATED: progress not met") 
                    return True
                # truncate when training
                elif self.is_evaluating == False and self.step_counter >= self.MAX_STEPS:
                    print(f"TRUNCATED: max steps exceeded")
                    return True
                # truncate when evaluation: might be longer
                elif self.is_evaluating == True and self.step_counter >= self.MAX_STEPS_EVAL:
                    print("TRUNCATED: EVAL LAP COMPLETE")
                    return True
                else:
                    return False

            case _:
                raise Exception("Unknown truncate condition for reward function.")

    def get_observation(self):

        # Get Position and Orientation of F1tenth
        odom, lidar = self.get_data()
        odom = process_odom(odom)
        
        num_points = self.LIDAR_POINTS
        
        # init state
        # state = []
        limited_odom = None
        
        # Add odom data
        match (self.odom_observation_mode):
            case 'no_position':
                limited_odom = odom[2:]
            case 'lidar_only':
                limited_odom = odom[-2:] 
            case _:
                limited_odom = odom 
        
        # Add lidar data:
        match self.LIDAR_PROCESSING:
            case 'pretrained_ae':
                processed_lidar_range = process_ae_lidar(lidar, self.ae_lidar_model, is_latent_only=True)
                visualized_range = reconstruct_ae_latent(lidar, self.ae_lidar_model, processed_lidar_range)
                #TODO: get rid of hard coded lidar points num
                scan = create_lidar_msg(lidar, 683, visualized_range)
            case 'avg':
                processed_lidar_range = avg_lidar(lidar, num_points)
                visualized_range = processed_lidar_range
                scan = create_lidar_msg(lidar, num_points, visualized_range)
            case 'raw':
                processed_lidar_range = np.array(lidar.ranges.tolist())
                processed_lidar_range = np.nan_to_num(processed_lidar_range, posinf=-5, nan=-5, neginf=-5).tolist()  
                visualized_range = processed_lidar_range
                scan = create_lidar_msg(lidar, num_points, visualized_range)
        
        self.processed_publisher.publish(scan)

        # state += processed_lidar_range

        # Add extra observation:
        extra_observation = []
        for extra_observation in self.EXTRA_OBSERVATIONS:
            match extra_observation:
                case 'prev_ang_vel':
                    if self.full_current_state:
                        extra_observation += [self.full_current_state[7]]
                    else:
                        extra_observation += [odom[7]]

        full_state = odom + processed_lidar_range
        
        # is using lidar scan stack for temporal info
        if self.LIDAR_OBS_STACK_SIZE > 1:
            # if is first observation, fill stack with current observation
            if len(self.lidar_obs_stack) <= 1:
                for _ in range(0,self.OBSERVATION_SIZE):
                    self.lidar_obs_stack.append(processed_lidar_range)
            # add current observation to stack.
            else:
                self.lidar_obs_stack.append(processed_lidar_range)

        #######################################################
        ####### FORMING ACTUAL STATE TO BE PASSED ON ##########

        #### Check if should pass a dict state
        if self.IS_AUTO_ENCODER_ALG:
            
            # is using lidar scan stack for temporal info
            if self.LIDAR_OBS_STACK_SIZE > 1:
                state = {
                    'image': np.array(self.lidar_obs_stack).reshape((self.LIDAR_OBS_STACK_SIZE,-1)), # e.g. shape (683,) -> (1,683)
                    'vector': np.array(limited_odom), # e.g. shape (2,)
                }
            
            # not using scan stack
            else:
                state = {
                    'image': np.array([processed_lidar_range]).reshape((self.LIDAR_OBS_STACK_SIZE,-1)), # e.g. shape (683,) -> (1,683)
                    'vector': np.array(limited_odom), # e.g. shape (2,)
                }

        #### normal algorithm: flat state
        else:
            # is using lidar scan stack for temporal info
            if self.LIDAR_OBS_STACK_SIZE > 1:
                flattened_lidar_stack = list(chain(*self.lidar_obs_stack))
                state = limited_odom + flattened_lidar_stack + extra_observation
            # not using scan stack
            else:
                state = state = limited_odom + processed_lidar_range + extra_observation

        return state, full_state, lidar.ranges

    def compute_reward(self, state, next_state, raw_lidar_range):
        '''Compute reward based on FULL states: [*odom, *lidar, *extra]'''
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
                    close_to_wall_penalize_factor = 1 / (1 + np.exp(35 * (dist_to_wall - 0.35))) #y=\frac{1}{1+e^{35\left(x-0.35\right)}}
                    reward -= reward * close_to_wall_penalize_factor * weight
                    reward_info.update({"dist_to_wall":["avg",dist_to_wall]})
                    print(f"--- Wall proximity penalty factor: {weight} * {close_to_wall_penalize_factor}")   
                case 'turn':
                    angular_vel_diff = abs(state[7] - next_state[7])
                    turning_penalty_factor = 1 - (1 / (1 + np.exp(12 * (angular_vel_diff - 0.35)))) #y=1-\frac{1}{1+e^{12\left(x-0.35\right)}}
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

        self.steps_since_last_goal += 1

        if current_distance < self.REWARD_RANGE:
            # print(f'Goal #{self.goals_reached} Reached')
            reward += 2
            self.goals_reached += 1

            # Updating Goal Position
            new_x, new_y, _, _ = self.track_waypoints[(self.start_waypoint_index + self.goals_reached) % len(self.track_waypoints)]
            self.goal_position = [new_x, new_y]

            self.update_goal_service(new_x, new_y)

            self.steps_since_last_goal = 0
        
        if self.steps_since_last_goal >= 20:
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
        if self.step_progress < 0.02:
            self.progress_not_met_cnt += 1
        else:
            self.progress_not_met_cnt = 0

        reward += self.step_progress

        print(f"Step progress: {self.step_progress}")
       
        self.steps_since_last_goal += 1

        if current_distance < self.REWARD_RANGE:
            # print(f'Goal #{self.goals_reached} Reached')
            # reward += 2
            self.goals_reached += 1

            # Updating Goal Position
            new_x, new_y, _, _ = self.track_waypoints[(self.start_waypoint_index + self.goals_reached) % len(self.track_waypoints)]
            self.goal_position = [new_x, new_y]

            self.update_goal_service(new_x, new_y)

            self.steps_since_last_goal = 0

        if self.progress_not_met_cnt >= 5:
            reward -= 2
            

        if has_collided(raw_range, self.COLLISION_RANGE) or has_flipped_over(next_state[2:6]):
            reward -= 2.5
            

        info = {}

        return reward, info

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
                string += 'BRU'
                # string += f'Car Velocity: {observation[0]}\n'
                # string += f'Car Angular Velocity: {observation[1]}\n'
                # string += f'Lidar: {observation[2:]}\n'
            case _:
                string += f'Position: {observation[:2]}\n'
                string += f'Orientation: {observation[2:6]}\n'
                string += f'Car Velocity: {observation[6]}\n'
                string += f'Car Angular Velocity: {observation[7]}\n'
                string += f'Lidar: {observation[8:]}\n'

        return string