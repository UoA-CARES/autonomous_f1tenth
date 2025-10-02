import math
import rclpy
import numpy as np
from rclpy import Future
import random
from environment_interfaces.srv import Reset
from environments.F1tenthEnvironment import F1tenthEnvironment
from .util import get_track_math_defs, process_ae_lidar, process_odom, avg_lidar, create_lidar_msg, get_all_goals_and_waypoints_in_multi_tracks, ackermann_to_twist, reconstruct_ae_latent, has_collided, has_flipped_over, get_training_stages
from .util_track_progress import TrackMathDef
from .waypoints import waypoints
from std_srvs.srv import SetBool
from typing import Literal, List, Optional, Tuple
import torch
from datetime import datetime
import yaml
import scipy

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
                 is_staged_training=False,
                 config_path='/home/anyone/autonomous_f1tenth/src/environments/config/config.yaml',
                 ):
        super().__init__('car_track', car_name, max_steps, step_length)

        

        #####################################################################################################################
        # CHANGE SETTINGS HERE, might be specific to environment, therefore not moved to config file (for now at least).
        
        # Load configuration from YAML file
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        # Reward configuration
        self.BASE_REWARD_FUNCTION:Literal["goal_hitting", "progressive"] = 'progressive'
        self.EXTRA_REWARD_TERMS:List[Literal['penalize_turn']] = []
        self.REWARD_MODIFIERS:List[Tuple[Literal['turn','wall_proximity'],float]] = [('turn', 0.3), ('wall_proximity', 0.7)] # [ (penalize_turn", 0.3), (penalize_wall_proximity, 0.7) ]

        # Observation configuration
        self.LIDAR_PROCESSING:Literal["avg","pretrained_ae", "raw", "ae"] = 'ae'
        self.LIDAR_POINTS = 683 #683
        self.EXTRA_OBSERVATIONS:List[Literal['prev_ang_vel']] = []

        # Evaluation settings - configure train/eval split based on track
        if track == 'narrow_multi_track':


            self.MULTI_TRACK_TRAIN_EVAL_SPLIT = (12/15) # 12 train, 3 eval
        else:
            self.MULTI_TRACK_TRAIN_EVAL_SPLIT = 0.5
            
        # Toggles whether to use staged training on multi-tracks
        self.IS_STAGED_TRAINING = is_staged_training
        
        if self.IS_STAGED_TRAINING:
            self.TRAINING_STAGES = get_training_stages(track)
            self.CURRENT_TRAINING_STAGE = 0
            self.TRAINING_IDX = self.TRAINING_STAGES[self.CURRENT_TRAINING_STAGE][0]
            self.EVAL_IDX = self.TRAINING_STAGES[self.CURRENT_TRAINING_STAGE][1]


        #optional stuff
        pretrained_ae_path = "/home/anyone/autonomous_f1tenth/lidar_ae_ftg_rand.pt" #"/ws/lidar_ae_ftg_rand.pt"
        self.ENCODER = None
        self.DECODER = None
        
        # Speed and turn limit
        self.MAX_ACTIONS = np.asarray([config['actions']['max_speed'], config['actions']['max_turn']])
        self.MIN_ACTIONS = np.asarray([config['actions']['min_speed'], config['actions']['min_turn']])

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
        self.OBSERVATION_SIZE = odom_observation_size + self.LIDAR_POINTS+ self.get_extra_observation_size()
        #self.OBSERVATION_SIZE = {"lidar": self.LIDAR_POINTS, "vector": odom_observation_size}

        self.COLLISION_RANGE = collision_range
        self.REWARD_RANGE = reward_range

        self.ODOM_OBSERVATION_MODE = observation_mode
        self.TRACK = track
        self.IS_MULTI_TRACK = 'multi_track' in track or track == 'staged_tracks'


        # initialize track progress utilities
        self.PREV_T = None
        self.ALL_TRACK_MODELS = None
        self.TRACK_MODEL = None
        self.STEP_PROGRESS = 0
        
        # Evaluation related setup ---------------------------------------------------
        self.IS_EVAL = False

        if self.LIDAR_PROCESSING == 'pretrained_ae':
            from .autoencoders.lidar_autoencoder import LidarConvAE
            self.AE_LIDAR_MODEL = LidarConvAE()
            self.AE_LIDAR_MODEL.load_state_dict(torch.load(pretrained_ae_path))
            self.AE_LIDAR_MODEL.eval()
        
        if self.LIDAR_PROCESSING == 'ae':
            from .autoencoders.lidar_autoencoder import LidarConvAE
            self.AE_LIDAR_MODEL = LidarConvAE(encoder=self.encoder, decoder=self.decoder)
            if self.IS_EVAL:
                self.AE_LIDAR_MODEL.eval()
            else:
                self.AE_LOSS_FUNCTION = torch.nn.MSELoss()
                self.AE_OPTIMIZER = torch.optim.Adam(self.AE_LIDAR_MODEL.parameters(), lr=1e-3)

        # reward function specific setup:
        if self.BASE_REWARD_FUNCTION == 'progressive':
            self.PROGRESS_NOT_MET_CNT = 0


        # Reset Client -----------------------------------------------

        self.GOALS_REACHED = 0
        self.START_WAYPOINT_INDEX = 0
        self.STEPS_SINCE_LAST_GOAL = 0
        self.FULL_CURRENT_STATE = None

        if not self.IS_MULTI_TRACK:
            if "test_track" in track:
                track_key = track[0:-4] # "test_track_xx_xxx" -> "test_track_xx", here due to test_track's different width variants having the same waypoints.    
            else:
                track_key = track

            self.TRACK_WAYPOINTS = waypoints[track_key]
            self.TRACK_MODEL = TrackMathDef(np.array(self.TRACK_WAYPOINTS)[:,:2])
        else:
            _, self.ALL_TRACK_WAYPOINTS = get_all_goals_and_waypoints_in_multi_tracks(track)
            if self.IS_STAGED_TRAINING:
                self.CURRENT_TRACK_KEY = list(self.ALL_TRACK_WAYPOINTS.keys())[self.TRAINING_IDX[0]]
            else:
                self.CURRENT_TRACK_KEY = list(self.ALL_TRACK_WAYPOINTS.keys())[0]

            # set track models
            self.ALL_TRACK_MODELS = get_track_math_defs(self.ALL_TRACK_WAYPOINTS)
            self.TRACK_MODEL = self.ALL_TRACK_MODELS[self.CURRENT_TRACK_KEY]



        if self.IS_MULTI_TRACK:
            if self.IS_STAGED_TRAINING:
                self.EVAL_TRACK_BEGIN_IDX = None
                self.get_logger().info(f"Track '{track}', {self.TRAINING_IDX} training, {self.EVAL_IDX} evaluation")
            else:
                # define from which track in the track lists to be used for eval only
                self.EVAL_TRACK_BEGIN_IDX = int(len(self.ALL_TRACK_WAYPOINTS)*self.MULTI_TRACK_TRAIN_EVAL_SPLIT)
                # Debug logging
                total_tracks = len(self.ALL_TRACK_WAYPOINTS)
                training_tracks = self.EVAL_TRACK_BEGIN_IDX
                eval_tracks = total_tracks - training_tracks
                self.get_logger().info(f"Track '{track}' split: {total_tracks} total, {training_tracks} training, {eval_tracks} evaluation (split={self.MULTI_TRACK_TRAIN_EVAL_SPLIT})")
            
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

    def reset(self):
        self.step_counter = 0
        self.STEPS_SINCE_LAST_GOAL = 0
        self.GOALS_REACHED = 0

        self.set_velocity(0, 0)


        if self.IS_MULTI_TRACK:
            # Check if we have dedicated evaluation tracks

            if self.EVAL_TRACK_BEGIN_IDX is not None and self.EVAL_TRACK_BEGIN_IDX >= len(self.ALL_TRACK_WAYPOINTS):
                # No dedicated eval tracks (split = 1.0), use all tracks for both training and evaluation
                if self.IS_EVAL:
                    # For evaluation, cycle through all tracks sequentially
                    all_track_keys = list(self.ALL_TRACK_WAYPOINTS.keys())
                    self.CURRENT_TRACK_KEY = all_track_keys[self.EVAL_TRACK_IDX]
                    self.EVAL_TRACK_IDX += 1
                    self.EVAL_TRACK_IDX = self.EVAL_TRACK_IDX % len(all_track_keys)
                else:
                    # For training, choose random track from all tracks
                    self.CURRENT_TRACK_KEY = random.choice(list(self.ALL_TRACK_WAYPOINTS.keys()))
            else:
                # We have dedicated evaluation tracks (split < 1.0)
                if self.IS_EVAL:
                    # Evaluating: loop through eval tracks sequentially

                    if self.IS_STAGED_TRAINING:
                        eval_track_key_list = list(self.ALL_TRACK_WAYPOINTS.keys())[self.EVAL_IDX[0]:self.EVAL_IDX[1] + 1]
                    else:
                        eval_track_key_list = list(self.ALL_TRACK_WAYPOINTS.keys())[self.EVAL_TRACK_BEGIN_IDX:]

                    self.CURRENT_TRACK_KEY = eval_track_key_list[self.EVAL_TRACK_IDX]
                    self.EVAL_TRACK_IDX += 1
                    self.EVAL_TRACK_IDX = self.EVAL_TRACK_IDX % len(eval_track_key_list)
                else:
                    # Training: choose a random track that is not used for evaluation

                    if self.IS_STAGED_TRAINING:
                        self.CURRENT_TRACK_KEY = random.choice(list(self.ALL_TRACK_WAYPOINTS.keys())[self.TRAINING_IDX[0]:self.TRAINING_IDX[1] + 1])
                    else:
                        self.CURRENT_TRACK_KEY = random.choice(list(self.ALL_TRACK_WAYPOINTS.keys())[:self.EVAL_TRACK_BEGIN_IDX])

            
            self.TRACK_WAYPOINTS = self.ALL_TRACK_WAYPOINTS[self.CURRENT_TRACK_KEY]

        # start at beginning of track when evaluating
        if self.IS_EVAL:
            car_x, car_y, car_yaw, index = self.TRACK_WAYPOINTS[10]
        # start the car randomly along the track
        else:
            car_x, car_y, car_yaw, index = random.choice(self.TRACK_WAYPOINTS)

        # Update goal pointer to reflect starting position
        self.START_WAYPOINT_INDEX = index
        x,y,_,_ = self.TRACK_WAYPOINTS[self.START_WAYPOINT_INDEX+1 if self.START_WAYPOINT_INDEX+1 < len(self.TRACK_WAYPOINTS) else 0]# point toward next goal
        self.goal_position = [x,y]

        self.call_reset_service(car_x=car_x, car_y=car_y, car_Y=car_yaw, goal_x=x, goal_y=y, car_name=self.NAME)

        # Get initial observation
        self.call_step(pause=False)
        state, full_state , _ = self.get_observation()
        self.FULL_CURRENT_STATE = full_state
        self.call_step(pause=True)

        info = {}

        # get track progress related info
        # set new track model if its multi track
        if self.IS_MULTI_TRACK:
            self.TRACK_MODEL = self.ALL_TRACK_MODELS[self.CURRENT_TRACK_KEY]
        self.PREV_T = self.TRACK_MODEL.get_closest_point_on_spline(full_state[:2], t_only=True)

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
        self.step_counter += 1
        
        full_state = self.FULL_CURRENT_STATE

        self.call_step(pause=False)

        lin_vel, steering_angle = action
        self.set_velocity(lin_vel, steering_angle)
        self.sleep()
        
        next_state, full_next_state, raw_lidar_range = self.get_observation()
        self.call_step(pause=True)

        self.FULL_CURRENT_STATE = full_next_state
        
        if not self.PREV_T:
            self.PREV_T = self.TRACK_MODEL.get_closest_point_on_spline(full_state[:2], t_only=True)

        t2 = self.TRACK_MODEL.get_closest_point_on_spline(full_next_state[:2], t_only=True)
        self.STEP_PROGRESS = self.TRACK_MODEL.get_distance_along_track_parametric(self.PREV_T, t2, approximate=True)
        self.center_line_offset = self.TRACK_MODEL.get_distance_to_spline_point(t2, full_next_state[:2])

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
                self.step_counter >= self.MAX_STEPS
            case 'progressive':
                return self.PROGRESS_NOT_MET_CNT >= 5 or \
                self.step_counter >= self.MAX_STEPS
            case _:
                raise Exception("Unknown truncate condition for reward function.")


    def get_observation(self):

        # Get Position and Orientation of F1tenth
        odom, lidar = self.get_data()
        odom = process_odom(odom)
        
        num_points = self.LIDAR_POINTS
        
        # init state
        state = {}
        
        # Add odom data
        match (self.ODOM_OBSERVATION_MODE):
            case 'no_position':
                state["vector"] = odom[2:]
            case 'lidar_only':
                state["vector"] = odom[-2:] 
            case _:
                state["vector"] = odom 
                
        match self.LIDAR_PROCESSING:
            case 'pretrained_ae':
                processed_lidar_range = process_ae_lidar(lidar, self.AE_LIDAR_MODEL, is_latent_only=True)
                visualized_range = reconstruct_ae_latent(lidar, self.AE_LIDAR_MODEL, processed_lidar_range)
                #TODO: get rid of hard coded lidar points num
                scan = create_lidar_msg(lidar, num_points, visualized_range)
            case 'ae':
                lidar_data = np.array(lidar.ranges)
                lidar_data = np.nan_to_num(lidar_data, posinf=-5)
                if not self.IS_EVAL:
                    sampled_data = scipy.signal.resample(lidar_data, 512)
                    self.train_autoencoder(sampled_data)
                # Reduce lidar points to 10 for the message
                processed_lidar_range = process_ae_lidar(lidar, self.AE_LIDAR_MODEL, is_latent_only=True)
                scan = create_lidar_msg(lidar, num_points, processed_lidar_range)
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

        if self.LIDAR_PROCESSING == 'ae':
            state["lidar"] = lidar_data.tolist()
            full_state = odom + lidar_data.tolist()
        else:
            raise Exception(f"Current state configuration can only work with 'ae'")
        
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
                    close_to_wall_penalize_factor = 1 / (1 + np.exp(50 * (dist_to_wall - 0.3))) #y=\frac{1}{1+e^{35\left(x-0.5\right)}}
                    reward -= reward * close_to_wall_penalize_factor * weight
                    reward_info.update({"dist_to_wall":["avg",dist_to_wall]})
                    print(f"--- Wall proximity penalty factor: {weight} * {close_to_wall_penalize_factor}")   
                case 'turn':
                    angular_vel_diff = abs(state[7] - next_state[7])
                    turning_penalty_factor = 1 - (1 / (1 + np.exp(15 * (angular_vel_diff - 0.5)))) #y=1-\frac{1}{1+e^{15\left(x-0.3\right)}}
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
        print(f"Position {next_state[:2]} --> {goal_position}")
        
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

    ##########################################################################################
    ########################## Utility Functions #############################################
    ##########################################################################################
    

    def increment_stage(self):
        if not self.IS_STAGED_TRAINING:
            return
        
        if self.CURRENT_TRAINING_STAGE < len(self.TRAINING_STAGES) - 1:
            self.CURRENT_TRAINING_STAGE += 1
            self.TRAINING_IDX = self.TRAINING_STAGES[self.CURRENT_TRAINING_STAGE][0]
            self.EVAL_IDX = self.TRAINING_STAGES[self.CURRENT_TRAINING_STAGE][1]
            self.get_logger().info(f"Incremented to training stage {self.CURRENT_TRAINING_STAGE}. Training indices: {self.TRAINING_IDX}, Evaluation indices: {self.EVAL_IDX}")
        else:
            self.get_logger().info("Already at the last training stage. No increment performed.")

    def train_autoencoder(self, lidar_data):
        """
        Train the autoencoder using the processed latent representation and reconstructed range.
        """
        self.AE_LIDAR_MODEL.train()
        
        latent_tensor = torch.tensor(lidar_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        reconstructed_range = self.AE_LIDAR_MODEL(latent_tensor)
        loss = self.AE_LOSS_FUNCTION(reconstructed_range, latent_tensor)
        
        self.AE_OPTIMIZER.zero_grad()
        loss.backward()
        self.AE_OPTIMIZER.step()
        print(f"Autoencoder Loss: {loss.item()}")
        
    def set_ae(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        print("Environment set with encoder and decoder.")

