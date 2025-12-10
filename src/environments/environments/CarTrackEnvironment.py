import math
import rclpy
import numpy as np
import random
from environments.F1tenthEnvironment import F1tenthEnvironment
from .util import get_track_math_defs, process_ae_lidar, process_odom, avg_lidar, create_lidar_msg, get_all_goals_and_waypoints_in_multi_tracks, ackermann_to_twist, reconstruct_ae_latent, has_collided, has_flipped_over
from .util_track_progress import TrackMathDef
from .waypoints import waypoints
from std_srvs.srv import SetBool
from typing import Literal, List, Optional, Tuple
import torch
from datetime import datetime
import yaml

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
        self.REWARD_MODIFIERS:List[Tuple[Literal['turn','wall_proximity'],float]] = [('turn', 0.5), ('wall_proximity', 0.5)] # [ (penalize_turn", 0.3), (penalize_wall_proximity, 0.7) ]

        # Observation configuration
        self.LIDAR_PROCESSING:Literal["avg","pretrained_ae", "raw"] = 'avg'
        self.LIDAR_POINTS = 10 #682
        self.EXTRA_OBSERVATIONS:List[Literal['prev_ang_vel']] = []

        # Evaluation settings
        self.MULTI_TRACK_TRAIN_EVAL_SPLIT=0.5 

        #optional stuff
        pretrained_ae_path = "/home/anyone/autonomous_f1tenth/lidar_ae_ftg_rand.pt" #"/ws/lidar_ae_ftg_rand.pt"

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
            self.PROGRESS_NOT_MET_CNT = 0
        self.STEPS_SINCE_LAST_GOAL = 0

        self.goals_reached = 0
        self.start_waypoint_index = 0
        self.steps_since_last_goal = 0
        self.full_current_state = None

        if not self.is_multi_track:
            if "test_track" in track:
                track_key = track[0:-4] # "test_track_xx_xxx" -> "test_track_xx", here due to test_track's different width variants having the same waypoints.
            else:
                track_key = track

            self.track_waypoints = waypoints[track_key]
            self.track_model = TrackMathDef(np.array(self.track_waypoints)[:,:2])
            
        else:
            _, self.all_track_waypoints = get_all_goals_and_waypoints_in_multi_tracks(track)
            self.current_track_key = list(self.all_track_waypoints.keys())[0]

            # set current track waypoints
            self.track_waypoints = self.all_track_waypoints[self.current_track_key]

            # set track models
            self.all_track_models = get_track_math_defs(self.all_track_waypoints)
            self.track_model = self.all_track_models[self.current_track_key]


        # Evaluation related setup ---------------------------------------------------
        self.is_evaluating = False

        if self.is_multi_track:
            # define from which track in the track lists to be used for eval only
            self.eval_track_begin_idx = int(len(self.all_track_waypoints)*self.MULTI_TRACK_TRAIN_EVAL_SPLIT)
            # idx used to loop through eval tracks sequentially
            self.eval_track_idx = 0

        self.get_logger().info('Environment Setup Complete')

        ##################################################################################################################### 

    def reset(self):
        self.STEP_COUNTER = 0
        self.STEPS_SINCE_LAST_GOAL = 0
        self.GOALS_REACHED = 0
        self.set_velocity(0, 0)
        
        if self.is_multi_track:
            # Evaluating: loop through eval tracks sequentially
            if self.is_evaluating:
                eval_track_key_list = list(self.all_track_waypoints.keys())[self.eval_track_begin_idx:]
                self.current_track_key = eval_track_key_list[self.eval_track_idx]
                self.eval_track_idx += 1
                self.eval_track_idx = self.eval_track_idx % len(eval_track_key_list)

            # Training: choose a random track that is not used for evaluation
            else:
                self.current_track_key = random.choice(list(self.all_track_waypoints.keys())[:self.eval_track_begin_idx])
            
            self.track_waypoints = self.all_track_waypoints[self.current_track_key]

        # start at beginning of track when evaluating
        if self.is_evaluating:
            car_x, car_y, car_yaw, index = self.track_waypoints[10]
        # start the car randomly along the track
        else:
            car_x, car_y, car_yaw, index = random.choice(self.track_waypoints)
        
        # Update goal pointer to reflect starting position
        self.start_waypoint_index = index
        x,y,_,_ = self.track_waypoints[self.start_waypoint_index+1 if self.start_waypoint_index+1 < len(self.track_waypoints) else 0]# point toward next goal
        self.goal_position = [x,y]

        self.call_reset_service(car_x=car_x, car_y=car_y, car_Y=car_yaw, goal_x=x, goal_y=y, car_name=self.NAME)

        self.call_step(pause=False)
        state, full_state , _ = self.get_observation()
        self.full_current_state = full_state
        self.call_step(pause=True)

        if self.IS_MULTI_TRACK:
            self.CURR_TRACK_MODEL = self.ALL_TRACK_MODELS[self.CURR_TRACK]
        self.PREV_CLOSEST_POINT = self.CURR_TRACK_MODEL.get_closest_point_on_spline(full_state[:2], t_only=True)

        if self.BASE_REWARD_FUNCTION == 'progressive':
            self.progress_not_met_cnt = 0

        return state, info
    
    def start_eval(self):
        self.EVAL_TRACK_IDX = 0
        self.IS_EVAL = True
    
    def stop_eval(self):
        self.IS_EVAL = False

    def step(self, action):
        self.step_counter += 1
        
        full_state = self.full_current_state

        self.call_step(pause=False)
        
        lin_vel, steering_angle = action
        
        # simulate delay between NN output from previous step and action now
        # if not self.is_evaluating:
        action_delay = np.random.uniform(0.064, 0.084)  # 74ms ± 10ms needs be remeasured
        time.sleep(action_delay)
        
        # take action and wait
        if not self.is_evaluating:
            time.sleep(0.074)  # 74ms delay to simulate delay between nn output from previous step and action now

        # take action and wait
        if is_training:
            lin_vel, steering_angle = super().randomise_action(action)
        else:
            lin_vel, steering_angle = action

        self.set_velocity(lin_vel, steering_angle)
        
        # action delay based on training stage
        if self.current_training_stage == 0:
            action_delay = 0 
            print(f"No action delay  stage: {self.current_training_stage}")
        elif self.current_training_stage == 1:
            action_delay = 0.010
            print(f"10ms action delay  stage: {self.current_training_stage}")
        elif self.current_training_stage == 2:
            action_delay = 0.030
            print(f"30ms action delay  stage: {self.current_training_stage}")
        elif self.current_training_stage >= 3:
            action_delay = np.random.uniform(0.073, 0.075)  # 74ms ± 1ms
            print(f"{action_delay*1000:.1f}ms action delay  stage: {self.current_training_stage}")
            
        time.sleep(action_delay)
        
        self.set_velocity(lin_vel, steering_angle)
        
        self.sleep()
        next_state, full_next_state, raw_lidar_range = self.get_observation()
        
        # simulate sensor-to-NN delay
        # if not self.is_evaluating:
        sensor_delay = np.random.uniform(0.0017, 0.0037) # 2.7ms ± 1ms needs be remeasured
        time.sleep(sensor_delay)
            
        self.call_step(pause=True)

        self.full_current_state = full_next_state
        
        # calculate progress along track
        if not self.prev_t:
            self.prev_t = self.track_model.get_closest_point_on_spline(full_state[:2], t_only=True)

        t2 = self.CURR_TRACK_MODEL.get_closest_point_on_spline(full_next_state[:2], t_only=True)
        self.STEP_PROGRESS = self.CURR_TRACK_MODEL.get_distance_along_track_parametric(self.PREV_CLOSEST_POINT, t2, approximate=True)
        self.center_line_offset = self.CURR_TRACK_MODEL.get_distance_to_spline_point(t2, full_next_state[:2])
        self.PREV_CLOSEST_POINT = t2

        if abs(self.STEP_PROGRESS) > (full_next_state[6]/10*3): 
            self.STEP_PROGRESS = full_next_state[6]/10*0.8 

        reward, reward_info = self.compute_reward(full_state, full_next_state, raw_lidar_range)
        terminated = self.is_terminated(full_next_state, raw_lidar_range)
        truncated = self.is_truncated()

        info = {
            'linear_velocity':["avg", full_next_state[6]],
            'angular_velocity_diff':["avg", abs(full_next_state[7] - full_state[7])],
            'traveled distance': ['sum', self.STEP_PROGRESS]
        }
        info.update(reward_info)

        if self.is_evaluating and (terminated or truncated):
            self.eval_track_idx

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
        odom, lidar = self.get_data()
        odom = process_odom(odom)
        num_points = self.LIDAR_POINTS
        state = {}
        
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
                scan = create_lidar_msg(lidar, num_points, visualized_range)
            case 'ae':
                lidar_data = np.array(lidar.ranges)
                lidar_data = np.nan_to_num(lidar_data, posinf=-5)
                if not self.IS_EVAL:
                    sampled_data = scipy.signal.resample(lidar_data, 512)
                    self.train_autoencoder(sampled_data)
                processed_lidar_range = process_ae_lidar(lidar, self.AE_LIDAR_MODEL, is_latent_only=True)
                scan = create_lidar_msg(lidar, num_points, processed_lidar_range)
            case 'avg':
                processed_lidar_range = avg_lidar(lidar, num_points)
                visualized_range = processed_lidar_range
                scan = create_lidar_msg(lidar, num_points, visualized_range)
            case 'uneven_median':
                processed_lidar_range = uneven_median_lidar(lidar, num_points)
                visualized_range = processed_lidar_range
                scan = create_lidar_msg(lidar, num_points, visualized_range)
            case 'raw':
                processed_lidar_range = np.array(lidar.ranges.tolist())
                processed_lidar_range = np.nan_to_num(processed_lidar_range, posinf=-5, nan=-1, neginf=-5).tolist()  
                visualized_range = processed_lidar_range
                scan = create_lidar_msg(lidar, num_points, visualized_range)
        
        self.PROCESSED_PUBLISHER.publish(scan)
        if self.LIDAR_PROCESSING == 'ae':
            state["lidar"] = lidar_data.tolist()
            full_state = odom + lidar_data.tolist()
        else:
            full_state = odom + processed_lidar_range
        return state, full_state, lidar.ranges
    
    def compute_reward(self, state, next_state, raw_lidar_range):
        reward = 0
        reward_info = {}

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

        for term in self.EXTRA_REWARD_TERMS:
            match term:
                case 'penalize_turn':
                    turn_penalty = abs(state[7] - next_state[7])*0.12
                    reward -= turn_penalty
                    reward_info.update({"turn_penalty":("avg",turn_penalty)})

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

    def calculate_goal_hitting_reward(self, state, next_state, raw_range):
        reward = 0
        goal_position = self.GOAL_POSITION
        current_distance = math.dist(goal_position, next_state[:2])
        previous_distance = math.dist(goal_position, state[:2])
        reward += previous_distance - current_distance
        self.STEPS_SINCE_LAST_GOAL += 1

        if current_distance < self.REWARD_RANGE:
            reward += 2
            self.GOALS_REACHED += 1
            new_x, new_y, _, _ = self.TRACK_WAYPOINTS[(self.SPAWN_INDEX + self.GOALS_REACHED) % len(self.TRACK_WAYPOINTS)]
            self.GOAL_POSITION = [new_x, new_y]
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
        goal_position = self.GOAL_POSITION
        current_distance = math.dist(goal_position, next_state[:2])
        
        if self.STEP_PROGRESS < 0.02:
            self.PROGRESS_NOT_MET_CNT += 1
        else:
            self.PROGRESS_NOT_MET_CNT = 0
        reward += self.STEP_PROGRESS
        self.STEPS_SINCE_LAST_GOAL += 1

        if current_distance < self.REWARD_RANGE:
            self.GOALS_REACHED += 1
            new_x, new_y, _, _ = self.TRACK_WAYPOINTS[(self.SPAWN_INDEX + self.GOALS_REACHED) % len(self.TRACK_WAYPOINTS)]
            self.GOAL_POSITION = [new_x, new_y]
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