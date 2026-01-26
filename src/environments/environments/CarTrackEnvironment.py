import math
import numpy as np
import random
from environments.F1tenthEnvironment import F1tenthEnvironment
from .util import process_ae_lidar, process_odom, avg_lidar, create_lidar_msg, reconstruct_ae_latent, has_collided, has_flipped_over, get_training_stages
from typing import Literal, List, Tuple
import torch
import scipy
import time

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
        super().__init__('car_track', car_name, reward_range, max_steps, collision_range, step_length, 10, track, observation_mode)

        #####################################################################################################################
        # Reward configuration -----------------------------------------
        self.BASE_REWARD_FUNCTION:Literal["goal_hitting", "progressive"] = 'progressive'
        self.EXTRA_REWARD_TERMS:List[Literal['penalize_turn']] = []
        self.REWARD_MODIFIERS:List[Tuple[Literal['turn','wall_proximity'],float]] = [('turn', 0.3), ('wall_proximity', 0.7)] 
        if track == 'narrow_multi_track':
            self.MULTI_TRACK_TRAIN_EVAL_SPLIT = (12/15)
        else:
            self.MULTI_TRACK_TRAIN_EVAL_SPLIT = 0.5

        #####################################################################################################################   
        # Staging configuration ----------------------------------------
        self.IS_STAGED_TRAINING = is_staged_training
        if self.IS_STAGED_TRAINING:
            self.TRAINING_STAGES = get_training_stages(track)
            self.TRAINING_IDX = self.TRAINING_STAGES[self.CURRENT_TRAINING_STAGE][0]
            self.EVAL_IDX = self.TRAINING_STAGES[self.CURRENT_TRAINING_STAGE][1]
        self.CURRENT_TRAINING_STAGE = 0
        #####################################################################################################################
        # Environment configuration -------------------------------------
        if self.BASE_REWARD_FUNCTION == 'progressive':
            self.PROGRESS_NOT_MET_CNT = 0
        self.STEPS_SINCE_LAST_GOAL = 0

        if self.IS_MULTI_TRACK:
            if self.IS_STAGED_TRAINING:
                self.CURR_TRACK = list(self.ALL_TRACK_WAYPOINTS.keys())[self.TRAINING_IDX[0]]
                self.EVAL_TRACK_BEGIN_IDX = None
                self.get_logger().info(f"Track '{track}', {self.TRAINING_IDX} training, {self.EVAL_IDX} evaluation")
            else:
                self.EVAL_TRACK_BEGIN_IDX = int(len(self.ALL_TRACK_WAYPOINTS)*self.MULTI_TRACK_TRAIN_EVAL_SPLIT)
            self.EVAL_TRACK_IDX = 0

        self.get_logger().info('Environment Setup Complete')

        ##################################################################################################################### 

    def reset(self):
        self.STEP_COUNTER = 0
        self.STEPS_SINCE_LAST_GOAL = 0
        self.GOALS_REACHED = 0
        self.set_velocity(0, 0)

        if self.IS_MULTI_TRACK:
            if self.EVAL_TRACK_BEGIN_IDX is not None and self.EVAL_TRACK_BEGIN_IDX >= len(self.ALL_TRACK_WAYPOINTS):
                if self.IS_EVAL:
                    all_track_keys = list(self.ALL_TRACK_WAYPOINTS.keys())
                    self.CURR_TRACK = all_track_keys[self.EVAL_TRACK_IDX]
                    self.EVAL_TRACK_IDX += 1
                    self.EVAL_TRACK_IDX = self.EVAL_TRACK_IDX % len(all_track_keys)
                else:
                    self.CURR_TRACK = random.choice(list(self.ALL_TRACK_WAYPOINTS.keys()))
            else:
                if self.IS_EVAL:
                    if self.IS_STAGED_TRAINING:
                        eval_track_key_list = list(self.ALL_TRACK_WAYPOINTS.keys())[self.EVAL_IDX[0]:self.EVAL_IDX[1] + 1]
                    else:
                        eval_track_key_list = list(self.ALL_TRACK_WAYPOINTS.keys())[self.EVAL_TRACK_BEGIN_IDX:]
                    self.CURR_TRACK = eval_track_key_list[self.EVAL_TRACK_IDX]
                    self.EVAL_TRACK_IDX += 1
                    self.EVAL_TRACK_IDX = self.EVAL_TRACK_IDX % len(eval_track_key_list)
                else:
                    if self.IS_STAGED_TRAINING:
                        self.CURR_TRACK = random.choice(list(self.ALL_TRACK_WAYPOINTS.keys())[self.TRAINING_IDX[0]:self.TRAINING_IDX[1] + 1])
                    else:
                        self.CURR_TRACK = random.choice(list(self.ALL_TRACK_WAYPOINTS.keys())[:self.EVAL_TRACK_BEGIN_IDX])
            self.CURR_WAYPOINTS = self.ALL_TRACK_WAYPOINTS[self.CURR_TRACK]

        if self.IS_EVAL:
            car_x, car_y, car_yaw, index = self.CURR_WAYPOINTS[10]
        else:
            car_x, car_y, car_yaw, index = random.choice(self.CURR_WAYPOINTS)

        self.SPAWN_INDEX = index
        x,y,_,_ = self.CURR_WAYPOINTS[self.SPAWN_INDEX+1 if self.SPAWN_INDEX+1 < len(self.CURR_WAYPOINTS) else 0]# point toward next goal
        self.GOAL_POSITION = [x,y]
        self.call_reset_service(car_x=car_x, car_y=car_y, car_Y=car_yaw, goal_x=x, goal_y=y, car_name=self.NAME)

        self.call_step(pause=False)
        state, full_state , _ = self.get_observation()
        self.CURR_STATE = full_state
        self.call_step(pause=True)

        if self.IS_MULTI_TRACK:
            self.CURR_TRACK_MODEL = self.ALL_TRACK_MODELS[self.CURR_TRACK]
        self.PREV_CLOSEST_POINT = self.CURR_TRACK_MODEL.get_closest_point_on_spline(full_state[:2], t_only=True)

        if self.BASE_REWARD_FUNCTION == 'progressive':
            self.PROGRESS_NOT_MET_CNT = 0
        info = {}
        return state, info
    
    def start_eval(self):
        self.EVAL_TRACK_IDX = 0
        self.IS_EVAL = True
    
    def stop_eval(self):
        self.IS_EVAL = False

    def step(self, action):
        self.STEP_COUNTER += 1
        full_state = self.CURR_STATE
        self.call_step(pause=False)

        lin_vel, steering_angle = action
        if not self.IS_EVAL:
            time.sleep(0.074)  # 74ms delay to simulate delay between nn output from previous step and action now
        self.set_velocity(lin_vel, steering_angle)
        # action delay based on training stage
        if self.CURRENT_TRAINING_STAGE == 0:
            action_delay = 0 
            print(f"No action delay  stage: {self.CURRENT_TRAINING_STAGE}")
        elif self.CURRENT_TRAINING_STAGE == 1:
            action_delay = 0.010
            print(f"10ms action delay  stage: {self.CURRENT_TRAINING_STAGE}")
        elif self.CURRENT_TRAINING_STAGE == 2:
            action_delay = 0.030
            print(f"30ms action delay  stage: {self.CURRENT_TRAINING_STAGE}")
        elif self.CURRENT_TRAINING_STAGE >= 3:
            action_delay = np.random.uniform(0.073, 0.075)  # 74ms ± 1ms
            print(f"{action_delay*1000:.1f}ms action delay  stage: {self.CURRENT_TRAINING_STAGE}")
        time.sleep(action_delay)
        next_state, full_next_state, raw_lidar_range = self.get_observation()
        self.call_step(pause=True)
        # simulate sensor-to-NN delay
        # if not self.is_evaluating:
        sensor_delay = np.random.uniform(0.0017, 0.0037) # 2.7ms ± 1ms needs be remeasured
        time.sleep(sensor_delay)

        self.CURR_STATE = full_next_state
        if not self.PREV_CLOSEST_POINT:
            self.PREV_CLOSEST_POINT = self.CURR_TRACK_MODEL.get_closest_point_on_spline(full_state[:2], t_only=True)

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
            new_x, new_y, _, _ = self.CURR_WAYPOINTS[(self.SPAWN_INDEX + self.GOALS_REACHED) % len(self.CURR_WAYPOINTS)]
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
            new_x, new_y, _, _ = self.CURR_WAYPOINTS[(self.SPAWN_INDEX + self.GOALS_REACHED) % len(self.CURR_WAYPOINTS)]
            self.GOAL_POSITION = [new_x, new_y]
            self.update_goal_service(new_x, new_y)
            self.STEPS_SINCE_LAST_GOAL = 0

        if self.PROGRESS_NOT_MET_CNT >= 5:
            reward -= 2
        if has_collided(raw_range, self.COLLISION_RANGE) or has_flipped_over(next_state[2:6]):
            reward -= 2.5

        info = {}
        return reward, info
    
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
        self.AE_LIDAR_MODEL.train()
        latent_tensor = torch.tensor(lidar_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        reconstructed_range = self.AE_LIDAR_MODEL(latent_tensor)
        loss = self.AE_LOSS_FUNCTION(reconstructed_range, latent_tensor)     
        self.AE_OPTIMIZER.zero_grad()
        loss.backward()
        self.AE_OPTIMIZER.step()
        print(f"Autoencoder Loss: {loss.item()}")
        
    def set_ae(self, encoder, decoder):
        self.ENCODER = encoder
        self.DECODER = decoder
        print("Environment set with encoder and decoder.")