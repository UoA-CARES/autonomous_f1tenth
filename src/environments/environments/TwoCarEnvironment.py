import math
import rclpy
import numpy as np
from rclpy import Future
import random
from environment_interfaces.srv import Reset
from environments.F1tenthEnvironment import F1tenthEnvironment
from .util import has_collided, has_flipped_over, findOccurrences
from .util import get_track_math_defs, process_ae_lidar, process_odom, avg_lidar, create_lidar_msg, get_all_goals_and_waypoints_in_multi_tracks, twist_to_ackermann, reconstruct_ae_latent, lateral_translation
from .util_track_progress import TrackMathDef
from .waypoints import waypoints
from std_srvs.srv import SetBool
from typing import Literal, List, Optional, Tuple
from std_msgs.msg import String
import torch
from datetime import datetime
from message_filters import Subscriber, ApproximateTimeSynchronizer
from nav_msgs.msg import Odometry
import yaml
import time


class TwoCarEnvironment(F1tenthEnvironment):

    MULTI_TRACK_TRAIN_EVAL_SPLIT = 5/6
    LIDAR_POINTS = 10
    REWARD_MODIFIERS:List[Tuple[Literal['turn','wall_proximity', 'racing'],float]] = [('turn', 0.3), ('wall_proximity', 0.7), ('racing', 1)]
    LIDAR_PROCESSING:Literal["avg","pretrained_ae", "raw"] = 'avg' 

    def __init__(self, 
                 car_name, 
                 reward_range=0.5, 
                 max_steps=200, 
                 collision_range=0.2, 
                 step_length=0.5, 
                 track='track_1',
                 observation_mode='lidar_only',
                 config_path='/home/anyone/autonomous_f1tenth/src/environments/config/config.yaml',
                 ):
        super().__init__('two_car', car_name, reward_range, max_steps, collision_range, step_length, track, observation_mode)

        #####################################################################################################################
        # Read in params from init and config
        
        # Init params
        self.TRACK = track
        self.OBSERVATION_MODE = observation_mode

        #####################################################################################################################
        # Initialise other vars

        # Track progress utilities
        self.PREV_CLOSEST_POINT = None
        TwoCarEnvironment.ALL_TRACK_MODELS = None
        TwoCarEnvironment.ALL_TRACK_WAYPOINTS = None
        self.CURR_TRACK_MODEL = None
        self.CURR_TRACK = None
        self.CURR_WAYPOINTS = None
        self.STEP_PROGRESS = 0
        self.PROGRESS_NOT_MET_COUNTER = 0

        # Distance covered
        self.EP_PROGRESS1 = 0
        self.EP_PROGRESS2 = 0
        self.LAST_POS1 = [0, 0]
        self.LAST_POS2 = [0, 0]

        # Reset client
        self.GOALS_REACHED = 0
        self.SPAWN_INDEX = 0
        self.STEPS_WITHOUT_GOAL = 0
        self.CURR_STATE = None #Can reformat this var

        # Eval utilities
        self.IS_EVAL = False
        self.EVAL_TRACKS_IDX = 0
        self.CURR_EVAL_IDX = 0

        #####################################################################################################################

        # AE
        if TwoCarEnvironment.LIDAR_PROCESSING == 'pretrained_ae':
            from .autoencoders.lidar_autoencoder import LidarConvAE
            self.AE_LIDAR = LidarConvAE()
            self.AE_LIDAR.load_state_dict(torch.load("/home/anyone/autonomous_f1tenth/lidar_ae_ftg_rand.pt"))
            self.AE_LIDAR.eval()

        # Observation Size
        match self.OBSERVATION_MODE:
            case 'lidar_only':
                odom_size = 2
            case 'no_position':
                odom_size = 6
            case _:
                odom_size = 10
        TwoCarEnvironment.OBSERVATION_SIZE = odom_size + TwoCarEnvironment.LIDAR_POINTS
        
        # Track info
        TwoCarEnvironment.IS_MULTI_TRACK = 'multi_track' in self.TRACK
        if TwoCarEnvironment.IS_MULTI_TRACK:
            # Get all track infos
            _, TwoCarEnvironment.ALL_TRACK_WAYPOINTS = get_all_goals_and_waypoints_in_multi_tracks(self.TRACK)
            TwoCarEnvironment.ALL_TRACK_MODELS = get_track_math_defs(TwoCarEnvironment.ALL_TRACK_WAYPOINTS)
            
            # Get current track infos (should start empty?)
            self.CURR_TRACK = list(TwoCarEnvironment.ALL_TRACK_WAYPOINTS.keys())[0] # Should it always be the first one? Should it be initialized empty?
            self.CURR_WAYPOINTS = TwoCarEnvironment.ALL_TRACK_WAYPOINTS[self.CURR_TRACK]
            self.CURR_TRACK_MODEL = TwoCarEnvironment.ALL_TRACK_MODELS[self.CURR_TRACK]

            # Set eval track indexes
            self.EVAL_TRACKS_IDX = int(len(TwoCarEnvironment.ALL_TRACK_WAYPOINTS)*TwoCarEnvironment.MULTI_TRACK_TRAIN_EVAL_SPLIT)   
        else:
            if "test_track" in self.TRACK:
                track_key = self.TRACK[0:-4] # "test_track_xx_xxx" -> "test_track_xx", here due to test_track's different width variants having the same waypoints.
            else:
                track_key = self.TRACK

            self.CURR_WAYPOINTS = waypoints[track_key] #from waypoints.py
            self.CURR_TRACK_MODEL = TrackMathDef(np.array(self.CURR_WAYPOINTS)[:,:2])
            
        #####################################################################################################################
        # Odom subscribers
        self.ODOM_SUB_1 = Subscriber(
            self,
            Odometry,
            f'/f1tenth/odometry',
        )

        self.ODOM_SUB_2 = Subscriber(
            self,
            Odometry,
            f'/f2tenth/odometry',
        )

        self.ODOM_MESSAGE_FILTER = ApproximateTimeSynchronizer(
            [self.ODOM_SUB_1, self.ODOM_SUB_2],
            10,
            0.1,
        )

        self.ODOM_MESSAGE_FILTER.registerCallback(self.odom_message_filter_callback)
        self.ODOM_OBSERVATION_FUTURE = Future()

        #####################################################################################################################

        # Publish and subscribe to status topic

        self.STATUS_PUB = self.create_publisher(
            String,
            '/status',
            10
        )

        self.STATUS_SUB = self.create_subscription(
            String,
            '/status',
            self.status_callback,
            10)
        
        self.STATUS_LOCK_PUB = self.create_publisher(
            String,
            '/status_lock',
            10
        )

        self.STATUS_LOCK_SUB = self.create_subscription(
            String,
            '/status_lock',
            self.status_lock_callback,
            10)
        
        self.STATUS = 'r_f1tenth'

        self.STATUS_OBSERVATION_FUTURE = Future()
        self.STATUS_LOCK = 'off'

        self.get_logger().info('Environment Setup Complete')

        #####################################################################################################################

#    ____ _        _    ____ ____    _____ _   _ _   _  ____ _____ ___ ___  _   _ ____  
#   / ___| |      / \  / ___/ ___|  |  ___| | | | \ | |/ ___|_   _|_ _/ _ \| \ | / ___| 
#  | |   | |     / _ \ \___ \___ \  | |_  | | | |  \| | |     | |  | | | | |  \| \___ \ 
#  | |___| |___ / ___ \ ___) |__) | |  _| | |_| | |\  | |___  | |  | | |_| | |\  |___) |
#   \____|_____/_/   \_\____/____/  |_|    \___/|_| \_|\____| |_| |___\___/|_| \_|____/ 

    def odom_message_filter_callback(self, odom1: Odometry, odom2: Odometry):
        self.ODOM_OBSERVATION_FUTURE.set_result({'odom1': odom1, 'odom2': odom2})                                                                            
    
    def randomize_yaw(self, yaw, percentage=0.5):
        factor = 1 + random.uniform(-percentage, percentage)
        return yaw + factor
    
    def reset(self):
        self.STEP_COUNTER = 0

        self.STEPS_WITHOUT_GOAL = 0
        self.GOALS_REACHED = 0

        self.set_velocity(0, 0)
        if self.NAME == 'f2tenth':
            while ('respawn' not in self.STATUS):
                rclpy.spin_until_future_complete(self, self.STATUS_OBSERVATION_FUTURE, timeout_sec=10)
                if (self.STATUS_OBSERVATION_FUTURE.result()) == None:
                    state, full_state , _ = self.get_observation()

                    self.CURR_STATE = full_state
                    info = {}
                    return state, info
                self.STATUS_OBSERVATION_FUTURE = Future()
            track, goal, spawn = self.parse_status(self.STATUS)
            self.CURR_TRACK = track
            self.GOAL_POS = [goal[0], goal[1]]
            self.SPAWN_INDEX = spawn
            self.CURR_WAYPOINTS = TwoCarEnvironment.ALL_TRACK_WAYPOINTS[self.CURR_TRACK]
            if self.IS_EVAL:
                eval_track_key_list = list(TwoCarEnvironment.ALL_TRACK_WAYPOINTS.keys())[self.EVAL_TRACKS_IDX:]
                self.CURR_EVAL_IDX += 1
                self.CURR_EVAL_IDX = self.CURR_EVAL_IDX % len(eval_track_key_list)
            self.publish_status('ready')
        else:
            self.car_spawn()
            i = 0
            while(self.STATUS != 'ready'):
                rclpy.spin_until_future_complete(self, self.STATUS_OBSERVATION_FUTURE, timeout_sec=10)
                if (self.STATUS_OBSERVATION_FUTURE.result() == None):
                    break
                self.STATUS_OBSERVATION_FUTURE = Future()
                
        

        # Get initial observation
        self.call_step(pause=False)
        state, full_state , _ = self.get_observation()

        self.CURR_STATE = full_state

        self.call_step(pause=True)

        info = {}

        # get track progress related info
        # set new track model if its multi track

        if TwoCarEnvironment.IS_MULTI_TRACK:
            self.CURR_TRACK_MODEL = TwoCarEnvironment.ALL_TRACK_MODELS[self.CURR_TRACK]
        self.PREV_CLOSEST_POINT = self.CURR_TRACK_MODEL.get_closest_point_on_spline(full_state[:2], t_only=True)
        self.EP_PROGRESS1 = 0
        self.EP_PROGRESS2 = 0
        self.LAST_POS1 = [None, None]
        self.LAST_POS2 = [None, None]
        # reward function specific resets
        self.PROGRESS_NOT_MET_COUNTER = 0


        self.publish_status('')
        self.change_status_lock('off')
        return state, info
    
    def car_spawn(self):
        if TwoCarEnvironment.IS_MULTI_TRACK:
            # Evaluating: loop through eval tracks sequentially
            if self.IS_EVAL:
                eval_track_key_list = list(TwoCarEnvironment.ALL_TRACK_WAYPOINTS.keys())[self.EVAL_TRACKS_IDX:]
                self.CURR_TRACK = eval_track_key_list[self.CURR_EVAL_IDX]
                self.CURR_EVAL_IDX += 1
                self.CURR_EVAL_IDX = self.CURR_EVAL_IDX % len(eval_track_key_list)

            # Training: choose a random track that is not used for evaluation
            else:
                self.CURR_TRACK = random.choice(list(TwoCarEnvironment.ALL_TRACK_WAYPOINTS.keys())[:self.EVAL_TRACKS_IDX])
            
            self.CURR_WAYPOINTS = TwoCarEnvironment.ALL_TRACK_WAYPOINTS[self.CURR_TRACK]
        else:
            self.CURR_TRACK = self.TRACK

        if (self.CURR_TRACK[-3:]).isdigit():
            width = int(self.CURR_TRACK[-3:])
        else: 
            width = 300

        car_x, car_y, car_yaw, index = random.choice(self.CURR_WAYPOINTS)

        # Update goal pointer to reflect starting position
        order = random.choice([1, 2])
        translation2 = random.random()

        translation1 = 0.15 + random.random()*0.3
        translation2 = -0.2 - random.random()*1.5

        if width > 200:
            if order == 1:
                car_2_x, car_2_y = lateral_translation((car_x, car_y), car_yaw, translation1) # translation 0.5 prev
                car_x, car_y = lateral_translation((car_x, car_y), car_yaw, translation2) # translation -1.5 prev
            else:
                car_2_x, car_2_y = lateral_translation((car_x, car_y), car_yaw, translation2) # translation 0.5 prev
                car_x, car_y = lateral_translation((car_x, car_y), car_yaw, translation1) # translation -1.5 prev
            car_2_yaw = car_yaw
        else:
            if order == 1:
                car_2_offset = random.randint(8, 16)
                car_2_index = (index + car_2_offset) % len(self.CURR_WAYPOINTS)
                car_2_x, car_2_y, car_2_yaw, _ = self.CURR_WAYPOINTS[car_2_index]
            else:
                car_2_offset = random.randint(8, 16)
                car_2_index = (index - car_2_offset) % len(self.CURR_WAYPOINTS)
                car_2_x, car_2_y, car_2_yaw, _ = self.CURR_WAYPOINTS[car_2_index]

        self.SPAWN_INDEX = index
        x,y,_,_ = self.CURR_WAYPOINTS[self.SPAWN_INDEX+1 if self.SPAWN_INDEX+1 < len(self.CURR_WAYPOINTS) else 0]# point toward next goal
        goal_x, goal_y, _, _ = self.CURR_WAYPOINTS[self.SPAWN_INDEX+3 if self.SPAWN_INDEX+3 < len(self.CURR_WAYPOINTS) else 0]
        self.GOAL_POS = [x,y]


        # Spawn car
        self.call_reset_service(car_x=car_x, car_y=car_y, car_Y=car_yaw, goal_x=goal_x, goal_y=goal_y, car_name='f1tenth')
        self.call_reset_service(car_x=car_2_x, car_y=car_2_y, car_Y=car_2_yaw, goal_x=goal_x, goal_y=goal_y, car_name='f2tenth')

        if self.NAME == 'f1tenth':
            string =  'respawn_' + str(self.CURR_TRACK) + '_' + str(self.GOAL_POS)+ '_' + str(self.SPAWN_INDEX) + ', car1'
        else:
            string =  'respawn_' + str(self.CURR_TRACK) + '_' + str(self.GOAL_POS)+ '_' + str(self.SPAWN_INDEX) + ', car2'
        self.publish_status(string)
    
    def start_eval(self):
        self.CURR_EVAL_IDX = 0
        self.IS_EVAL = True

    def stop_eval(self):
        self.IS_EVAL = False

    def step(self, action):
        self.STEP_COUNTER += 1
        
        # get current state
        full_state = self.CURR_STATE

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
        self.CURR_STATE = full_next_state
        
        # calculate progress along track
        if not self.PREV_CLOSEST_POINT:
            self.PREV_CLOSEST_POINT = self.CURR_TRACK_MODEL.get_closest_point_on_spline(full_state[:2], t_only=True)

        t2 = self.CURR_TRACK_MODEL.get_closest_point_on_spline(full_next_state[:2], t_only=True)
        self.STEP_PROGRESS = self.CURR_TRACK_MODEL.get_distance_along_track_parametric(self.PREV_CLOSEST_POINT, t2, approximate=True)
        self.center_line_offset = self.CURR_TRACK_MODEL.get_distance_to_spline_point(t2, full_next_state[:2])

        self.PREV_CLOSEST_POINT = t2

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
            self.CURR_EVAL_IDX

        if ((terminated or truncated) and self.STATUS_LOCK == 'off'):
            self.change_status_lock('on')
            string = 'r_' + str(self.NAME)
            self.publish_status(string)
            self.STATUS=string
        if ((not truncated) and ('r' in self.STATUS)):
            truncated = True
        return next_state, reward, terminated, truncated, info

    def is_terminated(self, state, ranges):
        return has_collided(ranges, self.COLLISION_RANGE) \
            or has_flipped_over(state[2:6])

    def is_truncated(self):
        return self.PROGRESS_NOT_MET_COUNTER >= 5 or \
        self.STEP_COUNTER >= self.MAX_STEPS



    def get_observation(self):

        # Get Position and Orientation of F1tenth
        odom, lidar = self.get_data()
        odom = process_odom(odom)
        
        num_points = TwoCarEnvironment.LIDAR_POINTS
        
        # init state
        state = []
        
        # Add odom data
        match (self.OBSERVATION_MODE):
            case 'no_position':
                state += odom[2:]
            case 'lidar_only':
                state += odom[-2:] 
            case _:
                state += odom 
        
        # Add lidar data:
        match TwoCarEnvironment.LIDAR_PROCESSING:
            case 'pretrained_ae':
                processed_lidar_range = process_ae_lidar(lidar, self.AE_LIDAR, is_latent_only=True)
                visualized_range = reconstruct_ae_latent(lidar, self.AE_LIDAR, processed_lidar_range)
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

        
        full_state = odom + processed_lidar_range
        
        return state, full_state, lidar.ranges

    def compute_reward(self, state, next_state, raw_lidar_range):
        '''Compute reward based on FULL states: odom + lidar + extra'''
        reward = 0
        reward_info = {}

        odom1, odom2 = self.get_odoms()
        if self.LAST_POS1[0] == None:
            self.LAST_POS1 = odom1[:2]
        if self.LAST_POS2[0] == None:
            self.LAST_POS2 = odom2[:2]
        progression1 = self.CURR_TRACK_MODEL.get_distance_along_track(self.LAST_POS1, odom1[:2])
        progression2 = self.CURR_TRACK_MODEL.get_distance_along_track(self.LAST_POS2, odom2[:2])
        if abs(progression1) < 1:
            self.EP_PROGRESS1 += progression1
        if abs(progression2) < 1:
            self.EP_PROGRESS2 += progression2
        #self.get_logger().info("Episode progression 1: " + str(self.EP_PROGRESS1) + " , episode progression 2: " + str(self.EP_PROGRESS2))
        
        if self.NAME == 'f1tenth':
            base_reward, base_reward_info = self.calculate_total_progress_reward(self.EP_PROGRESS1, progression1, next_state, raw_lidar_range)
        else:
            base_reward, base_reward_info = self.calculate_total_progress_reward(self.EP_PROGRESS2, progression2, next_state, raw_lidar_range)

        # calculate base reward
        #base_reward, base_reward_info = self.calculate_progressive_reward(state, next_state, raw_lidar_range)
        reward += base_reward
        reward_info.update(base_reward_info)
        
        # calculate reward modifiers:
        for modifier_type, weight in TwoCarEnvironment.REWARD_MODIFIERS:
            match modifier_type:
                case 'wall_proximity':
                    dist_to_wall = min(raw_lidar_range)
                    close_to_wall_penalize_factor = 1 / (1 + np.exp(35 * (dist_to_wall - 0.5))) #y=\frac{1}{1+e^{35\left(x-0.5\right)}}
                    reward -= reward * close_to_wall_penalize_factor * weight
                    reward_info.update({"dist_to_wall":["avg",dist_to_wall]})
                    #print(f"--- Wall proximity penalty factor: {weight} * {close_to_wall_penalize_factor}")   
                case 'turn':
                    steering_angle1 = twist_to_ackermann(state[7], state[6], L=0.325)
                    steering_angle2 = twist_to_ackermann(next_state[7], next_state[6], L=0.325)
                    angle_diff = abs(steering_angle1 - steering_angle2)
                    if angle_diff > 3:
                        turning_penalty_factor = 0
                    else:
                        turning_penalty_factor = 1 - (1 / (1 + np.exp(15 * (angle_diff - 0.3)))) #y=1-\frac{1}{1+e^{15\left(x-0.3\right)}}
                    
                    reward -= reward * turning_penalty_factor * weight
                    #print(f"--- Turning penalty factor: {weight} * {turning_penalty_factor}")
                case 'racing':
                    # point1 = self.CURR_TRACK_MODEL.get_closest_point_on_spline(odom1[:2], t_only=True)
                    # point2 = self.CURR_TRACK_MODEL.get_closest_point_on_spline(odom2[:2], t_only=True)
                    if self.NAME == 'f1tenth':
                        if self.EP_PROGRESS1 == self.EP_PROGRESS2:
                            modifier=0
                        else:
                            modifier = (self.EP_PROGRESS1 - self.EP_PROGRESS2)/abs(self.EP_PROGRESS1 - self.EP_PROGRESS2)
                    else:
                        if self.EP_PROGRESS1 == self.EP_PROGRESS2:
                            modifier=0
                        else:
                            modifier = (self.EP_PROGRESS2 - self.EP_PROGRESS1)/abs(self.EP_PROGRESS2 - self.EP_PROGRESS1)
                    reward += reward * modifier * weight
                    self.LAST_POS1 = odom1[:2]
                    self.LAST_POS2 = odom2[:2]  

        return reward, reward_info
    
    ##########################################################################################
    ########################## Reward Calculation ############################################
    ##########################################################################################
    
    def calculate_progressive_reward(self, state, next_state, raw_range):
        reward = 0

        goal_position = self.GOAL_POS

        current_distance = math.dist(goal_position, next_state[:2])
        
        # keep track of non moving steps
        if self.STEP_PROGRESS < 0.02:
            self.PROGRESS_NOT_MET_COUNTER += 1
        else:
            self.PROGRESS_NOT_MET_COUNTER = 0

        reward += self.STEP_PROGRESS

        #print(f"Step progress: {self.step_progress}")
       
        self.STEPS_WITHOUT_GOAL += 1

        if current_distance < self.REWARD_RANGE:
            #print(f'Goal #{self.goals_reached} Reached')
            # reward += 2
            self.GOALS_REACHED += 1

            # Updating Goal Position
            new_x, new_y, _, _ = self.CURR_WAYPOINTS[(self.SPAWN_INDEX + self.GOALS_REACHED) % len(self.CURR_WAYPOINTS)]
            self.GOAL_POS = [new_x, new_y]


            self.update_goal_service(new_x, new_y)

            self.STEPS_WITHOUT_GOAL = 0

        if self.PROGRESS_NOT_MET_COUNTER >= 5:
            reward -= 2

        if has_collided(raw_range, self.COLLISION_RANGE) or has_flipped_over(next_state[2:6]):
            reward -= 2.5

        info = {}

        return reward, info

    def calculate_total_progress_reward(self, total_progression, step_progression, next_state, raw_range):
        reward = 0

        if step_progression < 0.02:
            self.PROGRESS_NOT_MET_COUNTER += 1
        else:
            self.PROGRESS_NOT_MET_COUNTER = 0

        reward += total_progression

        if self.PROGRESS_NOT_MET_COUNTER >= 5:
            reward -= 2
        if has_collided(raw_range, self.COLLISION_RANGE) or has_flipped_over(next_state[2:6]):
            reward -= 2.5

        info = {}

        return reward, info


    ##########################################################################################
    ########################## Utility Functions #############################################
    ##########################################################################################
    
    def get_odoms(self):
        # Get odom of both cars

        rclpy.spin_until_future_complete(self, self.ODOM_OBSERVATION_FUTURE)
        future = self.ODOM_OBSERVATION_FUTURE
        self.ODOM_OBSERVATION_FUTURE = Future()
        data = future.result()
        odom1 = process_odom(data['odom1'])
        odom2 = process_odom(data['odom2'])
        return odom1, odom2
    
    def publish_status(self, status):
        msg = String()
        msg.data = str(status)
        self.STATUS_PUB.publish(msg)

    def status_callback(self, msg):
        self.STATUS = msg.data
        self.STATUS_OBSERVATION_FUTURE.set_result({'status': msg}) 
        #self.get_logger().info(str(self.NAME) + "reads " + str(self.STATUS))

    def change_status_lock(self, change):
        msg = String()
        msg.data = str(change)
        self.STATUS_LOCK_PUB.publish(msg)
        self.STATUS_LOCK = change

    def status_lock_callback(self, msg):
        self.STATUS_LOCK = msg.data

    def parse_status(self, msg):
        indexes = findOccurrences(msg, '_')
        comma = findOccurrences(msg, ',')
        track = msg[(indexes[0]+1):indexes[2]]
        goalx = float(msg[(indexes[2]+2):(comma[0])])
        goaly = float(msg[(comma[0]+2):(indexes[3]-1)])
        goal = goalx, goaly
        spawn_index = int(msg[(indexes[3]+1):comma[1]])
        return track, goal, spawn_index
