import math
import rclpy
import numpy as np
from rclpy import Future
import random
from environment_interfaces.srv import Reset
from environments.F1tenthEnvironment import F1tenthEnvironment
from .util import has_collided, has_flipped_over
from .util import get_track_math_defs, process_ae_lidar, process_odom, avg_lidar, create_lidar_msg, get_all_goals_and_waypoints_in_multi_tracks, ackermann_to_twist, reconstruct_ae_latent, lateral_translation
from .util_track_progress import TrackMathDef
from .waypoints import waypoints
from std_srvs.srv import SetBool
from typing import Literal, List, Optional, Tuple
import torch
from datetime import datetime
from message_filters import Subscriber, ApproximateTimeSynchronizer
from nav_msgs.msg import Odometry


class TwoCarEnvironment(F1tenthEnvironment):


    def __init__(self, 
                 car_name, 
                 reward_range=0.5, 
                 max_steps=3000, 
                 collision_range=0.2, 
                 step_length=0.5, 
                 track='track_1',
                 observation_mode='lidar_only',
                 ):
        
        max_steps = 200
        super().__init__('two_car', car_name, max_steps, step_length)

        

        #####################################################################################################################
        # CHANGE SETTINGS HERE, might be specific to environment, therefore not moved to config file (for now at least).

        # Reward configuration
        self.EXTRA_REWARD_TERMS:List[Literal['penalize_turn']] = []
        self.REWARD_MODIFIERS:List[Tuple[Literal['turn','wall_proximity', 'racing'],float]] = [('turn', 0.3), ('wall_proximity', 0.7), ('racing', 1)] # [ (penalize_turn", 0.3), (penalize_wall_proximity, 0.7) ]

        # Observation configuration
        self.LIDAR_PROCESSING:Literal["avg","pretrained_ae", "raw"] = 'avg'
        self.LIDAR_POINTS = 10 #682
        self.EXTRA_OBSERVATIONS:List[Literal['prev_ang_vel']] = []

        # Evaluation settings
        self.MULTI_TRACK_TRAIN_EVAL_SPLIT=0.5 

        #optional stuff
        pretrained_ae_path = "/home/anyone/autonomous_f1tenth/lidar_ae_ftg_rand.pt" #"/ws/lidar_ae_ftg_rand.pt"

        # Speed and turn limit
        self.MAX_ACTIONS = np.asarray([2, 0.434])
        self.MIN_ACTIONS = np.asarray([0.3, -0.434])

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
        self.progress_not_met_cnt = 0


        # Reset Client -----------------------------------------------

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

        # Subscribe to both car's odometry --------------------------------------------
        self.odom_sub_1 = Subscriber(
            self,
            Odometry,
            f'/f1tenth/odometry',
        )

        self.odom_sub_2 = Subscriber(
            self,
            Odometry,
            f'/f1tenth_2/odometry',
        )

        self.odom_message_filter = ApproximateTimeSynchronizer(
            [self.odom_sub_1, self.odom_sub_2],
            10,
            0.1,
        )

        self.odom_message_filter.registerCallback(self.odom_message_filter_callback)

        self.odom_observation_future = Future()

        if self.is_multi_track:
            # define from which track in the track lists to be used for eval only
            self.eval_track_begin_idx = int(len(self.all_track_waypoints)*self.MULTI_TRACK_TRAIN_EVAL_SPLIT)
            # idx used to loop through eval tracks sequentially
            self.eval_track_idx = 0

        self.get_logger().info('Environment Setup Complete')



#    ____ _        _    ____ ____    _____ _   _ _   _  ____ _____ ___ ___  _   _ ____  
#   / ___| |      / \  / ___/ ___|  |  ___| | | | \ | |/ ___|_   _|_ _/ _ \| \ | / ___| 
#  | |   | |     / _ \ \___ \___ \  | |_  | | | |  \| | |     | |  | | | | |  \| \___ \ 
#  | |___| |___ / ___ \ ___) |__) | |  _| | |_| | |\  | |___  | |  | | |_| | |\  |___) |
#   \____|_____/_/   \_\____/____/  |_|    \___/|_| \_|\____| |_| |___\___/|_| \_|____/ 

    def odom_message_filter_callback(self, odom1: Odometry, odom2: Odometry):
        self.odom_observation_future.set_result({'odom1': odom1, 'odom2': odom2})                                                                             

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
        self.step_counter = 0
        self.steps_since_last_goal = 0
        self.goals_reached = 0

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
        
        if (self.current_track_key[-3:]).isdigit():
            width = int(self.current_track_key[-3:])
        else: 
            width = 300
        
        car_x, car_y, car_yaw, index = random.choice(self.track_waypoints)
        #car_yaw = self.randomize_yaw(car_yaw, 0.25)

        #car_2_offset = random.randint(8, 16)  
        #car_2_index = (index + car_2_offset) % len(self.track_waypoints)
        #car_2_x, car_2_y, car_2_yaw, _ = self.track_waypoints[car_2_index]
        #car_2_yaw = self.randomize_yaw(car_2_yaw, 0.25)
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
                car_2_index = (index + car_2_offset) % len(self.track_waypoints)
                car_2_x, car_2_y, car_2_yaw, _ = self.track_waypoints[car_2_index]
            else:
                car_2_offset = random.randint(8, 16)
                car_2_index = (index - car_2_offset) % len(self.track_waypoints)
                car_2_x, car_2_y, car_2_yaw, _ = self.track_waypoints[car_2_index]
        
        
        # Update goal pointer to reflect starting position
        self.start_waypoint_index = index
        x,y,_,_ = self.track_waypoints[self.start_waypoint_index+1 if self.start_waypoint_index+1 < len(self.track_waypoints) else 0]# point toward next goal
        self.goal_position = [x,y]

        self.call_reset_service(car_x=car_x, car_y=car_y, car_Y=car_yaw, goal_x=x, goal_y=y, car_name='f1tenth')
        self.call_reset_service(car_x=car_2_x, car_y=car_2_y, car_Y=car_2_yaw, goal_x=x, goal_y=y, car_name='f1tenth_2')
        self.get_logger().info('Goal position:' + str(x) + ', ' + str(y))
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
        self.progress_not_met_cnt = 0

        return state, info
    
    def start_eval(self):
        self.eval_track_idx = 0
        self.is_evaluating = True

    def stop_eval(self):
        self.is_evaluating = False

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

        return next_state, reward, terminated, truncated, info

    def is_terminated(self, state, ranges):
        return has_collided(ranges, self.COLLISION_RANGE) \
            or has_flipped_over(state[2:6])

    def is_truncated(self):
        return self.progress_not_met_cnt >= 5 or \
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

        # Add extra observation:
        for extra_observation in self.EXTRA_OBSERVATIONS:
            match extra_observation:
                case 'prev_ang_vel':
                    if self.full_current_state:
                        state += [self.full_current_state[7]]
                    else:
                        state += [state[7]]

        
        full_state = odom + processed_lidar_range

        return state, full_state, lidar.ranges

    def compute_reward(self, state, next_state, raw_lidar_range):
        '''Compute reward based on FULL states: odom + lidar + extra'''
        reward = 0
        reward_info = {}

        # calculate base reward
        base_reward, base_reward_info = self.calculate_progressive_reward(state, next_state, raw_lidar_range)
        reward += base_reward
        reward_info.update(base_reward_info)

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
                    #print(f"--- Wall proximity penalty factor: {weight} * {close_to_wall_penalize_factor}")   
                case 'turn':
                    angular_vel_diff = abs(state[7] - next_state[7])
                    turning_penalty_factor = 1 - (1 / (1 + np.exp(15 * (angular_vel_diff - 0.3)))) #y=1-\frac{1}{1+e^{15\left(x-0.3\right)}}
                    reward -= reward * turning_penalty_factor * weight
                    #print(f"--- Turning penalty factor: {weight} * {turning_penalty_factor}")
                case 'racing':
                    odom1, odom2 = self.get_odoms()
                    point1 = self.track_model.get_closest_point_on_spline(odom1[:2], t_only=True)
                    point2 = self.track_model.get_closest_point_on_spline(odom2[:2], t_only=True)
                    if self.NAME == 'f1tenth':
                        if point1 == point2:
                            modifier=0
                        else:
                            modifier = (point1 > point2)
                    else:
                        if point1 == point2:
                            modifier=0
                        else:
                            modifier = (point2 > point1)
                    reward += reward * modifier * weight  

        return reward, reward_info
    
    ##########################################################################################
    ########################## Reward Calculation ############################################
    ##########################################################################################
    
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

        #print(f"Step progress: {self.step_progress}")
       
        self.steps_since_last_goal += 1

        if current_distance < self.REWARD_RANGE:
            #print(f'Goal #{self.goals_reached} Reached')
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
    
    def get_odoms(self):
        # Get odom of both cars

        rclpy.spin_until_future_complete(self, self.odom_observation_future)
        future = self.odom_observation_future
        self.odom_observation_future = Future()
        data = future.result()
        odom1 = process_odom(data['odom1'])
        odom2 = process_odom(data['odom2'])
        return odom1, odom2