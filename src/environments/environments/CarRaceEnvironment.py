import math
import rclpy
import numpy as np
from rclpy import Future
import random
from environment_interfaces.srv import Reset
from environments.F1tenthEnvironment import F1tenthEnvironment
from .util import get_track_math_defs, process_ae_lidar, process_odom, avg_lidar, create_lidar_msg, get_all_goals_and_waypoints_in_multi_tracks, ackermann_to_twist, reconstruct_ae_latent, has_collided, has_flipped_over
from .util_track_progress import TrackMathDef
from .waypoints import waypoints
from std_srvs.srv import SetBool
from typing import Literal, List, Optional, Tuple
import torch
from datetime import datetime

class CarRaceEnvironment(F1tenthEnvironment):

    """
    CarRace Environment:
        Currently a test environment only.

        Task:
            Agent races an opponent

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
            Its linear and angular velocity (Twist)
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
        super().__init__('car_race', car_name, max_steps, step_length)

        

        #####################################################################################################################
        # CHANGE SETTINGS HERE, might be specific to environment, therefore not moved to config file (for now at least).

        # Observation configuration
        self.LIDAR_PROCESSING:Literal["avg","pretrained_ae", "raw"] = 'avg'
        self.LIDAR_POINTS = 10 #682
        self.EXTRA_OBSERVATIONS:List[Literal['prev_ang_vel']] = []


        #optional stuff
        pretrained_ae_path = "/home/anyone/autonomous_f1tenth/lidar_ae_ftg_rand.pt" #"/ws/lidar_ae_ftg_rand.pt"

        # Speed and turn limit
        self.MAX_ACTIONS = np.asarray([3, 0.434])
        self.MIN_ACTIONS = np.asarray([0, -0.434])

        #####################################################################################################################

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

        self.odom_observation_mode = observation_mode
        self.track = track

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

        # Reset Client -----------------------------------------------

        self.goals_reached = 0
        self.start_waypoint_index = 0
        self.full_current_state = None


        if "test_track" in track:
            track_key = track[0:-4] # "test_track_xx_xxx" -> "test_track_xx", here due to test_track's different width variants having the same waypoints.
        else:
            track_key = track

        self.track_waypoints = waypoints[track_key]
        self.track_model = TrackMathDef(np.array(self.track_waypoints)[:,:2])


        # Evaluation related setup ---------------------------------------------------
        self.is_evaluating = False


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
        
        # start at beginning of track when evaluating
        if self.is_evaluating:
            car_x, car_y, car_yaw, index = self.track_waypoints[10]
            car_2_x, car_2_y, car_2_yaw, _ = self.track_waypoints[16]
        # start the car randomly along the track
        else:
            car_x, car_y, car_yaw, index = random.choice(self.track_waypoints)
            car_2_x, car_2_y, car_2_yaw, _ = self.track_waypoints[index+2 if index+20 < len(self.track_waypoints) else 0]
        
        # Update goal pointer to reflect starting position
        self.start_waypoint_index = index
        x,y,_,_ = self.track_waypoints[self.start_waypoint_index+1 if self.start_waypoint_index+1 < len(self.track_waypoints) else 0]# point toward next goal
        self.goal_position = [x,y]

        self.call_reset_service(car_x=car_x, car_y=car_y, car_Y=car_yaw, car_name=self.NAME)
        self.call_reset_service(car_x=car_2_x, car_y=car_2_y, car_Y=car_2_yaw, car_name='f1tenth_2')

        # Get initial observation
        self.call_step(pause=False)
        state, full_state , _ = self.get_observation()
        self.full_current_state = full_state
        self.call_step(pause=True)

        info = {}

        # get track progress related info
        # set new track model if its multi track
        
        self.prev_t = self.track_model.get_closest_point_on_spline(full_state[:2], t_only=True)

        # reward function specific resets
        

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

        match self.BASE_REWARD_FUNCTION:

            case 'goal_hitting':
                return self.steps_since_last_goal >= 20 or \
                self.step_counter >= self.MAX_STEPS
            case 'progressive':
                return self.progress_not_met_cnt >= 5 or \
                self.step_counter >= self.MAX_STEPS
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

    ##########################################################################################
    ########################## Utility Functions #############################################
    ##########################################################################################

    def call_reset_service(self, car_x, car_y, car_Y, car_name):
        """
        Reset the car and goal position
        """

        request = Reset.Request()
        request.car_name = car_name
        request.cx = float(car_x)
        request.cy = float(car_y)
        request.cyaw = float(car_Y)
        request.flag = "car_and_goal"

        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        return future.result()
    
    def sleep(self):
        while not self.timer_future.done():
            rclpy.spin_once(self)

        self.timer_future = Future()
    
    def parse_observation(self, observation):
        
        string = f'CarRace Observation\n'

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