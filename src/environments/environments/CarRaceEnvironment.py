import rclpy
import numpy as np
from rclpy import Future
import random
from environment_interfaces.srv import Reset
from environments.F1tenthEnvironment import F1tenthEnvironment
from .util import process_ae_lidar, process_odom, avg_lidar, create_lidar_msg, reconstruct_ae_latent
from .util_track_progress import TrackMathDef
from .waypoints import waypoints
from typing import Literal, List
import torch
import yaml

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
                 config_path='/home/anyone/autonomous_f1tenth/src/environments/config/config.yaml',
                 ):
        super().__init__('car_race', car_name, max_steps, step_length)

        

        #####################################################################################################################
        # CHANGE SETTINGS HERE, might be specific to environment, therefore not moved to config file (for now at least).
        
        # Load configuration from YAML file
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        # Observation configuration
        self.LIDAR_PROCESSING:Literal["avg","pretrained_ae", "raw"] = 'avg'
        self.LIDAR_POINTS = 10 #682
        self.EXTRA_OBSERVATIONS:List[Literal['prev_ang_vel']] = []


        #optional stuff
        pretrained_ae_path = "/home/anyone/autonomous_f1tenth/lidar_ae_ftg_rand.pt" #"/ws/lidar_ae_ftg_rand.pt"

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

        self.ODOM_OBSERVATION_MODE = observation_mode
        self.TRACK = track

        # initialize track progress utilities
        self.PREV_T = None
        self.ALL_TRACK_MODELS = None
        self.TRACK_MODEL = None
        self.STEP_PROGRESS = 0

        if self.LIDAR_PROCESSING == 'pretrained_ae':
            from .autoencoders.lidar_autoencoder import LidarConvAE
            self.AE_LIDAR_MODEL = LidarConvAE()
            self.AE_LIDAR_MODEL.load_state_dict(torch.load(pretrained_ae_path))
            self.AE_LIDAR_MODEL.eval()

        # Reset Client -----------------------------------------------

        self.GOALS_REACHED = 0
        self.START_WAYPOINT_INDEX = 0
        self.FULL_CURRENT_STATE = None


        if "test_track" in track:
            track_key = track[0:-4] # "test_track_xx_xxx" -> "test_track_xx", here due to test_track's different width variants having the same waypoints.
        else:
            track_key = track

        self.TRACK_WAYPOINTS = waypoints[track_key]
        self.TRACK_MODEL = TrackMathDef(np.array(self.TRACK_WAYPOINTS)[:,:2])


        # Evaluation related setup ---------------------------------------------------
        self.IS_EVAL = False

        self.STEP_COUNTER = 0

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
        self.STEP_COUNTER = 0

        self.set_velocity(0, 0)
        
        # start at beginning of track when evaluating
        if self.IS_EVAL:
            car_x, car_y, car_yaw, index = self.TRACK_WAYPOINTS[10]
            car_2_x, car_2_y, car_2_yaw, _ = self.TRACK_WAYPOINTS[16]
        # start the car randomly along the track
        else:
            car_x, car_y, car_yaw, index = random.choice(self.TRACK_WAYPOINTS)
            car_2_x, car_2_y, car_2_yaw, _ = self.TRACK_WAYPOINTS[index+2 if index+20 < len(self.TRACK_WAYPOINTS) else 0]

        # Update goal pointer to reflect starting position
        self.START_WAYPOINT_INDEX = index

        self.call_reset_service(car_x=car_x, car_y=car_y, car_Y=car_yaw, car_name=self.NAME)
        self.call_reset_service(car_x=car_2_x, car_y=car_2_y, car_Y=car_2_yaw, car_name='f1tenth_2')

        # Get initial observation
        self.call_step(pause=False)
        state, full_state , _ = self.get_observation()
        self.FULL_CURRENT_STATE = full_state
        self.call_step(pause=True)

        info = {}
        
        self.PREV_T = self.TRACK_MODEL.get_closest_point_on_spline(full_state[:2], t_only=True)

        return state, info

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
        request.flag = "car"

        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        return future.result()