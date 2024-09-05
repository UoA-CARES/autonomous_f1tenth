import numpy as np
import random
import math

import scipy.signal
import torch.types
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from .goal_positions import goal_positions
from .waypoints import waypoints
from .util_track_progress import TrackMathDef
import torch
import scipy

from rclpy.impl import rcutils_logger
logger = rcutils_logger.RcutilsLogger(name="util_log")

def get_quaternion_from_euler(roll, pitch, yaw):
  """
  Convert an Euler angle to a quaternion.
   
  Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.
 
  Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
  """
  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
  return [qx, qy, qz, qw]

def get_euler_from_quarternion(w,x,y,z):
    """
    Convert a quaternion w, x, y, z to Euler angles [roll, pitch, yaw]

    Args:
    quaternion (list or tuple): A list or tuple containing the quaternion components [w, x, y, z]

    Returns:
    tuple: A tuple containing the Euler angles (roll, pitch, yaw)
    """

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def generate_position(inner_bound=3, outer_bound=8):
        inner_bound = float(inner_bound)
        outer_bound = float(outer_bound)

        x_pos = random.uniform(-outer_bound, outer_bound)
        x_pos = x_pos + inner_bound if x_pos >= 0 else x_pos - inner_bound
        y_pos = random.uniform(-outer_bound, outer_bound)
        y_pos = y_pos + inner_bound if y_pos >= 0 else y_pos - inner_bound

        return [x_pos, y_pos]
      
def process_odom(odom: Odometry):
        pose = odom.pose.pose
        position = pose.position
        orientation = pose.orientation

        twist = odom.twist.twist
        lin_vel = twist.linear
        ang_vel = twist.angular

        return [position.x, position.y, orientation.w, orientation.x, orientation.y, orientation.z, lin_vel.x,
                ang_vel.z]

def reduce_lidar(lidar: LaserScan, num_points: int):
        num_outputs = num_points
        ranges = lidar.ranges
        ranges = np.nan_to_num(ranges, nan=float(10), posinf=float(10), neginf=float(10))
        ranges = ranges[1:]
        idx = np.round(np.linspace(1, len(ranges) - 1, num_outputs)).astype(int)
       
        new_range = []

        for index in idx:
            new_range.append(ranges[index])

        return new_range

# Reduce lidar so all values are facing forward from the robot

def avg_lidar(lidar: LaserScan, num_points: int):
  
        ranges = lidar.ranges
        ranges = np.nan_to_num(ranges, nan=float(10), posinf=float(10), neginf=float(10))
        ranges = ranges[1:]
                                                   
        new_range = []

        angle = 240/num_points
        iter = 240/len(ranges)
        num_ind = np.ceil(angle/iter)

        x = 1
        sum = ranges[0]

        while(x < len(ranges)):
                if(x%num_ind == 0):
                        new_range.append(float(sum/num_ind))
                        sum = 0
                sum += ranges[x]
                x += 1
        if(sum > 0):
                new_range.append(float(sum/(len(ranges)%num_ind)))
        
        return new_range

def process_ae_lidar(lidar:LaserScan, ae_model, is_latent_only=True):
    range_list = np.array(lidar.ranges)
    range_list = np.nan_to_num(range_list, posinf=-5)
    range_list = scipy.signal.resample(range_list, 512)
    range_tensor = torch.tensor(range_list, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    if (is_latent_only):
         return ae_model.encoder(range_tensor).tolist()[0]
    else:
        return ae_model(range_tensor).tolist()[0][0]

def process_ae_lidar_beta_vae(lidar:LaserScan, ae_model, is_latent_only=True):
    range_list = np.array(lidar.ranges)
    range_list = np.nan_to_num(range_list, posinf=-5)
    range_list = scipy.signal.resample(range_list, 512)
    range_tensor = torch.tensor(range_list, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    if (is_latent_only):
         return ae_model.get_latent(range_tensor)
    else:
        print(ae_model.get_latent(range_tensor))
        return ae_model.generate(range_tensor).tolist()[0][0]

def reconstruct_ae_latent(original_lidar:LaserScan, ae_model, latent:list):
    latent_tensor = torch.tensor(latent)
    reconstructed_range = ae_model.decoder(latent_tensor).tolist()[0] #####
    reconstructed_range = scipy.signal.resample(reconstructed_range, len(original_lidar.ranges))
    return np.array(reconstructed_range,dtype=np.float32).tolist()

def create_lidar_msg(lidar: LaserScan, num_points: int, lidar_range: list):

    scan = LaserScan()
    scan.header.stamp.sec = lidar.header.stamp.sec
    scan.header.stamp.nanosec = lidar.header.stamp.nanosec
    scan.header.frame_id = lidar.header.frame_id
    scan.angle_min = lidar.angle_min
    scan.angle_max = lidar.angle_min
    scan.angle_increment = lidar.angle_max*2/(num_points-1) #240/num_points * (3.142 / 180)
    # scan.time_increment =  scan.angle_increment/ lidar.angle_increment * lidar.time_increment # processed ang / orig ang = processed time / orig time
    scan.range_min = lidar.range_min
    scan.range_max = lidar.range_max
    scan.ranges = lidar_range

    return scan



def forward_reduce_lidar(lidar: LaserScan):
    num_outputs = 10
    ranges = lidar.ranges
    max_angle = abs(lidar.angle_max)
    ideal_angle = 1.396
    angle_incr = lidar.angle_increment
    
    ranges = np.nan_to_num(ranges, nan=float(10), posinf=float(10), neginf=float(-10))
    ranges = ranges[1:]
    idx_cut = int((max_angle-ideal_angle)/angle_incr)
    idx = np.round(np.linspace(idx_cut, len(ranges)-(1+idx_cut), num_outputs)).astype(int)
    new_range = []

    for index in idx:
        new_range.append(float(ranges[index]))

    return new_range

def get_all_goals_and_waypoints_in_multi_tracks(track_name):
    all_car_goals = {}
    all_car_waypoints = {}

    if track_name == 'multi_track_full':

        # Goal position
        austin_gp = goal_positions['austin_track']
        budapest_gp = [[x + 200, y] for x, y in goal_positions['budapest_track']]
        hockenheim_gp = [[x + 300, y] for x, y in goal_positions['hockenheim_track']]
        melbourne_gp = [[x + 500, y] for x, y in goal_positions['melbourne_track']]
        saopaolo_gp = [[x + 600, y] for x, y in goal_positions['saopaolo_track']]
        shanghai_gp = [[x + 750, y] for x, y in goal_positions['shanghai_track']]

        all_car_goals = {
            'austin_track': austin_gp,
            'budapest_track': budapest_gp,
            'hockenheim_track': hockenheim_gp,
            'melbourne_track': melbourne_gp,
            'saopaolo_track': saopaolo_gp,
            'shanghai_track': shanghai_gp,
        }

        # Waypoints
        austin_wp = waypoints['austin_track']
        budapest_wp = [(x + 200, y, yaw, index) for x, y, yaw, index in waypoints['budapest_track']]
        hockenheim_wp = [(x + 300, y, yaw, index) for x, y, yaw, index in waypoints['hockenheim_track']]
        melbourne_wp =[(x + 500, y, yaw, index) for x, y, yaw, index in waypoints['melbourne_track']]
        saopaolo_wp = [(x + 600, y, yaw, index) for x, y, yaw, index in waypoints['saopaolo_track']]
        shanghai_wp = [(x + 750, y, yaw, index) for x, y, yaw, index in waypoints['shanghai_track']]

        all_car_waypoints = {
            'austin_track': austin_wp,
            'budapest_track': budapest_wp,
            'hockenheim_track': hockenheim_wp,
            'melbourne_track': melbourne_wp,
            'saopaolo_track': saopaolo_wp,
            'shanghai_track': shanghai_wp
        }


    elif track_name == 'multi_track':
        # multi_track
        # Goal position
        austin_gp = goal_positions['austin_track']
        budapest_gp = [[x + 200, y] for x, y in goal_positions['budapest_track']]
        hockenheim_gp = [[x + 300, y] for x, y in goal_positions['hockenheim_track']]

        all_car_goals = {
            'austin_track': austin_gp,
            'budapest_track': budapest_gp,
            'hockenheim_track': hockenheim_gp,
        }

        # Waypoints
        austin_wp = waypoints['austin_track']
        budapest_wp = [(x + 200, y, yaw, index) for x, y, yaw, index in waypoints['budapest_track']]
        hockenheim_wp = [(x + 300, y, yaw, index) for x, y, yaw, index in waypoints['hockenheim_track']]

        all_car_waypoints = {
            'austin_track': austin_wp,
            'budapest_track': budapest_wp,
            'hockenheim_track': hockenheim_wp
        }

    elif track_name == 'multi_track_testing':
        # multi_track_testing
        # Goal position
        melbourne_gp = goal_positions['melbourne_track']
        saopaolo_gp = [[x + 100, y] for x, y in goal_positions['saopaolo_track']]
        shanghai_gp = [[x + 250, y] for x, y in goal_positions['shanghai_track']]

        all_car_goals = {
            'melbourne_track': melbourne_gp,
            'saopaolo_track': saopaolo_gp,
            'shanghai_track': shanghai_gp,
        }

        # Waypoints
        melbourne_wp = waypoints['melbourne_track']
        saopaolo_wp = [(x + 100, y, yaw, index) for x, y, yaw, index in waypoints['saopaolo_track']]
        shanghai_wp = [(x + 250, y, yaw, index) for x, y, yaw, index in waypoints['shanghai_track']]

        all_car_waypoints = {
            'melbourne_track': melbourne_wp,
            'saopaolo_track': saopaolo_wp,
            'shanghai_track': shanghai_wp
        }

    elif track_name == 'multi_track_01':

        WIDTHS = [150, 200, 250, 300, 350]
        TRACKS = ['track_01', 'track_02', 'track_03', 'track_04', 'track_05', 'track_06']
        
        # Usage of goals deprecated
        all_car_goals = None

        all_car_waypoints = {
             
        }

        i = 0
        
        # loop through each track
        for track in TRACKS:
             # loop through each width variant of each track
             for width in WIDTHS:
                  # combine to get the correct key for returned dict
                  track_name = f"{track}_{str(width)}"
                  # set correct x offset
                  global_wp = [(x + i*30, y, yaw, index) for x, y, yaw, index in waypoints[track]]
                  all_car_waypoints.update({track_name : global_wp})
                  i += 1

    elif track_name == 'multi_track_test_01':
        WIDTHS = [150, 200, 250, 300, 350]
        TRACKS = ['test_track_01','test_track_02']
        
        # Usage of goals deprecated
        all_car_goals = None

        all_car_waypoints = {
             
        }

        i = 0
        
        for track in TRACKS:
             for width in WIDTHS:
                  track_name = f"{track}_{str(width)}"
                  global_wp = [(x + i*30, y, yaw, index) for x, y, yaw, index in waypoints[track]]
                  all_car_waypoints.update({track_name : global_wp})
                  i += 1
         
        # Waypoint
        # track_01_100_wp = [(x, y, yaw, index) for x, y, yaw, index in waypoints['track_01']]
        # track_01_150_wp = [(x + 10, y, yaw, index) for x, y, yaw, index in waypoints['track_01']]
        # track_01_200_wp = [(x + 20, y, yaw, index) for x, y, yaw, index in waypoints['track_01']]

        # all_car_waypoints = {
        #     'track_01_100': track_01_100_wp,
        #     'track_01_150': track_01_150_wp,
        #     'track_01_200': track_01_200_wp
        # }

    elif track_name == 'multi_track_02':
        TRACKS = ['track_01','track_02','track_03','track_04','narrow_track_01','narrow_track_02','narrow_track_03', 'track_05', 'track_06', 'narrow_track_04']
        WIDE_WIDTH = [150, 200, 250, 300, 350]
        NARROW_WIDTH = [100, 150]
        all_car_goals = None

        all_car_waypoints = { } 
        
        i = 0
        # loop through each track
        for track in TRACKS:
            if "narrow" in track:
                width_range = NARROW_WIDTH
            else:
                width_range = WIDE_WIDTH
            # loop through each width variant of each track
            for width in width_range:
                # combine to get the correct key for returned dict
                track_name = f"{track}_{str(width)}"
                # set correct x offset
                global_wp = [(x + i*30, y, yaw, index) for x, y, yaw, index in waypoints[track]]
                all_car_waypoints.update({track_name : global_wp})
                i += 1

       

    return all_car_goals, all_car_waypoints

def get_track_math_defs(tracks_waypoints:dict) -> dict[str,TrackMathDef]:
    '''Expect {trackname: [Waypoint]}, output {trackname: TrackMathDef}'''
    track_math_models = {}

    for track_name in tracks_waypoints.keys():
        track_math_models[track_name] = TrackMathDef(np.array(tracks_waypoints[track_name])[:,:2])
        # print(track_math_models)
    return track_math_models

def twist_to_ackermann(omega, linear_v, L):
    '''
    Convert angular velocity about center of turn to Ackerman steering angle.

    Parameters:
    - omega: angular velocity about center of turn in rad/s
    - v: Vehicle speed in m/s
    - L: Wheelbase of the vehicle in m

    Returns:
    - delta: Ackerman steering angle in radians

    Derivation:
    R = v / omega 
    R = L / tan(delta)  equation 10 from https://www.researchgate.net/publication/228464812_Electric_Vehicle_Stability_with_Rear_Electronic_Differential_Traction#pf3
    tan(delta) = L * omega / v
    delta = arctan(L * omega/ v)
    '''
    if linear_v == 0:
        return 0

    delta = math.atan((L * omega) / linear_v)

    return delta


def ackermann_to_twist(delta, linear_v, L):
    try: 
        omega = math.tan(delta)*linear_v/L
    except ZeroDivisionError:
        print("Wheelbase must be greater than zero")
        return 0
    return omega

