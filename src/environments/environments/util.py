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

def uneven_median_lidar(lidar: LaserScan, num_points: int):
        ranges = lidar.ranges
        ranges = np.nan_to_num(ranges, nan=float(10), posinf=float(10), neginf=float(10))
        new_range = []
        
        window_size = [121, 70, 60 ,50, 40, 40, 50, 60, 70, 122]
        
        if len(ranges) != sum(window_size):
            raise Exception("Lidar length and window size do not match")
        
        if len(window_size) != num_points:
            raise Exception("Window size length and num_points do not match")
        
        start = 0
        for window in window_size:
            end = start + window
            window_ranges = ranges[start:end]
            new_range.append(float(np.median(window_ranges)))
            start = end
            
        return new_range

def avg_lidar_w_consensus(lidar:LaserScan, num_points:int):
    '''For each 'sector', count non hitting rays, if non hitting rays >= 50% consider entire sector non-hitting. Otherwise use avg of hitting rays.'''
    ranges = lidar.ranges
    ranges = np.nan_to_num(ranges, nan=float(-5), posinf=float(-5), neginf=float(-5))
    # ranges = ranges[1:]
                                                
    # Calculate sector size
    sector_size = len(ranges) // num_points
    processed_data = []
    
    for i in range(num_points):
        # Get the corresponding sector
        sector = ranges[i * sector_size: (i + 1) * sector_size]
        
        # Count non-hitting rays (-5 values)
        non_hitting_count = np.sum(sector == -5) # sector == -5 returns a boolean array, true being any -5 occurance. sum just count trues. chatgpt black magic right here
        
        if non_hitting_count > sector_size / 2:
            # If more non-hitting rays, return 10
            processed_data.append(float(10))
        else:
            # Return average of hitting rays (values other than -5)
            hitting_rays = sector[sector != -5]
            if len(hitting_rays) > 0:
                processed_data.append(float(np.mean(hitting_rays)))
            else:
                processed_data.append(float(10))  # All rays are non-hitting
        
    return processed_data

# This function is terrible at detecting obstacles....
def process_lidar_med_filt(lidar:LaserScan, window_size:int, nan_to = -5): #-> np.ArrayLike:
    ranges = np.array(lidar.ranges.tolist())
    ranges = np.nan_to_num(ranges, posinf=nan_to, nan=nan_to, neginf=nan_to).tolist()  
    processed_ranges = scipy.ndimage.median_filter(ranges, window_size, mode='nearest').tolist()
    return processed_ranges

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
    elif track_name == 'multi_track_wide':
        # Goal position
        all_car_goals = None

        # Waypoints
        track_01_wp = waypoints['track_01']
        track_02_wp = [(x + 30, y, yaw, index) for x, y, yaw, index in waypoints['track_02']]
        track_03_wp = [(x + 60, y, yaw, index) for x, y, yaw, index in waypoints['track_03']]
        track_04_wp = [(x + 90, y, yaw, index) for x, y, yaw, index in waypoints['track_04']]
        track_05_wp = [(x + 120, y, yaw, index) for x, y, yaw, index in waypoints['track_05']]
        track_06_wp = [(x + 150, y, yaw, index) for x, y, yaw, index in waypoints['track_06']]

        all_car_waypoints = {
            'track_01': track_01_wp,
            'track_02': track_02_wp,
            'track_03': track_03_wp,
            'track_04': track_04_wp,
            'track_05': track_05_wp,
            'track_06': track_06_wp
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
    elif track_name == 'multi_track_02':

        WIDTHS = [350] #[150, 200, 250, 300, 350]
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
                  
    elif track_name == 'staged_tracks':
        WIDTHS = [350, 300, 250, 200, 150]
        TRACKS = ['track_01', 'track_02', 'track_03', 'track_04', 'track_05', 'track_06']
        
        # Usage of goals deprecated
        all_car_goals = None
        all_car_waypoints = {     
        }
        i = 0
        
        for width in WIDTHS:
            for track in TRACKS:
                  track_name = f"{track}_{str(width)}"
                  global_wp = [(x + i*30, y, yaw, index) for x, y, yaw, index in waypoints[track_name]]
                  all_car_waypoints.update({track_name : global_wp})
                  i += 1

    elif track_name == 'narrow_multi_track':
        # Goal position - goals deprecated for narrow tracks
        all_car_goals = None

        # Waypoints - reordered with vary_track_width_new first, with larger spacing
        vary_track_width_new_wp = [(x + 1, y, yaw, index) for x, y, yaw, index in waypoints['vary_track_width_new']]
        spiral_track_wp = [(x + 22, y, yaw, index) for x, y, yaw, index in waypoints['spiral_track']]  
        track_01_1m_wp = [(x + 40, y, yaw, index) for x, y, yaw, index in waypoints['track_01_1m']]
        track_02_1m_wp = [(x + 49, y, yaw, index) for x, y, yaw, index in waypoints['track_02_1m']]  
        track_03_1m_wp = [(x + 58, y, yaw, index) for x, y, yaw, index in waypoints['track_03_1m']]  
        track_04_1m_wp = [(x + 67, y, yaw, index) for x, y, yaw, index in waypoints['track_04_1m']]  
        track_05_1m_wp = [(x + 76, y, yaw, index) for x, y, yaw, index in waypoints['track_05_1m']]  
        track_06_1m_wp = [(x + 85, y, yaw, index) for x, y, yaw, index in waypoints['track_06_1m']]  
        narrow_track_01_wp = [(x + 94, y, yaw, index) for x, y, yaw, index in waypoints['narrow_track_01']]  
        narrow_track_02_wp = [(x + 125, y, yaw, index) for x, y, yaw, index in waypoints['narrow_track_02']]
        narrow_track_03_wp = [(x + 156, y, yaw, index) for x, y, yaw, index in waypoints['narrow_track_03']]
        narrow_track_04_wp = [(x + 187, y, yaw, index) for x, y, yaw, index in waypoints['narrow_track_04']]
        narrow_track_05_wp = [(x + 218, y, yaw, index) for x, y, yaw, index in waypoints['narrow_track_05']]
        narrow_track_06_wp = [(x + 249, y, yaw, index) for x, y, yaw, index in waypoints['narrow_track_06']]
        track_01_2m_wp = [(x + 280, y, yaw, index) for x, y, yaw, index in waypoints['track_01_2m']]
        track_02_2m_wp = [(x + 296, y, yaw, index) for x, y, yaw, index in waypoints['track_02_2m']]
        track_03_2m_wp = [(x + 312, y, yaw, index) for x, y, yaw, index in waypoints['track_03_2m']]
        track_04_2m_wp = [(x + 328, y, yaw, index) for x, y, yaw, index in waypoints['track_04_2m']]
        track_05_2m_wp = [(x + 344, y, yaw, index) for x, y, yaw, index in waypoints['track_05_2m']]
        track_06_2m_wp = [(x + 360, y, yaw, index) for x, y, yaw, index in waypoints['track_06_2m']]

        all_car_waypoints = {
            'vary_track_width_new': vary_track_width_new_wp,
            'spiral_track': spiral_track_wp,
            # train vvvvv eval ^^^^^
            'track_01_1m': track_01_1m_wp,
            'track_02_1m': track_02_1m_wp,
            'track_03_1m': track_03_1m_wp,
            'track_04_1m': track_04_1m_wp,
            'track_05_1m': track_05_1m_wp,
            'track_06_1m': track_06_1m_wp,
            'narrow_track_01': narrow_track_01_wp,
            'narrow_track_02': narrow_track_02_wp,
            'narrow_track_03': narrow_track_03_wp,
            'narrow_track_04': narrow_track_04_wp,
            'narrow_track_05': narrow_track_05_wp,
            'narrow_track_06': narrow_track_06_wp,
            'track_01_2m': track_01_2m_wp,
            'track_02_2m': track_02_2m_wp,
            'track_03_2m': track_03_2m_wp,
            'track_04_2m': track_04_2m_wp,
            'track_05_2m': track_05_2m_wp,
            'track_06_2m': track_06_2m_wp
        }

    elif track_name == 'narrow_multi_track':
        # Goal position - goals deprecated for narrow tracks
        all_car_goals = None

        # Waypoints - reordered with vary_track_width_new first, with larger spacing
        bumpy_track_wp = waypoints['bumpy_track']
        vary_track_width_new_wp = [(x + 1, y, yaw, index) for x, y, yaw, index in waypoints['vary_track_width_new']]
        spiral_track_wp = [(x + 22, y, yaw, index) for x, y, yaw, index in waypoints['spiral_track']]  
        track_01_1m_wp = [(x + 40, y, yaw, index) for x, y, yaw, index in waypoints['track_01_1m']]
        track_02_1m_wp = [(x + 49, y, yaw, index) for x, y, yaw, index in waypoints['track_02_1m']]  
        track_03_1m_wp = [(x + 58, y, yaw, index) for x, y, yaw, index in waypoints['track_03_1m']]  
        track_04_1m_wp = [(x + 67, y, yaw, index) for x, y, yaw, index in waypoints['track_04_1m']]  
        track_05_1m_wp = [(x + 76, y, yaw, index) for x, y, yaw, index in waypoints['track_05_1m']]  
        track_06_1m_wp = [(x + 85, y, yaw, index) for x, y, yaw, index in waypoints['track_06_1m']]  
        narrow_track_01_wp = [(x + 94, y, yaw, index) for x, y, yaw, index in waypoints['narrow_track_01']]  
        narrow_track_02_wp = [(x + 125, y, yaw, index) for x, y, yaw, index in waypoints['narrow_track_02']]
        narrow_track_03_wp = [(x + 156, y, yaw, index) for x, y, yaw, index in waypoints['narrow_track_03']]
        narrow_track_04_wp = [(x + 187, y, yaw, index) for x, y, yaw, index in waypoints['narrow_track_04']]
        narrow_track_05_wp = [(x + 218, y, yaw, index) for x, y, yaw, index in waypoints['narrow_track_05']]
        narrow_track_06_wp = [(x + 249, y, yaw, index) for x, y, yaw, index in waypoints['narrow_track_06']]

        all_car_waypoints = {
            'bumpy_track': bumpy_track_wp,
            'vary_track_width_new': vary_track_width_new_wp,
            'spiral_track': spiral_track_wp,
            'narrow_track_01': narrow_track_01_wp,
            'narrow_track_02': narrow_track_02_wp,
            'narrow_track_03': narrow_track_03_wp,
            'narrow_track_04': narrow_track_04_wp,
            'narrow_track_05': narrow_track_05_wp,
            'track_01_1m': track_01_1m_wp,
            'track_02_1m': track_02_1m_wp,
            'track_03_1m': track_03_1m_wp,
            'track_04_1m': track_04_1m_wp,
            # eval vvvvv train ^^^^^
            'narrow_track_06': narrow_track_06_wp,
            'track_05_1m': track_05_1m_wp,
            'track_06_1m': track_06_1m_wp
        }

    elif track_name == 'narrow_multi_track':
        # Goal position - goals deprecated for narrow tracks
        all_car_goals = None

        # Waypoints - reordered with vary_track_width_new first, with larger spacing
        vary_track_width_new_wp = [(x + 1, y, yaw, index) for x, y, yaw, index in waypoints['vary_track_width_new']]
        spiral_track_wp = [(x + 22, y, yaw, index) for x, y, yaw, index in waypoints['spiral_track']]  
        track_01_1m_wp = [(x + 40, y, yaw, index) for x, y, yaw, index in waypoints['track_01_1m']]
        track_02_1m_wp = [(x + 49, y, yaw, index) for x, y, yaw, index in waypoints['track_02_1m']]  
        track_03_1m_wp = [(x + 58, y, yaw, index) for x, y, yaw, index in waypoints['track_03_1m']]  
        track_04_1m_wp = [(x + 67, y, yaw, index) for x, y, yaw, index in waypoints['track_04_1m']]  
        track_05_1m_wp = [(x + 76, y, yaw, index) for x, y, yaw, index in waypoints['track_05_1m']]  
        track_06_1m_wp = [(x + 85, y, yaw, index) for x, y, yaw, index in waypoints['track_06_1m']]  
        narrow_track_01_wp = [(x + 94, y, yaw, index) for x, y, yaw, index in waypoints['narrow_track_01']]  
        narrow_track_02_wp = [(x + 125, y, yaw, index) for x, y, yaw, index in waypoints['narrow_track_02']]
        narrow_track_03_wp = [(x + 156, y, yaw, index) for x, y, yaw, index in waypoints['narrow_track_03']]
        narrow_track_04_wp = [(x + 187, y, yaw, index) for x, y, yaw, index in waypoints['narrow_track_04']]
        narrow_track_05_wp = [(x + 218, y, yaw, index) for x, y, yaw, index in waypoints['narrow_track_05']]
        narrow_track_06_wp = [(x + 249, y, yaw, index) for x, y, yaw, index in waypoints['narrow_track_06']]
        track_01_2m_wp = [(x + 265, y, yaw, index) for x, y, yaw, index in waypoints['track_01_2m']]
        track_02_2m_wp = [(x + 281, y, yaw, index) for x, y, yaw, index in waypoints['track_02_2m']]
        track_03_2m_wp = [(x + 297, y, yaw, index) for x, y, yaw, index in waypoints['track_03_2m']]
        track_04_2m_wp = [(x + 313, y, yaw, index) for x, y, yaw, index in waypoints['track_04_2m']]
        track_05_2m_wp = [(x + 329, y, yaw, index) for x, y, yaw, index in waypoints['track_05_2m']]
        track_06_2m_wp = [(x + 345, y, yaw, index) for x, y, yaw, index in waypoints['track_06_2m']]

        all_car_waypoints = {
            'vary_track_width_new': vary_track_width_new_wp,
            'spiral_track': spiral_track_wp,
            # train vvvvv eval ^^^^^
            'track_01_1m': track_01_1m_wp,
            'track_02_1m': track_02_1m_wp,
            'track_03_1m': track_03_1m_wp,
            'track_04_1m': track_04_1m_wp,
            'track_05_1m': track_05_1m_wp,
            'track_06_1m': track_06_1m_wp,
            'narrow_track_01': narrow_track_01_wp,
            'narrow_track_02': narrow_track_02_wp,
            'narrow_track_03': narrow_track_03_wp,
            'narrow_track_04': narrow_track_04_wp,
            'narrow_track_05': narrow_track_05_wp,
            'narrow_track_06': narrow_track_06_wp,
            'track_01_2m': track_01_2m_wp,
            'track_02_2m': track_02_2m_wp,
            'track_03_2m': track_03_2m_wp,
            'track_04_2m': track_04_2m_wp,
            'track_05_2m': track_05_2m_wp,
            'track_06_2m': track_06_2m_wp
        }

    return all_car_goals, all_car_waypoints

def get_training_stages(track_name):
    # Stage track indices
    if track_name == 'narrow_multi_track':
        return {
            0: [(14, 19), (0, 1)],    # Training (start, end), Eval (start, end), both inclusive
            1: [(8, 13), (0, 1)],
            2: [(2, 13), (0, 1)],
        }
    elif track_name == 'staged_tracks':
        return {
            0: [(0, 3), (4, 5)],
            1: [(6, 9), (10, 11)],
            2: [(12, 15), (16, 17)],
            3: [(18, 21), (22, 23)],
            4: [(24, 27), (28, 29)],
        }
    else:
        raise Exception(f"Track {track_name} not designed for staged training.")

def get_track_math_defs(tracks_waypoints:dict) -> dict[str,TrackMathDef]:
    '''Expect {trackname: [Waypoint]}, output {trackname: TrackMathDef}'''
    track_math_models = {}

    for track_name in tracks_waypoints.keys():
        track_math_models[track_name] = TrackMathDef(np.array(tracks_waypoints[track_name])[:,:2])
        print(track_math_models)
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

# Terminations
def has_collided(lidar_ranges, collision_range):
    return any(0 < ray < collision_range for ray in lidar_ranges)

def has_flipped_over(quaternion):
    _, x, y, _ = quaternion
    return abs(x) > 0.5 or abs(y) > 0.5

def reached_goal(car_pos, goal_pos, reward_range):
    distance = math.dist(car_pos, goal_pos)
    return distance < reward_range

def lateral_translation(spline_location, angle, shift):
    x, y = spline_location
    x1 = x + shift*math.cos(angle+(math.pi/2))
    y1 = y + shift*math.sin(angle+(math.pi/2))
    return x1, y1

def findOccurrences(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]