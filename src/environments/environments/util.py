import numpy as np
import random
import math
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from .goal_positions import goal_positions
from .waypoints import waypoints

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
        if num_points < 1:
          raise Exception("Make sure number of points is more than 0")
        
        ranges = lidar.ranges
        ranges = np.nan_to_num(ranges, nan=float(10), posinf=float(10), neginf=float(10))
        ranges = ranges[1:]
                                                   
        new_range = []
         
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
        new_range.append(ranges[index])

    return new_range

def get_all_goals_and_waypoints_in_multi_tracks(track_name):
    all_car_goals = {}
    all_car_waypoints = {}

    if track_name == 'multi_track':
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

    return all_car_goals, all_car_waypoints

def twist_to_ackermann(omega, linear_v, L):
    '''
    Convert CG angular velocity to Ackerman steering angle.

    Parameters:
    - omega: CG angular velocity in rad/s
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
