import numpy as np
import random
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

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

def process_lidar(lidar: LaserScan):
    ranges = lidar.ranges
    ranges = np.nan_to_num(ranges, posinf=float(-1), neginf=float(-1))
    ranges = list(ranges)

    intensities = list(lidar.intensities)
    return ranges, intensities

def avg_reduce_lidar(lidar: LaserScan):
        ranges = lidar.ranges
        ranges = np.nan_to_num(ranges, posinf=float(-1), neginf=float(-1))
        ranges = list(ranges)

        reduced_range = []

        for i in range(10):
            avg = sum(ranges[i * 64: i * 64 + 64]) / 64
            reduced_range.append(avg)

        return reduced_range

def reduce_lidar(lidar: LaserScan):
        ranges = lidar.ranges
        ranges = np.nan_to_num(ranges, posinf=float(10), neginf=float(0))
        ranges = list(ranges)

        reduced_range = []

        for i in range(10):
            sample = ranges[i * 64]
            reduced_range.append(sample)

        return reduced_range