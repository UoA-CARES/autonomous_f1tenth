import math
import random

import numpy as np
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation


def get_quaternion_from_euler(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    return [qx, qy, qz, qw]


def get_euler_from_quarternion(w, x, y, z):
    # scipy expects xyzw order
    roll, pitch, yaw = Rotation.from_quat([x, y, z, w]).as_euler("xyz")
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
    return [
        position.x,
        position.y,
        orientation.w,
        orientation.x,
        orientation.y,
        orientation.z,
        lin_vel.x,
        ang_vel.z,
    ]


def twist_to_ackermann(omega, linear_v, L):
    if linear_v == 0:
        return 0
    delta = math.atan((L * omega) / linear_v)
    return delta


def ackermann_to_twist(delta, linear_v, L):
    try:
        omega = math.tan(delta) * linear_v / L
    except ZeroDivisionError:
        print("Wheelbase must be greater than zero")
        return 0
    return omega


def has_flipped_over(
    quaternion_wxyz: list[float], tilt_limit_rad: float = math.radians(60.0)
):
    """
    Returns True if roll or pitch exceeds tilt_limit_rad.
    Expects quaternion in [w, x, y, z] order.
    """
    _, x, y, _ = quaternion_wxyz
    up_z = 1.0 - 2.0 * (x * x + y * y)  # z-component of rotated world up vector
    return up_z < math.cos(tilt_limit_rad)


def lateral_translation(spline_location, angle, shift):
    x, y = spline_location
    x1 = x + shift * math.cos(angle + (math.pi / 2))
    y1 = y + shift * math.sin(angle + (math.pi / 2))
    return x1, y1


def find_occurrences(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]
