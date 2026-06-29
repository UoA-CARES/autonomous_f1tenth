import math

import numpy as np
import scipy.ndimage
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import LaserScan


def get_euler_from_quaternion(w, x, y, z):
    roll, pitch, yaw = Rotation.from_quat([x, y, z, w]).as_euler("xyz")
    return roll, pitch, yaw


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


def avg_lidar(lidar: LaserScan, num_points: int):
    ranges = np.nan_to_num(
        lidar.ranges, nan=float(10), posinf=float(10), neginf=float(10)
    )
    ranges = ranges[1:]
    new_range = []
    angle = 240 / num_points
    increment = 240 / len(ranges)
    num_indices = np.ceil(angle / increment)
    index = 1
    total = ranges[0]

    while index < len(ranges):
        if index % num_indices == 0:
            new_range.append(float(total / num_indices))
            total = 0
        total += ranges[index]
        index += 1
    if total > 0:
        new_range.append(float(total / (len(ranges) % num_indices)))
    return new_range


def avg_lidar_w_consensus(lidar: LaserScan, num_points: int):
    ranges = np.nan_to_num(
        lidar.ranges, nan=float(-5), posinf=float(-5), neginf=float(-5)
    )
    sector_size = len(ranges) // num_points
    processed_data = []

    for index in range(num_points):
        sector = ranges[index * sector_size : (index + 1) * sector_size]
        non_hitting_count = np.sum(sector == -5)
        if non_hitting_count > sector_size / 2:
            processed_data.append(float(10))
        else:
            hitting_rays = sector[sector != -5]
            if len(hitting_rays) > 0:
                processed_data.append(float(np.mean(hitting_rays)))
            else:
                processed_data.append(float(10))
    return processed_data


def uneven_median_lidar(lidar: LaserScan, num_points: int):
    ranges = np.asarray(lidar.ranges, dtype=float)
    ranges = np.nan_to_num(
        ranges,
        nan=float(10),
        posinf=float(10),
        neginf=float(10),
    )
    if num_points < 1:
        raise ValueError("num_points must be at least 1")
    if len(ranges) < num_points:
        raise ValueError(
            f"Lidar scan has {len(ranges)} rays but {num_points} points were requested"
        )

    if num_points == 10:
        # Scale the original uneven 683-ray windows to the hardware resolution.
        reference_windows = np.asarray(
            [121, 70, 60, 50, 40, 40, 50, 60, 70, 122],
            dtype=float,
        )
        boundaries = np.rint(
            np.concatenate(([0.0], np.cumsum(reference_windows)))
            * len(ranges)
            / reference_windows.sum()
        ).astype(int)
        boundaries[0] = 0
        boundaries[-1] = len(ranges)
        sectors = [
            ranges[boundaries[index] : boundaries[index + 1]]
            for index in range(num_points)
        ]
    else:
        sectors = np.array_split(ranges, num_points)

    return [float(np.median(sector)) for sector in sectors]


def process_lidar_med_filt(lidar: LaserScan, window_size: int, nan_to=-5):
    ranges = np.array(lidar.ranges.tolist())
    ranges = np.nan_to_num(ranges, posinf=nan_to, nan=nan_to, neginf=nan_to).tolist()
    return scipy.ndimage.median_filter(ranges, window_size, mode="nearest").tolist()


def create_lidar_msg(lidar: LaserScan, num_points: int, lidar_range: list):
    scan = LaserScan()
    scan.header.stamp.sec = lidar.header.stamp.sec
    scan.header.stamp.nanosec = lidar.header.stamp.nanosec
    scan.header.frame_id = lidar.header.frame_id
    scan.angle_min = lidar.angle_min
    scan.angle_max = lidar.angle_min
    scan.angle_increment = lidar.angle_max * 2 / (num_points - 1)
    scan.range_min = lidar.range_min
    scan.range_max = lidar.range_max
    scan.ranges = lidar_range
    return scan


def forward_reduce_lidar(lidar: LaserScan):
    num_outputs = 10
    ideal_angle = 1.396
    ranges = lidar.ranges
    max_angle = abs(lidar.angle_max)
    angle_incr = lidar.angle_increment
    ranges = np.nan_to_num(ranges, nan=float(10), posinf=float(10), neginf=float(-10))
    ranges = ranges[1:]
    idx_cut = int((max_angle - ideal_angle) / angle_incr)
    idx = np.round(
        np.linspace(idx_cut, len(ranges) - (1 + idx_cut), num_outputs)
    ).astype(int)
    return [float(ranges[index]) for index in idx]


def ackermann_to_twist(delta, linear_v, wheelbase):
    try:
        return math.tan(delta) * linear_v / wheelbase
    except ZeroDivisionError:
        print("Wheelbase must be greater than zero")
        return 0
