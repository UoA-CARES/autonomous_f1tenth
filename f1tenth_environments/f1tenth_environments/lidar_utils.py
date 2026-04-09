import numpy as np
import scipy
import scipy.signal
import torch
from sensor_msgs.msg import LaserScan


def avg_lidar(lidar: LaserScan, num_points: int):
    ranges = np.nan_to_num(
        lidar.ranges, nan=float(10), posinf=float(10), neginf=float(10)
    )
    ranges = ranges[1:]
    new_range = []
    angle = 240 / num_points
    iter_step = 240 / len(ranges)
    num_ind = np.ceil(angle / iter_step)
    x = 1
    total = ranges[0]

    while x < len(ranges):
        if x % num_ind == 0:
            new_range.append(float(total / num_ind))
            total = 0
        total += ranges[x]
        x += 1
    if total > 0:
        new_range.append(float(total / (len(ranges) % num_ind)))
    return new_range


def avg_lidar_w_consensus(lidar: LaserScan, num_points: int):
    # For each 'sector', count non hitting rays, if non hitting rays >= 50% consider entire sector non-hitting. Otherwise use avg of hitting rays.
    ranges = np.nan_to_num(
        lidar.ranges, nan=float(-5), posinf=float(-5), neginf=float(-5)
    )
    sector_size = len(ranges) // num_points
    processed_data = []

    for i in range(num_points):
        sector = ranges[i * sector_size : (i + 1) * sector_size]
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
    ranges = lidar.ranges
    ranges = np.nan_to_num(ranges, nan=float(10), posinf=float(10), neginf=float(10))
    new_range = []

    window_size = [121, 70, 60, 50, 40, 40, 50, 60, 70, 122]

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


# This function is terrible at detecting obstacles....
def process_lidar_med_filt(
    lidar: LaserScan, window_size: int, nan_to=-5
):  # -> np.ArrayLike:
    ranges = np.array(lidar.ranges.tolist())
    ranges = np.nan_to_num(ranges, posinf=nan_to, nan=nan_to, neginf=nan_to).tolist()
    processed_ranges = scipy.ndimage.median_filter(
        ranges, window_size, mode="nearest"
    ).tolist()
    return processed_ranges


def process_ae_lidar(lidar: LaserScan, ae_model, is_latent_only=True):
    range_list = np.array(lidar.ranges)
    range_list = np.nan_to_num(range_list, posinf=-5)
    range_list = scipy.signal.resample(range_list, 512)
    range_tensor = (
        torch.tensor(range_list, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    )

    if is_latent_only:
        return ae_model.encoder(range_tensor).tolist()[0]
    else:
        return ae_model(range_tensor).tolist()[0][0]


def process_ae_lidar_beta_vae(lidar: LaserScan, ae_model, is_latent_only=True):
    range_list = np.array(lidar.ranges)
    range_list = np.nan_to_num(range_list, posinf=-5)
    range_list = scipy.signal.resample(range_list, 512)
    range_tensor = (
        torch.tensor(range_list, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    )

    if is_latent_only:
        return ae_model.get_latent(range_tensor)
    else:
        print(ae_model.get_latent(range_tensor))
    return ae_model.generate(range_tensor).tolist()[0][0]


def reconstruct_ae_latent(original_lidar: LaserScan, ae_model, latent: list):
    latent_tensor = torch.tensor(latent)
    reconstructed_range = ae_model.decoder(latent_tensor).tolist()[0]
    reconstructed_range = scipy.signal.resample(
        reconstructed_range, len(original_lidar.ranges)
    )
    return np.array(reconstructed_range, dtype=np.float32).tolist()


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
    new_range = []
    for index in idx:
        new_range.append(float(ranges[index]))
    return new_range


def has_collided(lidar_ranges, collision_range):
    return any(0 < ray < collision_range for ray in lidar_ranges)
