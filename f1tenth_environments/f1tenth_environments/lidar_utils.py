from typing import Sequence

import numpy as np
import rclpy
import scipy
from sensor_msgs.msg import LaserScan


def _validate_positive_count(count: int, name: str) -> None:
    if count < 1:
        raise ValueError(f"{name} must be >= 1")


def _sanitize_lidar_ranges(
    lidar: LaserScan,
    *,
    nan_to: float,
    posinf_to: float,
    neginf_to: float,
) -> np.ndarray:
    ranges = np.asarray(lidar.ranges, dtype=np.float64)
    ranges = np.nan_to_num(ranges, nan=nan_to, posinf=posinf_to, neginf=neginf_to)
    return ranges


def _split_into_non_empty_sectors(
    ranges: np.ndarray, num_points: int
) -> list[np.ndarray]:
    _validate_positive_count(num_points, "num_points")
    if len(ranges) < num_points:
        raise ValueError("num_points cannot exceed the number of lidar rays")
    return [sector for sector in np.array_split(ranges, num_points) if len(sector) > 0]


def avg_lidar(lidar: LaserScan, num_points: int) -> list[float]:
    """Downsample a full scan by averaging contiguous angular sectors.

    Steps:
    1) Replace NaN and ±Inf in `lidar.ranges` with 10.0 meters.
    2) Split the scan into `num_points` contiguous sectors.
    3) Return one value per sector: the arithmetic mean of rays in that sector.

    Returns a list of length `num_points` (unless input is invalid and raises).
    """
    ranges = _sanitize_lidar_ranges(
        lidar,
        nan_to=10.0,
        posinf_to=10.0,
        neginf_to=10.0,
    )

    sectors = _split_into_non_empty_sectors(ranges, num_points)
    return [float(np.mean(sector)) for sector in sectors]


def create_lidar_msg(
    lidar: LaserScan, num_points: int, lidar_range: Sequence[float]
) -> LaserScan:
    """Create a LaserScan message for visualization from reduced lidar data."""
    if num_points < 2:
        raise ValueError("num_points must be >= 2")

    scan = LaserScan()
    scan.header.stamp.sec = lidar.header.stamp.sec
    scan.header.stamp.nanosec = lidar.header.stamp.nanosec
    scan.header.frame_id = lidar.header.frame_id
    scan.angle_min = lidar.angle_min
    scan.angle_max = lidar.angle_max
    scan.angle_increment = (lidar.angle_max - lidar.angle_min) / (num_points - 1)
    scan.range_min = lidar.range_min
    scan.range_max = lidar.range_max
    scan.ranges = list(lidar_range)
    return scan


def has_collided(lidar_ranges: Sequence[float], collision_range: float) -> bool:
    """Return True if any lidar ray reports an obstacle within collision range."""
    return any(0 < ray < collision_range for ray in lidar_ranges)


def _adaptive_k(
    n_valid_beams: int, fraction: float = 0.15, floor: int = 2, cap: int = 5
) -> int:
    return int(np.clip(round(n_valid_beams * fraction), floor, cap))


def _sector_distance(beams: np.ndarray, k: int) -> float:
    """
    Returns robust closest-obstacle distance for a sector.

    Sentinel values:
      -1.0  no finite returns at all — sensor blind spot or all-NaN
       otherwise raw distance in the same units as the input beams, caller is responsible for normalisation
    """
    finite = beams[np.isfinite(beams)]
    if len(finite) == 0:
        return -1.0  # explicitly no data, not "clear"
    k_actual = min(k, len(finite))
    return float(np.mean(np.partition(finite, k_actual - 1)[:k_actual]))


def lidar_to_state(
    lidar_scan: LaserScan,
    num_points: int,
    forward_half_angle: float | None = None,
    n_forward: int = 4,
    k_fraction: float = 0.15,
    k_floor: int = 2,
    k_cap: int = 5,
) -> np.ndarray:
    """
    Converts a raw LIDAR scan into a compact normalised state vector for RL.

    Example output for n_sectors=10, n_forward=4, forward_half_angle=20°:

                        FORWARD (0°)
                            |
              -20°          |          +20°
                \     F1 F2 | F3 F4   /
                 \    |  |  |  |  |  /
          FL\    |  |  |  |  |  |  |  |   /FR
              \  |  |  |  |  |  |  |  |  /
               [ LS  FL  F1  F2  F3  F4  FR  RS ]
                                                         (n_sectors=8 shown)

    Physical layout (top-down, car facing up):

                         ^ forward
                         |
                  ______|||______
                 |  F1 | | | F4 |    <- 4 narrow forward sectors (~10° each)
                 |FL   |   |  FR|    <- fore-left / fore-right (~40° each)
                 |LS   |car|  RS|    <- side-left / side-right (~80° each)
                 |_____|___|_____|

    Output vector (left to right = left to right physically):

      index:  [ 0      1      2      3      4      5      6      7      8      9  ]
      region: [ Lside  Lfore  Fwd1   Fwd2   Fwd3   Fwd4   Rfore  Rside         ]

      value:   -1.0    no data at all (blind spot / all-NaN returns)
               0.0     obstacle at min_range  (right next to sensor)
               0.5     obstacle at mid-range
               1.0     clear to max_range

    Robust obstacle detection — each sector value is the mean of the k
    smallest finite returns, where k = clamp(n_valid_beams * 0.15, 2, 5).
    Requires k beams to agree before registering a close obstacle,
    preventing single-beam phantom walls from triggering the agent.

                  raw:   [ 0.45  0.43  7.2   NaN  0.44  8.1  8.0 ]
                                  ^--- k=3 mean of 3 smallest finite
                  out:     0.44m  (ignores the outlier 7.2 and NaN)
    """
    # --- unpack LaserScan message ---
    raw_scan = np.array(lidar_scan.ranges, dtype=float)
    min_range = lidar_scan.range_min
    max_range = lidar_scan.range_max
    angle_min_deg = np.degrees(lidar_scan.angle_min)
    angle_max_deg = np.degrees(lidar_scan.angle_max)

    # derive after unpacking so validation has real values to check against
    if forward_half_angle is None:
        forward_half_angle = (angle_max_deg - angle_min_deg) / 2.0

    # --- parameter validation ---
    if num_points <= 0:
        raise ValueError(f"num_points must be > 0, got {num_points}")
    if n_forward <= 0:
        raise ValueError(f"n_forward must be > 0, got {n_forward}")
    if n_forward > num_points:
        raise ValueError(
            f"n_forward ({n_forward}) cannot exceed num_points ({num_points})"
        )
    if forward_half_angle <= 0:
        raise ValueError(f"forward_half_angle must be > 0, got {forward_half_angle}")
    if forward_half_angle > (angle_max_deg - angle_min_deg) / 2:
        raise ValueError(
            f"forward_half_angle ({forward_half_angle}°) exceeds half the scan FOV "
            f"({(angle_max_deg - angle_min_deg) / 2}°)"
        )

    # --- mask invalid returns using sensor's own range limits ---
    scan = raw_scan.copy()
    scan[(scan < min_range) | (scan > max_range)] = np.nan

    # --- sector allocation ---
    n_beams = len(scan)
    beam_angles = np.linspace(angle_min_deg, angle_max_deg, n_beams)

    n_remaining = num_points - n_forward
    n_left = n_remaining // 2
    n_right = n_remaining - n_left

    eps = 1e-9
    boundaries = np.concatenate(
        [
            np.linspace(angle_min_deg, -forward_half_angle, n_left + 1),
            np.linspace(-forward_half_angle, forward_half_angle, n_forward + 1)[1:],
            np.linspace(forward_half_angle, angle_max_deg + eps, n_right + 1)[1:],
        ]
    )

    state = []
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        mask = (beam_angles >= lo) & (beam_angles < hi)
        sector_beams = scan[mask]

        n_valid = int(np.isfinite(sector_beams).sum())
        k = _adaptive_k(n_valid, k_fraction, floor=k_floor, cap=k_cap)
        raw_dist = _sector_distance(sector_beams, k)

        if raw_dist < 0:
            state.append(-1.0)
        else:
            normalised = (raw_dist - min_range) / (max_range - min_range)
            state.append(float(np.clip(normalised, 0.0, 1.0)))

    return np.array(state, dtype=np.float32)


def process_avg_lidar(lidar: LaserScan, num_points: int) -> np.ndarray:
    return lidar_to_state(
        lidar_scan=lidar,
        num_points=num_points,
        n_forward=num_points,
        k_fraction=1.0,  # average all beams in sector, no k-filtering
        k_cap=1000,  # effectively no k cap
    )


def state_to_laserscan(
    state: np.ndarray,
    original_scan: LaserScan,
    forward_half_angle: float = None,
    n_forward: int = 4,
) -> LaserScan:
    num_points = len(state)
    min_range = original_scan.range_min
    max_range = original_scan.range_max
    angle_min_deg = np.degrees(original_scan.angle_min)
    angle_max_deg = np.degrees(original_scan.angle_max)

    if forward_half_angle is None:
        forward_half_angle = (angle_max_deg - angle_min_deg) / 2.0

    # --- sector boundaries (mirrors lidar_to_state) ---
    n_remaining = num_points - n_forward
    n_left = n_remaining // 2
    n_right = n_remaining - n_left

    eps = 1e-9
    full_fov = np.isclose(forward_half_angle, (angle_max_deg - angle_min_deg) / 2.0)

    if full_fov:
        boundaries = np.linspace(angle_min_deg, angle_max_deg + eps, num_points + 1)
    else:
        boundaries = np.concatenate(
            [
                np.linspace(angle_min_deg, -forward_half_angle, n_left + 1),
                np.linspace(-forward_half_angle, forward_half_angle, n_forward + 1)[1:],
                np.linspace(forward_half_angle, angle_max_deg + eps, n_right + 1)[1:],
            ]
        )

    # --- sector midpoint angles ---
    midpoints_rad = np.radians((boundaries[:-1] + boundaries[1:]) / 2.0)

    # --- denormalise state values to metres ---
    ranges = []
    for val in state:
        if val < 0:
            ranges.append(float(max_range))
        else:
            dist = val * (max_range - min_range) + min_range
            ranges.append(float(np.clip(dist, min_range, max_range)))

    # --- intensities — unique per sector for colour banding ---
    intensities = [float(i) / (num_points - 1) for i in range(num_points)]

    # --- build LaserScan message ---
    msg = LaserScan()
    msg.header.stamp = rclpy.clock.Clock().now().to_msg()
    msg.header.frame_id = original_scan.header.frame_id
    msg.angle_min = float(midpoints_rad[0])
    msg.angle_max = float(midpoints_rad[-1])
    msg.angle_increment = float(
        (midpoints_rad[-1] - midpoints_rad[0]) / (num_points - 1)
    )
    msg.range_min = min_range
    msg.range_max = max_range
    msg.time_increment = 0.0
    msg.scan_time = original_scan.scan_time
    msg.ranges = ranges
    msg.intensities = intensities

    return msg


if __name__ == "__main__":
    # Basic manual sanity check for comparing the two reducers.
    dummy = LaserScan()
    dummy.range_min = 0.0
    dummy.range_max = 10.0
    dummy.angle_min = float(np.radians(-135.0))
    dummy.angle_max = float(np.radians(135.0))

    # Build a synthetic scan with random baseline distances, obstacle pockets,
    # and a few invalid readings.
    beam_count = 1080
    rng = np.random.default_rng(42)
    base_scan = rng.uniform(0.5, 9.5, size=beam_count).astype(np.float32)

    # Add structured close-obstacle regions.
    base_scan[180:240] = rng.uniform(0.35, 1.1, size=60).astype(np.float32)
    base_scan[520:580] = rng.uniform(0.6, 1.8, size=60).astype(np.float32)

    # Inject invalid/anomalous samples.
    base_scan[120] = np.nan
    base_scan[121] = np.inf
    base_scan[122] = -np.inf
    base_scan[300] = 0.0
    base_scan[301] = 12.0

    dummy.ranges = base_scan.tolist()

    demo_num_points = 10
    avg_out = avg_lidar(dummy, demo_num_points)
    proc_out = process_avg_lidar(dummy, demo_num_points)
    reconstructed_scan = state_to_laserscan(
        state=proc_out,
        original_scan=dummy,
        # Must match process_avg_lidar configuration (all sectors are "forward").
        forward_half_angle=135.0,
        n_forward=demo_num_points,
    )

    print("=== Lidar reducer comparison ===")
    print(f"num_points: {demo_num_points}")
    print(f"avg_lidar (meters):      {np.asarray(avg_out)}")
    print(f"process_avg_lidar (norm): {proc_out}")
    print(f"state_to_laserscan (m):   {np.asarray(reconstructed_scan.ranges)}")
    print(
        "note: process_avg_lidar output is normalized to [0,1], while avg_lidar"
        " is in meters"
    )
