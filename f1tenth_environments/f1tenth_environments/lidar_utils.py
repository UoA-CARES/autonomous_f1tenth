from typing import Sequence

import numpy as np
from sensor_msgs.msg import LaserScan


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


class LidarProcessor:
    """Convert lidar scans to RL state and back using configured sectoring."""

    def __init__(
        self,
        num_points: int,
        forward_half_angle: float | None = None,
        n_forward: int = 4,
        k_fraction: float = 0.15,
        k_floor: int = 2,
        k_cap: int = 5,
    ) -> None:
        if num_points <= 0:
            raise ValueError(f"num_points must be > 0, got {num_points}")
        if n_forward <= 0:
            raise ValueError(f"n_forward must be > 0, got {n_forward}")
        if n_forward > num_points:
            raise ValueError(
                f"n_forward ({n_forward}) cannot exceed num_points ({num_points})"
            )
        self.num_points = num_points
        self.forward_half_angle = forward_half_angle
        self.n_forward = n_forward
        self.k_fraction = k_fraction
        self.k_floor = k_floor
        self.k_cap = k_cap

    def _adaptive_k(self, n_valid_beams: int) -> int:
        return int(
            np.clip(
                round(n_valid_beams * self.k_fraction),
                self.k_floor,
                self.k_cap,
            )
        )

    @staticmethod
    def _sector_distance(beams: np.ndarray, k: int) -> float:
        finite = beams[np.isfinite(beams)]
        if len(finite) == 0:
            return -1.0
        k_actual = min(k, len(finite))
        return float(np.mean(np.partition(finite, k_actual - 1)[:k_actual]))

    def _resolved_forward_half_angle(
        self,
        angle_min_deg: float,
        angle_max_deg: float,
    ) -> float:
        resolved = (
            (angle_max_deg - angle_min_deg) / 2.0
            if self.forward_half_angle is None
            else self.forward_half_angle
        )
        if resolved <= 0:
            raise ValueError(f"forward_half_angle must be > 0, got {resolved}")
        if resolved > (angle_max_deg - angle_min_deg) / 2:
            raise ValueError(
                f"forward_half_angle ({resolved}°) exceeds half the scan FOV "
                f"({(angle_max_deg - angle_min_deg) / 2}°)"
            )
        return float(resolved)

    def _boundaries(
        self,
        angle_min_deg: float,
        angle_max_deg: float,
        forward_half_angle: float,
    ) -> np.ndarray:
        n_remaining = self.num_points - self.n_forward
        n_left = n_remaining // 2
        n_right = n_remaining - n_left
        eps = 1e-9

        full_fov = np.isclose(forward_half_angle, (angle_max_deg - angle_min_deg) / 2.0)
        if full_fov:
            return np.linspace(angle_min_deg, angle_max_deg + eps, self.num_points + 1)

        return np.concatenate(
            [
                np.linspace(angle_min_deg, -forward_half_angle, n_left + 1),
                np.linspace(
                    -forward_half_angle, forward_half_angle, self.n_forward + 1
                )[1:],
                np.linspace(forward_half_angle, angle_max_deg + eps, n_right + 1)[1:],
            ]
        )

    def lidar_to_state(self, lidar_scan: LaserScan) -> np.ndarray:
        """
        Converts a raw LIDAR scan into a compact normalised state vector for RL.

        Sector layout (top-down, car facing up, 10 points / n_forward=4 / fwd=45°):

                                ^ forward (0°)
                                |
                    -45°        |       +45°
                        \  F1 F2 F3 F4 /
                         \ |  |  |  | /
                    L2 --  |  |  |  | -- R2
                    /   \  |  |  |  |  /   \
                L1       \ |  |  |  | /     R1
                           |__car__|

        Beam angular widths per region (270° FOV example):
        Left  sectors  (L1, L2):  each ~45°   <- coarse side coverage
        Forward sectors (F1–F4):  each ~22.5° <- fine forward resolution
        Right sectors  (R1, R2):  each ~45°   <- coarse side coverage

        Output vector (index 0 = leftmost, index N-1 = rightmost):

        idx:  [  0     1     2     3     4     5     6     7     8     9  ]
        name: [  L1    L2    F1    F2    F3    F4    R1    R2             ]

        value:  -1.0   no data (blind spot / all-NaN returns in sector)
                0.0   obstacle at min_range
                0.5   obstacle at mid-range
                1.0   clear to max_range

        Robust obstacle detection per sector:

        raw beams:  [ 0.45  0.43  NaN  7.20  0.44  8.10  8.00 ]
                            ↑ k = clamp(n_valid × 0.15, floor=2, cap=5)
                            ↑ mean of k-smallest finite values
        sector out:   0.44m  (phantom 7.2 and NaN discarded — k beams must agree)

        NaN / out-of-range handling:
        - values outside [range_min, range_max]  →  NaN  (masked before sectoring)
        - sectors where all returns are NaN      →  -1.0 sentinel (not "clear")
        - sectors with fewer than k valid beams  →  k shrinks to n_valid gracefully

        Parameters:
        lidar_scan  raw ROS2 LaserScan message — range limits and angles
                    taken directly from the message, nothing hardcoded

        Returns:
            np.ndarray shape (num_points,) dtype float32
        """

        raw_scan = np.array(lidar_scan.ranges, dtype=float)
        min_range = lidar_scan.range_min
        max_range = lidar_scan.range_max
        angle_min_deg = np.degrees(lidar_scan.angle_min)
        angle_max_deg = np.degrees(lidar_scan.angle_max)
        forward_half_angle = self._resolved_forward_half_angle(
            angle_min_deg,
            angle_max_deg,
        )

        scan = raw_scan.copy()
        scan[(scan < min_range) | (scan > max_range)] = np.nan

        beam_angles = np.linspace(angle_min_deg, angle_max_deg, len(scan))
        boundaries = self._boundaries(
            angle_min_deg,
            angle_max_deg,
            forward_half_angle,
        )

        state: list[float] = []
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            mask = (beam_angles >= lo) & (beam_angles < hi)
            sector_beams = scan[mask]

            n_valid = int(np.isfinite(sector_beams).sum())
            k = self._adaptive_k(n_valid)
            raw_dist = self._sector_distance(sector_beams, k)

            if raw_dist < 0:
                state.append(-1.0)
            else:
                normalised = (raw_dist - min_range) / (max_range - min_range)
                state.append(float(np.clip(normalised, 0.0, 1.0)))

        return np.array(state, dtype=np.float32)

    def state_to_laserscan(
        self, state: np.ndarray, original_scan: LaserScan
    ) -> LaserScan:
        if len(state) != self.num_points:
            raise ValueError(
                f"state length ({len(state)}) must match num_points ({self.num_points})"
            )

        min_range = original_scan.range_min
        max_range = original_scan.range_max
        angle_min_deg = np.degrees(original_scan.angle_min)
        angle_max_deg = np.degrees(original_scan.angle_max)
        forward_half_angle = self._resolved_forward_half_angle(
            angle_min_deg,
            angle_max_deg,
        )
        boundaries = self._boundaries(
            angle_min_deg,
            angle_max_deg,
            forward_half_angle,
        )
        midpoints_rad = np.radians((boundaries[:-1] + boundaries[1:]) / 2.0)

        ranges = []
        for val in state:
            if val < 0:
                ranges.append(float(max_range))
            else:
                dist = val * (max_range - min_range) + min_range
                ranges.append(float(np.clip(dist, min_range, max_range)))

        intensities = [float(i) / (self.num_points - 1) for i in range(self.num_points)]

        msg = LaserScan()
        msg.header.stamp = original_scan.header.stamp
        msg.header.frame_id = original_scan.header.frame_id
        msg.angle_min = float(midpoints_rad[0])
        msg.angle_max = float(midpoints_rad[-1])
        msg.angle_increment = float(
            (midpoints_rad[-1] - midpoints_rad[0]) / (self.num_points - 1)
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
    processor = LidarProcessor(
        num_points=demo_num_points,
        n_forward=demo_num_points,
        k_fraction=1.0,
        k_cap=1000,
    )
    proc_out = processor.lidar_to_state(dummy)
    reconstructed_scan = processor.state_to_laserscan(proc_out, dummy)

    print("=== Lidar reducer comparison ===")
    print(f"num_points: {demo_num_points}")
    print(f"avg_lidar (meters):      {np.asarray(avg_out)}")
    print(f"process_avg_lidar (norm): {proc_out}")
    print(f"state_to_laserscan (m):   {np.asarray(reconstructed_scan.ranges)}")
    print(
        "note: process_avg_lidar output is normalized to [0,1], while avg_lidar"
        " is in meters"
    )
