import rclpy
import numpy as np
from .controller import Controller


def main():
    rclpy.init()
    param_node = rclpy.create_node("params")
    param_node.declare_parameters(
        "",
        [
            ("car_name", "ftg_car"),
            ("track_name", "multi_track"),
        ],
    )
    params = param_node.get_parameters(["car_name", "track_name"])
    car_name, _ = [param.value for param in params]

    controller = Controller("ftg_policy_", car_name, 0.1)
    policy = FollowTheGapPolicy()
    policy_id = "ftg"
    state = controller.get_observation(policy_id)

    while rclpy.ok():
        action = policy.select_action(state)
        controller.get_logger().info(f"Selected action: {action}")
        state = controller.step(action, policy_id)

    rclpy.shutdown()


class FollowTheGapPolicy:
    """
    Improved Follow-The-Gap controller addressing performance bottlenecks:
    - Vectorized obstacle detection (no list append/delete in hot path)
    - Configurable lidar ray count (not hardcoded to 10)
    - Cached geometry calculations (meeting_dist computed once)
    - Dynamic speed control based on gap width and obstacle proximity
    - Proper numpy angle wrapping (no while loops)
    - Abstract state indexing for robustness
    """

    def __init__(
        self,
        turn_angle: float = 0.4667,
        min_turn_radius: float = 0.625,
        lidar_angle: float = 1.396,
        min_lidar_range: float = 0.08,
        obstacle_max_val: float = 4.0,
        min_velocity: float = 0.1,
        max_velocity: float = 3.0,
        odom_offset: int = 8,
    ):
        """
        Args:
            turn_angle: Maximum steering angle (rad)
            min_turn_radius: Minimum turning radius (m)
            lidar_angle: Half-width of lidar FOV (rad)
            min_lidar_range: Minimum lidar detection range (m)
            obstacle_max_val: Maximum lidar range to consider obstacle (m)
            min_velocity: Minimum linear velocity (m/s)
            max_velocity: Maximum linear velocity (m/s)
            odom_offset: Offset in state vector where lidar readings start
        """
        self.turn_angle = turn_angle
        self.min_turn_radius = min_turn_radius
        self.lidar_angle = lidar_angle
        self.min_lidar_range = min_lidar_range
        self.obstacle_max_val = obstacle_max_val
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.odom_offset = odom_offset

        # Cached geometry
        self.meeting_dist = np.sqrt(
            2 * self.min_turn_radius**2
            - 2 * self.min_turn_radius**2 * np.cos(2 * self.lidar_angle)
        )

        # Obstacle buffer and chassis width for border distance calculation
        self.obstacle_buffer = 0.001
        self.chassis_width = 0.16
        self.buffer_sum_sq = (self.obstacle_buffer + self.chassis_width) ** 2

    def _merge_blocked_intervals(
        self, blocked_intervals: np.ndarray
    ) -> list[tuple[float, float]]:
        if blocked_intervals.size == 0:
            return []

        sorted_intervals = blocked_intervals[np.argsort(blocked_intervals[:, 0])]
        merged_intervals: list[tuple[float, float]] = []
        current_start = float(sorted_intervals[0, 0])
        current_end = float(sorted_intervals[0, 1])

        for start, end in sorted_intervals[1:]:
            start = float(start)
            end = float(end)
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                merged_intervals.append((current_start, current_end))
                current_start = start
                current_end = end

        merged_intervals.append((current_start, current_end))
        return merged_intervals

    def _compute_free_gaps(
        self,
        merged_intervals: list[tuple[float, float]],
        search_left: float,
        search_right: float,
    ) -> list[tuple[float, float]]:
        if not merged_intervals:
            return [(search_right, search_left)]

        free_gaps: list[tuple[float, float]] = []
        current_angle = search_right

        for blocked_start, blocked_end in merged_intervals:
            if blocked_start > current_angle:
                free_gaps.append((current_angle, blocked_start))
            current_angle = max(current_angle, blocked_end)

        if current_angle < search_left:
            free_gaps.append((current_angle, search_left))

        return free_gaps

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action using simplified gap-finding logic."""
        state_array = np.asarray(state, dtype=float).reshape(-1)
        search_left = self.lidar_angle
        search_right = -self.lidar_angle

        # Extract lidar rays from state (all rays after odom_offset)
        lidar_ranges = state_array[self.odom_offset :]
        num_rays = len(lidar_ranges)

        if num_rays == 0:
            return np.asarray([self.min_velocity, 0.0])

        # Generate angle for each lidar ray
        angles = np.linspace(-self.lidar_angle, self.lidar_angle, num_rays)

        # Identify obstacles (vectorized check)
        is_obstacle = (lidar_ranges > self.min_lidar_range) & (
            lidar_ranges < self.obstacle_max_val
        )
        if not np.any(is_obstacle):
            return np.asarray([self.max_velocity, 0.0])

        obs_angles = angles[is_obstacle]
        obs_ranges = lidar_ranges[is_obstacle]

        # Calculate border angles for each obstacle
        border_dists = np.sqrt(np.maximum(1e-6, obs_ranges**2 - self.buffer_sum_sq))
        border_angle_offsets = np.arccos(np.clip(border_dists / obs_ranges, 0, 1))

        # Construct left and right border angles for each obstacle
        left_borders = obs_angles + border_angle_offsets
        right_borders = obs_angles - border_angle_offsets

        blocked_intervals = np.column_stack(
            (
                np.maximum(right_borders, search_right),
                np.minimum(left_borders, search_left),
            )
        )
        valid_intervals = blocked_intervals[:, 0] < blocked_intervals[:, 1]
        blocked_intervals = blocked_intervals[valid_intervals]

        merged_intervals = self._merge_blocked_intervals(blocked_intervals)
        free_gaps = self._compute_free_gaps(merged_intervals, search_left, search_right)

        if not free_gaps:
            return np.asarray([self.min_velocity, 0.0])

        gap_widths = np.asarray(
            [gap_end - gap_start for gap_start, gap_end in free_gaps], dtype=float
        )
        widest_idx = int(np.argmax(gap_widths))
        gap_start, gap_end = free_gaps[widest_idx]
        gap_width = gap_end - gap_start

        if gap_width <= 0:
            return np.asarray([self.min_velocity, 0.0])

        target_angle = (gap_start + gap_end) / 2.0

        # Normalize angle to [-π, π]
        target_angle = np.arctan2(np.sin(target_angle), np.cos(target_angle))

        # Dynamic speed: reduce near obstacles, increase in open gaps
        min_obs_range = np.min(obs_ranges)

        # Speed decreases as obstacle gets closer or gap narrower
        danger_proximity = 1.0 - np.clip(min_obs_range / self.obstacle_max_val, 0, 1)
        gap_openness = np.clip(gap_width / (2 * self.lidar_angle), 0, 1)

        speed = self.max_velocity * (0.7 * gap_openness + 0.3 * (1 - danger_proximity))
        speed = np.clip(speed, self.min_velocity, self.max_velocity)

        return np.asarray([speed, target_angle])


if __name__ == "__main__":
    main()
