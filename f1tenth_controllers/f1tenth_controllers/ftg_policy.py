import rclpy
import numpy as np
from typing import Literal
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
        obstacle_max_val: float = 2.2,
        min_velocity: float = 0.5,
        max_velocity: float = 3.75,
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

        # FTG behavior toggles (defaults chosen for stable 10-point lidar operation).
        self.use_disparity_extension = False
        self.gap_selection_mode: Literal["widest", "heading_bias"] = "widest"
        self.speed_mode: Literal["linear_gap_speed", "nonlinear_gap_speed"] = (
            "nonlinear_gap_speed"
        )
        self.fallback_mode: Literal["crawl_straight", "brake_hold"] = "crawl_straight"
        self.use_steering_smoothing = False
        self.steering_smoothing_alpha = 0.6
        self.prev_target_angle = 0.0

        self.disparity_threshold = 0.5
        self.gap_heading_weight = 0.35

    def _extract_lidar_ranges(self, state_array: np.ndarray) -> np.ndarray:
        return state_array[self.odom_offset :]

    def _compute_ray_angles(self, num_rays: int) -> np.ndarray:
        return np.linspace(-self.lidar_angle, self.lidar_angle, num_rays)

    def _preprocess_lidar(self, lidar_ranges: np.ndarray) -> np.ndarray:
        # FTG-specific post processing after upstream lidar processing.
        processed = np.asarray(lidar_ranges, dtype=float)
        processed = np.where(np.isfinite(processed), processed, self.obstacle_max_val)
        processed = np.clip(processed, self.min_lidar_range, self.obstacle_max_val)
        return processed

    def _apply_disparity_extension(
        self, lidar_ranges: np.ndarray, angles: np.ndarray
    ) -> np.ndarray:
        if not self.use_disparity_extension:
            return lidar_ranges

        if len(lidar_ranges) < 2:
            return lidar_ranges

        base_ranges = np.asarray(lidar_ranges, dtype=float)
        extended_ranges = base_ranges.copy()
        angle_per_ray = float(np.mean(np.abs(np.diff(angles))))
        if angle_per_ray <= 0:
            return extended_ranges

        safe_radius = self.obstacle_buffer + self.chassis_width
        range_diffs = np.abs(np.diff(base_ranges))
        disparity_indices = np.where(range_diffs > self.disparity_threshold)[0]

        for idx in disparity_indices:
            left = base_ranges[idx]
            right = base_ranges[idx + 1]
            short_idx = idx if left <= right else idx + 1
            short_range = max(float(base_ranges[short_idx]), self.min_lidar_range)

            if short_range <= safe_radius:
                extend_rays = len(extended_ranges)
            else:
                half_width_angle = float(
                    np.arcsin(np.clip(safe_radius / short_range, 0.0, 1.0))
                )
                extend_rays = int(np.ceil(half_width_angle / angle_per_ray))

            if extend_rays <= 0:
                continue

            if short_idx == idx:
                end = min(len(extended_ranges), idx + 1 + extend_rays)
                extended_ranges[idx + 1 : end] = short_range
            else:
                start = max(0, idx + 1 - extend_rays)
                extended_ranges[start : idx + 1] = short_range

        return extended_ranges

    def _identify_obstacles(self, lidar_ranges: np.ndarray) -> np.ndarray:
        return (lidar_ranges > self.min_lidar_range) & (
            lidar_ranges < self.obstacle_max_val
        )

    def _compute_blocked_intervals(
        self,
        obs_angles: np.ndarray,
        obs_ranges: np.ndarray,
        search_left: float,
        search_right: float,
    ) -> np.ndarray:
        border_dists = np.sqrt(np.maximum(1e-6, obs_ranges**2 - self.buffer_sum_sq))
        border_angle_offsets = np.arccos(np.clip(border_dists / obs_ranges, 0, 1))

        left_borders = obs_angles + border_angle_offsets
        right_borders = obs_angles - border_angle_offsets

        blocked_intervals = np.column_stack(
            (
                np.maximum(right_borders, search_right),
                np.minimum(left_borders, search_left),
            )
        )
        valid_intervals = blocked_intervals[:, 0] < blocked_intervals[:, 1]
        return blocked_intervals[valid_intervals]

    def _score_gap(self, gap_start: float, gap_end: float) -> float:
        gap_width = gap_end - gap_start
        match self.gap_selection_mode:
            case "widest":
                return gap_width
            case "heading_bias":
                gap_center = (gap_start + gap_end) / 2.0
                return gap_width - self.gap_heading_weight * abs(gap_center)
            case _:
                raise ValueError(
                    f"Unknown gap_selection_mode: {self.gap_selection_mode}"
                )

    def _compute_fallback_action(self) -> np.ndarray:
        match self.fallback_mode:
            case "brake_hold":
                return np.asarray([0.0, 0.0])
            case "crawl_straight":
                return np.asarray([self.min_velocity, 0.0])
            case _:
                raise ValueError(f"Unknown fallback_mode: {self.fallback_mode}")

    def _select_gap(self, free_gaps: list[tuple[float, float]]) -> tuple[float, float]:
        scores = np.asarray(
            [self._score_gap(gap_start, gap_end) for gap_start, gap_end in free_gaps],
            dtype=float,
        )
        best_idx = int(np.argmax(scores))
        return free_gaps[best_idx]

    def _compute_target_angle(self, gap_start: float, gap_end: float) -> float:
        target_angle = (gap_start + gap_end) / 2.0
        target_angle = np.arctan2(np.sin(target_angle), np.cos(target_angle))
        return target_angle

    def _compute_speed(
        self, gap_width: float, target_angle: float, min_obs_range: float
    ) -> float:
        gap_openness = np.clip(gap_width / (2 * self.lidar_angle), 0, 1)
        danger_proximity = 1.0 - np.clip(min_obs_range / self.obstacle_max_val, 0, 1)

        match self.speed_mode:
            case "nonlinear_gap_speed":
                speed = (
                    self.max_velocity
                    * np.sqrt(gap_openness)
                    * (1.0 - 0.3 * danger_proximity**2)
                )
            case "linear_gap_speed":
                speed = self.max_velocity * (
                    0.85 * gap_openness + 0.15 * (1 - danger_proximity)
                )
            case _:
                raise ValueError(f"Unknown speed_mode: {self.speed_mode}")

        turn_factor = 1.0 - np.clip(abs(target_angle) / self.turn_angle, 0, 1)
        speed *= 0.6 + 0.4 * turn_factor
        return float(np.clip(speed, self.min_velocity, self.max_velocity))

    def _apply_steering_smoothing(self, target_angle: float) -> float:
        if not self.use_steering_smoothing:
            self.prev_target_angle = float(target_angle)
            return float(target_angle)

        alpha = float(np.clip(self.steering_smoothing_alpha, 0.0, 1.0))
        smoothed = alpha * target_angle + (1.0 - alpha) * self.prev_target_angle
        self.prev_target_angle = float(smoothed)
        return float(smoothed)

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
        """Select action using staged Follow-The-Gap logic."""
        state_array = np.asarray(state, dtype=float).reshape(-1)
        search_left = self.lidar_angle
        search_right = -self.lidar_angle

        lidar_ranges = self._extract_lidar_ranges(state_array)
        num_rays = len(lidar_ranges)

        if num_rays == 0:
            return self._compute_fallback_action()

        angles = self._compute_ray_angles(num_rays)
        lidar_ranges = self._preprocess_lidar(lidar_ranges)
        lidar_ranges = self._apply_disparity_extension(lidar_ranges, angles)

        is_obstacle = self._identify_obstacles(lidar_ranges)
        if not np.any(is_obstacle):
            return np.asarray([self.max_velocity, 0.0])

        obs_angles = angles[is_obstacle]
        obs_ranges = lidar_ranges[is_obstacle]

        blocked_intervals = self._compute_blocked_intervals(
            obs_angles,
            obs_ranges,
            search_left,
            search_right,
        )

        merged_intervals = self._merge_blocked_intervals(blocked_intervals)
        free_gaps = self._compute_free_gaps(merged_intervals, search_left, search_right)

        if not free_gaps:
            return self._compute_fallback_action()

        gap_start, gap_end = self._select_gap(free_gaps)
        gap_width = gap_end - gap_start

        if gap_width <= 0:
            return self._compute_fallback_action()

        target_angle = self._compute_target_angle(gap_start, gap_end)
        target_angle = self._apply_steering_smoothing(target_angle)
        min_obs_range = np.min(obs_ranges)
        speed = self._compute_speed(gap_width, target_angle, min_obs_range)

        target_angle = np.clip(target_angle, -self.turn_angle, self.turn_angle)

        return np.asarray([speed, target_angle])


if __name__ == "__main__":
    main()
