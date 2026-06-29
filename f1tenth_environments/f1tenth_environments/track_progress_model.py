import numpy as np
from scipy.interpolate import splev
from scipy import integrate, optimize, interpolate


class TrackProgressModel:
    """Parametric closed-track geometry and progress helper.

    Provides spline projection, coordinate evaluation, and arc-distance helpers
    for closed-loop track geometry.

    Conventions:
    - Spline coordinate `spline_coord` is normalized to `[0, 1)`.
    - World XY vectors are named `world_coord_xy`.
    - Positive signed distance means "forward along track direction".
    - Signed distance always chooses the shorter direction around the loop.
    """

    def __init__(self, points: list) -> None:
        """Fit a closed parametric spline from ordered track centerline points."""
        if len(points) < 2:
            raise ValueError("At least two points are required to fit a track spline.")

        points_arr = np.asarray(points, dtype=np.float64)
        self.points = points_arr
        self.next_points = np.roll(points_arr, -1, axis=0)
        self.segment_vectors = self.next_points - self.points
        self.segment_lengths = np.linalg.norm(self.segment_vectors, axis=1)
        self.segment_lengths = np.where(
            self.segment_lengths > 1e-9,
            self.segment_lengths,
            1e-9,
        )
        self.cumulative_segment_lengths = np.concatenate(
            ([0.0], np.cumsum(self.segment_lengths))
        )
        self.waypoint_lap_length = float(self.cumulative_segment_lengths[-1])

        closed_points = np.append(points_arr, [points_arr[0]], axis=0)

        x_points = closed_points[:, 0]
        y_points = closed_points[:, 1]
        spline_param_grid = np.linspace(0.0, 1.0, len(x_points))

        self.spline_x_tck = interpolate.splrep(spline_param_grid, x_points, k=2)
        self.spline_y_tck = interpolate.splrep(spline_param_grid, y_points, k=2)
        self._lap_length_cache = None

    def track_distance_from_world_coord(self, world_coord_xy: np.ndarray) -> float:
        """Project a world coordinate onto the waypoint polyline and return arc distance."""
        point = np.asarray(world_coord_xy, dtype=np.float64)
        point_vectors = point - self.points
        segment_length_sq = self.segment_lengths * self.segment_lengths
        segment_fractions = np.sum(point_vectors * self.segment_vectors, axis=1)
        segment_fractions = np.clip(segment_fractions / segment_length_sq, 0.0, 1.0)
        projections = self.points + segment_fractions[:, None] * self.segment_vectors
        distances_sq = np.sum((projections - point) ** 2, axis=1)
        segment_index = int(np.argmin(distances_sq))
        return float(
            (
                self.cumulative_segment_lengths[segment_index]
                + segment_fractions[segment_index] * self.segment_lengths[segment_index]
            )
            % self.waypoint_lap_length
        )

    def forward_distance_between_track_distances(
        self,
        from_track_distance: float,
        to_track_distance: float,
    ) -> float:
        """Forward arc distance between wrapped waypoint-arc distances."""
        return float(
            (to_track_distance - from_track_distance) % self.waypoint_lap_length
        )

    def signed_delta_between_track_distances(
        self,
        from_track_distance: float,
        to_track_distance: float,
    ) -> float:
        """Shortest signed delta between wrapped waypoint-arc distances."""
        forward_delta = self.forward_distance_between_track_distances(
            from_track_distance,
            to_track_distance,
        )
        if forward_delta <= self.waypoint_lap_length / 2.0:
            return forward_delta
        return float(forward_delta - self.waypoint_lap_length)

    def signed_delta_between_world_coords(
        self,
        from_world_coord_xy: np.ndarray,
        to_world_coord_xy: np.ndarray,
    ) -> float:
        """Project world coordinates to the waypoint arc and return signed progress."""
        from_track_distance = self.track_distance_from_world_coord(from_world_coord_xy)
        to_track_distance = self.track_distance_from_world_coord(to_world_coord_xy)
        return self.signed_delta_between_track_distances(
            from_track_distance,
            to_track_distance,
        )

    @staticmethod
    def _normalize_spline_coord(spline_coord: float) -> float:
        """Wrap any scalar spline-coordinate value into the canonical interval [0, 1)."""
        return float(spline_coord) % 1.0

    def distance_from_spline_coord_to_world_coord_xy(
        self,
        spline_coord: float,
        world_coord_xy: np.ndarray,
    ) -> float:
        """Euclidean distance from spline point at `spline_coord` to `world_coord_xy`."""
        spline_world_coord_xy = self.spline_coord_to_world_coord_xy(spline_coord)
        return float(np.linalg.norm(spline_world_coord_xy - world_coord_xy))

    def world_coord_to_spline_coord(self, world_coord_xy: np.ndarray) -> float:
        """Project a world XY coordinate onto the track and return spline coordinate."""
        result = optimize.differential_evolution(
            self.distance_from_spline_coord_to_world_coord_xy,
            bounds=[(0.0, 1.0)],
            args=([world_coord_xy]),
        )
        return float(result.x[0])

    def spline_coord_to_world_coord_xy(self, spline_coord: float) -> np.ndarray:
        """Return the world XY coordinate at a given normalized spline coordinate."""
        normalized_spline_coord = self._normalize_spline_coord(spline_coord)
        x = float(interpolate.splev(normalized_spline_coord, self.spline_x_tck))
        y = float(interpolate.splev(normalized_spline_coord, self.spline_y_tck))
        return np.asarray([x, y], dtype=np.float64)

    def linear_distance_between_spline_coords(
        self,
        from_spline_coord: float,
        to_spline_coord: float,
    ) -> float:
        """Euclidean distance between spline points at two spline coordinates."""
        return float(
            np.linalg.norm(
                self.spline_coord_to_world_coord_xy(from_spline_coord)
                - self.spline_coord_to_world_coord_xy(to_spline_coord)
            )
        )

    def linear_distance_between_world_coords(
        self,
        from_world_coord_xy: np.ndarray,
        to_world_coord_xy: np.ndarray,
    ) -> float:
        """Project world coordinates, then return Euclidean distance between spline points."""
        from_spline_coord = self.world_coord_to_spline_coord(from_world_coord_xy)
        to_spline_coord = self.world_coord_to_spline_coord(to_world_coord_xy)
        return self.linear_distance_between_spline_coords(
            from_spline_coord,
            to_spline_coord,
        )

    def _arc_length_integrand(self, spline_coord: float) -> float:
        """Return local speed magnitude along the spline at `spline_coord`."""
        normalized_spline_coord = self._normalize_spline_coord(spline_coord)
        dxdt = splev(normalized_spline_coord, self.spline_x_tck, der=1)
        dydt = splev(normalized_spline_coord, self.spline_y_tck, der=1)
        return float(np.sqrt(dxdt**2 + dydt**2))

    def _forward_arc_length_unwrapped(
        self,
        from_spline_coord: float,
        to_spline_coord: float,
    ) -> float:
        """Forward arc length on an unwrapped interval where `to_spline_coord >= from_spline_coord`."""
        if to_spline_coord < from_spline_coord:
            raise ValueError(
                "to_spline_coord must be >= from_spline_coord for unwrapped length."
            )
        length, _ = integrate.quad(
            self._arc_length_integrand,
            from_spline_coord,
            to_spline_coord,
        )
        return float(length)

    def _lap_length(self) -> float:
        """Total closed-loop arc length of the track spline."""
        if self._lap_length_cache is None:
            self._lap_length_cache = self._forward_arc_length_unwrapped(0.0, 1.0)
        return float(self._lap_length_cache)

    def _forward_arc_length(
        self,
        from_spline_coord: float,
        to_spline_coord: float,
    ) -> float:
        """Arc length from `from_spline_coord` to `to_spline_coord` moving forward."""
        from_norm = self._normalize_spline_coord(from_spline_coord)
        to_norm = self._normalize_spline_coord(to_spline_coord)

        if from_norm <= to_norm:
            return self._forward_arc_length_unwrapped(from_norm, to_norm)

        length_a = self._forward_arc_length_unwrapped(from_norm, 1.0)
        length_b = self._forward_arc_length_unwrapped(0.0, to_norm)
        return float(length_a + length_b)

    def arc_distance_between_spline_coords(
        self,
        from_spline_coord: float,
        to_spline_coord: float,
        signed: bool = True,
    ) -> float:
        """Shortest arc distance between spline coordinates; signed if requested."""
        forward_fraction = (
            self._normalize_spline_coord(to_spline_coord)
            - self._normalize_spline_coord(from_spline_coord)
        ) % 1.0

        if forward_fraction <= 0.5:
            shortest_distance = self._forward_arc_length(
                from_spline_coord,
                to_spline_coord,
            )
            return shortest_distance

        shortest_distance = self._forward_arc_length(
            from_spline_coord=to_spline_coord,
            to_spline_coord=from_spline_coord,
        )
        if signed:
            return -shortest_distance
        return shortest_distance

    def arc_distance_between_world_coords(
        self,
        from_world_coord_xy: np.ndarray,
        to_world_coord_xy: np.ndarray,
        signed: bool = True,
    ) -> float:
        """Project world coordinates, then return shortest arc distance."""
        from_spline_coord = self.world_coord_to_spline_coord(from_world_coord_xy)
        to_spline_coord = self.world_coord_to_spline_coord(to_world_coord_xy)
        return self.arc_distance_between_spline_coords(
            from_spline_coord,
            to_spline_coord,
            signed=signed,
        )

    def signed_arc_distance_between_spline_coords(
        self,
        from_spline_coord: float,
        to_spline_coord: float,
    ) -> float:
        """Return signed shortest arc distance between two spline coordinates."""
        return self.arc_distance_between_spline_coords(
            from_spline_coord,
            to_spline_coord,
            signed=True,
        )

    def signed_arc_distance_between_world_coords(
        self,
        from_world_coord_xy: np.ndarray,
        to_world_coord_xy: np.ndarray,
    ) -> float:
        """Project world coordinates, then return signed shortest arc distance."""
        return self.arc_distance_between_world_coords(
            from_world_coord_xy,
            to_world_coord_xy,
            signed=True,
        )

    def world_coord_xy_from_spline_distance(
        self,
        origin_spline_coord: float,
        signed_distance: float,
    ) -> np.ndarray:
        """Return world XY coordinate at signed arc distance from `origin_spline_coord`.

        Supports arbitrary signed distances (including multiple laps) by reducing the
        request modulo lap length and solving on a monotonic unwrapped interval.
        """
        origin_norm = self._normalize_spline_coord(origin_spline_coord)

        if np.isclose(signed_distance, 0.0, atol=1e-9):
            return self.spline_coord_to_world_coord_xy(origin_norm)

        lap_length = self._lap_length()
        if lap_length <= 0.0:
            raise RuntimeError("Track lap length must be positive.")

        reduced_distance = float(np.fmod(signed_distance, lap_length))
        if np.isclose(reduced_distance, 0.0, atol=1e-9):
            return self.spline_coord_to_world_coord_xy(origin_norm)

        if reduced_distance > 0.0:

            def solve_target(target: float) -> float:
                return (
                    self._forward_arc_length_unwrapped(origin_norm, target)
                    - reduced_distance
                )

            root = optimize.root_scalar(
                solve_target,
                bracket=(origin_norm, origin_norm + 1.0),
                method="brentq",
            )
        else:
            backward_distance = -reduced_distance

            def solve_target(target: float) -> float:
                return (
                    self._forward_arc_length_unwrapped(target, origin_norm)
                    - backward_distance
                )

            root = optimize.root_scalar(
                solve_target,
                bracket=(origin_norm - 1.0, origin_norm),
                method="brentq",
            )

        if not root.converged:
            raise RuntimeError("Could not solve for coordinate at requested distance.")

        target_spline_coord = float(root.root) % 1.0
        return self.spline_coord_to_world_coord_xy(target_spline_coord)
