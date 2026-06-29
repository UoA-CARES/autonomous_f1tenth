from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import math

import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from . import geometry_utils
from .lidar_processor import LidarProcessor

OdomMode = Literal["velocity_only", "orientation_velocity"]
LidarMode = Literal["processed", "raw"]

ODOM_STATE_SIZES: dict[OdomMode, int] = {
    "velocity_only": 2,
    "orientation_velocity": 4,
}


def calculate_motion_state_bounds(
    min_speed: float,
    max_speed: float,
    max_turn: float,
    wheelbase_m: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return odom normalization bounds derived from symmetric action limits."""
    min_speed = float(min_speed)
    max_speed = float(max_speed)
    max_turn = abs(float(max_turn))

    linear_velocity_bounds = (min_speed, max_speed)

    max_abs_speed = max(abs(min_speed), abs(max_speed))
    max_abs_angular_velocity = abs(
        geometry_utils.ackermann_to_twist(max_turn, max_abs_speed, wheelbase_m)
    )

    if np.isclose(max_abs_angular_velocity, 0.0):
        max_abs_angular_velocity = 1.0

    angular_velocity_bounds = (-max_abs_angular_velocity, max_abs_angular_velocity)
    return linear_velocity_bounds, angular_velocity_bounds


@dataclass(slots=True)
class StateData:
    """Bundle of observation-side data derived from synced ROS messages.

    This is the full observation-side result object returned by the builder.
    It stores structured odometry plus multiple lidar representations so the
    environment and reward logic can use whichever view they need.
    """

    odometry: Odometry
    laser_scan: LaserScan

    lidar_sanitised_data: np.ndarray

    state: np.ndarray
    lidar_state_scan: LaserScan

    def position_xy(self) -> tuple[float, float]:
        """Return odometry position as (x, y)."""
        position = self.odometry.pose.pose.position
        return float(position.x), float(position.y)

    def linear_velocity(self) -> float:
        """Return odometry linear velocity (x)."""
        return float(self.odometry.twist.twist.linear.x)

    def angular_velocity(self) -> float:
        """Return odometry angular velocity (z)."""
        return float(self.odometry.twist.twist.angular.z)

    def quaternion_wxyz(self) -> list[float]:
        """Return odometry orientation quaternion in (w, x, y, z) order."""
        orientation = self.odometry.pose.pose.orientation
        return [
            float(orientation.w),
            float(orientation.x),
            float(orientation.y),
            float(orientation.z),
        ]


class StateBuilder:
    """Construct policy observations from synced odometry and lidar messages.

    The builder centralises the lidar-mode branching that is currently spread
    across environment code:
    - raw lidar mode -> full scan sanitise + normalise
    - processed lidar mode -> sector-compressed lidar state

    It also prepares a sanitised lidar array for reward/termination logic.
    """

    def __init__(
        self,
        odom_mode: OdomMode,
        lidar_mode: LidarMode,
        lidar_state_size: int,
        min_speed: float,
        max_speed: float,
        max_turn: float,
        wheelbase_m: float,
        forward_half_angle: float | None = 45.0,
        n_forward: int = 5,
        k_fraction: float = 0.15,
        k_floor: int = 2,
        k_cap: int = 5,
    ) -> None:

        self.odom_mode = odom_mode
        self.lidar_mode = lidar_mode

        if self.odom_mode not in ODOM_STATE_SIZES:
            raise ValueError(
                f"Unsupported odom_mode '{self.odom_mode}'. "
                f"Supported modes: {list(ODOM_STATE_SIZES.keys())}"
            )

        self.linear_velocity_bounds, self.angular_velocity_bounds = (
            calculate_motion_state_bounds(
                min_speed=min_speed,
                max_speed=max_speed,
                max_turn=max_turn,
                wheelbase_m=wheelbase_m,
            )
        )

        self._validate_bounds(self.linear_velocity_bounds, "linear_velocity_bounds")
        self._validate_bounds(self.angular_velocity_bounds, "angular_velocity_bounds")

        self.lidar_state_size = lidar_state_size

        self.lidar_processor = LidarProcessor(
            num_points=lidar_state_size,
            forward_half_angle=forward_half_angle,
            n_forward=n_forward,
            k_fraction=k_fraction,
            k_floor=k_floor,
            k_cap=k_cap,
        )

    @property
    def odom_state_size(self) -> int:
        return ODOM_STATE_SIZES[self.odom_mode]

    @property
    def policy_state_size(self) -> int:
        return self.odom_state_size + self.lidar_state_size

    def build_state(
        self,
        odom_msg: Odometry,
        lidar_scan: LaserScan,
    ) -> StateData:
        """Build all observation-side outputs from one synced odom/lidar pair."""
        odom_state = self._build_odom_state(odom_msg)

        lidar_state, lidar_state_scan = self._build_lidar_state(lidar_scan)

        lidar_sanitised_data = self.lidar_processor.sanitize_lidar(
            lidar_scan,
            invalid_value=float(lidar_scan.range_max),
        )

        state = np.concatenate([odom_state, lidar_state]).astype(np.float32, copy=False)

        return StateData(
            odometry=odom_msg,
            laser_scan=lidar_scan,
            state=state,
            lidar_sanitised_data=lidar_sanitised_data,
            lidar_state_scan=lidar_state_scan,
        )

    def _build_odom_state(self, odom_msg: Odometry) -> np.ndarray:
        """Convert ROS odometry into normalized [0, 1] policy odometry features."""
        orientation = odom_msg.pose.pose.orientation
        twist = odom_msg.twist.twist

        linear_velocity: Final = float(twist.linear.x)
        angular_velocity: Final = float(twist.angular.z)

        linear_velocity_norm = self._normalize_to_unit_interval(
            linear_velocity,
            self.linear_velocity_bounds,
        )
        angular_velocity_norm = self._normalize_to_unit_interval(
            angular_velocity,
            self.angular_velocity_bounds,
        )

        if self.odom_mode == "velocity_only":
            return np.asarray(
                [linear_velocity_norm, angular_velocity_norm],
                dtype=np.float32,
            )

        yaw = self._yaw_from_quaternion_xyzw(
            x=float(orientation.x),
            y=float(orientation.y),
            z=float(orientation.z),
            w=float(orientation.w),
        )
        yaw_sin_norm = 0.5 * (math.sin(yaw) + 1.0)
        yaw_cos_norm = 0.5 * (math.cos(yaw) + 1.0)

        return np.asarray(
            [
                yaw_sin_norm,
                yaw_cos_norm,
                linear_velocity_norm,
                angular_velocity_norm,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _validate_bounds(bounds: tuple[float, float], bounds_name: str) -> None:
        lower, upper = bounds
        if upper <= lower:
            raise ValueError(
                f"Invalid {bounds_name}: lower ({lower}) must be < upper ({upper})."
            )

    @staticmethod
    def _normalize_to_unit_interval(value: float, bounds: tuple[float, float]) -> float:
        """Min-max normalize and clip to [0, 1]."""
        lower, upper = bounds
        return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))

    @staticmethod
    def _yaw_from_quaternion_xyzw(x: float, y: float, z: float, w: float) -> float:
        """Extract yaw angle (rad) from quaternion in xyzw order."""
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _build_lidar_state(
        self,
        lidar_scan: LaserScan,
    ) -> tuple[np.ndarray, LaserScan]:
        """Convert a raw scan into policy lidar state and its visualisation scan."""
        if self.lidar_mode == "raw":
            return (
                self.lidar_processor.normalize_lidar(
                    lidar_scan,
                    invalid_value=-1.0,
                ),
                lidar_scan,
            )

        lidar_state = self.lidar_processor.lidar_to_state(lidar_scan)
        state_visualisation_scan = self.lidar_processor.state_to_laserscan(
            lidar_state,
            lidar_scan,
        )
        return lidar_state, state_visualisation_scan
