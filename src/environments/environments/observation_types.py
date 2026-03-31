from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from nav_msgs.msg import Odometry

ObservationMode = Literal["lidar_only", "no_position", "full_state"]


@dataclass(slots=True)
class OdomState:
    x: float
    y: float
    qw: float
    qx: float
    qy: float
    qz: float
    linear_velocity: float
    angular_velocity: float

    @classmethod
    def from_odometry(cls, odom_msg: Odometry) -> "OdomState":
        pose = odom_msg.pose.pose
        twist = odom_msg.twist.twist
        return cls(
            x=float(pose.position.x),
            y=float(pose.position.y),
            qw=float(pose.orientation.w),
            qx=float(pose.orientation.x),
            qy=float(pose.orientation.y),
            qz=float(pose.orientation.z),
            linear_velocity=float(twist.linear.x),
            angular_velocity=float(twist.angular.z),
        )

    def as_full_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.x,
                self.y,
                self.qw,
                self.qx,
                self.qy,
                self.qz,
                self.linear_velocity,
                self.angular_velocity,
            ],
            dtype=np.float32,
        )

    def quaternion_wxyz(self) -> list[float]:
        return [self.qw, self.qx, self.qy, self.qz]

    def quaternion_xyzw(self) -> list[float]:
        return [self.qx, self.qy, self.qz, self.qw]


@dataclass(slots=True)
class Observation:
    odom: OdomState
    lidar: np.ndarray

    def to_policy_array(self, mode: ObservationMode) -> np.ndarray:
        lidar = np.asarray(self.lidar, dtype=np.float32)

        if mode == "lidar_only":
            odom_part = np.asarray(
                [self.odom.linear_velocity, self.odom.angular_velocity],
                dtype=np.float32,
            )
        elif mode == "no_position":
            odom_part = np.asarray(
                [
                    self.odom.qw,
                    self.odom.qx,
                    self.odom.qy,
                    self.odom.qz,
                    self.odom.linear_velocity,
                    self.odom.angular_velocity,
                ],
                dtype=np.float32,
            )
        else:
            odom_part = self.odom.as_full_array()

        return np.concatenate([odom_part, lidar]).astype(np.float32, copy=False)

    def to_full_state_array(self) -> np.ndarray:
        return np.concatenate(
            [self.odom.as_full_array(), np.asarray(self.lidar, dtype=np.float32)]
        )
