"""Deterministic centreline-relative spawn geometry."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class SpawnPose:
    """World pose used by the evaluation-only environment reset seam."""

    x: float
    y: float
    yaw: float
    z: float = 0.0
    waypoint_index: int = 0
    lateral_offset_m: float = 0.0

    def as_reset_dict(self) -> dict[str, float | int]:
        return asdict(self)


def centreline_spawn_pose(
    waypoint, lateral_offset_m: float = 0.0
) -> SpawnPose:
    """Offset a waypoint along its left-pointing track normal."""
    x, y, yaw, waypoint_index = waypoint
    offset = float(lateral_offset_m)
    return SpawnPose(
        x=float(x) - math.sin(float(yaw)) * offset,
        y=float(y) + math.cos(float(yaw)) * offset,
        yaw=float(yaw),
        waypoint_index=int(waypoint_index),
        lateral_offset_m=offset,
    )


def side_by_side_spawn_poses(
    waypoint,
    *,
    left_agent: str,
    right_agent: str,
    lateral_offset_m: float,
) -> dict[str, dict[str, float | int]]:
    """Return equal and opposite poses at identical centreline progress."""
    if left_agent == right_agent:
        raise ValueError("left_agent and right_agent must be different")
    if lateral_offset_m <= 0.0:
        raise ValueError("lateral_offset_m must be positive")

    return {
        left_agent: centreline_spawn_pose(
            waypoint, lateral_offset_m
        ).as_reset_dict(),
        right_agent: centreline_spawn_pose(
            waypoint, -lateral_offset_m
        ).as_reset_dict(),
    }
