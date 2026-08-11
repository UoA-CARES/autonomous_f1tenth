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
    longitudinal_offset_m: float = 0.0

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


def staggered_spawn_poses(
    waypoint,
    *,
    lead_agent: str,
    chaser_agent: str,
    longitudinal_separation_m: float,
) -> dict[str, dict[str, float | int]]:
    """Place two cars in one lane, equally around a common start gate."""
    if lead_agent == chaser_agent:
        raise ValueError("lead_agent and chaser_agent must be different")
    separation = float(longitudinal_separation_m)
    if separation <= 0.0:
        raise ValueError("longitudinal_separation_m must be positive")

    x, y, yaw, waypoint_index = waypoint
    half_gap = separation / 2.0

    def pose(offset: float) -> dict[str, float | int]:
        return SpawnPose(
            x=float(x) + math.cos(float(yaw)) * offset,
            y=float(y) + math.sin(float(yaw)) * offset,
            yaw=float(yaw),
            waypoint_index=int(waypoint_index),
            longitudinal_offset_m=offset,
        ).as_reset_dict()

    return {lead_agent: pose(half_gap), chaser_agent: pose(-half_gap)}
