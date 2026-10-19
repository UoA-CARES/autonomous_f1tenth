"""Pure helpers for reproducible F1TENTH policy benchmarks."""

from .geometry import (
    SpawnPose,
    centreline_spawn_pose,
    seeded_waypoint,
    seeded_waypoint_index,
    staggered_spawn_poses,
)
from .monitor import LapMonitor, LapMonitorConfig, LapUpdate
from .protocol import (
    CheckpointRef,
    CrashAssessment,
    FinishResolution,
    HeatSpec,
    assess_crash,
    build_balanced_heats,
    resolve_finish_step,
)

__all__ = [
    "CheckpointRef",
    "CrashAssessment",
    "FinishResolution",
    "HeatSpec",
    "LapMonitor",
    "LapMonitorConfig",
    "LapUpdate",
    "SpawnPose",
    "assess_crash",
    "build_balanced_heats",
    "centreline_spawn_pose",
    "resolve_finish_step",
    "seeded_waypoint",
    "seeded_waypoint_index",
    "staggered_spawn_poses",
]
