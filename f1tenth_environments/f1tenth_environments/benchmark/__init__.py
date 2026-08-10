"""Pure helpers for reproducible F1TENTH policy benchmarks."""

from .geometry import (
    SpawnPose,
    centreline_spawn_pose,
    side_by_side_spawn_poses,
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
    "side_by_side_spawn_poses",
]
