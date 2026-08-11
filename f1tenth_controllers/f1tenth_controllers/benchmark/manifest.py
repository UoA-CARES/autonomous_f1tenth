"""Resolved benchmark manifest and repository provenance."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path


def git_worktree_state(path: Path) -> dict:
    """Return the exact revision and dirty state for one required repository."""
    path = Path(path).expanduser().resolve()

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(path), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    root = Path(git("rev-parse", "--show-toplevel")).resolve()
    return {
        "path": str(root),
        "revision": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current") or None,
        "dirty": bool(git("status", "--porcelain")),
    }


def git_revisions(repository_states: dict[str, dict]) -> dict[str, str]:
    return {
        name: state["revision"]
        for name, state in sorted(repository_states.items())
    }


def build_run_manifest(
    *,
    config: dict,
    policies: dict,
    repository_states: dict[str, dict],
    lap_length_m: float,
    time_trial_ids: list[str],
    heat_ids: list[str],
    centreline_start_pose: dict,
    side_by_side_start_poses: dict,
) -> dict:
    """Build the campaign declaration shared by pilot and full invocations."""
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_name": config["experiment_name"],
        "config_id": config["config_sha256"],
        "resolved_config": config,
        "checkpoints": {
            algorithm: policies[algorithm].manifest_entry()
            for algorithm in sorted(policies)
        },
        "repositories": repository_states,
        "git_revisions": git_revisions(repository_states),
        "track_resolution": {
            **config["track"],
            "lap_length_m": float(lap_length_m),
            "time_trial_start_pose": centreline_start_pose,
            "head_to_head_start_poses": side_by_side_start_poses,
        },
        "campaign": {
            "time_trial_ids": time_trial_ids,
            "head_to_head_heat_ids": heat_ids,
            "time_trial_count": len(time_trial_ids),
            "head_to_head_heat_count": len(heat_ids),
        },
        "simulation_fidelity": {
            "gazebo_physics_changed": False,
            "vehicle_or_sensor_changed": False,
            "observation_processing_changed": False,
            "policy_action_postprocessing": "none",
            "action_pipeline": config["environment"]["action_pipeline"],
            "simulator_time_source": "/clock",
            "simulator_seed_supported": config["environment"][
                "simulator_seed_supported"
            ],
            "evaluation_protocol_differences": [
                {
                    "setting": "fixed_test_track_and_exact_spawn_poses",
                    "reason": "reproducible benchmark starts",
                },
                {
                    "setting": "position_speed_multiplier",
                    "training_default": 0.9,
                    "benchmark_value": config["environment"][
                        "position_speed_multiplier"
                    ],
                    "reason": "equal primary/opponent competition limits",
                },
                {
                    "setting": "episode_limit",
                    "training_default_max_steps": 1000,
                    "benchmark_max_steps": config["environment"]["max_steps"],
                    "benchmark_timeout_sim_seconds": config["environment"][
                        "timeout_sim_seconds"
                    ],
                    "reason": (
                        "permit a full selected-track lap with the preserved raw MARL "
                        "action pipeline"
                    ),
                },
                {
                    "setting": "evaluation_service_timeout_wall_seconds",
                    "training_default": None,
                    "benchmark_value": config["runtime"][
                        "evaluation_service_timeout_wall_seconds"
                    ],
                    "reason": (
                        "fail a benchmark instead of hanging when Gazebo "
                        "transport is unavailable"
                    ),
                },
            ],
            "collision_evidence_limitation": (
                "The training LiDAR collision threshold does not identify a "
                "contact body; vehicle involvement and responsibility are "
                "therefore conservative inferences from synchronized pose, "
                "progress, velocity, collision, flip, and termination state."
            ),
            "seed_limitation": (
                "Protocol seeds are recorded and passed through reset, but the "
                "existing Gazebo launch exposes no simulator physics seed."
            ),
        },
    }
