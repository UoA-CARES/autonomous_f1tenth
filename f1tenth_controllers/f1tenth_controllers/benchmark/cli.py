"""Command-line entry point for reproducible MARL Gazebo evaluation."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.parameter import Parameter

import cares_reinforcement_learning
from f1tenth_environments import EnvironmentFactory
from f1tenth_environments.benchmark import (
    CheckpointRef,
    build_balanced_heats,
    centreline_spawn_pose,
    side_by_side_spawn_poses,
)

from .config import (
    EXPECTED_ALGORITHMS,
    load_experiment_config,
    resolve_checkpoint_specs,
    resolve_runtime_config,
)
from .manifest import (
    build_run_manifest,
    git_revisions,
    git_worktree_state,
)
from .policy import preflight_policies
from .results import ResultWriter
from .runner import BenchmarkRunner, build_trial_id


@dataclass(frozen=True, slots=True)
class TrialSpec:
    trial_id: str
    algorithm: str
    seed: int
    repetition: int


def _source_config_path() -> Path:
    source_path = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "marl_benchmark.json"
    )
    if source_path.is_file():
        return source_path
    return (
        Path(get_package_share_directory("f1tenth_controllers"))
        / "config"
        / "marl_benchmark.json"
    )


def _default_checkpoint_root() -> Path:
    configured = os.environ.get("F1TENTH_CHECKPOINT_DIR")
    if configured:
        return Path(configured).expanduser()
    source_root = Path(__file__).resolve().parents[3]
    source_checkpoints = source_root / "train_weights"
    if source_checkpoints.is_dir():
        return source_checkpoints
    package_share = Path(
        get_package_share_directory("f1tenth_controllers")
    )
    workspace_root = package_share.parents[3]
    return (
        workspace_root
        / "src"
        / "autonomous_f1tenth"
        / "train_weights"
    )


def build_trial_schedule(config: dict, policies: dict) -> list[TrialSpec]:
    seeds = [int(seed) for seed in config["time_trials"]["seeds"]]
    expected_repetitions = int(
        config["time_trials"]["trials_per_algorithm"]
    )
    if len(seeds) != expected_repetitions:
        raise ValueError(
            "time_trials.seeds count must match trials_per_algorithm"
        )

    trials = []
    for algorithm in sorted(policies):
        policy = policies[algorithm]
        for repetition, seed in enumerate(seeds):
            trials.append(
                TrialSpec(
                    trial_id=build_trial_id(
                        algorithm=algorithm,
                        checkpoint_sha256=policy.spec.sha256,
                        track=config["track"]["identifier"],
                        seed=seed,
                        repetition=repetition,
                        config_id=config["config_sha256"],
                    ),
                    algorithm=algorithm,
                    seed=seed,
                    repetition=repetition,
                )
            )
    if len({trial.trial_id for trial in trials}) != len(trials):
        raise RuntimeError("Generated duplicate time-trial IDs")
    return trials


def build_heat_schedule(config: dict, policies: dict):
    checkpoints = [
        CheckpointRef(
            algorithm=algorithm,
            filename=policies[algorithm].spec.filename,
            sha256=policies[algorithm].spec.sha256,
        )
        for algorithm in sorted(policies)
    ]
    return build_balanced_heats(
        checkpoints,
        seeds=[int(seed) for seed in config["race"]["seeds"]],
        track=config["track"]["identifier"],
        direction=config["track"]["direction"],
    )


def pilot_trials(trials: list[TrialSpec]) -> list[TrialSpec]:
    selected = {}
    for trial in trials:
        selected.setdefault(trial.algorithm, trial)
    return list(selected.values())


def pilot_heats(heats: list) -> list:
    selected = {}
    for heat in heats:
        pairing = (heat.algorithm_a, heat.algorithm_b)
        selected.setdefault(pairing, heat)
    return list(selected.values())


def environment_factory_config(config: dict) -> dict:
    values = config["environment"]
    factory_keys = {
        "max_steps",
        "step_sleep_time_ms",
        "command_latency_ms",
        "train_eval_split",
        "odom_mode",
        "lidar_mode",
        "lidar_state_size",
        "max_speed",
        "min_speed",
        "max_turn",
        "wall_proximity_reward_weight",
        "turn_reward_weight",
        "collision_penalty",
        "collision_range_m",
        "goal_reach_radius_m",
        "stall_progress_threshold_m",
        "stall_limit_steps",
        "position_speed_multiplier",
    }
    return {
        **{key: values[key] for key in factory_keys},
        "track": config["track"]["world"],
    }


def validate_runtime_environment(config: dict) -> None:
    expected_domain = int(config["runtime"]["ros_domain_id"])
    actual_domain = int(os.environ.get("ROS_DOMAIN_ID", "0"))
    if actual_domain != expected_domain:
        raise ValueError(
            f"Benchmark requires ROS_DOMAIN_ID={expected_domain}; "
            f"current value is {actual_domain}"
        )


def enable_simulator_time(environment) -> None:
    """Make the externally launched evaluation node consume Gazebo /clock."""
    results = environment.set_parameters(
        [Parameter("use_sim_time", Parameter.Type.BOOL, True)]
    )
    if not results or not all(result.successful for result in results):
        reasons = [result.reason for result in results]
        raise RuntimeError(
            f"Could not enable ROS simulator time: {reasons}"
        )


def validate_environment(
    environment,
    config: dict,
    policies: dict,
    *,
    expected_agent_count: int,
) -> None:
    """Fail before reset if runtime construction differs from the manifest."""
    expected = config["environment"]
    errors = []

    def equal(name: str, actual, configured) -> None:
        if isinstance(configured, float):
            matches = bool(np.isclose(actual, configured))
        else:
            matches = actual == configured
        if not matches:
            errors.append(
                f"{name}: runtime={actual!r}, configured={configured!r}"
            )

    equal("agent_count", len(environment.agents), expected_agent_count)
    equal("max_steps", environment.max_steps, expected["max_steps"])
    equal(
        "step_sleep_time_ms",
        environment.step_sleep_time_ms,
        expected["step_sleep_time_ms"],
    )
    equal(
        "command_latency_ms",
        environment.command_latency_ms,
        expected["command_latency_ms"],
    )
    equal(
        "collision_range_m",
        environment.collision_range_m,
        expected["collision_range_m"],
    )
    equal(
        "stall_progress_threshold_m",
        environment.stall_progress_threshold_m,
        expected["stall_progress_threshold_m"],
    )
    equal(
        "stall_limit_steps",
        environment.stall_limit_steps,
        expected["stall_limit_steps"],
    )
    equal(
        "position_speed_multiplier",
        environment.position_speed_multiplier,
        expected["position_speed_multiplier"],
    )
    equal("wheelbase_m", environment.wheelbase_m, expected["wheelbase_m"])
    equal(
        "odom_mode",
        environment.state_builder.odom_mode,
        expected["odom_mode"],
    )
    equal(
        "lidar_mode",
        environment.state_builder.lidar_mode,
        expected["lidar_mode"],
    )
    equal(
        "lidar_state_size",
        environment.state_builder.lidar_state_size,
        expected["lidar_state_size"],
    )
    equal(
        "minimum_speed",
        float(environment.min_actions[0]),
        expected["min_speed"],
    )
    equal(
        "maximum_speed",
        float(environment.max_actions[0]),
        expected["max_speed"],
    )
    equal(
        "maximum_turn",
        float(environment.max_actions[1]),
        expected["max_turn"],
    )

    track_name = config["track"]["identifier"]
    if track_name not in environment.tracks:
        errors.append(
            f"track {track_name!r} not in runtime tracks "
            f"{sorted(environment.tracks)}"
        )
    observation_size = environment.state_builder.policy_state_size
    for algorithm, policy in policies.items():
        if policy.observation_size != observation_size:
            errors.append(
                f"{algorithm} observation size {policy.observation_size} "
                f"does not match environment {observation_size}"
            )
        if policy.action_size != 2:
            errors.append(
                f"{algorithm} action size {policy.action_size} is not 2"
            )

    if errors:
        raise ValueError(
            "Benchmark environment fidelity validation failed:\n- "
            + "\n- ".join(errors)
        )


def _repository_states(checkpoint_root: Path) -> dict[str, dict]:
    autonomous_root = checkpoint_root.resolve().parent
    cares_root = Path(cares_reinforcement_learning.__file__).resolve().parent.parent
    f1_root = autonomous_root.parent / "f1tenth"
    return {
        "autonomous_f1tenth": git_worktree_state(autonomous_root),
        "cares_reinforcement_learning": git_worktree_state(cares_root),
        "f1tenth": git_worktree_state(f1_root),
    }


def _resolved_geometry(environment, config: dict) -> tuple[dict, dict]:
    track_name = config["track"]["identifier"]
    waypoint_index = int(config["track"]["start_waypoint_index"])
    waypoints = environment.tracks[track_name]
    if not 0 <= waypoint_index < len(waypoints):
        raise ValueError(
            f"Start waypoint {waypoint_index} outside {track_name}"
        )
    waypoint = waypoints[waypoint_index]
    centre = centreline_spawn_pose(waypoint).as_reset_dict()
    sides = side_by_side_spawn_poses(
        waypoint,
        left_agent="left",
        right_agent="right",
        lateral_offset_m=float(config["track"]["lateral_offset_m"]),
    )
    return centre, sides


def _selected_algorithms(arguments) -> set[str]:
    if not arguments.algorithm:
        return set(EXPECTED_ALGORITHMS)
    return set(arguments.algorithm)


def _run_time_trials(arguments, runner, writer, all_trials) -> None:
    trials = pilot_trials(all_trials) if arguments.pilot else all_trials
    selected_algorithms = _selected_algorithms(arguments)
    trials = [
        trial for trial in trials if trial.algorithm in selected_algorithms
    ]
    for index, trial in enumerate(trials, start=1):
        if writer.has_time_trial(trial.trial_id):
            print(f"[{index}/{len(trials)}] skip existing {trial.trial_id}")
            continue
        print(
            f"[{index}/{len(trials)}] time trial "
            f"{trial.algorithm} seed={trial.seed}"
        )
        runner.run_time_trial(
            trial.algorithm,
            seed=trial.seed,
            repetition=trial.repetition,
        )


def _run_head_to_head(arguments, runner, writer, all_heats) -> None:
    heats = pilot_heats(all_heats) if arguments.pilot else all_heats
    selected_algorithms = _selected_algorithms(arguments)
    heats = [
        heat
        for heat in heats
        if {
            heat.algorithm_a,
            heat.algorithm_b,
        }.issubset(selected_algorithms)
    ]
    if arguments.heat_id:
        selected_heat_ids = set(arguments.heat_id)
        unknown = selected_heat_ids - {heat.heat_id for heat in all_heats}
        if unknown:
            raise ValueError(f"Unknown heat IDs: {sorted(unknown)}")
        heats = [heat for heat in heats if heat.heat_id in selected_heat_ids]

    for index, heat in enumerate(heats, start=1):
        if writer.has_head_to_head(heat.heat_id):
            print(f"[{index}/{len(heats)}] skip existing {heat.heat_id}")
            continue
        print(
            f"[{index}/{len(heats)}] {heat.algorithm_a} vs "
            f"{heat.algorithm_b} seed={heat.seed} heat={heat.heat_id}"
        )
        runner.run_head_to_head(heat)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the six selected F1TENTH MARL checkpoints"
    )
    parser.add_argument(
        "mode",
        choices=("preflight", "time-trials", "head-to-head"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_source_config_path(),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=_default_checkpoint_root(),
    )
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--algorithm",
        action="append",
        choices=sorted(EXPECTED_ALGORITHMS),
    )
    parser.add_argument("--heat-id", action="append")
    return parser


def main(argv=None) -> None:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    config = load_experiment_config(arguments.config)
    specs = resolve_checkpoint_specs(config)
    policies = preflight_policies(specs, arguments.checkpoint_dir)

    if arguments.mode == "preflight":
        print(
            json.dumps(
                {
                    "config_id": config["config_sha256"],
                    "checkpoints": {
                        algorithm: policy.manifest_entry()
                        for algorithm, policy in sorted(policies.items())
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if arguments.result_dir is None:
        parser.error("--result-dir is required for simulator runs")

    config = resolve_runtime_config(config, pilot=arguments.pilot)
    validate_runtime_environment(config)
    expected_agents = 1 if arguments.mode == "time-trials" else 2
    environment = None
    try:
        environment = EnvironmentFactory().create(
            "MultiCarRace",
            config=environment_factory_config(config),
        )
        environment.evaluation_service_timeout_s = float(
            config["runtime"][
                "evaluation_service_timeout_wall_seconds"
            ]
        )
        enable_simulator_time(environment)
        validate_environment(
            environment,
            config,
            policies,
            expected_agent_count=expected_agents,
        )
        all_trials = build_trial_schedule(config, policies)
        all_heats = build_heat_schedule(config, policies)
        centre_pose, side_poses = _resolved_geometry(environment, config)
        repository_states = _repository_states(arguments.checkpoint_dir)
        track_model = environment.track_progress_models[
            config["track"]["identifier"]
        ]
        manifest = build_run_manifest(
            config=config,
            policies=policies,
            repository_states=repository_states,
            lap_length_m=track_model.waypoint_lap_length,
            time_trial_ids=[trial.trial_id for trial in all_trials],
            heat_ids=[heat.heat_id for heat in all_heats],
            centreline_start_pose=centre_pose,
            side_by_side_start_poses=side_poses,
        )
        writer = ResultWriter(arguments.result_dir)
        manifest = writer.write_manifest(manifest)
        revisions = git_revisions(repository_states)
        runner = BenchmarkRunner(
            environment=environment,
            policies=policies,
            config=config,
            result_writer=writer,
            manifest_id=manifest["manifest_id"],
            git_revisions=revisions,
        )
        writer.write_event(
            {
                "event": "benchmark_invocation_start",
                "mode": arguments.mode,
                "pilot": arguments.pilot,
                "manifest_id": manifest["manifest_id"],
            }
        )
        if arguments.mode == "time-trials":
            _run_time_trials(arguments, runner, writer, all_trials)
        else:
            _run_head_to_head(arguments, runner, writer, all_heats)
        writer.write_event(
            {
                "event": "benchmark_invocation_complete",
                "mode": arguments.mode,
                "pilot": arguments.pilot,
                "manifest_id": manifest["manifest_id"],
            }
        )
    finally:
        if environment is not None:
            if rclpy.ok():
                environment._stop_all_agents()
            environment.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
