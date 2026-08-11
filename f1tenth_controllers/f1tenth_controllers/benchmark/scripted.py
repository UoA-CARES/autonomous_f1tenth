"""Gazebo integration checks using the repository's existing pure pursuit."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from f1tenth_environments import EnvironmentFactory
from f1tenth_environments.benchmark import (
    CheckpointRef,
    build_balanced_heats,
)

from f1tenth_controllers.path_trackers.pure_pursuit import PurePursuit

from .cli import (
    _default_checkpoint_root,
    _repository_states,
    _source_config_path,
    enable_simulator_time,
    environment_factory_config,
    validate_environment,
    validate_runtime_environment,
)
from .config import load_experiment_config
from .manifest import git_revisions
from .results import ResultWriter
from .runner import BenchmarkRunner


@dataclass(frozen=True, slots=True)
class ScriptedSpec:
    filename: str
    sha256: str
    actor_id: str


def _source_sha256() -> str:
    source = Path(inspect.getfile(PurePursuit)).resolve()
    return hashlib.sha256(source.read_bytes()).hexdigest()


def _offset_path(waypoints: list, lateral_offset_m: float) -> np.ndarray:
    points = []
    for waypoint in waypoints:
        x, y, yaw = map(float, waypoint[:3])
        points.append(
            (
                x - math.sin(yaw) * lateral_offset_m,
                y + math.cos(yaw) * lateral_offset_m,
            )
        )
    return np.asarray(points, dtype=np.float64)


class PurePursuitAdapter:
    """Expose unchanged PurePursuit through the runner's synchronous API."""

    def __init__(
        self,
        *,
        environment,
        agent: str,
        waypoints: list,
        speed_mps: float,
        lateral_offset_m: float,
    ) -> None:
        self.environment = environment
        self.agent = agent
        self.speed_mps = float(speed_mps)
        self.controller = PurePursuit(
            path=_offset_path(waypoints, lateral_offset_m),
        )
        self.spec = ScriptedSpec(
            filename="f1tenth_controllers.path_trackers.pure_pursuit",
            sha256=_source_sha256(),
            actor_id=agent,
        )

    def act(self, _observation) -> np.ndarray:
        state_data = self.environment.previous_state_data[self.agent]
        state = np.asarray(
            [
                *state_data.position_xy(),
                *state_data.quaternion_wxyz(),
            ],
            dtype=np.float64,
        )
        action = np.asarray(
            self.controller.select_action(state, None, None),
            dtype=np.float32,
        )
        action[0] = self.speed_mps
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError(f"Pure pursuit returned invalid action {action}")
        return action


def _scripted_heat(config: dict, source_hash: str):
    references = [
        CheckpointRef(
            algorithm="PURE_PURSUIT_FAST",
            filename="pure_pursuit.py@0.8mps",
            sha256=source_hash,
        ),
        CheckpointRef(
            algorithm="PURE_PURSUIT_SLOW",
            filename="pure_pursuit.py@0.5mps",
            sha256=source_hash,
        ),
    ]
    return build_balanced_heats(
        references,
        seeds=[0],
        track=config["track"]["identifier"],
        direction=config["track"]["direction"],
    )[0]


def _manifest(config: dict, repositories: dict, source_hash: str) -> dict:
    return {
        "schema_version": 1,
        "experiment_name": "f1tenth_marl_scripted_integration",
        "config_id": config["config_sha256"],
        "resolved_config": config,
        "track": config["track"],
        "repositories": repositories,
        "controller": {
            "implementation": (
                "f1tenth_controllers.path_trackers.pure_pursuit.PurePursuit"
            ),
            "source_sha256": source_hash,
            "time_trial_speed_mps": 0.8,
            "race_fast_speed_mps": 0.8,
            "race_slow_speed_mps": 0.5,
            "race_lane_offsets_m": [0.3, -0.3],
        },
        "simulation_changes": [],
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Run synchronous Gazebo checks with existing pure pursuit"
    )
    parser.add_argument("mode", choices=("time-trial", "head-to-head"))
    parser.add_argument("--config", type=Path, default=_source_config_path())
    parser.add_argument("--result-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)

    config = load_experiment_config(arguments.config)
    validate_runtime_environment(config)
    environment = None
    try:
        expected_agents = 1 if arguments.mode == "time-trial" else 2
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
            {},
            expected_agent_count=expected_agents,
        )

        repositories = _repository_states(_default_checkpoint_root())
        source_hash = _source_sha256()
        writer = ResultWriter(arguments.result_dir)
        manifest = writer.write_manifest(
            _manifest(config, repositories, source_hash)
        )
        waypoints = environment.tracks[config["track"]["identifier"]]

        if arguments.mode == "time-trial":
            policies = {
                "PURE_PURSUIT": PurePursuitAdapter(
                    environment=environment,
                    agent=environment.car_name,
                    waypoints=waypoints,
                    speed_mps=0.8,
                    lateral_offset_m=0.0,
                )
            }
        else:
            opponent = next(
                agent
                for agent in environment.agents
                if agent != environment.car_name
            )
            policies = {
                "PURE_PURSUIT_FAST": PurePursuitAdapter(
                    environment=environment,
                    agent=environment.car_name,
                    waypoints=waypoints,
                    speed_mps=0.8,
                    lateral_offset_m=0.3,
                ),
                "PURE_PURSUIT_SLOW": PurePursuitAdapter(
                    environment=environment,
                    agent=opponent,
                    waypoints=waypoints,
                    speed_mps=0.5,
                    lateral_offset_m=-0.3,
                ),
            }

        runner = BenchmarkRunner(
            environment=environment,
            policies=policies,
            config=config,
            result_writer=writer,
            manifest_id=manifest["manifest_id"],
            git_revisions=git_revisions(repositories),
        )
        if arguments.mode == "time-trial":
            row = runner.run_time_trial(
                "PURE_PURSUIT",
                seed=0,
                repetition=0,
            )
        else:
            row = runner.run_head_to_head(
                _scripted_heat(config, source_hash)
            )
        print(json.dumps(row, indent=2, sort_keys=True))
    finally:
        if environment is not None:
            if rclpy.ok():
                environment._stop_all_agents()
            environment.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
