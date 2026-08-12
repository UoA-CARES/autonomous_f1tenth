from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from f1tenth_controllers.benchmark.results import ResultWriter
from f1tenth_controllers.benchmark.runner import BenchmarkRunner
from f1tenth_environments.benchmark.protocol import HeatSpec


@dataclass(frozen=True)
class FakeSpec:
    algorithm: str
    filename: str
    sha256: str
    actor_id: str = "f1tenth"


class ScriptedPolicy:
    def __init__(self, algorithm: str, progress_per_step_m: float) -> None:
        self.spec = FakeSpec(
            algorithm=algorithm,
            filename=f"{algorithm}.pth",
            sha256=algorithm.lower().ljust(64, "0"),
        )
        self.progress_per_step_m = progress_per_step_m
        self.observation_times = []

    def act(self, observation) -> np.ndarray:
        self.observation_times.append(float(observation[0]))
        return np.asarray(
            [self.progress_per_step_m, 0.0], dtype=np.float32
        )


class FakeState:
    def __init__(self, progress_m: float, speed_mps: float) -> None:
        self._progress_m = progress_m
        self._speed_mps = speed_mps
        self.lidar_sanitised_data = np.ones(9, dtype=np.float32)

    def position_xy(self):
        return self._progress_m, 0.0

    def linear_velocity(self) -> float:
        return self._speed_mps

    @staticmethod
    def quaternion_wxyz():
        return 1.0, 0.0, 0.0, 0.0


class FakeTrackModel:
    waypoint_lap_length = 4.0

    @staticmethod
    def track_distance_from_world_coord(position) -> float:
        return float(position[0]) % 4.0


class ScriptedEnvironment:
    car_name = "f1tenth"
    collision_range_m = 0.2
    stall_limit_steps = 5

    @staticmethod
    def get_clock():
        return SimpleNamespace(ros_time_is_active=True)

    def __init__(self, agents, *, crash_step=None, crash_agents=()) -> None:
        self.agents = list(agents)
        self.tracks = {"test_track": [(0.0, 0.0, 0.0, 0)]}
        self.current_track_model = FakeTrackModel()
        self.crash_step = crash_step
        self.crash_agents = set(crash_agents)
        self.crashed = set()
        self.stall_counters = {agent: 0 for agent in self.agents}
        self.stopped_agents = []
        self.reset_options = None
        self.step_counter = 0
        self.last_command_sim_time_s = None
        self.last_observation_sim_time_s = None
        self.progress = {agent: 0.0 for agent in self.agents}
        self.previous_state_data = {
            agent: FakeState(0.0, 0.0) for agent in self.agents
        }

    def _observations(self):
        return {
            agent: np.asarray(
                [float(self.step_counter), self.progress[agent]],
                dtype=np.float32,
            )
            for agent in self.agents
        }

    def reset(self, seed=None, options=None):
        self.reset_options = options
        self.step_counter = 0
        self.progress = {
            agent: float(options["spawn_poses"][agent]["x"])
            for agent in self.agents
        }
        self.crashed = set()
        self.previous_state_data = {
            agent: FakeState(self.progress[agent], 0.0)
            for agent in self.agents
        }
        return self._observations(), {agent: {} for agent in self.agents}

    def step(self, actions):
        self.last_command_sim_time_s = float(self.step_counter)
        self.step_counter += 1
        for agent, action in actions.items():
            increment = float(action[0])
            self.progress[agent] += increment
            self.previous_state_data[agent] = FakeState(
                self.progress[agent], increment
            )
        if self.crash_step == self.step_counter:
            self.crashed.update(self.crash_agents)
        self.last_observation_sim_time_s = float(self.step_counter)
        terminated = bool(self.crashed)
        return (
            self._observations(),
            {agent: 0.0 for agent in self.agents},
            {agent: terminated for agent in self.agents},
            {agent: False for agent in self.agents},
            {agent: {} for agent in self.agents},
        )

    def stop_agent(self, agent):
        self.stopped_agents.append(agent)


def benchmark_config() -> dict:
    return {
        "config_sha256": "config_hash",
        "track": {
            "identifier": "test_track",
            "direction": "counter_clockwise",
            "start_waypoint_selection": "seeded_sha256_modulo",
            "start_waypoint_seed_salt": "test-starts",
            "longitudinal_separation_m": 1.0,
        },
        "lap_monitor": {
            "sector_fractions": [0.25, 0.5, 0.75],
            "max_projection_jump_m": 1.0,
            "max_projection_speed_mps": 5.0,
        },
        "environment": {
            "max_steps": 20,
            "timeout_sim_seconds": 20.0,
        },
        "race": {
            "tie_tolerance_s": 0.001,
            "possible_contact_distance_m": 0.5,
            "rear_end_closing_speed_mps": 0.2,
        },
    }


def make_runner(
    tmp_path: Path,
    environment: ScriptedEnvironment,
    policies: dict[str, ScriptedPolicy],
) -> BenchmarkRunner:
    return BenchmarkRunner(
        environment=environment,
        policies=policies,
        config=benchmark_config(),
        result_writer=ResultWriter(tmp_path),
        manifest_id="manifest_test",
        git_revisions={"autonomous_f1tenth": "abc"},
    )


def make_heat() -> HeatSpec:
    return HeatSpec(
        heat_id="heat_test",
        checkpoint_id_a="FAST",
        algorithm_a="FAST",
        checkpoint_a="FAST.pth",
        checkpoint_sha256_a="fast".ljust(64, "0"),
        checkpoint_id_b="SLOW",
        algorithm_b="SLOW",
        checkpoint_b="SLOW.pth",
        checkpoint_sha256_b="slow".ljust(64, "0"),
        lead_checkpoint_id="FAST",
        lead_algorithm="FAST",
        chaser_checkpoint_id="SLOW",
        chaser_algorithm="SLOW",
        primary_checkpoint_id="FAST",
        primary_algorithm="FAST",
        opponent_checkpoint_id="SLOW",
        opponent_algorithm="SLOW",
        seed=42,
        track="test_track",
        direction="counter_clockwise",
    )


def test_time_trial_uses_first_command_time_and_interpolated_finish(
    tmp_path: Path,
) -> None:
    environment = ScriptedEnvironment(["f1tenth"])
    policy = ScriptedPolicy("FAST", 0.8)
    runner = make_runner(tmp_path, environment, {"FAST": policy})

    row = runner.run_time_trial("FAST", seed=7, repetition=0)

    assert row["completion_status"] == "completed"
    assert row["start_sim_time"] == pytest.approx(0.0)
    assert row["finish_sim_time"] == pytest.approx(5.0)
    assert row["lap_time"] == pytest.approx(5.0)
    assert row["distance_completed_m"] == pytest.approx(4.0)
    assert environment.reset_options["evaluation"] is True
    assert row["start_waypoint_index"] == 0


def test_time_trial_classifies_impossible_world_motion_as_teleport(
    tmp_path: Path,
) -> None:
    environment = ScriptedEnvironment(["f1tenth"])
    policy = ScriptedPolicy("FAST", 7.0)
    runner = make_runner(tmp_path, environment, {"FAST": policy})

    row = runner.run_time_trial("FAST", seed=7, repetition=0)

    assert row["completion_status"] == "dnf"
    assert row["dnf_reason"] == "teleport"
    assert row["teleport"] is True
    assert row["collision"] is False


def test_race_treats_impossible_world_motion_as_a_crash(
    tmp_path: Path,
) -> None:
    environment = ScriptedEnvironment(["f1tenth", "opponent_0"])
    policies = {
        "FAST": ScriptedPolicy("FAST", 7.0),
        "SLOW": ScriptedPolicy("SLOW", 0.5),
    }
    runner = make_runner(tmp_path, environment, policies)

    row = runner.run_head_to_head(make_heat())

    assert row["outcome_type"] == "crash_win"
    assert row["winner_checkpoint_id"] == "SLOW"
    assert row["teleport_checkpoint_ids"] == ["FAST"]
    assert row["crash_participant_checkpoint_ids"] == ["FAST"]
    assert row["dnf_reasons"] == {"FAST": "teleport"}


def test_race_uses_one_observation_snapshot_and_reports_along_track_lead(
    tmp_path: Path,
) -> None:
    environment = ScriptedEnvironment(["f1tenth", "opponent_0"])
    policies = {
        "FAST": ScriptedPolicy("FAST", 0.8),
        "SLOW": ScriptedPolicy("SLOW", 0.5),
    }
    runner = make_runner(tmp_path, environment, policies)

    row = runner.run_head_to_head(make_heat())

    assert row["outcome_type"] == "finish_win"
    assert row["winner_checkpoint_id"] == "FAST"
    assert row["lead_checkpoint_id"] == "FAST"
    assert row["chaser_checkpoint_id"] == "SLOW"
    assert row["start_separation_m"] == pytest.approx(1.0)
    assert row["start_waypoint_index"] == 0
    assert row["lead_m"] == pytest.approx(2.3125)
    assert policies["FAST"].observation_times == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert (
        policies["SLOW"].observation_times
        == policies["FAST"].observation_times
    )
    assert environment.reset_options["spawn_poses"]["f1tenth"][
        "longitudinal_offset_m"
    ] == pytest.approx(0.5)
    event = (
        tmp_path / "events.jsonl"
    ).read_text(encoding="utf-8").splitlines()[-1]
    assert '"monitor_evidence"' in event


def test_single_isolated_crash_produces_crash_win_with_attribution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    environment = ScriptedEnvironment(
        ["f1tenth", "opponent_0"],
        crash_step=2,
        crash_agents={"opponent_0"},
    )
    policies = {
        "FAST": ScriptedPolicy("FAST", 0.8),
        "SLOW": ScriptedPolicy("SLOW", 0.5),
    }
    runner = make_runner(tmp_path, environment, policies)

    def scripted_flags(env, agent):
        crashed = agent in env.crashed
        return {"collision": crashed, "flip": False, "crash": crashed}

    monkeypatch.setattr(
        "f1tenth_controllers.benchmark.runner._state_flags",
        scripted_flags,
    )
    row = runner.run_head_to_head(make_heat())

    assert row["outcome_type"] == "crash_win"
    assert row["winner_checkpoint_id"] == "FAST"
    assert row["crash_participant_checkpoint_ids"] == ["SLOW"]
    assert row["responsible_checkpoint_id"] == "SLOW"
    assert row["attribution_confidence"] == "medium"
    assert row["dnf_reasons"] == {"SLOW": "collision"}


def test_same_step_two_car_crash_is_double_crash_and_conservative(
    tmp_path: Path,
    monkeypatch,
) -> None:
    environment = ScriptedEnvironment(
        ["f1tenth", "opponent_0"],
        crash_step=2,
        crash_agents={"f1tenth", "opponent_0"},
    )
    policies = {
        "FAST": ScriptedPolicy("FAST", 0.5),
        "SLOW": ScriptedPolicy("SLOW", 0.5),
    }
    runner = make_runner(tmp_path, environment, policies)

    def scripted_flags(env, agent):
        crashed = agent in env.crashed
        return {"collision": crashed, "flip": False, "crash": crashed}

    monkeypatch.setattr(
        "f1tenth_controllers.benchmark.runner._state_flags",
        scripted_flags,
    )
    row = runner.run_head_to_head(make_heat())

    assert row["outcome_type"] == "double_crash"
    assert set(row["crash_participant_checkpoint_ids"]) == {"FAST", "SLOW"}
    assert row["responsible_checkpoint_id"] == "indeterminate"
    assert row["attribution_confidence"] == "low"
    assert row["dnf_reasons"] == {
        "FAST": "collision",
        "SLOW": "collision",
    }
