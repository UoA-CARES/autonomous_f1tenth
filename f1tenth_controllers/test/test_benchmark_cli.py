from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from f1tenth_controllers.benchmark.cli import (
    build_heat_schedule,
    build_trial_schedule,
    enable_simulator_time,
    environment_factory_config,
    pilot_heats,
    pilot_trials,
    validate_environment,
)
from f1tenth_controllers.benchmark.config import (
    load_experiment_config,
    resolve_runtime_config,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "marl_benchmark.json"
)


class FakePolicy:
    def __init__(self, algorithm: str) -> None:
        self.spec = SimpleNamespace(
            filename=f"{algorithm}.pth",
            sha256=algorithm.lower().ljust(64, "0"),
        )
        self.observation_size = 11
        self.action_size = 2


def policies() -> dict[str, FakePolicy]:
    return {
        algorithm: FakePolicy(algorithm)
        for algorithm in ("MATD3", "MAPPO", "MASAC", "ITD3", "IPPO", "ISAC")
    }


def test_full_and_pilot_campaign_sizes_are_explicit() -> None:
    config = load_experiment_config(CONFIG_PATH)
    all_policies = policies()

    trials = build_trial_schedule(config, all_policies)
    heats = build_heat_schedule(config, all_policies)

    assert len(trials) == 60
    assert len({trial.trial_id for trial in trials}) == 60
    assert len(pilot_trials(trials)) == 6
    assert len(heats) == 120
    assert len({heat.heat_id for heat in heats}) == 120
    assert len(pilot_heats(heats)) == 15


def test_pilot_timeout_has_distinct_configuration_identity() -> None:
    full = load_experiment_config(CONFIG_PATH)
    pilot = resolve_runtime_config(full, pilot=True)

    assert full["environment"]["timeout_sim_seconds"] == 600.0
    assert pilot["environment"]["timeout_sim_seconds"] == 30.0
    assert pilot["campaign_mode"] == "pilot"
    assert pilot["config_sha256"] != full["config_sha256"]
    assert resolve_runtime_config(full, pilot=False) is full


def test_factory_config_uses_selected_world_and_fair_multiplier() -> None:
    config = load_experiment_config(CONFIG_PATH)

    factory_config = environment_factory_config(config)

    assert factory_config["track"] == "test_track_02_350"
    assert factory_config["position_speed_multiplier"] == 1.0
    assert factory_config["step_sleep_time_ms"] == 100
    assert factory_config["command_latency_ms"] == 30
    assert factory_config["lidar_state_size"] == 9
    assert "timeout_sim_seconds" not in factory_config


def fake_environment(config: dict):
    values = config["environment"]
    state_builder = SimpleNamespace(
        odom_mode=values["odom_mode"],
        lidar_mode=values["lidar_mode"],
        lidar_state_size=values["lidar_state_size"],
        policy_state_size=11,
    )
    return SimpleNamespace(
        agents=["f1tenth", "opponent_1"],
        max_steps=values["max_steps"],
        step_sleep_time_ms=values["step_sleep_time_ms"],
        command_latency_ms=values["command_latency_ms"],
        collision_range_m=values["collision_range_m"],
        stall_progress_threshold_m=values["stall_progress_threshold_m"],
        stall_limit_steps=values["stall_limit_steps"],
        position_speed_multiplier=values["position_speed_multiplier"],
        wheelbase_m=values["wheelbase_m"],
        state_builder=state_builder,
        min_actions=np.asarray(
            [values["min_speed"], -values["max_turn"]],
            dtype=np.float32,
        ),
        max_actions=np.asarray(
            [values["max_speed"], values["max_turn"]],
            dtype=np.float32,
        ),
        tracks={config["track"]["identifier"]: [(0.0, 0.0, 0.0, 0)]},
    )


def test_runtime_fidelity_validation_accepts_exact_environment() -> None:
    config = load_experiment_config(CONFIG_PATH)

    validate_environment(
        fake_environment(config),
        config,
        policies(),
        expected_agent_count=2,
    )


def test_runtime_fidelity_validation_rejects_competition_disadvantage() -> None:
    config = load_experiment_config(CONFIG_PATH)
    environment = fake_environment(config)
    environment.position_speed_multiplier = 0.9

    with pytest.raises(
        ValueError,
        match="position_speed_multiplier",
    ):
        validate_environment(
            environment,
            config,
            policies(),
            expected_agent_count=2,
        )


class FakeClock:
    ros_time_is_active = True

    @staticmethod
    def now():
        return SimpleNamespace(nanoseconds=42_000_000_000)


class FakeSimulatorTimeNode:
    def __init__(self) -> None:
        self.parameter = None

    def set_parameters(self, parameters):
        self.parameter = parameters[0]
        return [SimpleNamespace(successful=True, reason="")]

    @staticmethod
    def get_clock():
        return FakeClock()


def test_simulator_time_is_explicitly_enabled(monkeypatch) -> None:
    node = FakeSimulatorTimeNode()
    monkeypatch.setattr("rclpy.spin_once", lambda *_args, **_kwargs: None)

    enable_simulator_time(node)

    assert node.parameter.name == "use_sim_time"
    assert node.parameter.value is True
