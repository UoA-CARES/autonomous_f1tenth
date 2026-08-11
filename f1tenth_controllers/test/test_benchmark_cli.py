from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from f1tenth_controllers.benchmark.cli import (
    _default_result_root,
    build_heat_schedule,
    build_trial_schedule,
    declared_campaign_schedules,
    enable_simulator_time,
    environment_factory_config,
    pilot_heats,
    pilot_trials,
    resolve_trial_seeds,
    select_checkpoint_specs,
    validate_environment,
    validate_mode_policy_count,
    validate_runtime_environment,
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
            algorithm=algorithm,
            filename=f"{algorithm}_selected.pth",
            sha256=algorithm.lower().ljust(64, "0"),
        )
        self.observation_size = 11
        self.action_size = 2


def policies() -> dict[str, FakePolicy]:
    return {
        algorithm: FakePolicy(algorithm)
        for algorithm in ("MATD3", "MAPPO", "MASAC", "ITD3", "IPPO", "ISAC")
    }


def test_default_results_live_under_home_and_allow_override(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.delenv("F1TENTH_BENCHMARK_RESULTS_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert _default_result_root() == tmp_path / "f1tenth_benchmark_results"

    override = tmp_path / "external_results"
    monkeypatch.setenv("F1TENTH_BENCHMARK_RESULTS_DIR", str(override))
    assert _default_result_root() == override


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

    declared_trials, declared_heats = declared_campaign_schedules(
        trials,
        heats,
        pilot=True,
    )
    assert len(declared_trials) == 6
    assert len(declared_heats) == 15

    full_trials, full_heats = declared_campaign_schedules(
        trials,
        heats,
        pilot=False,
    )
    assert full_trials is trials
    assert full_heats is heats


def test_single_discovered_checkpoint_builds_time_trials_only() -> None:
    config = load_experiment_config(CONFIG_PATH)
    single_policy = {"ISAC": FakePolicy("ISAC")}

    trials = build_trial_schedule(config, single_policy)
    heats = build_heat_schedule(config, single_policy)

    assert len(trials) == 10
    assert len(pilot_trials(trials)) == 1
    assert heats == []
    validate_mode_policy_count("time-trials", single_policy)
    with pytest.raises(ValueError, match="at least two"):
        validate_mode_policy_count("head-to-head", single_policy)


def test_explicit_algorithm_selection_must_be_discovered() -> None:
    specs = [SimpleNamespace(algorithm="ISAC", checkpoint_id="ISAC_seed")]

    selected = select_checkpoint_specs(
        SimpleNamespace(algorithm=["ISAC"], checkpoint=None), specs
    )
    assert selected == specs

    with pytest.raises(ValueError, match="not present"):
        select_checkpoint_specs(
            SimpleNamespace(algorithm=["MATD3"], checkpoint=None), specs
        )


def test_short_seed_list_is_extended_deterministically() -> None:
    config = load_experiment_config(CONFIG_PATH)
    config["time_trials"]["trials_per_algorithm"] = 5
    config["time_trials"]["seeds"] = [42, 100]

    assert resolve_trial_seeds(config) == [42, 100, 101, 102, 103]


def test_same_algorithm_variants_remain_distinct_competitors() -> None:
    config = load_experiment_config(CONFIG_PATH)
    variants = {
        "MASAC_seed_1": FakePolicy("MASAC"),
        "MASAC_seed_2": FakePolicy("MASAC"),
    }
    variants["MASAC_seed_1"].spec.filename = "MASAC_seed_1.pth"
    variants["MASAC_seed_2"].spec.filename = "MASAC_seed_2.pth"
    variants["MASAC_seed_1"].spec.sha256 = "1" * 64
    variants["MASAC_seed_2"].spec.sha256 = "2" * 64

    assert len(build_trial_schedule(config, variants)) == 20
    heats = build_heat_schedule(config, variants)
    assert len(heats) == 8
    assert {heat.algorithm_a for heat in heats} == {"MASAC"}
    assert {heat.lead_checkpoint_id for heat in heats} == set(variants)


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


def test_runtime_environment_requires_declared_ros_domain(
    monkeypatch,
) -> None:
    config = load_experiment_config(CONFIG_PATH)
    monkeypatch.setenv("ROS_DOMAIN_ID", "77")
    validate_runtime_environment(config)

    monkeypatch.setenv("ROS_DOMAIN_ID", "0")
    with pytest.raises(ValueError, match="ROS_DOMAIN_ID=77"):
        validate_runtime_environment(config)


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
