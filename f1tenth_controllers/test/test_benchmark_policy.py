import json
from pathlib import Path

import pytest

from f1tenth_controllers.benchmark.config import (
    EXPECTED_ALGORITHMS,
    load_experiment_config,
    resolve_checkpoint_specs,
)
from f1tenth_controllers.benchmark.policy import preflight_policies


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parent
CONFIG_PATH = PACKAGE_ROOT / "config" / "marl_benchmark.json"
CHECKPOINT_ROOT = REPOSITORY_ROOT / "train_weights"


def test_checked_in_checkpoint_mapping_is_explicit_and_complete() -> None:
    config = load_experiment_config(CONFIG_PATH)
    specs = resolve_checkpoint_specs(config)

    assert {spec.algorithm for spec in specs} == EXPECTED_ALGORITHMS
    assert len({spec.filename for spec in specs}) == len(EXPECTED_ALGORITHMS)
    assert all(spec.actor_id == "f1tenth" for spec in specs)
    assert config["environment"]["position_speed_multiplier"] == 1.0
    assert config["environment"]["action_pipeline"].startswith(
        "training_raw_policy_output"
    )


def test_checkpoint_mapping_rejects_missing_algorithm(tmp_path: Path) -> None:
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    del config["checkpoints"]["ISAC"]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="must contain exactly"):
        load_experiment_config(path)


def test_checkpoint_mapping_rejects_historical_paths(tmp_path: Path) -> None:
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    config["checkpoints"]["ISAC"]["filename"] = "old_runs/isac.pth"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    loaded = load_experiment_config(path)
    with pytest.raises(ValueError, match="must not contain a path"):
        resolve_checkpoint_specs(loaded)


@pytest.mark.skipif(
    not CHECKPOINT_ROOT.is_dir(),
    reason="authoritative local checkpoints are not installed",
)
def test_all_authoritative_checkpoints_pass_strict_preflight() -> None:
    config = load_experiment_config(CONFIG_PATH)
    adapters = preflight_policies(
        resolve_checkpoint_specs(config),
        CHECKPOINT_ROOT,
    )

    assert set(adapters) == EXPECTED_ALGORITHMS
    assert all(adapter.observation_size == 11 for adapter in adapters.values())
    assert all(adapter.action_size == 2 for adapter in adapters.values())
