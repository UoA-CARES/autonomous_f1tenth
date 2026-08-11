import hashlib
import json
from pathlib import Path

import pytest

from f1tenth_controllers.benchmark.config import (
    EXPECTED_ALGORITHMS,
    load_experiment_config,
    resolve_checkpoint_config,
    resolve_checkpoint_specs,
)
from f1tenth_controllers.benchmark.policy import preflight_policies


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parent
CONFIG_PATH = PACKAGE_ROOT / "config" / "marl_benchmark.json"
CHECKPOINT_ROOT = REPOSITORY_ROOT / "train_weights"


def test_checked_in_config_declares_discovery_without_file_identities() -> None:
    config = load_experiment_config(CONFIG_PATH)

    assert "checkpoints" not in config
    assert config["checkpoint_discovery"] == {
        "filename_pattern": "<ALGORITHM>_<name>.pth",
        "actor_id": "f1tenth",
    }
    assert config["environment"]["position_speed_multiplier"] == 1.0
    assert config["environment"]["action_pipeline"].startswith(
        "training_raw_policy_output"
    )


def test_discovers_seedless_checkpoint_and_records_identity(tmp_path: Path) -> None:
    config = load_experiment_config(CONFIG_PATH)
    contents = b"selected checkpoint"
    checkpoint = tmp_path / "isac_uploaded_model.pth"
    checkpoint.write_bytes(contents)

    specs = resolve_checkpoint_specs(config, tmp_path)

    assert len(specs) == 1
    spec = specs[0]
    assert spec.algorithm == "ISAC"
    assert spec.filename == checkpoint.name
    assert spec.sha256 == hashlib.sha256(contents).hexdigest()
    assert spec.size_bytes == len(contents)
    assert spec.actor_id == "f1tenth"
    assert spec.expected_family == "sac"


def test_resolved_config_identity_includes_discovered_checkpoint(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(CONFIG_PATH)
    checkpoint = tmp_path / "ISAC_candidate.pth"
    checkpoint.write_bytes(b"first")
    first_specs = resolve_checkpoint_specs(config, tmp_path)
    first = resolve_checkpoint_config(config, first_specs)

    checkpoint.write_bytes(b"second")
    second_specs = resolve_checkpoint_specs(config, tmp_path)
    second = resolve_checkpoint_config(config, second_specs)

    assert first["config_sha256"] != second["config_sha256"]
    assert first["resolved_checkpoints"]["ISAC"]["sha256"] == (
        first_specs[0].sha256
    )
    assert "resolved_checkpoints" not in config


def test_discovery_sorts_supported_algorithm_prefixes(tmp_path: Path) -> None:
    config = load_experiment_config(CONFIG_PATH)
    (tmp_path / "masac_candidate.pth").write_bytes(b"one")
    (tmp_path / "IPPO_candidate_without_seed.pth").write_bytes(b"two")

    specs = resolve_checkpoint_specs(config, tmp_path)

    assert [spec.algorithm for spec in specs] == ["IPPO", "MASAC"]


@pytest.mark.parametrize(
    "filename, message",
    [
        ("ISAC.pth", "must match"),
        ("TD3_model.pth", "Unsupported checkpoint algorithm prefix"),
    ],
)
def test_discovery_rejects_ambiguous_filename(
    tmp_path: Path, filename: str, message: str
) -> None:
    config = load_experiment_config(CONFIG_PATH)
    (tmp_path / filename).write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match=message):
        resolve_checkpoint_specs(config, tmp_path)


def test_discovery_rejects_empty_checkpoint(tmp_path: Path) -> None:
    config = load_experiment_config(CONFIG_PATH)
    (tmp_path / "ISAC_empty.pth").touch()

    with pytest.raises(ValueError, match="is empty"):
        resolve_checkpoint_specs(config, tmp_path)


def test_discovery_rejects_multiple_files_for_one_algorithm(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(CONFIG_PATH)
    (tmp_path / "ISAC_first.pth").write_bytes(b"first")
    (tmp_path / "isac_second.pth").write_bytes(b"second")

    with pytest.raises(ValueError, match="Multiple ISAC checkpoints"):
        resolve_checkpoint_specs(config, tmp_path)


def test_discovery_does_not_search_nested_run_directories(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(CONFIG_PATH)
    historical = tmp_path / "historical_run"
    historical.mkdir()
    (historical / "ISAC_old.pth").write_bytes(b"old")

    with pytest.raises(FileNotFoundError, match="directly"):
        resolve_checkpoint_specs(config, tmp_path)


def test_runtime_settings_must_be_positive_and_finite(tmp_path: Path) -> None:
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    config["runtime"]["evaluation_service_timeout_wall_seconds"] = 0
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="positive and finite"):
        load_experiment_config(path)


@pytest.mark.skipif(
    not CHECKPOINT_ROOT.is_dir(),
    reason="local checkpoints are not installed",
)
def test_all_discovered_checkpoints_pass_strict_preflight() -> None:
    config = load_experiment_config(CONFIG_PATH)
    specs = resolve_checkpoint_specs(config, CHECKPOINT_ROOT)
    adapters = preflight_policies(specs, CHECKPOINT_ROOT)

    assert set(adapters).issubset(EXPECTED_ALGORITHMS)
    assert set(adapters) == {spec.algorithm for spec in specs}
    assert all(adapter.observation_size == 11 for adapter in adapters.values())
    assert all(adapter.action_size == 2 for adapter in adapters.values())
