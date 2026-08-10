"""Strict loading for the checked-in benchmark experiment configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


EXPECTED_ALGORITHMS = frozenset(
    {"MATD3", "MAPPO", "MASAC", "ITD3", "IPPO", "ISAC"}
)


@dataclass(frozen=True, slots=True)
class CheckpointSpec:
    """One authoritative selected checkpoint."""

    algorithm: str
    filename: str
    sha256: str
    size_bytes: int
    actor_id: str

    @property
    def expected_family(self) -> str:
        if self.algorithm in {"MATD3", "ITD3"}:
            return "td3"
        if self.algorithm in {"MAPPO", "IPPO"}:
            return "ppo"
        return "sac"


def _canonical_sha256(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_experiment_config(path: Path) -> dict:
    """Load and validate the stable JSON experiment declaration."""
    path = Path(path)
    with path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    if config.get("schema_version") != 1:
        raise ValueError("Unsupported benchmark schema_version")

    checkpoints = config.get("checkpoints")
    if not isinstance(checkpoints, dict):
        raise ValueError("Benchmark config must contain checkpoint mappings")
    algorithms = set(checkpoints)
    if algorithms != EXPECTED_ALGORITHMS:
        raise ValueError(
            "Checkpoint mapping must contain exactly "
            f"{sorted(EXPECTED_ALGORITHMS)}; got {sorted(algorithms)}"
        )

    environment = config.get("environment", {})
    expected_pipeline = (
        "training_raw_policy_output_direct_to_environment_clip"
    )
    if environment.get("action_pipeline") != expected_pipeline:
        raise ValueError(
            "Benchmark action_pipeline must preserve MARL training behavior"
        )
    if environment.get("position_speed_multiplier") != 1.0:
        raise ValueError(
            "Head-to-head position_speed_multiplier must be 1.0"
        )

    config["config_sha256"] = _canonical_sha256(config)
    return config


def resolve_checkpoint_specs(config: dict) -> list[CheckpointSpec]:
    """Convert the explicit mapping into validated immutable specifications."""
    specs = []
    for algorithm in sorted(EXPECTED_ALGORITHMS):
        values = config["checkpoints"][algorithm]
        filename = str(values["filename"])
        if Path(filename).name != filename:
            raise ValueError(
                f"Checkpoint filename for {algorithm} must not contain a path"
            )
        sha256 = str(values["sha256"]).lower()
        if len(sha256) != 64 or any(
            character not in "0123456789abcdef" for character in sha256
        ):
            raise ValueError(f"Invalid SHA256 for {algorithm}")
        size_bytes = int(values["size_bytes"])
        if size_bytes <= 0:
            raise ValueError(f"Invalid checkpoint size for {algorithm}")
        actor_id = str(values["actor_id"]).strip()
        if not actor_id:
            raise ValueError(f"Missing actor_id for {algorithm}")
        specs.append(
            CheckpointSpec(
                algorithm=algorithm,
                filename=filename,
                sha256=sha256,
                size_bytes=size_bytes,
                actor_id=actor_id,
            )
        )
    return specs
