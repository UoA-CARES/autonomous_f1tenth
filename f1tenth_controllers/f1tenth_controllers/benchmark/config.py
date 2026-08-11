"""Strict loading for the checked-in benchmark experiment configuration."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
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

    lap_monitor = config.get("lap_monitor", {})
    if lap_monitor.get("max_projection_speed_mps") != environment.get(
        "max_speed"
    ):
        raise ValueError(
            "lap_monitor.max_projection_speed_mps must match the "
            "training environment max_speed"
        )

    runtime = config.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("Benchmark config must contain runtime settings")
    try:
        ros_domain_id = int(runtime["ros_domain_id"])
        service_timeout = float(
            runtime["evaluation_service_timeout_wall_seconds"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid benchmark runtime settings") from error
    if not 0 <= ros_domain_id <= 232:
        raise ValueError("runtime.ros_domain_id must be between 0 and 232")
    if not math.isfinite(service_timeout) or service_timeout <= 0.0:
        raise ValueError(
            "runtime.evaluation_service_timeout_wall_seconds must be "
            "positive and finite"
        )

    config["config_sha256"] = _canonical_sha256(config)
    return config


def resolve_runtime_config(config: dict, *, pilot: bool) -> dict:
    """Resolve a pilot without allowing its rows into the full campaign."""
    if not pilot:
        return config

    resolved = deepcopy(config)
    resolved.pop("config_sha256", None)
    pilot_timeout = float(resolved["pilot"]["timeout_sim_seconds"])
    if pilot_timeout <= 0.0:
        raise ValueError("pilot.timeout_sim_seconds must be positive")
    if pilot_timeout >= float(
        resolved["environment"]["timeout_sim_seconds"]
    ):
        raise ValueError(
            "Pilot timeout must be shorter than the full timeout"
        )
    resolved["environment"]["timeout_sim_seconds"] = pilot_timeout
    resolved["campaign_mode"] = "pilot"
    resolved["config_sha256"] = _canonical_sha256(resolved)
    return resolved


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
