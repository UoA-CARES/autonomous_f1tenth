"""Strict loading for the benchmark configuration and local checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CheckpointSpec:
    """One authoritative selected checkpoint."""

    algorithm: str
    filename: str
    sha256: str
    size_bytes: int
    actor_id: str

    @property
    def checkpoint_id(self) -> str:
        """Stable human-readable identity for this exact checkpoint file."""
        return Path(self.filename).stem

    @property
    def expected_family(self) -> str | None:
        if self.algorithm.endswith("TD3"):
            return "td3"
        if self.algorithm.endswith("PPO"):
            return "ppo"
        if self.algorithm.endswith("SAC"):
            return "sac"
        return None


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

    if config.get("schema_version") != 4:
        raise ValueError("Unsupported benchmark schema_version")

    discovery = config.get("checkpoint_discovery")
    if not isinstance(discovery, dict):
        raise ValueError(
            "Benchmark config must contain checkpoint_discovery settings"
        )
    if discovery.get("filename_pattern") != "<ALGORITHM>_<name>.pth":
        raise ValueError("Unsupported checkpoint filename_pattern")
    actor_id = str(discovery.get("actor_id", "")).strip()
    if actor_id != "f1tenth":
        raise ValueError(
            "checkpoint_discovery.actor_id must preserve the training "
            "actor identity 'f1tenth'"
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

    track = config.get("track", {})
    if track.get("start_waypoint_selection") != "seeded_sha256_modulo":
        raise ValueError(
            "track.start_waypoint_selection must be seeded_sha256_modulo"
        )
    start_salt = track.get("start_waypoint_seed_salt")
    if not isinstance(start_salt, str) or not start_salt.strip():
        raise ValueError(
            "track.start_waypoint_seed_salt must be a non-empty string"
        )

    separation = track.get("longitudinal_separation_m")
    if not isinstance(separation, (int, float)) or separation <= 0.0:
        raise ValueError("track.longitudinal_separation_m must be positive")

    lap_monitor = config.get("lap_monitor", {})
    if lap_monitor.get("max_projection_speed_mps") != environment.get(
        "max_speed"
    ):
        raise ValueError(
            "lap_monitor.max_projection_speed_mps must match the "
            "training environment max_speed"
        )
    if (
        lap_monitor.get("projection_jump_policy")
        != "discard_progress_and_reanchor"
    ):
        raise ValueError(
            "lap_monitor.projection_jump_policy must discard progress "
            "and reanchor"
        )
    if (
        lap_monitor.get("teleport_policy")
        != "world_displacement_exceeds_projection_bound"
    ):
        raise ValueError(
            "lap_monitor.teleport_policy must use the world displacement "
            "projection bound"
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


def resolve_checkpoint_config(
    config: dict, specs: list[CheckpointSpec]
) -> dict:
    """Bind runtime-discovered checkpoint identities to the configuration."""
    resolved = deepcopy(config)
    resolved.pop("config_sha256", None)
    resolved["resolved_checkpoints"] = {
        spec.checkpoint_id: {
            **asdict(spec),
            "checkpoint_id": spec.checkpoint_id,
        }
        for spec in specs
    }
    resolved["config_sha256"] = _canonical_sha256(resolved)
    return resolved


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


def resolve_time_trial_count(
    config: dict, *, trials_per_algorithm: int | None
) -> dict:
    """Override how many time-trial repetitions run per checkpoint."""
    if trials_per_algorithm is None:
        return config

    if trials_per_algorithm <= 0:
        raise ValueError("--trials-per-algorithm must be positive")

    resolved = deepcopy(config)
    resolved.pop("config_sha256", None)
    resolved["time_trials"]["trials_per_algorithm"] = trials_per_algorithm
    resolved["config_sha256"] = _canonical_sha256(resolved)
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _algorithm_from_filename(path: Path) -> str:
    prefix, separator, remainder = path.stem.partition("_")
    algorithm = prefix.upper()
    if path.suffix.lower() != ".pth" or not separator or not remainder:
        raise ValueError(
            f"Checkpoint filename {path.name!r} must match "
            "<ALGORITHM>_<name>.pth"
        )
    return algorithm


def resolve_checkpoint_specs(
    config: dict, checkpoint_root: Path
) -> list[CheckpointSpec]:
    """Discover child checkpoints and record immutable identity."""
    checkpoint_root = Path(checkpoint_root).expanduser()
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {checkpoint_root}"
        )

    checkpoint_paths = sorted(
        (
            path
            for path in checkpoint_root.iterdir()
            if path.is_file() and path.suffix.lower() == ".pth"
        ),
        key=lambda path: path.name.lower(),
    )
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No .pth checkpoints found directly in {checkpoint_root}"
        )

    actor_id = config["checkpoint_discovery"]["actor_id"]
    specs = []
    checkpoint_ids = set()
    for checkpoint_path in checkpoint_paths:
        algorithm = _algorithm_from_filename(checkpoint_path)
        checkpoint_id = checkpoint_path.stem
        if checkpoint_id in checkpoint_ids:
            raise ValueError(
                f"Duplicate checkpoint identity {checkpoint_id!r} in "
                f"{checkpoint_root}"
            )
        size_bytes = checkpoint_path.stat().st_size
        if size_bytes <= 0:
            raise ValueError(
                f"Checkpoint file is empty: {checkpoint_path.name!r}"
            )
        specs.append(CheckpointSpec(
            algorithm=algorithm,
            filename=checkpoint_path.name,
            sha256=_sha256_file(checkpoint_path),
            size_bytes=size_bytes,
            actor_id=actor_id,
        ))
        checkpoint_ids.add(checkpoint_id)

    return sorted(specs, key=lambda spec: spec.checkpoint_id.lower())
