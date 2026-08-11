"""Strict CARES actor adapter with no deployment-only action processing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from cares_reinforcement_learning.algorithm.algorithm_factory import (
    AlgorithmFactory,
)

from f1tenth_controllers.rl_policy import (
    _build_marl_observation_size,
    _configure_actor_from_any_checkpoint,
    _load_marl_actor_weights,
    _load_network_config,
    _marl_action,
)

from .config import CheckpointSpec


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_family(checkpoint: dict) -> str:
    actor = checkpoint.get("actor")
    if not isinstance(actor, dict):
        raise ValueError("Checkpoint has no actor state dictionary")
    has_mean = "mean_linear.weight" in actor
    has_log_std_head = "log_std_linear.weight" in actor
    if has_mean != has_log_std_head:
        raise ValueError("Checkpoint contains only one SAC actor head")
    if has_mean:
        return "sac"
    if "log_std" in checkpoint and "value_normaliser" in checkpoint:
        return "ppo"
    if "target_actor" in checkpoint and "policy_noise" in checkpoint:
        return "td3"
    return "unknown"


def _json_safe(value):
    return json.loads(json.dumps(value, default=str))


@dataclass(slots=True)
class PolicyAdapter:
    """One selected actor using standard CARES evaluation inference."""

    spec: CheckpointSpec
    checkpoint_path: Path
    policy: object
    observation_size: int
    action_size: int
    checkpoint_metadata: dict
    algorithm_configuration: dict

    @classmethod
    def load(
        cls, spec: CheckpointSpec, checkpoint_root: Path
    ) -> "PolicyAdapter":
        checkpoint_path = Path(checkpoint_root) / spec.filename
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Missing {spec.algorithm} checkpoint: {checkpoint_path}"
            )
        actual_size = checkpoint_path.stat().st_size
        if actual_size != spec.size_bytes:
            raise ValueError(
                f"{spec.algorithm} checkpoint size mismatch: "
                f"expected {spec.size_bytes}, got {actual_size}"
            )
        actual_sha256 = _sha256_file(checkpoint_path)
        if actual_sha256 != spec.sha256:
            raise ValueError(
                f"{spec.algorithm} checkpoint SHA256 mismatch: "
                f"expected {spec.sha256}, got {actual_sha256}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        if not isinstance(checkpoint, dict):
            raise ValueError(
                f"{spec.algorithm} checkpoint top level must be a dictionary"
            )
        family = _checkpoint_family(checkpoint)
        if spec.expected_family is not None and family != spec.expected_family:
            raise ValueError(
                f"{spec.algorithm} expected {spec.expected_family} checkpoint "
                f"structure, got {family}; top-level keys={sorted(checkpoint)}"
            )

        network_config = _load_network_config(spec.algorithm)
        actor_observation_size, action_size, actor_path = (
            _configure_actor_from_any_checkpoint(
                spec.algorithm,
                network_config,
                checkpoint_path,
            )
        )
        observation_size, raw_observation_size = _build_marl_observation_size(
            spec.algorithm,
            actor_observation_size,
            network_config,
            [spec.actor_id],
            {spec.actor_id: [spec.actor_id]},
        )
        policy = AlgorithmFactory().create_network(
            observation_size,
            action_size,
            config=network_config,
        )
        if policy is None:
            raise RuntimeError(
                f"CARES did not construct a {spec.algorithm} policy"
            )
        _load_marl_actor_weights(policy, actor_path, spec.actor_id)

        actor = checkpoint["actor"]
        actor_shapes = {
            key: list(value.shape)
            for key, value in actor.items()
            if hasattr(value, "shape")
        }
        config_dump = (
            network_config.model_dump()
            if hasattr(network_config, "model_dump")
            else network_config.dict()
        )
        adapter = cls(
            spec=spec,
            checkpoint_path=checkpoint_path,
            policy=policy,
            observation_size=raw_observation_size,
            action_size=action_size,
            checkpoint_metadata={
                **asdict(spec),
                "path": str(checkpoint_path.resolve()),
                "family": family,
                "top_level_keys": sorted(checkpoint),
                "actor_state_shapes": actor_shapes,
            },
            algorithm_configuration=_json_safe(config_dump),
        )
        adapter.preflight()
        return adapter

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Return the raw deterministic CARES evaluation action."""
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (self.observation_size,):
            raise ValueError(
                f"{self.spec.algorithm} expected observation shape "
                f"{(self.observation_size,)}, got {observation.shape}"
            )
        if not np.isfinite(observation).all():
            raise ValueError(
                f"{self.spec.algorithm} received a non-finite observation"
            )
        action = np.asarray(
            _marl_action(
                self.policy,
                self.spec.actor_id,
                observation,
            ),
            dtype=np.float32,
        )
        if action.shape != (self.action_size,):
            raise ValueError(
                f"{self.spec.algorithm} returned action shape {action.shape}"
            )
        if not np.isfinite(action).all():
            raise ValueError(
                f"{self.spec.algorithm} returned a non-finite action"
            )
        if np.any(np.abs(action) > 1.0 + 1e-6):
            raise ValueError(
                f"{self.spec.algorithm} returned action outside [-1, 1]"
            )
        return action

    def preflight(self) -> None:
        """Verify dimensions, finiteness, and deterministic evaluation mode."""
        observation = np.linspace(
            0.0, 1.0, self.observation_size, dtype=np.float32
        )
        first = self.act(observation)
        second = self.act(observation.copy())
        if not np.array_equal(first, second):
            raise ValueError(
                f"{self.spec.algorithm} evaluation action is not deterministic"
            )

    def manifest_entry(self) -> dict:
        return {
            **self.checkpoint_metadata,
            "observation_size": self.observation_size,
            "action_size": self.action_size,
            "algorithm_configuration": self.algorithm_configuration,
            "evaluation_inference": "CARES act(evaluation=True)",
            "action_postprocessing": "none",
        }


def preflight_policies(
    specs: list[CheckpointSpec], checkpoint_root: Path
) -> dict[str, PolicyAdapter]:
    """Load all declared policies without substituting another checkpoint."""
    adapters = {
        spec.checkpoint_id: PolicyAdapter.load(spec, checkpoint_root)
        for spec in specs
    }
    if len(adapters) != len(specs):
        raise RuntimeError("Duplicate checkpoint identities in preflight")
    return adapters
