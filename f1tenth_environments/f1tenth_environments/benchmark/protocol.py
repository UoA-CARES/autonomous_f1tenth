"""Deterministic heat scheduling and race-outcome helpers."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass

from .monitor import LapUpdate


@dataclass(frozen=True, slots=True)
class CheckpointRef:
    checkpoint_id: str
    algorithm: str
    filename: str
    sha256: str


@dataclass(frozen=True, slots=True)
class HeatSpec:
    """One balanced head-to-head assignment."""

    heat_id: str
    checkpoint_id_a: str
    algorithm_a: str
    checkpoint_a: str
    checkpoint_sha256_a: str
    checkpoint_id_b: str
    algorithm_b: str
    checkpoint_b: str
    checkpoint_sha256_b: str
    lead_checkpoint_id: str
    lead_algorithm: str
    chaser_checkpoint_id: str
    chaser_algorithm: str
    primary_checkpoint_id: str
    primary_algorithm: str
    opponent_checkpoint_id: str
    opponent_algorithm: str
    seed: int
    track: str
    direction: str


def _heat_id(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def build_balanced_heats(
    checkpoints: list[CheckpointRef],
    seeds: list[int],
    track: str,
    direction: str,
) -> list[HeatSpec]:
    """Build all checkpoint pairings with lead/chaser and slot balance."""
    if len({item.checkpoint_id for item in checkpoints}) != len(checkpoints):
        raise ValueError("checkpoint identities must be unique")
    if not seeds:
        raise ValueError("at least one seed is required")

    ordered_checkpoints = sorted(
        checkpoints, key=lambda item: item.checkpoint_id
    )
    heats: list[HeatSpec] = []
    checkpoint_pairs = itertools.combinations(ordered_checkpoints, 2)
    for checkpoint_a, checkpoint_b in checkpoint_pairs:
        for seed in seeds:
            for a_leads in (True, False):
                for a_in_primary_slot in (True, False):
                    lead = checkpoint_a if a_leads else checkpoint_b
                    chaser = checkpoint_b if a_leads else checkpoint_a
                    primary = (
                        checkpoint_a if a_in_primary_slot else checkpoint_b
                    )
                    opponent = (
                        checkpoint_b if a_in_primary_slot else checkpoint_a
                    )
                    payload = {
                        "checkpoint_id_a": checkpoint_a.checkpoint_id,
                        "algorithm_a": checkpoint_a.algorithm,
                        "checkpoint_a": checkpoint_a.filename,
                        "checkpoint_sha256_a": checkpoint_a.sha256,
                        "checkpoint_id_b": checkpoint_b.checkpoint_id,
                        "algorithm_b": checkpoint_b.algorithm,
                        "checkpoint_b": checkpoint_b.filename,
                        "checkpoint_sha256_b": checkpoint_b.sha256,
                        "lead_checkpoint_id": lead.checkpoint_id,
                        "lead_algorithm": lead.algorithm,
                        "chaser_checkpoint_id": chaser.checkpoint_id,
                        "chaser_algorithm": chaser.algorithm,
                        "primary_checkpoint_id": primary.checkpoint_id,
                        "primary_algorithm": primary.algorithm,
                        "opponent_checkpoint_id": opponent.checkpoint_id,
                        "opponent_algorithm": opponent.algorithm,
                        "seed": int(seed),
                        "track": track,
                        "direction": direction,
                    }
                    heats.append(
                        HeatSpec(heat_id=_heat_id(payload), **payload)
                    )

    heat_ids = [heat.heat_id for heat in heats]
    if len(heat_ids) != len(set(heat_ids)):
        raise RuntimeError("generated duplicate heat IDs")
    return heats


@dataclass(frozen=True, slots=True)
class FinishResolution:
    outcome_type: str
    winner: str | None
    finish_sim_time: float | None
    lead_m: float | None


def _progress_at_time(update: LapUpdate, sim_time: float) -> float:
    duration = update.sim_time - update.previous_sim_time
    if duration <= 0.0:
        return update.unwrapped_progress_m
    fraction = (sim_time - update.previous_sim_time) / duration
    fraction = min(max(fraction, 0.0), 1.0)
    progress_delta = (
        update.unwrapped_progress_m - update.previous_unwrapped_progress_m
    )
    return update.previous_unwrapped_progress_m + fraction * progress_delta


def resolve_finish_step(
    updates: dict[str, LapUpdate],
    lap_length_m: float,
    tie_tolerance_s: float,
) -> FinishResolution | None:
    """Resolve one synchronous step containing one or two valid finishes."""
    finishers = {
        car: update.finish_sim_time
        for car, update in updates.items()
        if update.finish_crossed and update.finish_sim_time is not None
    }
    if not finishers:
        return None

    ordered = sorted(finishers.items(), key=lambda item: item[1])
    within_tolerance = (
        len(ordered) > 1
        and abs(ordered[0][1] - ordered[1][1]) <= tie_tolerance_s
    )
    if within_tolerance:
        return FinishResolution(
            outcome_type="tie",
            winner=None,
            finish_sim_time=min(ordered[0][1], ordered[1][1]),
            lead_m=0.0,
        )

    winner, finish_time = ordered[0]
    losers = [car for car in updates if car != winner]
    if len(losers) != 1:
        raise ValueError("finish resolution requires exactly two cars")
    loser_progress = _progress_at_time(updates[losers[0]], finish_time)
    lead_m = max(0.0, float(lap_length_m) - loser_progress)
    return FinishResolution(
        outcome_type="finish_win",
        winner=winner,
        finish_sim_time=finish_time,
        lead_m=lead_m,
    )


@dataclass(frozen=True, slots=True)
class CrashAssessment:
    participants: tuple[str, ...]
    responsible_car: str
    attribution_confidence: str
    evidence: tuple[str, ...]


def assess_crash(
    crash_signals: dict[str, bool],
    *,
    pair_distance_m: float,
    progress_m: dict[str, float],
    speed_mps: dict[str, float],
    possible_contact_distance_m: float = 0.5,
    rear_end_closing_speed_mps: float = 0.2,
) -> CrashAssessment:
    """Attribute a crash only when existing signals support the inference."""
    signalled = tuple(
        sorted(car for car, crashed in crash_signals.items() if crashed)
    )
    if not signalled:
        return CrashAssessment(
            (), "indeterminate", "none", ("no_crash_signal",)
        )

    if len(signalled) == 1 and pair_distance_m > possible_contact_distance_m:
        car = signalled[0]
        return CrashAssessment(
            participants=(car,),
            responsible_car=car,
            attribution_confidence="medium",
            evidence=("isolated_crash_signal", "cars_separated"),
        )

    participants = tuple(sorted(crash_signals))
    if len(progress_m) == 2 and len(speed_mps) == 2:
        trailing, leading = sorted(progress_m, key=progress_m.get)
        closing_speed = speed_mps[trailing] - speed_mps[leading]
        if closing_speed >= rear_end_closing_speed_mps:
            return CrashAssessment(
                participants=participants,
                responsible_car=trailing,
                attribution_confidence="medium",
                evidence=(
                    "possible_vehicle_contact",
                    "trailing_car_closing_on_leading_car",
                ),
            )

    return CrashAssessment(
        participants=participants,
        responsible_car="indeterminate",
        attribution_confidence="low",
        evidence=("possible_vehicle_contact", "insufficient_contact_evidence"),
    )


def heat_as_dict(heat: HeatSpec) -> dict:
    """Return a stable JSON-serializable heat representation."""
    return asdict(heat)
