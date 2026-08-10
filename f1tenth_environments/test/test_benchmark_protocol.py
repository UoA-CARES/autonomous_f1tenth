from dataclasses import replace

import pytest

from f1tenth_environments.benchmark.monitor import LapUpdate
from f1tenth_environments.benchmark.protocol import (
    CheckpointRef,
    assess_crash,
    build_balanced_heats,
    resolve_finish_step,
)


def _finish_update(
    finish_time: float, end_progress: float = 40.2
) -> LapUpdate:
    return LapUpdate(
        accepted=True,
        reason=None,
        previous_sim_time=4.0,
        sim_time=5.0,
        previous_unwrapped_progress_m=39.0,
        unwrapped_progress_m=end_progress,
        signed_step_m=end_progress - 39.0,
        finish_crossed=True,
        finish_sim_time=finish_time,
    )


def _non_finish_update(start: float, end: float) -> LapUpdate:
    return LapUpdate(
        accepted=True,
        reason=None,
        previous_sim_time=4.0,
        sim_time=5.0,
        previous_unwrapped_progress_m=start,
        unwrapped_progress_m=end,
        signed_step_m=end - start,
    )


def test_two_same_step_finishes_use_interpolated_times() -> None:
    result = resolve_finish_step(
        {"car_a": _finish_update(4.2), "car_b": _finish_update(4.7)},
        lap_length_m=40.0,
        tie_tolerance_s=0.001,
    )

    assert result is not None
    assert result.outcome_type == "finish_win"
    assert result.winner == "car_a"


def test_same_step_finishes_within_tolerance_are_tied() -> None:
    result = resolve_finish_step(
        {"car_a": _finish_update(4.2000), "car_b": _finish_update(4.2005)},
        lap_length_m=40.0,
        tie_tolerance_s=0.001,
    )

    assert result is not None
    assert result.outcome_type == "tie"
    assert result.winner is None
    assert result.lead_m == pytest.approx(0.0)


def test_lead_interpolates_loser_progress_across_wrap_boundary() -> None:
    result = resolve_finish_step(
        {
            "winner": _finish_update(4.5),
            "loser": _non_finish_update(38.5, 39.5),
        },
        lap_length_m=40.0,
        tie_tolerance_s=0.001,
    )

    assert result is not None
    assert result.lead_m == pytest.approx(1.0)


def test_balanced_heat_generation_is_unique_and_counterbalanced() -> None:
    checkpoints = [
        CheckpointRef("A", "a.pth", "a" * 64),
        CheckpointRef("B", "b.pth", "b" * 64),
        CheckpointRef("C", "c.pth", "c" * 64),
    ]

    heats = build_balanced_heats(
        checkpoints,
        seeds=[42, 100],
        track="test_track_02_350",
        direction="counter_clockwise",
    )

    assert len(heats) == 3 * 2 * 4
    assert len({heat.heat_id for heat in heats}) == len(heats)
    ab_heats = [
        heat
        for heat in heats
        if {heat.algorithm_a, heat.algorithm_b} == {"A", "B"}
    ]
    assert {heat.left_algorithm for heat in ab_heats} == {"A", "B"}
    assert {heat.primary_algorithm for heat in ab_heats} == {"A", "B"}


def test_heat_id_changes_when_an_assignment_changes() -> None:
    checkpoints = [
        CheckpointRef("A", "a.pth", "a" * 64),
        CheckpointRef("B", "b.pth", "b" * 64),
    ]
    heats = build_balanced_heats(checkpoints, [42], "track", "ccw")

    assert heats[0].heat_id != heats[1].heat_id
    assert replace(heats[0], heat_id=heats[1].heat_id) != heats[0]


def test_isolated_wall_crash_attributes_only_signalled_car() -> None:
    assessment = assess_crash(
        {"car_a": True, "car_b": False},
        pair_distance_m=2.0,
        progress_m={"car_a": 10.0, "car_b": 8.0},
        speed_mps={"car_a": 1.0, "car_b": 1.0},
    )

    assert assessment.participants == ("car_a",)
    assert assessment.responsible_car == "car_a"
    assert assessment.attribution_confidence == "medium"


def test_ambiguous_close_contact_is_indeterminate() -> None:
    assessment = assess_crash(
        {"car_a": True, "car_b": True},
        pair_distance_m=0.2,
        progress_m={"car_a": 10.0, "car_b": 10.1},
        speed_mps={"car_a": 1.0, "car_b": 1.0},
    )

    assert assessment.participants == ("car_a", "car_b")
    assert assessment.responsible_car == "indeterminate"
    assert assessment.attribution_confidence == "low"


def test_clear_rear_end_motion_attributes_trailing_car() -> None:
    assessment = assess_crash(
        {"car_a": True, "car_b": True},
        pair_distance_m=0.2,
        progress_m={"car_a": 9.8, "car_b": 10.0},
        speed_mps={"car_a": 1.5, "car_b": 1.0},
    )

    assert assessment.responsible_car == "car_a"
    assert assessment.attribution_confidence == "medium"
