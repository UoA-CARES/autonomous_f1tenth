import pytest

from f1tenth_environments.benchmark.monitor import (
    LapMonitor,
    LapMonitorConfig,
    SectorSequence,
)


def _monitor(start_distance: float = 0.0) -> LapMonitor:
    return LapMonitor(
        LapMonitorConfig(
            lap_length_m=40.0,
            sector_fractions=(0.25, 0.5, 0.75),
            max_projection_jump_m=15.0,
        ),
        start_track_distance_m=start_distance,
        start_sim_time=0.0,
    )


def test_forward_wraparound_completes_one_sector_validated_lap() -> None:
    monitor = _monitor(start_distance=35.0)

    updates = [
        monitor.update(5.0, 1.0),
        monitor.update(15.0, 2.0),
        monitor.update(25.0, 3.0),
        monitor.update(35.0, 4.0),
    ]

    assert updates[-1].finish_crossed
    assert updates[-1].finish_sim_time == pytest.approx(4.0)
    assert monitor.sectors.complete


def test_backward_finish_crossing_is_rejected() -> None:
    monitor = _monitor(start_distance=1.0)

    update = monitor.update(39.0, 1.0)

    assert update.signed_step_m == pytest.approx(-2.0)
    assert not update.finish_crossed
    assert not monitor.finished


def test_starting_near_finish_does_not_complete_immediately() -> None:
    monitor = _monitor(start_distance=39.5)

    update = monitor.update(0.5, 1.0)

    assert update.unwrapped_progress_m == pytest.approx(1.0)
    assert not update.finish_crossed


def test_sector_sequence_rejects_incorrect_order() -> None:
    sectors = SectorSequence(gate_count=3)

    assert not sectors.pass_gate(1)
    assert sectors.invalid
    assert not sectors.complete


def test_projection_teleport_invalidates_monitor() -> None:
    monitor = LapMonitor(
        LapMonitorConfig(lap_length_m=40.0, max_projection_jump_m=2.0),
        start_track_distance_m=0.0,
        start_sim_time=0.0,
    )

    update = monitor.update(10.0, 1.0)

    assert not update.accepted
    assert update.reason == "projection_jump"
    assert not monitor.valid


def test_projection_limit_scales_with_actual_simulator_time() -> None:
    monitor = LapMonitor(
        LapMonitorConfig(
            lap_length_m=40.0,
            max_projection_jump_m=1.0,
            max_projection_speed_mps=2.0,
        ),
        start_track_distance_m=0.0,
        start_sim_time=0.0,
    )

    accepted = monitor.update(3.0, 1.0)

    assert accepted.accepted
    assert accepted.max_allowed_step_m == pytest.approx(3.0)


def test_projection_limit_rejects_motion_above_physical_bound() -> None:
    monitor = LapMonitor(
        LapMonitorConfig(
            lap_length_m=40.0,
            max_projection_jump_m=1.0,
            max_projection_speed_mps=2.0,
        ),
        start_track_distance_m=0.0,
        start_sim_time=0.0,
    )

    rejected = monitor.update(3.1, 1.0)

    assert not rejected.accepted
    assert rejected.reason == "projection_jump"
    assert rejected.max_allowed_step_m == pytest.approx(3.0)


def test_multiple_finish_crossings_emit_only_one_finish_event() -> None:
    monitor = _monitor()
    monitor.update(10.0, 1.0)
    monitor.update(20.0, 2.0)
    monitor.update(30.0, 3.0)
    first = monitor.update(0.5, 4.0)

    second = monitor.update(1.0, 5.0)

    assert first.finish_crossed
    assert not second.finish_crossed
    assert second.reason == "already_finished"
    assert second.finish_sim_time == first.finish_sim_time


def test_finish_time_is_interpolated_between_simulator_samples() -> None:
    monitor = _monitor()
    monitor.update(10.0, 1.0)
    monitor.update(20.0, 2.0)
    monitor.update(30.0, 3.0)
    monitor.update(39.0, 4.0)

    finish = monitor.update(1.0, 5.0)

    assert finish.finish_crossed
    assert finish.finish_sim_time == pytest.approx(4.5)
