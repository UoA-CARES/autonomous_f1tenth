"""Lap validation using wrapped distances from ``TrackProgressModel``."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class LapMonitorConfig:
    """Geometry-independent limits for one lap monitor."""

    lap_length_m: float
    sector_fractions: tuple[float, ...] = (0.25, 0.5, 0.75)
    max_projection_jump_m: float = 1.0
    max_projection_speed_mps: float = 0.0

    def __post_init__(self) -> None:
        if self.lap_length_m <= 0.0:
            raise ValueError("lap_length_m must be positive")
        if self.max_projection_jump_m <= 0.0:
            raise ValueError("max_projection_jump_m must be positive")
        if self.max_projection_speed_mps < 0.0:
            raise ValueError("max_projection_speed_mps must be nonnegative")
        if any(
            not 0.0 < fraction < 1.0 for fraction in self.sector_fractions
        ):
            raise ValueError(
                "sector fractions must be strictly between zero and one"
            )
        if tuple(sorted(self.sector_fractions)) != self.sector_fractions:
            raise ValueError("sector fractions must be in ascending order")
        if len(set(self.sector_fractions)) != len(self.sector_fractions):
            raise ValueError("sector fractions must be unique")


@dataclass(slots=True)
class SectorSequence:
    """Require virtual sector gates to be recorded in declared order."""

    gate_count: int
    next_gate: int = 0
    invalid: bool = False

    def pass_gate(self, gate_index: int) -> bool:
        """Record one gate, invalidating the sequence if it is out of order."""
        if self.invalid or gate_index != self.next_gate:
            self.invalid = True
            return False
        self.next_gate += 1
        return True

    @property
    def complete(self) -> bool:
        return not self.invalid and self.next_gate == self.gate_count


@dataclass(frozen=True, slots=True)
class LapUpdate:
    """Result of incorporating one synchronous simulator sample."""

    accepted: bool
    reason: str | None
    previous_sim_time: float
    sim_time: float
    previous_unwrapped_progress_m: float
    unwrapped_progress_m: float
    signed_step_m: float
    max_allowed_step_m: float = 0.0
    sample_discarded: bool = False
    passed_sector_indices: tuple[int, ...] = ()
    finish_crossed: bool = False
    finish_sim_time: float | None = None


@dataclass(slots=True)
class LapMonitor:
    """Validate a forward lap without treating a seam crossing as a finish."""

    config: LapMonitorConfig
    start_track_distance_m: float
    start_sim_time: float
    last_track_distance_m: float = field(init=False)
    last_sim_time: float = field(init=False)
    unwrapped_progress_m: float = field(init=False, default=0.0)
    sectors: SectorSequence = field(init=False)
    invalid_reason: str | None = field(init=False, default=None)
    finish_sim_time: float | None = field(init=False, default=None)
    projection_jump_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.last_track_distance_m = (
            float(self.start_track_distance_m) % self.config.lap_length_m
        )
        self.last_sim_time = float(self.start_sim_time)
        self.sectors = SectorSequence(len(self.config.sector_fractions))

    @property
    def finished(self) -> bool:
        return self.finish_sim_time is not None

    @property
    def valid(self) -> bool:
        return self.invalid_reason is None and not self.sectors.invalid

    def _signed_step(self, track_distance_m: float) -> float:
        lap_length = self.config.lap_length_m
        current = float(track_distance_m) % lap_length
        forward = (current - self.last_track_distance_m) % lap_length
        if forward <= lap_length / 2.0:
            return forward
        return forward - lap_length

    def _invalid_update(
        self,
        *,
        reason: str,
        track_distance_m: float,
        sim_time: float,
        signed_step_m: float,
        max_allowed_step_m: float,
    ) -> LapUpdate:
        previous_time = self.last_sim_time
        previous_progress = self.unwrapped_progress_m
        self.invalid_reason = reason
        self.last_track_distance_m = (
            float(track_distance_m) % self.config.lap_length_m
        )
        self.last_sim_time = float(sim_time)
        return LapUpdate(
            accepted=False,
            reason=reason,
            previous_sim_time=previous_time,
            sim_time=float(sim_time),
            previous_unwrapped_progress_m=previous_progress,
            unwrapped_progress_m=previous_progress,
            signed_step_m=signed_step_m,
            max_allowed_step_m=max_allowed_step_m,
        )

    def update(self, track_distance_m: float, sim_time: float) -> LapUpdate:
        """Add one projected sample and return any new gate/finish event."""
        sim_time = float(sim_time)
        if sim_time <= self.last_sim_time:
            return self._invalid_update(
                reason="non_monotonic_sim_time",
                track_distance_m=track_distance_m,
                sim_time=sim_time,
                signed_step_m=0.0,
                max_allowed_step_m=self.config.max_projection_jump_m,
            )

        simulator_delta_s = sim_time - self.last_sim_time
        max_allowed_step_m = (
            self.config.max_projection_jump_m
            + self.config.max_projection_speed_mps * simulator_delta_s
        )

        if not self.valid:
            return LapUpdate(
                accepted=False,
                reason=self.invalid_reason or "invalid_sector_order",
                previous_sim_time=self.last_sim_time,
                sim_time=sim_time,
                previous_unwrapped_progress_m=self.unwrapped_progress_m,
                unwrapped_progress_m=self.unwrapped_progress_m,
                signed_step_m=0.0,
                max_allowed_step_m=max_allowed_step_m,
            )

        if self.finished:
            return LapUpdate(
                accepted=True,
                reason="already_finished",
                previous_sim_time=self.last_sim_time,
                sim_time=sim_time,
                previous_unwrapped_progress_m=self.unwrapped_progress_m,
                unwrapped_progress_m=self.unwrapped_progress_m,
                signed_step_m=0.0,
                max_allowed_step_m=max_allowed_step_m,
                finish_sim_time=self.finish_sim_time,
            )

        signed_step = self._signed_step(track_distance_m)
        if abs(signed_step) > max_allowed_step_m:
            previous_time = self.last_sim_time
            previous_progress = self.unwrapped_progress_m
            self.projection_jump_count += 1
            self.last_track_distance_m = (
                float(track_distance_m) % self.config.lap_length_m
            )
            self.last_sim_time = sim_time
            return LapUpdate(
                accepted=True,
                reason="projection_jump_discarded",
                previous_sim_time=previous_time,
                sim_time=sim_time,
                previous_unwrapped_progress_m=previous_progress,
                unwrapped_progress_m=previous_progress,
                signed_step_m=signed_step,
                max_allowed_step_m=max_allowed_step_m,
                sample_discarded=True,
            )

        previous_time = self.last_sim_time
        previous_progress = self.unwrapped_progress_m
        current_progress = previous_progress + signed_step
        passed_sectors: list[int] = []

        if signed_step > 0.0:
            while self.sectors.next_gate < len(self.config.sector_fractions):
                gate_index = self.sectors.next_gate
                gate_progress = (
                    self.config.sector_fractions[gate_index]
                    * self.config.lap_length_m
                )
                if not previous_progress < gate_progress <= current_progress:
                    break
                if not self.sectors.pass_gate(gate_index):
                    return self._invalid_update(
                        reason="invalid_sector_order",
                        track_distance_m=track_distance_m,
                        sim_time=sim_time,
                        signed_step_m=signed_step,
                        max_allowed_step_m=max_allowed_step_m,
                    )
                passed_sectors.append(gate_index)

        finish_crossed = False
        finish_time = None
        lap_length = self.config.lap_length_m
        if (
            signed_step > 0.0
            and previous_progress < lap_length <= current_progress
            and self.sectors.complete
        ):
            interpolation = (lap_length - previous_progress) / signed_step
            finish_time = previous_time + interpolation * (
                sim_time - previous_time
            )
            self.finish_sim_time = finish_time
            finish_crossed = True

        self.last_track_distance_m = float(track_distance_m) % lap_length
        self.last_sim_time = sim_time
        self.unwrapped_progress_m = current_progress

        return LapUpdate(
            accepted=True,
            reason=None,
            previous_sim_time=previous_time,
            sim_time=sim_time,
            previous_unwrapped_progress_m=previous_progress,
            unwrapped_progress_m=current_progress,
            signed_step_m=signed_step,
            max_allowed_step_m=max_allowed_step_m,
            passed_sector_indices=tuple(passed_sectors),
            finish_crossed=finish_crossed,
            finish_sim_time=finish_time,
        )
