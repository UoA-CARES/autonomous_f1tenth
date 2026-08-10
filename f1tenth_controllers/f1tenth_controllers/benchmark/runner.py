"""Synchronous time-trial and two-car benchmark orchestration."""

from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from f1tenth_environments import geometry_utils, lidar_processor
from f1tenth_environments.benchmark import (
    LapMonitor,
    LapMonitorConfig,
    assess_crash,
    centreline_spawn_pose,
    resolve_finish_step,
    side_by_side_spawn_poses,
)
from f1tenth_environments.benchmark.protocol import HeatSpec

from .policy import PolicyAdapter
from .results import ResultWriter, stable_id


def build_trial_id(
    *,
    algorithm: str,
    checkpoint_sha256: str,
    track: str,
    seed: int,
    repetition: int,
    config_id: str,
) -> str:
    return stable_id(
        "trial",
        {
            "algorithm": algorithm,
            "checkpoint_sha256": checkpoint_sha256,
            "track": track,
            "seed": int(seed),
            "repetition": int(repetition),
            "config_id": config_id,
        },
    )


def _state_flags(environment, agent: str) -> dict[str, bool]:
    state_data = environment.previous_state_data[agent]
    collision = lidar_processor.has_collided(
        state_data.lidar_sanitised_data,
        environment.collision_range_m,
    )
    flip = geometry_utils.has_flipped_over(state_data.quaternion_wxyz())
    return {
        "collision": bool(collision),
        "flip": bool(flip),
        "crash": bool(collision or flip),
    }


def _dnf_reason(
    *,
    collision: bool,
    flip: bool,
    stall: bool,
    timeout: bool,
    monitor_reason: str | None = None,
) -> str | None:
    if monitor_reason is not None:
        return monitor_reason
    if collision and flip:
        return "collision_and_flip"
    if collision:
        return "collision"
    if flip:
        return "flip"
    if stall:
        return "stall"
    if timeout:
        return "timeout"
    return None


class BenchmarkRunner:
    """Run policies through one existing synchronous multi-car environment."""

    def __init__(
        self,
        *,
        environment,
        policies: dict[str, PolicyAdapter],
        config: dict,
        result_writer: ResultWriter,
        manifest_id: str,
        git_revisions: dict[str, str],
    ) -> None:
        self.environment = environment
        self.policies = policies
        self.config = config
        self.result_writer = result_writer
        self.manifest_id = manifest_id
        self.git_revisions = git_revisions
        self.track_name = config["track"]["identifier"]
        self.direction = config["track"]["direction"]
        self.config_id = config["config_sha256"]

    @property
    def _waypoint(self):
        waypoint_index = int(self.config["track"]["start_waypoint_index"])
        waypoints = self.environment.tracks[self.track_name]
        if not 0 <= waypoint_index < len(waypoints):
            raise ValueError(
                f"Start waypoint {waypoint_index} is outside "
                f"{self.track_name} with {len(waypoints)} waypoints"
            )
        return waypoints[waypoint_index]

    def _lap_monitor(
        self,
        start_track_distance_m: float,
        start_sim_time: float,
    ) -> LapMonitor:
        monitor_config = self.config["lap_monitor"]
        return LapMonitor(
            LapMonitorConfig(
                lap_length_m=(
                    self.environment.current_track_model.waypoint_lap_length
                ),
                sector_fractions=tuple(
                    monitor_config["sector_fractions"]
                ),
                max_projection_jump_m=float(
                    monitor_config["max_projection_jump_m"]
                ),
            ),
            start_track_distance_m=start_track_distance_m,
            start_sim_time=start_sim_time,
        )

    def _track_distance(self, agent: str) -> float:
        position = self.environment.previous_state_data[agent].position_xy()
        return self.environment.current_track_model.track_distance_from_world_coord(
            np.asarray(position, dtype=np.float64)
        )

    def _sim_times(self) -> tuple[float, float]:
        command_time = self.environment.last_command_sim_time_s
        observation_time = self.environment.last_observation_sim_time_s
        if command_time is None or observation_time is None:
            raise RuntimeError("Environment did not expose simulator timestamps")
        return float(command_time), float(observation_time)

    def run_time_trial(
        self,
        algorithm: str,
        *,
        seed: int,
        repetition: int,
    ) -> dict:
        if self.environment.agents != [self.environment.car_name]:
            raise ValueError(
                "Time trials require a Gazebo launch with zero opponents"
            )
        policy = self.policies[algorithm]
        agent = self.environment.car_name
        spawn_pose = centreline_spawn_pose(self._waypoint).as_reset_dict()
        observations, _ = self.environment.reset(
            seed=seed,
            options={
                "evaluation": True,
                "track_name": self.track_name,
                "spawn_poses": {agent: spawn_pose},
            },
        )
        initial_track_distance = self._track_distance(agent)
        monitor = None
        start_sim_time = None
        speeds: list[float] = []
        collision = flip = stall = timeout = False
        dnf_reason = None
        finish_time = None

        while True:
            action = policy.act(observations[agent])
            (
                observations,
                _rewards,
                terminateds,
                truncateds,
                _infos,
            ) = self.environment.step({agent: action})
            command_time, observation_time = self._sim_times()
            if monitor is None:
                start_sim_time = command_time
                monitor = self._lap_monitor(
                    initial_track_distance,
                    start_sim_time,
                )

            update = monitor.update(
                self._track_distance(agent),
                observation_time,
            )
            speed = abs(
                self.environment.previous_state_data[agent].linear_velocity()
            )
            speeds.append(float(speed))
            flags = _state_flags(self.environment, agent)
            collision = flags["collision"]
            flip = flags["flip"]
            stall = (
                self.environment.stall_counters[agent]
                >= self.environment.stall_limit_steps
            )
            elapsed = observation_time - float(start_sim_time)
            timeout = (
                elapsed >= self.config["environment"]["timeout_sim_seconds"]
                or self.environment.step_counter
                >= self.config["environment"]["max_steps"]
            )

            if update.finish_crossed:
                finish_time = update.finish_sim_time
                break
            if not update.accepted:
                dnf_reason = update.reason
                break
            if terminateds[agent] or truncateds[agent] or timeout:
                dnf_reason = _dnf_reason(
                    collision=collision,
                    flip=flip,
                    stall=stall,
                    timeout=timeout,
                )
                break

        completed = finish_time is not None
        row = {
            "trial_id": build_trial_id(
                algorithm=algorithm,
                checkpoint_sha256=policy.spec.sha256,
                track=self.track_name,
                seed=seed,
                repetition=repetition,
                config_id=self.config_id,
            ),
            "algorithm": algorithm,
            "checkpoint_filename": policy.spec.filename,
            "checkpoint_sha256": policy.spec.sha256,
            "actor_id": policy.spec.actor_id,
            "track": self.track_name,
            "direction": self.direction,
            "seed": seed,
            "start_sim_time": start_sim_time,
            "finish_sim_time": finish_time,
            "lap_time": (
                float(finish_time) - float(start_sim_time)
                if completed
                else None
            ),
            "completion_status": "completed" if completed else "dnf",
            "dnf_reason": None if completed else dnf_reason,
            "distance_completed_m": min(
                max(monitor.unwrapped_progress_m, 0.0),
                monitor.config.lap_length_m,
            ),
            "average_speed_mps": float(np.mean(speeds)) if speeds else 0.0,
            "maximum_speed_mps": max(speeds, default=0.0),
            "collision": collision,
            "flip": flip,
            "stall": stall,
            "timeout": timeout,
            "git_revisions": self.git_revisions,
            "manifest_id": self.manifest_id,
            "config_id": self.config_id,
        }
        self.result_writer.write_time_trial(row)
        self.result_writer.write_event(
            {
                "event": "time_trial_finish" if completed else "time_trial_dnf",
                "trial_id": row["trial_id"],
                "sim_time": finish_time or observation_time,
                "reason": row["dnf_reason"],
                "manifest_id": self.manifest_id,
            }
        )
        return row

    def _race_assignments(
        self, heat: HeatSpec
    ) -> tuple[dict[str, str], dict[str, dict]]:
        if len(self.environment.agents) != 2:
            raise ValueError(
                "Head-to-head heats require exactly one Gazebo opponent"
            )
        primary = self.environment.car_name
        opponent = next(
            agent for agent in self.environment.agents if agent != primary
        )
        algorithms_by_agent = {
            primary: heat.primary_algorithm,
            opponent: heat.opponent_algorithm,
        }
        left_agent = next(
            agent
            for agent, algorithm in algorithms_by_agent.items()
            if algorithm == heat.left_algorithm
        )
        right_agent = next(
            agent for agent in algorithms_by_agent if agent != left_agent
        )
        spawn_poses = side_by_side_spawn_poses(
            self._waypoint,
            left_agent=left_agent,
            right_agent=right_agent,
            lateral_offset_m=float(
                self.config["track"]["lateral_offset_m"]
            ),
        )
        return algorithms_by_agent, spawn_poses

    def _race_progress(self, monitors: dict[str, LapMonitor]) -> dict[str, float]:
        return {
            agent: monitor.unwrapped_progress_m
            for agent, monitor in monitors.items()
        }

    def run_head_to_head(self, heat: HeatSpec) -> dict:
        algorithms_by_agent, spawn_poses = self._race_assignments(heat)
        observations, _ = self.environment.reset(
            seed=heat.seed,
            options={
                "evaluation": True,
                "track_name": self.track_name,
                "spawn_poses": spawn_poses,
            },
        )
        initial_track_distances = {
            agent: self._track_distance(agent)
            for agent in self.environment.agents
        }
        monitors: dict[str, LapMonitor] = {}
        active_agents = set(self.environment.agents)
        dnf_reasons: dict[str, str] = {}
        outcome_type = "invalid_heat"
        winner_agent = None
        outcome_time = None
        lead_m = None
        crash_assessment = assess_crash(
            {agent: False for agent in self.environment.agents},
            pair_distance_m=math.inf,
            progress_m={},
            speed_mps={},
        )
        final_updates = {}

        while True:
            actions = {
                agent: self.policies[algorithms_by_agent[agent]].act(
                    observations[agent]
                )
                for agent in active_agents
            }
            (
                observations,
                _rewards,
                terminateds,
                truncateds,
                _infos,
            ) = self.environment.step(actions)
            command_time, observation_time = self._sim_times()
            if not monitors:
                monitors = {
                    agent: self._lap_monitor(
                        initial_track_distances[agent],
                        command_time,
                    )
                    for agent in self.environment.agents
                }

            final_updates = {
                agent: monitors[agent].update(
                    self._track_distance(agent),
                    observation_time,
                )
                for agent in self.environment.agents
            }
            if any(not update.accepted for update in final_updates.values()):
                outcome_type = "invalid_heat"
                for agent, update in final_updates.items():
                    if not update.accepted:
                        dnf_reasons[agent] = update.reason or "invalid_monitor"
                outcome_time = observation_time
                break

            finish = resolve_finish_step(
                final_updates,
                self.environment.current_track_model.waypoint_lap_length,
                float(self.config["race"]["tie_tolerance_s"]),
            )
            if finish is not None:
                outcome_type = finish.outcome_type
                winner_agent = finish.winner
                outcome_time = finish.finish_sim_time
                lead_m = finish.lead_m
                break

            flags = {
                agent: _state_flags(self.environment, agent)
                for agent in self.environment.agents
            }
            crashers = {
                agent
                for agent in active_agents
                if flags[agent]["crash"]
            }
            if crashers:
                for agent in crashers:
                    dnf_reasons[agent] = _dnf_reason(
                        collision=flags[agent]["collision"],
                        flip=flags[agent]["flip"],
                        stall=False,
                        timeout=False,
                    ) or "crash"
                positions = {
                    agent: np.asarray(
                        self.environment.previous_state_data[
                            agent
                        ].position_xy(),
                        dtype=np.float64,
                    )
                    for agent in self.environment.agents
                }
                first, second = self.environment.agents
                pair_distance = float(
                    np.linalg.norm(positions[first] - positions[second])
                )
                speeds = {
                    agent: float(
                        self.environment.previous_state_data[
                            agent
                        ].linear_velocity()
                    )
                    for agent in self.environment.agents
                }
                crash_assessment = assess_crash(
                    {
                        agent: flags[agent]["crash"]
                        for agent in self.environment.agents
                    },
                    pair_distance_m=pair_distance,
                    progress_m=self._race_progress(monitors),
                    speed_mps=speeds,
                    possible_contact_distance_m=float(
                        self.config["race"][
                            "possible_contact_distance_m"
                        ]
                    ),
                    rear_end_closing_speed_mps=float(
                        self.config["race"][
                            "rear_end_closing_speed_mps"
                        ]
                    ),
                )
                outcome_time = observation_time
                if len(crashers) == 1 and len(active_agents) == 2:
                    outcome_type = "crash_win"
                    winner_agent = next(
                        agent
                        for agent in active_agents
                        if agent not in crashers
                    )
                    loser_agent = next(iter(crashers))
                    lead_m = (
                        monitors[winner_agent].unwrapped_progress_m
                        - monitors[loser_agent].unwrapped_progress_m
                    )
                elif len(crashers) == 2:
                    outcome_type = "double_crash"
                else:
                    outcome_type = "both_dnf"
                break

            for agent in tuple(active_agents):
                if not truncateds[agent]:
                    continue
                stall = (
                    self.environment.stall_counters[agent]
                    >= self.environment.stall_limit_steps
                )
                timeout = (
                    self.environment.step_counter
                    >= self.config["environment"]["max_steps"]
                )
                dnf_reasons[agent] = "stall" if stall else "timeout"
                self.environment.stop_agent(agent)
                active_agents.remove(agent)

            if not active_agents:
                outcome_type = (
                    "timeout_draw"
                    if dnf_reasons
                    and all(
                        reason == "timeout"
                        for reason in dnf_reasons.values()
                    )
                    else "both_dnf"
                )
                outcome_time = observation_time
                break

            start_time = min(
                monitor.start_sim_time for monitor in monitors.values()
            )
            if (
                observation_time - start_time
                >= self.config["environment"]["timeout_sim_seconds"]
            ):
                for agent in active_agents:
                    dnf_reasons[agent] = "timeout"
                outcome_type = (
                    "timeout_draw"
                    if len(dnf_reasons) == len(self.environment.agents)
                    else "both_dnf"
                )
                outcome_time = observation_time
                break

            if any(terminateds.values()) and not crashers:
                outcome_type = "invalid_heat"
                dnf_reasons["heat"] = "termination_without_crash_evidence"
                outcome_time = observation_time
                break

        winner_algorithm = (
            algorithms_by_agent[winner_agent]
            if winner_agent is not None
            else None
        )
        final_progress = {
            algorithms_by_agent[agent]: monitors[agent].unwrapped_progress_m
            for agent in self.environment.agents
        }
        row = {
            **asdict(heat),
            "outcome_type": outcome_type,
            "winner": winner_algorithm,
            "finish_or_crash_sim_time": outcome_time,
            "lead_m": lead_m,
            "final_progress_m": final_progress,
            "crash_participants": [
                algorithms_by_agent[agent]
                for agent in crash_assessment.participants
            ],
            "responsible_car": (
                algorithms_by_agent.get(
                    crash_assessment.responsible_car,
                    crash_assessment.responsible_car,
                )
            ),
            "attribution_confidence": (
                crash_assessment.attribution_confidence
            ),
            "attribution_evidence": crash_assessment.evidence,
            "dnf_reasons": {
                algorithms_by_agent.get(agent, agent): reason
                for agent, reason in dnf_reasons.items()
            },
            "git_revisions": self.git_revisions,
            "manifest_id": self.manifest_id,
            "config_id": self.config_id,
        }
        self.result_writer.write_head_to_head(row)
        self.result_writer.write_event(
            {
                "event": "head_to_head_result",
                "heat_id": heat.heat_id,
                "sim_time": outcome_time,
                "outcome_type": outcome_type,
                "winner": winner_algorithm,
                "manifest_id": self.manifest_id,
            }
        )
        return row
