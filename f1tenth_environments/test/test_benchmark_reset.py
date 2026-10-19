import math
import types

import pytest

from f1tenth_environments.benchmark.geometry import (
    centreline_spawn_pose,
    seeded_waypoint,
    seeded_waypoint_index,
    staggered_spawn_poses,
)
from f1tenth_environments.multi_f1tenth_environment import (
    MultiF1TenthEnvironment,
)


def test_centreline_pose_uses_left_track_normal() -> None:
    pose = centreline_spawn_pose((10.0, 20.0, math.pi / 2.0, 7), 0.3)

    assert pose.x == pytest.approx(9.7)
    assert pose.y == pytest.approx(20.0)
    assert pose.yaw == pytest.approx(math.pi / 2.0)
    assert pose.waypoint_index == 7


def test_seeded_waypoint_selection_is_reproducible_and_varied() -> None:
    waypoints = [
        (float(index), 0.0, 0.0, index) for index in range(64)
    ]
    seeds = range(1000, 1100)
    indices = [
        seeded_waypoint_index(seed, len(waypoints), salt="test-starts")
        for seed in seeds
    ]

    assert indices == [
        seeded_waypoint_index(seed, len(waypoints), salt="test-starts")
        for seed in seeds
    ]
    assert len(set(indices)) > 1
    assert seeded_waypoint(
        waypoints, 1000, salt="test-starts"
    ) == waypoints[indices[0]]


def test_seeded_waypoint_selection_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="waypoint_count"):
        seeded_waypoint_index(1, 0, salt="test-starts")
    with pytest.raises(ValueError, match="salt"):
        seeded_waypoint_index(1, 10, salt="")


def test_staggered_poses_have_common_lane_and_safe_separation() -> None:
    poses = staggered_spawn_poses(
        (10.0, 20.0, 0.0, 7),
        lead_agent="f1tenth",
        chaser_agent="opponent_1",
        longitudinal_separation_m=1.0,
    )

    lead = poses["f1tenth"]
    chaser = poses["opponent_1"]
    assert lead["x"] - chaser["x"] == pytest.approx(1.0)
    assert lead["y"] == pytest.approx(chaser["y"])
    assert lead["yaw"] == pytest.approx(chaser["yaw"])
    assert lead["longitudinal_offset_m"] == pytest.approx(0.5)
    assert chaser["longitudinal_offset_m"] == pytest.approx(-0.5)
    assert lead["waypoint_index"] == chaser["waypoint_index"] == 7


def _reset_seam_env() -> MultiF1TenthEnvironment:
    env = MultiF1TenthEnvironment.__new__(MultiF1TenthEnvironment)
    env.car_name = "f1tenth"
    env.agents = ["f1tenth", "opponent_1"]
    env.tracks = {
        "track_a": [(0.0, 0.0, 0.0, 0)],
        "track_b": [(1.0, 1.0, 0.0, 0)],
    }
    env.track_progress_models = {
        "track_a": object(),
        "track_b": object(),
    }
    env.spawn_indices = {}
    env.set_pose_calls = []
    env._set_model_pose = lambda **kwargs: env.set_pose_calls.append(kwargs)
    return env


def _automatic_reset_env() -> MultiF1TenthEnvironment:
    env = MultiF1TenthEnvironment.__new__(MultiF1TenthEnvironment)
    env.car_name = "f1tenth"
    env.opponent_car_names = ["opponent_1", "opponent_2", "opponent_3"]
    env.agents = [env.car_name, *env.opponent_car_names]
    env.tracks = {
        "track_a": [
            (float(index), float(index + 100), 0.1 * index, index)
            for index in range(32)
        ]
    }
    env.track_progress_models = {"track_a": object()}
    env._select_track_name = lambda: "track_a"
    env.set_pose_calls = []
    env._set_model_pose = lambda **kwargs: env.set_pose_calls.append(kwargs)
    return env


def test_training_reset_randomly_assigns_agents_to_existing_slots(
    monkeypatch,
) -> None:
    env = _automatic_reset_env()
    env.is_eval = False
    monkeypatch.setattr(
        "f1tenth_environments.multi_f1tenth_environment.random.randrange",
        lambda _waypoint_count: 3,
    )

    def fixed_shuffle(agent_ids) -> None:
        agent_ids[:] = [
            "opponent_2",
            "f1tenth",
            "opponent_3",
            "opponent_1",
        ]

    monkeypatch.setattr(
        "f1tenth_environments.multi_f1tenth_environment.random.shuffle",
        fixed_shuffle,
    )

    env._reset_positions()

    assert env.spawn_indices == {
        "opponent_2": 3,
        "f1tenth": 11,
        "opponent_3": 15,
        "opponent_1": 19,
    }
    assert env.spawn_index == 11
    assert [call["model_name"] for call in env.set_pose_calls] == env.agents
    assert [call["x"] for call in env.set_pose_calls] == [11.0, 19.0, 3.0, 15.0]


def test_automatic_evaluation_reset_retains_fixed_agent_order(
    monkeypatch,
) -> None:
    env = _automatic_reset_env()
    env.is_eval = True

    def unexpected_shuffle(_agent_ids) -> None:
        raise AssertionError("evaluation must not randomise agent identities")

    monkeypatch.setattr(
        "f1tenth_environments.multi_f1tenth_environment.random.shuffle",
        unexpected_shuffle,
    )

    env._reset_positions()

    assert env.spawn_indices == {
        "f1tenth": 10,
        "opponent_1": 18,
        "opponent_2": 22,
        "opponent_3": 26,
    }
    assert env.spawn_index == 10


def test_evaluation_reset_positions_use_exact_track_and_world_poses() -> None:
    env = _reset_seam_env()
    poses = {
        "f1tenth": {
            "x": 1.0,
            "y": 2.0,
            "z": 0.0,
            "yaw": 0.5,
            "waypoint_index": 12,
        },
        "opponent_1": {
            "x": 1.0,
            "y": 1.4,
            "z": 0.0,
            "yaw": 0.5,
            "waypoint_index": 12,
        },
    }

    env._reset_positions(
        {
            "track_name": "track_b",
            "spawn_poses": poses,
        }
    )

    assert env.current_track == "track_b"
    assert env.current_waypoints is env.tracks["track_b"]
    assert env.current_track_model is env.track_progress_models["track_b"]
    assert env.spawn_index == 12
    assert env.spawn_indices == {"f1tenth": 12, "opponent_1": 12}
    assert env.set_pose_calls == [
        {
            "model_name": "f1tenth",
            "x": 1.0,
            "y": 2.0,
            "z": 0.0,
            "yaw": 0.5,
        },
        {
            "model_name": "opponent_1",
            "x": 1.0,
            "y": 1.4,
            "z": 0.0,
            "yaw": 0.5,
        },
    ]


def test_evaluation_reset_requires_pose_for_every_environment_agent() -> None:
    env = _reset_seam_env()

    with pytest.raises(ValueError, match="must match environment agents"):
        env._reset_positions(
            {
                "track_name": "track_b",
                "spawn_poses": {
                    "f1tenth": {"x": 1.0, "y": 2.0, "yaw": 0.5}
                },
            }
        )


def test_sim_time_accessor_uses_ros_clock_nanoseconds() -> None:
    env = MultiF1TenthEnvironment.__new__(MultiF1TenthEnvironment)
    env.get_clock = lambda: types.SimpleNamespace(
        now=lambda: types.SimpleNamespace(nanoseconds=12_345_000_000)
    )

    assert env._sim_time_seconds() == pytest.approx(12.345)
