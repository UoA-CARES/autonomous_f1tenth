from f1tenth_controllers.benchmark.manifest import (
    build_run_manifest,
    git_revisions,
)


class FakePolicy:
    def __init__(self, algorithm: str) -> None:
        self.algorithm = algorithm

    def manifest_entry(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "sha256": f"{self.algorithm}_hash",
            "observation_size": 11,
            "action_size": 2,
        }


def test_manifest_records_provenance_schedule_and_fidelity_differences(
) -> None:
    config = {
        "experiment_name": "benchmark",
        "config_sha256": "config_hash",
        "track": {"identifier": "test_track_02_350"},
        "lap_monitor": {
            "projection_jump_policy": "discard_progress_and_reanchor",
            "teleport_policy": (
                "world_displacement_exceeds_projection_bound"
            ),
        },
        "environment": {
            "action_pipeline": (
                "training_raw_policy_output_direct_to_environment_clip"
            ),
            "simulator_seed_supported": False,
            "position_speed_multiplier": 1.0,
            "max_steps": 6000,
            "timeout_sim_seconds": 600.0,
        },
        "runtime": {
            "evaluation_service_timeout_wall_seconds": 5.0,
        },
    }
    repositories = {
        "autonomous_f1tenth": {
            "revision": "abc",
            "dirty": False,
        },
        "cares_reinforcement_learning": {
            "revision": "def",
            "dirty": True,
        },
    }

    manifest = build_run_manifest(
        config=config,
        policies={
            "MATD3": FakePolicy("MATD3"),
            "ISAC": FakePolicy("ISAC"),
        },
        repository_states=repositories,
        lap_length_m=179.0,
        time_trial_ids=["trial_1"],
        heat_ids=["heat_1", "heat_2"],
        start_resolution={
            "strategy": "sha256_seed_salt_modulo_waypoint_count",
            "time_trials_by_seed": {"1000": {"waypoint_index": 7}},
            "head_to_head_by_seed": {"42": {"waypoint_index": 12}},
        },
    )

    assert manifest["campaign"]["time_trial_count"] == 1
    assert manifest["campaign"]["head_to_head_heat_count"] == 2
    assert manifest["git_revisions"] == {
        "autonomous_f1tenth": "abc",
        "cares_reinforcement_learning": "def",
    }
    assert manifest["checkpoints"]["ISAC"]["observation_size"] == 11
    assert manifest["track_resolution"]["lap_length_m"] == 179.0
    assert (
        manifest["track_resolution"]["start_resolution"]["strategy"]
        == "sha256_seed_salt_modulo_waypoint_count"
    )
    assert not manifest["simulation_fidelity"]["gazebo_physics_changed"]
    differences = manifest["simulation_fidelity"][
        "evaluation_protocol_differences"
    ]
    assert {difference["setting"] for difference in differences} == {
        "seeded_random_centreline_and_lead_chaser_spawn_poses",
        "projection_jump_handling",
        "position_speed_multiplier",
        "episode_limit",
        "evaluation_service_timeout_wall_seconds",
    }


def test_git_revisions_are_sorted() -> None:
    assert list(
        git_revisions(
            {
                "f1tenth": {"revision": "2"},
                "autonomous_f1tenth": {"revision": "1"},
            }
        )
    ) == ["autonomous_f1tenth", "f1tenth"]
