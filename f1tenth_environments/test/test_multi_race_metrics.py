import importlib
import sys
import types

import pytest


def _install_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    sys.modules[name] = module
    return module


def _noop(*args, **kwargs):
    return None


def _install_ros_stubs() -> None:
    geometry_msgs = _install_module("geometry_msgs")
    geometry_msgs.msg = _install_module("geometry_msgs.msg")
    geometry_msgs.msg.Point = type("Point", (), {})
    geometry_msgs.msg.Pose = type("Pose", (), {})
    geometry_msgs.msg.Twist = type("Twist", (), {})

    message_filters = _install_module("message_filters")
    message_filters.ApproximateTimeSynchronizer = type(
        "ApproximateTimeSynchronizer",
        (),
        {"__init__": _noop, "registerCallback": _noop},
    )
    message_filters.Subscriber = type(
        "Subscriber",
        (),
        {"__init__": _noop},
    )

    nav_msgs = _install_module("nav_msgs")
    nav_msgs.msg = _install_module("nav_msgs.msg")
    nav_msgs.msg.Odometry = type("Odometry", (), {})

    rclpy = _install_module("rclpy")
    rclpy.spin_once = _noop
    rclpy.spin_until_future_complete = _noop
    rclpy.node = _install_module("rclpy.node")
    rclpy.node.Node = type("Node", (), {})
    rclpy.qos = _install_module("rclpy.qos")
    rclpy.qos.QoSProfile = type(
        "QoSProfile",
        (),
        {"__init__": _noop},
    )
    rclpy.qos.qos_profile_sensor_data = object()

    ros_gz_interfaces = _install_module("ros_gz_interfaces")
    ros_gz_interfaces.msg = _install_module("ros_gz_interfaces.msg")
    ros_gz_interfaces.msg.Entity = type(
        "Entity",
        (),
        {"MODEL": 1},
    )
    ros_gz_interfaces.srv = _install_module("ros_gz_interfaces.srv")
    ros_gz_interfaces.srv.ControlWorld = type(
        "ControlWorld",
        (),
        {"Request": type("Request", (), {})},
    )
    ros_gz_interfaces.srv.SetEntityPose = type(
        "SetEntityPose",
        (),
        {"Request": type("Request", (), {})},
    )

    sensor_msgs = _install_module("sensor_msgs")
    sensor_msgs.msg = _install_module("sensor_msgs.msg")
    sensor_msgs.msg.LaserScan = type("LaserScan", (), {})

    pettingzoo = _install_module("pettingzoo")
    pettingzoo.ParallelEnv = type("ParallelEnv", (), {})

    gymnasium = _install_module("gymnasium")
    gymnasium.spaces = types.SimpleNamespace(Box=type("Box", (), {}))


_install_ros_stubs()
MultiF1TenthEnvironment = importlib.import_module(
    "f1tenth_environments.multi_f1tenth_environment"
).MultiF1TenthEnvironment
F1tenthEnvironment = importlib.import_module(
    "f1tenth_environments.f1tenth_environment"
).F1tenthEnvironment




class FakeTrackProgressModel:
    waypoint_lap_length = 40.0

    def track_distance_from_world_coord(self, position) -> float:
        return float(position[0]) % self.waypoint_lap_length

    def forward_distance_between_track_distances(
        self, from_distance: float, to_distance: float
    ) -> float:
        return float((to_distance - from_distance) % self.waypoint_lap_length)

    def signed_delta_between_track_distances(
        self, from_distance: float, to_distance: float
    ) -> float:
        forward = self.forward_distance_between_track_distances(
            from_distance, to_distance
        )
        if forward <= self.waypoint_lap_length / 2.0:
            return forward
        return forward - self.waypoint_lap_length


class FakeStateData:
    def __init__(self, x: float, y: float, linear_velocity: float = 0.0):
        self._position = (x, y)
        self._linear_velocity = linear_velocity
        self.state = []

    def position_xy(self) -> tuple[float, float]:
        return self._position

    def linear_velocity(self) -> float:
        return self._linear_velocity


def _metric_env() -> MultiF1TenthEnvironment:
    env = MultiF1TenthEnvironment.__new__(MultiF1TenthEnvironment)
    env.car_name = "f1tenth"
    env.agents = ["f1tenth", "opponent_0", "opponent_1"]
    env.current_waypoints = [
        (0.0, 0.0, 0.0, 0),
        (10.0, 0.0, 0.0, 1),
        (20.0, 0.0, 0.0, 2),
        (30.0, 0.0, 0.0, 3),
    ]
    env.goal_reach_radius_m = 0.5
    env.current_track_model = FakeTrackProgressModel()
    env.race_origin_track_distance = 0.0
    env.spawn_indices = {"f1tenth": 0, "opponent_0": 0, "opponent_1": 0}
    env.goals_reached = {"f1tenth": 0, "opponent_0": 0, "opponent_1": 0}
    env.agent_goals = {
        "f1tenth": (10.0, 0.0),
        "opponent_0": (10.0, 0.0),
        "opponent_1": (10.0, 0.0),
    }
    env.overtake_counts = {agent: 0 for agent in env.agents}
    env.pole_position_steps = {agent: 0 for agent in env.agents}
    env.total_linear_velocity = {agent: 0.0 for agent in env.agents}
    env.step_counter = 1
    return env


def test_multi_agent_race_position_uses_polyline_track_distance() -> None:
    env = _metric_env()
    env.race_origin_track_distance = 2.0

    assert env._get_agent_race_position(
        "f1tenth", FakeStateData(5.0, 0.0)
    ) == pytest.approx(3.0)


def test_multi_agent_race_position_unwraps_lap_boundary() -> None:
    env = _metric_env()
    previous_positions = {"f1tenth": 39.5, "opponent_0": 38.0}
    wrapped_positions = {"f1tenth": 0.5, "opponent_0": 39.0}

    positions = env._unwrap_race_positions(previous_positions, wrapped_positions)

    assert positions["f1tenth"] == pytest.approx(40.5)
    assert positions["opponent_0"] == pytest.approx(39.0)


def test_distance_to_opponents_uses_single_agent_sign_convention() -> None:
    env = _metric_env()
    race_positions = {
        "f1tenth": 1.5,
        "opponent_0": 2.25,
        "opponent_1": 0.25,
    }

    distances = env._get_opponent_distance_info("f1tenth", race_positions)

    assert distances["opponent_0"] == pytest.approx(0.75)
    assert distances["opponent_1"] == pytest.approx(-1.25)


def test_overtake_count_uses_previous_and_current_race_positions() -> None:
    env = _metric_env()
    previous_positions = {
        "f1tenth": 1.0,
        "opponent_0": 2.0,
        "opponent_1": 0.0,
    }
    current_positions = {
        "f1tenth": 2.5,
        "opponent_0": 2.4,
        "opponent_1": 0.5,
    }

    assert env._count_new_agent_overtakes(
        "f1tenth", previous_positions, current_positions
    ) == 1
    assert env._count_new_agent_overtakes(
        "opponent_0", previous_positions, current_positions
    ) == 0


def test_pole_position_requires_agent_to_lead_all_others() -> None:
    env = _metric_env()

    assert env._is_agent_in_pole_position(
        "f1tenth",
        {"f1tenth": 3.0, "opponent_0": 2.0, "opponent_1": 1.0},
    )
    assert not env._is_agent_in_pole_position(
        "f1tenth",
        {"f1tenth": 2.0, "opponent_0": 2.0, "opponent_1": 1.0},
    )


def test_goal_progress_advances_to_next_unreached_waypoint() -> None:
    env = _metric_env()
    env.agents = ["f1tenth"]
    env.spawn_indices = {"f1tenth": 1}
    env.goals_reached = {"f1tenth": 0}
    env.agent_goals = {"f1tenth": (20.0, 0.0)}

    env._update_goal_progress("f1tenth", FakeStateData(20.0, 0.0))

    assert env.goals_reached["f1tenth"] == 1
    assert env.agent_goals["f1tenth"] == (30.0, 0.0)


def test_metric_info_uses_race_positions_for_exact_reported_keys() -> None:
    env = _metric_env()
    env.overtake_counts["f1tenth"] = 2
    race_positions = {
        "f1tenth": 3.0,
        "opponent_0": 4.5,
        "opponent_1": 1.5,
    }

    info = env._build_agent_metric_info(
        "f1tenth",
        FakeStateData(0.0, 0.0, linear_velocity=2.0),
        race_positions,
        True,
        False,
        {"f1tenth": (0.0, 0.0), "opponent_0": (4.5, 0.0), "opponent_1": (1.5, 0.0)},
    )

    assert info["agent_track_position"] == pytest.approx(3.0)
    assert info["distance_to_opponents"] == {
        "opponent_0": 1.5,
        "opponent_1": -1.5,
    }
    assert info["distance_to_opponent_0"] == pytest.approx(1.5)
    assert info["opponent_0_track_position"] == pytest.approx(4.5)
    assert info["overtakes"] == 2
    assert info["agent_overtakes"] == 2
    assert info["overtakes_live"] == 2
    assert info["overtakes_per_episode"] == 2
    assert info["number_of_overtakes_per_episode"] == 2
    assert info["time_in_pole_position"] == 0


def test_metric_info_reports_live_overtakes_before_episode_end() -> None:
    env = _metric_env()
    env.overtake_counts["f1tenth"] = 2
    race_positions = {
        "f1tenth": 3.0,
        "opponent_0": 4.5,
        "opponent_1": 1.5,
    }

    info = env._build_agent_metric_info(
        "f1tenth",
        FakeStateData(0.0, 0.0, linear_velocity=2.0),
        race_positions,
        False,
        False,
        {"f1tenth": (0.0, 0.0), "opponent_0": (4.5, 0.0), "opponent_1": (1.5, 0.0)},
    )

    assert info["overtakes"] == 2
    assert info["agent_overtakes"] == 2
    assert info["overtakes_live"] == 2
    assert info["overtakes_per_episode"] == 0
    assert info["number_of_overtakes_per_episode"] == 0


def test_single_agent_transition_keeps_overtakes_terminal_only() -> None:
    env = F1tenthEnvironment.__new__(F1tenthEnvironment)
    env.car_name = "f1tenth"
    env.opponent_car_names = ["opponent_0"]
    env.latest_opponent_odometries = {}
    env.previous_state_data = FakeStateData(0.0, 0.0)
    env.previous_race_positions = {"f1tenth": 1.0, "opponent_0": 2.0}
    env.overtakes_per_episode = 1
    env.overtake_rear_margin_m = 0.25
    env.overtake_front_margin_m = 0.25
    env.overtake_eligible_opponents = {"opponent_0"}
    env.overtaken_opponents = set()
    env.pole_position_steps = 0
    env.total_linear_velocity = 0.0
    env.step_counter = 1

    current_state = FakeStateData(0.0, 0.0, linear_velocity=2.0)
    race_positions = {"f1tenth": 2.7, "opponent_0": 2.0}

    env._build_state_data = lambda: current_state
    env._set_simulation_paused = _noop
    env._compute_reward = lambda previous_state_data, current_state_data: (0.0, {})
    env._get_race_positions = lambda state_data: race_positions
    env._unwrap_race_positions = lambda previous, current: current
    env._is_agent_in_pole_position = lambda positions: False
    env._is_terminated = lambda state_data: False
    env._is_truncated = lambda: False
    env._get_opponent_distance_info = lambda positions: {"opponent_0": -0.7}

    _, _, _, _, info = env._transition()

    assert env.overtakes_per_episode == 2
    assert info["overtakes_per_episode"] == 0
    assert "overtakes" not in info
    assert "agent_overtakes" not in info
    assert "overtakes_live" not in info
    assert "number_of_overtakes_per_episode" not in info

