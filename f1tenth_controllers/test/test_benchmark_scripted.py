from types import SimpleNamespace

import numpy as np

from f1tenth_controllers.benchmark.scripted import (
    PurePursuitAdapter,
    _offset_path,
)


class FakeState:
    @staticmethod
    def position_xy():
        return 0.0, 0.0

    @staticmethod
    def quaternion_wxyz():
        return 1.0, 0.0, 0.0, 0.0


def test_offset_path_uses_waypoint_left_normal() -> None:
    path = _offset_path([(1.0, 2.0, 0.0, 0)], 0.3)

    np.testing.assert_allclose(path, np.asarray([[1.0, 2.3]]))


def test_existing_pure_pursuit_adapter_returns_declared_speed() -> None:
    environment = SimpleNamespace(
        previous_state_data={"f1tenth": FakeState()}
    )
    adapter = PurePursuitAdapter(
        environment=environment,
        agent="f1tenth",
        waypoints=[
            (0.0, 0.0, 0.0, 0),
            (1.0, 0.0, 0.0, 1),
            (2.0, 0.0, 0.0, 2),
        ],
        speed_mps=0.8,
        lateral_offset_m=0.0,
    )

    action = adapter.act(np.zeros(11, dtype=np.float32))

    assert action.shape == (2,)
    assert np.all(np.isfinite(action))
    assert action[0] == np.float32(0.8)
