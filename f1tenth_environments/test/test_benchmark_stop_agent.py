import pytest

from f1tenth_environments.multi_f1tenth_environment import (
    MultiF1TenthEnvironment,
)


class FakePublisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, message) -> None:
        self.messages.append(message)


def test_stop_agent_publishes_one_zero_twist() -> None:
    environment = MultiF1TenthEnvironment.__new__(
        MultiF1TenthEnvironment
    )
    publisher = FakePublisher()
    environment.cmd_vel_pubs = {"f1tenth": publisher}

    environment.stop_agent("f1tenth")

    assert len(publisher.messages) == 1
    assert publisher.messages[0].linear.x == pytest.approx(0.0)
    assert publisher.messages[0].angular.z == pytest.approx(0.0)


def test_stop_agent_rejects_unknown_name() -> None:
    environment = MultiF1TenthEnvironment.__new__(
        MultiF1TenthEnvironment
    )
    environment.cmd_vel_pubs = {}

    with pytest.raises(ValueError, match="Unknown agent"):
        environment.stop_agent("missing")
