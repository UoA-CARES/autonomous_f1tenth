from types import SimpleNamespace

import pytest

from f1tenth_environments import f1tenth_environment as environment_module
from f1tenth_environments.f1tenth_environment import F1tenthEnvironment


class FakeFuture:
    def __init__(self, done: bool) -> None:
        self._done = done
        self.cancelled = False

    def done(self) -> bool:
        return self._done

    def cancel(self) -> None:
        self.cancelled = True


def test_training_service_wait_remains_unbounded(monkeypatch) -> None:
    environment = F1tenthEnvironment.__new__(F1tenthEnvironment)
    calls = []
    monkeypatch.setattr(
        environment_module.rclpy,
        "spin_until_future_complete",
        lambda node, future, timeout_sec=None: calls.append(timeout_sec),
    )

    environment._wait_for_evaluation_service(
        FakeFuture(done=True),
        operation="test",
    )

    assert calls == [None]


def test_evaluation_service_wait_times_out_and_cancels(monkeypatch) -> None:
    environment = F1tenthEnvironment.__new__(F1tenthEnvironment)
    environment.evaluation_service_timeout_s = 5.0
    future = FakeFuture(done=False)
    calls = []
    monkeypatch.setattr(
        environment_module.rclpy,
        "spin_until_future_complete",
        lambda node, pending, timeout_sec=None: calls.append(
            SimpleNamespace(
                pending=pending,
                timeout_sec=timeout_sec,
            )
        ),
    )

    with pytest.raises(TimeoutError, match="5.0 wall seconds"):
        environment._wait_for_evaluation_service(
            future,
            operation="world control",
        )

    assert calls[0].timeout_sec == 5.0
    assert future.cancelled
