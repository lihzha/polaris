import pytest

from polaris.pi05_droid_native_lifecycle import NativeEvaluatorLifecycle


class _Closer:
    def __init__(self, label, events, error=None):
        self.label = label
        self.events = events
        self.error = error
        self.calls = 0

    def close(self):
        self.calls += 1
        self.events.append(self.label)
        if self.error is not None:
            raise self.error


def test_lifecycle_closes_env_publishes_ready_then_closes_simulation_once(tmp_path):
    events = []
    env = _Closer("env.close", events)
    simulation = _Closer("simulation.close", events)
    lifecycle = NativeEvaluatorLifecycle(simulation)
    lifecycle.bind_environment(env)

    def publish(path, payload):
        events.append(("publish", path, payload))
        return {"path": str(path)}

    payload = {"status": "simulation_app_close_pending"}
    ready = tmp_path / "ready.json"
    lifecycle.prepare_close_ready(publish, ready, payload)
    lifecycle.close()
    assert events == ["env.close", ("publish", ready, payload), "simulation.close"]
    assert env.calls == simulation.calls == 1
    with pytest.raises(RuntimeError, match="more than once"):
        lifecycle.close()


@pytest.mark.parametrize("env_error", [ValueError("env failed"), SystemExit(0)])
def test_lifecycle_env_failure_never_enters_zero_masking_simulation_close(env_error):
    events = []
    env = _Closer("env.close", events, env_error)
    simulation = _Closer("simulation.close", events)
    lifecycle = NativeEvaluatorLifecycle(simulation)
    lifecycle.bind_environment(env)
    with pytest.raises(RuntimeError, match="pre-Kit cleanup failed") as captured:
        lifecycle.close()
    assert captured.value.__cause__ is env_error
    assert events == ["env.close"]
    assert env.calls == 1
    assert simulation.calls == 0


def test_lifecycle_publish_failure_never_enters_zero_masking_simulation_close(
    tmp_path,
):
    events = []
    env = _Closer("env.close", events)
    simulation = _Closer("simulation.close", events)
    lifecycle = NativeEvaluatorLifecycle(simulation)
    lifecycle.bind_environment(env)

    def publish(path, payload):
        events.append(("publish", path, payload))
        raise OSError("publish failed")

    payload = {"status": "simulation_app_close_pending"}
    ready = tmp_path / "ready.json"
    lifecycle.prepare_close_ready(publish, ready, payload)
    with pytest.raises(RuntimeError, match="pre-Kit cleanup failed") as captured:
        lifecycle.close()
    assert isinstance(captured.value.__cause__, OSError)
    assert events == ["env.close", ("publish", ready, payload)]
    assert env.calls == 1
    assert simulation.calls == 0


def test_lifecycle_env_failure_does_not_observe_simulation_close_failure():
    events = []
    env_error = ValueError("env failed")
    simulation_error = OSError("simulation failed")
    env = _Closer("env.close", events, env_error)
    simulation = _Closer("simulation.close", events, simulation_error)
    lifecycle = NativeEvaluatorLifecycle(simulation)
    lifecycle.bind_environment(env)
    with pytest.raises(RuntimeError, match="pre-Kit cleanup failed") as captured:
        lifecycle.close()
    assert captured.value.__cause__ is env_error
    assert events == ["env.close"]
    assert env.calls == 1
    assert simulation.calls == 0


def test_lifecycle_without_environment_still_closes_simulation():
    events = []
    simulation = _Closer("simulation.close", events)
    lifecycle = NativeEvaluatorLifecycle(simulation)
    lifecycle.close()
    assert events == ["simulation.close"]
