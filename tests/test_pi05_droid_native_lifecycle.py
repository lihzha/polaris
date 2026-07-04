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


def test_lifecycle_always_closes_simulation_and_propagates_env_failure():
    events = []
    env = _Closer("env.close", events, ValueError("env failed"))
    simulation = _Closer("simulation.close", events)
    lifecycle = NativeEvaluatorLifecycle(simulation)
    lifecycle.bind_environment(env)
    with pytest.raises(ValueError, match="env failed"):
        lifecycle.close()
    assert events == ["env.close", "simulation.close"]
    assert env.calls == simulation.calls == 1


def test_lifecycle_always_closes_simulation_and_propagates_publish_failure(tmp_path):
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
    with pytest.raises(OSError, match="publish failed"):
        lifecycle.close()
    assert events == ["env.close", ("publish", ready, payload), "simulation.close"]
    assert env.calls == simulation.calls == 1


def test_lifecycle_preserves_both_environment_and_simulation_close_failures():
    events = []
    env_error = ValueError("env failed")
    simulation_error = OSError("simulation failed")
    env = _Closer("env.close", events, env_error)
    simulation = _Closer("simulation.close", events, simulation_error)
    lifecycle = NativeEvaluatorLifecycle(simulation)
    lifecycle.bind_environment(env)
    with pytest.raises(BaseExceptionGroup) as captured:
        lifecycle.close()
    assert captured.value.exceptions == (env_error, simulation_error)
    assert events == ["env.close", "simulation.close"]
    assert env.calls == simulation.calls == 1


def test_lifecycle_without_environment_still_closes_simulation():
    events = []
    simulation = _Closer("simulation.close", events)
    lifecycle = NativeEvaluatorLifecycle(simulation)
    lifecycle.close()
    assert events == ["simulation.close"]
