import copy
from pathlib import Path

import pytest

from polaris.joint_velocity_smoke import (
    validate_immutable_joint_velocity_smoke,
    validate_joint_velocity_smoke,
)
from scripts import smoke_joint_velocity_controller as controller


class _CloseProbe:
    def __init__(self, *, error=None):
        self.calls = 0
        self.error = error

    def close(self):
        self.calls += 1
        if self.error is not None:
            raise self.error


def _write_valid_child(path: Path, payload: dict):
    child = copy.deepcopy(payload)
    child.pop("completion")
    child["lifecycle"] = {
        "env_close": "complete",
        "simulation_app_close": "pending_child_exit",
        "capture_stage": "kit_child_after_env_close_before_simulation_app_close",
    }
    validated = validate_joint_velocity_smoke(child, require_parent_completion=False)
    controller._write_child_capture(path, validated)


def test_parent_publishes_only_valid_close_capture_after_zero_child_exit(
    tmp_path,
    valid_joint_velocity_smoke_payload,
):
    raw_path = tmp_path / "raw.partial"
    output_path = tmp_path / "joint_velocity_smoke.json"
    _write_valid_child(raw_path, valid_joint_velocity_smoke_payload)

    identity = controller.finalize_child_capture(
        raw_path, output_path, child_exit_code=0
    )
    assert identity["status"] == "pass"
    assert validate_immutable_joint_velocity_smoke(output_path) == identity
    assert raw_path.exists()
    assert (raw_path.stat().st_mode & 0o777) == 0o400


def test_nonzero_or_incomplete_child_never_publishes_pass_artifact(
    tmp_path,
    valid_joint_velocity_smoke_payload,
):
    raw_path = tmp_path / "raw.partial"
    output_path = tmp_path / "joint_velocity_smoke.json"
    _write_valid_child(raw_path, valid_joint_velocity_smoke_payload)

    with pytest.raises(ValueError, match="exited nonzero"):
        controller.finalize_child_capture(raw_path, output_path, child_exit_code=1)
    assert not output_path.exists()

    raw_path.chmod(0o600)
    with pytest.raises(ValueError, match="mode-0400"):
        controller.finalize_child_capture(raw_path, output_path, child_exit_code=0)
    assert not output_path.exists()


def test_simulation_app_close_is_attempted_when_environment_close_raises():
    env = _CloseProbe(error=RuntimeError("env close failed"))
    app = _CloseProbe()
    with pytest.raises(RuntimeError, match="env close failed"):
        controller._close_kit_resources(env, app)
    assert env.calls == 1
    assert app.calls == 1

    env = _CloseProbe()
    app = _CloseProbe(error=RuntimeError("app close failed"))
    with pytest.raises(RuntimeError, match="app close failed"):
        controller._close_kit_resources(env, app)
    assert env.calls == 1
    assert app.calls == 1
