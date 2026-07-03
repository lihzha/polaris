import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from polaris.joint_velocity_smoke import (
    validate_immutable_joint_velocity_smoke,
    validate_joint_velocity_smoke,
)
from scripts import smoke_joint_velocity_controller as controller
from scripts.polaris import (
    finalize_pi05_droid_jointvelocity_controller_smoke as finalizer,
)


def _write_valid_child(raw_path: Path, ready_path: Path, payload: dict):
    child = copy.deepcopy(payload)
    child.pop("completion")
    child["lifecycle"] = {
        "env_close": "complete",
        "simulation_app_close": "pending_child_exit",
        "capture_stage": "kit_child_after_env_close_before_simulation_app_close",
    }
    validated = validate_joint_velocity_smoke(child, require_parent_completion=False)
    raw_bytes = controller._write_immutable_json(raw_path, validated)
    controller._write_immutable_json(
        ready_path, controller._child_ready_payload(raw_path, raw_bytes)
    )


def test_parent_publishes_only_valid_close_capture_after_zero_child_exit(
    tmp_path,
    valid_joint_velocity_smoke_payload,
):
    raw_path = tmp_path / "raw.partial"
    ready_path = tmp_path / "raw.partial.ready.json"
    failure_path = tmp_path / "raw.partial.failure.json"
    output_path = tmp_path / "joint_velocity_smoke.json"
    _write_valid_child(raw_path, ready_path, valid_joint_velocity_smoke_payload)

    identity = controller.finalize_child_capture(
        raw_path, ready_path, failure_path, output_path, child_exit_code=0
    )
    assert identity["status"] == "pass"
    assert validate_immutable_joint_velocity_smoke(output_path) == identity
    assert raw_path.exists()
    assert (raw_path.stat().st_mode & 0o777) == 0o444
    assert (ready_path.stat().st_mode & 0o777) == 0o444


def test_nonzero_or_incomplete_child_never_publishes_pass_artifact(
    tmp_path,
    valid_joint_velocity_smoke_payload,
):
    raw_path = tmp_path / "raw.partial"
    ready_path = tmp_path / "raw.partial.ready.json"
    failure_path = tmp_path / "raw.partial.failure.json"
    output_path = tmp_path / "joint_velocity_smoke.json"
    _write_valid_child(raw_path, ready_path, valid_joint_velocity_smoke_payload)

    with pytest.raises(ValueError, match="exited nonzero"):
        controller.finalize_child_capture(
            raw_path, ready_path, failure_path, output_path, child_exit_code=1
        )
    assert not output_path.exists()

    raw_path.chmod(0o600)
    with pytest.raises(ValueError, match="mode-0444"):
        controller.finalize_child_capture(
            raw_path, ready_path, failure_path, output_path, child_exit_code=0
        )
    assert not output_path.exists()


def test_parent_rejects_zero_exit_without_both_bound_child_artifacts(
    tmp_path,
    valid_joint_velocity_smoke_payload,
):
    raw_path = tmp_path / "raw.partial"
    ready_path = tmp_path / "raw.partial.ready.json"
    failure_path = tmp_path / "raw.partial.failure.json"
    output_path = tmp_path / "joint_velocity_smoke.json"

    with pytest.raises(ValueError, match="raw capture"):
        controller.finalize_child_capture(
            raw_path, ready_path, failure_path, output_path, child_exit_code=0
        )
    child = copy.deepcopy(valid_joint_velocity_smoke_payload)
    child.pop("completion")
    child["lifecycle"] = {
        "env_close": "complete",
        "simulation_app_close": "pending_child_exit",
        "capture_stage": "kit_child_after_env_close_before_simulation_app_close",
    }
    controller._write_immutable_json(
        raw_path,
        validate_joint_velocity_smoke(child, require_parent_completion=False),
    )
    with pytest.raises(ValueError, match="ready marker"):
        controller.finalize_child_capture(
            raw_path, ready_path, failure_path, output_path, child_exit_code=0
        )
    assert not output_path.exists()


def test_parent_rejects_tampered_ready_binding_and_child_failure(
    tmp_path,
    valid_joint_velocity_smoke_payload,
):
    raw_path = tmp_path / "raw.partial"
    ready_path = tmp_path / "raw.partial.ready.json"
    failure_path = tmp_path / "raw.partial.failure.json"
    output_path = tmp_path / "joint_velocity_smoke.json"
    _write_valid_child(raw_path, ready_path, valid_joint_velocity_smoke_payload)

    ready_path.chmod(0o644)
    ready = json.loads(ready_path.read_bytes())
    ready["raw_result"]["sha256"] = "0" * 64
    ready_path.unlink()
    controller._write_immutable_json(ready_path, ready)
    with pytest.raises(ValueError, match="does not bind"):
        controller.finalize_child_capture(
            raw_path, ready_path, failure_path, output_path, child_exit_code=0
        )

    controller._write_immutable_json(
        failure_path,
        {
            "schema_version": 1,
            "status": "failure",
            "stage": "run_controller_capture",
            "exception": {
                "type": "builtins.RuntimeError",
                "message": "hidden live failure",
                "traceback": "RuntimeError: hidden live failure\n",
            },
        },
    )
    with pytest.raises(ValueError, match="hidden live failure"):
        controller.finalize_child_capture(
            raw_path, ready_path, failure_path, output_path, child_exit_code=0
        )
    assert not output_path.exists()


@pytest.mark.parametrize("error", [RuntimeError("capture failed"), SystemExit(0)])
def test_child_failure_status_forces_hard_nonzero_without_kit_close(
    tmp_path, monkeypatch, error
):
    failure_path = tmp_path / "failure.json"
    hard_exit_codes = []

    def hard_exit(code):
        hard_exit_codes.append(code)
        raise RuntimeError("hard-exit sentinel")

    monkeypatch.setattr(controller.os, "_exit", hard_exit)
    with pytest.raises(RuntimeError, match="hard-exit sentinel"):
        controller._abort_kit_child(
            failure_path, stage="run_controller_capture", error=error
        )
    assert hard_exit_codes == [1]
    failure = json.loads(failure_path.read_bytes())
    assert failure["status"] == "failure"
    assert failure["stage"] == "run_controller_capture"
    assert (failure_path.stat().st_mode & 0o777) == 0o444


def test_success_source_orders_ready_publication_immediately_before_kit_close():
    source = Path(controller.__file__).read_text(encoding="utf-8")
    ready_index = source.index(
        "_write_immutable_json(\n            args_cli.ready_json,"
    )
    close_index = source.index("simulation_app.close()", ready_index)
    abort_index = source.index("except BaseException as error:", close_index)
    assert ready_index < close_index < abort_index
    assert "_close_kit_resources" not in source
    assert "os._exit(1)" in source


def test_systemexit_zero_is_nonzero_in_a_real_child_process(tmp_path):
    repository = Path(controller.__file__).parents[1]
    failure_path = tmp_path / "systemexit.failure.json"
    code = (
        "from pathlib import Path; "
        "from scripts.smoke_joint_velocity_controller import _abort_kit_child; "
        f"_abort_kit_child(Path({str(failure_path)!r}), "
        "stage='adversarial_systemexit_zero', error=SystemExit(0))"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{repository / 'src'}:{repository}"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    failure = json.loads(failure_path.read_bytes())
    assert failure["exception"]["type"] == "builtins.SystemExit"
    assert failure["exception"]["message"] == "0"


def test_immutable_reader_rejects_links_and_noncanonical_json(tmp_path):
    original = tmp_path / "original.json"
    controller._write_immutable_json(original, {"value": 1})
    hardlink = tmp_path / "hardlink.json"
    os.link(original, hardlink)
    with pytest.raises(ValueError, match="one mode-0444 regular link"):
        controller._read_immutable_json(original, "hardlinked status")

    original.unlink()
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(hardlink)
    with pytest.raises(ValueError, match="readable regular file"):
        controller._read_immutable_json(symlink, "symlinked status")

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(b'{"value":1,"value":1}\n')
    duplicate.chmod(0o444)
    with pytest.raises(ValueError, match="canonical JSON"):
        controller._read_immutable_json(duplicate, "duplicate-key status")


@pytest.mark.parametrize(
    "reader", [controller._read_immutable_json, finalizer._read_canonical_json]
)
def test_immutable_readers_reject_same_size_in_place_mutation(
    tmp_path, monkeypatch, reader
):
    path = tmp_path / "same-size.json"
    controller._write_immutable_json(path, {"value": 1})
    original_stat = path.stat()
    real_read = os.read
    mutated = False

    def read_then_mutate(descriptor, size):
        nonlocal mutated
        chunk = real_read(descriptor, size)
        if chunk and not mutated:
            mutated = True
            replacement = path.read_bytes().replace(b'"value":1', b'"value":2')
            assert len(replacement) == original_stat.st_size
            path.chmod(0o644)
            path.write_bytes(replacement)
            path.chmod(0o444)
            os.utime(
                path,
                ns=(
                    original_stat.st_atime_ns,
                    original_stat.st_mtime_ns + 1_000_000_000,
                ),
            )
        return chunk

    monkeypatch.setattr(os, "read", read_then_mutate)
    with pytest.raises(ValueError, match="changed while it was being read"):
        reader(path, "same-size mutation")
