from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from polaris import app_launcher_startup_diagnostic as diagnostic


ROOT = Path(__file__).parents[1]
GPU_UUID = "GPU-01234567-89ab-cdef-0123-456789abcdef"
SOURCE_TREE_SHA256 = "1" * 64
SOURCE_APPROVAL_SHA256 = "2" * 64
IMPLEMENTATION_COMMIT = "3" * 40


def runtime_environment(**updates: str) -> dict[str, str]:
    environment = {
        "SLURM_JOB_ID": "12345",
        "SLURM_STEP_ID": "0",
        "SLURM_JOB_GPUS": "2",
        "SLURM_STEP_GPUS": "2",
        "CUDA_VISIBLE_DEVICES": "0",
        "NVIDIA_VISIBLE_DEVICES": GPU_UUID,
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        "BATCH_VERIFIED_POLARIS_SOURCE_TREE_SHA256": SOURCE_TREE_SHA256,
        "SOURCE_APPROVAL_SHA256": SOURCE_APPROVAL_SHA256,
        "POLARIS_IMPLEMENTATION_COMMIT": IMPLEMENTATION_COMMIT,
    }
    environment.update(updates)
    return environment


def nvidia_smi_output(*, uuid: str = GPU_UUID, minor: int = 2) -> str:
    return f"{uuid}, NVIDIA L40S, 580.105.08, {minor}\n"


def cgroup_text(*, job_id: str = "12345", step_id: str = "0") -> str:
    return f"0::/slurm/uid_1000/job_{job_id}/step_{step_id}/task_0\n"


def physical_device_node(*, index: int = 2, minor: int = 2) -> dict[str, object]:
    return {
        "path": f"/dev/nvidia{index}",
        "mode": "0660",
        "file_type": "character",
        "device_major": 195,
        "device_minor": minor,
    }


def synthetic_runtime(*, pid: int = 4444) -> dict[str, object]:
    return diagnostic.capture_runtime_context(
        python_argv=["scripts/eval.py"],
        source_root=ROOT,
        expected_gpu_uuid=GPU_UUID,
        environ=runtime_environment(),
        nvidia_smi_output=nvidia_smi_output(),
        cgroup_text=cgroup_text(),
        device_nodes=[physical_device_node()],
        pid=pid,
        ppid=3333,
        executable="/.venv/bin/python",
        cwd=ROOT,
    )


def preexec_value(runtime: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "profile": diagnostic.PREEXEC_PROFILE,
        "status": "captured_before_public_eval_exec",
        "startup_diagnostic": diagnostic.STARTUP_DIAGNOSTIC_MODE,
        "runtime": runtime,
        "launcher_argv": ["app_launcher_startup_diagnostic.py"],
        "target_argv": ["/.venv/bin/python", "scripts/eval.py"],
        "zero_work_counters": dict(diagnostic.ZERO_WORK_COUNTERS),
        "bounded_diagnostic_counts": {
            "nvidia_smi_invocations": 1,
            "preexec_artifacts": 1,
            "preclose_artifacts": 0,
            "ready_artifacts": 0,
            "simulation_app_close_calls": 0,
        },
    }


def publish_preexec(path: Path, runtime: dict[str, object]) -> dict[str, object]:
    return diagnostic.publish_immutable_json(path, preexec_value(runtime))


def test_nvidia_smi_requires_exactly_one_expected_l40s() -> None:
    value = diagnostic.parse_nvidia_smi_output(nvidia_smi_output())
    assert value == {
        "uuid": GPU_UUID,
        "name": "NVIDIA L40S",
        "driver_version": "580.105.08",
        "minor_number": 2,
        "command": list(diagnostic.NVIDIA_SMI_COMMAND),
        "row_count": 1,
    }


@pytest.mark.parametrize(
    "output",
    [
        "",
        nvidia_smi_output()
        + nvidia_smi_output(uuid="GPU-fedcba98-7654-3210-fedc-ba9876543210", minor=3),
        f"{GPU_UUID}, NVIDIA L40S, 580.105.08\n",
        "GPU-not-a-uuid, NVIDIA L40S, 580.105.08, 2\n",
        f"{GPU_UUID}, NVIDIA A100-SXM4-80GB, 580.105.08, 2\n",
        f"{GPU_UUID}, NVIDIA L40S, 580.95.05, 2\n",
        f"{GPU_UUID}, NVIDIA L40S, 580.105.08, N/A\n",
    ],
)
def test_nvidia_smi_rejects_missing_extra_and_malformed_rows(output: str) -> None:
    with pytest.raises(ValueError):
        diagnostic.parse_nvidia_smi_output(output)


def test_cgroup_requires_one_matching_job_step_path() -> None:
    value = diagnostic.parse_cgroup_text(cgroup_text(), job_id="12345", step_id="0")
    assert value["job_step_path"] == "/slurm/uid_1000/job_12345/step_0/task_0"
    assert len(value["records"]) == 1


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "0::/\n",
        cgroup_text(job_id="99999"),
        cgroup_text(step_id="1"),
        (cgroup_text() + "1:devices:/slurm/uid_1000/job_12345/step_0/other_task\n"),
        cgroup_text() + cgroup_text(),
        "malformed\n",
    ],
)
def test_cgroup_rejects_missing_multiple_mismatched_and_duplicate_evidence(
    raw: str,
) -> None:
    with pytest.raises(ValueError):
        diagnostic.parse_cgroup_text(raw, job_id="12345", step_id="0")


def test_device_nodes_require_exactly_one_matching_physical_gpu() -> None:
    control = {
        "path": "/dev/nvidiactl",
        "mode": "0660",
        "file_type": "character",
        "device_major": 195,
        "device_minor": 255,
    }
    value = diagnostic.validate_device_nodes(
        [physical_device_node(), control], expected_minor_number=2
    )
    assert value["physical_count"] == 1
    assert value["physical"] == [physical_device_node()]


@pytest.mark.parametrize(
    "records,expected_minor",
    [
        ([], 2),
        ([physical_device_node(), physical_device_node(index=3, minor=3)], 2),
        ([physical_device_node(index=2, minor=3)], 3),
        ([physical_device_node()], 3),
        (
            [
                {
                    **physical_device_node(),
                    "file_type": "other",
                }
            ],
            2,
        ),
    ],
)
def test_device_nodes_reject_missing_extra_and_mismatched_gpu_nodes(
    records: list[dict[str, object]], expected_minor: int
) -> None:
    with pytest.raises(ValueError):
        diagnostic.validate_device_nodes(records, expected_minor_number=expected_minor)


def test_runtime_context_cross_binds_process_step_gpu_cgroup_and_source() -> None:
    value = synthetic_runtime()
    assert value["process"]["pid"] == 4444
    assert value["slurm"]["job_id"] == 12345
    assert value["slurm"]["step_id"] == 0
    assert value["nvidia_smi"]["uuid"] == GPU_UUID
    assert value["device_nodes"]["physical_count"] == 1
    assert value["source"]["eval_script"]["path"] == str(ROOT / "scripts/eval.py")
    assert value["source"]["approval"] == {
        "BATCH_VERIFIED_POLARIS_SOURCE_TREE_SHA256": SOURCE_TREE_SHA256,
        "POLARIS_IMPLEMENTATION_COMMIT": IMPLEMENTATION_COMMIT,
        "SOURCE_APPROVAL_SHA256": SOURCE_APPROVAL_SHA256,
    }


@pytest.mark.parametrize(
    "updates",
    [
        {"SLURM_JOB_ID": ""},
        {"SLURM_JOB_ID": "batch"},
        {"SLURM_STEP_ID": ""},
        {"SLURM_STEP_ID": "batch"},
        {"SLURM_JOB_GPUS": ""},
        {"SLURM_STEP_GPUS": ""},
        {"SLURM_STEP_GPUS": "3"},
        {"SLURM_JOB_GPUS": "3", "SLURM_STEP_GPUS": "3"},
        {"NVIDIA_VISIBLE_DEVICES": "GPU-fedcba98-7654-3210-fedc-ba9876543210"},
    ],
)
def test_runtime_context_rejects_missing_or_mismatched_environment(
    updates: dict[str, str],
) -> None:
    with pytest.raises(ValueError):
        diagnostic.capture_runtime_context(
            python_argv=["scripts/eval.py"],
            source_root=ROOT,
            expected_gpu_uuid=GPU_UUID,
            environ=runtime_environment(**updates),
            nvidia_smi_output=nvidia_smi_output(),
            cgroup_text=cgroup_text(),
            device_nodes=[physical_device_node()],
        )


def test_runtime_context_rejects_nvidia_uuid_mismatch() -> None:
    with pytest.raises(ValueError, match="nvidia-smi GPU UUID"):
        diagnostic.capture_runtime_context(
            python_argv=["scripts/eval.py"],
            source_root=ROOT,
            expected_gpu_uuid=GPU_UUID,
            environ=runtime_environment(),
            nvidia_smi_output=nvidia_smi_output(
                uuid="GPU-fedcba98-7654-3210-fedc-ba9876543210"
            ),
            cgroup_text=cgroup_text(),
            device_nodes=[physical_device_node()],
        )


def test_runtime_context_rejects_diagnostic_module_origin_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    escaped = tmp_path / "app_launcher_startup_diagnostic.py"
    escaped.write_text("# adversarial shadow\n", encoding="utf-8")
    monkeypatch.setattr(diagnostic, "__file__", str(escaped))
    with pytest.raises(ValueError, match="escaped the approved source root"):
        diagnostic.capture_runtime_context(
            python_argv=["scripts/eval.py"],
            source_root=ROOT,
            expected_gpu_uuid=GPU_UUID,
            environ=runtime_environment(),
            nvidia_smi_output=nvidia_smi_output(),
            cgroup_text=cgroup_text(),
            device_nodes=[physical_device_node()],
            executable="/.venv/bin/python",
            cwd=ROOT,
        )


def test_immutable_publication_is_canonical_nonreplacing_and_mode_0444(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.json"
    identity = diagnostic.publish_immutable_json(path, {"z": 1, "a": [True, None]})
    assert path.read_bytes() == b'{"a":[true,null],"z":1}\n'
    metadata = path.stat()
    assert stat.S_IMODE(metadata.st_mode) == 0o444
    assert metadata.st_nlink == 1
    assert identity["sha256"] == diagnostic._sha256(path.read_bytes())
    with pytest.raises(FileExistsError):
        diagnostic.publish_immutable_json(path, {"replacement": True})
    assert path.read_bytes() == b'{"a":[true,null],"z":1}\n'


def test_immutable_publication_rejects_symlink_destination(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("unchanged", encoding="utf-8")
    destination = tmp_path / "artifact.json"
    destination.symlink_to(target)
    with pytest.raises(FileExistsError):
        diagnostic.publish_immutable_json(destination, {"replacement": True})
    assert target.read_text(encoding="utf-8") == "unchanged"


def test_stable_read_rejects_writable_hardlinked_and_noncanonical_json(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.json"
    path.write_text('{"value":1}\n', encoding="ascii")
    path.chmod(0o644)
    with pytest.raises(ValueError, match="mode-0444"):
        diagnostic.stable_read_immutable_json(path)
    path.chmod(0o444)
    linked = tmp_path / "linked.json"
    os.link(path, linked)
    with pytest.raises(ValueError, match="mode-0444"):
        diagnostic.stable_read_immutable_json(path)
    linked.unlink()
    path.chmod(0o644)
    path.write_text('{"value": 1}\n', encoding="ascii")
    path.chmod(0o444)
    with pytest.raises(ValueError, match="canonical"):
        diagnostic.stable_read_immutable_json(path)


def test_target_argv_is_closed_to_exact_public_entrypoint(tmp_path: Path) -> None:
    preexec = tmp_path / "preexec.json"
    preclose = tmp_path / "preclose.json"
    target = [
        "/.venv/bin/python",
        "scripts/eval.py",
        "--startup-diagnostic",
        "app_launcher_only",
        "--startup-diagnostic-preexec-path",
        str(preexec),
        "--startup-diagnostic-preclose-path",
        str(preclose),
        "--startup-diagnostic-expected-gpu-uuid",
        GPU_UUID,
    ]
    assert (
        diagnostic.validate_target_argv(
            target,
            preexec_path=preexec,
            preclose_path=preclose,
            expected_gpu_uuid=GPU_UUID,
        )
        == target
    )
    assert diagnostic.python_argv_after_exec(target) == target[1:]
    for mutation in (
        ["python", *target[1:]],
        [*target, "--startup-diagnostic", "app_launcher_only"],
        [item for item in target if item != "app_launcher_only"],
    ):
        with pytest.raises(ValueError):
            diagnostic.validate_target_argv(
                mutation,
                preexec_path=preexec,
                preclose_path=preclose,
                expected_gpu_uuid=GPU_UUID,
            )


def test_closed_eval_environment_matches_public_shape_plus_step_handoff() -> None:
    value = diagnostic.build_closed_eval_environment(
        inherited=runtime_environment(),
        source_root=Path("/polaris-source"),
        data_root=Path("/physical/PolaRiS-Hub"),
        cache_root=Path("/cache"),
        preexec_sha256="4" * 64,
    )
    assert (
        value["PATH"] == "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    )
    assert value["PYTHONPATH"] == (
        "/polaris-source/src:"
        "/polaris-source/third_party/openpi/packages/openpi-client/src"
    )
    assert value["SLURM_JOB_ID"] == "12345"
    assert value["SLURM_STEP_ID"] == "0"
    assert value["SLURM_JOB_GPUS"] == "2"
    assert value["SLURM_STEP_GPUS"] == "2"
    assert value["NVIDIA_VISIBLE_DEVICES"] == GPU_UUID
    assert "CUDA_VISIBLE_DEVICES" not in value
    assert set(value) == {
        "PATH",
        "LANG",
        "LC_ALL",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "VK_DRIVER_FILES",
        "ACCEPT_EULA",
        "OMNI_KIT_ACCEPT_EULA",
        "PRIVACY_CONSENT",
        "OMNI_KIT_ALLOW_ROOT",
        "PYTHONUNBUFFERED",
        "PYTHONPATH",
        "POLARIS_DATA_PATH",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "HOME",
        "SLURM_JOB_ID",
        "SLURM_STEP_ID",
        "SLURM_JOB_GPUS",
        "SLURM_STEP_GPUS",
        "POLARIS_STARTUP_DIAGNOSTIC_PREEXEC_SHA256",
        *diagnostic.SOURCE_IDENTITY_ENVIRONMENT,
    }


def test_context_continuity_rejects_process_step_gpu_cgroup_device_and_source_drift() -> (
    None
):
    runtime = synthetic_runtime()
    preexec = preexec_value(runtime)
    diagnostic._validate_context_continuity(preexec, runtime)
    mutation_paths = [
        ("process", "pid"),
        ("process", "python_argv"),
        ("slurm", "step_id"),
        ("slurm", "step_gpu_index"),
        ("nvidia_smi", "uuid"),
        ("cgroup", "raw_sha256"),
        ("device_nodes", "physical"),
        ("source", "eval_script", "sha256"),
    ]
    for path in mutation_paths:
        changed = json.loads(json.dumps(runtime))
        cursor = changed
        for component in path[:-1]:
            cursor = cursor[component]
        cursor[path[-1]] = "mismatch"
        with pytest.raises(ValueError, match="continuity mismatch"):
            diagnostic._validate_context_continuity(preexec, changed)


def test_preexec_publishes_bound_evidence_and_builds_exact_exec_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preexec_path = tmp_path / "preexec.json"
    preclose_path = tmp_path / "preclose.json"
    target = [
        "/.venv/bin/python",
        "scripts/eval.py",
        "--startup-diagnostic",
        "app_launcher_only",
        "--startup-diagnostic-preexec-path",
        str(preexec_path),
        "--startup-diagnostic-preclose-path",
        str(preclose_path),
        "--startup-diagnostic-expected-gpu-uuid",
        GPU_UUID,
    ]
    environment = runtime_environment()
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    runtime = synthetic_runtime(pid=os.getpid())
    runtime["process"]["python_argv"] = target[1:]
    captures = []

    def capture_runtime(**kwargs):
        captures.append(kwargs)
        return runtime

    monkeypatch.setattr(diagnostic, "capture_runtime_context", capture_runtime)

    actual_target, closed_environment, identity = diagnostic.prepare_public_eval_exec(
        target_argv=target,
        preexec_path=preexec_path,
        preclose_path=preclose_path,
        expected_gpu_uuid=GPU_UUID,
        source_root=ROOT,
        data_root=Path("/physical/PolaRiS-Hub"),
        cache_root=Path("/cache"),
    )

    assert actual_target == target
    assert captures[0]["python_argv"] == target[1:]
    assert (
        closed_environment["POLARIS_STARTUP_DIAGNOSTIC_PREEXEC_SHA256"]
        == identity["sha256"]
    )
    preexec, _ = diagnostic.stable_read_immutable_json(preexec_path)
    assert preexec["runtime"]["process"]["pid"] == os.getpid()
    assert preexec["runtime"]["process"]["python_argv"] == target[1:]
    assert preexec["target_argv"] == target
    assert preexec["zero_work_counters"] == diagnostic.ZERO_WORK_COUNTERS


def test_real_execve_python_argv_matches_normalized_target(tmp_path: Path) -> None:
    capture = tmp_path / "capture_argv.py"
    capture.write_text(
        "import json, sys\nprint(json.dumps(sys.argv, separators=(',', ':')))\n",
        encoding="utf-8",
    )
    target = [sys.executable, str(capture), "scripts/eval.py", "--headless"]
    launcher = (
        "import os,sys; "
        "os.execve(sys.argv[1], sys.argv[1:], "
        "{'PATH': os.environ.get('PATH', '')})"
    )
    completed = subprocess.run(
        [sys.executable, "-c", launcher, *target],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == target[1:]


def test_cli_uses_execve_without_a_fork(monkeypatch: pytest.MonkeyPatch) -> None:
    target = ["/.venv/bin/python", "scripts/eval.py"]
    environment = {"CLOSED": "1"}
    monkeypatch.setattr(
        diagnostic,
        "_parse_cli",
        lambda _: type(
            "Args",
            (),
            {
                "target": target,
                "preexec_output": Path("/preexec.json"),
                "preclose_output": Path("/preclose.json"),
                "expected_batch_gpu_uuid": GPU_UUID,
                "source_root": Path("/polaris-source"),
                "data_root": Path("/data"),
                "cache_root": Path("/cache"),
            },
        )(),
    )
    monkeypatch.setattr(
        diagnostic,
        "prepare_public_eval_exec",
        lambda **_: (target, environment, {"sha256": "4" * 64}),
    )
    observed = []

    class ExecveCalled(RuntimeError):
        pass

    def fake_execve(path, argv, env):
        observed.append((path, argv, env, os.getpid()))
        raise ExecveCalled

    monkeypatch.setattr(os, "execve", fake_execve)
    with pytest.raises(ExecveCalled):
        diagnostic.main([])
    assert observed == [("/.venv/bin/python", target, environment, os.getpid())]


class FakeSimulationApp:
    def __init__(self, error: BaseException | None = None):
        self.error = error
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.error is not None:
            raise self.error


def test_app_launcher_only_publishes_preclose_ready_then_closes_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = synthetic_runtime(pid=os.getpid())
    preexec_path = tmp_path / "preexec.json"
    preexec_identity = publish_preexec(preexec_path, runtime)
    preclose_path = tmp_path / "preclose.json"
    monkeypatch.setenv(
        "POLARIS_STARTUP_DIAGNOSTIC_PREEXEC_SHA256",
        preexec_identity["sha256"],
    )
    monkeypatch.setattr(diagnostic, "capture_runtime_context", lambda **_: runtime)
    monkeypatch.setattr(diagnostic, "_forbidden_loaded_modules", lambda: [])
    app = FakeSimulationApp()
    artifacts = diagnostic.run_app_launcher_only_diagnostic(
        simulation_app=app,
        preexec_path=preexec_path,
        preclose_path=preclose_path,
        expected_gpu_uuid=GPU_UUID,
    )
    assert app.close_calls == 1
    preclose, _ = diagnostic.stable_read_immutable_json(preclose_path)
    ready, _ = diagnostic.stable_read_immutable_json(
        diagnostic.ready_path_for(preclose_path)
    )
    assert preclose["status"] == "simulation_app_close_pending"
    assert ready["status"] == "ready_for_simulation_app_close"
    assert ready["preclose"] == artifacts["preclose"]
    assert set(preclose["zero_work_counters"].values()) == {0}
    assert preclose["bounded_diagnostic_counts"]["nvidia_smi_invocations"] == 2


@pytest.mark.parametrize("close_error", [RuntimeError("close failed"), SystemExit(0)])
def test_close_failure_is_forced_nonzero_without_false_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    close_error: BaseException,
) -> None:
    runtime = synthetic_runtime(pid=os.getpid())
    preexec_path = tmp_path / "preexec.json"
    preexec_identity = publish_preexec(preexec_path, runtime)
    preclose_path = tmp_path / "preclose.json"
    monkeypatch.setenv(
        "POLARIS_STARTUP_DIAGNOSTIC_PREEXEC_SHA256",
        preexec_identity["sha256"],
    )
    monkeypatch.setattr(diagnostic, "capture_runtime_context", lambda **_: runtime)
    monkeypatch.setattr(diagnostic, "_forbidden_loaded_modules", lambda: [])
    app = FakeSimulationApp(close_error)
    with pytest.raises(
        RuntimeError, match="AppLauncher diagnostic SimulationApp.close"
    ) as captured:
        diagnostic.run_app_launcher_only_diagnostic(
            simulation_app=app,
            preexec_path=preexec_path,
            preclose_path=preclose_path,
            expected_gpu_uuid=GPU_UUID,
        )
    assert captured.value.__cause__ is close_error
    assert app.close_calls == 1
    assert preclose_path.is_file()
    assert diagnostic.ready_path_for(preclose_path).is_file()
    error_name = f"{type(close_error).__module__}.{type(close_error).__qualname__}"
    assert (
        f"POLARIS_STARTUP_DIAGNOSTIC_CLOSE_ERROR={error_name}"
        in capsys.readouterr().err
    )


def test_forbidden_module_boundary_fails_before_publication_or_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = synthetic_runtime(pid=os.getpid())
    preexec_path = tmp_path / "preexec.json"
    preexec_identity = publish_preexec(preexec_path, runtime)
    preclose_path = tmp_path / "preclose.json"
    monkeypatch.setenv(
        "POLARIS_STARTUP_DIAGNOSTIC_PREEXEC_SHA256",
        preexec_identity["sha256"],
    )
    monkeypatch.setattr(diagnostic, "capture_runtime_context", lambda **_: runtime)
    monkeypatch.setattr(
        diagnostic, "_forbidden_loaded_modules", lambda: ["polaris.policy"]
    )
    app = FakeSimulationApp()
    with pytest.raises(RuntimeError, match="forbidden modules"):
        diagnostic.run_app_launcher_only_diagnostic(
            simulation_app=app,
            preexec_path=preexec_path,
            preclose_path=preclose_path,
            expected_gpu_uuid=GPU_UUID,
        )
    assert app.close_calls == 0
    assert not preclose_path.exists()


@pytest.mark.parametrize("prefix", diagnostic.FORBIDDEN_MODULE_PREFIXES)
def test_every_forbidden_module_prefix_is_detected(
    prefix: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_name = f"{prefix}.adversarial_leaf"
    monkeypatch.setitem(sys.modules, module_name, object())
    assert module_name in diagnostic._forbidden_loaded_modules()
