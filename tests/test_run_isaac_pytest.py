from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_isaac_pytest


class _ExitSignal(RuntimeError):
    def __init__(self, code: int):
        super().__init__(f"exit {code}")
        self.code = code


def _invoke_child(
    monkeypatch,
    *,
    pytest_exit_code: int = 0,
    pytest_error=None,
    controller_import_error=None,
    close_error=None,
    report_error=None,
    observations=None,
) -> int:
    if observations is None:
        observations = {}
    observations["close_calls"] = 0
    observations["reported_exit_codes"] = []
    observations["events"] = []

    class _App:
        def close(self):
            observations["close_calls"] += 1
            observations["events"].append("close")
            if close_error is not None:
                raise close_error

    class _AppLauncher:
        def __init__(self, _config):
            self.app = _App()

    def fake_pytest_main(arguments):
        observations["pytest_arguments"] = list(arguments)
        observations["pytest_sys_argv"] = list(sys.argv)
        if pytest_error is not None:
            raise pytest_error
        return pytest_exit_code

    fake_pytest = SimpleNamespace(main=fake_pytest_main)
    fake_controller = SimpleNamespace(
        DifferentialIKController=type(
            "DifferentialIKController",
            (),
            {"__module__": "isaaclab.controllers.differential_ik"},
        )
    )
    modules = {
        "isaaclab.app": SimpleNamespace(AppLauncher=_AppLauncher),
        "isaaclab.controllers.differential_ik": fake_controller,
        "omni.log": SimpleNamespace(),
        "pytest": fake_pytest,
    }
    real_import_module = run_isaac_pytest.importlib.import_module

    def fake_import_module(name):
        observations["events"].append(f"import:{name}")
        observations.setdefault(
            "first_import_environment",
            {
                run_isaac_pytest.CHILD_PROCESS_ENV: run_isaac_pytest.os.environ.get(
                    run_isaac_pytest.CHILD_PROCESS_ENV
                ),
                run_isaac_pytest.CHILD_RESULT_FD_ENV: run_isaac_pytest.os.environ.get(
                    run_isaac_pytest.CHILD_RESULT_FD_ENV
                ),
                run_isaac_pytest.EXIT_CODE_FILE_ENV: run_isaac_pytest.os.environ.get(
                    run_isaac_pytest.EXIT_CODE_FILE_ENV
                ),
            },
        )
        if (
            name == "isaaclab.controllers.differential_ik"
            and controller_import_error is not None
        ):
            raise controller_import_error
        if name in modules:
            return modules[name]
        return real_import_module(name)

    monkeypatch.setattr(run_isaac_pytest.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(
        run_isaac_pytest.os,
        "_exit",
        lambda code: (_ for _ in ()).throw(_ExitSignal(code)),
    )
    monkeypatch.setattr(sys, "argv", ["run_isaac_pytest.py", "-q", "tests/test_x.py"])
    monkeypatch.setenv(run_isaac_pytest.CHILD_PROCESS_ENV, "1")
    monkeypatch.setenv(run_isaac_pytest.CHILD_RESULT_FD_ENV, "99")
    monkeypatch.delenv(run_isaac_pytest.EXIT_CODE_FILE_ENV, raising=False)

    def fake_prepare_child_result_descriptor():
        observations["events"].append("prepare")
        run_isaac_pytest.os.environ.pop(
            run_isaac_pytest.CHILD_RESULT_FD_ENV,
            None,
        )
        return 99

    monkeypatch.setattr(
        run_isaac_pytest,
        "_prepare_child_result_descriptor",
        fake_prepare_child_result_descriptor,
    )
    monkeypatch.setattr(
        run_isaac_pytest,
        "_write_child_exit_code",
        lambda code, descriptor: (
            observations["events"].append("report"),
            observations["reported_exit_codes"].append((code, descriptor)),
        ),
    )
    if report_error is not None:
        monkeypatch.setattr(
            run_isaac_pytest.traceback,
            "print_exception",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(report_error),
        )

    with pytest.raises(_ExitSignal) as captured:
        run_isaac_pytest._child_main(  # noqa: SLF001
            ["-q", "tests/test_x.py"]
        )
    return captured.value.code


def _invoke_parent(
    monkeypatch,
    *,
    reported_exit_code=0,
    process_return_code=0,
    run_error=None,
    exit_code_file=None,
    observations=None,
) -> int:
    if observations is None:
        observations = {}

    def fake_run_child(arguments):
        observations["child_arguments"] = list(arguments)
        if run_error is not None:
            raise run_error
        return reported_exit_code, process_return_code

    monkeypatch.setattr(run_isaac_pytest, "_run_isaac_child", fake_run_child)
    monkeypatch.setattr(
        run_isaac_pytest.os,
        "_exit",
        lambda code: (_ for _ in ()).throw(_ExitSignal(code)),
    )
    monkeypatch.setattr(sys, "argv", ["run_isaac_pytest.py", "-q", "tests/test_x.py"])
    monkeypatch.delenv(run_isaac_pytest.CHILD_PROCESS_ENV, raising=False)
    if exit_code_file is None:
        monkeypatch.delenv(run_isaac_pytest.EXIT_CODE_FILE_ENV, raising=False)
    else:
        monkeypatch.setenv(
            run_isaac_pytest.EXIT_CODE_FILE_ENV,
            str(exit_code_file),
        )

    with pytest.raises(_ExitSignal) as captured:
        run_isaac_pytest.main()
    return captured.value.code


@pytest.mark.parametrize("pytest_exit_code", range(6))
def test_bootstrap_propagates_exact_pytest_exit_code(monkeypatch, pytest_exit_code):
    observations = {}
    assert (
        _invoke_child(
            monkeypatch,
            pytest_exit_code=pytest_exit_code,
            observations=observations,
        )
        == pytest_exit_code
    )
    assert observations["reported_exit_codes"] == [(pytest_exit_code, 99)]


@pytest.mark.parametrize("pytest_exit_code", [0, 2])
def test_bootstrap_close_failure_forces_one(monkeypatch, pytest_exit_code):
    observations = {}
    assert (
        _invoke_child(
            monkeypatch,
            pytest_exit_code=pytest_exit_code,
            close_error=RuntimeError("close failed"),
            observations=observations,
        )
        == 1
    )
    assert observations["reported_exit_codes"] == [(pytest_exit_code, 99)]


def test_bootstrap_pytest_failure_closes_once_and_exits_one(monkeypatch):
    observations = {}
    assert (
        _invoke_child(
            monkeypatch,
            pytest_error=RuntimeError("pytest failed"),
            observations=observations,
        )
        == 1
    )
    assert observations["close_calls"] == 1


def test_bootstrap_import_failure_closes_once_and_exits_one(monkeypatch):
    observations = {}
    assert (
        _invoke_child(
            monkeypatch,
            controller_import_error=ImportError("controller failed"),
            observations=observations,
        )
        == 1
    )
    assert observations["close_calls"] == 1


def test_bootstrap_reporting_failure_cannot_preserve_success(monkeypatch):
    assert (
        _invoke_child(
            monkeypatch,
            pytest_exit_code=0,
            close_error=RuntimeError("close failed"),
            report_error=OSError("stderr failed"),
        )
        == 1
    )


def test_bootstrap_flush_failure_still_forces_nonzero_exit(monkeypatch):
    class _BadStream:
        def flush(self):
            raise OSError("flush failed")

    monkeypatch.setattr(
        run_isaac_pytest.os,
        "_exit",
        lambda code: (_ for _ in ()).throw(_ExitSignal(code)),
    )
    with pytest.raises(_ExitSignal) as captured:
        run_isaac_pytest._flush_and_exit(0, streams=(_BadStream(),))  # noqa: SLF001
    assert captured.value.code == 1


@pytest.mark.parametrize("pytest_exit_code", range(6))
def test_bootstrap_publishes_exact_immutable_exit_code(
    monkeypatch, tmp_path, pytest_exit_code
):
    exit_code_file = tmp_path / "isaac-pytest.exit"
    assert (
        _invoke_parent(
            monkeypatch,
            reported_exit_code=pytest_exit_code,
            process_return_code=pytest_exit_code,
            exit_code_file=exit_code_file,
        )
        == pytest_exit_code
    )
    assert exit_code_file.read_bytes() == f"{pytest_exit_code}\n".encode("ascii")
    assert stat.S_IMODE(exit_code_file.stat().st_mode) == 0o444
    assert list(tmp_path.glob(".*.tmp")) == []


def test_bootstrap_exit_code_file_never_overwrites(monkeypatch, tmp_path):
    exit_code_file = tmp_path / "isaac-pytest.exit"
    exit_code_file.write_bytes(b"sentinel\n")

    assert (
        _invoke_parent(
            monkeypatch,
            reported_exit_code=0,
            exit_code_file=exit_code_file,
        )
        == 1
    )
    assert exit_code_file.read_bytes() == b"sentinel\n"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_bootstrap_exit_code_file_rejects_dangling_symlink(monkeypatch, tmp_path):
    exit_code_file = tmp_path / "isaac-pytest.exit"
    exit_code_file.symlink_to(tmp_path / "missing-target")

    assert (
        _invoke_parent(
            monkeypatch,
            reported_exit_code=0,
            exit_code_file=exit_code_file,
        )
        == 1
    )
    assert exit_code_file.is_symlink()
    assert not exit_code_file.resolve().exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_bootstrap_exit_code_publish_failure_forces_one(monkeypatch, tmp_path):
    exit_code_file = tmp_path / "missing" / "isaac-pytest.exit"
    assert (
        _invoke_parent(
            monkeypatch,
            reported_exit_code=0,
            exit_code_file=exit_code_file,
        )
        == 1
    )
    assert not exit_code_file.exists()


def test_parent_child_runner_failure_publishes_one(monkeypatch, tmp_path):
    exit_code_file = tmp_path / "isaac-pytest.exit"
    assert (
        _invoke_parent(
            monkeypatch,
            exit_code_file=exit_code_file,
            run_error=RuntimeError("child launch failed"),
        )
        == 1
    )
    assert exit_code_file.read_bytes() == b"1\n"
    assert stat.S_IMODE(exit_code_file.stat().st_mode) == 0o444


def test_bootstrap_flush_failure_publishes_one(monkeypatch, tmp_path):
    class _BadStream:
        def flush(self):
            raise OSError("flush failed")

    exit_code_file = tmp_path / "isaac-pytest.exit"
    monkeypatch.setattr(
        run_isaac_pytest.os,
        "_exit",
        lambda code: (_ for _ in ()).throw(_ExitSignal(code)),
    )
    with pytest.raises(_ExitSignal) as captured:
        run_isaac_pytest._flush_and_exit(  # noqa: SLF001
            0,
            streams=(_BadStream(),),
            exit_code_file=exit_code_file,
        )
    assert captured.value.code == 1
    assert exit_code_file.read_bytes() == b"1\n"
    assert stat.S_IMODE(exit_code_file.stat().st_mode) == 0o444


@pytest.mark.parametrize("operation", ["write", "fsync", "link"])
def test_bootstrap_publication_operation_failure_forces_one(
    monkeypatch, tmp_path, operation
):
    exit_code_file = tmp_path / "isaac-pytest.exit"

    def fail(*_args, **_kwargs):
        raise OSError(f"{operation} failed")

    monkeypatch.setattr(run_isaac_pytest.os, operation, fail)
    assert (
        _invoke_parent(
            monkeypatch,
            reported_exit_code=0,
            exit_code_file=exit_code_file,
        )
        == 1
    )
    assert not exit_code_file.exists()
    assert not exit_code_file.is_symlink()


def test_bootstrap_unlink_failure_preserves_final_and_leaves_rejectable_temp(
    monkeypatch, tmp_path
):
    exit_code_file = tmp_path / "isaac-pytest.exit"
    temporary = tmp_path / ".isaac-pytest.exit.tmp"
    original_unlink = Path.unlink

    def reject_temporary_unlink(path, *args, **kwargs):
        if path == temporary:
            raise OSError("temporary unlink failed")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", reject_temporary_unlink)
    assert (
        _invoke_parent(
            monkeypatch,
            reported_exit_code=0,
            exit_code_file=exit_code_file,
        )
        == 0
    )
    assert exit_code_file.read_bytes() == b"0\n"
    assert stat.S_IMODE(exit_code_file.stat().st_mode) == 0o444
    assert temporary.exists()


def test_bootstrap_forwards_arguments_and_isolates_pytest_sys_argv(monkeypatch):
    observations = {}
    assert _invoke_child(monkeypatch, observations=observations) == 0
    assert observations["pytest_arguments"] == ["-q", "tests/test_x.py"]
    assert observations["pytest_sys_argv"] == ["run_isaac_pytest.py"]
    assert observations["events"][0] == "prepare"
    report_index = observations["events"].index("report")
    assert observations["events"][report_index + 1] == "close"
    assert observations["first_import_environment"] == {
        run_isaac_pytest.CHILD_PROCESS_ENV: None,
        run_isaac_pytest.CHILD_RESULT_FD_ENV: None,
        run_isaac_pytest.EXIT_CODE_FILE_ENV: None,
    }


@pytest.mark.parametrize(
    ("reported_exit_code", "process_return_code", "expected"),
    [
        (0, 0, 0),
        (1, 0, 1),
        (3, 0, 3),
        (5, 0, 5),
        (5, 5, 5),
        (None, 0, 1),
        (6, 0, 1),
        (0, 1, 1),
        (5, 1, 1),
        (0, -11, 1),
    ],
)
def test_parent_resolves_child_result_fail_closed(
    reported_exit_code, process_return_code, expected
):
    assert (
        run_isaac_pytest._resolve_child_exit_code(  # noqa: SLF001
            reported_exit_code,
            process_return_code,
        )
        == expected
    )


def test_child_result_pipe_is_noninheritable_exact_and_closed(monkeypatch):
    read_descriptor, write_descriptor = run_isaac_pytest.os.pipe()
    monkeypatch.setenv(
        run_isaac_pytest.CHILD_RESULT_FD_ENV,
        str(write_descriptor),
    )

    prepared = run_isaac_pytest._prepare_child_result_descriptor()  # noqa: SLF001
    assert prepared == write_descriptor
    assert not run_isaac_pytest.os.get_inheritable(prepared)
    assert run_isaac_pytest.CHILD_RESULT_FD_ENV not in run_isaac_pytest.os.environ
    run_isaac_pytest._write_child_exit_code(5, prepared)  # noqa: SLF001

    assert run_isaac_pytest.os.read(read_descriptor, 2) == b"\x05"
    assert run_isaac_pytest.os.read(read_descriptor, 2) == b""
    with pytest.raises(OSError):
        run_isaac_pytest.os.fstat(write_descriptor)
    run_isaac_pytest.os.close(read_descriptor)


def test_child_report_commit_survives_descriptor_close_failure(monkeypatch):
    read_descriptor, write_descriptor = run_isaac_pytest.os.pipe()
    original_close = run_isaac_pytest.os.close

    def fail_write_descriptor_close(descriptor):
        if descriptor == write_descriptor:
            raise OSError("close failed after commit")
        return original_close(descriptor)

    monkeypatch.setattr(
        run_isaac_pytest.os,
        "close",
        fail_write_descriptor_close,
    )
    run_isaac_pytest._write_child_exit_code(0, write_descriptor)  # noqa: SLF001
    assert run_isaac_pytest.os.read(read_descriptor, 2) == b"\x00"
    original_close(write_descriptor)
    assert run_isaac_pytest.os.read(read_descriptor, 2) == b""
    original_close(read_descriptor)


def test_child_diagnostic_failure_occurs_before_report_commit(monkeypatch):
    import builtins

    observations = {}
    original_print = builtins.print

    def fail_report_marker(*args, **kwargs):
        if args and str(args[0]).startswith("ISAAC_PYTEST_CHILD_REPORTED_EXIT_CODE="):
            raise OSError("marker flush failed")
        return original_print(*args, **kwargs)

    monkeypatch.setattr(builtins, "print", fail_report_marker)
    assert _invoke_child(monkeypatch, observations=observations) == 1
    assert observations["reported_exit_codes"] == []
    assert "report" not in observations["events"]
    assert observations["close_calls"] == 1


@pytest.mark.parametrize(
    ("payload", "expected_reported"),
    [(b"\x05", 5), (b"", None), (b"\x05\x00", None), (b"\x06", None)],
)
def test_run_child_requires_exact_one_byte_and_eof(
    monkeypatch, payload, expected_reported
):
    observations = {}

    class _Process:
        pid = 12345

        def __init__(self, descriptor):
            self._descriptor = descriptor

        def wait(self, *, timeout):
            observations["wait_timeout"] = timeout
            if payload:
                run_isaac_pytest.os.write(self._descriptor, payload)
            run_isaac_pytest.os.close(self._descriptor)
            return 0

        def poll(self):
            return 0

    def fake_popen(command, *, env, pass_fds, start_new_session):
        observations["command"] = command
        observations["environment"] = env
        observations["start_new_session"] = start_new_session
        return _Process(run_isaac_pytest.os.dup(pass_fds[0]))

    monkeypatch.setenv(run_isaac_pytest.EXIT_CODE_FILE_ENV, "/not/for/the/child")
    monkeypatch.setattr(run_isaac_pytest.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        run_isaac_pytest,
        "_kill_lingering_process_group",
        lambda _pid: False,
    )

    reported, process_code = run_isaac_pytest._run_isaac_child(  # noqa: SLF001
        ["-q", "tests/test_x.py"]
    )

    assert (reported, process_code) == (expected_reported, 0)
    assert observations["command"] == [
        sys.executable,
        str(Path(run_isaac_pytest.__file__).resolve()),
        "-q",
        "tests/test_x.py",
    ]
    assert observations["environment"][run_isaac_pytest.CHILD_PROCESS_ENV] == "1"
    assert run_isaac_pytest.EXIT_CODE_FILE_ENV not in observations["environment"]
    assert observations["start_new_session"] is True
    assert observations["wait_timeout"] == run_isaac_pytest.CHILD_TIMEOUT_SECONDS


def test_run_child_rejects_missing_eof(monkeypatch):
    retained_descriptor = []

    class _Process:
        pid = 12345

        def wait(self, *, timeout):
            del timeout
            run_isaac_pytest.os.write(retained_descriptor[0], b"\x00")
            return 0

        def poll(self):
            return 0

    def fake_popen(_command, *, env, pass_fds, start_new_session):
        del env, start_new_session
        retained_descriptor.append(run_isaac_pytest.os.dup(pass_fds[0]))
        return _Process()

    monkeypatch.setattr(run_isaac_pytest.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        run_isaac_pytest,
        "_kill_lingering_process_group",
        lambda _pid: False,
    )
    try:
        assert run_isaac_pytest._run_isaac_child([]) == (None, 0)  # noqa: SLF001
    finally:
        run_isaac_pytest.os.close(retained_descriptor[0])


def test_run_child_rejects_and_cleans_lingering_group_after_normal_exit(monkeypatch):
    observations = {}

    class _Process:
        pid = 23456

        def wait(self, *, timeout):
            del timeout
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(
        run_isaac_pytest.subprocess,
        "Popen",
        lambda *_args, **_kwargs: _Process(),
    )
    monkeypatch.setattr(
        run_isaac_pytest,
        "_kill_lingering_process_group",
        lambda pid: observations.setdefault("cleanup_pid", pid) and True,
    )
    with pytest.raises(RuntimeError, match="surviving process group"):
        run_isaac_pytest._run_isaac_child([])  # noqa: SLF001
    assert observations["cleanup_pid"] == 23456


def test_run_child_timeout_terminates_then_kills_process_group(monkeypatch):
    observations = {"wait_calls": 0}

    class _Process:
        pid = 54321

        def wait(self, *, timeout):
            observations["wait_calls"] += 1
            if observations["wait_calls"] <= 2:
                raise run_isaac_pytest.subprocess.TimeoutExpired("child", timeout)
            return -9

        def poll(self):
            return None

    def fake_popen(_command, *, env, pass_fds, start_new_session):
        del env, pass_fds, start_new_session
        return _Process()

    monkeypatch.setattr(run_isaac_pytest.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        run_isaac_pytest,
        "_kill_lingering_process_group",
        lambda pid: observations.setdefault("cleanup_pid", pid) and False,
    )
    monkeypatch.setattr(
        run_isaac_pytest.os,
        "killpg",
        lambda pid, sig: observations.setdefault("killpg", []).append((pid, sig)),
    )

    with pytest.raises(run_isaac_pytest.subprocess.TimeoutExpired):
        run_isaac_pytest._run_isaac_child([])  # noqa: SLF001
    assert observations["killpg"] == [
        (54321, run_isaac_pytest.signal.SIGTERM),
        (54321, run_isaac_pytest.signal.SIGKILL),
    ]
    assert observations["wait_calls"] == 3
    assert observations["cleanup_pid"] == 54321


def test_timeout_cleans_descendants_when_leader_exits_during_term_grace(monkeypatch):
    observations = {"wait_calls": 0}

    class _Process:
        pid = 65432

        def wait(self, *, timeout):
            observations["wait_calls"] += 1
            if observations["wait_calls"] == 1:
                raise run_isaac_pytest.subprocess.TimeoutExpired("child", timeout)
            return -15

        def poll(self):
            return None

    monkeypatch.setattr(
        run_isaac_pytest.subprocess,
        "Popen",
        lambda *_args, **_kwargs: _Process(),
    )
    monkeypatch.setattr(
        run_isaac_pytest.os,
        "killpg",
        lambda pid, sig: observations.setdefault("signals", []).append((pid, sig)),
    )
    monkeypatch.setattr(
        run_isaac_pytest,
        "_kill_lingering_process_group",
        lambda pid: observations.setdefault("cleanup_pid", pid) and True,
    )

    with pytest.raises(run_isaac_pytest.subprocess.TimeoutExpired):
        run_isaac_pytest._run_isaac_child([])  # noqa: SLF001
    assert observations["signals"] == [
        (65432, run_isaac_pytest.signal.SIGTERM),
        (65432, run_isaac_pytest.signal.SIGKILL),
    ]
    assert observations["cleanup_pid"] == 65432
    assert observations["wait_calls"] == 3


def test_unreapable_timeout_still_runs_cleanup_and_closes_descriptors(monkeypatch):
    observations = {"wait_calls": 0, "descriptors": []}
    real_pipe = run_isaac_pytest.os.pipe
    original_term_handler = run_isaac_pytest.signal.getsignal(
        run_isaac_pytest.signal.SIGTERM
    )

    def tracked_pipe():
        pair = real_pipe()
        observations["descriptors"].extend(pair)
        return pair

    class _Process:
        pid = 67543

        def wait(self, *, timeout):
            observations["wait_calls"] += 1
            raise run_isaac_pytest.subprocess.TimeoutExpired("child", timeout)

        def poll(self):
            return None

    monkeypatch.setattr(run_isaac_pytest.os, "pipe", tracked_pipe)
    monkeypatch.setattr(
        run_isaac_pytest.subprocess,
        "Popen",
        lambda *_args, **_kwargs: _Process(),
    )
    monkeypatch.setattr(
        run_isaac_pytest.os,
        "killpg",
        lambda pid, sig: observations.setdefault("signals", []).append((pid, sig)),
    )
    monkeypatch.setattr(
        run_isaac_pytest,
        "_kill_lingering_process_group",
        lambda pid: observations.setdefault("cleanup_pid", pid) and False,
    )

    with pytest.raises(RuntimeError, match="cleanup failed"):
        run_isaac_pytest._run_isaac_child([])  # noqa: SLF001
    assert observations["signals"] == [
        (67543, run_isaac_pytest.signal.SIGTERM),
        (67543, run_isaac_pytest.signal.SIGKILL),
    ]
    assert observations["cleanup_pid"] == 67543
    assert observations["wait_calls"] == 5
    for descriptor in observations["descriptors"]:
        with pytest.raises(OSError):
            run_isaac_pytest.os.fstat(descriptor)
    assert (
        run_isaac_pytest.signal.getsignal(run_isaac_pytest.signal.SIGTERM)
        == original_term_handler
    )


def test_parent_write_descriptor_close_failure_kills_child_and_closes_pipe(
    monkeypatch,
):
    observations = {"descriptors": [], "close_failed": False}
    real_pipe = run_isaac_pytest.os.pipe
    real_close = run_isaac_pytest.os.close

    def tracked_pipe():
        pair = real_pipe()
        observations["descriptors"].extend(pair)
        return pair

    class _Process:
        pid = 78654

        def wait(self, *, timeout):
            del timeout
            return -9

        def poll(self):
            return -9

    def fail_first_write_close(descriptor):
        if (
            descriptor == observations["descriptors"][1]
            and not observations["close_failed"]
        ):
            observations["close_failed"] = True
            raise OSError("parent write close failed")
        return real_close(descriptor)

    monkeypatch.setattr(run_isaac_pytest.os, "pipe", tracked_pipe)
    monkeypatch.setattr(run_isaac_pytest.os, "close", fail_first_write_close)
    monkeypatch.setattr(
        run_isaac_pytest.subprocess,
        "Popen",
        lambda *_args, **_kwargs: _Process(),
    )
    monkeypatch.setattr(
        run_isaac_pytest.os,
        "killpg",
        lambda pid, sig: observations.setdefault("signals", []).append((pid, sig)),
    )
    monkeypatch.setattr(
        run_isaac_pytest,
        "_kill_lingering_process_group",
        lambda pid: observations.setdefault("cleanup_pid", pid) and False,
    )

    with pytest.raises(OSError, match="parent write close failed"):
        run_isaac_pytest._run_isaac_child([])  # noqa: SLF001
    assert observations["signals"] == [(78654, run_isaac_pytest.signal.SIGKILL)]
    assert observations["cleanup_pid"] == 78654
    for descriptor in observations["descriptors"]:
        with pytest.raises(OSError):
            run_isaac_pytest.os.fstat(descriptor)


def test_parent_term_handler_kills_and_reaps_child_group(monkeypatch):
    observations = {"wait_calls": 0}
    original_term_handler = run_isaac_pytest.signal.getsignal(
        run_isaac_pytest.signal.SIGTERM
    )

    class _Process:
        pid = 87654

        def wait(self, *, timeout):
            del timeout
            observations["wait_calls"] += 1
            if observations["wait_calls"] == 1:
                handler = run_isaac_pytest.signal.getsignal(
                    run_isaac_pytest.signal.SIGTERM
                )
                handler(run_isaac_pytest.signal.SIGTERM, None)
            return -9

        def poll(self):
            return None

    monkeypatch.setattr(
        run_isaac_pytest.subprocess,
        "Popen",
        lambda *_args, **_kwargs: _Process(),
    )
    monkeypatch.setattr(
        run_isaac_pytest.os,
        "killpg",
        lambda pid, sig: observations.setdefault("signals", []).append((pid, sig)),
    )
    monkeypatch.setattr(
        run_isaac_pytest,
        "_kill_lingering_process_group",
        lambda pid: observations.setdefault("cleanup_pid", pid) and False,
    )

    with pytest.raises(run_isaac_pytest._ParentSignalError):  # noqa: SLF001
        run_isaac_pytest._run_isaac_child([])  # noqa: SLF001
    assert observations["signals"] == [(87654, run_isaac_pytest.signal.SIGKILL)]
    assert observations["wait_calls"] == 2
    assert observations["cleanup_pid"] == 87654
    assert (
        run_isaac_pytest.signal.getsignal(run_isaac_pytest.signal.SIGTERM)
        == original_term_handler
    )


def test_lingering_process_group_is_killed_and_observed_gone(monkeypatch):
    observations = {"zero_checks": 0, "signals": []}

    def fake_killpg(pid, sig):
        observations["signals"].append((pid, sig))
        if sig == 0:
            observations["zero_checks"] += 1
            if observations["zero_checks"] >= 2:
                raise ProcessLookupError

    monkeypatch.setattr(run_isaac_pytest.os, "killpg", fake_killpg)
    assert run_isaac_pytest._kill_lingering_process_group(76543) is True  # noqa: SLF001
    assert observations["signals"] == [
        (76543, 0),
        (76543, run_isaac_pytest.signal.SIGKILL),
        (76543, 0),
    ]


def test_run_child_spawn_failure_closes_both_pipe_ends(monkeypatch):
    descriptors = []
    real_pipe = run_isaac_pytest.os.pipe

    def tracked_pipe():
        pair = real_pipe()
        descriptors.extend(pair)
        return pair

    monkeypatch.setattr(run_isaac_pytest.os, "pipe", tracked_pipe)
    monkeypatch.setattr(
        run_isaac_pytest.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("spawn failed")),
    )

    with pytest.raises(OSError, match="spawn failed"):
        run_isaac_pytest._run_isaac_child([])  # noqa: SLF001
    assert len(descriptors) == 2
    for descriptor in descriptors:
        with pytest.raises(OSError):
            run_isaac_pytest.os.fstat(descriptor)


@pytest.mark.parametrize("pytest_exit_code", [0, 5])
def test_parent_survives_child_native_zero_exit_and_publishes_result(
    tmp_path, pytest_exit_code
):
    package_root = tmp_path / "fake_modules"
    (package_root / "isaaclab" / "controllers").mkdir(parents=True)
    (package_root / "omni").mkdir()
    (package_root / "isaaclab" / "__init__.py").write_text("")
    (package_root / "isaaclab" / "controllers" / "__init__.py").write_text("")
    (package_root / "omni" / "__init__.py").write_text("")
    (package_root / "omni" / "log.py").write_text("")
    (package_root / "isaaclab" / "app.py").write_text(
        "import os\n"
        "class _App:\n"
        "    def close(self):\n"
        "        os._exit(0)\n"
        "class AppLauncher:\n"
        "    def __init__(self, config):\n"
        "        self.app = _App()\n"
    )
    (package_root / "isaaclab" / "controllers" / "differential_ik.py").write_text(
        "class DifferentialIKController:\n    pass\n"
    )
    (package_root / "pytest.py").write_text(
        "import os\n"
        "def main(arguments):\n"
        "    return int(os.environ['FAKE_PYTEST_EXIT_CODE'])\n"
    )

    exit_code_file = tmp_path / f"parent-{pytest_exit_code}.exit"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(package_root)
    environment["FAKE_PYTEST_EXIT_CODE"] = str(pytest_exit_code)
    environment[run_isaac_pytest.EXIT_CODE_FILE_ENV] = str(exit_code_file)
    environment.pop(run_isaac_pytest.CHILD_PROCESS_ENV, None)
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(run_isaac_pytest.__file__).resolve()),
            "-q",
            "tests/fake.py",
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == pytest_exit_code, completed.stderr
    assert exit_code_file.read_bytes() == f"{pytest_exit_code}\n".encode("ascii")
    assert stat.S_IMODE(exit_code_file.stat().st_mode) == 0o444
    assert f"reported={pytest_exit_code};process=0;resolved={pytest_exit_code}" in (
        completed.stdout
    )
