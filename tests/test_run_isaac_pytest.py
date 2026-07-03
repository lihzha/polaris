from __future__ import annotations

import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_isaac_pytest


class _ExitSignal(RuntimeError):
    def __init__(self, code: int):
        super().__init__(f"exit {code}")
        self.code = code


def _invoke_main(
    monkeypatch,
    *,
    pytest_exit_code: int = 0,
    pytest_error=None,
    controller_import_error=None,
    close_error=None,
    report_error=None,
    observations=None,
    exit_code_file=None,
) -> int:
    if observations is None:
        observations = {}
    observations["close_calls"] = 0

    class _App:
        def close(self):
            observations["close_calls"] += 1
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
    if exit_code_file is None:
        monkeypatch.delenv(run_isaac_pytest.EXIT_CODE_FILE_ENV, raising=False)
    else:
        monkeypatch.setenv(
            run_isaac_pytest.EXIT_CODE_FILE_ENV,
            str(exit_code_file),
        )
    if report_error is not None:
        monkeypatch.setattr(
            run_isaac_pytest.traceback,
            "print_exception",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(report_error),
        )

    with pytest.raises(_ExitSignal) as captured:
        run_isaac_pytest.main()
    return captured.value.code


@pytest.mark.parametrize("pytest_exit_code", [0, 1, 2, 4, 5])
def test_bootstrap_propagates_exact_pytest_exit_code(monkeypatch, pytest_exit_code):
    assert (
        _invoke_main(monkeypatch, pytest_exit_code=pytest_exit_code) == pytest_exit_code
    )


@pytest.mark.parametrize("pytest_exit_code", [0, 2])
def test_bootstrap_close_failure_forces_one(monkeypatch, pytest_exit_code):
    assert (
        _invoke_main(
            monkeypatch,
            pytest_exit_code=pytest_exit_code,
            close_error=RuntimeError("close failed"),
        )
        == 1
    )


def test_bootstrap_pytest_failure_closes_once_and_exits_one(monkeypatch):
    observations = {}
    assert (
        _invoke_main(
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
        _invoke_main(
            monkeypatch,
            controller_import_error=ImportError("controller failed"),
            observations=observations,
        )
        == 1
    )
    assert observations["close_calls"] == 1


def test_bootstrap_reporting_failure_cannot_preserve_success(monkeypatch):
    assert (
        _invoke_main(
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


@pytest.mark.parametrize("pytest_exit_code", [0, 1, 2, 4, 5])
def test_bootstrap_publishes_exact_immutable_exit_code(
    monkeypatch, tmp_path, pytest_exit_code
):
    exit_code_file = tmp_path / "isaac-pytest.exit"
    assert (
        _invoke_main(
            monkeypatch,
            pytest_exit_code=pytest_exit_code,
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
        _invoke_main(
            monkeypatch,
            pytest_exit_code=0,
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
        _invoke_main(
            monkeypatch,
            pytest_exit_code=0,
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
        _invoke_main(
            monkeypatch,
            pytest_exit_code=0,
            exit_code_file=exit_code_file,
        )
        == 1
    )
    assert not exit_code_file.exists()


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


@pytest.mark.parametrize("failure", ["import", "pytest", "close", "report"])
def test_bootstrap_failures_publish_one(monkeypatch, tmp_path, failure):
    exit_code_file = tmp_path / "isaac-pytest.exit"
    kwargs = {"exit_code_file": exit_code_file}
    if failure == "import":
        kwargs["controller_import_error"] = ImportError("controller failed")
    elif failure == "pytest":
        kwargs["pytest_error"] = RuntimeError("pytest failed")
    elif failure == "close":
        kwargs["close_error"] = RuntimeError("close failed")
    else:
        kwargs["close_error"] = RuntimeError("close failed")
        kwargs["report_error"] = OSError("stderr failed")

    assert _invoke_main(monkeypatch, **kwargs) == 1
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
        _invoke_main(
            monkeypatch,
            pytest_exit_code=0,
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
        _invoke_main(
            monkeypatch,
            pytest_exit_code=0,
            exit_code_file=exit_code_file,
        )
        == 0
    )
    assert exit_code_file.read_bytes() == b"0\n"
    assert stat.S_IMODE(exit_code_file.stat().st_mode) == 0o444
    assert temporary.exists()


def test_bootstrap_forwards_arguments_and_isolates_pytest_sys_argv(monkeypatch):
    observations = {}
    assert _invoke_main(monkeypatch, observations=observations) == 0
    assert observations["pytest_arguments"] == ["-q", "tests/test_x.py"]
    assert observations["pytest_sys_argv"] == ["run_isaac_pytest.py"]
