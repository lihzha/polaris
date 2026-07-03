from __future__ import annotations

import sys
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


def test_bootstrap_forwards_arguments_and_isolates_pytest_sys_argv(monkeypatch):
    observations = {}
    assert _invoke_main(monkeypatch, observations=observations) == 0
    assert observations["pytest_arguments"] == ["-q", "tests/test_x.py"]
    assert observations["pytest_sys_argv"] == ["run_isaac_pytest.py"]
