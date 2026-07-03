#!/usr/bin/env python3
"""Launch Isaac Sim before collecting tests that import Isaac Lab extensions."""

from __future__ import annotations

import importlib
import os
import sys
import traceback


def _report_exception(error: BaseException) -> None:
    """Report a bootstrap failure without risking its nonzero exit status."""

    try:
        traceback.print_exception(type(error), error, error.__traceback__)
    except BaseException:
        pass


def _flush_and_exit(exit_code: int, *, streams=None) -> None:
    """Best-effort flush, then unconditionally bypass Kit/atexit hooks."""

    if streams is None:
        streams = (sys.stdout, sys.stderr)
    final_exit_code = exit_code
    try:
        for stream in streams:
            try:
                stream.flush()
            except BaseException:
                final_exit_code = 1
    finally:
        os._exit(final_exit_code)


def main() -> None:
    pytest_args = sys.argv[1:]
    sys.argv = [sys.argv[0]]

    launcher = None
    exit_code = 1
    try:
        app_launcher_module = importlib.import_module("isaaclab.app")
        app_launcher_type = app_launcher_module.AppLauncher
        launcher = app_launcher_type({"headless": True, "enable_cameras": False})
        controller_module = importlib.import_module(
            "isaaclab.controllers.differential_ik"
        )
        importlib.import_module("omni.log")
        pytest = importlib.import_module("pytest")

        print(f"ISAAC_PYTEST_ARGS={pytest_args!r}", flush=True)
        print(
            f"ISAAC_CONTROLLER={controller_module.DifferentialIKController.__module__}",
            flush=True,
        )
        print("ISAAC_PYTEST_BOOTSTRAP_OK", flush=True)
        exit_code = int(pytest.main(pytest_args))
        print(f"ISAAC_PYTEST_EXIT_CODE={exit_code}", flush=True)
    except BaseException as error:
        exit_code = 1
        _report_exception(error)
    finally:
        if launcher is not None:
            try:
                launcher.app.close()
            except BaseException as error:
                exit_code = 1
                _report_exception(error)
        # Kit teardown can otherwise mask pytest's nonzero status. Bypass
        # interpreter/atexit hooks after the app has closed so Slurm sees the
        # exact test result. This helper reaches os._exit even if flushing
        # itself fails.
        _flush_and_exit(exit_code)


if __name__ == "__main__":
    main()
