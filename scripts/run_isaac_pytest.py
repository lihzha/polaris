#!/usr/bin/env python3
"""Launch Isaac Sim before collecting tests that import Isaac Lab extensions."""

from __future__ import annotations

import importlib
import sys


def main() -> int:
    pytest_args = sys.argv[1:]
    sys.argv = [sys.argv[0]]

    app_launcher_module = importlib.import_module("isaaclab.app")
    app_launcher_type = app_launcher_module.AppLauncher
    launcher = app_launcher_type({"headless": True, "enable_cameras": False})
    try:
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
        return int(pytest.main(pytest_args))
    finally:
        launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
