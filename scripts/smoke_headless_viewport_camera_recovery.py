#!/usr/bin/env python3
"""Force and verify recovery of Isaac Lab's default headless viewport camera."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--environment", default="DROID-BlockStackKitchen")
parser.add_argument("--output-json", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = True


def _publish(path: Path, payload: dict[str, object]) -> None:
    serialized = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        path.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    simulation_app = AppLauncher(args_cli).app
    env = None
    payload: dict[str, object]
    error: BaseException | None = None
    try:
        import gymnasium as gym
        import omni.usd
        from isaaclab.sim import SimulationContext
        from isaaclab_tasks.utils import parse_env_cfg
        from pxr import UsdGeom

        import polaris.environments  # noqa: F401
        from polaris.headless_viewport import (
            DEFAULT_VIEWPORT_CAMERA_PRIM_PATH,
            HEADLESS_VIEWPORT_RECOVERY_PROFILE,
            install_viewport_camera_guard,
        )

        forced_missing = False
        recovery_messages: list[str] = []

        def stage_getter():
            nonlocal forced_missing
            stage = omni.usd.get_context().get_stage()
            if stage is not None and not forced_missing:
                if stage.GetPrimAtPath(DEFAULT_VIEWPORT_CAMERA_PRIM_PATH).IsValid():
                    stage.RemovePrim(DEFAULT_VIEWPORT_CAMERA_PRIM_PATH)
                forced_missing = True
                if stage.GetPrimAtPath(DEFAULT_VIEWPORT_CAMERA_PRIM_PATH).IsValid():
                    raise RuntimeError("forced viewport-camera removal did not persist")
            return stage

        installed = install_viewport_camera_guard(
            SimulationContext,
            stage_getter=stage_getter,
            camera_definer=lambda stage, path: UsdGeom.Camera.Define(stage, path),
            emit=recovery_messages.append,
        )
        if not installed:
            raise RuntimeError("viewport-camera guard was already installed")

        env_cfg = parse_env_cfg(
            args_cli.environment,
            device="cuda",
            num_envs=1,
            use_fabric=True,
        )
        env = gym.make(args_cli.environment, cfg=env_cfg)
        stage = omni.usd.get_context().get_stage()
        camera_valid = (
            stage is not None
            and stage.GetPrimAtPath(DEFAULT_VIEWPORT_CAMERA_PRIM_PATH).IsValid()
        )
        if not forced_missing or not camera_valid or len(recovery_messages) != 1:
            raise RuntimeError(
                "headless viewport recovery did not close its exact forced-missing "
                "camera contract"
            )
        payload = {
            "schema_version": 1,
            "status": "success",
            "environment": args_cli.environment,
            "profile": HEADLESS_VIEWPORT_RECOVERY_PROFILE,
            "camera_prim_path": DEFAULT_VIEWPORT_CAMERA_PRIM_PATH,
            "forced_missing": forced_missing,
            "camera_valid_after_recovery": camera_valid,
            "recovery_messages": recovery_messages,
        }
    except BaseException as caught:
        error = caught
        payload = {
            "schema_version": 1,
            "status": "failure",
            "environment": args_cli.environment,
            "error_type": f"{type(caught).__module__}.{type(caught).__qualname__}",
            "error_message": str(caught),
            "traceback": "".join(
                traceback.format_exception(type(caught), caught, caught.__traceback__)
            ),
        }
    finally:
        if env is not None:
            env.close()
        simulation_app.close()
        _publish(args_cli.output_json, payload)

    if error is not None:
        raise error


if __name__ == "__main__":
    main()
