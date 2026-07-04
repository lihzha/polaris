#!/usr/bin/env python3
"""Force and verify recovery of Isaac Lab's default headless viewport camera."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import traceback


def _parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", default="DROID-BlockStackKitchen")
    parser.add_argument("--output-json", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = True
    return args_cli, AppLauncher


def _publish(path: Path, payload: dict[str, object]) -> dict[str, object]:
    serialized = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path = path.resolve(strict=False)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        offset = 0
        while offset < len(serialized):
            offset += os.write(descriptor, serialized[offset:])
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)

    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o444
        or before.st_nlink != 1
    ):
        raise RuntimeError("published viewport-smoke JSON is not sealed")
    published = path.read_bytes()
    after = path.lstat()
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        stat.S_IMODE(before.st_mode),
        before.st_nlink,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        stat.S_IMODE(after.st_mode),
        after.st_nlink,
    )
    if published != serialized or identity != after_identity:
        raise RuntimeError("published viewport-smoke JSON changed on readback")
    return {
        "path": str(path),
        "size_bytes": len(published),
        "sha256": hashlib.sha256(published).hexdigest(),
        "mode": "0444",
    }


def _publish_success_and_close(
    output_json: Path,
    payload: dict[str, object],
    simulation_app,
) -> None:
    raw_identity = _publish(output_json, payload)
    ready_marker = output_json.resolve(strict=False).with_name(
        output_json.name + ".ready.json"
    )
    ready_payload = {
        "schema_version": 1,
        "stage": "simulation_app_close_pending",
        "raw_result": raw_identity,
    }
    print(
        "POLARIS_VIEWPORT_RECOVERY_RAW="
        f"{raw_identity['path']};size={raw_identity['size_bytes']};"
        f"sha256={raw_identity['sha256']};mode={raw_identity['mode']}",
        flush=True,
    )
    print(f"POLARIS_VIEWPORT_RECOVERY_READY={ready_marker}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    _publish(ready_marker, ready_payload)
    simulation_app.close()


def _remove_composed_root_prim(stage, path: str, *, remove_spec) -> int:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError("default viewport camera was absent before forced removal")
    prim_specs = list(prim.GetPrimStack())
    if not prim_specs:
        raise RuntimeError("default viewport camera has no removable prim specs")
    for prim_spec in prim_specs:
        if str(prim_spec.path) != path:
            raise RuntimeError("default viewport camera prim stack path mismatch")
        if remove_spec(prim_spec) is not True:
            raise RuntimeError("failed to remove a viewport-camera prim spec")
    if stage.GetPrimAtPath(path).IsValid():
        raise RuntimeError("forced composed viewport-camera removal did not persist")
    return len(prim_specs)


def main() -> int:
    args_cli, app_launcher_type = _parse_args()
    simulation_app = None
    env = None
    payload: dict[str, object]
    error: BaseException | None = None
    try:
        simulation_app = app_launcher_type(args_cli).app
        import gymnasium as gym
        import omni.usd
        from isaaclab.sim import SimulationContext
        from isaaclab_tasks.utils import parse_env_cfg
        from pxr import Sdf, UsdGeom

        import polaris.environments  # noqa: F401
        from polaris.headless_viewport import (
            DEFAULT_VIEWPORT_CAMERA_PRIM_PATH,
            HEADLESS_VIEWPORT_RECOVERY_PROFILE,
            install_viewport_camera_guard,
        )

        forced_missing = False
        recovery_messages: list[str] = []

        def remove_spec(prim_spec):
            edit = Sdf.BatchNamespaceEdit()
            edit.Add(Sdf.NamespaceEdit.Remove(prim_spec.path))
            return prim_spec.layer.Apply(edit)

        def stage_getter():
            nonlocal forced_missing
            stage = omni.usd.get_context().get_stage()
            if stage is not None and not forced_missing:
                removed_specs = _remove_composed_root_prim(
                    stage,
                    DEFAULT_VIEWPORT_CAMERA_PRIM_PATH,
                    remove_spec=remove_spec,
                )
                forced_missing = True
                print(
                    "POLARIS_VIEWPORT_FORCED_REMOVAL="
                    f"prim_path={DEFAULT_VIEWPORT_CAMERA_PRIM_PATH};"
                    f"prim_specs={removed_specs}",
                    flush=True,
                )
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
            "stage": "simulation_app_close_pending",
            "environment": args_cli.environment,
            "profile": HEADLESS_VIEWPORT_RECOVERY_PROFILE,
            "camera_prim_path": DEFAULT_VIEWPORT_CAMERA_PRIM_PATH,
            "forced_missing": forced_missing,
            "camera_valid_after_recovery": camera_valid,
            "recovery_messages": recovery_messages,
        }
        env.close()
        env = None
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
        if env is not None:
            try:
                env.close()
            except BaseException:
                traceback.print_exc()

    if error is not None:
        traceback.print_exception(type(error), error, error.__traceback__)
        try:
            _publish(args_cli.output_json, payload)
        except BaseException:
            traceback.print_exc()
        if simulation_app is not None:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        return 1

    try:
        _publish_success_and_close(args_cli.output_json, payload, simulation_app)
        return 0
    except BaseException as caught:
        traceback.print_exception(type(caught), caught, caught.__traceback__)
        sys.stdout.flush()
        sys.stderr.flush()
        if simulation_app is not None:
            os._exit(1)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
