#!/usr/bin/env python3
"""Close-aware parent/Kit-child smoke for native DROID velocity control."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import traceback
from pathlib import Path
from typing import NoReturn


def _canonical_json(payload: dict) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _declared_absolute_path(path: str | Path, field: str) -> Path:
    rendered = os.fspath(path)
    if (
        not isinstance(rendered, str)
        or not rendered.startswith("/")
        or rendered.startswith("//")
        or "\0" in rendered
        or rendered != os.path.normpath(rendered)
    ):
        raise ValueError(f"{field} must use one normalized absolute path spelling")
    return Path(rendered)


def _write_immutable_json(path: Path, payload: dict) -> bytes:
    """Publish one canonical, fsynced, non-overwriting mode-0444 JSON file."""

    rendered = _canonical_json(payload)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as output:
            output.write(rendered)
            output.flush()
            os.fsync(output.fileno())
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return rendered


def _read_immutable_json(path: Path, field: str) -> tuple[dict, bytes, os.stat_result]:
    """Read one exact immutable inode without following or racing a replacement."""

    path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{field} is not one readable regular file") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o444
        ):
            raise ValueError(f"{field} must be one mode-0444 regular link")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        rendered = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.stat(path, follow_symlinks=False)
    stable_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mode,
        before.st_nlink,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if (
        stable_identity
        != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mode,
            after.st_nlink,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        or stable_identity
        != (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mode,
            current.st_nlink,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        or not stat.S_ISREG(current.st_mode)
        or stat.S_IMODE(current.st_mode) != 0o444
        or current.st_nlink != 1
    ):
        raise ValueError(f"{field} changed while it was being read")
    try:
        value = json.loads(rendered)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} is not strict JSON") from error
    if not isinstance(value, dict) or rendered != _canonical_json(value):
        raise ValueError(f"{field} is not canonical JSON")
    return value, rendered, before


def _child_ready_payload(raw_path: str | Path, raw_bytes: bytes) -> dict:
    raw_path = _declared_absolute_path(raw_path, "child raw capture")
    return {
        "schema_version": 1,
        "status": "success",
        "stage": "kit_child_after_env_close_before_simulation_app_close",
        "raw_result": {
            "path": str(raw_path),
            "size_bytes": len(raw_bytes),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "mode": "0444",
        },
    }


def _abort_kit_child(
    failure_path: Path, *, stage: str, error: BaseException
) -> NoReturn:
    """Durably report the original failure, then bypass Kit teardown hooks."""

    try:
        try:
            try:
                formatted = "".join(
                    traceback.format_exception(type(error), error, error.__traceback__)
                )
            except BaseException:
                formatted = "traceback formatting failed"
            try:
                message = str(error)
            except BaseException:
                message = "exception stringification failed"
            failure = {
                "schema_version": 1,
                "status": "failure",
                "stage": stage,
                "exception": {
                    "type": f"{type(error).__module__}.{type(error).__qualname__}",
                    "message": message,
                    "traceback": formatted,
                },
            }
            try:
                _write_immutable_json(failure_path, failure)
            except BaseException as persistence_error:
                try:
                    print(
                        "POLARIS_JOINT_VELOCITY_FAILURE_PERSISTENCE="
                        f"{type(persistence_error).__name__}: {persistence_error}",
                        file=sys.stderr,
                        flush=True,
                    )
                except BaseException:
                    pass
            try:
                traceback.print_exception(
                    type(error), error, error.__traceback__, file=sys.stderr
                )
            except BaseException:
                pass
        except BaseException:
            pass
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            # Failure paths deliberately do not call SimulationApp.close(): in
            # the pinned Kit runtime that call can hard-exit zero and mask this
            # error.
            os._exit(1)


def finalize_child_capture(
    raw_path: Path,
    ready_path: Path,
    failure_path: Path,
    output_path: Path,
    *,
    child_exit_code: int,
) -> dict:
    """Publish only after validated raw+ready evidence and child exit zero."""

    from polaris.joint_velocity_smoke import (
        publish_immutable_joint_velocity_smoke,
        validate_joint_velocity_smoke,
    )

    raw_path = Path(raw_path)
    ready_path = Path(ready_path)
    failure_path = Path(failure_path)
    if failure_path.exists() or failure_path.is_symlink():
        failure, _, _ = _read_immutable_json(failure_path, "Kit child failure")
        exception = failure.get("exception")
        if (
            set(failure) != {"schema_version", "status", "stage", "exception"}
            or failure.get("schema_version") != 1
            or type(failure.get("schema_version")) is not int
            or failure.get("status") != "failure"
            or not isinstance(failure.get("stage"), str)
            or not failure["stage"]
            or not isinstance(exception, dict)
            or set(exception) != {"type", "message", "traceback"}
            or not all(isinstance(exception.get(key), str) for key in exception)
        ):
            raise ValueError("Kit child failure status schema mismatch")
        raise ValueError(
            "Kit child reported failure at "
            f"{failure['stage']}: {exception['type']}: {exception['message']}"
        )
    if child_exit_code != 0:
        raise ValueError(f"Kit child exited nonzero: {child_exit_code}")
    raw, raw_bytes, _ = _read_immutable_json(raw_path, "Kit child raw capture")
    _, ready_bytes, _ = _read_immutable_json(ready_path, "Kit child ready marker")
    expected_ready = _child_ready_payload(raw_path, raw_bytes)
    if ready_bytes != _canonical_json(expected_ready):
        raise ValueError("Kit child ready marker does not bind the raw capture")
    validated_raw = validate_joint_velocity_smoke(raw, require_parent_completion=False)
    final_payload = dict(validated_raw)
    final_payload.pop("status", None)
    final_payload.pop("case_count", None)
    final_payload["lifecycle"] = {
        "env_close": "complete",
        "simulation_app_close": "invoked_then_child_exited_zero",
        "capture_stage": "stdlib_parent_after_kit_child_exit",
    }
    final_payload["completion"] = {
        "child_exit_code": 0,
        "publication_stage": "stdlib_parent_after_child_exit",
        "child_capture_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "child_capture_size": len(raw_bytes),
        "child_capture_mode": "0444",
        "child_capture_path": str(
            _declared_absolute_path(raw_path, "child raw capture")
        ),
        "child_ready_marker_sha256": hashlib.sha256(ready_bytes).hexdigest(),
        "child_ready_marker_size": len(ready_bytes),
        "child_ready_marker_mode": "0444",
        "child_ready_marker_path": str(
            _declared_absolute_path(ready_path, "child ready marker")
        ),
    }
    return publish_immutable_joint_velocity_smoke(output_path, final_payload)


def _run_capture(args_cli, env, runtime_contract) -> dict:
    import torch

    from polaris.joint_velocity_smoke import (
        SMOKE_PROFILE,
        build_joint_velocity_smoke_cases,
    )
    from polaris.pi05_droid_jointvelocity_contract import (
        NATIVE_GRIPPER_DRIVE_PROFILE,
        NATIVE_GRIPPER_PRECONDITION_STEPS,
        PANDA_ARM_JOINT_NAMES,
        PI05_DROID_JOINTVELOCITY_PROFILE,
    )

    root_env = getattr(env, "unwrapped", env)
    if args_cli.expected_gripper_drive_profile != NATIVE_GRIPPER_DRIVE_PROFILE:
        raise ValueError("Smoke expected gripper drive profile mismatch")
    robot = root_env.scene["robot"]
    finger_term = root_env.action_manager._terms["finger_joint"]
    arm_ids, arm_names = robot.find_joints(
        list(PANDA_ARM_JOINT_NAMES), preserve_order=True
    )
    if tuple(arm_names) != PANDA_ARM_JOINT_NAMES:
        raise ValueError(f"Panda arm order mismatch: {arm_names}")
    finger_ids, finger_names = robot.find_joints(["finger_joint"], preserve_order=True)
    if finger_names != ["finger_joint"]:
        raise ValueError(f"Finger joint mismatch: {finger_names}")

    def numpy(value):
        return value.detach().cpu().numpy()

    zero_action = torch.zeros((1, 8), device=env.device)
    results = []
    for case in build_joint_velocity_smoke_cases(args_cli.command_magnitude):
        observation, _ = env.reset(expensive=False)
        for _ in range(args_cli.settle_steps):
            observation, _, terminated, truncated, _ = env.step(
                zero_action, expensive=False
            )
            if bool(terminated[0]) or bool(truncated[0]):
                raise RuntimeError(f"Smoke settle ended for {case['label']}")
        if case["kind"] == "gripper" and case["precondition_finger_target"] > 0.5:
            precondition_action = zero_action.clone()
            precondition_action[:, 7] = 1.0
            for _ in range(NATIVE_GRIPPER_PRECONDITION_STEPS):
                observation, _, terminated, truncated, _ = env.step(
                    precondition_action, expensive=False
                )
                if bool(terminated[0]) or bool(truncated[0]):
                    raise RuntimeError(
                        f"Smoke gripper precondition ended for {case['label']}"
                    )
        q_before = numpy(observation["policy"]["arm_joint_pos"])[0]
        dq_before = numpy(observation["policy"]["arm_joint_vel"])[0]
        finger_position_before = float(numpy(robot.data.joint_pos[:, finger_ids])[0, 0])
        finger_velocity_before = float(numpy(robot.data.joint_vel[:, finger_ids])[0, 0])
        action = torch.tensor(case["action"], device=env.device).reshape(1, -1)
        observation, _, terminated, truncated, _ = env.step(action, expensive=False)
        q_after = numpy(observation["policy"]["arm_joint_pos"])[0]
        dq_after = numpy(observation["policy"]["arm_joint_vel"])[0]
        finger_position_after = float(numpy(robot.data.joint_pos[:, finger_ids])[0, 0])
        finger_velocity_after = float(numpy(robot.data.joint_vel[:, finger_ids])[0, 0])
        arm_term = root_env.action_manager._terms["arm"]
        result = {
            **case,
            "joint_position_before": q_before.tolist(),
            "joint_velocity_before": dq_before.tolist(),
            "joint_position_after": q_after.tolist(),
            "joint_velocity_after": dq_after.tolist(),
            "processed_joint_velocity": numpy(arm_term.processed_actions)[0].tolist(),
            "articulation_joint_velocity_target": numpy(
                robot.data.joint_vel_target[:, arm_ids]
            )[0].tolist(),
            "soft_joint_position_limits": numpy(
                robot.data.soft_joint_pos_limits[:, arm_ids]
            )[0].tolist(),
            "finger_position_target": float(
                numpy(robot.data.joint_pos_target[:, finger_ids])[0, 0]
            ),
            "processed_finger_position_target": float(
                numpy(finger_term.processed_actions)[0, 0]
            ),
            "finger_position_before": finger_position_before,
            "finger_velocity_before": finger_velocity_before,
            "finger_position_after": finger_position_after,
            "finger_velocity_after": finger_velocity_after,
            "finger_average_slew_rad_s": (
                (finger_position_after - finger_position_before) * 15.0
            ),
            "terminated": bool(terminated[0]),
            "truncated": bool(truncated[0]),
        }
        results.append(result)

    env.reset(expensive=False)
    reset_excitation = zero_action.clone()
    reset_excitation[:, 0] = args_cli.command_magnitude
    env.step(reset_excitation, expensive=False)
    observation, _ = env.reset(expensive=False)
    observation, _, terminated, truncated, _ = env.step(zero_action, expensive=False)
    if bool(terminated[0]) or bool(truncated[0]):
        raise RuntimeError("Reset probe ended the episode")
    reset_probe = {
        "default_joint_position": numpy(robot.data.default_joint_pos[:, arm_ids])[
            0
        ].tolist(),
        "joint_position": numpy(observation["policy"]["arm_joint_pos"])[0].tolist(),
        "joint_velocity": numpy(observation["policy"]["arm_joint_vel"])[0].tolist(),
        "joint_velocity_target": numpy(robot.data.joint_vel_target[:, arm_ids])[
            0
        ].tolist(),
        "default_finger_position": float(
            numpy(robot.data.default_joint_pos[:, finger_ids])[0, 0]
        ),
        "finger_position": float(numpy(robot.data.joint_pos[:, finger_ids])[0, 0]),
        "finger_velocity": float(numpy(robot.data.joint_vel[:, finger_ids])[0, 0]),
        "finger_position_target": float(
            numpy(robot.data.joint_pos_target[:, finger_ids])[0, 0]
        ),
    }
    return {
        "schema_version": 1,
        "smoke_profile": SMOKE_PROFILE,
        "controller_profile": PI05_DROID_JOINTVELOCITY_PROFILE,
        "environment": args_cli.environment,
        "command_magnitude": args_cli.command_magnitude,
        "settle_steps": args_cli.settle_steps,
        "expected_gripper_drive_profile": args_cli.expected_gripper_drive_profile,
        "gripper_precondition_steps": NATIVE_GRIPPER_PRECONDITION_STEPS,
        "runtime_contract": runtime_contract,
        "cases": results,
        "reset_probe": reset_probe,
    }


def _kit_child_main(argv: list[str]) -> int:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", default="DROID-FoodBussing")
    parser.add_argument("--command-magnitude", type=float, default=0.25)
    parser.add_argument("--settle-steps", type=int, default=5)
    parser.add_argument("--expected-gripper-drive-profile", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--kit-child", action="store_true")
    parser.add_argument("--raw-json", type=Path, required=True)
    parser.add_argument("--ready-json", type=Path, required=True)
    parser.add_argument("--failure-json", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args(argv)
    args_cli.enable_cameras = True
    args_cli.headless = True
    env = None
    stage = "launch_simulation_app"
    try:
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app
        stage = "initialize_environment"
        import gymnasium as gym
        from isaaclab_tasks.utils import parse_env_cfg

        import polaris.environments  # noqa: F401
        from polaris.environments.droid_cfg import (
            DroidJointVelocityActionCfg,
            DroidJointVelocityEventCfg,
            DroidJointVelocityObservationCfg,
        )
        from polaris.environments.robot_cfg import (
            make_nvidia_droid_joint_velocity_cfg,
        )
        from polaris.joint_velocity_runtime import validate_joint_velocity_runtime

        env_cfg = parse_env_cfg(
            args_cli.environment,
            device=args_cli.device,
            num_envs=1,
            use_fabric=True,
        )
        env_cfg.scene.robot = make_nvidia_droid_joint_velocity_cfg()
        env_cfg.actions = DroidJointVelocityActionCfg()
        env_cfg.events = DroidJointVelocityEventCfg()
        env_cfg.observations = DroidJointVelocityObservationCfg()
        env = gym.make(args_cli.environment, cfg=env_cfg)
        env.reset(expensive=False)
        stage = "validate_joint_velocity_runtime"
        runtime_contract = validate_joint_velocity_runtime(
            env,
            expected_gripper_drive_profile=args_cli.expected_gripper_drive_profile,
        )
        stage = "run_controller_capture"
        payload = _run_capture(args_cli, env, runtime_contract)
        if payload is None:
            raise RuntimeError("Kit child produced no capture")
        stage = "close_environment"
        env.close()
        payload["lifecycle"] = {
            "env_close": "complete",
            "simulation_app_close": "pending_child_exit",
            "capture_stage": "kit_child_after_env_close_before_simulation_app_close",
        }
        from polaris.joint_velocity_smoke import validate_joint_velocity_smoke

        stage = "validate_child_capture"
        validated = validate_joint_velocity_smoke(
            payload, require_parent_completion=False
        )
        stage = "publish_child_capture"
        raw_bytes = _write_immutable_json(args_cli.raw_json, validated)
        stage = "publish_ready_then_invoke_simulation_app_close"
        _write_immutable_json(
            args_cli.ready_json, _child_ready_payload(args_cli.raw_json, raw_bytes)
        )
        simulation_app.close()
    except BaseException as error:
        _abort_kit_child(args_cli.failure_json, stage=stage, error=error)
    return 0


def _parent_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", required=True)
    known, _ = parser.parse_known_args(argv)
    output_path = _declared_absolute_path(known.output_json, "smoke output")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.with_name(output_path.name + ".child-close.json")
    ready_path = raw_path.with_name(raw_path.name + ".ready.json")
    failure_path = raw_path.with_name(raw_path.name + ".failure.json")
    for path, field in (
        (raw_path, "child capture"),
        (ready_path, "child ready marker"),
        (failure_path, "child failure status"),
    ):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing existing {field} {path}")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        *argv,
        "--kit-child",
        "--raw-json",
        str(raw_path),
        "--ready-json",
        str(ready_path),
        "--failure-json",
        str(failure_path),
    ]
    completed = subprocess.run(command, check=False)
    finalize_child_capture(
        raw_path,
        ready_path,
        failure_path,
        output_path,
        child_exit_code=completed.returncode,
    )
    print(f"Immutable joint-velocity smoke: {output_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--kit-child" in argv:
        return _kit_child_main(argv)
    return _parent_main(argv)


if __name__ == "__main__":
    sys.exit(main())
