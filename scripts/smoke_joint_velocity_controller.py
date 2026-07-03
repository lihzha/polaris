#!/usr/bin/env python3
"""Close-aware parent/Kit-child smoke for native DROID velocity control."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def _write_child_capture(path: Path, payload: dict) -> None:
    rendered = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as output:
            output.write(rendered)
            output.flush()
            os.fsync(output.fileno())
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _close_kit_resources(env, simulation_app) -> None:
    """Attempt Kit close even when environment close fails."""

    try:
        if env is not None:
            env.close()
    finally:
        simulation_app.close()


def finalize_child_capture(
    raw_path: Path, output_path: Path, *, child_exit_code: int
) -> dict:
    """Publish only after the close-aware Kit child has exited zero."""

    from polaris.joint_velocity_smoke import (
        publish_immutable_joint_velocity_smoke,
        validate_joint_velocity_smoke,
    )

    if child_exit_code != 0:
        raise ValueError(f"Kit child exited nonzero: {child_exit_code}")
    raw_path = Path(raw_path)
    if raw_path.is_symlink() or not raw_path.is_file():
        raise ValueError("Kit child did not publish a regular close capture")
    raw_stat = raw_path.stat()
    if raw_stat.st_nlink != 1 or (raw_stat.st_mode & 0o777) != 0o400:
        raise ValueError("Kit child close capture must be one mode-0400 link")
    raw_bytes = raw_path.read_bytes()
    raw = json.loads(raw_bytes)
    canonical_raw = (
        json.dumps(
            raw,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    if raw_bytes != canonical_raw:
        raise ValueError("Kit child close capture is not canonical JSON")
    validated_raw = validate_joint_velocity_smoke(raw, require_parent_completion=False)
    final_payload = dict(validated_raw)
    final_payload.pop("status", None)
    final_payload.pop("case_count", None)
    final_payload["lifecycle"] = {
        "env_close": "complete",
        "simulation_app_close": "complete",
        "capture_stage": "stdlib_parent_after_kit_child_exit",
    }
    final_payload["completion"] = {
        "child_exit_code": 0,
        "publication_stage": "stdlib_parent_after_child_exit",
        "child_capture_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "child_capture_size": len(raw_bytes),
        "child_capture_mode": "0400",
        "child_capture_path": str(raw_path.resolve()),
    }
    return publish_immutable_joint_velocity_smoke(output_path, final_payload)


def _run_capture(args_cli, env, runtime_contract) -> dict:
    import torch

    from polaris.joint_velocity_smoke import (
        SMOKE_PROFILE,
        build_joint_velocity_smoke_cases,
    )
    from polaris.pi05_droid_jointvelocity_contract import (
        PANDA_ARM_JOINT_NAMES,
        PI05_DROID_JOINTVELOCITY_PROFILE,
    )

    root_env = getattr(env, "unwrapped", env)
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
        q_before = numpy(observation["policy"]["arm_joint_pos"])[0]
        dq_before = numpy(observation["policy"]["arm_joint_vel"])[0]
        action = torch.tensor(case["action"], device=env.device).reshape(1, -1)
        observation, _, terminated, truncated, _ = env.step(action, expensive=False)
        q_after = numpy(observation["policy"]["arm_joint_pos"])[0]
        dq_after = numpy(observation["policy"]["arm_joint_vel"])[0]
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
    }
    return {
        "schema_version": 1,
        "smoke_profile": SMOKE_PROFILE,
        "controller_profile": PI05_DROID_JOINTVELOCITY_PROFILE,
        "environment": args_cli.environment,
        "command_magnitude": args_cli.command_magnitude,
        "settle_steps": args_cli.settle_steps,
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
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--kit-child", action="store_true")
    parser.add_argument("--raw-json", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args(argv)
    args_cli.enable_cameras = True
    args_cli.headless = True
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    env = None
    payload = None
    try:
        import gymnasium as gym
        from isaaclab_tasks.utils import parse_env_cfg

        import polaris.environments  # noqa: F401
        from polaris.environments.droid_cfg import (
            DroidJointVelocityActionCfg,
            DroidJointVelocityObservationCfg,
        )
        from polaris.environments.robot_cfg import NVIDIA_DROID_JOINT_VELOCITY
        from polaris.joint_velocity_runtime import validate_joint_velocity_runtime

        env_cfg = parse_env_cfg(
            args_cli.environment,
            device=args_cli.device,
            num_envs=1,
            use_fabric=True,
        )
        env_cfg.scene.robot = NVIDIA_DROID_JOINT_VELOCITY.copy()
        env_cfg.actions = DroidJointVelocityActionCfg()
        env_cfg.observations = DroidJointVelocityObservationCfg()
        env = gym.make(args_cli.environment, cfg=env_cfg)
        runtime_contract = validate_joint_velocity_runtime(env)
        payload = _run_capture(args_cli, env, runtime_contract)
    except BaseException:
        _close_kit_resources(env, simulation_app)
        raise

    if payload is None:
        _close_kit_resources(env, simulation_app)
        raise RuntimeError("Kit child produced no capture")
    try:
        if env is not None:
            env.close()
    except BaseException:
        simulation_app.close()
        raise

    # This mode-0400 record is deliberately non-final. SimulationApp.close can
    # terminate Kit without returning, so only the stdlib parent may promote it,
    # and only after observing this child exit zero.
    payload["lifecycle"] = {
        "env_close": "complete",
        "simulation_app_close": "pending_child_exit",
        "capture_stage": "kit_child_after_env_close_before_simulation_app_close",
    }
    from polaris.joint_velocity_smoke import validate_joint_velocity_smoke

    try:
        validated = validate_joint_velocity_smoke(
            payload, require_parent_completion=False
        )
        _write_child_capture(args_cli.raw_json, validated)
    except BaseException:
        simulation_app.close()
        raise
    simulation_app.close()
    return 0


def _parent_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    known, _ = parser.parse_known_args(argv)
    if known.output_json.exists():
        raise FileExistsError(f"Refusing to overwrite {known.output_json}")
    known.output_json.parent.mkdir(parents=True, exist_ok=True)
    raw_path = known.output_json.with_name(known.output_json.name + ".child-close.json")
    if raw_path.exists() or raw_path.is_symlink():
        raise FileExistsError(f"Refusing existing child capture {raw_path}")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        *argv,
        "--kit-child",
        "--raw-json",
        str(raw_path),
    ]
    completed = subprocess.run(command, check=False)
    finalize_child_capture(
        raw_path, known.output_json, child_exit_code=completed.returncode
    )
    print(f"Immutable joint-velocity smoke: {known.output_json}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--kit-child" in argv:
        return _kit_child_main(argv)
    return _parent_main(argv)


if __name__ == "__main__":
    sys.exit(main())
