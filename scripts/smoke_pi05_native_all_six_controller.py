#!/usr/bin/env python3
"""Run the no-model native all-six coupled controller smoke in real Isaac."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _write_immutable(path: Path, value: Any) -> bytes:
    rendered = _canonical_bytes(value)
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
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return rendered


def _read_immutable(path: Path, field: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink():
        raise ValueError(f"{field} must not be a symlink")
    file_stat = path.stat()
    if (
        not stat.S_ISREG(file_stat.st_mode)
        or file_stat.st_nlink != 1
        or stat.S_IMODE(file_stat.st_mode) != 0o444
    ):
        raise ValueError(f"{field} must be one mode-0444 regular link")
    rendered = path.read_bytes()
    value = json.loads(rendered)
    if rendered != _canonical_bytes(value):
        raise ValueError(f"{field} must be canonical JSON")
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value, rendered


def _child_main(argv: list[str]) -> int:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", default="DROID-FoodBussing")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--raw-json", type=Path, required=True)
    parser.add_argument("--ready-json", type=Path, required=True)
    parser.add_argument("--failure-json", type=Path, required=True)
    parser.add_argument("--kit-child", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args(argv)
    args.enable_cameras = True
    args.headless = True
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    env = None
    stage = "initialize"
    try:
        import gymnasium as gym
        import torch
        from isaaclab_tasks.utils import parse_env_cfg

        import polaris.environments  # noqa: F401
        from polaris.environments.droid_cfg import (
            DroidJointVelocityActionCfg,
            DroidJointVelocityEventCfg,
            DroidJointVelocityObservationCfg,
        )
        from polaris.environments.robot_cfg import NVIDIA_DROID_JOINT_VELOCITY
        from polaris.joint_velocity_runtime import validate_joint_velocity_runtime
        from polaris.native_all_six_smoke import (
            PRECONDITION_STEPS,
            SETTLE_STEPS,
            coupled_scenario_plans,
            validate_native_all_six_smoke,
        )
        from polaris.native_gripper_runtime import native_gripper_reset_report
        from polaris.pi05_droid_jointvelocity_contract import (
            NATIVE_GRIPPER_DRIVE_PROFILE,
        )

        stage = "create_environment"
        env_cfg = parse_env_cfg(
            args.environment,
            device=args.device,
            num_envs=1,
            use_fabric=True,
        )
        env_cfg.scene.robot = NVIDIA_DROID_JOINT_VELOCITY.copy()
        env_cfg.actions = DroidJointVelocityActionCfg()
        env_cfg.events = DroidJointVelocityEventCfg()
        env_cfg.observations = DroidJointVelocityObservationCfg()
        env = gym.make(args.environment, cfg=env_cfg)
        env.reset(expensive=False)
        stage = "validate_runtime"
        runtime_contract = validate_joint_velocity_runtime(
            env,
            expected_gripper_drive_profile=NATIVE_GRIPPER_DRIVE_PROFILE,
        )
        root_env = getattr(env, "unwrapped", env)
        arm_term = root_env.action_manager._terms["arm"]
        zero_open = torch.zeros((1, 8), dtype=torch.float32, device=env.device)
        zero_closed = zero_open.clone()
        zero_closed[:, 7] = 1.0
        scenarios = []
        for plan in coupled_scenario_plans():
            stage = f"reset_{plan['label']}"
            env.reset(expensive=False)
            reset_write = native_gripper_reset_report(env)
            for _ in range(SETTLE_STEPS):
                _, _, terminated, truncated, _ = env.step(zero_open, expensive=False)
                if bool(terminated[0]) or bool(truncated[0]):
                    raise RuntimeError(f"settle ended in {plan['label']}")
                arm_term.record_native_all_joint_post_policy_step()
            if plan["precondition"] == "closed":
                for _ in range(PRECONDITION_STEPS):
                    _, _, terminated, truncated, _ = env.step(
                        zero_closed, expensive=False
                    )
                    if bool(terminated[0]) or bool(truncated[0]):
                        raise RuntimeError(
                            f"closed precondition ended in {plan['label']}"
                        )
                    arm_term.record_native_all_joint_post_policy_step()
            arm_term.reset_native_all_joint_dynamic_report()
            terminated_values = []
            truncated_values = []
            for action_values in plan["actions"]:
                action = torch.tensor(
                    action_values,
                    dtype=torch.float32,
                    device=env.device,
                ).reshape(1, 8)
                _, _, terminated, truncated, _ = env.step(action, expensive=False)
                arm_term.record_native_all_joint_post_policy_step()
                terminated_values.append(bool(terminated[0]))
                truncated_values.append(bool(truncated[0]))
            scenarios.append(
                {
                    **plan,
                    "terminated": terminated_values,
                    "truncated": truncated_values,
                    "reset_write": reset_write,
                    "dynamic": arm_term.native_all_joint_dynamic_report(
                        include_samples=True
                    ),
                }
            )

        stage = "close_environment"
        env.close()
        env = None
        payload = {
            "schema_version": 1,
            "profile": "pi05_droid_native_all_six_coupled_controller_smoke_v1",
            "controller_profile": "openpi_pi05_droid_native_jointvelocity_v1",
            "gripper_profile": runtime_contract["all_six_gripper"]["profile"],
            "environment": args.environment,
            "runtime_contract": runtime_contract,
            "mimic_joint_contract": runtime_contract["all_six_gripper"][
                "mimic_joint_contract"
            ],
            "scenario_plans": coupled_scenario_plans(),
            "scenarios": scenarios,
            "lifecycle": {
                "env_close": "complete",
                "simulation_app_close": "pending_immediate_invocation",
                "publication": "kit_child_before_simulation_app_close",
            },
        }
        stage = "validate_child_payload"
        validated = validate_native_all_six_smoke(
            payload, require_parent_completion=False
        )
        stage = "publish_child_payload"
        raw_bytes = _write_immutable(args.raw_json, validated)
        ready = {
            "schema_version": 1,
            "profile": "pi05_droid_native_all_six_coupled_controller_ready_v1",
            "status": "ready_for_simulation_app_close",
            "raw_path": str(args.raw_json),
            "raw_size": len(raw_bytes),
            "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        }
        _write_immutable(args.ready_json, ready)
        stage = "close_simulation_app"
        simulation_app.close()
        return 0
    except BaseException as error:
        if env is not None:
            try:
                env.close()
            except BaseException:
                pass
        try:
            _write_immutable(
                args.failure_json,
                {
                    "schema_version": 1,
                    "profile": "pi05_droid_native_all_six_coupled_controller_failure_v1",
                    "stage": stage,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        except BaseException:
            pass
        try:
            simulation_app.close()
        except BaseException:
            pass
        raise


def _parent_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    known, _ = parser.parse_known_args(argv)
    output = known.output_json
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = output.with_name(output.name + ".child-close.json")
    ready = raw.with_name(raw.name + ".ready.json")
    failure = raw.with_name(raw.name + ".failure.json")
    for path in (output, raw, ready, failure):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing existing smoke path: {path}")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        *argv,
        "--kit-child",
        "--raw-json",
        str(raw),
        "--ready-json",
        str(ready),
        "--failure-json",
        str(failure),
    ]
    child = subprocess.run(command, check=False)
    if child.returncode != 0:
        raise RuntimeError(f"all-six Kit child exited {child.returncode}")
    if failure.exists() or failure.is_symlink():
        raise RuntimeError("all-six Kit child published a failure artifact")
    child_payload, raw_bytes = _read_immutable(raw, "all-six child capture")
    ready_payload, ready_bytes = _read_immutable(ready, "all-six ready marker")
    expected_ready = {
        "schema_version": 1,
        "profile": "pi05_droid_native_all_six_coupled_controller_ready_v1",
        "status": "ready_for_simulation_app_close",
        "raw_path": str(raw),
        "raw_size": len(raw_bytes),
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }
    if ready_payload != expected_ready:
        raise ValueError("all-six ready marker does not bind the child capture")
    from polaris.native_all_six_smoke import (
        publish_immutable_native_all_six_smoke,
        validate_native_all_six_smoke,
    )

    validate_native_all_six_smoke(child_payload, require_parent_completion=False)
    final = dict(child_payload)
    final["lifecycle"] = {
        "env_close": "complete",
        "simulation_app_close": "invoked_then_child_exited_zero",
        "publication": "stdlib_parent_after_child_exit",
    }
    final["completion"] = {
        "child_exit_code": 0,
        "raw_path": str(raw),
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "raw_size": len(raw_bytes),
        "ready_path": str(ready),
        "ready_sha256": hashlib.sha256(ready_bytes).hexdigest(),
        "ready_size": len(ready_bytes),
    }
    artifact = publish_immutable_native_all_six_smoke(output, final)
    print(f"native_all_six_smoke_path={artifact['path']}")
    print(f"native_all_six_smoke_sha256={artifact['sha256']}")
    return 0


def main() -> int:
    if "--kit-child" in sys.argv[1:]:
        return _child_main(sys.argv[1:])
    return _parent_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
