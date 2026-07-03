"""Standalone Isaac smoke for native ``pi05_droid`` joint-velocity control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--environment", default="DROID-FoodBussing")
parser.add_argument("--command-magnitude", type=float, default=0.25)
parser.add_argument("--settle-steps", type=int, default=5)
parser.add_argument("--output-json", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _numpy(value):
    return value.detach().cpu().numpy()


def main() -> int:
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    import polaris.environments  # noqa: F401
    from polaris.environments.droid_cfg import (
        DroidJointVelocityActionCfg,
        DroidJointVelocityObservationCfg,
    )
    from polaris.environments.robot_cfg import NVIDIA_DROID_JOINT_VELOCITY
    from polaris.joint_velocity_runtime import validate_joint_velocity_runtime
    from polaris.joint_velocity_smoke import (
        SMOKE_PROFILE,
        build_joint_velocity_smoke_cases,
        validate_joint_velocity_smoke,
    )
    from polaris.pi05_droid_jointvelocity_contract import (
        PANDA_ARM_JOINT_NAMES,
        PI05_DROID_JOINTVELOCITY_PROFILE,
    )

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
    root_env = getattr(env, "unwrapped", env)
    robot = root_env.scene["robot"]
    arm_ids, arm_names = robot.find_joints(
        list(PANDA_ARM_JOINT_NAMES), preserve_order=True
    )
    if tuple(arm_names) != PANDA_ARM_JOINT_NAMES:
        raise ValueError(f"Panda arm order mismatch: {arm_names}")
    finger_ids, finger_names = robot.find_joints(["finger_joint"], preserve_order=True)
    if finger_names != ["finger_joint"]:
        raise ValueError(f"Finger joint mismatch: {finger_names}")

    zero_action = torch.zeros((1, 8), device=env.device)
    results = []
    try:
        for case in build_joint_velocity_smoke_cases(args_cli.command_magnitude):
            observation, _ = env.reset(expensive=False)
            for _ in range(args_cli.settle_steps):
                observation, _, terminated, truncated, _ = env.step(
                    zero_action, expensive=False
                )
                if bool(terminated[0]) or bool(truncated[0]):
                    raise RuntimeError(f"Smoke settle ended for {case['label']}")
            q_before = _numpy(observation["policy"]["arm_joint_pos"])[0]
            dq_before = _numpy(observation["policy"]["arm_joint_vel"])[0]
            action = torch.tensor(case["action"], device=env.device).reshape(1, -1)
            observation, _, terminated, truncated, _ = env.step(action, expensive=False)
            q_after = _numpy(observation["policy"]["arm_joint_pos"])[0]
            dq_after = _numpy(observation["policy"]["arm_joint_vel"])[0]
            arm_term = root_env.action_manager._terms["arm"]
            result = {
                **case,
                "joint_position_before": q_before.tolist(),
                "joint_velocity_before": dq_before.tolist(),
                "joint_position_after": q_after.tolist(),
                "joint_velocity_after": dq_after.tolist(),
                "processed_joint_velocity": _numpy(arm_term.processed_actions)[
                    0
                ].tolist(),
                "articulation_joint_velocity_target": _numpy(
                    robot.data.joint_vel_target[:, arm_ids]
                )[0].tolist(),
                "soft_joint_position_limits": _numpy(
                    robot.data.soft_joint_pos_limits[:, arm_ids]
                )[0].tolist(),
                "finger_position_target": float(
                    _numpy(robot.data.joint_pos_target[:, finger_ids])[0, 0]
                ),
                "terminated": bool(terminated[0]),
                "truncated": bool(truncated[0]),
            }
            results.append(result)

        # Install a nonzero target, reset, and take one zero-command step.  The
        # probe proves both state restoration and that no velocity target leaks
        # across episode reset.
        env.reset(expensive=False)
        reset_excitation = zero_action.clone()
        reset_excitation[:, 0] = args_cli.command_magnitude
        env.step(reset_excitation, expensive=False)
        observation, _ = env.reset(expensive=False)
        observation, _, terminated, truncated, _ = env.step(
            zero_action, expensive=False
        )
        if bool(terminated[0]) or bool(truncated[0]):
            raise RuntimeError("Reset probe ended the episode")
        reset_probe = {
            "default_joint_position": _numpy(robot.data.default_joint_pos[:, arm_ids])[
                0
            ].tolist(),
            "joint_position": _numpy(observation["policy"]["arm_joint_pos"])[
                0
            ].tolist(),
            "joint_velocity": _numpy(observation["policy"]["arm_joint_vel"])[
                0
            ].tolist(),
            "joint_velocity_target": _numpy(robot.data.joint_vel_target[:, arm_ids])[
                0
            ].tolist(),
        }
    finally:
        env.close()

    payload = {
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
    validated = validate_joint_velocity_smoke(payload)
    args_cli.output_json.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output_json.write_text(
        json.dumps(validated, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Joint-velocity smoke passed: {validated['case_count']} cases; "
        f"artifact={args_cli.output_json}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        simulation_app.close()
