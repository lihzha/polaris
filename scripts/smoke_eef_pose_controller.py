"""Headless scripted smoke test for PolaRiS absolute EEF pose control.

This entrypoint launches Isaac Sim. It is intentionally not part of the CPU
unit-test suite.
"""

import argparse
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--environment", default="DROID-FoodBussing")
parser.add_argument("--hold-steps", type=int, default=45)
parser.add_argument("--position-delta", type=float, default=0.04)
parser.add_argument("--rotation-degrees", type=float, default=15.0)
parser.add_argument("--position-tolerance", type=float, default=0.01)
parser.add_argument("--rotation-tolerance-degrees", type=float, default=5.0)
parser.add_argument("--frame-position-tolerance", type=float, default=1e-5)
parser.add_argument("--frame-rotation-tolerance-degrees", type=float, default=0.01)
parser.add_argument(
    "--output-json",
    type=Path,
    default=None,
    help="Optional machine-readable per-axis result summary.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _wxyz_to_xyzw(quaternion):
    return quaternion[[1, 2, 3, 0]]


def main() -> int:
    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.utils import math as math_utils
    from scipy.spatial.transform import Rotation

    import polaris.environments  # noqa: F401
    from polaris.config import LAP_EEF_FRAME
    from polaris.environments.droid_cfg import EefPoseActionCfg
    from polaris.policy.lap_eef_pose_client import anchor_action_chunk

    env_cfg = parse_env_cfg(
        args_cli.environment,
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )
    env_cfg.actions = EefPoseActionCfg()
    env = gym.make(args_cli.environment, cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]
    link0_index = robot.data.body_names.index("panda_link0")
    link8_index = robot.data.body_names.index(LAP_EEF_FRAME)

    def direct_link8_pose():
        position, quaternion = math_utils.subtract_frame_transforms(
            robot.data.body_pos_w[:, link0_index],
            robot.data.body_quat_w[:, link0_index],
            robot.data.body_pos_w[:, link8_index],
            robot.data.body_quat_w[:, link8_index],
        )
        return (
            position[0].detach().cpu().numpy(),
            quaternion[0].detach().cpu().numpy(),
        )

    def validate_observation_frame(observation):
        observed_position = observation["policy"]["eef_pos"][0].detach().cpu().numpy()
        observed_quaternion = (
            observation["policy"]["eef_quat"][0].detach().cpu().numpy()
        )
        direct_position, direct_quaternion = direct_link8_pose()
        position_error = float(np.linalg.norm(observed_position - direct_position))
        observed_rotation = Rotation.from_quat(_wxyz_to_xyzw(observed_quaternion))
        direct_rotation = Rotation.from_quat(_wxyz_to_xyzw(direct_quaternion))
        rotation_error = float((direct_rotation.inv() * observed_rotation).magnitude())
        if (
            position_error > args_cli.frame_position_tolerance
            or rotation_error > math.radians(args_cli.frame_rotation_tolerance_degrees)
        ):
            raise RuntimeError(
                "LAP EEF observation is not the articulation's direct panda_link8 "
                f"pose: position_error={position_error}, rotation_error={rotation_error}"
            )
        return position_error, rotation_error

    angle = math.radians(args_cli.rotation_degrees)
    delta = args_cli.position_delta
    test_cases = [
        ("hold", np.zeros(6)),
        ("translate +x", np.array([delta, 0.0, 0.0, 0.0, 0.0, 0.0])),
        ("translate +y", np.array([0.0, delta, 0.0, 0.0, 0.0, 0.0])),
        ("translate +z", np.array([0.0, 0.0, delta, 0.0, 0.0, 0.0])),
        ("rotate +x", np.array([0.0, 0.0, 0.0, angle, 0.0, 0.0])),
        ("rotate +y", np.array([0.0, 0.0, 0.0, 0.0, angle, 0.0])),
        ("rotate +z", np.array([0.0, 0.0, 0.0, 0.0, 0.0, angle])),
    ]

    failures = 0
    results = []
    try:
        for label, pose_delta in test_cases:
            observation, _ = env.reset(expensive=False)
            reset_frame_position_error, reset_frame_rotation_error = (
                validate_observation_frame(observation)
            )
            anchor_position = observation["policy"]["eef_pos"][0].detach().cpu().numpy()
            anchor_quaternion = (
                observation["policy"]["eef_quat"][0].detach().cpu().numpy()
            )
            lap_delta = np.concatenate([pose_delta, np.array([1.0])])[None, :]
            target = anchor_action_chunk(lap_delta, anchor_position, anchor_quaternion)[
                0
            ]
            action = torch.as_tensor(target, device=env.device).reshape(1, -1)

            for _ in range(args_cli.hold_steps):
                observation, _, terminated, truncated, _ = env.step(
                    action, expensive=False
                )
                if bool(terminated[0]) or bool(truncated[0]):
                    raise RuntimeError(f"Episode ended during smoke case {label!r}")

            actual_position = observation["policy"]["eef_pos"][0].detach().cpu().numpy()
            actual_quaternion = (
                observation["policy"]["eef_quat"][0].detach().cpu().numpy()
            )
            final_frame_position_error, final_frame_rotation_error = (
                validate_observation_frame(observation)
            )
            position_error = float(np.linalg.norm(actual_position - target[:3]))
            target_rotation = Rotation.from_quat(_wxyz_to_xyzw(target[3:7]))
            actual_rotation = Rotation.from_quat(_wxyz_to_xyzw(actual_quaternion))
            rotation_error = float(
                (target_rotation.inv() * actual_rotation).magnitude()
            )
            passed = (
                position_error <= args_cli.position_tolerance
                and rotation_error <= math.radians(args_cli.rotation_tolerance_degrees)
            )
            failures += int(not passed)
            results.append(
                {
                    "case": label,
                    "passed": passed,
                    "position_error_m": position_error,
                    "rotation_error_rad": rotation_error,
                    "target_position": target[:3].tolist(),
                    "actual_position": actual_position.tolist(),
                    "target_quaternion_wxyz": target[3:7].tolist(),
                    "actual_quaternion_wxyz": actual_quaternion.tolist(),
                    "reset_frame_position_error_m": reset_frame_position_error,
                    "reset_frame_rotation_error_rad": reset_frame_rotation_error,
                    "final_frame_position_error_m": final_frame_position_error,
                    "final_frame_rotation_error_rad": final_frame_rotation_error,
                }
            )
            print(
                f"{label:>14}: {'PASS' if passed else 'FAIL'} "
                f"position_error={position_error * 1000:.2f}mm "
                f"rotation_error={math.degrees(rotation_error):.2f}deg"
            )
    finally:
        env.close()

    print(f"EEF pose smoke: {len(test_cases) - failures}/{len(test_cases)} passed")
    if args_cli.output_json is not None:
        args_cli.output_json.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output_json.write_text(
            json.dumps(
                {
                    "environment": args_cli.environment,
                    "eef_frame": LAP_EEF_FRAME,
                    "hold_steps": args_cli.hold_steps,
                    "position_delta_m": args_cli.position_delta,
                    "rotation_delta_deg": args_cli.rotation_degrees,
                    "position_tolerance_m": args_cli.position_tolerance,
                    "rotation_tolerance_deg": args_cli.rotation_tolerance_degrees,
                    "frame_position_tolerance_m": args_cli.frame_position_tolerance,
                    "frame_rotation_tolerance_deg": args_cli.frame_rotation_tolerance_degrees,
                    "passed": failures == 0,
                    "results": results,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return int(failures > 0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        simulation_app.close()
