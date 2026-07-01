"""Headless scripted smoke test for PolaRiS absolute EEF pose control.

This entrypoint launches Isaac Sim. It is intentionally not part of the CPU
unit-test suite.
"""

import argparse
import math
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--environment", default="DROID-FoodBussing")
parser.add_argument("--hold-steps", type=int, default=45)
parser.add_argument("--position-delta", type=float, default=0.04)
parser.add_argument("--rotation-degrees", type=float, default=15.0)
parser.add_argument("--position-tolerance", type=float, default=0.01)
parser.add_argument("--rotation-tolerance-degrees", type=float, default=5.0)
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
    from scipy.spatial.transform import Rotation

    import polaris.environments  # noqa: F401
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
    try:
        for label, pose_delta in test_cases:
            observation, _ = env.reset(expensive=False)
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
            print(
                f"{label:>14}: {'PASS' if passed else 'FAIL'} "
                f"position_error={position_error * 1000:.2f}mm "
                f"rotation_error={math.degrees(rotation_error):.2f}deg"
            )
    finally:
        env.close()

    print(f"EEF pose smoke: {len(test_cases) - failures}/{len(test_cases)} passed")
    return int(failures > 0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        simulation_app.close()
