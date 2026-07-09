#!/usr/bin/env python3
"""Close-aware L40S smoke for the official-DROID position controller."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import smoke_joint_velocity_controller as lifecycle


def _numpy(value):
    return value.detach().cpu().numpy()


def _run_capture(args, env, runtime_contract):
    import numpy as np
    import torch

    from polaris.pi05_droid_position_adapter import (
        PositionActionTargetLimitError,
        adapt_official_droid_action,
        exact_position_target_guard_from_live_limits,
        expected_position_limit_contract,
    )
    from polaris.pi05_droid_position_contract import PANDA_ARM_JOINT_NAMES
    from polaris.pi05_droid_position_runtime import make_position_safety_report
    from polaris.pi05_droid_position_smoke import (
        POSITION_SMOKE_PROFILE,
        build_position_smoke_cases,
    )

    root = getattr(env, "unwrapped", env)
    robot = root.scene["robot"]
    arm_ids, arm_names = robot.find_joints(
        list(PANDA_ARM_JOINT_NAMES), preserve_order=True
    )
    finger_ids, finger_names = robot.find_joints(["finger_joint"], preserve_order=True)
    if tuple(arm_names) != PANDA_ARM_JOINT_NAMES or finger_names != ["finger_joint"]:
        raise ValueError("position smoke articulation order mismatch")
    cases = []
    for definition in build_position_smoke_cases(args.command_magnitude):
        observation, _ = env.reset(expensive=False)
        arm_term = root.action_manager._terms["arm"]
        finger_term = root.action_manager._terms["finger_joint"]
        q_observation = _numpy(observation["policy"]["arm_joint_pos"])[0]
        q_live = _numpy(robot.data.joint_pos[:, arm_ids])[0]
        if q_observation.dtype != np.float32 or not np.array_equal(
            q_observation, q_live
        ):
            raise ValueError("position smoke observation/live q mismatch")
        adapter = adapt_official_droid_action(definition["raw_action"], q_live)
        action = torch.as_tensor(
            adapter["emitted_absolute_action"], device=env.device
        ).reshape(1, 8)
        observation, _, terminated, truncated, _ = env.step(action, expensive=False)
        arm_term.record_native_all_joint_post_policy_step()
        hold = arm_term.consume_position_target_hold_report()
        dynamic = arm_term.native_all_joint_dynamic_report(include_samples=False)
        safety = make_position_safety_report(dynamic, outer_steps=1)
        cases.append(
            {
                **definition,
                "adapter": adapter,
                "processed_joint_position_target": _numpy(arm_term.processed_actions)[
                    0
                ].tolist(),
                "articulation_joint_position_target": _numpy(
                    robot.data.joint_pos_target[:, arm_ids]
                )[0].tolist(),
                "processed_finger_position_target": float(
                    _numpy(finger_term.processed_actions)[0, 0]
                ),
                "articulation_finger_position_target": float(
                    _numpy(robot.data.joint_pos_target[:, finger_ids])[0, 0]
                ),
                "target_hold": hold,
                "safety": safety,
                "measured_joint_position_after": _numpy(
                    observation["policy"]["arm_joint_pos"]
                )[0].tolist(),
                "measured_joint_velocity_after": _numpy(
                    observation["policy"]["arm_joint_vel"]
                )[0].tolist(),
                "terminated": bool(terminated[0]),
                "truncated": bool(truncated[0]),
            }
        )

    # Two consecutive commands from one conceptual open-loop chunk. The
    # second must anchor to measured q after step one, not the prior target.
    observation, _ = env.reset(expensive=False)
    arm_term = root.action_manager._terms["arm"]
    raw_reanchor = np.zeros(8, dtype=np.float64)
    raw_reanchor[0] = 0.5
    q_step1 = _numpy(robot.data.joint_pos[:, arm_ids])[0].copy()
    first = adapt_official_droid_action(raw_reanchor, q_step1)
    first_action = torch.as_tensor(
        first["emitted_absolute_action"], device=env.device
    ).reshape(1, 8)
    observation, _, terminated, truncated, _ = env.step(first_action, expensive=False)
    if bool(terminated[0]) or bool(truncated[0]):
        raise RuntimeError("fresh-reanchor step one terminated")
    arm_term.record_native_all_joint_post_policy_step()
    first_hold = arm_term.consume_position_target_hold_report()
    q_after1 = _numpy(robot.data.joint_pos[:, arm_ids])[0].copy()
    q_step2 = _numpy(observation["policy"]["arm_joint_pos"])[0].copy()
    if not np.array_equal(q_after1, q_step2):
        raise ValueError("fresh-reanchor live/observation q mismatch")
    second = adapt_official_droid_action(raw_reanchor, q_step2)
    stale_second = np.asarray(
        first["absolute_joint_position_target_rad"], dtype=np.float64
    ) + 0.2 * np.asarray(second["clipped_action"][:7], dtype=np.float64)
    second_action = torch.as_tensor(
        second["emitted_absolute_action"], device=env.device
    ).reshape(1, 8)
    observation, _, terminated, truncated, _ = env.step(second_action, expensive=False)
    if bool(terminated[0]) or bool(truncated[0]):
        raise RuntimeError("fresh-reanchor step two terminated")
    arm_term.record_native_all_joint_post_policy_step()
    second_hold = arm_term.consume_position_target_hold_report()
    reanchor_safety = make_position_safety_report(
        arm_term.native_all_joint_dynamic_report(include_samples=False),
        outer_steps=2,
    )
    target2 = np.asarray(second["absolute_joint_position_target_rad"])
    if np.array_equal(target2.astype(np.float32), stale_second.astype(np.float32)):
        raise ValueError(
            "fresh-reanchor probe did not separate measured and stale anchors"
        )
    fresh_reanchor_probe = {
        "raw_action": raw_reanchor.tolist(),
        "step1_measured_joint_position": q_step1.tolist(),
        "step1_absolute_target": first["absolute_joint_position_target_rad"],
        "step1_measured_joint_position_after": q_after1.tolist(),
        "step1_target_hold": first_hold,
        "step2_measured_joint_position": q_step2.tolist(),
        "step2_absolute_target": second["absolute_joint_position_target_rad"],
        "stale_prior_target_anchor_result": stale_second.tolist(),
        "step2_target_hold": second_hold,
        "fresh_measurement_equals_step1_after": True,
        "step2_differs_from_stale_target_anchor": True,
        "safety": reanchor_safety,
    }

    # Exercise the independent action-term guard directly. A target beyond the
    # live upper soft limit must raise during process_actions, before the first
    # Articulation.set_joint_position_target call.
    env.reset(expensive=False)
    arm_term = root.action_manager._terms["arm"]
    live_hard_limits = _numpy(robot.data.joint_pos_limits[:, arm_ids])
    live_soft_limits = _numpy(robot.data.soft_joint_pos_limits[:, arm_ids])
    target_guard_limits = exact_position_target_guard_from_live_limits(
        live_hard_limits, live_soft_limits
    )
    target = _numpy(robot.data.joint_pos[:, arm_ids]).copy()
    joint_index = 3
    target[0, joint_index] = np.nextafter(
        target_guard_limits[0, joint_index, 1],
        np.float32(np.inf),
        dtype=np.float32,
    )
    if not (
        target[0, joint_index] > live_hard_limits[0, joint_index, 1]
        and target[0, joint_index] <= live_soft_limits[0, joint_index, 1]
    ):
        raise ValueError("adversarial q4 target did not isolate hard/soft intersection")
    before_target = _numpy(robot.data.joint_pos_target[:, arm_ids]).copy()
    try:
        arm_term.process_actions(torch.as_tensor(target, device=env.device))
    except PositionActionTargetLimitError as error:
        exception_type = type(error).__name__
        exception_message = str(error)
    else:
        raise RuntimeError("adversarial position target reached the setter path")
    after_target = _numpy(robot.data.joint_pos_target[:, arm_ids]).copy()
    guard = {
        "position_limit_contract": expected_position_limit_contract(),
        "joint_index": joint_index,
        "controlling_bound_source": "live_joint_pos_limits",
        "hard_upper_limit_rad": float(live_hard_limits[0, joint_index, 1]),
        "soft_upper_limit_rad": float(live_soft_limits[0, joint_index, 1]),
        "intersection_guard_upper_limit_rad": float(
            target_guard_limits[0, joint_index, 1]
        ),
        "adversarial_target_rad": float(target[0, joint_index]),
        "adversarial_is_one_float32_step_above_guard": True,
        "adversarial_inside_soft_limit": True,
        "adversarial_outside_hard_limit": True,
        "articulation_target_before": before_target[0].tolist(),
        "articulation_target_after": after_target[0].tolist(),
        "exception_type": exception_type,
        "exception_message": exception_message,
        "setter_unchanged": bool(np.array_equal(before_target, after_target)),
    }
    return {
        "schema_version": 1,
        "smoke_profile": POSITION_SMOKE_PROFILE,
        "controller_profile": ("openpi_pi05_droid_fresh_jointdelta_position_v1"),
        "environment": args.environment,
        "command_magnitude": args.command_magnitude,
        "runtime_contract": runtime_contract,
        "cases": cases,
        "limit_guard_probe": guard,
        "fresh_reanchor_probe": fresh_reanchor_probe,
    }


def _kit_child(argv: list[str]) -> int:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", default="DROID-FoodBussing")
    parser.add_argument("--command-magnitude", type=float, default=1.5)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--kit-child", action="store_true")
    parser.add_argument("--raw-json", type=Path, required=True)
    parser.add_argument("--ready-json", type=Path, required=True)
    parser.add_argument("--failure-json", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args(argv)
    args.enable_cameras = True
    args.headless = True
    env = None
    stage = "launch_simulation_app"
    try:
        app = AppLauncher(args).app
        stage = "initialize_position_environment"
        import gymnasium as gym
        from isaaclab_tasks.utils import parse_env_cfg

        import polaris.environments  # noqa: F401
        from polaris.environments.pi05_droid_position_cfg import (
            DroidPositionAdapterActionCfg,
            DroidPositionAdapterEventCfg,
            DroidPositionAdapterObservationCfg,
        )
        from polaris.environments.pi05_droid_position_robot_cfg import (
            make_nvidia_droid_position_adapter_cfg,
        )
        from polaris.pi05_droid_position_runtime import (
            capture_position_adapter_runtime,
        )
        from polaris.pi05_droid_position_smoke import validate_position_smoke

        cfg = parse_env_cfg(
            args.environment, device=args.device, num_envs=1, use_fabric=True
        )
        cfg.scene.robot = make_nvidia_droid_position_adapter_cfg()
        cfg.actions = DroidPositionAdapterActionCfg()
        cfg.events = DroidPositionAdapterEventCfg()
        cfg.observations = DroidPositionAdapterObservationCfg()
        env = gym.make(args.environment, cfg=cfg)
        env.reset(expensive=False)
        runtime = capture_position_adapter_runtime(env)
        stage = "run_position_controller_capture"
        payload = _run_capture(args, env, runtime)
        stage = "close_environment"
        env.close()
        payload["lifecycle"] = {
            "env_close": "complete",
            "simulation_app_close": "pending_child_exit",
            "capture_stage": "kit_child_after_env_close_before_simulation_app_close",
        }
        validated = validate_position_smoke(payload, require_parent_completion=False)
        raw = lifecycle._write_immutable_json(args.raw_json, validated)
        lifecycle._write_immutable_json(
            args.ready_json, lifecycle._child_ready_payload(args.raw_json, raw)
        )
        app.close()
    except BaseException as error:
        lifecycle._abort_kit_child(args.failure_json, stage=stage, error=error)
    return 0


def _parent(argv: list[str]) -> int:
    from polaris.pi05_droid_position_smoke import (
        publish_position_smoke,
        validate_position_smoke,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", required=True)
    known, _ = parser.parse_known_args(argv)
    output = lifecycle._declared_absolute_path(known.output_json, "position smoke")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output.with_name(output.name + ".child-close.json")
    ready_path = raw_path.with_name(raw_path.name + ".ready.json")
    failure_path = raw_path.with_name(raw_path.name + ".failure.json")
    for path in (output, raw_path, ready_path, failure_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing existing smoke artifact: {path}")
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
    if failure_path.exists() or completed.returncode != 0:
        raise RuntimeError("position controller Kit child failed")
    raw, raw_bytes, _ = lifecycle._read_immutable_json(raw_path, "position raw")
    _, ready_bytes, _ = lifecycle._read_immutable_json(ready_path, "position ready")
    if ready_bytes != lifecycle._canonical_json(
        lifecycle._child_ready_payload(raw_path, raw_bytes)
    ):
        raise ValueError("position ready marker does not bind raw capture")
    validated = validate_position_smoke(raw, require_parent_completion=False)
    final = dict(validated)
    final["lifecycle"] = {
        "env_close": "complete",
        "simulation_app_close": "invoked_then_child_exited_zero",
        "capture_stage": "stdlib_parent_after_kit_child_exit",
    }
    final["completion"] = {
        "child_exit_code": 0,
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "ready_sha256": hashlib.sha256(ready_bytes).hexdigest(),
    }
    final["status"] = "pass"
    final["case_count"] = len(final["cases"])
    publish_position_smoke(output, final)
    print(f"Immutable DROID position controller smoke: {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    return _kit_child(argv) if "--kit-child" in argv else _parent(argv)


if __name__ == "__main__":
    raise SystemExit(main())
