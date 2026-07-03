"""Headless scripted smoke test for PolaRiS absolute EEF pose control.

This entrypoint launches Isaac Sim. It is intentionally not part of the CPU
unit-test suite.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import traceback
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
    required=True,
    help="Required atomic machine-readable success/failure record.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = True


def _exception_evidence(error: BaseException) -> dict[str, str]:
    try:
        message = str(error)
    except BaseException:
        message = "<unprintable exception>"
    try:
        formatted_traceback = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
    except BaseException:
        formatted_traceback = "<traceback unavailable>"
    return {
        "type": f"{type(error).__module__}.{type(error).__qualname__}",
        "message": message,
        "traceback": formatted_traceback,
    }


def _print_exception(error: BaseException) -> None:
    try:
        traceback.print_exception(
            type(error), error, error.__traceback__, file=sys.stderr
        )
        sys.stdout.flush()
        sys.stderr.flush()
    except BaseException:
        pass


def _best_effort_failure_log(message: str) -> None:
    try:
        print(message, flush=True)
    except BaseException:
        pass


def _strict_json_value(value):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    raise TypeError(f"Unsupported smoke JSON value: {type(value).__name__}")


def _strict_json_bytes(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(_strict_json_value(payload), indent=2, allow_nan=False) + "\n"
    ).encode()


def _atomic_write_strict_json(path: Path, payload: dict[str, object]) -> None:
    serialized = _strict_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("xb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, path)
        path.chmod(0o444)
        published_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(published_fd)
        finally:
            os.close(published_fd)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def _result_payload(
    state: dict[str, object], *, finalized: bool, exit_code: int
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "finalized": finalized,
        "environment": args_cli.environment,
        "eef_frame": state["eef_frame"],
        "hold_steps": args_cli.hold_steps,
        "position_delta_m": args_cli.position_delta,
        "rotation_delta_deg": args_cli.rotation_degrees,
        "position_tolerance_m": args_cli.position_tolerance,
        "rotation_tolerance_deg": args_cli.rotation_tolerance_degrees,
        "frame_position_tolerance_m": args_cli.frame_position_tolerance,
        "frame_rotation_tolerance_deg": args_cli.frame_rotation_tolerance_degrees,
        "stage": state["stage"],
        "case": state["case"],
        "exit_code": exit_code,
        "raw_ik_safety_capture": state["raw_capture"],
        "ik_safety_episodes": state["safety_reports"],
        "ik_safety_adversarial": state["adversarial_result"],
        "passed": (
            finalized
            and exit_code == 0
            and state["failure"] is None
            and not state["close_failures"]
            and not state["persistence_failures"]
        ),
        "results": state["results"],
        "failure": state["failure"],
        "close_failures": state["close_failures"],
        "persistence_failures": state["persistence_failures"],
    }


def _raw_is_eligible_for_close(
    state: dict[str, object],
    *,
    exit_code: int,
    raw_published: bool,
    simulation_app,
) -> bool:
    return (
        raw_published
        and simulation_app is not None
        and exit_code == 0
        and state["stage"] == "simulation_app_close_pending"
        and state["case"] is None
        and state["failure"] is None
        and not state["close_failures"]
        and not state["persistence_failures"]
    )


def _wxyz_to_xyzw(quaternion):
    return quaternion[[1, 2, 3, 0]]


def _strict_vector_evidence(tensor):
    raw_values = [float(value) for value in tensor.detach().cpu().reshape(-1).tolist()]
    finite_mask = [math.isfinite(value) for value in raw_values]
    finite_values = [
        abs(value)
        for value, finite in zip(raw_values, finite_mask, strict=True)
        if finite
    ]
    return {
        "values": [
            value if finite else None
            for value, finite in zip(raw_values, finite_mask, strict=True)
        ],
        "finite_mask": finite_mask,
        "finite_count": sum(finite_mask),
        "max_abs": max(finite_values, default=None),
    }


def main(state: dict[str, object]) -> int:
    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab_tasks.utils import parse_env_cfg
    from scipy.spatial.transform import Rotation

    import polaris.environments  # noqa: F401
    from polaris.config import LAP_EEF_FRAME
    from polaris.eef_runtime_contract import validate_eef_runtime_frame
    from polaris.eef_runtime_contract import validate_eef_runtime_safety
    from polaris.eef_ik_safety import validate_one_step_adversarial_report
    from polaris.eef_ik_safety import CURRENT_JOINT_SOFT_LIMIT_TOLERANCE_RAD
    from polaris.eef_runtime_contract import begin_eef_safety_episode
    from polaris.eef_runtime_contract import eef_episode_safety_report
    from polaris.eef_gripper_runtime import install_eef_gripper_runtime
    from polaris.eef_gripper_runtime import record_eef_gripper_post_policy_step
    from polaris.eef_gripper_runtime import validate_eef_gripper_post_reset
    from polaris.environments.droid_cfg import EgoLapEefPoseActionCfg
    from polaris.environments.robot_cfg import configure_eef_pose_joint_safety
    from polaris.policy.lap_eef_pose_client import anchor_action_chunk

    state["eef_frame"] = LAP_EEF_FRAME
    state["stage"] = "build_environment"
    env_cfg = parse_env_cfg(
        args_cli.environment,
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )
    env_cfg.actions = EgoLapEefPoseActionCfg()
    configure_eef_pose_joint_safety(
        env_cfg.scene.robot,
        physx_cfg=env_cfg.sim.physx,
        enable_gripper_velocity_limit=True,
    )
    robot_usd_path = Path(env_cfg.scene.robot.spawn.usd_path)
    env = gym.make(args_cli.environment, cfg=env_cfg)
    state["env"] = env
    env.reset(expensive=False)
    gripper_runtime_contract = install_eef_gripper_runtime(
        env, robot_usd_path=robot_usd_path
    )

    def validate_observation_frame(observation):
        result = validate_eef_runtime_frame(
            env,
            observation,
            position_tolerance=args_cli.frame_position_tolerance,
            rotation_tolerance_radians=math.radians(
                args_cli.frame_rotation_tolerance_degrees
            ),
        )
        return result["position_error_m"], result["rotation_error_rad"]

    angle = math.radians(args_cli.rotation_degrees)
    delta = args_cli.position_delta
    test_cases = [
        ("hold", np.zeros(6)),
        ("translate +x", np.array([delta, 0.0, 0.0, 0.0, 0.0, 0.0])),
        ("translate -x", np.array([-delta, 0.0, 0.0, 0.0, 0.0, 0.0])),
        ("translate +y", np.array([0.0, delta, 0.0, 0.0, 0.0, 0.0])),
        ("translate -y", np.array([0.0, -delta, 0.0, 0.0, 0.0, 0.0])),
        ("translate +z", np.array([0.0, 0.0, delta, 0.0, 0.0, 0.0])),
        ("translate -z", np.array([0.0, 0.0, -delta, 0.0, 0.0, 0.0])),
        ("rotate +x", np.array([0.0, 0.0, 0.0, angle, 0.0, 0.0])),
        ("rotate -x", np.array([0.0, 0.0, 0.0, -angle, 0.0, 0.0])),
        ("rotate +y", np.array([0.0, 0.0, 0.0, 0.0, angle, 0.0])),
        ("rotate -y", np.array([0.0, 0.0, 0.0, 0.0, -angle, 0.0])),
        ("rotate +z", np.array([0.0, 0.0, 0.0, 0.0, 0.0, angle])),
        ("rotate -z", np.array([0.0, 0.0, 0.0, 0.0, 0.0, -angle])),
    ]

    failures = 0
    results = state["results"]
    safety_reports = state["safety_reports"]
    state["stage"] = "capture_runtime_safety"
    arm_term = env.unwrapped.action_manager._terms["arm"]
    initial_capture = arm_term.safety_report()
    state["raw_capture"] = initial_capture
    print(
        "POLARIS_EEF_IK_SAFETY_CAPTURE="
        + json.dumps(initial_capture, sort_keys=True, allow_nan=False),
        flush=True,
    )
    state["stage"] = "validate_runtime_safety_capture"
    validated_initial_capture = validate_eef_runtime_safety(env)
    if validated_initial_capture != initial_capture:
        raise RuntimeError("Live EEF safety report changed during initial validation")

    for case_index, (label, pose_delta) in enumerate(test_cases):
        state["stage"] = "reset_case"
        state["case"] = label
        observation, _ = env.reset(expensive=False)
        validate_eef_gripper_post_reset(env, gripper_runtime_contract)
        begin_eef_safety_episode(env, case_index)
        state["stage"] = "validate_reset_frame"
        reset_frame_position_error, reset_frame_rotation_error = (
            validate_observation_frame(observation)
        )
        anchor_position = observation["policy"]["eef_pos"][0].detach().cpu().numpy()
        anchor_quaternion = observation["policy"]["eef_quat"][0].detach().cpu().numpy()
        # The first hold case closes the gripper long enough to prove the
        # EEF-only 5/120 driver-target slew reaches its unchanged endpoint.
        # All remaining pose cases keep the reset-default gripper open.
        gripper_open = 0.0 if case_index == 0 else 1.0
        lap_delta = np.concatenate([pose_delta, np.array([gripper_open])])[None, :]
        target = anchor_action_chunk(lap_delta, anchor_position, anchor_quaternion)[0]
        action = torch.as_tensor(target, device=env.device).reshape(1, -1)

        state["stage"] = "execute_case"
        for _ in range(args_cli.hold_steps):
            observation, _, terminated, truncated, _ = env.step(action, expensive=False)
            record_eef_gripper_post_policy_step(env)
            if bool(terminated[0]) or bool(truncated[0]):
                raise RuntimeError(f"Episode ended during smoke case {label!r}")

        actual_position = observation["policy"]["eef_pos"][0].detach().cpu().numpy()
        actual_quaternion = observation["policy"]["eef_quat"][0].detach().cpu().numpy()
        final_frame_position_error, final_frame_rotation_error = (
            validate_observation_frame(observation)
        )
        position_error = float(np.linalg.norm(actual_position - target[:3]))
        target_rotation = Rotation.from_quat(_wxyz_to_xyzw(target[3:7]))
        actual_rotation = Rotation.from_quat(_wxyz_to_xyzw(actual_quaternion))
        rotation_error = float((target_rotation.inv() * actual_rotation).magnitude())
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
        state["stage"] = "validate_case_safety"
        validate_eef_runtime_safety(env)
        safety_report = eef_episode_safety_report(env, case_index)
        expected_apply_calls = args_cli.hold_steps * 8
        if safety_report["counters"]["apply_calls"] != expected_apply_calls:
            raise RuntimeError(
                f"Smoke case {label!r} expected {expected_apply_calls} "
                "physics-substep apply calls, got "
                f"{safety_report['counters']['apply_calls']}"
            )
        safety_reports.append(safety_report)
        print(
            f"{label:>14}: {'PASS' if passed else 'FAIL'} "
            f"position_error={position_error * 1000:.2f}mm "
            f"rotation_error={math.degrees(rotation_error):.2f}deg"
        )

    # One bounded adversarial target proves that the guard activates while
    # preserving a finite simulator state. Never hold this target beyond
    # the one policy step; reset immediately after evidence capture.
    adversarial_index = len(test_cases)
    state["stage"] = "reset_adversarial_case"
    state["case"] = "oversized absolute +x target for one policy step"
    observation, _ = env.reset(expensive=False)
    validate_eef_gripper_post_reset(env, gripper_runtime_contract)
    begin_eef_safety_episode(env, adversarial_index)
    anchor_position = observation["policy"]["eef_pos"][0].detach().cpu().numpy()
    anchor_quaternion = observation["policy"]["eef_quat"][0].detach().cpu().numpy()
    oversized_delta = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    oversized_target = anchor_action_chunk(
        oversized_delta, anchor_position, anchor_quaternion
    )[0]
    state["stage"] = "execute_adversarial_case"
    adversarial_observation, _, terminated, truncated, _ = env.step(
        torch.as_tensor(oversized_target, device=env.device).reshape(1, -1),
        expensive=False,
    )
    record_eef_gripper_post_policy_step(env)
    robot = env.unwrapped.scene["robot"]
    joint_pos = robot.data.joint_pos[:, arm_term._joint_ids]
    joint_vel = robot.data.joint_vel[:, arm_term._joint_ids]
    eef_state_is_finite = bool(
        torch.isfinite(adversarial_observation["policy"]["eef_pos"]).all()
        and torch.isfinite(adversarial_observation["policy"]["eef_quat"]).all()
    )
    joint_state_is_finite = bool(
        torch.isfinite(joint_pos).all() and torch.isfinite(joint_vel).all()
    )
    captured_soft_limits = torch.as_tensor(
        initial_capture["soft_joint_pos_limits_rad"],
        dtype=joint_pos.dtype,
        device=joint_pos.device,
    ).unsqueeze(0)
    joint_soft_limit_violation = torch.maximum(
        torch.clamp(captured_soft_limits[..., 0] - joint_pos, min=0.0),
        torch.clamp(joint_pos - captured_soft_limits[..., 1], min=0.0),
    )
    joint_pos_within_soft_limits = bool(
        torch.isfinite(joint_soft_limit_violation).all()
        and (joint_soft_limit_violation <= CURRENT_JOINT_SOFT_LIMIT_TOLERANCE_RAD).all()
    )
    state_is_finite = eef_state_is_finite and joint_state_is_finite
    joint_state_evidence = {
        "joint_names": list(arm_term._joint_names),
        "joint_pos_rad": _strict_vector_evidence(joint_pos[0]),
        "joint_vel_rad_s": _strict_vector_evidence(joint_vel[0]),
        "soft_limit_violation_rad": _strict_vector_evidence(
            joint_soft_limit_violation[0]
        ),
        "soft_limit_tolerance_rad": CURRENT_JOINT_SOFT_LIMIT_TOLERANCE_RAD,
        "position_within_captured_soft_limits": (joint_pos_within_soft_limits),
    }
    state["stage"] = "validate_adversarial_case"
    adversarial_safety = eef_episode_safety_report(env, adversarial_index)
    json.dumps(adversarial_safety, allow_nan=False)
    counters = adversarial_safety["counters"]
    try:
        guard_evidence = validate_one_step_adversarial_report(adversarial_safety)
        guard_error = ""
    except ValueError as error:
        guard_evidence = None
        guard_error = str(error)
    adversarial_passed = (
        not bool(terminated[0])
        and not bool(truncated[0])
        and state_is_finite
        and joint_pos_within_soft_limits
        and guard_evidence is not None
    )
    adversarial_result = {
        "case": "oversized absolute +x target for one policy step",
        "passed": adversarial_passed,
        "state_is_finite": state_is_finite,
        "eef_state_is_finite": eef_state_is_finite,
        "joint_state_is_finite": joint_state_is_finite,
        "joint_pos_within_captured_soft_limits": (joint_pos_within_soft_limits),
        "joint_state": joint_state_evidence,
        "terminated": bool(terminated[0]),
        "truncated": bool(truncated[0]),
        "guard_evidence": guard_evidence,
        "guard_error": guard_error,
        "ik_safety": adversarial_safety,
    }
    state["adversarial_result"] = adversarial_result
    failures += int(not adversarial_passed)
    print(
        " adversarial: "
        f"{'PASS' if adversarial_passed else 'FAIL'} "
        f"apply_calls={counters['apply_calls']} "
        f"slew_events={counters['slew_limit_events']} "
        f"joint_finite={joint_state_is_finite} "
        f"joint_in_limits={joint_pos_within_soft_limits} "
        f"guard_error={guard_error or 'none'}",
        flush=True,
    )
    state["stage"] = "reset_after_adversarial_case"
    env.reset(expensive=False)
    validate_eef_gripper_post_reset(env, gripper_runtime_contract)

    total_checks = len(test_cases) + 1
    print(f"EEF pose smoke: {total_checks - failures}/{total_checks} passed")
    state["stage"] = "main_complete"
    state["case"] = None
    return int(failures > 0)


if __name__ == "__main__":
    state = {
        "stage": "launch_simulation_app",
        "case": None,
        "eef_frame": None,
        "raw_capture": None,
        "results": [],
        "safety_reports": [],
        "adversarial_result": None,
        "failure": None,
        "close_failures": [],
        "persistence_failures": [],
        "env": None,
    }
    exit_code = 1
    simulation_app = None
    try:
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app
        state["stage"] = "run_smoke"
        exit_code = main(state)
    except BaseException as run_error:
        state["failure"] = _exception_evidence(run_error)
        _print_exception(run_error)
        exit_code = 1

    env = state["env"]
    if state["failure"] is None:
        state["stage"] = "close_environment"
        state["case"] = None
    if env is not None:
        try:
            env.close()
        except BaseException as close_error:
            close_evidence = _exception_evidence(close_error)
            close_evidence["component"] = "environment"
            state["close_failures"].append(close_evidence)
            _print_exception(close_error)
            exit_code = 1

    if (
        simulation_app is not None
        and exit_code == 0
        and state["failure"] is None
        and not state["close_failures"]
    ):
        state["stage"] = "simulation_app_close_pending"
        state["case"] = None
    raw_published = False
    try:
        _atomic_write_strict_json(
            args_cli.output_json,
            _result_payload(state, finalized=False, exit_code=exit_code),
        )
        raw_published = True
    except BaseException as persistence_error:
        persistence_evidence = _exception_evidence(persistence_error)
        persistence_evidence["phase"] = "publish_immutable_raw"
        state["persistence_failures"].append(persistence_evidence)
        _print_exception(persistence_error)
        exit_code = 1

    raw_is_eligible = _raw_is_eligible_for_close(
        state,
        exit_code=exit_code,
        raw_published=raw_published,
        simulation_app=simulation_app,
    )
    raw_ready = False
    if raw_is_eligible:
        try:
            raw_stat = args_cli.output_json.stat()
            if raw_stat.st_mode & 0o777 != 0o444:
                raise RuntimeError("Immutable smoke raw JSON mode is not 0444")
            raw_bytes = args_cli.output_json.read_bytes()
            raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
            ready_marker = args_cli.output_json.with_name(
                args_cli.output_json.name + ".ready.json"
            )
            marker_payload = {
                "schema_version": 1,
                "stage": "simulation_app_close_pending",
                "raw_result": {
                    "path": str(args_cli.output_json),
                    "size_bytes": len(raw_bytes),
                    "sha256": raw_sha256,
                    "mode": "0444",
                },
            }
            marker_sha256 = hashlib.sha256(
                _strict_json_bytes(marker_payload)
            ).hexdigest()
            print(f"POLARIS_SMOKE_RAW_PREPARED={args_cli.output_json}", flush=True)
            print(f"POLARIS_SMOKE_RAW_SHA256={raw_sha256}", flush=True)
            print(f"POLARIS_SMOKE_READY_MARKER_PATH={ready_marker}", flush=True)
            print(
                f"POLARIS_SMOKE_READY_MARKER_EXPECTED_SHA256={marker_sha256}",
                flush=True,
            )
            sys.stdout.flush()
            sys.stderr.flush()
            raw_ready = True
            _atomic_write_strict_json(ready_marker, marker_payload)
            simulation_app.close()
        except BaseException as persistence_error:
            raw_ready = False
            _print_exception(persistence_error)
            exit_code = 1
            _best_effort_failure_log(
                "POLARIS_SMOKE_RAW_FAILURE=ready_marker_or_simulation_close_failed",
            )
    else:
        exit_code = 1
        _best_effort_failure_log(
            "POLARIS_SMOKE_RAW_FAILURE="
            f"stage={state['stage']},exit_code={exit_code},published={raw_published}",
        )

    # SimulationApp.close() hard-exits the pinned Isaac process.  It is called
    # immediately after the durable marker publication above, with no fallible
    # statement in between.  The host attests the untouched raw bytes only
    # after srun returns zero.
    if simulation_app is not None and not raw_ready:
        _best_effort_failure_log("POLARIS_SIMULATION_APP_CLOSE_SKIPPED=raw_not_ready")
        exit_code = 1

    if exit_code != 0 and simulation_app is not None:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except BaseException:
            pass
        finally:
            os._exit(1)
    sys.exit(exit_code)
