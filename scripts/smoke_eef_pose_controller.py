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
from polaris.config import EEF_CONTROLLER_BASELINE_PROFILE
from polaris.config import EEF_CONTROLLER_PROFILES


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--environment", default="DROID-FoodBussing")
parser.add_argument(
    "--eef-controller-profile",
    choices=EEF_CONTROLLER_PROFILES,
    default=EEF_CONTROLLER_BASELINE_PROFILE,
)
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

DELAYED_CLOSE_REPLAY_PROFILE = "eef_open115_then_close5_same_arm_pose_v1"
DELAYED_CLOSE_OPEN_POLICY_STEPS = 115
DELAYED_CLOSE_CLOSE_POLICY_STEPS = 5
DELAYED_CLOSE_TRANSITION_SUBSTEPS = 38
DELAYED_CLOSE_LIMITED_APPLIES = DELAYED_CLOSE_TRANSITION_SUBSTEPS - 1
CLOSE_ARM_VELOCITY_HEADROOM_PROFILE = "arm_velocity_max_over_limit_le_0p95_v1"
CLOSE_ARM_VELOCITY_HEADROOM_MAX_RATIO = 0.95


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
    payload = {
        "schema_version": 2,
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
        "gripper_delayed_close_replay": state["delayed_close_result"],
        "gripper_close_velocity_headroom": state["close_headroom_result"],
        "terminal_failure_evidence": state["terminal_failure_evidence"],
        "failure": state["failure"],
        "close_failures": state["close_failures"],
        "persistence_failures": state["persistence_failures"],
    }
    if args_cli.eef_controller_profile != EEF_CONTROLLER_BASELINE_PROFILE:
        payload["eef_controller_profile"] = args_cli.eef_controller_profile
        payload["schema_version"] = 3
    if state.get("concurrent_close_result") is not None:
        payload["concurrent_arm_gripper_discriminator"] = state[
            "concurrent_close_result"
        ]
    return payload


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


def _capture_terminal_failure_evidence(state: dict[str, object]) -> dict[str, object]:
    """Best-effort live evidence captured before either environment is closed."""

    evidence = {
        "schema_version": 1,
        "profile": "eef_smoke_live_terminal_failure_capture_v1",
        "status": "unavailable",
        "stage": state.get("stage"),
        "case": state.get("case"),
        "episode_index": state.get("active_episode_index"),
        "safety_report": None,
        "current_joint_velocity_abort": None,
        "arm_joint_names": None,
        "arm_joint_velocity_rad_s": None,
        "all_six_gripper_state": None,
        "driver_target_slew": None,
        "capture_error": None,
    }
    env = state.get("env")
    episode_index = state.get("active_episode_index")
    reporter = state.get("episode_safety_reporter")
    if env is None or type(episode_index) is not int or not callable(reporter):
        return evidence
    try:
        safety_report = reporter(env, episode_index)
        runtime = getattr(env, "unwrapped", env)
        terms = runtime.action_manager._terms
        arm_term = terms["arm"]
        finger_term = terms["finger_joint"]
        robot = runtime.scene["robot"]
        target_slew_reporter = getattr(
            finger_term, "gripper_target_slew_dynamic_report", None
        )
        if not callable(target_slew_reporter):
            raise RuntimeError("Terminal finger target-slew reporter is unavailable")
        driver_target_slew = target_slew_reporter()
        gripper_dynamic = safety_report["gripper_runtime_dynamic"]
        if driver_target_slew != gripper_dynamic["driver_target_slew"]:
            raise RuntimeError("Terminal target-slew report changed during capture")

        arm_joint_names = list(arm_term._joint_names)
        arm_joint_velocity = _strict_vector_evidence(
            robot.data.joint_vel[:, arm_term._joint_ids][0]
        )
        if len(arm_joint_names) != 7 or len(arm_joint_velocity["values"]) != 7:
            raise RuntimeError("Terminal arm velocity evidence is not seven-joint")
        current_abort = safety_report["current_joint_velocity_abort"]
        if (
            current_abort is not None
            and current_abort["joint_velocity_rad_s"] != arm_joint_velocity["values"]
        ):
            raise RuntimeError("Terminal arm velocity/current-abort binding drift")

        gripper_joint_names = list(gripper_dynamic["joint_names"])
        gripper_joint_indices = list(gripper_dynamic["joint_indices"])
        if len(gripper_joint_names) != 6 or len(gripper_joint_indices) != 6:
            raise RuntimeError("Terminal gripper evidence is not all-six")
        all_six_gripper_state = {
            "joint_names": gripper_joint_names,
            "joint_indices": gripper_joint_indices,
            **{
                output_field: _strict_vector_evidence(
                    getattr(robot.data, source_field)[:, gripper_joint_indices][0]
                )
                for output_field, source_field in (
                    ("joint_position_rad", "joint_pos"),
                    ("joint_velocity_rad_s", "joint_vel"),
                    ("joint_acceleration_rad_s2", "joint_acc"),
                    ("joint_position_target_rad", "joint_pos_target"),
                    ("joint_velocity_target_rad_s", "joint_vel_target"),
                )
            },
        }
        if any(
            len(all_six_gripper_state[field]["values"]) != 6
            for field in (
                "joint_position_rad",
                "joint_velocity_rad_s",
                "joint_acceleration_rad_s2",
                "joint_position_target_rad",
                "joint_velocity_target_rad_s",
            )
        ):
            raise RuntimeError("Terminal all-six gripper vector width drift")
        evidence.update(
            {
                "status": "captured",
                "safety_report": safety_report,
                "current_joint_velocity_abort": current_abort,
                "arm_joint_names": arm_joint_names,
                "arm_joint_velocity_rad_s": arm_joint_velocity,
                "all_six_gripper_state": all_six_gripper_state,
                "driver_target_slew": driver_target_slew,
            }
        )
    except BaseException as capture_error:
        evidence["status"] = "capture_failed"
        evidence["capture_error"] = _exception_evidence(capture_error)
    return evidence


def _arm_velocity_headroom_evidence(
    safety_report: dict[str, object], *, episode_index: int
) -> dict[str, object]:
    maxima = list(safety_report["maxima"]["abs_joint_vel_rad_s"])
    limits = list(safety_report["joint_velocity_limits_rad_s"])
    joint_names = list(safety_report["joint_names"])
    if (
        safety_report["episode_index"] != episode_index
        or len(joint_names) != 7
        or len(maxima) != 7
        or len(limits) != 7
        or any(
            not math.isfinite(float(maximum)) or float(maximum) < 0.0
            for maximum in maxima
        )
        or any(
            not math.isfinite(float(limit)) or float(limit) <= 0.0 for limit in limits
        )
    ):
        raise RuntimeError("Close arm-velocity headroom input drift")
    ratios = [
        float(maximum) / float(limit)
        for maximum, limit in zip(maxima, limits, strict=True)
    ]
    maximum_ratio = max(ratios)
    return {
        "episode_index": episode_index,
        "joint_names": joint_names,
        "max_abs_joint_velocity_rad_s": maxima,
        "joint_velocity_limit_rad_s": limits,
        "velocity_to_limit_ratio": ratios,
        "maximum_ratio": maximum_ratio,
        "threshold_ratio": CLOSE_ARM_VELOCITY_HEADROOM_MAX_RATIO,
        "passed": maximum_ratio <= CLOSE_ARM_VELOCITY_HEADROOM_MAX_RATIO,
    }


def main(state: dict[str, object]) -> int:
    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab_tasks.utils import parse_env_cfg
    from scipy.spatial.transform import Rotation

    import polaris.environments  # noqa: F401
    from polaris.config import LAP_EEF_FRAME
    from polaris.config import EEF_CONTROLLER_CONCURRENT_ARM_GRIPPER_CANDIDATE_PROFILE
    from polaris.eef_controller_profile import configure_eef_controller_profile
    from polaris.eef_runtime_contract import validate_eef_runtime_frame
    from polaris.eef_runtime_contract import validate_eef_runtime_safety
    from polaris.eef_ik_safety import validate_one_step_adversarial_report
    from polaris.eef_ik_safety import CURRENT_JOINT_SOFT_LIMIT_TOLERANCE_RAD
    from polaris.eef_runtime_contract import begin_eef_safety_episode
    from polaris.eef_runtime_contract import eef_episode_safety_report
    from polaris.eef_gripper_runtime import install_eef_gripper_runtime
    from polaris.eef_gripper_runtime import GRIPPER_CLOSE_TRANSITION_APPLIES
    from polaris.eef_gripper_runtime import GRIPPER_CLOSE_TRANSITION_0P25_APPLIES
    from polaris.eef_gripper_runtime import record_eef_gripper_post_policy_step
    from polaris.eef_gripper_runtime import validate_eef_gripper_post_reset
    from polaris.environments.droid_cfg import EgoLapEefPoseActionCfg
    from polaris.environments.robot_cfg import configure_eef_pose_joint_safety
    from polaris.policy.lap_eef_pose_client import anchor_action_chunk

    state["eef_frame"] = LAP_EEF_FRAME
    record_concurrent_post_step = record_eef_gripper_post_policy_step
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
    controller_spec = configure_eef_controller_profile(
        env_cfg,
        profile=args_cli.eef_controller_profile,
    )
    target_slew_profile = controller_spec.target_slew_profile
    concurrent_v6 = (
        controller_spec.profile
        == EEF_CONTROLLER_CONCURRENT_ARM_GRIPPER_CANDIDATE_PROFILE
    )
    transition_substeps = (
        GRIPPER_CLOSE_TRANSITION_0P25_APPLIES
        if controller_spec.target_slew_rate_0p25_enabled
        else GRIPPER_CLOSE_TRANSITION_APPLIES
    )
    transition_policy_steps = math.ceil(transition_substeps / 8)

    def episode_safety_report(active_env, episode_index):
        return eef_episode_safety_report(
            active_env,
            episode_index,
            expected_gripper_target_slew_profile=target_slew_profile,
            expected_eef_controller_profile=controller_spec.profile,
        )

    def validate_runtime_safety(active_env):
        return validate_eef_runtime_safety(
            active_env,
            expected_gripper_target_slew_profile=target_slew_profile,
            expected_eef_controller_profile=controller_spec.profile,
        )

    state["episode_safety_reporter"] = episode_safety_report
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
    if controller_spec.profile == EEF_CONTROLLER_BASELINE_PROFILE:
        validated_initial_capture = validate_eef_runtime_safety(env)
    else:
        validated_initial_capture = validate_runtime_safety(env)
    if validated_initial_capture != initial_capture:
        raise RuntimeError("Live EEF safety report changed during initial validation")

    for case_index, (label, pose_delta) in enumerate(test_cases):
        state["stage"] = "reset_case"
        state["case"] = label
        state["active_episode_index"] = None
        observation, _ = env.reset(expensive=False)
        validate_eef_gripper_post_reset(env, gripper_runtime_contract)
        begin_eef_safety_episode(env, case_index)
        state["active_episode_index"] = case_index
        state["stage"] = "validate_reset_frame"
        reset_frame_position_error, reset_frame_rotation_error = (
            validate_observation_frame(observation)
        )
        anchor_position = observation["policy"]["eef_pos"][0].detach().cpu().numpy()
        anchor_quaternion = observation["policy"]["eef_quat"][0].detach().cpu().numpy()
        # The first hold case closes the gripper long enough to prove the
        # EEF-only 2.5/120 driver-target slew reaches its unchanged endpoint.
        # All remaining pose cases keep the reset-default gripper open.
        gripper_open = 0.0 if case_index == 0 else 1.0
        lap_delta = np.concatenate([pose_delta, np.array([gripper_open])])[None, :]
        target = anchor_action_chunk(lap_delta, anchor_position, anchor_quaternion)[0]
        action = torch.as_tensor(target, device=env.device).reshape(1, -1)

        state["stage"] = "execute_case"
        for _ in range(args_cli.hold_steps):
            observation, _, terminated, truncated, _ = env.step(action, expensive=False)
            record_concurrent_post_step(env)
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
        validate_runtime_safety(env)
        safety_report = episode_safety_report(env, case_index)
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

    # Replay the official failure boundary without changing the ordinary
    # 13-case contract: hold one reset-anchored arm pose open for policy steps
    # 0..114, then request the same arm pose with close for five policy steps.
    # At 2.5 rad/s and 120 Hz, the exact pi/4 endpoint is reached on close
    # substep 38, leaving two endpoint-hold substeps in the fifth policy step.
    delayed_close_index = len(test_cases)
    state["stage"] = "reset_delayed_close_replay"
    state["case"] = "open 115 policy steps then close at the same arm pose"
    state["active_episode_index"] = None
    observation, _ = env.reset(expensive=False)
    validate_eef_gripper_post_reset(env, gripper_runtime_contract)
    begin_eef_safety_episode(env, delayed_close_index)
    state["active_episode_index"] = delayed_close_index
    anchor_position = observation["policy"]["eef_pos"][0].detach().cpu().numpy()
    anchor_quaternion = observation["policy"]["eef_quat"][0].detach().cpu().numpy()
    zero_pose_delta = np.zeros(6)
    open_target = anchor_action_chunk(
        np.concatenate([zero_pose_delta, np.array([1.0])])[None, :],
        anchor_position,
        anchor_quaternion,
    )[0]
    close_target = anchor_action_chunk(
        np.concatenate([zero_pose_delta, np.array([0.0])])[None, :],
        anchor_position,
        anchor_quaternion,
    )[0]
    if not np.array_equal(open_target[:7], close_target[:7]):
        raise RuntimeError("Delayed-close replay changed the anchored arm target")
    delayed_terminated = False
    delayed_truncated = False
    for phase, action_target, policy_steps in (
        ("open", open_target, DELAYED_CLOSE_OPEN_POLICY_STEPS),
        ("close", close_target, transition_policy_steps),
    ):
        state["stage"] = f"execute_delayed_close_{phase}"
        action = torch.as_tensor(action_target, device=env.device).reshape(1, -1)
        for _ in range(policy_steps):
            _, _, terminated, truncated, _ = env.step(action, expensive=False)
            record_eef_gripper_post_policy_step(env)
            delayed_terminated = delayed_terminated or bool(terminated[0])
            delayed_truncated = delayed_truncated or bool(truncated[0])
            if delayed_terminated or delayed_truncated:
                raise RuntimeError(
                    f"Episode ended during delayed-close {phase!r} phase"
                )
    state["stage"] = "validate_delayed_close_replay"
    validate_runtime_safety(env)
    delayed_safety = episode_safety_report(env, delayed_close_index)
    delayed_counters = delayed_safety["counters"]
    delayed_target_slew = delayed_safety["gripper_runtime_dynamic"][
        "driver_target_slew"
    ]
    delayed_total_policy_steps = (
        DELAYED_CLOSE_OPEN_POLICY_STEPS + transition_policy_steps
    )
    delayed_apply_calls = delayed_total_policy_steps * 8
    delayed_abort_count = sum(
        delayed_counters[field]
        for field in (
            "current_joint_limit_aborts",
            "invariant_aborts",
            "nonfinite_aborts",
        )
    )
    closed_target_rad = delayed_safety["gripper_runtime_static"]["driver_target_slew"][
        "closed_target_rad"
    ]
    immediate_close_headroom = _arm_velocity_headroom_evidence(
        safety_reports[0], episode_index=0
    )
    delayed_close_headroom = _arm_velocity_headroom_evidence(
        delayed_safety, episode_index=delayed_close_index
    )
    close_headroom_passed = (
        immediate_close_headroom["passed"] and delayed_close_headroom["passed"]
    )
    state["close_headroom_result"] = {
        "profile": CLOSE_ARM_VELOCITY_HEADROOM_PROFILE,
        "threshold_ratio": CLOSE_ARM_VELOCITY_HEADROOM_MAX_RATIO,
        "passed": close_headroom_passed,
        "immediate_close_hold": immediate_close_headroom,
        "delayed_close_replay": delayed_close_headroom,
    }
    if concurrent_v6:
        state["close_headroom_result"]["completion_gate_applied"] = False
    delayed_passed = (
        not delayed_terminated
        and not delayed_truncated
        and delayed_safety["current_joint_velocity_abort"] is None
        and delayed_abort_count == 0
        and delayed_counters["apply_calls"] == delayed_apply_calls
        and delayed_counters["environment_substeps"] == delayed_apply_calls
        and delayed_target_slew["process_action_calls"] == delayed_total_policy_steps
        and delayed_target_slew["apply_calls"] == delayed_apply_calls
        and delayed_target_slew["endpoint_change_count"] == 1
        and delayed_target_slew["repeated_endpoint_process_count"]
        == delayed_total_policy_steps - 2
        and delayed_target_slew["slew_limited_apply_count"] == transition_substeps - 1
        and delayed_target_slew["endpoint_reached_apply_count"]
        == delayed_apply_calls - (transition_substeps - 1)
        and delayed_target_slew["last_requested_endpoint_rad"] == closed_target_rad
        and delayed_target_slew["last_applied_target_rad"] == closed_target_rad
        and (concurrent_v6 or delayed_close_headroom["passed"])
    )
    state["delayed_close_result"] = {
        "profile": DELAYED_CLOSE_REPLAY_PROFILE,
        "case": "open 115 policy steps then close at the same arm pose",
        "passed": delayed_passed,
        "episode_index": delayed_close_index,
        "open_policy_steps": DELAYED_CLOSE_OPEN_POLICY_STEPS,
        "close_policy_steps": transition_policy_steps,
        "close_transition_substeps": transition_substeps,
        "terminated": delayed_terminated,
        "truncated": delayed_truncated,
        "arm_abort_count": delayed_abort_count,
        "ik_safety": delayed_safety,
    }
    failures += int(not delayed_passed)
    if not concurrent_v6:
        failures += int(not immediate_close_headroom["passed"])
    print(
        " delayed close: "
        f"{'PASS' if delayed_passed else 'FAIL'} "
        f"process={delayed_target_slew['process_action_calls']} "
        f"apply={delayed_target_slew['apply_calls']} "
        f"limited={delayed_target_slew['slew_limited_apply_count']} "
        f"aborts={delayed_abort_count} "
        f"immediate_dq_ratio={immediate_close_headroom['maximum_ratio']:.6f} "
        f"delayed_dq_ratio={delayed_close_headroom['maximum_ratio']:.6f}",
        flush=True,
    )

    # V6 must prove that gripper transitions never own the arm target.  Move
    # through a distinct absolute EEF pose on every policy step while closing,
    # then continue moving while reopening so the contact/mimic impulse gate
    # observes the relevant endpoint.  Every physics apply must be accounted
    # as a fresh ordinary DLS/slew transaction.
    concurrent_discriminator_checks = 0
    if concurrent_v6:
        concurrent_index = len(test_cases) + 1
        state["stage"] = "reset_concurrent_arm_gripper_discriminator"
        state["case"] = "moving EEF through close and reopen transitions"
        state["active_episode_index"] = None
        observation, _ = env.reset(expensive=False)
        validate_eef_gripper_post_reset(env, gripper_runtime_contract)
        begin_eef_safety_episode(env, concurrent_index)
        state["active_episode_index"] = concurrent_index
        anchor_position = observation["policy"]["eef_pos"][0].detach().cpu().numpy()
        anchor_quaternion = observation["policy"]["eef_quat"][0].detach().cpu().numpy()
        moving_targets = []
        transition_terminated = False
        transition_truncated = False
        for phase, gripper_open in (("initial_open", 1.0),):
            target = anchor_action_chunk(
                np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_open]]),
                anchor_position,
                anchor_quaternion,
            )[0]
            moving_targets.append(target[:7].tolist())
            state["stage"] = f"execute_concurrent_{phase}"
            _, _, terminated, truncated, _ = env.step(
                torch.as_tensor(target, device=env.device).reshape(1, -1),
                expensive=False,
            )
            record_eef_gripper_post_policy_step(env)
            transition_terminated = transition_terminated or bool(terminated[0])
            transition_truncated = transition_truncated or bool(truncated[0])
        for phase, gripper_open in (("close", 0.0), ("reopen", 1.0)):
            for step in range(transition_policy_steps):
                fraction = (step + 1) / transition_policy_steps
                pose_delta = np.array(
                    [
                        0.02 * fraction,
                        (0.004 * fraction if phase == "reopen" else 0.0),
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        gripper_open,
                    ]
                )
                target = anchor_action_chunk(
                    pose_delta[None, :],
                    anchor_position,
                    anchor_quaternion,
                )[0]
                moving_targets.append(target[:7].tolist())
                state["stage"] = f"execute_concurrent_{phase}_moving"
                _, _, terminated, truncated, _ = env.step(
                    torch.as_tensor(target, device=env.device).reshape(1, -1),
                    expensive=False,
                )
                record_concurrent_post_step(env)
                transition_terminated = transition_terminated or bool(terminated[0])
                transition_truncated = transition_truncated or bool(truncated[0])
                if transition_terminated or transition_truncated:
                    raise RuntimeError(
                        f"Episode ended during concurrent {phase!r} discriminator"
                    )
        state["stage"] = "validate_concurrent_arm_gripper_discriminator"
        validate_runtime_safety(env)
        concurrent_safety = episode_safety_report(env, concurrent_index)
        concurrent_report = arm_term.controller_repair_candidate_report()
        concurrent_evidence = concurrent_report["concurrent_arm_gripper"]
        interlock = concurrent_report["gripper_close_arm_interlock"]
        interlock_counter_fields = (
            "remaining_substeps",
            "observed_endpoint_change_count",
            "activation_count",
            "active_apply_count",
            "released_apply_count",
            "anchor_capture_count",
            "anchor_target_apply_count",
            "anchor_first_exact_target_count",
            "anchor_refresh_count",
            "anchor_slew_limit_event_count",
            "anchor_slew_limited_joint_count",
            "anchor_position_limit_event_count",
            "anchor_position_limited_joint_count",
            "anchor_completion_count",
            "anchor_open_cancel_count",
        )
        telemetry = concurrent_safety["gripper_runtime_dynamic"][
            "open_endpoint_contact_mimic_impulse"
        ]
        concurrent_policy_steps = 1 + 2 * transition_policy_steps
        expected_apply_calls = concurrent_policy_steps * 8
        expected_closed_applies = transition_policy_steps * 8
        concurrent_passed = (
            not transition_terminated
            and not transition_truncated
            and concurrent_safety["counters"]["apply_calls"] == expected_apply_calls
            and concurrent_evidence["fresh_dls_target_applies"] == expected_apply_calls
            and concurrent_evidence["normal_target_setter_applies"]
            == expected_apply_calls
            and concurrent_evidence["closed_endpoint_fresh_dls_target_applies"]
            == expected_closed_applies
            and concurrent_evidence["closed_endpoint_distinct_desired_pose_count"]
            >= transition_policy_steps
            and concurrent_evidence["recovery_owned_target_applies"] == 0
            and concurrent_evidence["deferred_endpoint_transition_count"] == 0
            and concurrent_evidence["stored_target_replay_count"] == 0
            and interlock["enabled"] is False
            and interlock["configured_substeps"] == 0
            and interlock["endpoint_observed"] is False
            and interlock["anchor_valid"] is False
            and all(interlock[field] == 0 for field in interlock_counter_fields)
            and "arm_release_ramp" not in concurrent_report
            and telemetry["open_endpoint_samples"] > 0
            and telemetry["maximum_follower_diagnostic"] is not None
            and telemetry["passed"] is True
        )
        state["concurrent_close_result"] = {
            "profile": "moving_eef_close_reopen_fresh_dls_every_apply_v1",
            "passed": concurrent_passed,
            "episode_index": concurrent_index,
            "transition_substeps": transition_substeps,
            "transition_policy_steps": transition_policy_steps,
            "expected_apply_calls": expected_apply_calls,
            "expected_closed_endpoint_applies": expected_closed_applies,
            "distinct_policy_targets": moving_targets,
            "controller_report": concurrent_report,
            "open_endpoint_contact_mimic_impulse": telemetry,
            "ik_safety": concurrent_safety,
        }
        failures += int(not concurrent_passed)
        concurrent_discriminator_checks = 1
        print(
            " concurrent close/reopen: "
            f"{'PASS' if concurrent_passed else 'FAIL'} "
            f"fresh_dls={concurrent_evidence['fresh_dls_target_applies']} "
            f"closed_fresh={concurrent_evidence['closed_endpoint_fresh_dls_target_applies']} "
            f"distinct_closed={concurrent_evidence['closed_endpoint_distinct_desired_pose_count']} "
            f"open_samples={telemetry['open_endpoint_samples']} "
            f"coupled_failures={telemetry['coupled_impulse_failure_samples']}",
            flush=True,
        )

    # One bounded adversarial target proves that the guard activates while
    # preserving a finite simulator state. Never hold this target beyond
    # the one policy step; reset immediately after evidence capture.
    adversarial_index = len(test_cases) + 1 + concurrent_discriminator_checks
    state["stage"] = "reset_adversarial_case"
    state["case"] = "oversized absolute +x target for one policy step"
    state["active_episode_index"] = None
    observation, _ = env.reset(expensive=False)
    validate_eef_gripper_post_reset(env, gripper_runtime_contract)
    begin_eef_safety_episode(env, adversarial_index)
    state["active_episode_index"] = adversarial_index
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
    adversarial_safety = episode_safety_report(env, adversarial_index)
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
    state["active_episode_index"] = None

    total_checks = len(test_cases) + 2 + concurrent_discriminator_checks
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
        "delayed_close_result": None,
        "close_headroom_result": None,
        "concurrent_close_result": None,
        "terminal_failure_evidence": None,
        "active_episode_index": None,
        "episode_safety_reporter": None,
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
        # This is intentionally secondary evidence: capture failures are
        # recorded inside the payload and never replace the original error.
        try:
            state["terminal_failure_evidence"] = _capture_terminal_failure_evidence(
                state
            )
        except BaseException as capture_error:
            state["terminal_failure_evidence"] = {
                "schema_version": 1,
                "profile": "eef_smoke_live_terminal_failure_capture_v1",
                "status": "capture_failed",
                "stage": state.get("stage"),
                "case": state.get("case"),
                "episode_index": state.get("active_episode_index"),
                "safety_report": None,
                "current_joint_velocity_abort": None,
                "arm_joint_names": None,
                "arm_joint_velocity_rad_s": None,
                "all_six_gripper_state": None,
                "driver_target_slew": None,
                "capture_error": _exception_evidence(capture_error),
            }
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
