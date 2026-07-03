import copy

import numpy as np
import pytest

import polaris.joint_velocity_runtime as runtime
from polaris.joint_velocity_smoke import (
    SMOKE_PROFILE,
    build_joint_velocity_smoke_cases,
)
from polaris.pi05_droid_jointvelocity_contract import (
    NATIVE_GRIPPER_DAMPING,
    NATIVE_GRIPPER_DRIVE_PROFILE,
    NATIVE_GRIPPER_EFFORT_LIMIT,
    NATIVE_GRIPPER_PRECONDITION_STEPS,
    NATIVE_GRIPPER_STIFFNESS,
    NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S,
    PANDA_ARM_EFFORT_LIMITS,
    PANDA_ARM_JOINT_NAMES,
    PANDA_ARM_VELOCITY_LIMITS,
    PI05_DROID_ISAACLAB_SOURCE_SHA256,
    PI05_DROID_JOINTVELOCITY_PROFILE,
    PI05_DROID_POLARIS_RUNTIME_SOURCE_SHA256,
)


def _array_report(values, *, device="cuda:0"):
    array = np.asarray(values, dtype=np.float32)
    return {
        "shape": list(array.shape),
        "dtype": "torch.float32",
        "device": device,
        "values": array.tolist(),
    }


def make_joint_velocity_runtime_report():
    stiffness = np.zeros((1, 7), dtype=np.float32)
    damping = np.full((1, 7), 80.0, dtype=np.float32)
    effort = np.asarray([PANDA_ARM_EFFORT_LIMITS], dtype=np.float32)
    velocity = np.asarray([PANDA_ARM_VELOCITY_LIMITS], dtype=np.float32)
    buffered_drive = {
        "stiffness": _array_report(stiffness),
        "damping": _array_report(damping),
        "effort_limit": _array_report(effort),
        "velocity_limit": _array_report(velocity),
    }
    direct_drive = {
        name: {**copy.deepcopy(value), "device": "cpu"}
        for name, value in buffered_drive.items()
    }
    gripper_live = {
        "stiffness": _array_report([[NATIVE_GRIPPER_STIFFNESS]]),
        "damping": _array_report([[NATIVE_GRIPPER_DAMPING]]),
        "effort_limit": _array_report([[NATIVE_GRIPPER_EFFORT_LIMIT]]),
        "velocity_limit": _array_report([[NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S]]),
    }
    report = {
        "schema_version": 1,
        "profile": PI05_DROID_JOINTVELOCITY_PROFILE,
        "status": "pass",
        "isaaclab_version": "2.3.0",
        "isaaclab_source_sha256": dict(PI05_DROID_ISAACLAB_SOURCE_SHA256),
        "polaris_runtime_source_sha256": dict(PI05_DROID_POLARIS_RUNTIME_SOURCE_SHA256),
        "policy_frequency_hz": 15,
        "physics_frequency_hz": 120,
        "decimation": 8,
        "joint_names": list(PANDA_ARM_JOINT_NAMES),
        "action_term_class": (
            "isaaclab.envs.mdp.actions.joint_actions.JointVelocityAction"
        ),
        "action_cfg_class": (
            "isaaclab.envs.mdp.actions.actions_cfg.JointVelocityActionCfg"
        ),
        "scale": 1.0,
        "offset": 0.0,
        "use_default_offset": False,
        "clip": _array_report(
            np.broadcast_to(np.asarray([-1.0, 1.0], dtype=np.float32), (1, 7, 2))
        ),
        "action_buffers": {
            "raw_action": _array_report(np.zeros((1, 7), dtype=np.float32)),
            "processed_action": _array_report(np.zeros((1, 7), dtype=np.float32)),
        },
        "position_integration": "absent_by_exact_action_class",
        "velocity_drive": {
            "position_stiffness": 0.0,
            "velocity_damping": 80.0,
            "buffered": buffered_drive,
            "direct_physx": direct_drive,
        },
        "gripper": {
            "action_class": (
                "polaris.environments.droid_cfg.BinaryJointPositionZeroToOneAction"
            ),
            "joint_name": "finger_joint",
            "threshold": "closed_if_gt_0p5_else_open",
            "open_command": _array_report(np.zeros((1,), dtype=np.float32)),
            "closed_command": _array_report(
                np.full((1,), np.pi / 4.0, dtype=np.float32)
            ),
            "raw_action": _array_report(np.zeros((1, 1), dtype=np.float32)),
            "processed_action": _array_report(np.zeros((1, 1), dtype=np.float32)),
            "drive": {
                "profile": NATIVE_GRIPPER_DRIVE_PROFILE,
                "configured": {
                    "joint_names_expr": ["finger_joint"],
                    "stiffness": None,
                    "damping": None,
                    "effort_limit": NATIVE_GRIPPER_EFFORT_LIMIT,
                    "effort_limit_sim": NATIVE_GRIPPER_EFFORT_LIMIT,
                    "velocity_limit": NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S,
                    "velocity_limit_sim": NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S,
                },
                "actuator": {
                    **copy.deepcopy(gripper_live),
                    "effort_limit_sim": _array_report([[NATIVE_GRIPPER_EFFORT_LIMIT]]),
                    "velocity_limit_sim": _array_report(
                        [[NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S]]
                    ),
                },
                "direct_physx": {
                    name: {**copy.deepcopy(value), "device": "cpu"}
                    for name, value in gripper_live.items()
                },
            },
        },
    }
    report["runtime_sha256"] = runtime._canonical_sha256(report)
    return report


def make_joint_velocity_smoke_payload():
    command_magnitude = 0.25
    cases = []
    for case in build_joint_velocity_smoke_cases(command_magnitude):
        action = case["action"]
        expected_finger = float(np.float32(np.pi / 4.0)) if action[7] > 0.5 else 0.0
        if case["kind"] == "gripper":
            finger_before = float(np.float32(case["precondition_finger_target"]))
            finger_after = finger_before + 0.1 * case["expected_motion_sign"]
            finger_velocity_after = 1.0 * case["expected_motion_sign"]
        else:
            finger_before = 0.0
            finger_after = 0.0
            finger_velocity_after = 0.0
        cases.append(
            {
                **case,
                "joint_position_before": [0.0] * 7,
                "joint_velocity_before": [0.0] * 7,
                "joint_position_after": [0.01 * value for value in action[:7]],
                "joint_velocity_after": [0.5 * value for value in action[:7]],
                "processed_joint_velocity": action[:7],
                "articulation_joint_velocity_target": action[:7],
                "soft_joint_position_limits": [[-3.0, 3.0] for _ in range(7)],
                "finger_position_target": expected_finger,
                "processed_finger_position_target": expected_finger,
                "finger_position_before": finger_before,
                "finger_velocity_before": 0.0,
                "finger_position_after": finger_after,
                "finger_velocity_after": finger_velocity_after,
                "finger_average_slew_rad_s": ((finger_after - finger_before) * 15.0),
                "terminated": False,
                "truncated": False,
            }
        )
    return {
        "schema_version": 1,
        "smoke_profile": SMOKE_PROFILE,
        "controller_profile": PI05_DROID_JOINTVELOCITY_PROFILE,
        "environment": "DROID-FoodBussing",
        "command_magnitude": command_magnitude,
        "settle_steps": 5,
        "expected_gripper_drive_profile": NATIVE_GRIPPER_DRIVE_PROFILE,
        "gripper_precondition_steps": NATIVE_GRIPPER_PRECONDITION_STEPS,
        "runtime_contract": make_joint_velocity_runtime_report(),
        "cases": cases,
        "reset_probe": {
            "default_joint_position": [0.0] * 7,
            "joint_position": [0.0] * 7,
            "joint_velocity": [0.0] * 7,
            "joint_velocity_target": [0.0] * 7,
            "default_finger_position": 0.0,
            "finger_position": 0.0,
            "finger_velocity": 0.0,
            "finger_position_target": 0.0,
        },
        "lifecycle": {
            "env_close": "complete",
            "simulation_app_close": "invoked_then_child_exited_zero",
            "capture_stage": "stdlib_parent_after_kit_child_exit",
        },
        "completion": {
            "child_exit_code": 0,
            "publication_stage": "stdlib_parent_after_child_exit",
            "child_capture_sha256": "a" * 64,
            "child_capture_size": 1,
            "child_capture_mode": "0444",
            "child_capture_path": "/tmp/test-child-close.json",
            "child_ready_marker_sha256": "b" * 64,
            "child_ready_marker_size": 1,
            "child_ready_marker_mode": "0444",
            "child_ready_marker_path": "/tmp/test-child-close.json.ready.json",
        },
    }


@pytest.fixture
def valid_joint_velocity_smoke_payload():
    return make_joint_velocity_smoke_payload()
