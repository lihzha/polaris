import copy

import numpy as np
import pytest

import polaris.joint_velocity_runtime as runtime
from polaris.joint_velocity_smoke import (
    SMOKE_PROFILE,
    build_joint_velocity_smoke_cases,
)
from polaris.pi05_droid_jointvelocity_contract import (
    PANDA_ARM_EFFORT_LIMITS,
    PANDA_ARM_JOINT_NAMES,
    PANDA_ARM_VELOCITY_LIMITS,
    PI05_DROID_ISAACLAB_SOURCE_SHA256,
    PI05_DROID_JOINTVELOCITY_PROFILE,
    PI05_DROID_POLARIS_RUNTIME_SOURCE_SHA256,
)


def _array_report(values):
    array = np.asarray(values, dtype=np.float32)
    return {
        "shape": list(array.shape),
        "dtype": "torch.float32",
        "device": "cuda:0",
        "values": array.tolist(),
    }


def make_joint_velocity_runtime_report():
    stiffness = np.zeros((1, 7), dtype=np.float32)
    damping = np.full((1, 7), 80.0, dtype=np.float32)
    effort = np.asarray([PANDA_ARM_EFFORT_LIMITS], dtype=np.float32)
    velocity = np.asarray([PANDA_ARM_VELOCITY_LIMITS], dtype=np.float32)
    drive = {
        "stiffness": _array_report(stiffness),
        "damping": _array_report(damping),
        "effort_limit": _array_report(effort),
        "velocity_limit": _array_report(velocity),
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
        "position_integration": "absent_by_exact_action_class",
        "velocity_drive": {
            "position_stiffness": 0.0,
            "velocity_damping": 80.0,
            "buffered": copy.deepcopy(drive),
            "direct_physx": copy.deepcopy(drive),
        },
        "gripper": {
            "action_class": (
                "polaris.environments.droid_cfg.BinaryJointPositionZeroToOneAction"
            ),
            "joint_name": "finger_joint",
            "threshold": "closed_if_gt_0p5_else_open",
            "open_command": _array_report(np.zeros((1, 1), dtype=np.float32)),
            "closed_command": _array_report(
                np.full((1, 1), np.pi / 4.0, dtype=np.float32)
            ),
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
        "runtime_contract": make_joint_velocity_runtime_report(),
        "cases": cases,
        "reset_probe": {
            "default_joint_position": [0.0] * 7,
            "joint_position": [0.0] * 7,
            "joint_velocity": [0.0] * 7,
            "joint_velocity_target": [0.0] * 7,
        },
        "lifecycle": {
            "env_close": "complete",
            "simulation_app_close": "complete",
            "capture_stage": "stdlib_parent_after_kit_child_exit",
        },
        "completion": {
            "child_exit_code": 0,
            "publication_stage": "stdlib_parent_after_child_exit",
            "child_capture_sha256": "a" * 64,
            "child_capture_size": 1,
            "child_capture_mode": "0400",
            "child_capture_path": "/tmp/test-child-close.json",
        },
    }


@pytest.fixture
def valid_joint_velocity_smoke_payload():
    return make_joint_velocity_smoke_payload()
