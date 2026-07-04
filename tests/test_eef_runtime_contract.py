import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import traceback
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from polaris.eef_runtime_contract import atomic_write_runtime_contract
from polaris.eef_runtime_contract import atomic_write_episode_safety
from polaris.eef_runtime_contract import _validate_wrist_energy_brake_history
from polaris.eef_runtime_contract import aggregate_episode_safety
from polaris.eef_runtime_contract import build_eef_controller_repair_candidate_aggregate
from polaris.eef_runtime_contract import build_terminal_rollout_evidence
from polaris.eef_runtime_contract import EEF_RUNTIME_CONTRACT_SCHEMA_VERSION
from polaris.eef_runtime_contract import EEF_SAFETY_SIDECAR_SCHEMA_VERSION
from polaris.eef_runtime_contract import (
    EEF_RUNTIME_CONTRACT_VELOCITY_RECOVERY_SCHEMA_VERSION,
)
from polaris.eef_runtime_contract import (
    EEF_SAFETY_SIDECAR_VELOCITY_RECOVERY_SCHEMA_VERSION,
)
from polaris.eef_runtime_contract import eef_episode_safety_report
from polaris.eef_runtime_contract import load_episode_safety_sidecars
from polaris.eef_runtime_contract import reconcile_episode_safety_transactions
from polaris.eef_runtime_contract import validate_episode_safety_cadence
from polaris.eef_runtime_contract import validate_eef_runtime_frame
from polaris.eef_runtime_contract import validate_eef_runtime_safety
from polaris.eef_runtime_contract import validate_ego_lap_runtime_protocol
from polaris import eef_runtime_contract as runtime_contract_module
from polaris.config import EEF_CONTROLLER_BASELINE_PROFILE
from polaris.config import EEF_CONTROLLER_MIMIC_COMPLIANCE_CANDIDATE_PROFILE
from polaris.config import EEF_CONTROLLER_RELEASE_RAMP_CANDIDATE_PROFILE
from polaris.config import EEF_CONTROLLER_VELOCITY_RECOVERY_CANDIDATE_PROFILE
from polaris.eef_controller_profile import eef_controller_profile
from polaris.eef_controller_repair import ARM_RELEASE_RAMP_FORMULA_PROFILE
from polaris.eef_controller_repair import ARM_RELEASE_RAMP_FRACTION_PROFILE
from polaris.eef_controller_repair import ARM_RELEASE_RAMP_PROFILE
from polaris.eef_controller_repair import ARM_RELEASE_RAMP_STATE_PROFILE
from polaris.eef_controller_repair import ARM_RELEASE_RAMP_SUBSTEPS
from polaris.eef_controller_repair import ARM_RELEASE_RAMP_TRANSACTION_PROFILE
from polaris.eef_controller_repair import arm_release_ramp_fraction
from polaris.eef_gripper_failure_trace import EEF_ALL_SIX_GRIPPER_TRACE_PROFILE
from polaris.eef_gripper_runtime import (
    EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE,
)
from polaris.eef_ik_safety import EEF_IK_APPLY_CADENCE
from polaris.eef_ik_safety import EEF_IK_SAFETY_PROFILE
from polaris.eef_ik_safety import EEF_IK_WRIST_ENERGY_BRAKE_CANDIDATE_PROFILE
from polaris.eef_ik_safety import (
    EEF_IK_CURRENT_VELOCITY_RECOVERY_CANDIDATE_PROFILE,
)
from polaris.eef_ik_safety import current_joint_velocity_recovery_envelope
from polaris.eef_ik_safety import CURRENT_JOINT_VELOCITY_RECOVERY_CLEAN_SAMPLES_REQUIRED
from polaris.eef_ik_safety import (
    CURRENT_JOINT_VELOCITY_RECOVERY_ENVELOPE_FORMULA_PROFILE,
)
from polaris.eef_ik_safety import CURRENT_JOINT_VELOCITY_RECOVERY_HOLD_PROFILE
from polaris.eef_ik_safety import (
    CURRENT_JOINT_VELOCITY_RECOVERY_MAXIMUM_ACTIVE_SUBSTEPS,
)
from polaris.eef_ik_safety import (
    CURRENT_JOINT_VELOCITY_RECOVERY_PREDICTED_POSITION_PROFILE,
)
from polaris.eef_ik_safety import CURRENT_JOINT_VELOCITY_RECOVERY_PROFILE
from polaris.eef_ik_safety import (
    CURRENT_JOINT_VELOCITY_RECOVERY_RELATIVE_ENVELOPE_FLOAT32,
)
from polaris.eef_ik_safety import CURRENT_JOINT_VELOCITY_RECOVERY_SCHEMA_VERSION
from polaris.eef_ik_safety import CURRENT_JOINT_VELOCITY_RECOVERY_TRANSACTION_PROFILE
from polaris.eef_ik_safety import EEF_QUATERNION_UNIT_NORM_TOLERANCE
from polaris.eef_ik_safety import ARM_VELOCITY_TARGET_PROFILE
from polaris.eef_ik_safety import ARTICULATION_SOLVER_PROFILE
from polaris.eef_ik_safety import ARTICULATION_SOLVER_READBACK
from polaris.eef_ik_safety import CURRENT_JOINT_VELOCITY_ABORT_EVIDENCE_PROFILE
from polaris.eef_ik_safety import CURRENT_JOINT_VELOCITY_RECOVERY_ABORT_MESSAGES
from polaris.eef_ik_safety import current_joint_velocity_abort_evidence_sha256
from polaris.eef_ik_safety import format_current_joint_velocity_abort_message
from polaris.eef_ik_safety import JOINT_SLEW_FLOAT32_TOLERANCE_RAD
from polaris.eef_ik_safety import JOINT_VELOCITY_LIMIT_TOLERANCE_RAD_S
from polaris.eef_ik_safety import PANDA_EEF_JOINT_EFFORT_LIMITS
from polaris.eef_ik_safety import PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S
from polaris.eef_ik_safety import PANDA_EEF_PHYSICS_DT_FLOAT32
from polaris.eef_ik_safety import PANDA_EEF_SOLVER_POSITION_ITERATION_COUNT
from polaris.eef_ik_safety import PANDA_EEF_SOLVER_VELOCITY_ITERATION_COUNT
from polaris.eef_ik_safety import PANDA_EEF_PHYSX_SOLVER_TYPE
from polaris.eef_ik_safety import (
    PANDA_PHYSX_DERIVED_SOFT_JOINT_POS_LIMITS_FLOAT32_SHA256,
)
from polaris.eef_ik_safety import PANDA_PHYSX_DERIVED_SOFT_JOINT_POS_LIMITS_RAD
from polaris.eef_ik_safety import PANDA_PHYSX_HARD_JOINT_POS_LIMITS_FLOAT32_SHA256
from polaris.eef_ik_safety import PANDA_PHYSX_HARD_JOINT_POS_LIMITS_RAD
from polaris.eef_ik_safety import PANDA_SOFT_JOINT_POS_LIMITS_FLOAT32_SHA256
from polaris.eef_ik_safety import PANDA_SOFT_JOINT_POS_LIMITS_RAD
from polaris.eef_ik_safety import PANDA_TARGET_JOINT_POS_LIMITS_FLOAT32_SHA256
from polaris.eef_ik_safety import PHYSX_DERIVED_SOFT_LIMIT_PROFILE
from polaris.eef_ik_safety import PHYSX_HARD_LIMIT_PROFILE
from polaris.eef_ik_safety import TARGET_SOFT_LIMIT_GUARD_BAND_PROFILE
from polaris.eef_ik_safety import WRIST_ENERGY_BRAKE_JOINT_NAMES
from polaris.eef_ik_safety import WRIST_ENERGY_BRAKE_LATCH_SUBSTEPS
from polaris.eef_ik_safety import WRIST_ENERGY_BRAKE_PROFILE
from polaris.eef_ik_safety import WRIST_ENERGY_BRAKE_TARGET_SHIFT_FRACTION
from polaris.eef_ik_safety import validate_one_step_adversarial_report
from polaris.gripper_semantics import GRIPPER_THRESHOLD_PROFILE
from polaris.eval_artifacts import build_episode_artifact_identity
from polaris.eval_artifacts import EGO_LAP_ENVIRONMENT_RUNTIME_PROFILE
from polaris.eval_artifacts import EGO_LAP_TRACE_PROFILE
from polaris.eval_artifacts import EGO_LAP_TRACE_SCHEMA_VERSION
from polaris.eval_artifacts import empty_eval_results
from polaris.eval_artifacts import TRACE_QUERY_FIELDS


def _wxyz(rotation: Rotation) -> np.ndarray:
    return rotation.as_quat()[[3, 0, 1, 2]]


def _runtime_fixture(*, wrist_energy_brake=False):
    link0_rotation = Rotation.from_euler("z", 20, degrees=True)
    relative_rotation = Rotation.from_euler("xyz", [10, -5, 30], degrees=True)
    link8_rotation = link0_rotation * relative_rotation
    link0_position = np.array([0.1, -0.2, 0.3])
    relative_position = np.array([0.4, 0.05, 0.2])
    link8_position = link0_position + link0_rotation.apply(relative_position)
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_names=["panda_link0", "panda_link8"],
            body_pos_w=np.array([[link0_position, link8_position]]),
            body_quat_w=np.array([[_wxyz(link0_rotation), _wxyz(link8_rotation)]]),
        )
    )
    offset = SimpleNamespace(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0))
    controller = SimpleNamespace(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.01},
    )
    arm_term = SimpleNamespace(
        cfg=SimpleNamespace(
            body_name="panda_link8",
            body_offset=offset,
            controller=controller,
            scale=1.0,
            enable_wrist_energy_brake=wrist_energy_brake,
        ),
        action_dim=7,
        _body_idx=1,
        _joint_names=[f"panda_joint{index}" for index in range(1, 8)],
    )
    max_delta = [
        np.float32(np.float32(value) * np.float32(1.0 / 120.0)).item()
        for value in PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S
    ]
    soft_limits = [list(values) for values in PANDA_SOFT_JOINT_POS_LIMITS_RAD]
    soft_limits_f32 = np.asarray(soft_limits, dtype=np.float32)
    margin_f32 = np.asarray(max_delta, dtype=np.float32)
    target_limits = np.stack(
        (
            soft_limits_f32[:, 0] + margin_f32,
            soft_limits_f32[:, 1] - margin_f32,
        ),
        axis=-1,
    ).tolist()
    target_limit_sha256 = hashlib.sha256(
        np.asarray(target_limits, dtype="<f4").tobytes()
    ).hexdigest()
    soft_limit_sha256 = PANDA_SOFT_JOINT_POS_LIMITS_FLOAT32_SHA256
    arm_term.safety_report = lambda: {
        "episode_index": None,
        "profile": EEF_IK_SAFETY_PROFILE,
        "apply_actions_cadence": EEF_IK_APPLY_CADENCE,
        "physics_dt": 1.0 / 120.0,
        "control_dt": 1.0 / 15.0,
        "decimation": 8,
        "current_joint_soft_limit_tolerance_rad": 1e-5,
        "target_soft_limit_guard_band_profile": TARGET_SOFT_LIMIT_GUARD_BAND_PROFILE,
        "physx_hard_limit_profile": PHYSX_HARD_LIMIT_PROFILE,
        "physx_derived_soft_limit_profile": PHYSX_DERIVED_SOFT_LIMIT_PROFILE,
        "physx_hard_limit_write_count": 1,
        "arm_velocity_target_profile": ARM_VELOCITY_TARGET_PROFILE,
        "articulation_solver_profile": ARTICULATION_SOLVER_PROFILE,
        "articulation_solver_readback": ARTICULATION_SOLVER_READBACK,
        "physx_solver_type": PANDA_EEF_PHYSX_SOLVER_TYPE,
        "solver_position_iteration_count": (PANDA_EEF_SOLVER_POSITION_ITERATION_COUNT),
        "solver_velocity_iteration_count": (PANDA_EEF_SOLVER_VELOCITY_ITERATION_COUNT),
        "joint_velocity_limit_tolerance_rad_s": JOINT_VELOCITY_LIMIT_TOLERANCE_RAD_S,
        "eef_quaternion_unit_norm_tolerance": EEF_QUATERNION_UNIT_NORM_TOLERANCE,
        "joint_slew_float32_tolerance_rad": JOINT_SLEW_FLOAT32_TOLERANCE_RAD,
        "soft_joint_pos_limit_factor": 1.0,
        "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
        "joint_velocity_limits_rad_s": list(PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S),
        "joint_effort_limits": list(PANDA_EEF_JOINT_EFFORT_LIMITS),
        "max_delta_joint_pos_rad": list(max_delta),
        "target_soft_limit_margin_rad": list(max_delta),
        "target_joint_pos_limits_rad": target_limits,
        "target_joint_pos_limits_float32_sha256": target_limit_sha256,
        "physx_hard_joint_pos_limits_rad": json.loads(json.dumps(target_limits)),
        "physx_hard_joint_pos_limits_float32_sha256": target_limit_sha256,
        "physx_derived_soft_joint_pos_limits_rad": [
            list(pair) for pair in PANDA_PHYSX_DERIVED_SOFT_JOINT_POS_LIMITS_RAD
        ],
        "physx_derived_soft_joint_pos_limits_float32_sha256": (
            PANDA_PHYSX_DERIVED_SOFT_JOINT_POS_LIMITS_FLOAT32_SHA256
        ),
        "arm_velocity_target_rad_s": [0.0] * 7,
        "soft_joint_pos_limits_rad": soft_limits,
        "soft_joint_pos_limits_float32_sha256": soft_limit_sha256,
        "counters": {
            "apply_calls": 0,
            "environment_substeps": 0,
            "slew_limit_events": 0,
            "slew_limited_joints": 0,
            "position_limit_events": 0,
            "position_limited_joints": 0,
            "post_clamp_target_violations": 0,
            "current_joint_limit_aborts": 0,
            "invariant_aborts": 0,
            "nonfinite_aborts": 0,
            "dls_fallbacks": 0,
            "guard_diagnostics_dropped": 0,
        },
        "maxima": {
            "raw_delta_joint_pos_rad": [0.0] * 7,
            "applied_delta_joint_pos_rad": [0.0] * 7,
            "raw_target_soft_limit_violation_rad": [0.0] * 7,
            "post_clamp_target_soft_limit_violation_rad": [0.0] * 7,
            "post_clamp_target_guard_band_violation_rad": [0.0] * 7,
            "current_joint_soft_limit_violation_rad": [0.0] * 7,
            "current_physx_hard_limit_violation_rad": [0.0] * 7,
            "abs_joint_vel_rad_s": [0.0] * 7,
            "minimum_outer_joint_clearance_rad": [0.0] * 7,
        },
        "guard_diagnostics": [],
        "max_raw_delta_diagnostic": None,
        "current_joint_velocity_abort": None,
    }
    if wrist_energy_brake:
        base_reporter = arm_term.safety_report

        def candidate_report():
            report = base_reporter()
            report["profile"] = EEF_IK_WRIST_ENERGY_BRAKE_CANDIDATE_PROFILE
            report.update(
                {
                    "wrist_energy_brake_profile": WRIST_ENERGY_BRAKE_PROFILE,
                    "wrist_energy_brake_joint_names": list(
                        WRIST_ENERGY_BRAKE_JOINT_NAMES
                    ),
                    "wrist_energy_brake_latch_substeps": (
                        WRIST_ENERGY_BRAKE_LATCH_SUBSTEPS
                    ),
                    "wrist_energy_brake_target_shift_fraction": (
                        WRIST_ENERGY_BRAKE_TARGET_SHIFT_FRACTION
                    ),
                    "wrist_energy_brake_target_shift_threshold_rad": [
                        np.float32(
                            np.float32(max_delta[index])
                            * np.float32(WRIST_ENERGY_BRAKE_TARGET_SHIFT_FRACTION)
                        ).item()
                        for index in range(4, 7)
                    ],
                    "wrist_energy_brake_latch_remaining_substeps": [0],
                    "wrist_energy_brake_diagnostics": [],
                }
            )
            report["counters"].update(
                {
                    "wrist_energy_brake_trigger_events": 0,
                    "wrist_energy_brake_active_substeps": 0,
                    "wrist_energy_brake_attempted_joint_targets": 0,
                    "wrist_energy_brake_braked_joint_targets": 0,
                    "wrist_energy_brake_diagnostics_dropped": 0,
                }
            )
            return report

        arm_term.safety_report = candidate_report
    finger_term = SimpleNamespace(gripper_threshold_profile=GRIPPER_THRESHOLD_PROFILE)
    runtime = SimpleNamespace(
        max_episode_length=451,
        step_dt=1.0 / 15.0,
        physics_dt=1.0 / 120.0,
        cfg=SimpleNamespace(sim=SimpleNamespace(dt=1.0 / 120.0), decimation=8),
        scene={"robot": robot},
        action_manager=SimpleNamespace(
            _terms={"arm": arm_term, "finger_joint": finger_term}
        ),
    )
    env = SimpleNamespace(unwrapped=runtime, max_episode_length=451)
    observation = {
        "policy": {
            "eef_pos": relative_position[None, :],
            "eef_quat": _wxyz(relative_rotation)[None, :],
        }
    }
    return env, observation


def _episode_result(*, episode=0, length=2, numerical_failure=False):
    return {
        "episode": episode,
        "episode_length": length,
        "success": False,
        "progress": 0.0 if numerical_failure else 0.25,
        "numerical_failure": numerical_failure,
        "numerical_failure_reason": (
            "DifferentialIKNumericalError: guard" if numerical_failure else ""
        ),
    }


def _episode_safety(
    *,
    episode=0,
    length=2,
    numerical_failure=False,
    wrist_energy_brake=False,
    failure_substeps=3,
):
    env, _ = _runtime_fixture(wrist_energy_brake=wrist_energy_brake)
    report = env.unwrapped.action_manager._terms["arm"].safety_report()
    report["episode_index"] = episode
    apply_calls = (
        length * 8 if not numerical_failure else (length - 1) * 8 + failure_substeps
    )
    report["counters"]["apply_calls"] = apply_calls
    report["counters"]["environment_substeps"] = apply_calls
    zero_vector = {
        "values": [0.0] * 7,
        "finite_mask": [True] * 7,
        "finite_count": 7,
    }
    neutral_joint_pos = [
        (lower + upper) / 2.0 for lower, upper in report["target_joint_pos_limits_rad"]
    ]
    neutral_vector = {
        "values": neutral_joint_pos,
        "finite_mask": [True] * 7,
        "finite_count": 7,
    }
    report["max_raw_delta_diagnostic"] = {
        "kind": "max_raw_delta",
        "episode_index": episode,
        "policy_step": 0,
        "physics_substep": 0,
        "joint_pos_rad": neutral_vector,
        "raw_delta_joint_pos_rad": zero_vector,
        "raw_joint_pos_target_rad": neutral_vector,
        "safe_joint_pos_target_rad": neutral_vector,
        "pose_error_norm": 0.0,
        "jacobian_finite": True,
        "jacobian_max_abs": 0.0,
        "eef_quaternion_norm": None,
    }
    if numerical_failure:
        report["counters"]["nonfinite_aborts"] = 1
        report["guard_diagnostics"] = [
            {
                "kind": "nonfinite_abort",
                "episode_index": episode,
                "policy_step": length - 1,
                "physics_substep": failure_substeps - 1,
                "joint_pos_rad": None,
                "raw_delta_joint_pos_rad": None,
                "raw_joint_pos_target_rad": None,
                "safe_joint_pos_target_rad": None,
                "pose_error_norm": None,
                "jacobian_finite": None,
                "jacobian_max_abs": None,
                "eef_quaternion_norm": None,
            }
        ]
    return report


def _baseline_controller_report(safety: dict) -> dict:
    spec = eef_controller_profile(EEF_CONTROLLER_BASELINE_PROFILE)
    physical = list(safety["max_delta_joint_pos_rad"])
    return {
        "arm_slew_headroom": {
            "enabled": False,
            "profile": "panda_nominal_target_slew_0p95_physical_limit_v1",
            "ratio": 0.95,
            "physical_max_delta_joint_pos_rad": physical,
            "nominal_max_delta_joint_pos_rad": physical,
        },
        "gripper_close_arm_interlock": {
            "enabled": False,
            "profile": spec.close_interlock_profile,
            "configured_substeps": spec.close_interlock_substeps,
            "remaining_substeps": 0,
            "observed_endpoint_change_count": 0,
            "endpoint_observed": False,
            "activation_count": 0,
            "active_apply_count": 0,
            "anchor_valid": False,
            "anchor_capture_count": 0,
            "anchor_target_apply_count": 0,
            "anchor_first_exact_target_count": 0,
            "anchor_refresh_count": 0,
            "anchor_slew_limit_event_count": 0,
            "anchor_slew_limited_joint_count": 0,
            "anchor_position_limit_event_count": 0,
            "anchor_position_limited_joint_count": 0,
            "anchor_completion_count": 0,
            "anchor_open_cancel_count": 0,
            "last_activation_apply_index": None,
            "last_anchor_joint_pos_rad": None,
            "last_anchor_little_endian_float32_sha256": None,
            "max_abs_current_anchor_residual_rad": [0.0] * 7,
            "max_abs_target_anchor_residual_rad": [0.0] * 7,
            "max_abs_active_delta_joint_pos_rad": [0.0] * 7,
            "released_apply_count": 0,
            "max_abs_released_delta_joint_pos_rad": [0.0] * 7,
        },
    }


def _controller_aggregate(ik_safety: dict) -> dict:
    return {
        "profile": "polaris_eef_controller_repair_candidate_episode_aggregate_v1",
        "eef_controller_profile": EEF_CONTROLLER_BASELINE_PROFILE,
        "initial": _baseline_controller_report(ik_safety),
        "episodes": [
            {
                "episode_index": episode["episode_index"],
                "report": _baseline_controller_report(ik_safety),
            }
            for episode in ik_safety.get("episodes", [])
        ],
    }


def _empty_velocity_recovery_report() -> dict:
    limits = np.asarray(
        PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S,
        dtype=np.float32,
    ).tolist()
    envelopes = [current_joint_velocity_recovery_envelope(value) for value in limits]
    hard_limits = [list(row) for row in PANDA_PHYSX_HARD_JOINT_POS_LIMITS_RAD]
    return {
        "contract": {
            "schema_version": CURRENT_JOINT_VELOCITY_RECOVERY_SCHEMA_VERSION,
            "profile": CURRENT_JOINT_VELOCITY_RECOVERY_PROFILE,
            "envelope_formula_profile": (
                CURRENT_JOINT_VELOCITY_RECOVERY_ENVELOPE_FORMULA_PROFILE
            ),
            "relative_envelope_float32": (
                CURRENT_JOINT_VELOCITY_RECOVERY_RELATIVE_ENVELOPE_FLOAT32
            ),
            "maximum_active_substeps": (
                CURRENT_JOINT_VELOCITY_RECOVERY_MAXIMUM_ACTIVE_SUBSTEPS
            ),
            "clean_samples_required": (
                CURRENT_JOINT_VELOCITY_RECOVERY_CLEAN_SAMPLES_REQUIRED
            ),
            "hold_profile": CURRENT_JOINT_VELOCITY_RECOVERY_HOLD_PROFILE,
            "predicted_position_profile": (
                CURRENT_JOINT_VELOCITY_RECOVERY_PREDICTED_POSITION_PROFILE
            ),
            "hard_limit_profile": PHYSX_HARD_LIMIT_PROFILE,
            "release_ramp_profile": ARM_RELEASE_RAMP_PROFILE,
            "transaction_profile": CURRENT_JOINT_VELOCITY_RECOVERY_TRANSACTION_PROFILE,
            "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
            "velocity_limits_rad_s": limits,
            "velocity_envelopes_rad_s": envelopes,
            "physics_dt_float32": PANDA_EEF_PHYSICS_DT_FLOAT32,
            "hard_joint_position_limits_rad": hard_limits,
            "hard_joint_position_limits_little_endian_float32_sha256": (
                PANDA_PHYSX_HARD_JOINT_POS_LIMITS_FLOAT32_SHA256
            ),
        },
        "state": {
            "phase": "inactive",
            "active": False,
            "consecutive_active_substeps": 0,
            "consecutive_clean_samples": 0,
            "release_ramp_next_index": None,
        },
        "counters": {
            "residual_events": 0,
            "residual_joints": 0,
            "recovery_events": 0,
            "recovery_active_substeps": 0,
            "recovered_events": 0,
            "hold_target_applies": 0,
            "release_ramp_target_applies": 0,
            "sustained_aborts": 0,
            "current_hard_limit_aborts": 0,
            "predicted_limit_aborts": 0,
            "transaction_aborts": 0,
            "lower_endpoint_transition_aborts": 0,
        },
        "maxima": {
            "abs_velocity_to_limit_ratio": 0.0,
            "consecutive_recovery_substeps": 0,
            "abs_velocity_residual_excess_rad_s": [0.0] * 7,
        },
        "events": [],
    }


def _active_velocity_recovery_report(*, apply_index: int) -> dict:
    report = _empty_velocity_recovery_report()
    limits = np.asarray(report["contract"]["velocity_limits_rad_s"], dtype=np.float32)
    velocity = np.asarray([11.743, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    excess = np.maximum(np.abs(velocity) - limits, np.float32(0.0))
    ratio = (np.abs(velocity) / limits).astype(np.float32)
    position = np.asarray(
        [0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.0],
        dtype=np.float32,
    )
    predicted = (position + velocity * np.float32(PANDA_EEF_PHYSICS_DT_FLOAT32)).astype(
        np.float32
    )
    hard_limits = np.asarray(
        report["contract"]["hard_joint_position_limits_rad"],
        dtype=np.float32,
    )
    clearance = np.minimum(
        predicted - hard_limits[:, 0],
        hard_limits[:, 1] - predicted,
    ).astype(np.float32)
    snapshot = {
        "apply_index": apply_index,
        "policy_step": apply_index // 8,
        "physics_substep": apply_index % 8,
        "joint_pos_rad": position.tolist(),
        "joint_velocity_rad_s": velocity.tolist(),
        "joint_velocity_limit_rad_s": limits.tolist(),
        "joint_velocity_envelope_rad_s": report["contract"]["velocity_envelopes_rad_s"],
        "joint_velocity_limit_excess_rad_s": excess.tolist(),
        "velocity_to_limit_ratio": ratio.tolist(),
        "predicted_joint_pos_rad": predicted.tolist(),
        "predicted_hard_limit_clearance_rad": clearance.tolist(),
        "hold_target_rad": position.tolist(),
        "hold_position_target_readback_rad": position.tolist(),
        "hold_velocity_target_readback_rad_s": [0.0] * 7,
        "hold_effort_target_readback_nm": [0.0] * 7,
    }
    report["state"] = {
        "phase": "hold",
        "active": True,
        "consecutive_active_substeps": 1,
        "consecutive_clean_samples": 0,
        "release_ramp_next_index": None,
    }
    report["counters"].update(
        {
            "residual_events": 1,
            "residual_joints": 1,
            "recovery_events": 1,
            "recovery_active_substeps": 1,
            "hold_target_applies": 1,
        }
    )
    report["maxima"].update(
        {
            "abs_velocity_to_limit_ratio": float(ratio.max()),
            "consecutive_recovery_substeps": 1,
            "abs_velocity_residual_excess_rad_s": excess.tolist(),
        }
    )
    report["events"] = [
        {
            "event_index": 0,
            "start_apply_index": apply_index,
            "end_apply_index": None,
            "start_reason": "measured_velocity_above_float32_envelope",
            "end_reason": None,
            "deferred_lower_endpoint_transition_count": None,
            "lower_endpoint_transition_overflow_context": None,
            "recovery_completed_apply_index": None,
            "start": copy.deepcopy(snapshot),
            "last": copy.deepcopy(snapshot),
        }
    ]
    return report


def _predicted_terminal_velocity_recovery_report(*, apply_index: int) -> dict:
    report = _active_velocity_recovery_report(apply_index=apply_index - 1)
    terminal = copy.deepcopy(report["events"][0]["last"])
    velocity = np.asarray(terminal["joint_velocity_rad_s"], dtype=np.float32)
    hard_limits = np.asarray(
        report["contract"]["hard_joint_position_limits_rad"],
        dtype=np.float32,
    )
    delta = (velocity * np.float32(PANDA_EEF_PHYSICS_DT_FLOAT32)).astype(np.float32)
    position = np.asarray(
        [0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.0],
        dtype=np.float32,
    )
    position[0] = np.float32(hard_limits[0, 1] - np.float32(delta[0] / 2.0))
    predicted = (position + delta).astype(np.float32)
    clearance = np.minimum(
        predicted - hard_limits[:, 0],
        hard_limits[:, 1] - predicted,
    ).astype(np.float32)
    terminal.update(
        {
            "apply_index": apply_index,
            "policy_step": apply_index // 8,
            "physics_substep": apply_index % 8,
            "joint_pos_rad": position.tolist(),
            "predicted_joint_pos_rad": predicted.tolist(),
            "predicted_hard_limit_clearance_rad": clearance.tolist(),
            "hold_target_rad": None,
            "hold_position_target_readback_rad": None,
            "hold_velocity_target_readback_rad_s": None,
            "hold_effort_target_readback_nm": None,
        }
    )
    report["state"] = {
        "phase": "inactive",
        "active": False,
        "consecutive_active_substeps": 0,
        "consecutive_clean_samples": 0,
        "release_ramp_next_index": None,
    }
    report["counters"]["residual_events"] = 2
    report["counters"]["residual_joints"] = 2
    report["counters"]["predicted_limit_aborts"] = 1
    report["events"][0].update(
        {
            "end_apply_index": apply_index,
            "end_reason": "predicted_hard_limit_abort",
            "last": terminal,
        }
    )
    return report


def _lower_endpoint_terminal_velocity_recovery_report(*, apply_index: int) -> dict:
    report = _active_velocity_recovery_report(apply_index=apply_index - 1)
    terminal = copy.deepcopy(report["events"][0]["last"])
    terminal.update(
        {
            "apply_index": apply_index,
            "policy_step": apply_index // 8,
            "physics_substep": apply_index % 8,
            "hold_target_rad": None,
            "hold_position_target_readback_rad": None,
            "hold_velocity_target_readback_rad_s": None,
            "hold_effort_target_readback_nm": None,
        }
    )
    report["state"] = {
        "phase": "inactive",
        "active": False,
        "consecutive_active_substeps": 0,
        "consecutive_clean_samples": 0,
        "release_ramp_next_index": None,
    }
    report["counters"]["residual_events"] = 2
    report["counters"]["residual_joints"] = 2
    report["counters"]["lower_endpoint_transition_aborts"] = 1
    report["events"][0].update(
        {
            "end_apply_index": apply_index,
            "end_reason": "lower_endpoint_transition_overflow_abort",
            "deferred_lower_endpoint_transition_count": 2,
            "lower_endpoint_transition_overflow_context": "active_recovery",
            "last": terminal,
        }
    )
    return report


def _post_recovery_lower_endpoint_terminal_velocity_recovery_report() -> dict:
    report = _active_velocity_recovery_report(apply_index=0)
    terminal = copy.deepcopy(report["events"][0]["last"])
    position = np.asarray(
        [0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.0],
        dtype=np.float32,
    )
    hard_limits = np.asarray(
        report["contract"]["hard_joint_position_limits_rad"],
        dtype=np.float32,
    )
    terminal.update(
        {
            "apply_index": 18,
            "policy_step": 2,
            "physics_substep": 2,
            "joint_pos_rad": position.tolist(),
            "joint_velocity_rad_s": [0.0] * 7,
            "joint_velocity_limit_excess_rad_s": [0.0] * 7,
            "velocity_to_limit_ratio": [0.0] * 7,
            "predicted_joint_pos_rad": position.tolist(),
            "predicted_hard_limit_clearance_rad": np.minimum(
                position - hard_limits[:, 0],
                hard_limits[:, 1] - position,
            )
            .astype(np.float32)
            .tolist(),
            "hold_target_rad": None,
            "hold_position_target_readback_rad": None,
            "hold_velocity_target_readback_rad_s": None,
            "hold_effort_target_readback_nm": None,
        }
    )
    report["state"] = {
        "phase": "inactive",
        "active": False,
        "consecutive_active_substeps": 0,
        "consecutive_clean_samples": 0,
        "release_ramp_next_index": None,
    }
    report["counters"].update(
        {
            "recovery_active_substeps": 3,
            "recovered_events": 1,
            "hold_target_applies": 3,
            "release_ramp_target_applies": 16,
            "lower_endpoint_transition_aborts": 1,
        }
    )
    report["maxima"]["consecutive_recovery_substeps"] = 3
    report["events"][0].update(
        {
            "end_apply_index": 18,
            "end_reason": "lower_endpoint_transition_overflow_abort",
            "deferred_lower_endpoint_transition_count": 2,
            "lower_endpoint_transition_overflow_context": "post_recovery_resume",
            "recovery_completed_apply_index": 17,
            "last": terminal,
        }
    )
    return report


def _hard_limit_terminal_snapshot(
    report: dict,
    *,
    kind: str,
    apply_index: int,
    committed: bool,
) -> dict:
    limits = np.asarray(report["contract"]["velocity_limits_rad_s"], dtype=np.float32)
    hard_limits = np.asarray(
        report["contract"]["hard_joint_position_limits_rad"],
        dtype=np.float32,
    )
    position = np.asarray(
        [0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.0],
        dtype=np.float32,
    )
    velocity = np.zeros(7, dtype=np.float32)
    if kind == "current":
        position[0] = np.nextafter(hard_limits[0, 1], np.float32(np.inf))
    elif kind == "predicted":
        velocity[0] = np.float32(2.0)
        delta = np.float32(velocity[0] * np.float32(PANDA_EEF_PHYSICS_DT_FLOAT32))
        position[0] = np.float32(hard_limits[0, 1] - np.float32(delta / 2.0))
    else:
        raise AssertionError(f"unknown hard-limit fixture kind: {kind!r}")
    predicted = (position + velocity * np.float32(PANDA_EEF_PHYSICS_DT_FLOAT32)).astype(
        np.float32
    )
    clearance = np.minimum(
        predicted - hard_limits[:, 0],
        hard_limits[:, 1] - predicted,
    ).astype(np.float32)
    hold_target = position.tolist() if committed else None
    return {
        "apply_index": apply_index,
        "policy_step": apply_index // 8,
        "physics_substep": apply_index % 8,
        "joint_pos_rad": position.tolist(),
        "joint_velocity_rad_s": velocity.tolist(),
        "joint_velocity_limit_rad_s": limits.tolist(),
        "joint_velocity_envelope_rad_s": report["contract"]["velocity_envelopes_rad_s"],
        "joint_velocity_limit_excess_rad_s": np.maximum(
            np.abs(velocity) - limits,
            np.float32(0.0),
        )
        .astype(np.float32)
        .tolist(),
        "velocity_to_limit_ratio": (np.abs(velocity) / limits)
        .astype(np.float32)
        .tolist(),
        "predicted_joint_pos_rad": predicted.tolist(),
        "predicted_hard_limit_clearance_rad": clearance.tolist(),
        "hold_target_rad": hold_target,
        "hold_position_target_readback_rad": hold_target,
        "hold_velocity_target_readback_rad_s": [0.0] * 7 if committed else None,
        "hold_effort_target_readback_nm": [0.0] * 7 if committed else None,
    }


def _hard_limit_collision_velocity_recovery_report(
    *,
    kind: str,
    post_recovery: bool,
) -> dict:
    report = _active_velocity_recovery_report(apply_index=0 if post_recovery else 9)
    end_reason = {
        "current": "current_hard_limit_abort",
        "predicted": "predicted_hard_limit_abort",
    }[kind]
    terminal_apply = 18 if post_recovery else 10
    terminal = _hard_limit_terminal_snapshot(
        report,
        kind=kind,
        apply_index=terminal_apply,
        committed=False,
    )
    report["state"] = {
        "phase": "inactive",
        "active": False,
        "consecutive_active_substeps": 0,
        "consecutive_clean_samples": 0,
        "release_ramp_next_index": None,
    }
    abort_counter = (
        "current_hard_limit_aborts" if kind == "current" else "predicted_limit_aborts"
    )
    report["counters"][abort_counter] = 1
    if not post_recovery:
        report["events"][0].update(
            {
                "end_apply_index": terminal_apply,
                "end_reason": end_reason,
                "deferred_lower_endpoint_transition_count": 2,
                "lower_endpoint_transition_overflow_context": "active_recovery",
                "last": terminal,
            }
        )
        return report

    clean = _hard_limit_terminal_snapshot(
        report,
        kind="predicted",
        apply_index=17,
        committed=True,
    )
    # The clean completion owns a neutral state, not the predicted fixture's
    # near-boundary position/velocity.
    clean_position = np.asarray(
        [0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.0],
        dtype=np.float32,
    )
    hard_limits = np.asarray(
        report["contract"]["hard_joint_position_limits_rad"],
        dtype=np.float32,
    )
    clean.update(
        {
            "joint_pos_rad": clean_position.tolist(),
            "joint_velocity_rad_s": [0.0] * 7,
            "joint_velocity_limit_excess_rad_s": [0.0] * 7,
            "velocity_to_limit_ratio": [0.0] * 7,
            "predicted_joint_pos_rad": clean_position.tolist(),
            "predicted_hard_limit_clearance_rad": np.minimum(
                clean_position - hard_limits[:, 0],
                hard_limits[:, 1] - clean_position,
            )
            .astype(np.float32)
            .tolist(),
            "hold_target_rad": clean_position.tolist(),
            "hold_position_target_readback_rad": clean_position.tolist(),
        }
    )
    report["events"][0].update(
        {
            "end_apply_index": 17,
            "end_reason": "clean2_release_ramp_complete",
            "recovery_completed_apply_index": 17,
            "last": clean,
        }
    )
    report["counters"].update(
        {
            "recovery_events": 2,
            "recovery_active_substeps": 3,
            "recovered_events": 1,
            "hold_target_applies": 3,
            "release_ramp_target_applies": 16,
        }
    )
    report["maxima"]["consecutive_recovery_substeps"] = 3
    owner_start_reason = {
        "current": "current_hard_limit_violation",
        "predicted": "predicted_hard_limit_crossing",
    }[kind]
    report["events"].append(
        {
            "event_index": 1,
            "start_apply_index": 18,
            "end_apply_index": 18,
            "start_reason": owner_start_reason,
            "end_reason": end_reason,
            "deferred_lower_endpoint_transition_count": 2,
            "lower_endpoint_transition_overflow_context": "post_recovery_resume",
            "recovery_completed_apply_index": 17,
            "start": copy.deepcopy(terminal),
            "last": copy.deepcopy(terminal),
        }
    )
    return report


def _candidate_controller_report(
    safety: dict,
    *,
    initial: bool,
    profile: str = EEF_CONTROLLER_MIMIC_COMPLIANCE_CANDIDATE_PROFILE,
) -> dict:
    spec = eef_controller_profile(profile)
    physical = np.asarray(safety["max_delta_joint_pos_rad"], dtype=np.float32).tolist()
    nominal = np.multiply(
        np.asarray(physical, dtype=np.float32),
        np.float32(0.95),
        dtype=np.float32,
    ).tolist()
    report = {
        "arm_slew_headroom": {
            "enabled": True,
            "profile": "panda_nominal_target_slew_0p95_physical_limit_v1",
            "ratio": 0.95,
            "physical_max_delta_joint_pos_rad": physical,
            "nominal_max_delta_joint_pos_rad": nominal,
        },
        "gripper_close_arm_interlock": {
            "enabled": True,
            "profile": spec.close_interlock_profile,
            "configured_substeps": spec.close_interlock_substeps,
            "remaining_substeps": 0,
            "observed_endpoint_change_count": 0,
            "endpoint_observed": not initial,
            "activation_count": 0,
            "active_apply_count": 0,
            "anchor_valid": False,
            "anchor_capture_count": 0,
            "anchor_target_apply_count": 0,
            "anchor_first_exact_target_count": 0,
            "anchor_refresh_count": 0,
            "anchor_slew_limit_event_count": 0,
            "anchor_slew_limited_joint_count": 0,
            "anchor_position_limit_event_count": 0,
            "anchor_position_limited_joint_count": 0,
            "anchor_completion_count": 0,
            "anchor_open_cancel_count": 0,
            "last_activation_apply_index": None,
            "last_anchor_joint_pos_rad": None,
            "last_anchor_little_endian_float32_sha256": None,
            "max_abs_current_anchor_residual_rad": [0.0] * 7,
            "max_abs_target_anchor_residual_rad": [0.0] * 7,
            "max_abs_active_delta_joint_pos_rad": [0.0] * 7,
            "released_apply_count": 0,
            "max_abs_released_delta_joint_pos_rad": [0.0] * 7,
        },
    }
    if spec.arm_release_ramp_enabled:
        report["arm_release_ramp"] = {
            "enabled": True,
            "profile": ARM_RELEASE_RAMP_PROFILE,
            "state_profile": ARM_RELEASE_RAMP_STATE_PROFILE,
            "substeps": ARM_RELEASE_RAMP_SUBSTEPS,
            "fraction_profile": ARM_RELEASE_RAMP_FRACTION_PROFILE,
            "fractions_float32": [
                arm_release_ramp_fraction(index)
                for index in range(ARM_RELEASE_RAMP_SUBSTEPS)
            ],
            "formula_profile": ARM_RELEASE_RAMP_FORMULA_PROFILE,
            "transaction_profile": ARM_RELEASE_RAMP_TRANSACTION_PROFILE,
            "open_during_ramp_policy": (
                "continue_current_ramp_without_restart_or_skip_v1"
            ),
            "phase": "release",
            "next_index": None,
            "release_observed_count": 0,
            "ramp_started_count": 0,
            "ramp_completed_count": 0,
            "ramp_cancelled_by_reactivation_count": 0,
            "ramp_target_apply_count": 0,
            "cancelled_ramp_target_apply_count": 0,
            "ramp_limited_target_apply_count": 0,
            "ramp_limited_joint_target_count": 0,
            "last_target_apply_index": None,
            "last_ramp_index": None,
            "max_abs_nominal_to_ramped_target_change_rad": [0.0] * 7,
            "gripper_target_or_state_write_count": 0,
        }
    if spec.current_joint_velocity_recovery_enabled:
        report["current_joint_velocity_recovery"] = _empty_velocity_recovery_report()
    return report


def _all_six_trace(*, episode: int, length: int, apply_calls: int) -> dict:
    snapshot = {
        "joint_pos_rad": [0.0] * 6,
        "joint_vel_rad_s": [0.0] * 6,
        "joint_acc_rad_s2": [0.0] * 6,
        "joint_pos_target_rad": [0.0] * 6,
        "joint_vel_target_rad_s": [0.0] * 6,
        "joint_effort_target_nm": [0.0] * 6,
    }
    first = max(apply_calls - 64, 0)
    entries = [
        {
            "apply_index": apply_index,
            "policy_step": apply_index // 8,
            "physics_substep": apply_index % 8,
            "raw_action": 0.0,
            "requested_endpoint_rad": 0.0,
            "pre": copy.deepcopy(snapshot),
            "target_after_setter_rad": 0.0,
            "post": copy.deepcopy(snapshot),
        }
        for apply_index in range(first, apply_calls)
    ]
    return {
        "schema_version": 1,
        "profile": EEF_ALL_SIX_GRIPPER_TRACE_PROFILE,
        "episode_index": episode,
        "capacity": 64,
        "decimation": 8,
        "joint_names": [
            "finger_joint",
            "right_outer_knuckle_joint",
            "left_inner_finger_joint",
            "right_inner_finger_joint",
            "left_inner_finger_knuckle_joint",
            "right_inner_finger_knuckle_joint",
        ],
        "joint_indices": [7, 8, 9, 10, 11, 12],
        "process_action_calls": length,
        "total_apply_entries": apply_calls,
        "dropped_entries": first,
        "initial_snapshot": copy.deepcopy(snapshot),
        "entries": entries,
        "terminal_snapshot": copy.deepcopy(snapshot),
        "numerical_failure": False,
    }


def _attach_candidate_stub_gripper(safety: dict, *, length: int) -> dict:
    apply_calls = safety["counters"]["apply_calls"]
    safety["gripper_runtime_static"] = {
        "profile": "candidate-static-stub",
        "driver_target_slew": {
            "profile": EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
        },
        "mimic_compliance": {
            "profile": (
                "robotiq_2f85_live_physx_mimic_frequency100_damping1p2_candidate_v1"
            ),
            "enabled": True,
        },
    }
    safety["gripper_runtime_dynamic"] = {
        "apply_entry_samples": apply_calls,
        "post_policy_step_samples": length,
        "max_abs_joint_velocity_rad_s": [0.0] * 6,
        "max_abs_joint_acceleration_rad_s2": [0.0] * 6,
        "nonfinite_samples": 0,
        "driver_target_slew": {
            "profile": EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE,
            "process_action_calls": length,
            "apply_calls": apply_calls,
            "endpoint_change_count": 0,
        },
    }
    return safety


def _attach_stub_gripper_runtime(
    safety,
    *,
    post_policy_step_samples,
    target_process_calls,
    target_apply_calls,
):
    apply_calls = safety["counters"]["apply_calls"]
    safety["gripper_runtime_static"] = {
        "profile": "stub-static",
        "driver_target_slew": {"profile": "stub-target-slew"},
    }
    safety["gripper_runtime_dynamic"] = {
        "apply_entry_samples": apply_calls,
        "post_policy_step_samples": post_policy_step_samples,
        "nonfinite_samples": 0,
        "driver_target_slew": {
            "process_action_calls": target_process_calls,
            "apply_calls": target_apply_calls,
        },
    }
    return safety


def _candidate_trigger_runtime():
    env, observation = _runtime_fixture(wrist_energy_brake=True)
    report = env.unwrapped.action_manager._terms["arm"].safety_report()
    report["episode_index"] = 0
    report["counters"].update(
        {
            "apply_calls": 2,
            "environment_substeps": 2,
            "wrist_energy_brake_trigger_events": 1,
            "wrist_energy_brake_active_substeps": 1,
            "wrist_energy_brake_attempted_joint_targets": 1,
            "wrist_energy_brake_braked_joint_targets": 1,
        }
    )
    joint_pos = [0.0] * 7
    joint_vel = [0.0] * 7
    joint_vel[4] = 0.1
    previous_target = [0.0] * 7
    previous_target[4] = -0.02
    nominal_target = [0.0] * 7
    nominal_target[4] = 0.02
    applied_target = list(nominal_target)
    applied_target[4] = 0.0
    report["wrist_energy_brake_latch_remaining_substeps"] = [1]
    report["wrist_energy_brake_diagnostics"] = [
        {
            "episode_index": 0,
            "apply_index": 1,
            "policy_step": 0,
            "physics_substep": 1,
            "environment_index": 0,
            "reversal_detection_armed": True,
            "trigger_joint_mask": [True, False, False],
            "attempted_joint_mask": [True, False, False],
            "braked_joint_mask": [True, False, False],
            "joint_pos_rad": joint_pos,
            "joint_vel_rad_s": joint_vel,
            "previous_applied_target_rad": previous_target,
            "nominal_safe_target_rad": nominal_target,
            "applied_target_rad": applied_target,
            "target_shift_rad": [0.04, 0.0, 0.0],
        }
    ]
    env.unwrapped.action_manager._terms["arm"].safety_report = lambda: report
    return env, observation, report


def _candidate_two_substep_runtime():
    env, observation, report = _candidate_trigger_runtime()
    first = report["wrist_energy_brake_diagnostics"][0]
    follow_up = copy.deepcopy(first)
    follow_up.update(
        {
            "apply_index": 2,
            "policy_step": 0,
            "physics_substep": 2,
            "reversal_detection_armed": False,
            "trigger_joint_mask": [False, False, False],
            "previous_applied_target_rad": list(first["applied_target_rad"]),
            "target_shift_rad": [0.02, 0.0, 0.0],
        }
    )
    report["counters"].update(
        {
            "apply_calls": 3,
            "environment_substeps": 3,
            "wrist_energy_brake_active_substeps": 2,
            "wrist_energy_brake_attempted_joint_targets": 2,
            "wrist_energy_brake_braked_joint_targets": 2,
        }
    )
    report["wrist_energy_brake_latch_remaining_substeps"] = [0]
    report["wrist_energy_brake_diagnostics"].append(follow_up)
    return env, observation, report


def _environment_state(step: int) -> dict:
    return {
        "profile": "isaaclab_single_env_episode_sim_common_camera_counters_v1",
        "live_max_episode_length": 451,
        "episode_length": step,
        "sim_step_counter": 8 * step,
        "common_step_counter": step,
        "sensor_frame_counters": {
            "external_cam": step,
            "wrist_cam": step,
        },
    }


def _terminal_rollout(result: dict, *, failure_substeps: int = 3) -> dict:
    completed = result["episode_length"] - int(result["numerical_failure"])
    environment_after = _environment_state(completed)
    if result["numerical_failure"]:
        # The fixture's failed action aborts on its third attempted substep.
        environment_after["sim_step_counter"] += failure_substeps
    return build_terminal_rollout_evidence(
        episode_result=result,
        environment_before=_environment_state(0),
        environment_after=environment_after,
        terminated_false_count=completed,
        truncated_false_count=completed,
    )


def _current_velocity_abort_safety(
    *, episode: int = 0, length: int = 118
) -> tuple[dict, dict]:
    safety = _episode_safety(
        episode=episode,
        length=length,
        numerical_failure=True,
        failure_substeps=4,
    )
    safety["counters"]["nonfinite_aborts"] = 0
    safety["counters"]["invariant_aborts"] = 1
    safety["guard_diagnostics"][0]["kind"] = "current_joint_velocity_limit_abort"
    limits = np.asarray(safety["joint_velocity_limits_rad_s"], dtype=np.float32)
    velocity = np.asarray([0.1, -0.2, 0.3, -0.4, 2.75, 0.5, -3.0], dtype=np.float32)
    excess = np.maximum(np.abs(velocity) - limits, np.float32(0.0))
    mask = np.abs(velocity) > (
        limits + np.float32(JOINT_VELOCITY_LIMIT_TOLERANCE_RAD_S)
    )
    safety["maxima"]["abs_joint_vel_rad_s"] = np.abs(velocity).tolist()
    safety["current_joint_velocity_abort"] = {
        "profile": CURRENT_JOINT_VELOCITY_ABORT_EVIDENCE_PROFILE,
        "episode_index": episode,
        "policy_step": length - 1,
        "physics_substep": 3,
        "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
        "joint_velocity_rad_s": velocity.tolist(),
        "joint_velocity_limit_rad_s": limits.tolist(),
        "joint_velocity_limit_tolerance_rad_s": (JOINT_VELOCITY_LIMIT_TOLERANCE_RAD_S),
        "joint_velocity_limit_excess_rad_s": excess.tolist(),
        "exceeded_joint_mask": mask.tolist(),
    }
    result = _episode_result(
        episode=episode,
        length=length,
        numerical_failure=True,
    )
    result["numerical_failure_reason"] = (
        "DifferentialIKInvariantError: "
        f"{format_current_joint_velocity_abort_message(safety['current_joint_velocity_abort'])}"
    )
    return safety, result


def _artifact_identity_for_terminal(result: dict, terminal: dict) -> dict:
    return {
        "video": {
            "filename": f"episode_{result['episode']}.mp4",
            "size_bytes": 1,
            "sha256": "0" * 64,
            "frame_count": result["episode_length"],
            "height": 224,
            "width": 448,
        },
        "terminal_trace": {
            "filename": f"episode_{result['episode']:06d}.jsonl",
            "size_bytes": 1,
            "sha256": "1" * 64,
            "schema_version": 2,
            "trace_profile": EGO_LAP_TRACE_PROFILE,
            "episode_result": result,
            "terminal_rollout": terminal,
        },
    }


def _trace_common(event: str, episode: int) -> dict:
    return {
        "schema_version": EGO_LAP_TRACE_SCHEMA_VERSION,
        "trace_profile": EGO_LAP_TRACE_PROFILE,
        "timestamp": 1.0,
        "event": event,
        "episode": episode,
    }


def _trace_query(episode: int, query_index: int) -> dict:
    zero_chunk = [[0.0] * 7 for _ in range(16)]
    anchored_chunk = [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] for _ in range(16)]
    record = {field: None for field in TRACE_QUERY_FIELDS}
    record.update(_trace_common("query", episode))
    record.update(
        {
            "query": query_index,
            "step": query_index * 8,
            "instruction": "perform the task",
            "checkpoint_profile": "original_lap_public_3b_v1",
            "checkpoint_path": "/checkpoints/LAP-3B",
            "contract_sha256": "0" * 64,
            "policy_type": "flow",
            "response_semantics": "cumulative_delta_targets",
            "execution_horizon": 8,
            "ar_endpoint_interpolation_profile": None,
            "ar_endpoint_interpolation_steps": None,
            "gripper_execution_profile": "binary_model_open_gt_0p5_else_closed_v1",
            "gripper_threshold": 0.5,
            "action_sampler_profile": "flow_explicit_euler_t1_to_t0_v1",
            "flow_num_steps": 10,
            "initial_rng_seed": 0,
            "ar_max_decoding_steps": None,
            "ar_temperature": None,
            "ar_stop_at_eos": None,
            "frame_description": "robot base frame",
            "eef_frame": "panda_link8",
            "numeric_action_frame": "robot_base",
            "normalization_scope": "category",
            "normalization_stats_sha256": "1" * 64,
            "normalization_profile": "q99_train_matched_v1",
            "normalization_compute_dtype": "float32",
            "normalization_input_formula": "q99_input_eps1e-8_clip_zero0_v1",
            "normalization_output_formula": (
                "q99_output_eps1e-8_zeroq01_extrapolate_v1"
            ),
            "normalization_formula_probe_sha256": "2" * 64,
            "state_layout": "xyz+r6_first_two_rows+gripper_open",
            "state_layout_mode": "public_lap_train_matched_rows_v1",
            "polaris_profile": "panda_link8_eef_pose_single_arm_v1",
            "anchor_position": [0.0, 0.0, 0.0],
            "anchor_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "state": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            "server_delta_chunk": zero_chunk,
            "raw_delta_chunk": zero_chunk,
            "base_delta_chunk": zero_chunk,
            "anchored_action_chunk": anchored_chunk,
            "reasoning": None,
        }
    )
    return record


def _write_completed_trace(path: Path, result):
    episode = result["episode"]
    records = [
        {
            **_trace_common("reset", episode),
            "environment_runtime_profile": EGO_LAP_ENVIRONMENT_RUNTIME_PROFILE,
            "environment_before": _environment_state(0),
        }
    ]
    for step in range(result["episode_length"]):
        if step % 8 == 0:
            records.append(_trace_query(episode, step // 8))
        identity = {
            "query": step // 8,
            "step": step,
            "chunk_index": step % 8,
        }
        records.extend(
            [
                {
                    **_trace_common("action", episode),
                    **identity,
                    "raw_delta": [0.0] * 7,
                    "polaris_action": [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                },
                {
                    **_trace_common("execution", episode),
                    **identity,
                    "transition": {
                        "step_index": step,
                        "terminated": False,
                        "truncated": False,
                        "environment_before": _environment_state(step),
                        "environment_after": _environment_state(step + 1),
                        "counter_deltas": {
                            "episode_length": 1,
                            "sim_step_counter": 8,
                            "common_step_counter": 1,
                        },
                        "camera_frame_deltas": {
                            "external_cam": 1,
                            "wrist_cam": 1,
                        },
                    },
                },
            ]
        )
    records.append(
        {
            **_trace_common("episode_complete", episode),
            **result,
            "status": "completed",
            "terminal_rollout": _terminal_rollout(result),
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _video_probe(_path: Path, *, frames=450):
    return {"frame_count": frames, "height": 224, "width": 448}


def _prepare_episode_transaction(tmp_path: Path, *, episode: int):
    result = _episode_result(episode=episode, length=450)
    safety = _episode_safety(episode=episode, length=450)
    video_path = tmp_path / f"episode_{episode}.mp4"
    trace_dir = tmp_path / "policy_traces"
    trace_path = trace_dir / f"episode_{episode:06d}.jsonl"
    video_path.write_bytes(f"complete-video-{episode}".encode())
    _write_completed_trace(trace_path, result)
    identity = build_episode_artifact_identity(
        run_folder=tmp_path,
        trace_path=trace_path,
        episode_result=result,
        video_probe_fn=_video_probe,
    )
    sidecar_path = tmp_path / "ik_safety" / f"episode_{episode:06d}.json"
    payload = atomic_write_episode_safety(
        sidecar_path,
        eef_controller_profile=EEF_CONTROLLER_BASELINE_PROFILE,
        controller_repair_candidate=_baseline_controller_report(safety),
        arm_failure_substep_trace=None,
        all_six_gripper_trace=None,
        episode_index=episode,
        episode_result=result,
        safety=safety,
        artifact_identity=identity,
        terminal_rollout=_terminal_rollout(result),
    )
    return result, sidecar_path, payload


def test_runtime_protocol_requires_outer450_internal451_at_15hz():
    env, _ = _runtime_fixture()
    resolved = validate_ego_lap_runtime_protocol(env)
    assert resolved["episode_steps"] == 450
    assert resolved["live_max_episode_length"] == 451
    assert resolved["autoreset_margin_steps"] == 1
    assert resolved["policy_hz"] == 15.0
    assert resolved["physics_hz"] == 120.0
    assert resolved["decimation"] == 8

    env.max_episode_length = 450
    with pytest.raises(ValueError, match="451"):
        validate_ego_lap_runtime_protocol(env)
    env.max_episode_length = 451
    env.unwrapped.step_dt = 1.0 / 10.0
    with pytest.raises(ValueError, match="15 Hz"):
        validate_ego_lap_runtime_protocol(env)


def test_runtime_frame_matches_direct_link8_and_absolute_action_term():
    env, observation = _runtime_fixture()
    result = validate_eef_runtime_frame(env, observation)
    assert result["eef_frame"] == "panda_link8"
    assert result["position_error_m"] < 1e-12
    assert result["rotation_error_rad"] < 1e-12
    assert result["reference_frame"] == "panda_link0"
    assert result["controlled_body"] == "panda_link8"
    assert result["body_offset"] == "identity"
    assert result["command_type"] == "pose"
    assert result["use_relative_mode"] is False
    assert result["ik_method"] == "dls"
    assert result["dls_damping"] == 0.01
    assert result["arm_scale"] == 1.0
    assert result["arm_joint_names"] == [f"panda_joint{index}" for index in range(1, 8)]
    assert result["gripper_threshold_profile"] == GRIPPER_THRESHOLD_PROFILE
    assert result["ik_safety_profile"] == EEF_IK_SAFETY_PROFILE
    assert result["action_dim"] == 7

    safety = validate_eef_runtime_safety(env)
    assert safety["profile"] == EEF_IK_SAFETY_PROFILE
    assert safety["max_delta_joint_pos_rad"] == [
        np.float32(np.float32(value) * np.float32(1.0 / 120.0)).item()
        for value in PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S
    ]
    assert safety["target_soft_limit_margin_rad"] == safety["max_delta_joint_pos_rad"]
    assert safety["target_soft_limit_guard_band_profile"] == (
        TARGET_SOFT_LIMIT_GUARD_BAND_PROFILE
    )
    assert safety["target_joint_pos_limits_float32_sha256"] == (
        PANDA_TARGET_JOINT_POS_LIMITS_FLOAT32_SHA256
    )


def test_runtime_candidate_selects_exact_profile_schema_and_diagnostics(tmp_path):
    env, observation, report = _candidate_trigger_runtime()

    frame = validate_eef_runtime_frame(env, observation)
    assert frame["ik_safety_profile"] == EEF_IK_WRIST_ENERGY_BRAKE_CANDIDATE_PROFILE
    validated = validate_eef_runtime_safety(env)
    assert validated is report
    assert validated["wrist_energy_brake_profile"] == WRIST_ENERGY_BRAKE_PROFILE
    assert validated["wrist_energy_brake_joint_names"] == list(
        WRIST_ENERGY_BRAKE_JOINT_NAMES
    )
    assert validated["wrist_energy_brake_latch_substeps"] == 2
    assert validated["wrist_energy_brake_target_shift_fraction"] == 0.9
    assert validated["wrist_energy_brake_latch_remaining_substeps"] == [1]

    durable_result = _episode_result(length=450)
    durable = _episode_safety(length=450, wrist_energy_brake=True)
    validate_episode_safety_cadence(
        safety=durable,
        episode_result=durable_result,
    )
    aggregate = aggregate_episode_safety(
        durable,
        [
            {
                "episode_index": 0,
                "episode_result": durable_result,
                "artifact_identity": {},
                "cadence_evidence": {"apply_calls": 3600},
                "terminal_rollout": _terminal_rollout(durable_result),
                "safety": durable,
                "path": "episode_000000.json",
                "sha256": "0" * 64,
            }
        ],
    )
    assert aggregate["profile"] == EEF_IK_WRIST_ENERGY_BRAKE_CANDIDATE_PROFILE
    assert aggregate["wrist_energy_brake_profile"] == WRIST_ENERGY_BRAKE_PROFILE
    assert aggregate["episodes"][0]["wrist_energy_brake_diagnostics"] == []
    assert aggregate["episodes"][0]["wrist_energy_brake_latch_remaining_substeps"] == [
        0
    ]
    atomic_write_runtime_contract(
        tmp_path / "candidate-runtime.json",
        eef_controller_profile=EEF_CONTROLLER_BASELINE_PROFILE,
        controller_repair_candidate=_controller_aggregate(aggregate),
        protocol=validate_ego_lap_runtime_protocol(env),
        frame=frame,
        ik_safety=aggregate,
    )
    with pytest.raises(ValueError, match="profiles disagree"):
        atomic_write_runtime_contract(
            tmp_path / "candidate-runtime-mismatch.json",
            eef_controller_profile=EEF_CONTROLLER_BASELINE_PROFILE,
            controller_repair_candidate=_controller_aggregate(aggregate),
            protocol=validate_ego_lap_runtime_protocol(env),
            frame={**frame, "ik_safety_profile": EEF_IK_SAFETY_PROFILE},
            ik_safety=aggregate,
        )


def test_runtime_disabled_mode_retains_exact_base_schema():
    env, observation = _runtime_fixture()
    report = validate_eef_runtime_safety(env)
    frame = validate_eef_runtime_frame(env, observation)

    assert report["profile"] == EEF_IK_SAFETY_PROFILE
    assert frame["ik_safety_profile"] == EEF_IK_SAFETY_PROFILE
    assert not any(key.startswith("wrist_energy_brake_") for key in report)
    assert not any(key.startswith("wrist_energy_brake_") for key in report["counters"])


def test_runtime_candidate_rejects_mode_schema_and_static_tampering():
    env, _, report = _candidate_trigger_runtime()
    arm_term = env.unwrapped.action_manager._terms["arm"]

    arm_term.cfg.enable_wrist_energy_brake = False
    with pytest.raises(ValueError, match="schema drift"):
        validate_eef_runtime_safety(env)

    arm_term.cfg.enable_wrist_energy_brake = "true"
    with pytest.raises(ValueError, match="must be exactly bool"):
        validate_eef_runtime_safety(env)

    base_env, _ = _runtime_fixture()
    base_env.unwrapped.action_manager._terms["arm"].cfg.enable_wrist_energy_brake = True
    with pytest.raises(ValueError, match="schema drift"):
        validate_eef_runtime_safety(base_env)

    for field, tampered in (
        ("profile", EEF_IK_SAFETY_PROFILE),
        ("wrist_energy_brake_profile", "wrong"),
        ("wrist_energy_brake_joint_names", ["panda_joint7"]),
        ("wrist_energy_brake_latch_substeps", 3),
        ("wrist_energy_brake_target_shift_fraction", 0.8),
    ):
        env, _, report = _candidate_trigger_runtime()
        report[field] = tampered
        with pytest.raises(ValueError, match=field):
            validate_eef_runtime_safety(env)

    env, _, report = _candidate_trigger_runtime()
    report["wrist_energy_brake_target_shift_threshold_rad"][0] += 1e-3
    with pytest.raises(ValueError, match="target-shift threshold mismatch"):
        validate_eef_runtime_safety(env)


def test_runtime_candidate_rejects_dynamic_counter_and_diagnostic_tampering():
    env, _, report = _candidate_trigger_runtime()
    report["wrist_energy_brake_latch_remaining_substeps"] = [3]
    with pytest.raises(ValueError, match="latch state"):
        validate_eef_runtime_safety(env)

    env, _, report = _candidate_trigger_runtime()
    report["counters"]["wrist_energy_brake_trigger_events"] = 2
    with pytest.raises(ValueError, match="counter history"):
        validate_eef_runtime_safety(env)

    env, _, report = _candidate_trigger_runtime()
    report["counters"]["wrist_energy_brake_diagnostics_dropped"] = 1
    with pytest.raises(ValueError, match="diagnostic accounting"):
        validate_eef_runtime_safety(env)

    env, _, report = _candidate_trigger_runtime()
    report["wrist_energy_brake_diagnostics"][0].update(
        {"apply_index": 0, "policy_step": 0, "physics_substep": 0}
    )
    with pytest.raises(ValueError, match="arming cadence"):
        validate_eef_runtime_safety(env)

    env, _, report = _candidate_trigger_runtime()
    report["counters"]["apply_calls"] = 10
    report["counters"]["environment_substeps"] = 10
    with pytest.raises(ValueError, match="open-latch apply identity"):
        validate_eef_runtime_safety(env)

    env, _, report = _candidate_trigger_runtime()
    report["counters"]["wrist_energy_brake_attempted_joint_targets"] = 2
    with pytest.raises(ValueError, match="attempted-target count"):
        validate_eef_runtime_safety(env)

    env, _, report = _candidate_trigger_runtime()
    report["counters"]["wrist_energy_brake_braked_joint_targets"] = 0
    with pytest.raises(ValueError, match="effective-target count"):
        validate_eef_runtime_safety(env)

    env, _, report = _candidate_two_substep_runtime()
    validate_eef_runtime_safety(env)
    follow_up = report["wrist_energy_brake_diagnostics"][1]
    follow_up.update(
        {
            "apply_index": 1_000,
            "policy_step": 125,
            "physics_substep": 0,
        }
    )
    report["counters"]["apply_calls"] = 1_001
    report["counters"]["environment_substeps"] = 1_001
    with pytest.raises(ValueError, match="follow-up cadence"):
        validate_eef_runtime_safety(env)

    env, _, report = _candidate_two_substep_runtime()
    follow_up = report["wrist_energy_brake_diagnostics"][1]
    follow_up["previous_applied_target_rad"][4] = -0.01
    follow_up["target_shift_rad"][0] = 0.03
    with pytest.raises(ValueError, match="previous-target chain"):
        validate_eef_runtime_safety(env)

    env, _, report = _candidate_trigger_runtime()
    report["wrist_energy_brake_diagnostics"][0]["target_shift_rad"][0] += 1e-3
    with pytest.raises(ValueError, match="target-shift drift"):
        validate_eef_runtime_safety(env)

    env, _, report = _candidate_trigger_runtime()
    report["wrist_energy_brake_diagnostics"][0]["trigger_joint_mask"] = [
        False,
        True,
        False,
    ]
    with pytest.raises(ValueError, match="trigger-mask drift"):
        validate_eef_runtime_safety(env)

    env, _, report = _candidate_trigger_runtime()
    report["wrist_energy_brake_diagnostics"][0]["applied_target_rad"][4] = 0.01
    with pytest.raises(ValueError, match="applied-target drift"):
        validate_eef_runtime_safety(env)

    env, _, report = _candidate_trigger_runtime()
    report["wrist_energy_brake_diagnostics"][0]["unexpected"] = True
    with pytest.raises(ValueError, match="diagnostic schema drift"):
        validate_eef_runtime_safety(env)


def test_runtime_candidate_rejects_impossible_hidden_effective_count():
    diagnostics = []
    for index in range(32):
        global_active_ordinal = 2 + index
        trigger_record = global_active_ordinal % 2 == 0
        apply_index = (
            1
            + 3 * (global_active_ordinal // WRIST_ENERGY_BRAKE_LATCH_SUBSTEPS)
            + global_active_ordinal % WRIST_ENERGY_BRAKE_LATCH_SUBSTEPS
        )
        diagnostics.append(
            {
                "apply_index": apply_index,
                "reversal_detection_armed": trigger_record,
                "trigger_joint_mask": [trigger_record, False, False],
                "attempted_joint_mask": (
                    [True, True, True] if index == 0 else [False, False, False]
                ),
                "braked_joint_mask": [False, False, False],
                "previous_applied_target_rad": [0.0] * 7,
                "applied_target_rad": [0.0] * 7,
            }
        )
    counters = {
        "apply_calls": diagnostics[-1]["apply_index"] + 1,
        "wrist_energy_brake_trigger_events": 17,
        "wrist_energy_brake_active_substeps": 34,
        "wrist_energy_brake_attempted_joint_targets": 3,
        "wrist_energy_brake_braked_joint_targets": 3,
        "wrist_energy_brake_diagnostics_dropped": 2,
        "current_joint_limit_aborts": 0,
        "invariant_aborts": 0,
        "nonfinite_aborts": 0,
    }

    with pytest.raises(ValueError, match="hidden effective/attempted"):
        _validate_wrist_energy_brake_history(
            counters=counters,
            latch_remaining=[0],
            diagnostics=diagnostics,
            field="test",
        )


def test_runtime_candidate_allows_bound_target_state_abort_only_in_candidate():
    env, _, report = _candidate_trigger_runtime()
    report["counters"]["apply_calls"] = 3
    report["counters"]["environment_substeps"] = 3
    report["counters"]["invariant_aborts"] = 1
    report["guard_diagnostics"] = [
        {
            "kind": "wrist_energy_brake_target_state_abort",
            "episode_index": 0,
            "policy_step": 0,
            "physics_substep": 2,
            "joint_pos_rad": None,
            "raw_delta_joint_pos_rad": None,
            "raw_joint_pos_target_rad": None,
            "safe_joint_pos_target_rad": None,
            "pose_error_norm": None,
            "jacobian_finite": None,
            "jacobian_max_abs": None,
            "eef_quaternion_norm": None,
        }
    ]
    validate_eef_runtime_safety(env)

    durable = _episode_safety(
        numerical_failure=True,
        wrist_energy_brake=True,
    )
    durable["counters"]["nonfinite_aborts"] = 0
    durable["counters"]["invariant_aborts"] = 1
    durable["guard_diagnostics"][0]["kind"] = "wrist_energy_brake_target_state_abort"
    validate_episode_safety_cadence(
        safety=durable,
        episode_result=_episode_result(numerical_failure=True),
    )


def test_runtime_contract_is_atomic_and_has_exact_evidence_schema():
    env, observation = _runtime_fixture()
    protocol = validate_ego_lap_runtime_protocol(env)
    frame = validate_eef_runtime_frame(env, observation)
    safety = validate_eef_runtime_safety(env)
    aggregate_safety = aggregate_episode_safety(safety, [])

    with tempfile.TemporaryDirectory() as temporary_directory:
        path = Path(temporary_directory) / "nested" / "runtime.json"
        path.parent.mkdir(parents=True)
        path.write_text('{"stale": true}\n', encoding="utf-8")
        atomic_write_runtime_contract(
            path,
            eef_controller_profile=EEF_CONTROLLER_BASELINE_PROFILE,
            controller_repair_candidate=_controller_aggregate(aggregate_safety),
            protocol=protocol,
            frame=frame,
            ik_safety=aggregate_safety,
        )
        payload = json.loads(path.read_text(encoding="utf-8"))

        assert payload == {
            "schema_version": EEF_RUNTIME_CONTRACT_SCHEMA_VERSION,
            "eef_controller_profile": EEF_CONTROLLER_BASELINE_PROFILE,
            "controller_repair_candidate": _controller_aggregate(aggregate_safety),
            "protocol": {
                "profile": "ego_lap_eef_outer450_internal451_no_autoreset_v1",
                "episode_steps": 450,
                "live_max_episode_length": 451,
                "autoreset_margin_steps": 1,
                "policy_hz": 15.0,
                "step_dt": 1.0 / 15.0,
                "physics_hz": 120.0,
                "physics_dt": 1.0 / 120.0,
                "decimation": 8,
                "camera_sensor_names": ["external_cam", "wrist_cam"],
            },
            "frame": frame,
            "ik_safety": aggregate_safety,
        }
        assert not list(path.parent.glob(".*.tmp"))


def test_candidate_schema6_sidecar_resume_and_runtime_profile_propagation(
    tmp_path: Path,
    monkeypatch,
):
    result = _episode_result(length=450)
    safety = _attach_candidate_stub_gripper(
        _episode_safety(length=450),
        length=450,
    )
    controller_report = _candidate_controller_report(safety, initial=False)
    initial_report = _candidate_controller_report(safety, initial=True)
    all_six_trace = _all_six_trace(episode=0, length=450, apply_calls=3600)
    terminal = _terminal_rollout(result)
    sidecar_path = tmp_path / "ik_safety" / "episode_000000.json"

    def static_contract(value, *, expected_target_slew_profile):
        assert expected_target_slew_profile == (
            EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
        )
        return dict(value)

    def dynamic_contract(value, *, expected_target_slew_profile):
        assert expected_target_slew_profile == (
            EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
        )
        return dict(value)

    with monkeypatch.context() as patch:
        patch.setattr(
            runtime_contract_module,
            "validate_eef_gripper_static_contract",
            static_contract,
        )
        patch.setattr(
            runtime_contract_module,
            "validate_eef_gripper_dynamic_evidence",
            dynamic_contract,
        )
        patch.setattr(
            runtime_contract_module,
            "validate_eef_controller_safety_evidence",
            lambda *_args, expected_profile, **_kwargs: eef_controller_profile(
                expected_profile
            ),
        )
        payload = atomic_write_episode_safety(
            sidecar_path,
            eef_controller_profile=(EEF_CONTROLLER_MIMIC_COMPLIANCE_CANDIDATE_PROFILE),
            controller_repair_candidate=controller_report,
            arm_failure_substep_trace=None,
            all_six_gripper_trace=all_six_trace,
            episode_index=0,
            episode_result=result,
            safety=safety,
            artifact_identity=_artifact_identity_for_terminal(result, terminal),
            terminal_rollout=terminal,
            expected_gripper_target_slew_profile=(
                EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
            ),
        )
        assert payload["schema_version"] == 6
        assert payload["eef_controller_profile"] == (
            EEF_CONTROLLER_MIMIC_COMPLIANCE_CANDIDATE_PROFILE
        )
        assert payload["controller_repair_candidate"] == controller_report
        assert payload["all_six_gripper_trace"] == all_six_trace
        assert payload["arm_failure_substep_trace"] is None

        unsafe_nominal = copy.deepcopy(safety)
        unsafe_nominal["maxima"]["applied_delta_joint_pos_rad"][0] = (
            controller_report["arm_slew_headroom"]["nominal_max_delta_joint_pos_rad"][0]
            + 2e-6
        )
        with pytest.raises(ValueError, match="nominal arm-slew bound"):
            atomic_write_episode_safety(
                tmp_path / "unsafe_nominal.json",
                eef_controller_profile=(
                    EEF_CONTROLLER_MIMIC_COMPLIANCE_CANDIDATE_PROFILE
                ),
                controller_repair_candidate=controller_report,
                arm_failure_substep_trace=None,
                all_six_gripper_trace=all_six_trace,
                episode_index=0,
                episode_result=result,
                safety=unsafe_nominal,
                artifact_identity=_artifact_identity_for_terminal(result, terminal),
                terminal_rollout=terminal,
                expected_gripper_target_slew_profile=(
                    EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
                ),
            )

        unsafe_endpoint_count = copy.deepcopy(safety)
        unsafe_endpoint_count["gripper_runtime_dynamic"]["driver_target_slew"][
            "endpoint_change_count"
        ] = 1
        with pytest.raises(ValueError, match="endpoint-change cadence"):
            atomic_write_episode_safety(
                tmp_path / "unsafe_endpoint.json",
                eef_controller_profile=(
                    EEF_CONTROLLER_MIMIC_COMPLIANCE_CANDIDATE_PROFILE
                ),
                controller_repair_candidate=controller_report,
                arm_failure_substep_trace=None,
                all_six_gripper_trace=all_six_trace,
                episode_index=0,
                episode_result=result,
                safety=unsafe_endpoint_count,
                artifact_identity=_artifact_identity_for_terminal(result, terminal),
                terminal_rollout=terminal,
                expected_gripper_target_slew_profile=(
                    EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
                ),
            )

        with pytest.raises(ValueError, match="controller profile drift"):
            load_episode_safety_sidecars(sidecar_path.parent, [0])
        sidecars = load_episode_safety_sidecars(
            sidecar_path.parent,
            [0],
            expected_eef_controller_profile=(
                EEF_CONTROLLER_MIMIC_COMPLIANCE_CANDIDATE_PROFILE
            ),
            expected_gripper_target_slew_profile=(
                EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
            ),
        )
        aggregate = aggregate_episode_safety(
            safety,
            sidecars,
            expected_eef_controller_profile=(
                EEF_CONTROLLER_MIMIC_COMPLIANCE_CANDIDATE_PROFILE
            ),
            expected_gripper_target_slew_profile=(
                EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
            ),
        )
        controller_aggregate = build_eef_controller_repair_candidate_aggregate(
            live_safety=safety,
            initial_report=initial_report,
            sidecars=sidecars,
            expected_eef_controller_profile=(
                EEF_CONTROLLER_MIMIC_COMPLIANCE_CANDIDATE_PROFILE
            ),
            expected_gripper_target_slew_profile=(
                EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
            ),
        )
        env, observation = _runtime_fixture()
        runtime_path = tmp_path / "polaris_runtime_contract.json"
        atomic_write_runtime_contract(
            runtime_path,
            eef_controller_profile=(EEF_CONTROLLER_MIMIC_COMPLIANCE_CANDIDATE_PROFILE),
            controller_repair_candidate=controller_aggregate,
            protocol=validate_ego_lap_runtime_protocol(env),
            frame=validate_eef_runtime_frame(env, observation),
            ik_safety=aggregate,
            expected_gripper_target_slew_profile=(
                EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
            ),
        )
        runtime = json.loads(runtime_path.read_text())
        assert set(runtime) == {
            "schema_version",
            "eef_controller_profile",
            "controller_repair_candidate",
            "protocol",
            "frame",
            "ik_safety",
        }
        assert runtime["schema_version"] == 6
        assert runtime["eef_controller_profile"] == (
            EEF_CONTROLLER_MIMIC_COMPLIANCE_CANDIDATE_PROFILE
        )
        assert runtime["controller_repair_candidate"]["initial"] == initial_report
        assert runtime["controller_repair_candidate"]["episodes"] == [
            {"episode_index": 0, "report": controller_report}
        ]


def test_release_ramp_v4_sidecar_and_aggregate_are_profile_bound(
    tmp_path: Path,
    monkeypatch,
):
    result = _episode_result(length=450)
    safety = _attach_candidate_stub_gripper(
        _episode_safety(length=450),
        length=450,
    )
    report = _candidate_controller_report(
        safety,
        initial=False,
        profile=EEF_CONTROLLER_RELEASE_RAMP_CANDIDATE_PROFILE,
    )
    initial = _candidate_controller_report(
        safety,
        initial=True,
        profile=EEF_CONTROLLER_RELEASE_RAMP_CANDIDATE_PROFILE,
    )
    trace = _all_six_trace(episode=0, length=450, apply_calls=3600)
    terminal = _terminal_rollout(result)
    sidecar_path = tmp_path / "ik_safety" / "episode_000000.json"

    with monkeypatch.context() as patch:
        patch.setattr(
            runtime_contract_module,
            "validate_eef_gripper_static_contract",
            lambda value, **_kwargs: dict(value),
        )
        patch.setattr(
            runtime_contract_module,
            "validate_eef_gripper_dynamic_evidence",
            lambda value, **_kwargs: dict(value),
        )
        patch.setattr(
            runtime_contract_module,
            "validate_eef_controller_safety_evidence",
            lambda *_args, expected_profile, **_kwargs: eef_controller_profile(
                expected_profile
            ),
        )
        payload = atomic_write_episode_safety(
            sidecar_path,
            eef_controller_profile=EEF_CONTROLLER_RELEASE_RAMP_CANDIDATE_PROFILE,
            controller_repair_candidate=report,
            arm_failure_substep_trace=None,
            all_six_gripper_trace=trace,
            episode_index=0,
            episode_result=result,
            safety=safety,
            artifact_identity=_artifact_identity_for_terminal(result, terminal),
            terminal_rollout=terminal,
            expected_gripper_target_slew_profile=(
                EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
            ),
        )
        assert payload["eef_controller_profile"] == (
            EEF_CONTROLLER_RELEASE_RAMP_CANDIDATE_PROFILE
        )
        assert (
            payload["controller_repair_candidate"]["arm_release_ramp"]
            == (report["arm_release_ramp"])
        )
        sidecars = load_episode_safety_sidecars(
            sidecar_path.parent,
            [0],
            expected_eef_controller_profile=(
                EEF_CONTROLLER_RELEASE_RAMP_CANDIDATE_PROFILE
            ),
            expected_gripper_target_slew_profile=(
                EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
            ),
        )
        aggregate = build_eef_controller_repair_candidate_aggregate(
            live_safety=safety,
            initial_report=initial,
            sidecars=sidecars,
            expected_eef_controller_profile=(
                EEF_CONTROLLER_RELEASE_RAMP_CANDIDATE_PROFILE
            ),
            expected_gripper_target_slew_profile=(
                EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
            ),
        )
        assert aggregate["eef_controller_profile"] == (
            EEF_CONTROLLER_RELEASE_RAMP_CANDIDATE_PROFILE
        )
        assert aggregate["initial"]["arm_release_ramp"] == initial["arm_release_ramp"]
        assert (
            aggregate["episodes"][0]["report"]["arm_release_ramp"]
            == report["arm_release_ramp"]
        )

        missing = copy.deepcopy(report)
        missing.pop("arm_release_ramp")
        with pytest.raises(ValueError, match="schema drift"):
            atomic_write_episode_safety(
                tmp_path / "missing_ramp.json",
                eef_controller_profile=(EEF_CONTROLLER_RELEASE_RAMP_CANDIDATE_PROFILE),
                controller_repair_candidate=missing,
                arm_failure_substep_trace=None,
                all_six_gripper_trace=trace,
                episode_index=0,
                episode_result=result,
                safety=safety,
                artifact_identity=_artifact_identity_for_terminal(result, terminal),
                terminal_rollout=terminal,
                expected_gripper_target_slew_profile=(
                    EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
                ),
            )


def test_velocity_recovery_v5_sidecar_is_schema7_and_profile_bound(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result = _episode_result(length=450)
    safety = _attach_candidate_stub_gripper(
        _episode_safety(length=450),
        length=450,
    )
    safety["profile"] = EEF_IK_CURRENT_VELOCITY_RECOVERY_CANDIDATE_PROFILE
    safety["current_joint_velocity_abort"] = None
    safety["current_joint_velocity_recovery"] = _empty_velocity_recovery_report()
    report = _candidate_controller_report(
        safety,
        initial=False,
        profile=EEF_CONTROLLER_VELOCITY_RECOVERY_CANDIDATE_PROFILE,
    )
    trace = _all_six_trace(episode=0, length=450, apply_calls=3600)
    terminal = _terminal_rollout(result)
    sidecar_path = tmp_path / "ik_safety" / "episode_000000.json"

    with monkeypatch.context() as patch:
        patch.setattr(
            runtime_contract_module,
            "validate_eef_gripper_static_contract",
            lambda value, **_kwargs: dict(value),
        )
        patch.setattr(
            runtime_contract_module,
            "validate_eef_gripper_dynamic_evidence",
            lambda value, **_kwargs: dict(value),
        )
        patch.setattr(
            runtime_contract_module,
            "validate_eef_controller_safety_evidence",
            lambda *_args, expected_profile, **_kwargs: eef_controller_profile(
                expected_profile
            ),
        )
        payload = atomic_write_episode_safety(
            sidecar_path,
            eef_controller_profile=(EEF_CONTROLLER_VELOCITY_RECOVERY_CANDIDATE_PROFILE),
            controller_repair_candidate=report,
            arm_failure_substep_trace=None,
            all_six_gripper_trace=trace,
            episode_index=0,
            episode_result=result,
            safety=safety,
            artifact_identity=_artifact_identity_for_terminal(result, terminal),
            terminal_rollout=terminal,
            expected_gripper_target_slew_profile=(
                EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
            ),
        )
        assert (
            payload["schema_version"]
            == (EEF_SAFETY_SIDECAR_VELOCITY_RECOVERY_SCHEMA_VERSION)
            == 7
        )
        assert (
            payload["safety"]["current_joint_velocity_recovery"]["contract"][
                "schema_version"
            ]
            == 3
        )
        assert (
            payload["controller_repair_candidate"]["current_joint_velocity_recovery"]
            == safety["current_joint_velocity_recovery"]
        )

        sidecars = load_episode_safety_sidecars(
            sidecar_path.parent,
            [0],
            expected_eef_controller_profile=(
                EEF_CONTROLLER_VELOCITY_RECOVERY_CANDIDATE_PROFILE
            ),
            expected_gripper_target_slew_profile=(
                EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
            ),
        )
        assert sidecars[0]["schema_version"] == 7

        drifted = copy.deepcopy(payload)
        drifted["schema_version"] = EEF_SAFETY_SIDECAR_SCHEMA_VERSION
        sidecar_path.write_text(json.dumps(drifted), encoding="utf-8")
        with pytest.raises(ValueError, match="sidecar identity"):
            load_episode_safety_sidecars(
                sidecar_path.parent,
                [0],
                expected_eef_controller_profile=(
                    EEF_CONTROLLER_VELOCITY_RECOVERY_CANDIDATE_PROFILE
                ),
                expected_gripper_target_slew_profile=(
                    EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
                ),
            )


def test_prepared_sidecar_recovers_exact_missing_csv_row(tmp_path: Path):
    result = _episode_result(length=450)
    safety = _episode_safety(length=450)
    video_path = tmp_path / "episode_0.mp4"
    trace_dir = tmp_path / "policy_traces"
    trace_path = trace_dir / "episode_000000.jsonl"
    video_path.write_bytes(b"complete-video")
    _write_completed_trace(trace_path, result)
    identity = build_episode_artifact_identity(
        run_folder=tmp_path,
        trace_path=trace_path,
        episode_result=result,
        video_probe_fn=_video_probe,
    )
    sidecar_path = tmp_path / "ik_safety" / "episode_000000.json"
    atomic_write_episode_safety(
        sidecar_path,
        eef_controller_profile=EEF_CONTROLLER_BASELINE_PROFILE,
        controller_repair_candidate=_baseline_controller_report(safety),
        arm_failure_substep_trace=None,
        all_six_gripper_trace=None,
        episode_index=0,
        episode_result=result,
        safety=safety,
        artifact_identity=identity,
        terminal_rollout=_terminal_rollout(result),
    )

    recovered, changed = reconcile_episode_safety_transactions(
        empty_eval_results(),
        directory=sidecar_path.parent,
        run_folder=tmp_path,
        trace_dir=trace_dir,
        expected_rollouts=50,
        expected_horizon=450,
        video_probe_fn=_video_probe,
    )

    assert changed is True
    assert recovered.to_dict(orient="records") == [result]
    assert sidecar_path.is_file()


def test_transaction_recovery_is_idempotent_after_csv_commit(tmp_path: Path):
    result, sidecar_path, payload = _prepare_episode_transaction(tmp_path, episode=0)
    committed, changed = reconcile_episode_safety_transactions(
        pd.DataFrame([result]),
        directory=sidecar_path.parent,
        run_folder=tmp_path,
        trace_dir=tmp_path / "policy_traces",
        expected_rollouts=50,
        expected_horizon=450,
        video_probe_fn=_video_probe,
    )
    assert changed is False
    assert committed.to_dict(orient="records") == [result]
    assert json.loads(sidecar_path.read_text()) == payload

    drifted_safety = _episode_safety(length=450)
    drifted_safety["counters"]["slew_limit_events"] = 1
    drifted_safety["counters"]["slew_limited_joints"] = 1
    with pytest.raises(ValueError, match="Refusing to overwrite drifted"):
        atomic_write_episode_safety(
            sidecar_path,
            eef_controller_profile=EEF_CONTROLLER_BASELINE_PROFILE,
            controller_repair_candidate=_baseline_controller_report(drifted_safety),
            arm_failure_substep_trace=None,
            all_six_gripper_trace=None,
            episode_index=0,
            episode_result=result,
            safety=drifted_safety,
            artifact_identity=payload["artifact_identity"],
            terminal_rollout=payload["terminal_rollout"],
        )


def test_sidecar_loader_and_reconciler_reject_previous_v5_schema(tmp_path: Path):
    _result, sidecar_path, payload = _prepare_episode_transaction(tmp_path, episode=0)
    assert payload["schema_version"] == EEF_SAFETY_SIDECAR_SCHEMA_VERSION == 6
    payload["schema_version"] = 5
    sidecar_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid episode safety sidecar identity"):
        load_episode_safety_sidecars(sidecar_path.parent, [0])
    with pytest.raises(ValueError, match="Invalid prepared episode safety transaction"):
        reconcile_episode_safety_transactions(
            empty_eval_results(),
            directory=sidecar_path.parent,
            run_folder=tmp_path,
            trace_dir=tmp_path / "policy_traces",
            expected_rollouts=50,
            expected_horizon=450,
            video_probe_fn=_video_probe,
        )


def test_transaction_recovery_rejects_missing_or_multiple_prepared_sidecars(
    tmp_path: Path,
):
    result = _episode_result()
    with pytest.raises(ValueError, match="must equal the CSV prefix"):
        reconcile_episode_safety_transactions(
            pd.DataFrame([result]),
            directory=tmp_path / "ik_safety",
            run_folder=tmp_path,
            trace_dir=tmp_path / "policy_traces",
            expected_rollouts=50,
            expected_horizon=450,
            video_probe_fn=_video_probe,
        )

    _prepare_episode_transaction(tmp_path, episode=0)
    _prepare_episode_transaction(tmp_path, episode=1)
    with pytest.raises(ValueError, match="add exactly its next episode"):
        reconcile_episode_safety_transactions(
            empty_eval_results(),
            directory=tmp_path / "ik_safety",
            run_folder=tmp_path,
            trace_dir=tmp_path / "policy_traces",
            expected_rollouts=50,
            expected_horizon=450,
            video_probe_fn=_video_probe,
        )


def test_transaction_recovery_archives_uncommitted_artifacts_without_deleting_evidence(
    tmp_path: Path,
):
    video = tmp_path / "episode_0.mp4"
    trace = tmp_path / "policy_traces" / "episode_000000.jsonl"
    temporary_video = tmp_path / ".episode_0.tmp.mp4"
    trace.parent.mkdir(parents=True)
    video.write_bytes(b"uncommitted-video")
    trace.write_text("partial trace evidence\n")
    temporary_video.write_bytes(b"partial-video-evidence")

    frame, changed = reconcile_episode_safety_transactions(
        empty_eval_results(),
        directory=tmp_path / "ik_safety",
        run_folder=tmp_path,
        trace_dir=trace.parent,
        expected_rollouts=50,
        expected_horizon=450,
        video_probe_fn=_video_probe,
    )

    assert frame.empty
    assert changed is False
    assert not video.exists()
    assert not trace.exists()
    assert not temporary_video.exists()
    archived = list((tmp_path / "recovery_orphans" / "episode_000000").iterdir())
    assert len(archived) == 3
    assert sorted(path.read_bytes() for path in archived) == sorted(
        [
            b"uncommitted-video",
            b"partial trace evidence\n",
            b"partial-video-evidence",
        ]
    )


def test_runtime_aggregate_reconstructs_all_resume_history(tmp_path: Path):
    for episode in range(2):
        _prepare_episode_transaction(tmp_path, episode=episode)
    sidecars = load_episode_safety_sidecars(tmp_path / "ik_safety", [0, 1])
    env, _ = _runtime_fixture()
    live = env.unwrapped.action_manager._terms["arm"].safety_report()
    aggregate = aggregate_episode_safety(live, sidecars)

    assert aggregate["episodes_completed"] == 2
    assert aggregate["counters"]["apply_calls"] == 7200
    assert aggregate["counters"]["environment_substeps"] == 7200
    assert [item["episode_index"] for item in aggregate["episodes"]] == [0, 1]
    assert all(item["sidecar_sha256"] for item in aggregate["episodes"])


def test_velocity_recovery_aggregate_and_runtime_contract_are_schema7(
    tmp_path: Path,
) -> None:
    env, observation = _runtime_fixture()
    result = _episode_result(length=450)
    safety = _episode_safety(length=450)
    safety["profile"] = EEF_IK_CURRENT_VELOCITY_RECOVERY_CANDIDATE_PROFILE
    safety["current_joint_velocity_abort"] = None
    safety["current_joint_velocity_recovery"] = _empty_velocity_recovery_report()
    sidecar = {
        "episode_index": 0,
        "episode_result": result,
        "artifact_identity": {},
        "cadence_evidence": {"apply_calls": 3600},
        "terminal_rollout": _terminal_rollout(result),
        "safety": safety,
        "path": "episode_000000.json",
        "sha256": "7" * 64,
    }
    aggregate = aggregate_episode_safety(
        safety,
        [sidecar],
        expected_eef_controller_profile=(
            EEF_CONTROLLER_VELOCITY_RECOVERY_CANDIDATE_PROFILE
        ),
    )
    assert aggregate["profile"] == (EEF_IK_CURRENT_VELOCITY_RECOVERY_CANDIDATE_PROFILE)
    assert (
        aggregate["episodes"][0]["current_joint_velocity_recovery"]
        == (safety["current_joint_velocity_recovery"])
    )
    controller_aggregate = {
        "profile": "polaris_eef_controller_repair_candidate_episode_aggregate_v1",
        "eef_controller_profile": (EEF_CONTROLLER_VELOCITY_RECOVERY_CANDIDATE_PROFILE),
        "initial": _candidate_controller_report(
            safety,
            initial=True,
            profile=EEF_CONTROLLER_VELOCITY_RECOVERY_CANDIDATE_PROFILE,
        ),
        "episodes": [
            {
                "episode_index": 0,
                "report": _candidate_controller_report(
                    safety,
                    initial=False,
                    profile=EEF_CONTROLLER_VELOCITY_RECOVERY_CANDIDATE_PROFILE,
                ),
            }
        ],
    }
    frame = validate_eef_runtime_frame(env, observation)
    frame["ik_safety_profile"] = EEF_IK_CURRENT_VELOCITY_RECOVERY_CANDIDATE_PROFILE
    path = tmp_path / "velocity-recovery-runtime.json"
    atomic_write_runtime_contract(
        path,
        eef_controller_profile=EEF_CONTROLLER_VELOCITY_RECOVERY_CANDIDATE_PROFILE,
        controller_repair_candidate=controller_aggregate,
        protocol=validate_ego_lap_runtime_protocol(env),
        frame=frame,
        ik_safety=aggregate,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == (
        EEF_RUNTIME_CONTRACT_VELOCITY_RECOVERY_SCHEMA_VERSION
    )
    drifted = copy.deepcopy(controller_aggregate)
    drifted["episodes"][0]["report"]["current_joint_velocity_recovery"]["counters"][
        "residual_events"
    ] = 1
    with pytest.raises(ValueError, match="recovery|counter"):
        atomic_write_runtime_contract(
            tmp_path / "velocity-recovery-runtime-drift.json",
            eef_controller_profile=(EEF_CONTROLLER_VELOCITY_RECOVERY_CANDIDATE_PROFILE),
            controller_repair_candidate=drifted,
            protocol=validate_ego_lap_runtime_protocol(env),
            frame=frame,
            ik_safety=aggregate,
        )


def test_v5_terminal_recovery_binds_exact_event_digest_and_guard() -> None:
    result = _episode_result(length=2, numerical_failure=True)
    safety = _episode_safety(
        length=2,
        numerical_failure=True,
        failure_substeps=3,
    )
    recovery = _predicted_terminal_velocity_recovery_report(apply_index=10)
    velocity = recovery["events"][0]["last"]["joint_velocity_rad_s"]
    safety["profile"] = EEF_IK_CURRENT_VELOCITY_RECOVERY_CANDIDATE_PROFILE
    safety["current_joint_velocity_abort"] = None
    safety["current_joint_velocity_recovery"] = recovery
    safety["maxima"]["abs_joint_vel_rad_s"] = [abs(value) for value in velocity]
    safety["counters"]["nonfinite_aborts"] = 0
    safety["counters"]["invariant_aborts"] = 1
    safety["guard_diagnostics"][0]["kind"] = "predicted_joint_hard_position_limit_abort"
    encoded = json.dumps(
        recovery["events"][0],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    result["numerical_failure_reason"] = (
        "DifferentialIKInvariantError: "
        f"{CURRENT_JOINT_VELOCITY_RECOVERY_ABORT_MESSAGES['predicted_hard_limit_abort']} "
        f"(evidence_sha256={digest})"
    )
    validated = validate_episode_safety_cadence(
        safety=safety,
        episode_result=result,
    )
    assert validated["abort_count"] == 1

    drifted = copy.deepcopy(result)
    drifted["numerical_failure_reason"] = drifted["numerical_failure_reason"].replace(
        digest,
        "0" * 64,
    )
    with pytest.raises(ValueError, match="digest binding drift"):
        validate_episode_safety_cadence(safety=safety, episode_result=drifted)


@pytest.mark.parametrize("post_recovery", [False, True])
def test_v5_lower_endpoint_overflow_survives_result_sidecar_and_runtime(
    tmp_path: Path,
    monkeypatch,
    post_recovery: bool,
) -> None:
    length = 3 if post_recovery else 2
    result = _episode_result(length=length, numerical_failure=True)
    safety = _episode_safety(
        length=length,
        numerical_failure=True,
        failure_substeps=3,
    )
    recovery = (
        _post_recovery_lower_endpoint_terminal_velocity_recovery_report()
        if post_recovery
        else _lower_endpoint_terminal_velocity_recovery_report(apply_index=10)
    )
    safety["profile"] = EEF_IK_CURRENT_VELOCITY_RECOVERY_CANDIDATE_PROFILE
    safety["current_joint_velocity_abort"] = None
    safety["current_joint_velocity_recovery"] = recovery
    safety["maxima"]["abs_joint_vel_rad_s"] = [
        max(abs(start), abs(last))
        for start, last in zip(
            recovery["events"][0]["start"]["joint_velocity_rad_s"],
            recovery["events"][0]["last"]["joint_velocity_rad_s"],
            strict=True,
        )
    ]
    safety["counters"]["nonfinite_aborts"] = 0
    safety["counters"]["invariant_aborts"] = 1
    safety["guard_diagnostics"][0]["kind"] = (
        "measured_velocity_recovery_lower_endpoint_transition_abort"
    )
    encoded = json.dumps(
        recovery["events"][0],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    result["numerical_failure_reason"] = (
        "DifferentialIKInvariantError: "
        f"{CURRENT_JOINT_VELOCITY_RECOVERY_ABORT_MESSAGES['lower_endpoint_transition_overflow_abort']} "
        f"(evidence_sha256={digest})"
    )
    assert (
        validate_episode_safety_cadence(
            safety=safety,
            episode_result=result,
        )["abort_count"]
        == 1
    )

    safety = _attach_candidate_stub_gripper(safety, length=length)
    dynamic = safety["gripper_runtime_dynamic"]
    dynamic["post_policy_step_samples"] = length - 1
    committed_apply_calls = safety["counters"]["apply_calls"] - 1
    dynamic["driver_target_slew"]["apply_calls"] = committed_apply_calls
    dynamic["driver_target_slew"]["endpoint_change_count"] = 2
    controller_report = _candidate_controller_report(
        safety,
        initial=False,
        profile=EEF_CONTROLLER_VELOCITY_RECOVERY_CANDIDATE_PROFILE,
    )
    controller_report["current_joint_velocity_recovery"] = copy.deepcopy(recovery)
    all_six_trace = _all_six_trace(
        episode=0,
        length=length,
        apply_calls=committed_apply_calls,
    )
    all_six_trace["numerical_failure"] = True
    terminal = _terminal_rollout(result)
    sidecar_path = tmp_path / "ik_safety" / "episode_000000.json"

    with monkeypatch.context() as patch:
        patch.setattr(
            runtime_contract_module,
            "validate_eef_gripper_static_contract",
            lambda value, **_kwargs: dict(value),
        )
        patch.setattr(
            runtime_contract_module,
            "validate_eef_gripper_dynamic_evidence",
            lambda value, **_kwargs: dict(value),
        )
        patch.setattr(
            runtime_contract_module,
            "validate_eef_controller_safety_evidence",
            lambda *_args, expected_profile, **_kwargs: eef_controller_profile(
                expected_profile
            ),
        )
        patch.setattr(
            runtime_contract_module,
            "validate_arm_failure_substep_trace",
            lambda value, **_kwargs: dict(value),
        )
        payload = atomic_write_episode_safety(
            sidecar_path,
            eef_controller_profile=(EEF_CONTROLLER_VELOCITY_RECOVERY_CANDIDATE_PROFILE),
            controller_repair_candidate=controller_report,
            arm_failure_substep_trace={},
            all_six_gripper_trace=all_six_trace,
            episode_index=0,
            episode_result=result,
            safety=safety,
            artifact_identity=_artifact_identity_for_terminal(result, terminal),
            terminal_rollout=terminal,
            expected_gripper_target_slew_profile=(
                EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
            ),
        )
        assert payload["schema_version"] == 7
        assert payload["safety"]["current_joint_velocity_recovery"] == recovery

        aggregate = aggregate_episode_safety(
            safety,
            [{**payload, "path": str(sidecar_path), "sha256": "8" * 64}],
            expected_eef_controller_profile=(
                EEF_CONTROLLER_VELOCITY_RECOVERY_CANDIDATE_PROFILE
            ),
        )
        initial_report = _candidate_controller_report(
            safety,
            initial=True,
            profile=EEF_CONTROLLER_VELOCITY_RECOVERY_CANDIDATE_PROFILE,
        )
        controller_aggregate = {
            "profile": ("polaris_eef_controller_repair_candidate_episode_aggregate_v1"),
            "eef_controller_profile": (
                EEF_CONTROLLER_VELOCITY_RECOVERY_CANDIDATE_PROFILE
            ),
            "initial": initial_report,
            "episodes": [{"episode_index": 0, "report": controller_report}],
        }
        env, observation = _runtime_fixture()
        frame = validate_eef_runtime_frame(env, observation)
        frame["ik_safety_profile"] = EEF_IK_CURRENT_VELOCITY_RECOVERY_CANDIDATE_PROFILE
        runtime_path = tmp_path / "polaris-runtime.json"
        atomic_write_runtime_contract(
            runtime_path,
            eef_controller_profile=(EEF_CONTROLLER_VELOCITY_RECOVERY_CANDIDATE_PROFILE),
            controller_repair_candidate=controller_aggregate,
            protocol=validate_ego_lap_runtime_protocol(env),
            frame=frame,
            ik_safety=aggregate,
        )
        runtime_payload = json.loads(runtime_path.read_text(encoding="utf-8"))
        runtime_recovery = runtime_payload["ik_safety"]["episodes"][0][
            "current_joint_velocity_recovery"
        ]
        assert runtime_recovery["events"][0]["end_reason"] == (
            "lower_endpoint_transition_overflow_abort"
        )
        assert runtime_recovery["counters"]["lower_endpoint_transition_aborts"] == 1
        assert (
            runtime_recovery["events"][0]["deferred_lower_endpoint_transition_count"]
            == 2
        )
        assert runtime_recovery["events"][0][
            "lower_endpoint_transition_overflow_context"
        ] == ("post_recovery_resume" if post_recovery else "active_recovery")

        drifted_safety = copy.deepcopy(safety)
        drifted_safety["gripper_runtime_dynamic"]["driver_target_slew"][
            "endpoint_change_count"
        ] = 1
        with pytest.raises(ValueError, match="endpoint-change cadence drift"):
            atomic_write_episode_safety(
                tmp_path / "endpoint-count-drift.json",
                eef_controller_profile=(
                    EEF_CONTROLLER_VELOCITY_RECOVERY_CANDIDATE_PROFILE
                ),
                controller_repair_candidate=controller_report,
                arm_failure_substep_trace={},
                all_six_gripper_trace=all_six_trace,
                episode_index=0,
                episode_result=result,
                safety=drifted_safety,
                artifact_identity=_artifact_identity_for_terminal(result, terminal),
                terminal_rollout=terminal,
                expected_gripper_target_slew_profile=(
                    EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
                ),
            )


@pytest.mark.parametrize("kind", ["current", "predicted"])
@pytest.mark.parametrize("post_recovery", [False, True])
def test_v5_hard_limit_collision_survives_result_sidecar_and_runtime(
    tmp_path: Path,
    monkeypatch,
    kind: str,
    post_recovery: bool,
) -> None:
    length = 3 if post_recovery else 2
    result = _episode_result(length=length, numerical_failure=True)
    safety = _episode_safety(
        length=length,
        numerical_failure=True,
        failure_substeps=3,
    )
    recovery = _hard_limit_collision_velocity_recovery_report(
        kind=kind,
        post_recovery=post_recovery,
    )
    terminal_event = recovery["events"][-1]
    safety["profile"] = EEF_IK_CURRENT_VELOCITY_RECOVERY_CANDIDATE_PROFILE
    safety["current_joint_velocity_abort"] = None
    safety["current_joint_velocity_recovery"] = recovery
    safety["maxima"]["abs_joint_vel_rad_s"] = [
        max(
            abs(event[snapshot_name]["joint_velocity_rad_s"][joint_index])
            for event in recovery["events"]
            for snapshot_name in ("start", "last")
        )
        for joint_index in range(7)
    ]
    safety["counters"]["nonfinite_aborts"] = 0
    safety["counters"]["current_joint_limit_aborts"] = int(kind == "current")
    safety["counters"]["invariant_aborts"] = int(kind == "predicted")
    safety["guard_diagnostics"][0]["kind"] = {
        "current": "current_joint_hard_position_limit_abort",
        "predicted": "predicted_joint_hard_position_limit_abort",
    }[kind]
    encoded = json.dumps(
        terminal_event,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    end_reason = terminal_event["end_reason"]
    result["numerical_failure_reason"] = (
        "DifferentialIKInvariantError: "
        f"{CURRENT_JOINT_VELOCITY_RECOVERY_ABORT_MESSAGES[end_reason]} "
        f"(evidence_sha256={digest})"
    )
    assert (
        validate_episode_safety_cadence(
            safety=safety,
            episode_result=result,
        )["abort_count"]
        == 1
    )

    safety = _attach_candidate_stub_gripper(safety, length=length)
    dynamic = safety["gripper_runtime_dynamic"]
    dynamic["post_policy_step_samples"] = length - 1
    committed_apply_calls = safety["counters"]["apply_calls"] - 1
    dynamic["driver_target_slew"]["apply_calls"] = committed_apply_calls
    dynamic["driver_target_slew"]["endpoint_change_count"] = 2
    controller_report = _candidate_controller_report(
        safety,
        initial=False,
        profile=EEF_CONTROLLER_VELOCITY_RECOVERY_CANDIDATE_PROFILE,
    )
    controller_report["current_joint_velocity_recovery"] = copy.deepcopy(recovery)
    all_six_trace = _all_six_trace(
        episode=0,
        length=length,
        apply_calls=committed_apply_calls,
    )
    all_six_trace["numerical_failure"] = True
    terminal = _terminal_rollout(result)
    sidecar_path = tmp_path / "ik_safety" / "episode_000000.json"

    with monkeypatch.context() as patch:
        patch.setattr(
            runtime_contract_module,
            "validate_eef_gripper_static_contract",
            lambda value, **_kwargs: dict(value),
        )
        patch.setattr(
            runtime_contract_module,
            "validate_eef_gripper_dynamic_evidence",
            lambda value, **_kwargs: dict(value),
        )
        patch.setattr(
            runtime_contract_module,
            "validate_eef_controller_safety_evidence",
            lambda *_args, expected_profile, **_kwargs: eef_controller_profile(
                expected_profile
            ),
        )
        patch.setattr(
            runtime_contract_module,
            "validate_arm_failure_substep_trace",
            lambda value, **_kwargs: dict(value),
        )
        payload = atomic_write_episode_safety(
            sidecar_path,
            eef_controller_profile=(EEF_CONTROLLER_VELOCITY_RECOVERY_CANDIDATE_PROFILE),
            controller_repair_candidate=controller_report,
            arm_failure_substep_trace={},
            all_six_gripper_trace=all_six_trace,
            episode_index=0,
            episode_result=result,
            safety=safety,
            artifact_identity=_artifact_identity_for_terminal(result, terminal),
            terminal_rollout=terminal,
            expected_gripper_target_slew_profile=(
                EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
            ),
        )
        assert payload["schema_version"] == 7
        assert payload["safety"]["current_joint_velocity_recovery"] == recovery

        aggregate = aggregate_episode_safety(
            safety,
            [{**payload, "path": str(sidecar_path), "sha256": "9" * 64}],
            expected_eef_controller_profile=(
                EEF_CONTROLLER_VELOCITY_RECOVERY_CANDIDATE_PROFILE
            ),
        )
        initial_report = _candidate_controller_report(
            safety,
            initial=True,
            profile=EEF_CONTROLLER_VELOCITY_RECOVERY_CANDIDATE_PROFILE,
        )
        controller_aggregate = {
            "profile": "polaris_eef_controller_repair_candidate_episode_aggregate_v1",
            "eef_controller_profile": (
                EEF_CONTROLLER_VELOCITY_RECOVERY_CANDIDATE_PROFILE
            ),
            "initial": initial_report,
            "episodes": [{"episode_index": 0, "report": controller_report}],
        }
        env, observation = _runtime_fixture()
        frame = validate_eef_runtime_frame(env, observation)
        frame["ik_safety_profile"] = EEF_IK_CURRENT_VELOCITY_RECOVERY_CANDIDATE_PROFILE
        runtime_path = tmp_path / "polaris-runtime.json"
        atomic_write_runtime_contract(
            runtime_path,
            eef_controller_profile=(EEF_CONTROLLER_VELOCITY_RECOVERY_CANDIDATE_PROFILE),
            controller_repair_candidate=controller_aggregate,
            protocol=validate_ego_lap_runtime_protocol(env),
            frame=frame,
            ik_safety=aggregate,
        )
        runtime_payload = json.loads(runtime_path.read_text(encoding="utf-8"))
        runtime_event = runtime_payload["ik_safety"]["episodes"][0][
            "current_joint_velocity_recovery"
        ]["events"][-1]
        assert runtime_event["end_reason"] == end_reason
        assert runtime_event["deferred_lower_endpoint_transition_count"] == 2
        assert runtime_event["lower_endpoint_transition_overflow_context"] == (
            "post_recovery_resume" if post_recovery else "active_recovery"
        )
        assert runtime_event["recovery_completed_apply_index"] == (
            17 if post_recovery else None
        )

        drifted_safety = copy.deepcopy(safety)
        drifted_safety["gripper_runtime_dynamic"]["driver_target_slew"][
            "endpoint_change_count"
        ] = 1
        with pytest.raises(ValueError, match="endpoint-change cadence drift"):
            atomic_write_episode_safety(
                tmp_path / "endpoint-count-drift.json",
                eef_controller_profile=(
                    EEF_CONTROLLER_VELOCITY_RECOVERY_CANDIDATE_PROFILE
                ),
                controller_repair_candidate=controller_report,
                arm_failure_substep_trace={},
                all_six_gripper_trace=all_six_trace,
                episode_index=0,
                episode_result=result,
                safety=drifted_safety,
                artifact_identity=_artifact_identity_for_terminal(result, terminal),
                terminal_rollout=terminal,
                expected_gripper_target_slew_profile=(
                    EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
                ),
            )


def test_v5_open_recovery_allows_unrelated_digest_abort_and_horizon_reset() -> None:
    result = _episode_result(length=2, numerical_failure=True)
    result["numerical_failure_reason"] = (
        "DifferentialIKNumericalError: unrelated terminal evidence "
        f"(evidence_sha256={'a' * 64})"
    )
    safety = _episode_safety(
        length=2,
        numerical_failure=True,
        failure_substeps=3,
    )
    recovery = _active_velocity_recovery_report(apply_index=9)
    velocity = recovery["events"][0]["last"]["joint_velocity_rad_s"]
    safety["profile"] = EEF_IK_CURRENT_VELOCITY_RECOVERY_CANDIDATE_PROFILE
    safety["current_joint_velocity_abort"] = None
    safety["current_joint_velocity_recovery"] = recovery
    safety["maxima"]["abs_joint_vel_rad_s"] = [abs(value) for value in velocity]
    assert (
        validate_episode_safety_cadence(
            safety=safety,
            episode_result=result,
        )["abort_count"]
        == 1
    )

    legacy = copy.deepcopy(safety)
    legacy["profile"] = EEF_IK_SAFETY_PROFILE
    legacy.pop("current_joint_velocity_recovery")
    with pytest.raises(ValueError, match="abort evidence is missing"):
        validate_episode_safety_cadence(safety=legacy, episode_result=result)

    completed_safety = _episode_safety(length=2)
    completed_recovery = _active_velocity_recovery_report(apply_index=15)
    completed_safety["profile"] = EEF_IK_CURRENT_VELOCITY_RECOVERY_CANDIDATE_PROFILE
    completed_safety["current_joint_velocity_abort"] = None
    completed_safety["current_joint_velocity_recovery"] = completed_recovery
    completed_safety["maxima"]["abs_joint_vel_rad_s"] = [
        abs(value)
        for value in completed_recovery["events"][0]["last"]["joint_velocity_rad_s"]
    ]
    assert (
        validate_episode_safety_cadence(
            safety=completed_safety,
            episode_result=_episode_result(length=2),
        )["abort_count"]
        == 0
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda aggregate: aggregate.__setitem__("episodes_completed", 1),
            "completed-episode count",
        ),
        (
            lambda aggregate: aggregate["counters"].__setitem__(
                "apply_calls", aggregate["counters"]["apply_calls"] + 1
            ),
            "counters disagree",
        ),
        (
            lambda aggregate: aggregate["maxima"][
                "raw_delta_joint_pos_rad"
            ].__setitem__(0, 1.0),
            "maxima disagree",
        ),
        (
            lambda aggregate: aggregate["episodes"][1].__setitem__("episode_index", 3),
            "indices are not contiguous",
        ),
    ],
)
def test_runtime_writer_recomputes_aggregate_from_episode_entries(
    tmp_path: Path, mutation, match
):
    for episode in range(2):
        _prepare_episode_transaction(tmp_path, episode=episode)
    sidecars = load_episode_safety_sidecars(tmp_path / "ik_safety", [0, 1])
    env, observation = _runtime_fixture()
    live = env.unwrapped.action_manager._terms["arm"].safety_report()
    aggregate = aggregate_episode_safety(live, sidecars)
    mutation(aggregate)

    with pytest.raises(ValueError, match=match):
        atomic_write_runtime_contract(
            tmp_path / "runtime-mutated.json",
            eef_controller_profile=EEF_CONTROLLER_BASELINE_PROFILE,
            controller_repair_candidate=_controller_aggregate(aggregate),
            protocol=validate_ego_lap_runtime_protocol(env),
            frame=validate_eef_runtime_frame(env, observation),
            ik_safety=aggregate,
        )


def test_prepared_sidecar_recovery_rejects_csv_and_trace_drift(tmp_path: Path):
    result = _episode_result(length=450)
    safety = _episode_safety(length=450)
    (tmp_path / "episode_0.mp4").write_bytes(b"complete-video")
    trace_dir = tmp_path / "policy_traces"
    trace_path = trace_dir / "episode_000000.jsonl"
    _write_completed_trace(trace_path, result)
    identity = build_episode_artifact_identity(
        run_folder=tmp_path,
        trace_path=trace_path,
        episode_result=result,
        video_probe_fn=_video_probe,
    )
    sidecar_path = tmp_path / "ik_safety" / "episode_000000.json"
    atomic_write_episode_safety(
        sidecar_path,
        eef_controller_profile=EEF_CONTROLLER_BASELINE_PROFILE,
        controller_repair_candidate=_baseline_controller_report(safety),
        arm_failure_substep_trace=None,
        all_six_gripper_trace=None,
        episode_index=0,
        episode_result=result,
        safety=safety,
        artifact_identity=identity,
        terminal_rollout=_terminal_rollout(result),
    )

    drifted_row = {**result, "progress": 0.5}
    with pytest.raises(ValueError, match="CSV row differs"):
        reconcile_episode_safety_transactions(
            pd.DataFrame([drifted_row]),
            directory=sidecar_path.parent,
            run_folder=tmp_path,
            trace_dir=trace_dir,
            expected_rollouts=50,
            expected_horizon=450,
            video_probe_fn=_video_probe,
        )

    trace_path.write_text(trace_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact identity drift"):
        reconcile_episode_safety_transactions(
            empty_eval_results(),
            directory=sidecar_path.parent,
            run_folder=tmp_path,
            trace_dir=trace_dir,
            expected_rollouts=50,
            expected_horizon=450,
            video_probe_fn=_video_probe,
        )


def test_current_velocity_abort_preserves_sidecar_and_runtime_evidence(
    tmp_path: Path, monkeypatch
):
    safety, result = _current_velocity_abort_safety()
    terminal = _terminal_rollout(result, failure_substeps=4)
    env, observation = _runtime_fixture()
    arm_term = env.unwrapped.action_manager._terms["arm"]
    arm_term.episode_safety_report = lambda episode_index: (
        safety
        if episode_index == 0
        else (_ for _ in ()).throw(AssertionError("unexpected episode index"))
    )
    arm_term.safety_report = lambda: safety
    assert validate_eef_runtime_safety(env) is safety
    with pytest.raises(TypeError, match="unexpected keyword argument 'report'"):
        validate_eef_runtime_safety(env, report=safety)
    arm_term.safety_report = lambda: (_ for _ in ()).throw(
        AssertionError("provided report must not be fetched again")
    )

    def validate_exact_returned_report(
        passed_env,
        *,
        require_gripper_runtime=False,
        report=None,
        expected_gripper_target_slew_profile=None,
    ):
        assert passed_env is env
        assert require_gripper_runtime is True
        assert report is safety
        assert expected_gripper_target_slew_profile == (
            "eef_binary_driver_target_slew_rate2p5_"
            "from_live_limit5_per_120hz_substep_v2"
        )
        return report

    with monkeypatch.context() as patch:
        patch.setattr(
            "polaris.eef_runtime_contract._validate_eef_runtime_safety_report",
            validate_exact_returned_report,
        )
        assert eef_episode_safety_report(env, 0) is safety
    cadence = validate_episode_safety_cadence(safety=safety, episode_result=result)
    assert cadence == {
        "apply_calls": 940,
        "expected_decimation": 8,
        "failed_policy_step": 117,
        "failed_physics_substep": 3,
        "abort_count": 1,
    }
    evidence = safety["current_joint_velocity_abort"]
    assert evidence["exceeded_joint_mask"] == [
        False,
        False,
        False,
        False,
        True,
        False,
        True,
    ]
    digest = current_joint_velocity_abort_evidence_sha256(evidence)
    assert len(digest) == 64
    assert result["numerical_failure_reason"].endswith(f"evidence_sha256={digest})")

    sidecar_path = tmp_path / "ik_safety" / "episode_000000.json"
    payload = atomic_write_episode_safety(
        sidecar_path,
        eef_controller_profile=EEF_CONTROLLER_BASELINE_PROFILE,
        controller_repair_candidate=_baseline_controller_report(safety),
        arm_failure_substep_trace=None,
        all_six_gripper_trace=None,
        episode_index=0,
        episode_result=result,
        safety=safety,
        artifact_identity=_artifact_identity_for_terminal(result, terminal),
        terminal_rollout=terminal,
    )
    assert payload["schema_version"] == EEF_SAFETY_SIDECAR_SCHEMA_VERSION == 6
    assert payload["safety"]["current_joint_velocity_abort"] == evidence
    aggregate = aggregate_episode_safety(
        safety,
        [
            {
                **payload,
                "path": str(sidecar_path),
                "sha256": "2" * 64,
            }
        ],
    )
    assert aggregate["episodes"][0]["current_joint_velocity_abort"] == evidence
    runtime_path = tmp_path / "polaris_runtime_contract.json"
    atomic_write_runtime_contract(
        runtime_path,
        eef_controller_profile=EEF_CONTROLLER_BASELINE_PROFILE,
        controller_repair_candidate=_controller_aggregate(aggregate),
        protocol=validate_ego_lap_runtime_protocol(env),
        frame=validate_eef_runtime_frame(env, observation),
        ik_safety=aggregate,
    )
    runtime_payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert runtime_payload["schema_version"] == EEF_RUNTIME_CONTRACT_SCHEMA_VERSION == 6

    runtime_sign_drift = copy.deepcopy(aggregate)
    runtime_abort = runtime_sign_drift["episodes"][0]["current_joint_velocity_abort"]
    runtime_abort["joint_velocity_rad_s"][0] *= -1.0
    with pytest.raises(ValueError, match="result/reason digest binding drift"):
        atomic_write_runtime_contract(
            tmp_path / "runtime-sign-drift.json",
            eef_controller_profile=EEF_CONTROLLER_BASELINE_PROFILE,
            controller_repair_candidate=_controller_aggregate(runtime_sign_drift),
            protocol=validate_ego_lap_runtime_protocol(env),
            frame=validate_eef_runtime_frame(env, observation),
            ik_safety=runtime_sign_drift,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda safety: safety["current_joint_velocity_abort"].__setitem__(
            "unexpected", True
        ),
        lambda safety: safety["current_joint_velocity_abort"].__setitem__(
            "profile", "wrong"
        ),
        lambda safety: safety["current_joint_velocity_abort"].__setitem__(
            "episode_index", 1
        ),
        lambda safety: safety["current_joint_velocity_abort"].__setitem__(
            "policy_step", 116
        ),
        lambda safety: safety["current_joint_velocity_abort"].__setitem__(
            "physics_substep", 2
        ),
        lambda safety: safety["current_joint_velocity_abort"].__setitem__(
            "joint_names", ["wrong"] * 7
        ),
        lambda safety: safety["current_joint_velocity_abort"].__setitem__(
            "joint_velocity_rad_s", [0.0] * 6
        ),
        lambda safety: safety["current_joint_velocity_abort"][
            "joint_velocity_rad_s"
        ].__setitem__(4, float("nan")),
        lambda safety: safety["current_joint_velocity_abort"][
            "joint_velocity_limit_rad_s"
        ].__setitem__(4, 9.0),
        lambda safety: safety["current_joint_velocity_abort"].__setitem__(
            "joint_velocity_limit_tolerance_rad_s", 1e-4
        ),
        lambda safety: safety["current_joint_velocity_abort"][
            "joint_velocity_limit_excess_rad_s"
        ].__setitem__(4, 0.0),
        lambda safety: safety["current_joint_velocity_abort"].__setitem__(
            "exceeded_joint_mask", [False] * 7
        ),
        lambda safety: safety["maxima"]["abs_joint_vel_rad_s"].__setitem__(4, 2.7),
        lambda safety: safety.__setitem__("guard_diagnostics", []),
        lambda safety: safety["guard_diagnostics"][0].__setitem__("physics_substep", 2),
        lambda safety: safety["counters"].__setitem__("invariant_aborts", 0),
    ],
)
def test_current_velocity_abort_rejects_every_schema_and_binding_mutation(mutation):
    safety, result = _current_velocity_abort_safety()
    mutation(safety)
    with pytest.raises(ValueError, match="current-velocity|counter/diagnostic"):
        validate_episode_safety_cadence(safety=safety, episode_result=result)


@pytest.mark.parametrize("joint_index", range(7))
def test_current_velocity_abort_rejects_each_individual_signed_velocity_flip(
    joint_index,
):
    safety, result = _current_velocity_abort_safety()
    safety["current_joint_velocity_abort"]["joint_velocity_rad_s"][joint_index] *= -1.0

    with pytest.raises(ValueError, match="result/reason digest binding drift"):
        validate_episode_safety_cadence(safety=safety, episode_result=result)


@pytest.mark.parametrize(
    "reason_mutation",
    [
        lambda _reason: "DifferentialIKNumericalError: unrelated failure",
        lambda reason: f"{reason} drift",
        lambda reason: reason[:-65] + "0" * 64 + ")",
    ],
)
def test_current_velocity_abort_rejects_unrelated_or_drifted_failure_reason(
    reason_mutation,
):
    safety, result = _current_velocity_abort_safety()
    result["numerical_failure_reason"] = reason_mutation(
        result["numerical_failure_reason"]
    )

    with pytest.raises(ValueError, match="result/reason digest binding drift"):
        validate_episode_safety_cadence(safety=safety, episode_result=result)


def test_current_velocity_abort_digest_has_reproducible_consumer_canonicalization():
    safety, _result = _current_velocity_abort_safety()
    evidence = safety["current_joint_velocity_abort"]
    producer_digest = current_joint_velocity_abort_evidence_sha256(evidence)
    reordered = dict(reversed(list(evidence.items())))
    consumer_encoding = json.dumps(
        reordered,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    consumer_digest = hashlib.sha256(consumer_encoding).hexdigest()

    assert current_joint_velocity_abort_evidence_sha256(reordered) == producer_digest
    assert consumer_digest == producer_digest
    assert producer_digest == (
        "f7c69e1fa8ae3a36cdd17ad511de65ba1795f2ee3391d3ddb181d2369602d86d"
    )
    with pytest.raises(ValueError, match="not canonical JSON"):
        current_joint_velocity_abort_evidence_sha256(
            {**evidence, "joint_velocity_rad_s": [float("nan")] * 7}
        )


@pytest.mark.parametrize("threshold_joint_index", [0, 4])
def test_current_velocity_abort_validator_uses_direct_float32_threshold(
    threshold_joint_index,
):
    safety, result = _current_velocity_abort_safety()
    evidence = safety["current_joint_velocity_abort"]
    limits = np.asarray(evidence["joint_velocity_limit_rad_s"], dtype=np.float32)
    threshold = limits + np.float32(JOINT_VELOCITY_LIMIT_TOLERANCE_RAD_S)
    velocity = np.zeros(7, dtype=np.float32)
    velocity[threshold_joint_index] = threshold[threshold_joint_index]
    trigger_index = 1 if threshold_joint_index == 0 else 6
    velocity[trigger_index] = np.nextafter(threshold[trigger_index], np.float32(np.inf))
    excess = np.maximum(np.abs(velocity) - limits, np.float32(0.0))
    mask = np.abs(velocity) > threshold
    evidence["joint_velocity_rad_s"] = velocity.tolist()
    evidence["joint_velocity_limit_excess_rad_s"] = excess.tolist()
    evidence["exceeded_joint_mask"] = mask.tolist()
    safety["maxima"]["abs_joint_vel_rad_s"] = np.abs(velocity).tolist()
    result["numerical_failure_reason"] = (
        "DifferentialIKInvariantError: "
        f"{format_current_joint_velocity_abort_message(evidence)}"
    )

    cadence = validate_episode_safety_cadence(
        safety=safety,
        episode_result=result,
    )

    assert evidence["exceeded_joint_mask"][threshold_joint_index] is False
    assert evidence["exceeded_joint_mask"][trigger_index] is True
    assert (
        evidence["joint_velocity_limit_excess_rad_s"][threshold_joint_index]
        > JOINT_VELOCITY_LIMIT_TOLERANCE_RAD_S
    )
    assert cadence["failed_physics_substep"] == 3


def test_current_velocity_abort_requires_bidirectional_guard_maxima_and_failure():
    safety, result = _current_velocity_abort_safety()
    safety["current_joint_velocity_abort"] = None
    with pytest.raises(ValueError, match="current-velocity abort evidence is missing"):
        validate_episode_safety_cadence(safety=safety, episode_result=result)

    safety, result = _current_velocity_abort_safety()
    completed = {**result, "numerical_failure": False, "numerical_failure_reason": ""}
    with pytest.raises(ValueError, match="result/reason digest binding drift"):
        validate_episode_safety_cadence(safety=safety, episode_result=completed)

    safety, result = _current_velocity_abort_safety()
    safety["guard_diagnostics"][0]["kind"] = "nonfinite_abort"
    safety["counters"]["invariant_aborts"] = 0
    safety["counters"]["nonfinite_aborts"] = 1
    with pytest.raises(ValueError, match="current-velocity guard binding"):
        validate_episode_safety_cadence(safety=safety, episode_result=result)


def test_episode_safety_cadence_binds_success_and_failure_substep():
    completed = validate_episode_safety_cadence(
        safety=_episode_safety(), episode_result=_episode_result()
    )
    assert completed["apply_calls"] == 16
    assert completed["failed_policy_step"] is None

    failed = validate_episode_safety_cadence(
        safety=_episode_safety(numerical_failure=True),
        episode_result=_episode_result(numerical_failure=True),
    )
    assert failed["apply_calls"] == 11
    assert failed["failed_policy_step"] == 1
    assert failed["failed_physics_substep"] == 2

    contained_guard_abort = _episode_safety(numerical_failure=True)
    contained_guard_abort["counters"]["nonfinite_aborts"] = 0
    contained_guard_abort["counters"]["invariant_aborts"] = 1
    contained_guard_abort["counters"]["post_clamp_target_violations"] = 1
    contained_guard_abort["maxima"]["post_clamp_target_guard_band_violation_rad"][0] = (
        5e-6
    )
    diagnostic = json.loads(
        json.dumps(contained_guard_abort["max_raw_delta_diagnostic"])
    )
    diagnostic.update(
        {
            "kind": "post_clamp_position_invariant_abort",
            "policy_step": 1,
            "physics_substep": 2,
        }
    )
    contained_guard_abort["guard_diagnostics"] = [diagnostic]
    contained = validate_episode_safety_cadence(
        safety=contained_guard_abort,
        episode_result=_episode_result(numerical_failure=True),
    )
    assert contained["abort_count"] == 1
    assert contained["failed_policy_step"] == 1

    invalid = _episode_safety()
    invalid["counters"]["apply_calls"] -= 1
    invalid["counters"]["environment_substeps"] -= 1
    with pytest.raises(ValueError, match="Completed episode controller cadence"):
        validate_episode_safety_cadence(
            safety=invalid, episode_result=_episode_result()
        )


def test_episode_cadence_cross_binds_gripper_target_slew_action_order(monkeypatch):
    observed_profiles = []

    def validate_stub(value, **kwargs):
        observed_profiles.append(kwargs.get("expected_target_slew_profile"))
        return value

    monkeypatch.setattr(
        runtime_contract_module,
        "validate_eef_gripper_static_contract",
        validate_stub,
    )
    monkeypatch.setattr(
        runtime_contract_module,
        "validate_eef_gripper_dynamic_evidence",
        validate_stub,
    )

    completed_safety = _attach_stub_gripper_runtime(
        _episode_safety(),
        post_policy_step_samples=2,
        target_process_calls=2,
        target_apply_calls=16,
    )
    completed = validate_episode_safety_cadence(
        safety=completed_safety,
        episode_result=_episode_result(),
    )
    assert completed["apply_calls"] == 16
    assert observed_profiles == ["stub-target-slew"] * 3
    observed_profiles.clear()

    # ActionManager applies arm before finger. On an arm abort at the third
    # zero-indexed substep, arm evidence includes the failed 11th apply entry,
    # while the finger target setter completed only the prior 10 applies.
    failed_safety = _attach_stub_gripper_runtime(
        _episode_safety(numerical_failure=True),
        post_policy_step_samples=1,
        target_process_calls=2,
        target_apply_calls=10,
    )
    failed = validate_episode_safety_cadence(
        safety=failed_safety,
        episode_result=_episode_result(numerical_failure=True),
    )
    assert failed["failed_physics_substep"] == 2
    assert observed_profiles == ["stub-target-slew"] * 3
    observed_profiles.clear()

    drifted = copy.deepcopy(failed_safety)
    drifted["gripper_runtime_dynamic"]["driver_target_slew"]["apply_calls"] = 11
    with pytest.raises(ValueError, match="target-slew cadence mismatch"):
        validate_episode_safety_cadence(
            safety=drifted,
            episode_result=_episode_result(numerical_failure=True),
        )

    drifted = copy.deepcopy(completed_safety)
    drifted["gripper_runtime_dynamic"]["driver_target_slew"]["process_action_calls"] = 1
    with pytest.raises(ValueError, match="target-slew cadence mismatch"):
        validate_episode_safety_cadence(
            safety=drifted,
            episode_result=_episode_result(),
        )


def test_episode_safety_rejects_schema_counter_and_unknown_abort_tamper():
    extra = _episode_safety()
    extra["unexpected"] = True
    with pytest.raises(ValueError, match="schema drift"):
        validate_episode_safety_cadence(safety=extra, episode_result=_episode_result())

    mismatched = _episode_safety(numerical_failure=True)
    mismatched["counters"]["nonfinite_aborts"] = 2
    with pytest.raises(ValueError, match="counter history is impossible"):
        validate_episode_safety_cadence(
            safety=mismatched,
            episode_result=_episode_result(numerical_failure=True),
        )

    unknown = _episode_safety(numerical_failure=True)
    unknown["guard_diagnostics"][0]["kind"] = "unknown_abort"
    with pytest.raises(ValueError, match="kind is not allowed"):
        validate_episode_safety_cadence(
            safety=unknown,
            episode_result=_episode_result(numerical_failure=True),
        )

    dropped = _episode_safety()
    dropped["counters"]["guard_diagnostics_dropped"] = 1
    with pytest.raises(ValueError, match="dropped durable"):
        validate_episode_safety_cadence(
            safety=dropped, episode_result=_episode_result()
        )


def test_completed_episode_requires_consistent_max_raw_diagnostic():
    missing = _episode_safety()
    missing["max_raw_delta_diagnostic"] = None
    with pytest.raises(ValueError, match="lacks max-raw-delta"):
        validate_episode_safety_cadence(
            safety=missing, episode_result=_episode_result()
        )

    inconsistent = _episode_safety()
    inconsistent["maxima"]["raw_delta_joint_pos_rad"][0] = 0.25
    with pytest.raises(ValueError, match="disagrees with maxima"):
        validate_episode_safety_cadence(
            safety=inconsistent, episode_result=_episode_result()
        )

    incomplete_vectors = _episode_safety()
    incomplete_vectors["max_raw_delta_diagnostic"]["joint_pos_rad"] = None
    with pytest.raises(ValueError, match="max-raw diagnostic is non-finite"):
        validate_episode_safety_cadence(
            safety=incomplete_vectors, episode_result=_episode_result()
        )

    guard_band_max = _episode_safety()
    guard_band_max["maxima"]["post_clamp_target_guard_band_violation_rad"][0] = 2e-5
    guard_band_max["maxima"]["current_joint_soft_limit_violation_rad"][0] = 2e-5
    with pytest.raises(ValueError, match="guard-band recovery tolerance"):
        validate_episode_safety_cadence(
            safety=guard_band_max, episode_result=_episode_result()
        )

    unattributed_recovery = _episode_safety()
    unattributed_recovery["maxima"]["post_clamp_target_guard_band_violation_rad"][0] = (
        5e-6
    )
    with pytest.raises(ValueError, match="not attributable"):
        validate_episode_safety_cadence(
            safety=unattributed_recovery, episode_result=_episode_result()
        )

    guard_band_target = _episode_safety()
    guard_band_target["max_raw_delta_diagnostic"]["safe_joint_pos_target_rad"][
        "values"
    ][0] = guard_band_target["target_joint_pos_limits_rad"][0][1] + 2e-5
    with pytest.raises(ValueError, match="guard-band recovery tolerance"):
        validate_episode_safety_cadence(
            safety=guard_band_target, episode_result=_episode_result()
        )


def test_episode_safety_rejects_impossible_counter_and_diagnostic_history():
    impossible_events = _episode_safety()
    impossible_events["counters"]["slew_limit_events"] = 17
    impossible_events["counters"]["slew_limited_joints"] = 17
    with pytest.raises(ValueError, match="history is impossible"):
        validate_episode_safety_cadence(
            safety=impossible_events, episode_result=_episode_result()
        )

    multiple_abort = _episode_safety(numerical_failure=True)
    duplicate = json.loads(json.dumps(multiple_abort["guard_diagnostics"][0]))
    multiple_abort["guard_diagnostics"].append(duplicate)
    multiple_abort["counters"]["nonfinite_aborts"] = 2
    with pytest.raises(ValueError, match="history is impossible|exactly one"):
        validate_episode_safety_cadence(
            safety=multiple_abort,
            episode_result=_episode_result(numerical_failure=True),
        )

    out_of_order = _episode_safety(numerical_failure=True)
    fallback = json.loads(json.dumps(out_of_order["guard_diagnostics"][0]))
    fallback["kind"] = "dls_pseudoinverse_fallback"
    fallback["policy_step"] = 0
    fallback["physics_substep"] = 0
    out_of_order["guard_diagnostics"].append(fallback)
    out_of_order["counters"]["dls_fallbacks"] = 1
    with pytest.raises(ValueError, match="out of order"):
        validate_episode_safety_cadence(
            safety=out_of_order,
            episode_result=_episode_result(numerical_failure=True),
        )


def test_episode_sidecar_strict_json_rejects_nan(tmp_path: Path):
    result = _episode_result(length=450)
    terminal = _terminal_rollout(result)
    safety = _episode_safety(length=450)
    safety["maxima"]["raw_delta_joint_pos_rad"][0] = float("nan")
    with pytest.raises(ValueError, match="maximum raw_delta_joint_pos_rad is invalid"):
        atomic_write_episode_safety(
            tmp_path / "episode_000000.json",
            eef_controller_profile=EEF_CONTROLLER_BASELINE_PROFILE,
            controller_repair_candidate=_baseline_controller_report(safety),
            arm_failure_substep_trace=None,
            all_six_gripper_trace=None,
            episode_index=0,
            episode_result=result,
            safety=safety,
            artifact_identity={
                "video": {
                    "filename": "episode_0.mp4",
                    "size_bytes": 1,
                    "sha256": "0" * 64,
                    "frame_count": 450,
                    "height": 224,
                    "width": 448,
                },
                "terminal_trace": {
                    "filename": "episode_000000.jsonl",
                    "size_bytes": 1,
                    "sha256": "0" * 64,
                    "schema_version": 2,
                    "trace_profile": EGO_LAP_TRACE_PROFILE,
                    "episode_result": result,
                    "terminal_rollout": terminal,
                },
            },
            terminal_rollout=terminal,
        )
    assert not (tmp_path / "episode_000000.json").exists()


def test_episode_sidecar_binds_failure_sim_tail_to_apply_calls(tmp_path: Path):
    result = _episode_result(length=2, numerical_failure=True)
    safety = _episode_safety(length=2, numerical_failure=True)
    terminal = _terminal_rollout(result)
    identity = {
        "video": {
            "filename": "episode_0.mp4",
            "size_bytes": 1,
            "sha256": "0" * 64,
            "frame_count": 2,
            "height": 224,
            "width": 448,
        },
        "terminal_trace": {
            "filename": "episode_000000.jsonl",
            "size_bytes": 1,
            "sha256": "0" * 64,
            "schema_version": 2,
            "trace_profile": EGO_LAP_TRACE_PROFILE,
            "episode_result": result,
            "terminal_rollout": terminal,
        },
    }
    atomic_write_episode_safety(
        tmp_path / "valid.json",
        eef_controller_profile=EEF_CONTROLLER_BASELINE_PROFILE,
        controller_repair_candidate=_baseline_controller_report(safety),
        arm_failure_substep_trace=None,
        all_six_gripper_trace=None,
        episode_index=0,
        episode_result=result,
        safety=safety,
        artifact_identity=identity,
        terminal_rollout=terminal,
    )
    assert terminal["counter_deltas"]["sim_step_counter"] == 11

    mismatch = copy.deepcopy(terminal)
    mismatch["environment_after"]["sim_step_counter"] += 1
    mismatch["counter_deltas"]["sim_step_counter"] += 1
    mismatch_identity = copy.deepcopy(identity)
    mismatch_identity["terminal_trace"]["terminal_rollout"] = mismatch
    with pytest.raises(ValueError, match="sim-counter/apply-call binding"):
        atomic_write_episode_safety(
            tmp_path / "mismatch.json",
            eef_controller_profile=EEF_CONTROLLER_BASELINE_PROFILE,
            controller_repair_candidate=_baseline_controller_report(safety),
            arm_failure_substep_trace=None,
            all_six_gripper_trace=None,
            episode_index=0,
            episode_result=result,
            safety=safety,
            artifact_identity=mismatch_identity,
            terminal_rollout=mismatch,
        )


def test_runtime_frame_rejects_observation_and_controller_drift():
    env, observation = _runtime_fixture()
    observation["policy"]["eef_pos"] = observation["policy"]["eef_pos"].copy()
    observation["policy"]["eef_pos"][0, 0] += 0.01
    with pytest.raises(ValueError, match="direct panda_link0->panda_link8"):
        validate_eef_runtime_frame(env, observation)

    env, observation = _runtime_fixture()
    env.unwrapped.action_manager._terms["arm"].cfg.body_name = "base_link"
    with pytest.raises(ValueError, match="does not control physical panda_link8"):
        validate_eef_runtime_frame(env, observation)


def test_runtime_frame_rejects_nonidentity_offset_and_relative_mode():
    env, observation = _runtime_fixture()
    env.unwrapped.action_manager._terms["arm"].cfg.body_offset.pos = (0.0, 0.0, 0.01)
    with pytest.raises(ValueError, match="offset is not identity"):
        validate_eef_runtime_frame(env, observation)

    env, observation = _runtime_fixture()
    env.unwrapped.action_manager._terms["arm"].cfg.controller.use_relative_mode = True
    with pytest.raises(ValueError, match="not absolute pose"):
        validate_eef_runtime_frame(env, observation)


def test_runtime_safety_rejects_drift_and_unbounded_applied_delta():
    env, _ = _runtime_fixture()
    original_reporter = env.unwrapped.action_manager._terms["arm"].safety_report
    drifted = original_reporter()
    drifted["apply_actions_cadence"] = "policy_step"
    env.unwrapped.action_manager._terms["arm"].safety_report = lambda: drifted
    with pytest.raises(ValueError, match="apply_actions_cadence"):
        validate_eef_runtime_safety(env)

    for field, value in (
        ("articulation_solver_profile", "wrong"),
        ("articulation_solver_readback", "wrong"),
        ("physx_solver_type", 0),
        ("solver_position_iteration_count", 32),
        ("solver_velocity_iteration_count", 0),
    ):
        env, _ = _runtime_fixture()
        drifted = env.unwrapped.action_manager._terms["arm"].safety_report()
        drifted[field] = value
        env.unwrapped.action_manager._terms["arm"].safety_report = lambda: drifted
        with pytest.raises(ValueError, match=field):
            validate_eef_runtime_safety(env)

    env, _ = _runtime_fixture()
    unsafe = env.unwrapped.action_manager._terms["arm"].safety_report()
    unsafe["maxima"]["applied_delta_joint_pos_rad"][0] = 1.0
    env.unwrapped.action_manager._terms["arm"].safety_report = lambda: unsafe
    with pytest.raises(ValueError, match="exceeds its physics-substep bound"):
        validate_eef_runtime_safety(env)

    env, _ = _runtime_fixture()
    tampered = env.unwrapped.action_manager._terms["arm"].safety_report()
    tampered["soft_joint_pos_limits_rad"][0][0] += 1e-3
    tampered["soft_joint_pos_limits_float32_sha256"] = hashlib.sha256(
        np.asarray(tampered["soft_joint_pos_limits_rad"], dtype="<f4").tobytes()
    ).hexdigest()
    env.unwrapped.action_manager._terms["arm"].safety_report = lambda: tampered
    with pytest.raises(ValueError, match="canonical Panda float32"):
        validate_eef_runtime_safety(env)

    env, _ = _runtime_fixture()
    tampered = env.unwrapped.action_manager._terms["arm"].safety_report()
    tampered["target_joint_pos_limits_rad"][4][1] += 1e-3
    env.unwrapped.action_manager._terms["arm"].safety_report = lambda: tampered
    with pytest.raises(ValueError, match="target guard-band limits"):
        validate_eef_runtime_safety(env)

    env, _ = _runtime_fixture()
    tampered = env.unwrapped.action_manager._terms["arm"].safety_report()
    tampered["target_joint_pos_limits_float32_sha256"] = "0" * 64
    env.unwrapped.action_manager._terms["arm"].safety_report = lambda: tampered
    with pytest.raises(ValueError, match="target guard-band digest"):
        validate_eef_runtime_safety(env)

    env, _ = _runtime_fixture()
    tampered = env.unwrapped.action_manager._terms["arm"].safety_report()
    tampered["physx_derived_soft_joint_pos_limits_rad"][3][1] = tampered[
        "physx_hard_joint_pos_limits_rad"
    ][3][1]
    env.unwrapped.action_manager._terms["arm"].safety_report = lambda: tampered
    with pytest.raises(ValueError, match="PhysX-derived soft-limit readback"):
        validate_eef_runtime_safety(env)

    env, _ = _runtime_fixture()
    tampered = env.unwrapped.action_manager._terms["arm"].safety_report()
    tampered["physx_derived_soft_joint_pos_limits_float32_sha256"] = "0" * 64
    env.unwrapped.action_manager._terms["arm"].safety_report = lambda: tampered
    with pytest.raises(ValueError, match="PhysX-derived soft-limit readback"):
        validate_eef_runtime_safety(env)

    env, _ = _runtime_fixture()
    tampered = env.unwrapped.action_manager._terms["arm"].safety_report()
    tampered["arm_velocity_target_rad_s"][4] = 1e-3
    env.unwrapped.action_manager._terms["arm"].safety_report = lambda: tampered
    with pytest.raises(ValueError, match="velocity target must be exactly zero"):
        validate_eef_runtime_safety(env)

    env, _ = _runtime_fixture()
    tampered = env.unwrapped.action_manager._terms["arm"].safety_report()
    tampered["target_soft_limit_margin_rad"][0] = np.nextafter(
        np.float32(tampered["target_soft_limit_margin_rad"][0]),
        np.float32(np.inf),
    ).item()
    env.unwrapped.action_manager._terms["arm"].safety_report = lambda: tampered
    with pytest.raises(ValueError, match="must exactly equal"):
        validate_eef_runtime_safety(env)


def test_runtime_safety_uses_one_exact_float32_slew_tolerance():
    env, _ = _runtime_fixture()
    report = env.unwrapped.action_manager._terms["arm"].safety_report()
    bound = report["max_delta_joint_pos_rad"][0]
    report["maxima"]["applied_delta_joint_pos_rad"][0] = (
        bound + JOINT_SLEW_FLOAT32_TOLERANCE_RAD
    )
    env.unwrapped.action_manager._terms["arm"].safety_report = lambda: report
    validate_eef_runtime_safety(env)

    report["maxima"]["applied_delta_joint_pos_rad"][0] = (
        bound + 2 * JOINT_SLEW_FLOAT32_TOLERANCE_RAD
    )
    with pytest.raises(ValueError, match="exceeds its physics-substep bound"):
        validate_eef_runtime_safety(env)


def test_runtime_safety_diagnostic_requires_strict_null_and_finite_mask():
    env, _ = _runtime_fixture()
    report = env.unwrapped.action_manager._terms["arm"].safety_report()
    report["episode_index"] = 0
    diagnostic = {
        "kind": "nonfinite_abort",
        "episode_index": 0,
        "policy_step": 3,
        "physics_substep": 2,
        "joint_pos_rad": {
            "values": [0.0, None, 0.0, 0.0, 0.0, 0.0, 0.0],
            "finite_mask": [True, False, True, True, True, True, True],
            "finite_count": 6,
        },
        "raw_delta_joint_pos_rad": None,
        "raw_joint_pos_target_rad": None,
        "safe_joint_pos_target_rad": None,
        "pose_error_norm": None,
        "jacobian_finite": False,
        "jacobian_max_abs": None,
        "eef_quaternion_norm": None,
    }
    report["guard_diagnostics"] = [diagnostic]
    report["counters"]["nonfinite_aborts"] = 1
    env.unwrapped.action_manager._terms["arm"].safety_report = lambda: report
    validated = validate_eef_runtime_safety(env)
    json.dumps(validated, allow_nan=False)

    diagnostic["joint_pos_rad"]["values"][1] = 1.0
    with pytest.raises(ValueError, match="nonfinite value must be null"):
        validate_eef_runtime_safety(env)

    diagnostic["joint_pos_rad"]["values"][1] = None
    diagnostic["eef_quaternion_norm"] = float("nan")
    with pytest.raises(ValueError, match="eef_quaternion_norm is non-finite"):
        validate_eef_runtime_safety(env)


def test_runtime_safety_binds_exact_quaternion_unit_norm_tolerance():
    env, _ = _runtime_fixture()
    report = env.unwrapped.action_manager._terms["arm"].safety_report()
    validate_eef_runtime_safety(env)

    report["eef_quaternion_unit_norm_tolerance"] = (
        EEF_QUATERNION_UNIT_NORM_TOLERANCE * 2
    )
    env.unwrapped.action_manager._terms["arm"].safety_report = lambda: report
    with pytest.raises(ValueError, match="quaternion_tolerance"):
        validate_eef_runtime_safety(env)


def test_one_step_adversarial_smoke_requires_bounded_active_slew_guard():
    report = _episode_safety()
    report["counters"]["apply_calls"] = 8
    report["counters"]["environment_substeps"] = 8
    report["counters"]["slew_limit_events"] = 1
    report["counters"]["slew_limited_joints"] = 1
    report["maxima"]["applied_delta_joint_pos_rad"] = list(
        report["max_delta_joint_pos_rad"]
    )

    evidence = validate_one_step_adversarial_report(report)
    assert evidence["apply_calls"] == 8
    assert evidence["slew_limit_events"] == 1
    assert evidence["applied_within_bounds"] is True

    no_guard = json.loads(json.dumps(report))
    no_guard["counters"]["slew_limit_events"] = 0
    with pytest.raises(ValueError, match="did not activate"):
        validate_one_step_adversarial_report(no_guard)

    unsafe = json.loads(json.dumps(report))
    unsafe["maxima"]["applied_delta_joint_pos_rad"][0] += 2e-6
    with pytest.raises(ValueError, match="exceeded"):
        validate_one_step_adversarial_report(unsafe)

    smoke_source = (
        Path(__file__).parents[1] / "scripts" / "smoke_eef_pose_controller.py"
    ).read_text()
    for axis in "xyz":
        assert f'"translate +{axis}"' in smoke_source
        assert f'"translate -{axis}"' in smoke_source
        assert f'"rotate +{axis}"' in smoke_source
        assert f'"rotate -{axis}"' in smoke_source
    assert "robot.data.joint_pos[:, arm_term._joint_ids]" in smoke_source
    assert "robot.data.joint_vel[:, arm_term._joint_ids]" in smoke_source
    assert '"joint_state_is_finite"' in smoke_source
    assert '"joint_pos_rad"' in smoke_source
    assert '"joint_vel_rad_s"' in smoke_source
    assert '"max_abs"' in smoke_source
    assert '"position_within_captured_soft_limits"' in smoke_source
    assert "CURRENT_JOINT_SOFT_LIMIT_TOLERANCE_RAD" in smoke_source
    assert (
        smoke_source.index('state["raw_capture"] = initial_capture')
        < (
            smoke_source.index(
                "validated_initial_capture = validate_eef_runtime_safety(env)"
            )
        )
        < smoke_source.index("for case_index, (label, pose_delta)")
    )
    assert "os.link(temporary_path, path)" in smoke_source
    assert 'temporary_path.open("xb"' in smoke_source
    assert "path.chmod(0o444)" in smoke_source
    assert "os.fsync(directory_fd)" in smoke_source
    assert "allow_nan=False" in smoke_source
    assert 'required=True,\n    help="Required atomic' in smoke_source
    assert '"raw_ik_safety_capture"' in smoke_source
    assert '"stage": state["stage"]' in smoke_source
    assert '"case": state["case"]' in smoke_source
    assert 'formatted_traceback = "".join(' in smoke_source
    assert '"traceback": formatted_traceback' in smoke_source
    assert "except BaseException as run_error:" in smoke_source
    assert "except BaseException as close_error:" in smoke_source
    assert 'close_evidence["component"] = "environment"' in smoke_source
    assert "_result_payload(state, finalized=False" in smoke_source
    assert "_result_payload(state, finalized=True" not in smoke_source
    assert '"failure": state["failure"]' in smoke_source
    assert '"close_failures": state["close_failures"]' in smoke_source
    assert '"persistence_failures": state["persistence_failures"]' in smoke_source
    assert smoke_source.index("_print_exception(run_error)") < smoke_source.index(
        "simulation_app.close()"
    )
    pending_index = smoke_source.index(
        'state["stage"] = "simulation_app_close_pending"'
    )
    raw_write_index = smoke_source.index("_atomic_write_strict_json(", pending_index)
    prepared_index = smoke_source.index("POLARIS_SMOKE_RAW_PREPARED=", raw_write_index)
    marker_publish_index = smoke_source.index(
        "_atomic_write_strict_json(ready_marker, marker_payload)", prepared_index
    )
    simulation_close_index = smoke_source.index(
        "simulation_app.close()", marker_publish_index
    )
    assert (
        smoke_source.index("env.close()")
        < pending_index
        < raw_write_index
        < prepared_index
        < marker_publish_index
        < simulation_close_index
    )
    assert (
        "_atomic_write_strict_json(ready_marker, marker_payload)\n"
        "            simulation_app.close()"
    ) in smoke_source
    assert "POLARIS_SMOKE_RAW_READY=" not in smoke_source
    assert "POLARIS_SMOKE_RAW_PREPARED=" in smoke_source
    assert "POLARIS_SMOKE_RAW_SHA256=" in smoke_source
    assert "POLARIS_SMOKE_READY_MARKER_PATH=" in smoke_source
    assert "POLARIS_SMOKE_READY_MARKER_EXPECTED_SHA256=" in smoke_source
    assert "POLARIS_SIMULATION_APP_CLOSE_SKIPPED=raw_not_ready" in smoke_source
    assert "os._exit(1)" in smoke_source
    assert "def _best_effort_failure_log" in smoke_source
    assert (
        "else:\n        exit_code = 1\n        _best_effort_failure_log(\n"
        '            "POLARIS_SMOKE_RAW_FAILURE="'
    ) in smoke_source
    assert "finally:\n            os._exit(1)" in smoke_source
    assert "sys.stderr.flush()" in smoke_source


def test_smoke_raw_json_publication_is_strict_and_nonoverwriting(tmp_path):
    smoke_path = Path(__file__).parents[1] / "scripts" / "smoke_eef_pose_controller.py"
    parsed = ast.parse(smoke_path.read_text())
    helper_names = {
        "_strict_json_value",
        "_strict_json_bytes",
        "_atomic_write_strict_json",
        "_raw_is_eligible_for_close",
    }
    helper_nodes = [
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    assert {node.name for node in helper_nodes} == helper_names
    namespace = {"Path": Path, "json": json, "math": math, "os": os}
    exec(
        compile(ast.Module(helper_nodes, type_ignores=[]), str(smoke_path), "exec"),
        namespace,
    )
    writer = namespace["_atomic_write_strict_json"]
    eligible = namespace["_raw_is_eligible_for_close"]

    clean_state = {
        "stage": "simulation_app_close_pending",
        "case": None,
        "failure": None,
        "close_failures": [],
        "persistence_failures": [],
    }
    assert not eligible(
        clean_state, exit_code=0, raw_published=True, simulation_app=None
    )
    assert eligible(
        clean_state, exit_code=0, raw_published=True, simulation_app=object()
    )

    output = tmp_path / "raw.json"
    writer(output, {"value": 1.0, "nonfinite": math.inf})
    original = output.read_bytes()
    assert output.stat().st_mode & 0o777 == 0o444
    assert json.loads(original) == {"value": 1.0, "nonfinite": None}

    with pytest.raises(FileExistsError):
        writer(output, {"value": 2.0})
    assert output.read_bytes() == original
    assert list(tmp_path.glob("*.tmp")) == []


def test_smoke_failure_capture_preserves_live_arm_gripper_and_original_error():
    import torch

    smoke_path = Path(__file__).parents[1] / "scripts" / "smoke_eef_pose_controller.py"
    parsed = ast.parse(smoke_path.read_text())
    helper_names = {
        "_exception_evidence",
        "_strict_vector_evidence",
        "_capture_terminal_failure_evidence",
        "_arm_velocity_headroom_evidence",
    }
    helper_nodes = [
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    assert {node.name for node in helper_nodes} == helper_names
    namespace = {
        "math": math,
        "traceback": traceback,
        "CLOSE_ARM_VELOCITY_HEADROOM_MAX_RATIO": 0.95,
    }
    exec(
        compile(ast.Module(helper_nodes, type_ignores=[]), str(smoke_path), "exec"),
        namespace,
    )
    capture = namespace["_capture_terminal_failure_evidence"]

    arm_velocity = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]
    joint_vel = torch.zeros((1, 13), dtype=torch.float32)
    joint_vel[0, :7] = torch.tensor(arm_velocity)
    robot_data = SimpleNamespace(
        joint_pos=torch.arange(13, dtype=torch.float32).reshape(1, 13),
        joint_vel=joint_vel,
        joint_acc=torch.ones((1, 13), dtype=torch.float32),
        joint_pos_target=torch.arange(13, dtype=torch.float32).reshape(1, 13) / 10,
        joint_vel_target=torch.zeros((1, 13), dtype=torch.float32),
    )
    target_slew = {"profile": "target-slew", "apply_calls": 4}
    current_abort = {
        "joint_velocity_rad_s": [float(value) for value in joint_vel[0, :7].tolist()]
    }
    safety = {
        "episode_index": 0,
        "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
        "joint_velocity_limits_rad_s": [1.0] * 7,
        "maxima": {"abs_joint_vel_rad_s": [0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        "current_joint_velocity_abort": current_abort,
        "gripper_runtime_dynamic": {
            "joint_names": [f"gripper_{index}" for index in range(6)],
            "joint_indices": list(range(7, 13)),
            "driver_target_slew": target_slew,
        },
    }
    arm_term = SimpleNamespace(
        _joint_names=[f"panda_joint{index}" for index in range(1, 8)],
        _joint_ids=list(range(7)),
    )
    finger_term = SimpleNamespace(
        gripper_target_slew_dynamic_report=lambda: target_slew
    )
    runtime = SimpleNamespace(
        action_manager=SimpleNamespace(
            _terms={"arm": arm_term, "finger_joint": finger_term}
        ),
        scene={"robot": SimpleNamespace(data=robot_data)},
    )
    env = SimpleNamespace(unwrapped=runtime)
    original_failure = {"type": "OriginalSafetyAbort", "message": "do not mask"}
    state = {
        "stage": "execute_case",
        "case": "hold",
        "active_episode_index": 0,
        "env": env,
        "episode_safety_reporter": lambda _env, _episode: safety,
        "failure": original_failure,
    }
    evidence = capture(state)
    assert state["failure"] is original_failure
    assert evidence["status"] == "captured"
    assert evidence["safety_report"] is safety
    assert evidence["current_joint_velocity_abort"] is current_abort
    assert (
        evidence["arm_joint_velocity_rad_s"]["values"]
        == current_abort["joint_velocity_rad_s"]
    )
    assert evidence["driver_target_slew"] is target_slew
    assert len(evidence["all_six_gripper_state"]["joint_velocity_rad_s"]["values"]) == 6

    def failed_reporter(_env, _episode):
        raise RuntimeError("secondary capture failure")

    state["episode_safety_reporter"] = failed_reporter
    failed = capture(state)
    assert state["failure"] is original_failure
    assert failed["status"] == "capture_failed"
    assert failed["capture_error"]["message"] == "secondary capture failure"

    headroom = namespace["_arm_velocity_headroom_evidence"]
    boundary = headroom(safety, episode_index=0)
    assert boundary["maximum_ratio"] == 0.95
    assert boundary["passed"] is True
    safety["maxima"]["abs_joint_vel_rad_s"][0] = 0.950001
    assert headroom(safety, episode_index=0)["passed"] is False


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("ik_method", "pinv", "damped least-squares"),
        ("damping", 0.1, "DLS damping"),
        ("scale", 0.5, "action scale"),
        (
            "joint_names",
            list(reversed([f"panda_joint{i}" for i in range(1, 8)])),
            "joint order",
        ),
        ("gripper_profile", "closed_positive_gt_0p5", "gripper threshold semantics"),
    ],
)
def test_runtime_frame_rejects_controller_semantics_drift(field, value, match):
    env, observation = _runtime_fixture()
    arm = env.unwrapped.action_manager._terms["arm"]
    if field == "ik_method":
        arm.cfg.controller.ik_method = value
    elif field == "damping":
        arm.cfg.controller.ik_params["lambda_val"] = value
    elif field == "scale":
        arm.cfg.scale = value
    elif field == "joint_names":
        arm._joint_names = value
    else:
        env.unwrapped.action_manager._terms[
            "finger_joint"
        ].gripper_threshold_profile = value

    with pytest.raises(ValueError, match=match):
        validate_eef_runtime_frame(env, observation)
