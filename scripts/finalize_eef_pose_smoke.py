#!/usr/bin/env python3
"""Verify immutable PolaRiS smoke evidence and publish a separate attestation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
from typing import Any


EXPECTED_DIGEST = "fbf7535901c042fea5d901812ecd02c5fd81ade06c23c1499c32d66a859104de"
EXPECTED_TARGET_DIGEST = (
    "09b20ab18c35d6dc22a3edbc2beca2edff419e242dd07d74cd1d65df9ce67e0f"
)
EXPECTED_PHYSX_DERIVED_SOFT_DIGEST = (
    "dd7865f59efb23e96d7d4cbb5e129906b04a42b5e5c0941459bfc8866dd7ecd0"
)
EXPECTED_LIMITS = [
    [-2.8973000049591064, 2.8973000049591064],
    [-1.7627999782562256, 1.7627999782562256],
    [-2.8973000049591064, 2.8973000049591064],
    [-3.0717999935150146, -0.06979990005493164],
    [-2.8973000049591064, 2.8973000049591064],
    [-0.017499923706054688, 3.752500057220459],
    [-2.8973000049591064, 2.8973000049591064],
]
EXPECTED_CASES = [
    "hold",
    "translate +x",
    "translate -x",
    "translate +y",
    "translate -y",
    "translate +z",
    "translate -z",
    "rotate +x",
    "rotate -x",
    "rotate +y",
    "rotate -y",
    "rotate +z",
    "rotate -z",
]
ABORT_COUNTERS = (
    "current_joint_limit_aborts",
    "invariant_aborts",
    "nonfinite_aborts",
)
EXPECTED_JOINT_NAMES = [f"panda_joint{index}" for index in range(1, 8)]
EXPECTED_VELOCITY_LIMITS = [2.174999952316284] * 4 + [2.609999895095825] * 3
EXPECTED_EFFORT_LIMITS = [87.0] * 4 + [12.0] * 3
EXPECTED_MAX_DELTA = [0.018125001341104507] * 4 + [0.02174999937415123] * 3


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


EXPECTED_TARGET_LIMITS = [
    [_float32(lower + margin), _float32(upper - margin)]
    for (lower, upper), margin in zip(EXPECTED_LIMITS, EXPECTED_MAX_DELTA, strict=True)
]
EXPECTED_PHYSX_DERIVED_SOFT_LIMITS = [
    [-2.8791749477386475, 2.8791749477386475],
    [-1.7446749210357666, 1.7446749210357666],
    [-2.8791749477386475, 2.8791749477386475],
    [-3.0536749362945557, -0.08792495727539062],
    [-2.8755500316619873, 2.8755500316619873],
    [0.004250049591064453, 3.73075008392334],
    [-2.8755500316619873, 2.8755500316619873],
]
if (
    hashlib.sha256(
        b"".join(
            struct.pack("<f", value)
            for pair in EXPECTED_TARGET_LIMITS
            for value in pair
        )
    ).hexdigest()
    != EXPECTED_TARGET_DIGEST
):
    raise RuntimeError("Canonical Panda target guard-band digest drift")
if (
    hashlib.sha256(
        b"".join(
            struct.pack("<f", value)
            for pair in EXPECTED_PHYSX_DERIVED_SOFT_LIMITS
            for value in pair
        )
    ).hexdigest()
    != EXPECTED_PHYSX_DERIVED_SOFT_DIGEST
):
    raise RuntimeError("Canonical Panda PhysX-derived soft-limit digest drift")
RAW_FIELDS = {
    "schema_version",
    "finalized",
    "environment",
    "eef_frame",
    "hold_steps",
    "position_delta_m",
    "rotation_delta_deg",
    "position_tolerance_m",
    "rotation_tolerance_deg",
    "frame_position_tolerance_m",
    "frame_rotation_tolerance_deg",
    "stage",
    "case",
    "exit_code",
    "raw_ik_safety_capture",
    "ik_safety_episodes",
    "ik_safety_adversarial",
    "passed",
    "results",
    "gripper_delayed_close_replay",
    "gripper_close_velocity_headroom",
    "terminal_failure_evidence",
    "failure",
    "close_failures",
    "persistence_failures",
}
RESULT_FIELDS = {
    "case",
    "passed",
    "position_error_m",
    "rotation_error_rad",
    "target_position",
    "actual_position",
    "target_quaternion_wxyz",
    "actual_quaternion_wxyz",
    "reset_frame_position_error_m",
    "reset_frame_rotation_error_rad",
    "final_frame_position_error_m",
    "final_frame_rotation_error_rad",
}
SAFETY_FIELDS = {
    "episode_index",
    "profile",
    "apply_actions_cadence",
    "physics_dt",
    "control_dt",
    "decimation",
    "current_joint_soft_limit_tolerance_rad",
    "target_soft_limit_guard_band_profile",
    "physx_hard_limit_profile",
    "physx_derived_soft_limit_profile",
    "physx_hard_limit_write_count",
    "arm_velocity_target_profile",
    "articulation_solver_profile",
    "articulation_solver_readback",
    "physx_solver_type",
    "solver_position_iteration_count",
    "solver_velocity_iteration_count",
    "joint_velocity_limit_tolerance_rad_s",
    "eef_quaternion_unit_norm_tolerance",
    "joint_slew_float32_tolerance_rad",
    "soft_joint_pos_limit_factor",
    "joint_names",
    "joint_velocity_limits_rad_s",
    "joint_effort_limits",
    "max_delta_joint_pos_rad",
    "target_soft_limit_margin_rad",
    "target_joint_pos_limits_rad",
    "target_joint_pos_limits_float32_sha256",
    "physx_hard_joint_pos_limits_rad",
    "physx_hard_joint_pos_limits_float32_sha256",
    "physx_derived_soft_joint_pos_limits_rad",
    "physx_derived_soft_joint_pos_limits_float32_sha256",
    "arm_velocity_target_rad_s",
    "soft_joint_pos_limits_rad",
    "soft_joint_pos_limits_float32_sha256",
    "counters",
    "maxima",
    "guard_diagnostics",
    "max_raw_delta_diagnostic",
    "current_joint_velocity_abort",
    "gripper_runtime_static",
    "gripper_runtime_dynamic",
}
GRIPPER_RUNTIME_STATIC_FIELDS = {
    "profile",
    "joint_names",
    "gripper_joint_names",
    "gripper_joint_indices",
    "driver_joint_name",
    "driver_joint_index",
    "follower_joint_names",
    "follower_joint_indices",
    "actuator_joint_ownership",
    "device_partition",
    "driver_actuator",
    "mimic_joint_contract",
    "velocity_limits_before_write",
    "velocity_limits_after_write",
    "velocity_limit_write_contract",
    "driver_target_slew",
    "measured_velocity_is_hard_bounded_by_limit",
}
GRIPPER_RUNTIME_DYNAMIC_FIELDS = {
    "profile",
    "joint_names",
    "joint_indices",
    "apply_entry_samples",
    "post_policy_step_samples",
    "max_abs_joint_velocity_rad_s",
    "max_abs_joint_acceleration_rad_s2",
    "max_velocity_diagnostic",
    "terminal_state",
    "driver_target_slew",
    "nonfinite_samples",
    "dropped_diagnostics",
}
GRIPPER_TARGET_SLEW_STATIC_FIELDS = {
    "profile",
    "scope",
    "action_class",
    "driver_joint_name",
    "driver_joint_index",
    "endpoint_semantics_profile",
    "open_target_rad",
    "closed_target_rad",
    "physical_velocity_limit_source",
    "physical_velocity_limit_rad_s",
    "target_slew_rate_source",
    "target_slew_rate_factor",
    "target_slew_rate_rad_s",
    "physics_hz",
    "physics_dt",
    "max_target_step_rad",
    "float32_tolerance_rad",
    "reset_profile",
    "tensor_dtype",
    "tensor_device",
}
GRIPPER_TARGET_SLEW_DYNAMIC_FIELDS = {
    "profile",
    "process_action_calls",
    "apply_calls",
    "initialization_count",
    "endpoint_change_count",
    "repeated_endpoint_process_count",
    "slew_limited_apply_count",
    "endpoint_reached_apply_count",
    "live_limit_validation_count",
    "max_abs_target_step_rad",
    "max_abs_endpoint_error_before_step_rad",
    "max_abs_endpoint_error_after_step_rad",
    "initial_anchor_rad",
    "last_requested_endpoint_rad",
    "last_applied_target_rad",
}
GRIPPER_TENSOR_FIELDS = {
    "dtype",
    "device",
    "shape",
    "values",
    "finite_mask",
    "finite_count",
}
GRIPPER_RUNTIME_PROFILE = (
    "implicit_gripper_physx_velocity_limit5_followers5_"
    "cuda_actuator_cpu_static_physx_v1"
)
GRIPPER_TARGET_SLEW_PROFILE = (
    "eef_binary_driver_target_slew_rate2p5_from_live_limit5_per_120hz_substep_v2"
)
GRIPPER_TARGET_SLEW_ACTION_CLASS = "EefBinaryJointPositionTargetSlewAction"
GRIPPER_TARGET_SLEW_RESET_PROFILE = (
    "first_apply_after_action_reset_anchor_live_driver_position_v1"
)
GRIPPER_JOINT_NAMES = [
    "finger_joint",
    "right_outer_knuckle_joint",
    "left_inner_finger_joint",
    "right_inner_finger_joint",
    "left_inner_finger_knuckle_joint",
    "right_inner_finger_knuckle_joint",
]
EXPECTED_DROID_JOINT_NAMES = [*EXPECTED_JOINT_NAMES, *GRIPPER_JOINT_NAMES]
GRIPPER_PHYSICAL_VELOCITY_LIMIT = _float32(5.0)
GRIPPER_TARGET_SLEW_RATE_FACTOR = _float32(0.5)
GRIPPER_TARGET_SLEW_RATE = _float32(
    GRIPPER_PHYSICAL_VELOCITY_LIMIT * GRIPPER_TARGET_SLEW_RATE_FACTOR
)
GRIPPER_TARGET_SLEW_PHYSICS_DT_FLOAT32 = _float32(1.0 / 120.0)
GRIPPER_MAX_TARGET_STEP = _float32(
    GRIPPER_TARGET_SLEW_RATE * GRIPPER_TARGET_SLEW_PHYSICS_DT_FLOAT32
)
GRIPPER_CLOSED_TARGET = struct.unpack("<f", struct.pack("<f", math.pi / 4.0))[0]
GRIPPER_CLOSE_TRANSITION_APPLIES = math.ceil(
    GRIPPER_CLOSED_TARGET / GRIPPER_MAX_TARGET_STEP
)
GRIPPER_CLOSE_LIMITED_APPLIES = GRIPPER_CLOSE_TRANSITION_APPLIES - 1
if GRIPPER_CLOSE_TRANSITION_APPLIES != 38:
    raise RuntimeError("EEF gripper close transition count drift")
GRIPPER_MIN_ANCHOR = _float32(-_float32(1e-6))
GRIPPER_MAX_ANCHOR = _float32(GRIPPER_CLOSED_TARGET + _float32(1e-6))
COUNTER_FIELDS = {
    "apply_calls",
    "environment_substeps",
    "slew_limit_events",
    "slew_limited_joints",
    "position_limit_events",
    "position_limited_joints",
    "post_clamp_target_violations",
    "current_joint_limit_aborts",
    "invariant_aborts",
    "nonfinite_aborts",
    "dls_fallbacks",
    "guard_diagnostics_dropped",
}
MAXIMA_FIELDS = {
    "raw_delta_joint_pos_rad",
    "applied_delta_joint_pos_rad",
    "raw_target_soft_limit_violation_rad",
    "post_clamp_target_soft_limit_violation_rad",
    "post_clamp_target_guard_band_violation_rad",
    "current_joint_soft_limit_violation_rad",
    "current_physx_hard_limit_violation_rad",
    "abs_joint_vel_rad_s",
    "minimum_outer_joint_clearance_rad",
}
DIAGNOSTIC_FIELDS = {
    "kind",
    "episode_index",
    "policy_step",
    "physics_substep",
    "joint_pos_rad",
    "raw_delta_joint_pos_rad",
    "raw_joint_pos_target_rad",
    "safe_joint_pos_target_rad",
    "pose_error_norm",
    "jacobian_finite",
    "jacobian_max_abs",
    "eef_quaternion_norm",
}
DIAGNOSTIC_COUNTERS = {
    "current_joint_limit_abort": "current_joint_limit_aborts",
    "post_clamp_position_invariant_abort": "invariant_aborts",
    "post_clamp_slew_invariant_abort": "invariant_aborts",
    "current_eef_quaternion_invariant_abort": "invariant_aborts",
    "desired_eef_quaternion_invariant_abort": "invariant_aborts",
    "nonfinite_abort": "nonfinite_aborts",
    "dls_pseudoinverse_fallback": "dls_fallbacks",
}
ADVERSARIAL_FIELDS = {
    "case",
    "passed",
    "state_is_finite",
    "eef_state_is_finite",
    "joint_state_is_finite",
    "joint_pos_within_captured_soft_limits",
    "joint_state",
    "terminated",
    "truncated",
    "guard_evidence",
    "guard_error",
    "ik_safety",
}
DELAYED_CLOSE_FIELDS = {
    "profile",
    "case",
    "passed",
    "episode_index",
    "open_policy_steps",
    "close_policy_steps",
    "close_transition_substeps",
    "terminated",
    "truncated",
    "arm_abort_count",
    "ik_safety",
}
DELAYED_CLOSE_REPLAY_PROFILE = "eef_open115_then_close5_same_arm_pose_v1"
DELAYED_CLOSE_OPEN_POLICY_STEPS = 115
DELAYED_CLOSE_CLOSE_POLICY_STEPS = 5
DELAYED_CLOSE_APPLY_CALLS = (
    DELAYED_CLOSE_OPEN_POLICY_STEPS + DELAYED_CLOSE_CLOSE_POLICY_STEPS
) * 8
CLOSE_ARM_VELOCITY_HEADROOM_PROFILE = "arm_velocity_max_over_limit_le_0p95_v1"
CLOSE_ARM_VELOCITY_HEADROOM_MAX_RATIO = 0.95
CLOSE_HEADROOM_FIELDS = {
    "profile",
    "threshold_ratio",
    "passed",
    "immediate_close_hold",
    "delayed_close_replay",
}
CLOSE_HEADROOM_ENTRY_FIELDS = {
    "episode_index",
    "joint_names",
    "max_abs_joint_velocity_rad_s",
    "joint_velocity_limit_rad_s",
    "velocity_to_limit_ratio",
    "maximum_ratio",
    "threshold_ratio",
    "passed",
}


class VerificationError(ValueError):
    """Evidence or provenance does not satisfy the finalization contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _object(value: Any, field: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{field} must be an object")
    return value


def _typed_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _typed_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _typed_equal(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        )
    return actual == expected


def _list(value: Any, field: str, *, length: int | None = None) -> list[Any]:
    _require(isinstance(value, list), f"{field} must be an array")
    if length is not None:
        _require(len(value) == length, f"{field} must contain {length} entries")
    return value


def _finite_number(value: Any, field: str) -> float:
    _require(
        type(value) is float and math.isfinite(value),
        f"{field} must be a finite JSON float",
    )
    return value


def _exact_int(value: Any, expected: int, field: str) -> int:
    _require(
        type(value) is int and value == expected, f"{field} must be int {expected}"
    )
    return value


def _exact_float(value: Any, expected: float, field: str) -> float:
    actual = _finite_number(value, field)
    _require(actual == expected, f"{field} mismatch")
    return actual


def _finite_vector_evidence(value: Any, field: str) -> list[float]:
    evidence = _object(value, field)
    _require(
        set(evidence) == {"values", "finite_mask", "finite_count", "max_abs"},
        f"{field} schema drift",
    )
    values = _list(evidence.get("values"), f"{field}.values", length=7)
    mask = _list(evidence.get("finite_mask"), f"{field}.finite_mask", length=7)
    _require(mask == [True] * 7, f"{field}.finite_mask must be all true")
    _require(evidence.get("finite_count") == 7, f"{field}.finite_count must be 7")
    finite_values = [
        _finite_number(item, f"{field}.values[{index}]")
        for index, item in enumerate(values)
    ]
    maximum = _finite_number(evidence.get("max_abs"), f"{field}.max_abs")
    _require(
        math.isclose(
            maximum,
            max(abs(item) for item in finite_values),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        f"{field}.max_abs is inconsistent",
    )
    return finite_values


def _finite_vector(value: Any, field: str, *, length: int) -> list[float]:
    values = _list(value, field, length=length)
    return [
        _finite_number(item, f"{field}[{index}]") for index, item in enumerate(values)
    ]


def _exact_float_vector(value: Any, expected: list[float], field: str) -> list[float]:
    values = _finite_vector(value, field, length=len(expected))
    _require(values == expected, f"{field} mismatch")
    return values


def _normalize_quaternion_wxyz(quaternion: list[float]) -> list[float]:
    norm = math.sqrt(sum(item * item for item in quaternion))
    _require(norm > 0.0, "quaternion norm must be positive")
    return [item / norm for item in quaternion]


def _quaternion_angular_distance_wxyz(left: list[float], right: list[float]) -> float:
    left_unit = _normalize_quaternion_wxyz(left)
    right_unit = _normalize_quaternion_wxyz(right)
    dot = abs(sum(a * b for a, b in zip(left_unit, right_unit, strict=True)))
    return 2.0 * math.acos(min(1.0, max(0.0, dot)))


def _quaternion_multiply_wxyz(left: list[float], right: list[float]) -> list[float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return [
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ]


def _diagnostic_vector(value: Any, field: str) -> list[float | None] | None:
    if value is None:
        return None
    evidence = _object(value, field)
    _require(
        set(evidence) == {"values", "finite_mask", "finite_count"},
        f"{field} schema drift",
    )
    values = _list(evidence.get("values"), f"{field}.values", length=7)
    mask = _list(evidence.get("finite_mask"), f"{field}.finite_mask", length=7)
    _require(all(type(item) is bool for item in mask), f"{field}.finite_mask invalid")
    _require(
        type(evidence.get("finite_count")) is int
        and evidence["finite_count"] == sum(mask),
        f"{field}.finite_count invalid",
    )
    result: list[float | None] = []
    for index, (item, finite) in enumerate(zip(values, mask, strict=True)):
        if finite:
            result.append(_finite_number(item, f"{field}.values[{index}]"))
        else:
            _require(item is None, f"{field}.values[{index}] must be null")
            result.append(None)
    return result


def _validate_diagnostic(
    value: Any,
    field: str,
    *,
    episode_index: int,
    allowed_kinds: set[str],
    apply_calls: int,
) -> tuple[int, dict[str, Any]]:
    diagnostic = _object(value, field)
    _require(set(diagnostic) == DIAGNOSTIC_FIELDS, f"{field} schema drift")
    _require(
        type(diagnostic.get("kind")) is str and diagnostic["kind"] in allowed_kinds,
        f"{field} kind invalid",
    )
    _exact_int(diagnostic.get("episode_index"), episode_index, f"{field} episode")
    policy_step = diagnostic.get("policy_step")
    physics_substep = diagnostic.get("physics_substep")
    _require(type(policy_step) is int and policy_step >= 0, f"{field} policy_step")
    _require(
        type(physics_substep) is int and 0 <= physics_substep < 8,
        f"{field} physics_substep",
    )
    flattened = policy_step * 8 + physics_substep
    _require(flattened < apply_calls, f"{field} is out of cadence")
    for name in (
        "joint_pos_rad",
        "raw_delta_joint_pos_rad",
        "raw_joint_pos_target_rad",
        "safe_joint_pos_target_rad",
    ):
        _diagnostic_vector(diagnostic.get(name), f"{field}.{name}")
    for name in ("pose_error_norm", "jacobian_max_abs", "eef_quaternion_norm"):
        scalar = diagnostic.get(name)
        if scalar is not None:
            _require(
                _finite_number(scalar, f"{field}.{name}") >= 0.0, f"{field}.{name}"
            )
    jacobian_finite = diagnostic.get("jacobian_finite")
    _require(
        jacobian_finite is None or type(jacobian_finite) is bool,
        f"{field}.jacobian_finite",
    )
    return flattened, diagnostic


def _same_float32(actual: Any, expected: float) -> bool:
    return (
        type(actual) is float
        and math.isfinite(actual)
        and struct.pack("<f", actual) == struct.pack("<f", expected)
    )


def _validate_gripper_tensor(
    value: Any,
    *,
    field: str,
    expected: list[float],
    shape: list[int],
    device: str,
) -> dict[str, Any]:
    tensor = _object(value, field)
    _require(set(tensor) == GRIPPER_TENSOR_FIELDS, f"{field} schema drift")
    _require(tensor.get("dtype") == "torch.float32", f"{field} dtype")
    _require(tensor.get("device") == device, f"{field} device")
    _require(tensor.get("shape") == shape, f"{field} shape")
    values = _list(tensor.get("values"), f"{field}.values", length=len(expected))
    _require(
        tensor.get("finite_mask") == [True] * len(expected)
        and tensor.get("finite_count") == len(expected),
        f"{field} finite evidence",
    )
    _require(
        all(
            _same_float32(actual, wanted)
            for actual, wanted in zip(values, expected, strict=True)
        ),
        f"{field} values",
    )
    return tensor


def _validate_gripper_target_slew_static(value: Any, *, field: str) -> None:
    contract = _object(value, field)
    _require(
        set(contract) == GRIPPER_TARGET_SLEW_STATIC_FIELDS,
        f"{field} schema drift",
    )
    exact = {
        "profile": GRIPPER_TARGET_SLEW_PROFILE,
        "scope": "eef_pose_only_native_joint_position_unchanged_v1",
        "action_class": GRIPPER_TARGET_SLEW_ACTION_CLASS,
        "driver_joint_name": "finger_joint",
        "driver_joint_index": 7,
        "endpoint_semantics_profile": ("closed_positive_ge_0p5_inverse_open_gt_0p5_v1"),
        "physical_velocity_limit_source": (
            "live_implicit_actuator_velocity_limit_sim_float32_v1"
        ),
        "target_slew_rate_source": (
            "eef_profile_fraction_of_live_physical_velocity_limit_float32_v1"
        ),
        "physics_hz": 120.0,
        "physics_dt": 1.0 / 120.0,
        "float32_tolerance_rad": 1e-6,
        "reset_profile": GRIPPER_TARGET_SLEW_RESET_PROFILE,
        "tensor_dtype": "torch.float32",
        "tensor_device": "cuda:0",
    }
    for name, expected in exact.items():
        _require(
            type(contract.get(name)) is type(expected)
            and contract.get(name) == expected,
            f"{field}.{name}",
        )
    for name, expected in (
        ("open_target_rad", 0.0),
        ("closed_target_rad", GRIPPER_CLOSED_TARGET),
        ("physical_velocity_limit_rad_s", GRIPPER_PHYSICAL_VELOCITY_LIMIT),
        ("target_slew_rate_factor", GRIPPER_TARGET_SLEW_RATE_FACTOR),
        ("target_slew_rate_rad_s", GRIPPER_TARGET_SLEW_RATE),
        ("max_target_step_rad", GRIPPER_MAX_TARGET_STEP),
    ):
        _require(_same_float32(contract.get(name), expected), f"{field}.{name}")
    recomputed_rate = _float32(
        _float32(contract["physical_velocity_limit_rad_s"])
        * _float32(contract["target_slew_rate_factor"])
    )
    _require(
        _same_float32(recomputed_rate, contract["target_slew_rate_rad_s"]),
        f"{field} physical-limit/factor/rate binding",
    )
    recomputed_step = _float32(
        _float32(contract["target_slew_rate_rad_s"]) * _float32(contract["physics_dt"])
    )
    _require(
        _same_float32(recomputed_step, contract["max_target_step_rad"]),
        f"{field} rate/cadence cap binding",
    )


def _validate_gripper_static(value: Any, *, field: str) -> None:
    contract = _object(value, field)
    _require(set(contract) == GRIPPER_RUNTIME_STATIC_FIELDS, f"{field} schema drift")
    exact = {
        "profile": GRIPPER_RUNTIME_PROFILE,
        "joint_names": EXPECTED_DROID_JOINT_NAMES,
        "gripper_joint_names": GRIPPER_JOINT_NAMES,
        "gripper_joint_indices": list(range(7, 13)),
        "driver_joint_name": "finger_joint",
        "driver_joint_index": 7,
        "follower_joint_names": GRIPPER_JOINT_NAMES[1:],
        "follower_joint_indices": list(range(8, 13)),
        "actuator_joint_ownership": {
            "panda_shoulder": {
                "joint_names": EXPECTED_JOINT_NAMES[:4],
                "joint_indices": list(range(4)),
            },
            "panda_forearm": {
                "joint_names": EXPECTED_JOINT_NAMES[4:],
                "joint_indices": list(range(4, 7)),
            },
            "gripper": {"joint_names": ["finger_joint"], "joint_indices": [7]},
        },
        "device_partition": {
            "profile": "nvidia_droid_cuda_dynamic_actuator_cpu_static_physx_v1",
            "dynamic_articulation": "cuda:0",
            "implicit_actuator": "cuda:0",
            "static_physx": "cpu",
            "dtype": "torch.float32",
        },
        "measured_velocity_is_hard_bounded_by_limit": False,
    }
    for name, expected in exact.items():
        _require(_typed_equal(contract.get(name), expected), f"{field}.{name}")

    actuator = _object(contract.get("driver_actuator"), f"{field}.driver_actuator")
    actuator_fields = {
        "cfg_velocity_limit",
        "cfg_velocity_limit_sim",
        "cfg_effort_limit",
        "cfg_effort_limit_sim",
        "resolved_velocity_limit",
        "resolved_velocity_limit_sim",
        "resolved_effort_limit",
        "resolved_effort_limit_sim",
    }
    _require(set(actuator) == actuator_fields, f"{field}.driver_actuator schema")
    for name, expected in (
        ("cfg_velocity_limit", 5.0),
        ("cfg_velocity_limit_sim", 5.0),
        ("cfg_effort_limit", 200.0),
        ("cfg_effort_limit_sim", 200.0),
    ):
        _exact_float(actuator.get(name), expected, f"{field}.driver_actuator.{name}")
    for name in actuator_fields - {
        "cfg_velocity_limit",
        "cfg_velocity_limit_sim",
        "cfg_effort_limit",
        "cfg_effort_limit_sim",
    }:
        _validate_gripper_tensor(
            actuator[name],
            field=f"{field}.driver_actuator.{name}",
            expected=[200.0 if "effort" in name else 5.0],
            shape=[1, 1],
            device="cuda:0",
        )

    mimic = _object(contract.get("mimic_joint_contract"), f"{field}.mimic")
    _require(
        set(mimic)
        == {
            "profile",
            "robot_usd_sha256",
            "driver_joint_name",
            "driver_joint_index",
            "driver_joint_prim_path",
            "driver_physics_joint_type",
            "driver_exclude_from_articulation",
            "followers",
        },
        f"{field}.mimic schema",
    )
    _require(
        mimic.get("profile") == "robotiq_2f85_source_usd_physx_mimic_joint_v1"
        and mimic.get("robot_usd_sha256")
        == "d8379925b103963dbf3e7c85bcc4ae101b81b7c1d7dabe7d2e964f41d069ec44"
        and mimic.get("driver_joint_name") == "finger_joint"
        and mimic.get("driver_joint_index") == 7
        and mimic.get("driver_joint_prim_path")
        == "/panda/Gripper/Robotiq_2F_85/Joints/finger_joint"
        and mimic.get("driver_physics_joint_type") == "PhysicsRevoluteJoint"
        and mimic.get("driver_exclude_from_articulation") is False,
        f"{field}.mimic identity",
    )
    followers = _list(mimic.get("followers"), f"{field}.mimic.followers", length=5)
    follower_fields = {
        "joint_name",
        "joint_index",
        "prim_path",
        "physics_joint_type",
        "exclude_from_articulation",
        "mimic_axis",
        "reference_joint_path",
        "gearing",
        "natural_frequency_hz",
        "damping_ratio",
    }
    _require(
        [item.get("joint_name") for item in followers] == GRIPPER_JOINT_NAMES[1:]
        and [item.get("joint_index") for item in followers] == list(range(8, 13))
        and all(set(item) == follower_fields for item in followers)
        and all(
            item.get("physics_joint_type") == "PhysicsRevoluteJoint"
            and item.get("exclude_from_articulation") is False
            and item.get("prim_path")
            == f"/panda/Gripper/Robotiq_2F_85/Joints/{item['joint_name']}"
            and item.get("reference_joint_path") == mimic.get("driver_joint_prim_path")
            for item in followers
        ),
        f"{field}.mimic followers",
    )
    expected_mimic_values = [
        ("rotZ", -1.0, 1_000_000.0, 0.0),
        ("rotX", 1.0, 1_000.0, 0.05000000074505806),
        ("rotX", -1.0, 1_000.0, 0.05000000074505806),
        ("rotX", 1.0, 1_000.0, 0.05000000074505806),
        ("rotX", 1.0, 1_000.0, 0.05000000074505806),
    ]
    for index, (item, expected) in enumerate(
        zip(followers, expected_mimic_values, strict=True)
    ):
        _require(
            item.get("mimic_axis") == expected[0]
            and _same_float32(item.get("gearing"), expected[1])
            and _same_float32(item.get("natural_frequency_hz"), expected[2])
            and _same_float32(item.get("damping_ratio"), expected[3]),
            f"{field}.mimic follower {index} values",
        )

    before_values = [*EXPECTED_VELOCITY_LIMITS, 5.0, *([174.53292846679688] * 5)]
    after_values = [*EXPECTED_VELOCITY_LIMITS, *([5.0] * 6)]
    before = _validate_gripper_tensor(
        contract.get("velocity_limits_before_write"),
        field=f"{field}.before_write",
        expected=before_values,
        shape=[1, 13],
        device="cpu",
    )
    after = _validate_gripper_tensor(
        contract.get("velocity_limits_after_write"),
        field=f"{field}.after_write",
        expected=after_values,
        shape=[1, 13],
        device="cpu",
    )
    write = _object(contract.get("velocity_limit_write_contract"), f"{field}.write")
    _require(
        set(write)
        == {
            "profile",
            "setter",
            "timing",
            "call_count",
            "articulation_indices",
            "full_input",
        }
        and write.get("profile")
        == "live_root_physx_view_full_tensor_five_mimic_dofs_velocity_limit5_eef_production_v1"
        and write.get("setter") == "root_physx_view.set_dof_max_velocities"
        and write.get("timing") == "after_first_explicit_reset_before_first_apply_v1"
        and write.get("call_count") == 1
        and write.get("articulation_indices") == [0],
        f"{field}.write identity",
    )
    full_input = _validate_gripper_tensor(
        write.get("full_input"),
        field=f"{field}.write.full_input",
        expected=after_values,
        shape=[1, 13],
        device="cpu",
    )
    _require(_typed_equal(after, full_input), f"{field}.write/readback")
    _require(before["values"][:8] == after["values"][:8], f"{field}.driver drift")
    _validate_gripper_target_slew_static(
        contract.get("driver_target_slew"), field=f"{field}.target_slew"
    )


def _validate_gripper_target_slew_dynamic(
    value: Any,
    *,
    field: str,
    apply_calls: int,
    expect_closed_target: bool,
    expected_endpoint_change_count: int = 0,
) -> None:
    report = _object(value, field)
    _require(
        set(report) == GRIPPER_TARGET_SLEW_DYNAMIC_FIELDS,
        f"{field} schema drift",
    )
    _require(report.get("profile") == GRIPPER_TARGET_SLEW_PROFILE, f"{field}.profile")
    expected_process = apply_calls // 8
    counters = {
        name: report.get(name)
        for name in (
            "process_action_calls",
            "apply_calls",
            "initialization_count",
            "endpoint_change_count",
            "repeated_endpoint_process_count",
            "slew_limited_apply_count",
            "endpoint_reached_apply_count",
            "live_limit_validation_count",
        )
    }
    _require(
        all(type(item) is int and item >= 0 for item in counters.values()),
        f"{field}.counters",
    )
    _require(
        counters["process_action_calls"] == expected_process
        and counters["apply_calls"] == apply_calls
        and counters["live_limit_validation_count"] == apply_calls
        and counters["initialization_count"] == int(apply_calls > 0)
        and counters["endpoint_change_count"] == expected_endpoint_change_count
        and counters["repeated_endpoint_process_count"]
        == max(expected_process - 1 - expected_endpoint_change_count, 0)
        and counters["slew_limited_apply_count"]
        + counters["endpoint_reached_apply_count"]
        == apply_calls,
        f"{field}.cadence/history",
    )
    maxima = [
        _finite_number(report.get(name), f"{field}.{name}")
        for name in (
            "max_abs_target_step_rad",
            "max_abs_endpoint_error_before_step_rad",
            "max_abs_endpoint_error_after_step_rad",
        )
    ]
    _require(
        all(item >= 0.0 for item in maxima)
        and maxima[0] <= GRIPPER_MAX_TARGET_STEP + 1e-6
        and maxima[1] <= GRIPPER_MAX_ANCHOR
        and maxima[2] <= maxima[1] + 1e-6,
        f"{field}.maxima",
    )
    if apply_calls == 0:
        _require(
            maxima == [0.0, 0.0, 0.0]
            and report.get("initial_anchor_rad") is None
            and report.get("last_requested_endpoint_rad") is None
            and report.get("last_applied_target_rad") is None,
            f"{field}.empty state",
        )
    else:
        anchor = _finite_number(report.get("initial_anchor_rad"), f"{field}.anchor")
        endpoint = _finite_number(
            report.get("last_requested_endpoint_rad"), f"{field}.endpoint"
        )
        applied = _finite_number(
            report.get("last_applied_target_rad"), f"{field}.applied"
        )
        expected_endpoint = GRIPPER_CLOSED_TARGET if expect_closed_target else 0.0
        _require(
            GRIPPER_MIN_ANCHOR <= anchor <= GRIPPER_MAX_ANCHOR
            and _same_float32(endpoint, expected_endpoint)
            and GRIPPER_MIN_ANCHOR <= applied <= GRIPPER_MAX_ANCHOR,
            f"{field}.state",
        )
        if expect_closed_target:
            _require(
                apply_calls >= GRIPPER_CLOSE_TRANSITION_APPLIES
                and counters["slew_limited_apply_count"]
                == GRIPPER_CLOSE_LIMITED_APPLIES
                and counters["endpoint_reached_apply_count"]
                == apply_calls - GRIPPER_CLOSE_LIMITED_APPLIES
                and _same_float32(maxima[0], GRIPPER_MAX_TARGET_STEP)
                and _same_float32(maxima[1], GRIPPER_CLOSED_TARGET)
                and _same_float32(anchor, 0.0)
                and _same_float32(applied, GRIPPER_CLOSED_TARGET),
                f"{field}.closed transition",
            )
        else:
            _require(
                counters["slew_limited_apply_count"] == 0
                and counters["endpoint_reached_apply_count"] == apply_calls
                and maxima == [0.0, 0.0, 0.0]
                and _same_float32(anchor, 0.0)
                and _same_float32(applied, 0.0),
                f"{field}.open hold",
            )


def _validate_gripper_dynamic(
    value: Any,
    *,
    field: str,
    apply_calls: int,
    expect_closed_target: bool,
    expected_endpoint_change_count: int = 0,
) -> None:
    report = _object(value, field)
    _require(set(report) == GRIPPER_RUNTIME_DYNAMIC_FIELDS, f"{field} schema drift")
    _require(
        report.get("profile") == GRIPPER_RUNTIME_PROFILE
        and report.get("joint_names") == GRIPPER_JOINT_NAMES
        and report.get("joint_indices") == list(range(7, 13)),
        f"{field}.identity",
    )
    _require(
        report.get("apply_entry_samples") == apply_calls
        and report.get("post_policy_step_samples") == apply_calls // 8
        and report.get("nonfinite_samples") == 0
        and report.get("dropped_diagnostics") == 0,
        f"{field}.cadence",
    )
    physical_maxima = {}
    for name in (
        "max_abs_joint_velocity_rad_s",
        "max_abs_joint_acceleration_rad_s2",
    ):
        vector = _finite_vector(report.get(name), f"{field}.{name}", length=6)
        _require(all(item >= 0.0 for item in vector), f"{field}.{name}")
        physical_maxima[name] = vector
    diagnostic = report.get("max_velocity_diagnostic")
    terminal = report.get("terminal_state")
    if apply_calls == 0:
        _require(diagnostic is None and terminal is None, f"{field}.empty diagnostics")
    else:
        diagnostic = _object(diagnostic, f"{field}.max_velocity_diagnostic")
        _require(
            set(diagnostic)
            == {
                "sample_phase",
                "sample_index",
                "joint_position_rad",
                "joint_velocity_rad_s",
                "joint_acceleration_rad_s2",
                "joint_position_target_rad",
                "joint_velocity_target_rad_s",
            }
            and diagnostic.get("sample_phase") in {"apply_entry", "post_policy_step"}
            and type(diagnostic.get("sample_index")) is int
            and 0 <= diagnostic["sample_index"] < apply_calls + apply_calls // 8,
            f"{field}.max_velocity_diagnostic identity",
        )
        diagnostic_vectors = {}
        for name in (
            "joint_position_rad",
            "joint_velocity_rad_s",
            "joint_acceleration_rad_s2",
            "joint_position_target_rad",
            "joint_velocity_target_rad_s",
        ):
            diagnostic_vectors[name] = _finite_vector(
                diagnostic.get(name), f"{field}.diagnostic.{name}", length=6
            )
        _require(
            _same_float32(
                max(abs(item) for item in diagnostic_vectors["joint_velocity_rad_s"]),
                max(physical_maxima["max_abs_joint_velocity_rad_s"]),
            ),
            f"{field}.diagnostic/velocity maximum",
        )
        terminal = _object(terminal, f"{field}.terminal")
        _require(
            set(terminal)
            == {
                "sample_index",
                "joint_position_rad",
                "joint_velocity_rad_s",
                "joint_acceleration_rad_s2",
                "joint_position_target_rad",
                "joint_velocity_target_rad_s",
            }
            and terminal.get("sample_index") == (apply_calls // 8) * 9 - 1,
            f"{field}.terminal identity",
        )
        for name in set(terminal) - {"sample_index"}:
            _finite_vector(terminal.get(name), f"{field}.terminal.{name}", length=6)
    _validate_gripper_target_slew_dynamic(
        report.get("driver_target_slew"),
        field=f"{field}.target_slew",
        apply_calls=apply_calls,
        expect_closed_target=expect_closed_target,
        expected_endpoint_change_count=expected_endpoint_change_count,
    )


def _validate_safety_report(
    value: Any,
    *,
    field: str,
    episode_index: int | None,
    apply_calls: int,
    expect_closed_target: bool = False,
    expected_endpoint_change_count: int = 0,
) -> tuple[dict[str, int], dict[str, list[float]]]:
    report = _object(value, field)
    _require(set(report) == SAFETY_FIELDS, f"{field} schema drift")
    _require(
        report.get("current_joint_velocity_abort") is None,
        f"{field}.current_joint_velocity_abort must be null",
    )
    _validate_gripper_static(
        report.get("gripper_runtime_static"), field=f"{field}.gripper_static"
    )
    _validate_gripper_dynamic(
        report.get("gripper_runtime_dynamic"),
        field=f"{field}.gripper_dynamic",
        apply_calls=apply_calls,
        expect_closed_target=expect_closed_target,
        expected_endpoint_change_count=expected_endpoint_change_count,
    )
    if episode_index is None:
        _require(report.get("episode_index") is None, f"{field}.episode_index")
    else:
        _exact_int(report.get("episode_index"), episode_index, f"{field}.episode_index")
    for name, expected in (
        ("profile", "panda_velocity_physxlimit_solveriter1_v4"),
        ("apply_actions_cadence", "physics_substep"),
        (
            "target_soft_limit_guard_band_profile",
            "eef_physx_inner_hardlimit_one_substep_v2",
        ),
        ("physx_hard_limit_profile", "outer_minus_one_velocity_substep_v1"),
        (
            "physx_derived_soft_limit_profile",
            "isaaclab_midpoint_range_factor1_float32_v1",
        ),
        ("arm_velocity_target_profile", "zero_per_physics_substep_v1"),
        (
            "articulation_solver_profile",
            "tgs_position64_velocity1_eef_only_v1",
        ),
        (
            "articulation_solver_readback",
            "composed_usd_physx_articulation_api_all_env_roots_v1",
        ),
        ("target_joint_pos_limits_float32_sha256", EXPECTED_TARGET_DIGEST),
        ("physx_hard_joint_pos_limits_float32_sha256", EXPECTED_TARGET_DIGEST),
        (
            "physx_derived_soft_joint_pos_limits_float32_sha256",
            EXPECTED_PHYSX_DERIVED_SOFT_DIGEST,
        ),
        ("soft_joint_pos_limits_float32_sha256", EXPECTED_DIGEST),
    ):
        _require(
            type(report.get(name)) is str and report[name] == expected,
            f"{field}.{name}",
        )
    _exact_int(
        report.get("physx_solver_type"),
        1,
        f"{field}.physx_solver_type",
    )
    _exact_int(
        report.get("solver_position_iteration_count"),
        64,
        f"{field}.solver_position_iteration_count",
    )
    _exact_int(
        report.get("solver_velocity_iteration_count"),
        1,
        f"{field}.solver_velocity_iteration_count",
    )
    for name, expected in (
        ("physics_dt", 1.0 / 120.0),
        ("control_dt", 1.0 / 15.0),
        ("current_joint_soft_limit_tolerance_rad", 1e-5),
        ("eef_quaternion_unit_norm_tolerance", 1e-3),
        ("joint_slew_float32_tolerance_rad", 1e-6),
        ("joint_velocity_limit_tolerance_rad_s", 1e-5),
        ("soft_joint_pos_limit_factor", 1.0),
    ):
        _exact_float(report.get(name), expected, f"{field}.{name}")
    _exact_int(report.get("decimation"), 8, f"{field}.decimation")
    _exact_int(
        report.get("physx_hard_limit_write_count"),
        1,
        f"{field}.physx_hard_limit_write_count",
    )
    _require(
        report.get("joint_names") == EXPECTED_JOINT_NAMES
        and all(type(name) is str for name in report["joint_names"]),
        f"{field}.joint_names",
    )
    _exact_float_vector(
        report.get("joint_velocity_limits_rad_s"),
        EXPECTED_VELOCITY_LIMITS,
        f"{field}.joint_velocity_limits_rad_s",
    )
    _exact_float_vector(
        report.get("joint_effort_limits"),
        EXPECTED_EFFORT_LIMITS,
        f"{field}.joint_effort_limits",
    )
    _exact_float_vector(
        report.get("max_delta_joint_pos_rad"),
        EXPECTED_MAX_DELTA,
        f"{field}.max_delta_joint_pos_rad",
    )
    _exact_float_vector(
        report.get("target_soft_limit_margin_rad"),
        EXPECTED_MAX_DELTA,
        f"{field}.target_soft_limit_margin_rad",
    )
    target_limits = _list(
        report.get("target_joint_pos_limits_rad"),
        f"{field}.target_joint_pos_limits_rad",
        length=7,
    )
    for index, (actual, expected) in enumerate(
        zip(target_limits, EXPECTED_TARGET_LIMITS, strict=True)
    ):
        _exact_float_vector(
            actual, expected, f"{field}.target_joint_pos_limits_rad[{index}]"
        )
    physx_limits = _list(
        report.get("physx_hard_joint_pos_limits_rad"),
        f"{field}.physx_hard_joint_pos_limits_rad",
        length=7,
    )
    for index, (actual, expected) in enumerate(
        zip(physx_limits, EXPECTED_TARGET_LIMITS, strict=True)
    ):
        _exact_float_vector(
            actual,
            expected,
            f"{field}.physx_hard_joint_pos_limits_rad[{index}]",
        )
    physx_derived_soft_limits = _list(
        report.get("physx_derived_soft_joint_pos_limits_rad"),
        f"{field}.physx_derived_soft_joint_pos_limits_rad",
        length=7,
    )
    for index, (actual, expected) in enumerate(
        zip(
            physx_derived_soft_limits,
            EXPECTED_PHYSX_DERIVED_SOFT_LIMITS,
            strict=True,
        )
    ):
        _exact_float_vector(
            actual,
            expected,
            f"{field}.physx_derived_soft_joint_pos_limits_rad[{index}]",
        )
    _exact_float_vector(
        report.get("arm_velocity_target_rad_s"),
        [0.0] * 7,
        f"{field}.arm_velocity_target_rad_s",
    )
    limits = _list(report.get("soft_joint_pos_limits_rad"), f"{field}.limits", length=7)
    for index, (actual, expected) in enumerate(
        zip(limits, EXPECTED_LIMITS, strict=True)
    ):
        _exact_float_vector(actual, expected, f"{field}.limits[{index}]")

    counters = _object(report.get("counters"), f"{field}.counters")
    _require(set(counters) == COUNTER_FIELDS, f"{field}.counter schema drift")
    _require(
        all(type(item) is int and item >= 0 for item in counters.values()),
        f"{field}.counters invalid",
    )
    _require(counters["apply_calls"] == apply_calls, f"{field}.apply_calls")
    _require(
        counters["environment_substeps"] == apply_calls,
        f"{field}.environment_substeps",
    )
    for event_name, joint_name in (
        ("slew_limit_events", "slew_limited_joints"),
        ("position_limit_events", "position_limited_joints"),
    ):
        events = counters[event_name]
        joints = counters[joint_name]
        _require(
            events <= apply_calls and events <= joints <= 7 * events,
            f"{field}.{event_name}/{joint_name} impossible",
        )
    for name in (
        *ABORT_COUNTERS,
        "post_clamp_target_violations",
        "guard_diagnostics_dropped",
        "dls_fallbacks",
    ):
        _require(counters[name] == 0, f"{field}.{name} must be zero")

    maxima_value = _object(report.get("maxima"), f"{field}.maxima")
    _require(set(maxima_value) == MAXIMA_FIELDS, f"{field}.maxima schema drift")
    maxima = {
        name: _finite_vector(vector, f"{field}.maxima.{name}", length=7)
        for name, vector in maxima_value.items()
    }
    _require(
        all(
            item >= 0.0
            for name, vector in maxima.items()
            if name != "minimum_outer_joint_clearance_rad"
            for item in vector
        ),
        f"{field}.maxima must be nonnegative",
    )
    _require(
        all(
            item == 0.0 for item in maxima["post_clamp_target_soft_limit_violation_rad"]
        ),
        f"{field}.post-clamp maxima",
    )
    _require(
        all(
            item <= 1e-5
            for item in maxima["post_clamp_target_guard_band_violation_rad"]
        ),
        f"{field}.target guard-band maxima",
    )
    _require(
        all(
            guard <= current + 1e-6
            for guard, current in zip(
                maxima["post_clamp_target_guard_band_violation_rad"],
                maxima["current_joint_soft_limit_violation_rad"],
                strict=True,
            )
        ),
        f"{field}.target guard-band recovery attribution",
    )
    _require(
        all(item <= 1e-5 for item in maxima["current_joint_soft_limit_violation_rad"]),
        f"{field}.current-limit maxima",
    )
    _require(
        all(
            actual <= limit + 1e-5
            for actual, limit in zip(
                maxima["abs_joint_vel_rad_s"],
                EXPECTED_VELOCITY_LIMITS,
                strict=True,
            )
        ),
        f"{field}.joint-velocity maxima",
    )
    _require(
        all(
            slop <= margin + 1e-5
            for slop, margin in zip(
                maxima["current_physx_hard_limit_violation_rad"],
                EXPECTED_MAX_DELTA,
                strict=True,
            )
        ),
        f"{field}.PhysX hard-limit containment",
    )
    _require(
        all(
            math.isclose(
                violation,
                max(-clearance, 0.0),
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            for violation, clearance in zip(
                maxima["current_joint_soft_limit_violation_rad"],
                maxima["minimum_outer_joint_clearance_rad"],
                strict=True,
            )
        ),
        f"{field}.outer violation/clearance evidence",
    )
    _require(
        all(
            actual <= bound + 1e-6
            for actual, bound in zip(
                maxima["applied_delta_joint_pos_rad"], EXPECTED_MAX_DELTA, strict=True
            )
        ),
        f"{field}.applied slew maxima",
    )
    raw_slew_activated = any(
        raw_delta > bound
        for raw_delta, bound in zip(
            maxima["raw_delta_joint_pos_rad"], EXPECTED_MAX_DELTA, strict=True
        )
    )
    _require(
        (counters["slew_limit_events"] > 0) is raw_slew_activated,
        f"{field}.slew counters/maxima activation mismatch",
    )

    diagnostics = _list(report.get("guard_diagnostics"), f"{field}.diagnostics")
    _require(len(diagnostics) <= 32, f"{field}.diagnostics unbounded")
    _require(diagnostics == [], f"{field}.promotion diagnostics must be empty")
    mapped = {name: 0 for name in (*ABORT_COUNTERS, "dls_fallbacks")}
    indices = []
    for index, diagnostic_value in enumerate(diagnostics):
        flattened, diagnostic = _validate_diagnostic(
            diagnostic_value,
            f"{field}.diagnostics[{index}]",
            episode_index=episode_index,
            allowed_kinds=set(DIAGNOSTIC_COUNTERS),
            apply_calls=apply_calls,
        )
        indices.append(flattened)
        mapped[DIAGNOSTIC_COUNTERS[diagnostic["kind"]]] += 1
    _require(indices == sorted(indices), f"{field}.diagnostics out of order")
    for name, count in mapped.items():
        _require(counters[name] == count, f"{field}.{name}/diagnostic mismatch")

    max_raw = report.get("max_raw_delta_diagnostic")
    if apply_calls == 0:
        _require(max_raw is None, f"{field}.max-raw must be null")
        _require(
            all(item == 0 for item in counters.values()), f"{field}.initial counters"
        )
        _require(
            all(item == 0.0 for vector in maxima.values() for item in vector),
            f"{field}.initial maxima",
        )
    else:
        _, max_raw_diagnostic = _validate_diagnostic(
            max_raw,
            f"{field}.max_raw_delta_diagnostic",
            episode_index=episode_index,
            allowed_kinds={"max_raw_delta"},
            apply_calls=apply_calls,
        )
        raw_vector = _diagnostic_vector(
            max_raw_diagnostic["raw_delta_joint_pos_rad"],
            f"{field}.max_raw_delta_diagnostic.raw_delta_joint_pos_rad",
        )
        _require(
            raw_vector is not None and all(item is not None for item in raw_vector),
            f"{field}.max-raw finite",
        )
        diagnostic_max = max(abs(float(item)) for item in raw_vector)
        aggregate_max = max(maxima["raw_delta_joint_pos_rad"])
        _require(
            math.isclose(diagnostic_max, aggregate_max, rel_tol=0.0, abs_tol=1e-6),
            f"{field}.max-raw/maxima mismatch",
        )
        _require(
            max_raw_diagnostic.get("jacobian_finite") is True,
            f"{field}.max-raw jacobian",
        )
        _require(
            _finite_number(
                max_raw_diagnostic.get("pose_error_norm"),
                f"{field}.max-raw pose_error_norm",
            )
            >= 0.0,
            f"{field}.max-raw pose_error_norm",
        )
        _require(
            _finite_number(
                max_raw_diagnostic.get("jacobian_max_abs"),
                f"{field}.max-raw jacobian_max_abs",
            )
            >= 0.0,
            f"{field}.max-raw jacobian_max_abs",
        )
        _require(
            max_raw_diagnostic.get("eef_quaternion_norm") is None,
            f"{field}.max-raw eef_quaternion_norm must be null",
        )
        finite_vectors: dict[str, list[float]] = {}
        for name in (
            "joint_pos_rad",
            "raw_joint_pos_target_rad",
            "safe_joint_pos_target_rad",
        ):
            vector = _diagnostic_vector(
                max_raw_diagnostic.get(name), f"{field}.max_raw_delta_diagnostic.{name}"
            )
            _require(
                vector is not None and all(item is not None for item in vector),
                f"{field}.max-raw {name}",
            )
            finite_vectors[name] = [float(item) for item in vector]
        q_vector = finite_vectors["joint_pos_rad"]
        raw_target = finite_vectors["raw_joint_pos_target_rad"]
        safe_target = finite_vectors["safe_joint_pos_target_rad"]
        raw_delta = [float(item) for item in raw_vector]
        for index, (q, delta, raw, safe, bound, limits, target_limits) in enumerate(
            zip(
                q_vector,
                raw_delta,
                raw_target,
                safe_target,
                EXPECTED_MAX_DELTA,
                EXPECTED_LIMITS,
                EXPECTED_TARGET_LIMITS,
                strict=True,
            )
        ):
            _require(
                math.isclose(raw, q + delta, rel_tol=0.0, abs_tol=1e-6),
                f"{field}.max-raw joint {index} raw target identity",
            )
            _require(
                abs(safe - q) <= bound + 1e-6,
                f"{field}.max-raw joint {index} safe slew",
            )
            _require(
                limits[0] - 1e-5 <= safe <= limits[1] + 1e-5,
                f"{field}.max-raw joint {index} safe limits",
            )
            _require(
                target_limits[0] - 1e-5 <= safe <= target_limits[1] + 1e-5,
                f"{field}.max-raw joint {index} target guard band",
            )
    return counters, maxima


def _validate_ordinary_result(
    value: Any,
    *,
    index: int,
    position_tolerance: float,
    rotation_tolerance: float,
    frame_position_tolerance: float,
    frame_rotation_tolerance: float,
) -> None:
    field = f"ordinary[{index}]"
    result = _object(value, field)
    _require(set(result) == RESULT_FIELDS, f"{field} schema drift")
    _require(result.get("case") == EXPECTED_CASES[index], f"{field} case")
    _require(result.get("passed") is True, f"{field} did not pass")
    position_error = _finite_number(
        result.get("position_error_m"), f"{field}.position_error"
    )
    rotation_error = _finite_number(
        result.get("rotation_error_rad"), f"{field}.rotation_error"
    )
    _require(0.0 <= position_error <= position_tolerance, f"{field} position error")
    _require(0.0 <= rotation_error <= rotation_tolerance, f"{field} rotation error")
    target_position = _finite_vector(
        result.get("target_position"), f"{field}.target_position", length=3
    )
    actual_position = _finite_vector(
        result.get("actual_position"), f"{field}.actual_position", length=3
    )
    _require(
        math.isclose(
            math.dist(target_position, actual_position),
            position_error,
            rel_tol=0.0,
            abs_tol=1e-8,
        ),
        f"{field} position error inconsistent",
    )
    quaternions = {}
    for name in ("target_quaternion_wxyz", "actual_quaternion_wxyz"):
        quaternion = _finite_vector(result.get(name), f"{field}.{name}", length=4)
        _require(
            abs(math.sqrt(sum(item * item for item in quaternion)) - 1.0) <= 1e-3,
            f"{field}.{name} norm",
        )
        quaternions[name] = quaternion
    angular_distance = _quaternion_angular_distance_wxyz(
        quaternions["target_quaternion_wxyz"],
        quaternions["actual_quaternion_wxyz"],
    )
    _require(
        math.isclose(angular_distance, rotation_error, rel_tol=0.0, abs_tol=1e-7),
        f"{field} rotation error inconsistent",
    )
    for name, tolerance in (
        ("reset_frame_position_error_m", frame_position_tolerance),
        ("reset_frame_rotation_error_rad", frame_rotation_tolerance),
        ("final_frame_position_error_m", frame_position_tolerance),
        ("final_frame_rotation_error_rad", frame_rotation_tolerance),
    ):
        error = _finite_number(result.get(name), f"{field}.{name}")
        _require(0.0 <= error <= tolerance, f"{field}.{name} exceeded tolerance")


def _validate_case_target_geometry(ordinary: list[Any]) -> None:
    hold = _object(ordinary[0], "ordinary hold")
    hold_position = _finite_vector(
        hold.get("target_position"), "ordinary hold target_position", length=3
    )
    hold_quaternion = _finite_vector(
        hold.get("target_quaternion_wxyz"),
        "ordinary hold target_quaternion_wxyz",
        length=4,
    )
    translation_specs = (
        (1, 0, 1.0),
        (2, 0, -1.0),
        (3, 1, 1.0),
        (4, 1, -1.0),
        (5, 2, 1.0),
        (6, 2, -1.0),
    )
    for case_index, axis, sign in translation_specs:
        case = _object(ordinary[case_index], f"ordinary[{case_index}]")
        target_position = _finite_vector(
            case.get("target_position"),
            f"ordinary[{case_index}].target_position",
            length=3,
        )
        expected_position = hold_position.copy()
        expected_position[axis] += sign * 0.04
        _require(
            all(
                math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6)
                for actual, expected in zip(
                    target_position, expected_position, strict=True
                )
            ),
            f"ordinary[{case_index}] translation target geometry",
        )
        target_quaternion = _finite_vector(
            case.get("target_quaternion_wxyz"),
            f"ordinary[{case_index}].target_quaternion_wxyz",
            length=4,
        )
        _require(
            _quaternion_angular_distance_wxyz(target_quaternion, hold_quaternion)
            <= 1e-6,
            f"ordinary[{case_index}] translation changed target rotation",
        )

    rotation_specs = (
        (7, 0, 1.0),
        (8, 0, -1.0),
        (9, 1, 1.0),
        (10, 1, -1.0),
        (11, 2, 1.0),
        (12, 2, -1.0),
    )
    half_angle = math.radians(15.0) / 2.0
    for case_index, axis, sign in rotation_specs:
        case = _object(ordinary[case_index], f"ordinary[{case_index}]")
        target_position = _finite_vector(
            case.get("target_position"),
            f"ordinary[{case_index}].target_position",
            length=3,
        )
        _require(
            all(
                math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6)
                for actual, expected in zip(target_position, hold_position, strict=True)
            ),
            f"ordinary[{case_index}] rotation changed target position",
        )
        delta_quaternion = [math.cos(half_angle), 0.0, 0.0, 0.0]
        delta_quaternion[axis + 1] = sign * math.sin(half_angle)
        expected_quaternion = _quaternion_multiply_wxyz(
            hold_quaternion, delta_quaternion
        )
        target_quaternion = _finite_vector(
            case.get("target_quaternion_wxyz"),
            f"ordinary[{case_index}].target_quaternion_wxyz",
            length=4,
        )
        _require(
            _quaternion_angular_distance_wxyz(target_quaternion, expected_quaternion)
            <= 2e-6,
            f"ordinary[{case_index}] rotation target geometry",
        )


def _reject_constant(token: str) -> None:
    raise VerificationError(f"non-standard JSON constant {token!r}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _strict_json_from_bytes(data: bytes, field: str) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
        value = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"{field} is not strict UTF-8 JSON: {error}") from error
    return _object(value, field)


def _read_json_once(path: Path, field: str) -> tuple[dict[str, Any], bytes, str]:
    _require(path.is_file(), f"{field} does not exist: {path}")
    data = path.read_bytes()
    _require(bool(data), f"{field} is empty: {path}")
    return _strict_json_from_bytes(data, field), data, hashlib.sha256(data).hexdigest()


def _mode(path: Path) -> str:
    return f"{stat.S_IMODE(path.stat().st_mode):04o}"


def _file_identity(path: Path, field: str) -> dict[str, Any]:
    _require(path.is_file(), f"{field} does not exist: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return {
        "path": str(path),
        "size_bytes": size,
        "sha256": digest.hexdigest(),
        "mode": _mode(path),
    }


def _validate_close_headroom_entry(
    value: Any,
    *,
    field: str,
    episode_index: int,
    safety_report: dict[str, Any],
) -> dict[str, Any]:
    entry = _object(value, field)
    _require(set(entry) == CLOSE_HEADROOM_ENTRY_FIELDS, f"{field} schema drift")
    _exact_int(entry.get("episode_index"), episode_index, f"{field}.episode_index")
    _require(entry.get("joint_names") == EXPECTED_JOINT_NAMES, f"{field}.joint_names")
    expected_maxima = safety_report["maxima"]["abs_joint_vel_rad_s"]
    expected_limits = safety_report["joint_velocity_limits_rad_s"]
    _exact_float_vector(
        entry.get("max_abs_joint_velocity_rad_s"),
        expected_maxima,
        f"{field}.maxima",
    )
    _exact_float_vector(
        entry.get("joint_velocity_limit_rad_s"),
        expected_limits,
        f"{field}.limits",
    )
    ratios = _finite_vector(
        entry.get("velocity_to_limit_ratio"), f"{field}.ratios", length=7
    )
    recomputed = [
        maximum / limit
        for maximum, limit in zip(expected_maxima, expected_limits, strict=True)
    ]
    _require(
        all(
            math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
            for actual, expected in zip(ratios, recomputed, strict=True)
        ),
        f"{field}.ratio binding",
    )
    maximum_ratio = _finite_number(entry.get("maximum_ratio"), f"{field}.maximum")
    _exact_float(
        entry.get("threshold_ratio"),
        CLOSE_ARM_VELOCITY_HEADROOM_MAX_RATIO,
        f"{field}.threshold",
    )
    _require(
        math.isclose(maximum_ratio, max(recomputed), rel_tol=0.0, abs_tol=1e-12),
        f"{field}.maximum binding",
    )
    _require(
        entry.get("passed") is True
        and maximum_ratio <= CLOSE_ARM_VELOCITY_HEADROOM_MAX_RATIO,
        f"{field} lacks required five-percent arm-velocity headroom",
    )
    return {
        "maximum_ratio": maximum_ratio,
        "max_abs_joint_velocity_rad_s": list(expected_maxima),
    }


def _verify_raw(raw: dict[str, Any]) -> dict[str, Any]:
    _require(set(raw) == RAW_FIELDS, "raw top-level schema drift")
    _exact_int(raw.get("schema_version"), 2, "raw schema_version")
    _require(raw.get("finalized") is False, "raw finalized must be false")
    _require(raw.get("passed") is False, "raw passed must be false")
    _require(
        raw.get("stage") == "simulation_app_close_pending",
        "raw stage must be simulation_app_close_pending",
    )
    _require(raw.get("case") is None, "raw case must be null")
    _exact_int(raw.get("exit_code"), 0, "raw exit_code")
    _require(raw.get("failure") is None, "raw failure must be null")
    _require(
        raw.get("terminal_failure_evidence") is None,
        "successful raw terminal failure evidence must be null",
    )
    _require(raw.get("close_failures") == [], "raw close failures must be empty")
    _require(
        raw.get("persistence_failures") == [],
        "raw persistence failures must be empty",
    )
    _require(
        type(raw.get("environment")) is str
        and raw["environment"] == "DROID-FoodBussing",
        "raw environment",
    )
    _require(
        type(raw.get("eef_frame")) is str and raw["eef_frame"] == "panda_link8",
        "raw EEF frame",
    )
    _exact_int(raw.get("hold_steps"), 45, "raw hold_steps")
    for name, expected in (
        ("position_delta_m", 0.04),
        ("rotation_delta_deg", 15.0),
        ("position_tolerance_m", 0.01),
        ("rotation_tolerance_deg", 5.0),
        ("frame_position_tolerance_m", 1e-5),
        ("frame_rotation_tolerance_deg", 0.01),
    ):
        _exact_float(raw.get(name), expected, f"raw {name}")

    _validate_safety_report(
        raw.get("raw_ik_safety_capture"),
        field="raw capture",
        episode_index=None,
        apply_calls=0,
    )

    ordinary = _list(raw.get("results"), "ordinary results", length=13)
    _require(
        [entry.get("case") if isinstance(entry, dict) else None for entry in ordinary]
        == EXPECTED_CASES,
        "ordinary cases are missing or out of order",
    )
    for index, entry_value in enumerate(ordinary):
        _validate_ordinary_result(
            entry_value,
            index=index,
            position_tolerance=raw["position_tolerance_m"],
            rotation_tolerance=math.radians(raw["rotation_tolerance_deg"]),
            frame_position_tolerance=raw["frame_position_tolerance_m"],
            frame_rotation_tolerance=math.radians(raw["frame_rotation_tolerance_deg"]),
        )
    _validate_case_target_geometry(ordinary)

    reports = _list(raw.get("ik_safety_episodes"), "safety reports", length=13)
    for index, report_value in enumerate(reports):
        _validate_safety_report(
            report_value,
            field=f"safety[{index}]",
            episode_index=index,
            apply_calls=360,
            expect_closed_target=index == 0,
        )

    delayed_close = _object(
        raw.get("gripper_delayed_close_replay"), "delayed-close replay"
    )
    _require(
        set(delayed_close) == DELAYED_CLOSE_FIELDS,
        "delayed-close replay schema drift",
    )
    _require(
        delayed_close.get("profile") == DELAYED_CLOSE_REPLAY_PROFILE
        and delayed_close.get("case")
        == "open 115 policy steps then close at the same arm pose",
        "delayed-close replay identity",
    )
    _require(delayed_close.get("passed") is True, "delayed-close replay failed")
    _exact_int(delayed_close.get("episode_index"), 13, "delayed-close episode")
    _exact_int(
        delayed_close.get("open_policy_steps"),
        DELAYED_CLOSE_OPEN_POLICY_STEPS,
        "delayed-close open steps",
    )
    _exact_int(
        delayed_close.get("close_policy_steps"),
        DELAYED_CLOSE_CLOSE_POLICY_STEPS,
        "delayed-close close steps",
    )
    _exact_int(
        delayed_close.get("close_transition_substeps"),
        GRIPPER_CLOSE_TRANSITION_APPLIES,
        "delayed-close transition substeps",
    )
    _exact_int(delayed_close.get("arm_abort_count"), 0, "delayed-close aborts")
    _require(
        delayed_close.get("terminated") is False
        and delayed_close.get("truncated") is False,
        "delayed-close replay terminated or truncated",
    )
    delayed_counters, _ = _validate_safety_report(
        delayed_close.get("ik_safety"),
        field="delayed-close safety",
        episode_index=13,
        apply_calls=DELAYED_CLOSE_APPLY_CALLS,
        expect_closed_target=True,
        expected_endpoint_change_count=1,
    )
    _require(
        sum(delayed_counters[name] for name in ABORT_COUNTERS)
        == delayed_close["arm_abort_count"],
        "delayed-close abort counter binding",
    )
    close_headroom = _object(
        raw.get("gripper_close_velocity_headroom"), "close velocity headroom"
    )
    _require(
        set(close_headroom) == CLOSE_HEADROOM_FIELDS,
        "close velocity headroom schema drift",
    )
    _require(
        close_headroom.get("profile") == CLOSE_ARM_VELOCITY_HEADROOM_PROFILE,
        "close velocity headroom profile",
    )
    _exact_float(
        close_headroom.get("threshold_ratio"),
        CLOSE_ARM_VELOCITY_HEADROOM_MAX_RATIO,
        "close velocity headroom threshold",
    )
    _require(close_headroom.get("passed") is True, "close velocity headroom failed")
    immediate_headroom_summary = _validate_close_headroom_entry(
        close_headroom.get("immediate_close_hold"),
        field="immediate-close headroom",
        episode_index=0,
        safety_report=reports[0],
    )
    delayed_headroom_summary = _validate_close_headroom_entry(
        close_headroom.get("delayed_close_replay"),
        field="delayed-close headroom",
        episode_index=13,
        safety_report=delayed_close["ik_safety"],
    )

    adversarial = _object(raw.get("ik_safety_adversarial"), "adversarial")
    _require(set(adversarial) == ADVERSARIAL_FIELDS, "adversarial schema drift")
    _require(
        adversarial.get("case") == "oversized absolute +x target for one policy step",
        "adversarial case identity",
    )
    for field in (
        "passed",
        "state_is_finite",
        "eef_state_is_finite",
        "joint_state_is_finite",
        "joint_pos_within_captured_soft_limits",
    ):
        _require(adversarial.get(field) is True, f"adversarial {field} must be true")
    _require(adversarial.get("terminated") is False, "adversarial terminated")
    _require(adversarial.get("truncated") is False, "adversarial truncated")
    _require(adversarial.get("guard_error") == "", "adversarial guard error")

    joint_state = _object(adversarial.get("joint_state"), "adversarial joint_state")
    _require(
        set(joint_state)
        == {
            "joint_names",
            "joint_pos_rad",
            "joint_vel_rad_s",
            "soft_limit_violation_rad",
            "soft_limit_tolerance_rad",
            "position_within_captured_soft_limits",
        },
        "adversarial joint_state schema drift",
    )
    _require(
        joint_state.get("joint_names") == EXPECTED_JOINT_NAMES,
        "adversarial joint names",
    )
    _require(
        joint_state.get("position_within_captured_soft_limits") is True,
        "adversarial joint evidence says q is outside limits",
    )
    tolerance = _finite_number(
        joint_state.get("soft_limit_tolerance_rad"), "joint soft-limit tolerance"
    )
    _require(tolerance == 1e-5, "joint soft-limit tolerance mismatch")
    q = _finite_vector_evidence(joint_state.get("joint_pos_rad"), "adversarial q")
    dq = _finite_vector_evidence(joint_state.get("joint_vel_rad_s"), "adversarial dq")
    _require(
        all(
            abs(velocity) <= limit + 1e-6
            for velocity, limit in zip(dq, EXPECTED_VELOCITY_LIMITS, strict=True)
        ),
        "adversarial terminal dq exceeds configured velocity limits",
    )
    soft_violations = _finite_vector_evidence(
        joint_state.get("soft_limit_violation_rad"), "adversarial soft violations"
    )
    _require(
        all(item <= tolerance for item in soft_violations),
        "adversarial soft-limit violation evidence",
    )
    for index, (position, limits) in enumerate(zip(q, EXPECTED_LIMITS, strict=True)):
        _require(
            limits[0] - tolerance <= position <= limits[1] + tolerance,
            f"adversarial q[{index}] is outside captured limits",
        )

    counters, maxima = _validate_safety_report(
        adversarial.get("ik_safety"),
        field="adversarial safety",
        episode_index=14,
        apply_calls=8,
    )
    slew_events = counters.get("slew_limit_events")
    _require(
        type(slew_events) is int and slew_events > 0,
        "adversarial slew events must be positive",
    )

    guard = _object(adversarial.get("guard_evidence"), "guard evidence")
    _require(
        set(guard)
        == {
            "apply_calls",
            "slew_limit_events",
            "abort_count",
            "post_clamp_target_violations",
            "applied_within_bounds",
        },
        "guard evidence schema drift",
    )
    _exact_int(guard.get("apply_calls"), 8, "guard apply_calls")
    _exact_int(guard.get("abort_count"), 0, "guard abort_count")
    _exact_int(guard.get("post_clamp_target_violations"), 0, "guard post-clamp")
    _require(guard.get("applied_within_bounds") is True, "guard slew bound")
    _require(
        type(guard.get("slew_limit_events")) is int
        and guard["slew_limit_events"] == slew_events,
        "guard slew event count mismatch",
    )

    applied = maxima["applied_delta_joint_pos_rad"]
    bounds = EXPECTED_MAX_DELTA
    for index, (actual_value, bound_value) in enumerate(
        zip(applied, bounds, strict=True)
    ):
        actual = _finite_number(actual_value, f"applied[{index}]")
        bound = _finite_number(bound_value, f"bound[{index}]")
        _require(actual <= bound + 1e-6, f"joint {index} exceeded slew bound")
    limited_indices = [
        index
        for index, (raw_delta, bound) in enumerate(
            zip(maxima["raw_delta_joint_pos_rad"], EXPECTED_MAX_DELTA, strict=True)
        )
        if raw_delta > bound + 1e-6
    ]
    _require(limited_indices, "adversarial raw maxima never exceed a slew bound")
    for index in limited_indices:
        _require(
            applied[index] >= EXPECTED_MAX_DELTA[index] - 1e-6,
            f"adversarial joint {index} slew event lacks saturated applied maximum",
        )

    return {
        "ordinary_case_count": 13,
        "ordinary_case_order": EXPECTED_CASES,
        "ordinary_pass_count": 13,
        "ordinary_safety_report_count": 13,
        "ordinary_apply_calls_each": 360,
        "delayed_close": {
            "passed": True,
            "episode_index": 13,
            "open_policy_steps": DELAYED_CLOSE_OPEN_POLICY_STEPS,
            "close_policy_steps": DELAYED_CLOSE_CLOSE_POLICY_STEPS,
            "apply_calls": DELAYED_CLOSE_APPLY_CALLS,
            "close_transition_substeps": GRIPPER_CLOSE_TRANSITION_APPLIES,
            "slew_limited_apply_count": GRIPPER_CLOSE_LIMITED_APPLIES,
            "abort_count": 0,
            "arm_velocity_headroom": delayed_headroom_summary,
        },
        "immediate_close_arm_velocity_headroom": immediate_headroom_summary,
        "soft_limit_digest": EXPECTED_DIGEST,
        "target_limit_digest": EXPECTED_TARGET_DIGEST,
        "physx_derived_soft_limit_digest": EXPECTED_PHYSX_DERIVED_SOFT_DIGEST,
        "adversarial": {
            "passed": True,
            "apply_calls": 8,
            "slew_limit_events": slew_events,
            "abort_count": 0,
            "post_clamp_target_violations": 0,
            "eef_state_is_finite": True,
            "joint_state_is_finite": True,
            "joint_position_within_soft_limits": True,
        },
    }


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _strict_json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()


def _publish_nonoverwriting(path: Path, payload: dict[str, Any]) -> None:
    _require(not path.exists(), f"attestation already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(_strict_json_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
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
        temporary.unlink(missing_ok=True)


def _build_expected(args: argparse.Namespace) -> dict[str, Any]:
    _exact_int(args.srun_rc, 0, "srun_rc")
    _require(
        type(args.job_id) is int and args.job_id > 0, "job_id must be positive int"
    )
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    _require(
        slurm_job_id is not None
        and slurm_job_id.isdecimal()
        and int(slurm_job_id) == args.job_id,
        "CLI job_id does not match SLURM_JOB_ID",
    )
    _require(args.raw_result.resolve() != args.attestation.resolve(), "path collision")
    _require(
        args.raw_result.name == f"smoke-{args.job_id}.json",
        "raw result filename does not bind job_id",
    )
    _require(
        args.attestation.name == f"smoke-{args.job_id}.attestation.json",
        "attestation filename does not bind job_id",
    )
    _require(
        args.raw_result.parent.resolve() == args.attestation.parent.resolve(),
        "raw result and attestation directories differ",
    )
    for value, field, length in (
        (args.expected_polaris_commit, "expected PolaRiS commit", 40),
        (args.expected_smoke_sha256, "expected smoke SHA-256", 64),
        (args.expected_image_sha256, "expected image SHA-256", 64),
        (args.expected_finalizer_sha256, "expected finalizer SHA-256", 64),
        (
            args.expected_saved_job_script_sha256,
            "expected saved job script SHA-256",
            64,
        ),
    ):
        _require(
            len(value) == length
            and all(character in "0123456789abcdef" for character in value),
            f"{field} is malformed",
        )

    raw, raw_bytes, raw_sha256 = _read_json_once(args.raw_result, "raw result")
    _require(_mode(args.raw_result) == "0444", "raw result mode must be 0444")
    summary = _verify_raw(raw)

    marker_path = args.raw_result.with_name(args.raw_result.name + ".ready.json")
    marker, marker_bytes, marker_sha256 = _read_json_once(marker_path, "ready marker")
    _require(_mode(marker_path) == "0444", "ready marker mode must be 0444")
    _require(
        _typed_equal(
            marker,
            {
                "schema_version": 1,
                "stage": "simulation_app_close_pending",
                "raw_result": {
                    "path": str(args.raw_result),
                    "size_bytes": len(raw_bytes),
                    "sha256": raw_sha256,
                    "mode": "0444",
                },
            },
        ),
        "ready marker does not bind the exact raw result",
    )

    commit = _git(args.polaris_repo, "rev-parse", "HEAD")
    _require(commit == args.expected_polaris_commit, "PolaRiS commit mismatch")
    _require(_git(args.polaris_repo, "status", "--porcelain") == "", "repo dirty")
    smoke_identity = _file_identity(
        args.polaris_repo / "scripts" / "smoke_eef_pose_controller.py",
        "smoke source",
    )
    _require(
        smoke_identity["sha256"] == args.expected_smoke_sha256,
        "smoke source digest mismatch",
    )
    image_identity = _file_identity(args.container_image, "container image")
    _require(
        image_identity["sha256"] == args.expected_image_sha256,
        "container image digest mismatch",
    )
    runtime_script = _file_identity(args.runtime_job_script, "runtime job script")
    saved_script = _file_identity(args.saved_job_script, "saved job script")
    _require(
        runtime_script["sha256"] == saved_script["sha256"],
        "runtime/saved job script digest mismatch",
    )
    finalizer_identity = _file_identity(Path(__file__).resolve(), "finalizer")
    _require(
        saved_script["sha256"] == args.expected_saved_job_script_sha256,
        "saved job script expected digest mismatch",
    )
    _require(
        finalizer_identity["sha256"] == args.expected_finalizer_sha256,
        "finalizer expected digest mismatch",
    )

    return {
        "schema_version": 1,
        "finalized": True,
        "passed": True,
        "stage": "complete",
        "job_id": args.job_id,
        "srun_rc": args.srun_rc,
        "raw_result": {
            "path": str(args.raw_result),
            "size_bytes": len(raw_bytes),
            "sha256": raw_sha256,
            "mode": "0444",
            "ready_marker": {
                "path": str(marker_path),
                "size_bytes": len(marker_bytes),
                "sha256": marker_sha256,
                "mode": "0444",
            },
        },
        "validation_summary": summary,
        "provenance": {
            "polaris_repo": str(args.polaris_repo),
            "polaris_commit": commit,
            "smoke_source": smoke_identity,
            "container_image": image_identity,
            "runtime_job_script": runtime_script,
            "saved_job_script": saved_script,
            "finalizer": finalizer_identity,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("finalize", "verify"))
    parser.add_argument("--raw-result", required=True, type=Path)
    parser.add_argument("--attestation", required=True, type=Path)
    parser.add_argument("--srun-rc", required=True, type=int)
    parser.add_argument("--job-id", required=True, type=int)
    parser.add_argument("--runtime-job-script", required=True, type=Path)
    parser.add_argument("--saved-job-script", required=True, type=Path)
    parser.add_argument("--polaris-repo", required=True, type=Path)
    parser.add_argument("--expected-polaris-commit", required=True)
    parser.add_argument("--expected-smoke-sha256", required=True)
    parser.add_argument("--container-image", required=True, type=Path)
    parser.add_argument("--expected-image-sha256", required=True)
    parser.add_argument("--expected-finalizer-sha256", required=True)
    parser.add_argument("--expected-saved-job-script-sha256", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        expected = _build_expected(args)
        if args.mode == "finalize":
            _publish_nonoverwriting(args.attestation, expected)
        attestation, attestation_bytes, attestation_sha256 = _read_json_once(
            args.attestation, "attestation"
        )
        _require(_mode(args.attestation) == "0444", "attestation mode must be 0444")
        _require(_typed_equal(attestation, expected), "attestation content mismatch")
    except (
        OSError,
        subprocess.CalledProcessError,
        VerificationError,
    ) as error:
        print(f"SMOKE_ATTESTATION_FAIL={error}", file=sys.stderr, flush=True)
        return 1
    print(f"SMOKE_ATTESTATION_PASS={args.attestation}", flush=True)
    print(f"SMOKE_ATTESTATION_SIZE_BYTES={len(attestation_bytes)}", flush=True)
    print(f"SMOKE_ATTESTATION_SHA256={attestation_sha256}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
