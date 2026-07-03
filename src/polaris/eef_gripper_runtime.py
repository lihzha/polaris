"""Closed runtime evidence for the production EEF Robotiq gripper profile.

The physical articulation contains one driven ``finger_joint`` and five
passive PhysX mimic joints.  Isaac Lab exposes static PhysX drive properties on
CPU while cached articulation state and implicit-actuator tensors live on CUDA.
This module preserves that distinction and performs the one production write
needed by the passive followers.  The isolated rate-0.25 candidate additionally
wraps the existing USD spawner and authors compliant mimic parameters only
after the composed clone exists and before articulation initialization.  It
deliberately does not claim that a PhysX maximum-velocity setting is a hard
bound on measured passive-joint velocity.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from polaris.gripper_semantics import GRIPPER_THRESHOLD_PROFILE


EEF_GRIPPER_RUNTIME_PROFILE = (
    "implicit_gripper_physx_velocity_limit5_followers5_"
    "cuda_actuator_cpu_static_physx_v1"
)
EEF_GRIPPER_VELOCITY_WRITE_PROFILE = (
    "live_root_physx_view_full_tensor_five_mimic_dofs_velocity_limit5_eef_production_v1"
)
EEF_GRIPPER_VELOCITY_WRITE_SETTER = "root_physx_view.set_dof_max_velocities"
EEF_GRIPPER_VELOCITY_WRITE_TIMING = "after_first_explicit_reset_before_first_apply_v1"
EEF_GRIPPER_DEVICE_PARTITION_PROFILE = (
    "nvidia_droid_cuda_dynamic_actuator_cpu_static_physx_v1"
)
EEF_GRIPPER_MIMIC_PROFILE = "robotiq_2f85_source_usd_physx_mimic_joint_v1"
EEF_GRIPPER_MIMIC_COMPLIANCE_PROFILE = (
    "robotiq_2f85_live_physx_mimic_frequency100_damping1p2_candidate_v1"
)
EEF_GRIPPER_MIMIC_COMPLIANCE_SCOPE = (
    "eef_rate0p25_candidate_only_source_usd_immutable_v1"
)
EEF_GRIPPER_MIMIC_COMPLIANCE_TIMING = (
    "after_original_usd_spawn_before_articulation_initialization_v1"
)
EEF_GRIPPER_MIMIC_COMPLIANCE_SETTER = "UsdAttribute.Set_default_float_v1"
EEF_GRIPPER_MIMIC_COMPLIANCE_LIVE_ROOT_PROFILE = (
    "single_composed_world_env0_robot_root_v1"
)
EEF_GRIPPER_MIMIC_COMPLIANCE_ORIGINAL_SPAWN_IDENTITY = {
    "module": "isaaclab.sim.spawners.from_files.from_files",
    "qualname": "spawn_from_usd",
    "name": "spawn_from_usd",
}
EEF_GRIPPER_MIMIC_COMPLIANCE_OVERLAY_IDENTITY = {
    "module": "polaris.eef_gripper_runtime",
    "qualname": "eef_mimic_compliance_spawn_overlay",
    "name": "eef_mimic_compliance_spawn_overlay",
}
EEF_GRIPPER_MIMIC_COMPLIANCE_PHYSICS_HZ = 120.0
EEF_GRIPPER_MIMIC_COMPLIANCE_PHYSICS_DT = 1.0 / EEF_GRIPPER_MIMIC_COMPLIANCE_PHYSICS_HZ
EEF_GRIPPER_MIMIC_COMPLIANCE_NATURAL_FREQUENCY_RAD_S_FLOAT32 = float(np.float32(100.0))
EEF_GRIPPER_MIMIC_COMPLIANCE_DAMPING_RATIO_FLOAT32 = float(np.float32(1.2))
EEF_GRIPPER_MIMIC_COMPLIANCE_FREQUENCY_TIMESTEP_PRODUCT = 5.0 / 6.0
EEF_GRIPPER_MIMIC_COMPLIANCE_FOLLOWER_COUNT = 5
EEF_GRIPPER_MIMIC_COMPLIANCE_TOTAL_WRITE_COUNT = (
    2 * EEF_GRIPPER_MIMIC_COMPLIANCE_FOLLOWER_COUNT
)
EEF_GRIPPER_MIMIC_COMPLIANCE_EXPECTED_LIVE_ROOT = "/World/envs/env_0/robot"
GRIPPER_APPLY_ENTRY_SAMPLES_PER_POLICY_STEP = 8
GRIPPER_INTERLEAVED_SAMPLES_PER_POLICY_STEP = (
    GRIPPER_APPLY_ENTRY_SAMPLES_PER_POLICY_STEP + 1
)

EXPECTED_DROID_JOINT_NAMES = (
    *(f"panda_joint{index}" for index in range(1, 8)),
    "finger_joint",
    "right_outer_knuckle_joint",
    "left_inner_finger_joint",
    "right_inner_finger_joint",
    "left_inner_finger_knuckle_joint",
    "right_inner_finger_knuckle_joint",
)
EXPECTED_ARM_JOINT_NAMES = EXPECTED_DROID_JOINT_NAMES[:7]
EXPECTED_ARM_JOINT_INDICES = tuple(range(7))
DRIVEN_GRIPPER_JOINT_NAME = "finger_joint"
DRIVEN_GRIPPER_JOINT_INDEX = 7
GRIPPER_FOLLOWER_JOINT_NAMES = EXPECTED_DROID_JOINT_NAMES[8:]
GRIPPER_FOLLOWER_JOINT_INDICES = tuple(range(8, 13))
GRIPPER_JOINT_NAMES = EXPECTED_DROID_JOINT_NAMES[7:]
GRIPPER_JOINT_INDICES = tuple(range(7, 13))
EXPECTED_ACTUATOR_JOINT_OWNERSHIP = {
    "panda_shoulder": (EXPECTED_ARM_JOINT_NAMES[:4], EXPECTED_ARM_JOINT_INDICES[:4]),
    "panda_forearm": (EXPECTED_ARM_JOINT_NAMES[4:], EXPECTED_ARM_JOINT_INDICES[4:]),
    "gripper": ((DRIVEN_GRIPPER_JOINT_NAME,), (DRIVEN_GRIPPER_JOINT_INDEX,)),
}

EXPECTED_ARM_VELOCITY_LIMITS_FLOAT32 = (
    2.174999952316284,
    2.174999952316284,
    2.174999952316284,
    2.174999952316284,
    2.609999895095825,
    2.609999895095825,
    2.609999895095825,
)
GRIPPER_DRIVER_VELOCITY_LIMIT_FLOAT32 = 5.0
GRIPPER_FOLLOWER_DEFAULT_VELOCITY_LIMIT_FLOAT32 = 174.53292846679688
GRIPPER_FOLLOWER_VELOCITY_LIMIT_FLOAT32 = 5.0
EEF_GRIPPER_TARGET_SLEW_PROFILE = (
    "eef_binary_driver_target_slew_rate2p5_from_live_limit5_per_120hz_substep_v2"
)
EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE = (
    "eef_binary_driver_target_slew_rate1p25_from_live_limit5_"
    "per_120hz_substep_candidate_v1"
)
EEF_GRIPPER_TARGET_SLEW_ACTION_CLASS = "EefBinaryJointPositionTargetSlewAction"
EEF_GRIPPER_TARGET_SLEW_RESET_PROFILE = (
    "first_apply_after_action_reset_anchor_live_driver_position_v1"
)
GRIPPER_TARGET_SLEW_RATE_SOURCE = (
    "eef_profile_fraction_of_live_physical_velocity_limit_float32_v1"
)
GRIPPER_TARGET_SLEW_RATE_FACTOR_FLOAT32 = float(np.float32(0.5))
GRIPPER_TARGET_SLEW_RATE_RAD_S_FLOAT32 = float(
    np.multiply(
        np.float32(GRIPPER_DRIVER_VELOCITY_LIMIT_FLOAT32),
        np.float32(GRIPPER_TARGET_SLEW_RATE_FACTOR_FLOAT32),
        dtype=np.float32,
    )
)
GRIPPER_TARGET_SLEW_RATE_0P25_FACTOR_FLOAT32 = float(np.float32(0.25))
GRIPPER_TARGET_SLEW_RATE_0P25_RAD_S_FLOAT32 = float(
    np.multiply(
        np.float32(GRIPPER_DRIVER_VELOCITY_LIMIT_FLOAT32),
        np.float32(GRIPPER_TARGET_SLEW_RATE_0P25_FACTOR_FLOAT32),
        dtype=np.float32,
    )
)
GRIPPER_TARGET_SLEW_PHYSICS_HZ = 120.0
GRIPPER_TARGET_SLEW_PHYSICS_DT = 1.0 / GRIPPER_TARGET_SLEW_PHYSICS_HZ
GRIPPER_TARGET_SLEW_FLOAT32_TOLERANCE_RAD = 1e-6
GRIPPER_OPEN_TARGET_FLOAT32 = float(np.float32(0.0))
GRIPPER_CLOSED_TARGET_FLOAT32 = float(np.float32(np.pi / 4.0))
GRIPPER_TARGET_SLEW_MIN_ANCHOR_FLOAT32 = float(
    np.subtract(
        np.float32(GRIPPER_OPEN_TARGET_FLOAT32),
        np.float32(GRIPPER_TARGET_SLEW_FLOAT32_TOLERANCE_RAD),
        dtype=np.float32,
    )
)
GRIPPER_TARGET_SLEW_MAX_ANCHOR_FLOAT32 = float(
    np.add(
        np.float32(GRIPPER_CLOSED_TARGET_FLOAT32),
        np.float32(GRIPPER_TARGET_SLEW_FLOAT32_TOLERANCE_RAD),
        dtype=np.float32,
    )
)
GRIPPER_MAX_TARGET_STEP_FLOAT32 = float(
    np.multiply(
        np.float32(GRIPPER_TARGET_SLEW_RATE_RAD_S_FLOAT32),
        np.float32(GRIPPER_TARGET_SLEW_PHYSICS_DT),
        dtype=np.float32,
    )
)
GRIPPER_MAX_TARGET_STEP_0P25_FLOAT32 = float(
    np.multiply(
        np.float32(GRIPPER_TARGET_SLEW_RATE_0P25_RAD_S_FLOAT32),
        np.float32(GRIPPER_TARGET_SLEW_PHYSICS_DT),
        dtype=np.float32,
    )
)


@dataclass(frozen=True)
class _TargetSlewCloseSimulation:
    endpoint_apply_count: int
    limited_apply_count: int
    nextafter_correction_count: int


def _simulate_close_transition(
    max_target_step_rad: float,
) -> _TargetSlewCloseSimulation:
    """Mirror the bounded production float32 transition through its endpoint."""

    endpoint = np.float32(GRIPPER_CLOSED_TARGET_FLOAT32)
    maximum_step = np.float32(max_target_step_rad)
    previous = np.float32(GRIPPER_OPEN_TARGET_FLOAT32)
    apply_count = 0
    limited_count = 0
    nextafter_count = 0
    while previous != endpoint:
        if apply_count >= 1024:
            raise RuntimeError("EEF gripper close-transition simulation did not end")
        delta = np.subtract(endpoint, previous, dtype=np.float32)
        limited = bool(np.abs(delta) > maximum_step)
        step = np.clip(delta, -maximum_step, maximum_step).astype(np.float32)
        candidate = np.add(previous, step, dtype=np.float32)
        next_target = candidate if limited else endpoint
        applied_step = np.subtract(next_target, previous, dtype=np.float32)
        if np.abs(applied_step) > maximum_step:
            next_target = np.nextafter(
                next_target,
                previous,
                dtype=np.float32,
            )
            nextafter_count += 1
            applied_step = np.subtract(next_target, previous, dtype=np.float32)
        if not (
            np.isfinite(next_target)
            and np.isfinite(applied_step)
            and np.abs(applied_step) <= maximum_step
            and previous <= next_target <= endpoint
        ):
            raise RuntimeError("EEF gripper close-transition simulation invariant")
        previous = next_target
        apply_count += 1
        limited_count += int(limited)
    return _TargetSlewCloseSimulation(
        endpoint_apply_count=apply_count,
        limited_apply_count=limited_count,
        nextafter_correction_count=nextafter_count,
    )


GRIPPER_CLOSE_SETTLE_SUBSTEPS = 10
_GRIPPER_CLOSE_TRANSITION = _simulate_close_transition(GRIPPER_MAX_TARGET_STEP_FLOAT32)
_GRIPPER_CLOSE_TRANSITION_0P25 = _simulate_close_transition(
    GRIPPER_MAX_TARGET_STEP_0P25_FLOAT32
)
GRIPPER_CLOSE_TRANSITION_APPLIES = _GRIPPER_CLOSE_TRANSITION.endpoint_apply_count
GRIPPER_CLOSE_TRANSITION_0P25_APPLIES = (
    _GRIPPER_CLOSE_TRANSITION_0P25.endpoint_apply_count
)
if _GRIPPER_CLOSE_TRANSITION != _TargetSlewCloseSimulation(38, 37, 15):
    raise RuntimeError("Baseline EEF gripper close-transition count drift")
if _GRIPPER_CLOSE_TRANSITION_0P25 != _TargetSlewCloseSimulation(76, 75, 41):
    raise RuntimeError("Rate-0.25 EEF gripper close-transition count drift")


@dataclass(frozen=True)
class EefGripperTargetSlewProfileSpec:
    """One closed, float32-derived EEF target-slew/interlock profile."""

    profile: str
    action_class: str
    rate_factor_float32: float
    rate_rad_s_float32: float
    max_target_step_rad_float32: float
    close_transition_applies: int
    close_limited_applies: int
    close_nextafter_corrections: int
    close_interlock_profile: str
    close_interlock_substeps: int
    fixed_activation_anchor: bool


def _target_slew_profile_spec(
    *,
    profile: str,
    rate_factor_float32: float,
    rate_rad_s_float32: float,
    max_target_step_rad_float32: float,
    close_transition_applies: int,
    close_nextafter_corrections: int,
    close_interlock_profile: str | None = None,
    fixed_activation_anchor: bool = False,
) -> EefGripperTargetSlewProfileSpec:
    close_interlock_substeps = close_transition_applies + GRIPPER_CLOSE_SETTLE_SUBSTEPS
    return EefGripperTargetSlewProfileSpec(
        profile=profile,
        action_class=EEF_GRIPPER_TARGET_SLEW_ACTION_CLASS,
        rate_factor_float32=rate_factor_float32,
        rate_rad_s_float32=rate_rad_s_float32,
        max_target_step_rad_float32=max_target_step_rad_float32,
        close_transition_applies=close_transition_applies,
        close_limited_applies=close_transition_applies - 1,
        close_nextafter_corrections=close_nextafter_corrections,
        close_interlock_profile=(
            close_interlock_profile
            or f"eef_gripper_close_hold_arm_{close_interlock_substeps}_physics_substeps_v1"
        ),
        close_interlock_substeps=close_interlock_substeps,
        fixed_activation_anchor=fixed_activation_anchor,
    )


_EEF_GRIPPER_TARGET_SLEW_PROFILES = MappingProxyType(
    {
        EEF_GRIPPER_TARGET_SLEW_PROFILE: _target_slew_profile_spec(
            profile=EEF_GRIPPER_TARGET_SLEW_PROFILE,
            rate_factor_float32=GRIPPER_TARGET_SLEW_RATE_FACTOR_FLOAT32,
            rate_rad_s_float32=GRIPPER_TARGET_SLEW_RATE_RAD_S_FLOAT32,
            max_target_step_rad_float32=GRIPPER_MAX_TARGET_STEP_FLOAT32,
            close_transition_applies=GRIPPER_CLOSE_TRANSITION_APPLIES,
            close_nextafter_corrections=(
                _GRIPPER_CLOSE_TRANSITION.nextafter_correction_count
            ),
        ),
        EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE: (
            _target_slew_profile_spec(
                profile=EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE,
                rate_factor_float32=(GRIPPER_TARGET_SLEW_RATE_0P25_FACTOR_FLOAT32),
                rate_rad_s_float32=GRIPPER_TARGET_SLEW_RATE_0P25_RAD_S_FLOAT32,
                max_target_step_rad_float32=(GRIPPER_MAX_TARGET_STEP_0P25_FLOAT32),
                close_transition_applies=(GRIPPER_CLOSE_TRANSITION_0P25_APPLIES),
                close_nextafter_corrections=(
                    _GRIPPER_CLOSE_TRANSITION_0P25.nextafter_correction_count
                ),
                close_interlock_profile=(
                    "eef_gripper_close_fixed_activation_anchor_86_physics_substeps_v2"
                ),
                fixed_activation_anchor=True,
            )
        ),
    }
)


def eef_gripper_target_slew_profile(
    profile: str,
) -> EefGripperTargetSlewProfileSpec:
    """Resolve one exact target-slew profile from the closed mapping."""

    if type(profile) is not str or profile not in _EEF_GRIPPER_TARGET_SLEW_PROFILES:
        raise ValueError(
            f"Unknown PolaRiS EEF gripper target-slew profile: {profile!r}"
        )
    return _EEF_GRIPPER_TARGET_SLEW_PROFILES[profile]


def select_eef_gripper_target_slew_profile(
    *, enable_rate_0p25_candidate: bool
) -> EefGripperTargetSlewProfileSpec:
    """Select the baseline or the sole default-off rate candidate."""

    if type(enable_rate_0p25_candidate) is not bool:
        raise ValueError("EEF gripper rate-0.25 candidate flag must be bool")
    profile = (
        EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
        if enable_rate_0p25_candidate
        else EEF_GRIPPER_TARGET_SLEW_PROFILE
    )
    return eef_gripper_target_slew_profile(profile)


def validate_eef_gripper_close_arm_interlock_binding(
    *,
    target_slew_profile: str,
    interlock_profile: str,
    configured_substeps: int,
) -> EefGripperTargetSlewProfileSpec:
    """Bind one interlock duration to its exact installed target-slew profile."""

    spec = eef_gripper_target_slew_profile(target_slew_profile)
    if (
        type(interlock_profile) is not str
        or interlock_profile != spec.close_interlock_profile
        or type(configured_substeps) is not int
        or configured_substeps != spec.close_interlock_substeps
    ):
        raise ValueError(
            "PolaRiS EEF gripper close-interlock/target-slew profile mismatch: "
            f"target_slew={target_slew_profile!r}, "
            f"interlock={interlock_profile!r}, substeps={configured_substeps!r}"
        )
    return spec


EXPECTED_FULL_VELOCITY_LIMITS_BEFORE_WRITE = (
    *EXPECTED_ARM_VELOCITY_LIMITS_FLOAT32,
    GRIPPER_DRIVER_VELOCITY_LIMIT_FLOAT32,
    *([GRIPPER_FOLLOWER_DEFAULT_VELOCITY_LIMIT_FLOAT32] * 5),
)
EXPECTED_FULL_VELOCITY_LIMITS_AFTER_WRITE = (
    *EXPECTED_ARM_VELOCITY_LIMITS_FLOAT32,
    *([GRIPPER_DRIVER_VELOCITY_LIMIT_FLOAT32] * 6),
)

PINNED_DYNAMIC_DEVICE = "cuda:0"
PINNED_STATIC_PHYSX_DEVICE = "cpu"
PINNED_ACTUATOR_DEVICE = "cuda:0"
PINNED_TENSOR_DTYPE = "torch.float32"
EXPECTED_ROBOT_USD_SHA256 = (
    "d8379925b103963dbf3e7c85bcc4ae101b81b7c1d7dabe7d2e964f41d069ec44"
)
EXPECTED_ROBOT_USD_SIZE_BYTES = 14_156_155

EXPECTED_MIMIC_JOINT_SPECS = (
    ("right_outer_knuckle_joint", 8, "rotZ", -1.0, 1_000_000.0, 0.0),
    ("left_inner_finger_joint", 9, "rotX", 1.0, 1_000.0, 0.05000000074505806),
    ("right_inner_finger_joint", 10, "rotX", -1.0, 1_000.0, 0.05000000074505806),
    ("left_inner_finger_knuckle_joint", 11, "rotX", 1.0, 1_000.0, 0.05000000074505806),
    ("right_inner_finger_knuckle_joint", 12, "rotX", 1.0, 1_000.0, 0.05000000074505806),
)

TENSOR_EVIDENCE_FIELDS = {
    "dtype",
    "device",
    "shape",
    "values",
    "finite_mask",
    "finite_count",
}
MIMIC_JOINT_CONTRACT_FIELDS = {
    "profile",
    "robot_usd_sha256",
    "driver_joint_name",
    "driver_joint_index",
    "driver_joint_prim_path",
    "driver_physics_joint_type",
    "driver_exclude_from_articulation",
    "followers",
}
MIMIC_JOINT_ENTRY_FIELDS = {
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
MIMIC_COMPLIANCE_CALLABLE_IDENTITY_FIELDS = {"module", "qualname", "name"}
MIMIC_COMPLIANCE_SNAPSHOT_FIELDS = {
    "natural_frequency_rad_s",
    "damping_ratio",
}
MIMIC_COMPLIANCE_STRUCTURE_FIELDS = {
    "applied_mimic_api",
    "reference_joint_path",
    "gearing",
    "offset",
    "exclude_from_articulation",
}
MIMIC_COMPLIANCE_FOLLOWER_FIELDS = {
    "joint_name",
    "joint_index",
    "live_prim_path",
    "mimic_axis",
    "natural_frequency_attribute",
    "damping_ratio_attribute",
    "source",
    "before_spawn_write",
    "before_spawn_structure",
    "natural_frequency_write_count",
    "damping_ratio_write_count",
    "after_spawn_write",
    "after_spawn_structure",
    "post_reset_composed_usd_readback",
    "post_reset_composed_usd_structure",
}
MIMIC_COMPLIANCE_CONTRACT_FIELDS = {
    "profile",
    "enabled",
    "scope",
    "timing",
    "setter",
    "live_root_profile",
    "live_root_path",
    "original_spawn_func",
    "overlay_func",
    "original_spawn_call_count",
    "overlay_call_count",
    "physics_hz",
    "physics_dt",
    "target_natural_frequency_rad_s",
    "target_damping_ratio",
    "frequency_timestep_product",
    "follower_count",
    "natural_frequency_write_count",
    "damping_ratio_write_count",
    "total_write_count",
    "source_usd_sha256",
    "source_usd_unchanged_after_spawn_overlay",
    "followers",
}
WRITE_CONTRACT_FIELDS = {
    "profile",
    "setter",
    "timing",
    "call_count",
    "articulation_indices",
    "full_input",
}
STATIC_CONTRACT_FIELDS = {
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
MIMIC_COMPLIANCE_STATIC_FIELD = "mimic_compliance"
DRIVER_ACTUATOR_FIELDS = {
    "cfg_velocity_limit",
    "cfg_velocity_limit_sim",
    "cfg_effort_limit",
    "cfg_effort_limit_sim",
    "resolved_velocity_limit",
    "resolved_velocity_limit_sim",
    "resolved_effort_limit",
    "resolved_effort_limit_sim",
}
DYNAMIC_EVIDENCE_FIELDS = {
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
TARGET_SLEW_STATIC_FIELDS = {
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
TARGET_SLEW_DYNAMIC_FIELDS = {
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
MAX_VELOCITY_DIAGNOSTIC_FIELDS = {
    "sample_phase",
    "sample_index",
    "joint_position_rad",
    "joint_velocity_rad_s",
    "joint_acceleration_rad_s2",
    "joint_position_target_rad",
    "joint_velocity_target_rad_s",
}
TERMINAL_STATE_FIELDS = {
    "sample_index",
    "joint_position_rad",
    "joint_velocity_rad_s",
    "joint_acceleration_rad_s2",
    "joint_position_target_rad",
    "joint_velocity_target_rad_s",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _typed_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _typed_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _typed_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _same_float32(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or not isinstance(left, (int, float)):
        return False
    return bool(np.float32(left).tobytes() == np.float32(right).tobytes())


def _flat_values(value: Any) -> list[Any]:
    current = value
    for method in ("detach", "cpu"):
        operation = getattr(current, method, None)
        if callable(operation):
            current = operation()
    operation = getattr(current, "tolist", None)
    if callable(operation):
        current = operation()
    return np.asarray(current, dtype=object).reshape(-1).tolist()


def tensor_evidence(value: Any) -> dict[str, Any]:
    raw = _flat_values(value)
    values: list[float | None] = []
    finite_mask: list[bool] = []
    for item in raw:
        number = float(item)
        finite = math.isfinite(number)
        finite_mask.append(finite)
        values.append(number if finite else None)
    return {
        "dtype": str(getattr(value, "dtype", "missing")),
        "device": str(getattr(value, "device", "missing")),
        "shape": list(getattr(value, "shape", ())),
        "values": values,
        "finite_mask": finite_mask,
        "finite_count": sum(finite_mask),
    }


def validate_tensor_evidence(
    value: Any,
    *,
    field: str,
    shape: Sequence[int],
    device: str,
    expected: Sequence[float] | None = None,
) -> dict[str, Any]:
    _require(
        isinstance(value, dict) and set(value) == TENSOR_EVIDENCE_FIELDS,
        f"{field} schema",
    )
    _require(
        type(value["dtype"]) is str
        and type(value["device"]) is str
        and isinstance(value["shape"], list)
        and all(type(item) is int for item in value["shape"])
        and value["dtype"] == PINNED_TENSOR_DTYPE
        and value["device"] == device
        and value["shape"] == list(shape),
        f"{field} dtype/device/shape",
    )
    count = int(np.prod(shape))
    _require(
        isinstance(value["finite_mask"], list)
        and all(type(item) is bool for item in value["finite_mask"])
        and value["finite_mask"] == [True] * count
        and type(value["finite_count"]) is int
        and value["finite_count"] == count
        and isinstance(value["values"], list)
        and len(value["values"]) == count,
        f"{field} finiteness",
    )
    if expected is not None:
        _require(
            len(expected) == count
            and all(
                _same_float32(actual, wanted)
                for actual, wanted in zip(value["values"], expected, strict=True)
            ),
            f"{field} values",
        )
    return dict(value)


def _joint_indices_list(joint_ids: Any, *, joint_count: int) -> list[int]:
    if isinstance(joint_ids, slice):
        return list(range(joint_count))[joint_ids]
    current = joint_ids
    for method in ("detach", "cpu"):
        operation = getattr(current, method, None)
        if callable(operation):
            current = operation()
    operation = getattr(current, "tolist", None)
    if callable(operation):
        current = operation()
    _require(
        isinstance(current, (list, tuple))
        and all(type(item) is int for item in current),
        "joint indices exact integer sequence",
    )
    return list(current)


def _validated_live_joint_owner(
    *, owner: str, joint_names: Any, joint_ids: Any, live_joint_names: Sequence[str]
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    _require(
        isinstance(joint_names, (list, tuple))
        and all(type(name) is str for name in joint_names),
        f"{owner} joint names",
    )
    names = tuple(joint_names)
    indices = tuple(_joint_indices_list(joint_ids, joint_count=len(live_joint_names)))
    _require(
        len(names) == len(indices)
        and len(set(names)) == len(names)
        and len(set(indices)) == len(indices),
        f"{owner} ownership uniqueness",
    )
    _require(
        all(0 <= index < len(live_joint_names) for index in indices)
        and all(
            live_joint_names[index] == name
            for name, index in zip(names, indices, strict=True)
        ),
        f"{owner} name/index pairing",
    )
    _require(
        not set(names).intersection(GRIPPER_FOLLOWER_JOINT_NAMES)
        and not set(indices).intersection(GRIPPER_FOLLOWER_JOINT_INDICES),
        f"passive follower unexpectedly owned by {owner}",
    )
    return names, indices


def _expected_mimic_joint_contract() -> dict[str, Any]:
    root = "/panda/Gripper/Robotiq_2F_85/Joints"
    driver_path = f"{root}/{DRIVEN_GRIPPER_JOINT_NAME}"
    return {
        "profile": EEF_GRIPPER_MIMIC_PROFILE,
        "robot_usd_sha256": EXPECTED_ROBOT_USD_SHA256,
        "driver_joint_name": DRIVEN_GRIPPER_JOINT_NAME,
        "driver_joint_index": DRIVEN_GRIPPER_JOINT_INDEX,
        "driver_joint_prim_path": driver_path,
        "driver_physics_joint_type": "PhysicsRevoluteJoint",
        "driver_exclude_from_articulation": False,
        "followers": [
            {
                "joint_name": name,
                "joint_index": index,
                "prim_path": f"{root}/{name}",
                "physics_joint_type": "PhysicsRevoluteJoint",
                "exclude_from_articulation": False,
                "mimic_axis": axis,
                "reference_joint_path": driver_path,
                "gearing": gearing,
                "natural_frequency_hz": frequency,
                "damping_ratio": damping,
            }
            for name, index, axis, gearing, frequency, damping in EXPECTED_MIMIC_JOINT_SPECS
        ],
    }


def validate_mimic_joint_contract(value: Any) -> dict[str, Any]:
    _require(
        isinstance(value, dict) and set(value) == MIMIC_JOINT_CONTRACT_FIELDS,
        "mimic contract schema",
    )
    followers = value.get("followers")
    _require(
        isinstance(followers, list)
        and all(
            isinstance(item, dict) and set(item) == MIMIC_JOINT_ENTRY_FIELDS
            for item in followers
        ),
        "mimic follower schema",
    )
    _require(
        _typed_equal(value, _expected_mimic_joint_contract()),
        "mimic source-USD contract drift",
    )
    return dict(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_exact_mimic_api_instance(prim: Any, *, axis: str, field: str) -> str:
    get_applied_schemas = getattr(prim, "GetAppliedSchemas", None)
    _require(callable(get_applied_schemas), f"{field} applied schemas")
    applied = [str(token) for token in get_applied_schemas()]
    expected = f"PhysxMimicJointAPI:{axis}"
    mimic_instances = [
        token for token in applied if token.startswith("PhysxMimicJointAPI:")
    ]
    _require(
        mimic_instances == [expected],
        f"{field} exact applied mimic API instance drift",
    )
    return expected


def _capture_mimic_joint_contract(robot_usd_path: Path) -> dict[str, Any]:
    from pxr import Usd  # noqa: PLC0415

    path = Path(robot_usd_path).resolve()
    _require(
        path.stat().st_size == EXPECTED_ROBOT_USD_SIZE_BYTES
        and _file_sha256(path) == EXPECTED_ROBOT_USD_SHA256,
        "robot USD identity drift",
    )
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadNone)
    _require(stage is not None, "cannot open robot USD")
    expected = _expected_mimic_joint_contract()

    def static_joint(prim_path: str) -> tuple[Any, bool]:
        prim = stage.GetPrimAtPath(prim_path)
        _require(prim and prim.IsValid(), f"missing joint prim {prim_path}")
        excluded = prim.GetAttribute("physics:excludeFromArticulation").Get()
        _require(type(excluded) is bool, f"invalid joint exclusion {prim_path}")
        return prim, excluded

    driver, driver_excluded = static_joint(expected["driver_joint_prim_path"])
    result = {
        "profile": EEF_GRIPPER_MIMIC_PROFILE,
        "robot_usd_sha256": EXPECTED_ROBOT_USD_SHA256,
        "driver_joint_name": DRIVEN_GRIPPER_JOINT_NAME,
        "driver_joint_index": DRIVEN_GRIPPER_JOINT_INDEX,
        "driver_joint_prim_path": str(driver.GetPath()),
        "driver_physics_joint_type": driver.GetTypeName(),
        "driver_exclude_from_articulation": driver_excluded,
        "followers": [],
    }
    for specification in expected["followers"]:
        prim, excluded = static_joint(specification["prim_path"])
        _validate_exact_mimic_api_instance(
            prim,
            axis=specification["mimic_axis"],
            field=f"source mimic {specification['joint_name']}",
        )
        namespace = f"physxMimicJoint:{specification['mimic_axis']}"
        references = prim.GetRelationship(f"{namespace}:referenceJoint").GetTargets()
        _require(
            len(references) == 1, f"mimic reference count {specification['joint_name']}"
        )
        result["followers"].append(
            {
                "joint_name": specification["joint_name"],
                "joint_index": specification["joint_index"],
                "prim_path": str(prim.GetPath()),
                "physics_joint_type": prim.GetTypeName(),
                "exclude_from_articulation": excluded,
                "mimic_axis": specification["mimic_axis"],
                "reference_joint_path": str(references[0]),
                "gearing": float(prim.GetAttribute(f"{namespace}:gearing").Get()),
                "natural_frequency_hz": float(
                    prim.GetAttribute(f"{namespace}:naturalFrequency").Get()
                ),
                "damping_ratio": float(
                    prim.GetAttribute(f"{namespace}:dampingRatio").Get()
                ),
            }
        )
    return validate_mimic_joint_contract(result)


def _callable_identity(value: Any) -> dict[str, str]:
    _require(callable(value), "mimic compliance spawn callable")
    identity = {
        "module": getattr(value, "__module__", None),
        "qualname": getattr(value, "__qualname__", None),
        "name": getattr(value, "__name__", None),
    }
    _require(
        set(identity) == MIMIC_COMPLIANCE_CALLABLE_IDENTITY_FIELDS
        and all(type(item) is str and bool(item) for item in identity.values()),
        "mimic compliance spawn callable identity",
    )
    return identity


def _expected_original_spawn_func() -> Any:
    from isaaclab.sim.spawners.from_files.from_files import (  # noqa: PLC0415
        spawn_from_usd,
    )

    return spawn_from_usd


def _validate_original_spawn_func(value: Any) -> dict[str, str]:
    _require(
        value is _expected_original_spawn_func(),
        "mimic compliance original spawn callable object drift",
    )
    identity = _callable_identity(value)
    _require(
        _typed_equal(
            identity,
            EEF_GRIPPER_MIMIC_COMPLIANCE_ORIGINAL_SPAWN_IDENTITY,
        ),
        "mimic compliance original spawn callable identity drift",
    )
    return identity


def _current_live_stage_and_robot_roots(prim_path: str) -> tuple[Any, list[Any]]:
    import omni.usd  # noqa: PLC0415
    import isaaclab.sim as sim_utils  # noqa: PLC0415

    _require(type(prim_path) is str and bool(prim_path), "live robot prim expression")
    stage = omni.usd.get_context().get_stage()
    _require(stage is not None, "mimic compliance live USD stage")
    roots = list(sim_utils.find_matching_prims(prim_path, stage=stage))
    roots.sort(key=lambda prim: prim.GetPath().pathString)
    return stage, roots


def _prim_path_string(prim: Any) -> str:
    path = prim.GetPath()
    value = getattr(path, "pathString", None)
    if value is None:
        value = str(path)
    _require(type(value) is str and value.startswith("/"), "live prim path")
    return value


def _mimic_attribute_names(axis: str) -> tuple[str, str]:
    _require(axis in {"rotX", "rotY", "rotZ"}, "live mimic axis")
    namespace = f"physxMimicJoint:{axis}"
    return f"{namespace}:naturalFrequency", f"{namespace}:dampingRatio"


def _mimic_float_attribute(prim: Any, name: str, *, field: str) -> Any:
    attribute = prim.GetAttribute(name)
    _require(bool(attribute), f"missing {field} attribute")
    get_type_name = getattr(attribute, "GetTypeName", None)
    _require(callable(get_type_name), f"missing {field} attribute type")
    _require(str(get_type_name()) == "float", f"{field} attribute type drift")
    return attribute


def _mimic_bool_attribute(prim: Any, name: str, *, field: str) -> Any:
    attribute = prim.GetAttribute(name)
    _require(bool(attribute), f"missing {field} attribute")
    get_type_name = getattr(attribute, "GetTypeName", None)
    _require(callable(get_type_name), f"missing {field} attribute type")
    _require(str(get_type_name()) == "bool", f"{field} attribute type drift")
    return attribute


def _mimic_numeric_value(attribute: Any, *, field: str) -> float:
    value = attribute.Get()
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{field} finite scalar",
    )
    return float(value)


def _mimic_live_structure(
    *,
    prim: Any,
    specification: Mapping[str, Any],
    live_root: str,
) -> dict[str, Any]:
    axis = specification["mimic_axis"]
    applied_api = _validate_exact_mimic_api_instance(
        prim,
        axis=axis,
        field=f"live mimic {specification['joint_name']}",
    )
    namespace = f"physxMimicJoint:{axis}"
    relationship = prim.GetRelationship(f"{namespace}:referenceJoint")
    _require(bool(relationship), f"live mimic {specification['joint_name']} reference")
    targets = list(relationship.GetTargets())
    _require(
        len(targets) == 1,
        f"live mimic {specification['joint_name']} reference count",
    )
    target_path = getattr(targets[0], "pathString", None)
    if target_path is None:
        target_path = str(targets[0])
    expected_driver_path = live_root + _expected_mimic_joint_contract()[
        "driver_joint_prim_path"
    ].removeprefix("/panda")
    gearing_attribute = _mimic_float_attribute(
        prim,
        f"{namespace}:gearing",
        field=f"{specification['joint_name']} gearing",
    )
    offset_attribute = _mimic_float_attribute(
        prim,
        f"{namespace}:offset",
        field=f"{specification['joint_name']} offset",
    )
    exclusion_attribute = _mimic_bool_attribute(
        prim,
        "physics:excludeFromArticulation",
        field=f"{specification['joint_name']} exclusion",
    )
    gearing = _mimic_numeric_value(
        gearing_attribute,
        field=f"{specification['joint_name']} gearing",
    )
    offset = _mimic_numeric_value(
        offset_attribute,
        field=f"{specification['joint_name']} offset",
    )
    excluded = exclusion_attribute.Get()
    _require(
        type(target_path) is str
        and target_path == expected_driver_path
        and _same_float32(gearing, specification["gearing"])
        and _same_float32(offset, 0.0)
        and type(excluded) is bool
        and excluded is specification["exclude_from_articulation"],
        f"live mimic {specification['joint_name']} source structure drift",
    )
    return {
        "applied_mimic_api": applied_api,
        "reference_joint_path": target_path,
        "gearing": gearing,
        "offset": offset,
        "exclude_from_articulation": excluded,
    }


def _mimic_snapshot(
    *, natural_frequency: float, damping_ratio: float
) -> dict[str, float]:
    return {
        "natural_frequency_rad_s": float(natural_frequency),
        "damping_ratio": float(damping_ratio),
    }


def _live_mimic_bindings(
    *, stage: Any, roots: Sequence[Any], source_contract: Mapping[str, Any]
) -> tuple[str, list[dict[str, Any]]]:
    """Resolve the five composed follower attributes without authoring values."""

    source = validate_mimic_joint_contract(dict(source_contract))
    _require(len(roots) == 1, "mimic compliance live robot root count")
    live_root = _prim_path_string(roots[0])
    _require(
        live_root == EEF_GRIPPER_MIMIC_COMPLIANCE_EXPECTED_LIVE_ROOT,
        "mimic compliance live robot root path drift",
    )
    bindings: list[dict[str, Any]] = []
    for specification in source["followers"]:
        source_path = specification["prim_path"]
        _require(
            type(source_path) is str and source_path.startswith("/panda/"),
            "mimic compliance source follower path",
        )
        live_path = live_root + source_path.removeprefix("/panda")
        prim = stage.GetPrimAtPath(live_path)
        _require(
            bool(prim)
            and prim.IsValid()
            and _prim_path_string(prim) == live_path
            and prim.GetTypeName() == "PhysicsRevoluteJoint",
            f"mimic compliance live follower prim drift: {live_path}",
        )
        frequency_name, damping_name = _mimic_attribute_names(
            specification["mimic_axis"]
        )
        frequency_attribute = _mimic_float_attribute(
            prim,
            frequency_name,
            field=f"{specification['joint_name']} natural frequency",
        )
        damping_attribute = _mimic_float_attribute(
            prim,
            damping_name,
            field=f"{specification['joint_name']} damping ratio",
        )
        structure = _mimic_live_structure(
            prim=prim,
            specification=specification,
            live_root=live_root,
        )
        bindings.append(
            {
                "source": specification,
                "prim": prim,
                "live_prim_path": live_path,
                "natural_frequency_attribute_name": frequency_name,
                "damping_ratio_attribute_name": damping_name,
                "natural_frequency_attribute": frequency_attribute,
                "damping_ratio_attribute": damping_attribute,
                "structure": structure,
            }
        )
    _require(
        len(bindings) == EEF_GRIPPER_MIMIC_COMPLIANCE_FOLLOWER_COUNT,
        "mimic compliance follower count",
    )
    return live_root, bindings


def _read_live_mimic_snapshots(
    *, stage: Any, roots: Sequence[Any], source_contract: Mapping[str, Any]
) -> tuple[str, list[dict[str, Any]]]:
    live_root, bindings = _live_mimic_bindings(
        stage=stage,
        roots=roots,
        source_contract=source_contract,
    )
    snapshots: list[dict[str, Any]] = []
    for binding in bindings:
        snapshots.append(
            {
                "joint_name": binding["source"]["joint_name"],
                "joint_index": binding["source"]["joint_index"],
                "live_prim_path": binding["live_prim_path"],
                "mimic_axis": binding["source"]["mimic_axis"],
                "natural_frequency_attribute": binding[
                    "natural_frequency_attribute_name"
                ],
                "damping_ratio_attribute": binding["damping_ratio_attribute_name"],
                "snapshot": _mimic_snapshot(
                    natural_frequency=_mimic_numeric_value(
                        binding["natural_frequency_attribute"],
                        field=f"{binding['source']['joint_name']} natural frequency",
                    ),
                    damping_ratio=_mimic_numeric_value(
                        binding["damping_ratio_attribute"],
                        field=f"{binding['source']['joint_name']} damping ratio",
                    ),
                ),
                "structure": copy.deepcopy(binding["structure"]),
            }
        )
    return live_root, snapshots


def _write_spawned_mimic_compliance(
    *, stage: Any, roots: Sequence[Any], source_contract: Mapping[str, Any]
) -> tuple[str, list[dict[str, Any]]]:
    """Author the ten candidate values before articulation initialization."""

    live_root, bindings = _live_mimic_bindings(
        stage=stage,
        roots=roots,
        source_contract=source_contract,
    )
    # Phase 1: validate all five source states before authoring any opinion.
    prepared: list[dict[str, Any]] = []
    for binding in bindings:
        source = binding["source"]
        before = _mimic_snapshot(
            natural_frequency=_mimic_numeric_value(
                binding["natural_frequency_attribute"],
                field=f"{source['joint_name']} pre-write natural frequency",
            ),
            damping_ratio=_mimic_numeric_value(
                binding["damping_ratio_attribute"],
                field=f"{source['joint_name']} pre-write damping ratio",
            ),
        )
        _require(
            _same_float32(
                before["natural_frequency_rad_s"],
                source["natural_frequency_hz"],
            )
            and _same_float32(before["damping_ratio"], source["damping_ratio"]),
            f"mimic compliance pre-write/source drift: {source['joint_name']}",
        )
        prepared.append(
            {
                "binding": binding,
                "before": before,
                "before_structure": copy.deepcopy(binding["structure"]),
            }
        )

    # Phase 2: perform exactly two writes per fully validated follower.
    for item in prepared:
        binding = item["binding"]
        source = binding["source"]
        frequency_result = binding["natural_frequency_attribute"].Set(
            EEF_GRIPPER_MIMIC_COMPLIANCE_NATURAL_FREQUENCY_RAD_S_FLOAT32
        )
        damping_result = binding["damping_ratio_attribute"].Set(
            EEF_GRIPPER_MIMIC_COMPLIANCE_DAMPING_RATIO_FLOAT32
        )
        _require(
            type(frequency_result) is bool
            and frequency_result is True
            and type(damping_result) is bool
            and damping_result is True,
            f"mimic compliance USD write failed: {source['joint_name']}",
        )

    # Phase 3: read all values and untouched structure after all ten writes.
    followers: list[dict[str, Any]] = []
    for item in prepared:
        binding = item["binding"]
        source = binding["source"]
        before = item["before"]
        after = _mimic_snapshot(
            natural_frequency=_mimic_numeric_value(
                binding["natural_frequency_attribute"],
                field=f"{source['joint_name']} post-write natural frequency",
            ),
            damping_ratio=_mimic_numeric_value(
                binding["damping_ratio_attribute"],
                field=f"{source['joint_name']} post-write damping ratio",
            ),
        )
        _require(
            _same_float32(
                after["natural_frequency_rad_s"],
                EEF_GRIPPER_MIMIC_COMPLIANCE_NATURAL_FREQUENCY_RAD_S_FLOAT32,
            )
            and _same_float32(
                after["damping_ratio"],
                EEF_GRIPPER_MIMIC_COMPLIANCE_DAMPING_RATIO_FLOAT32,
            ),
            f"mimic compliance post-write readback drift: {source['joint_name']}",
        )
        after_structure = _mimic_live_structure(
            prim=binding["prim"],
            specification=source,
            live_root=live_root,
        )
        _require(
            _typed_equal(after_structure, item["before_structure"]),
            f"mimic compliance post-write structure drift: {source['joint_name']}",
        )
        followers.append(
            {
                "joint_name": source["joint_name"],
                "joint_index": source["joint_index"],
                "live_prim_path": binding["live_prim_path"],
                "mimic_axis": source["mimic_axis"],
                "natural_frequency_attribute": binding[
                    "natural_frequency_attribute_name"
                ],
                "damping_ratio_attribute": binding["damping_ratio_attribute_name"],
                "source": _mimic_snapshot(
                    natural_frequency=source["natural_frequency_hz"],
                    damping_ratio=source["damping_ratio"],
                ),
                "before_spawn_write": before,
                "before_spawn_structure": item["before_structure"],
                "natural_frequency_write_count": 1,
                "damping_ratio_write_count": 1,
                "after_spawn_write": after,
                "after_spawn_structure": after_structure,
                "post_reset_composed_usd_readback": None,
                "post_reset_composed_usd_structure": None,
            }
        )
    return live_root, followers


def _validate_mimic_compliance_snapshot(
    value: Any,
    *,
    expected_natural_frequency: float,
    expected_damping_ratio: float,
    field: str,
) -> dict[str, Any]:
    _require(
        isinstance(value, dict) and set(value) == MIMIC_COMPLIANCE_SNAPSHOT_FIELDS,
        f"{field} schema",
    )
    _require(
        _same_float32(value.get("natural_frequency_rad_s"), expected_natural_frequency)
        and _same_float32(value.get("damping_ratio"), expected_damping_ratio),
        f"{field} values",
    )
    return dict(value)


def _validate_mimic_compliance_structure(
    value: Any,
    *,
    source: Mapping[str, Any],
    field: str,
) -> dict[str, Any]:
    _require(
        isinstance(value, dict) and set(value) == MIMIC_COMPLIANCE_STRUCTURE_FIELDS,
        f"{field} schema",
    )
    expected_reference = (
        EEF_GRIPPER_MIMIC_COMPLIANCE_EXPECTED_LIVE_ROOT
        + _expected_mimic_joint_contract()["driver_joint_prim_path"].removeprefix(
            "/panda"
        )
    )
    _require(
        value.get("applied_mimic_api") == f"PhysxMimicJointAPI:{source['mimic_axis']}"
        and value.get("reference_joint_path") == expected_reference
        and _same_float32(value.get("gearing"), source["gearing"])
        and _same_float32(value.get("offset"), 0.0)
        and type(value.get("exclude_from_articulation")) is bool
        and value.get("exclude_from_articulation")
        is source["exclude_from_articulation"],
        f"{field} values",
    )
    return dict(value)


def validate_eef_gripper_mimic_compliance(
    value: Any,
    *,
    source_contract: Mapping[str, Any],
    require_post_reset_composed_usd_readback: bool = True,
) -> dict[str, Any]:
    """Validate the candidate-only pre-PhysX mimic-compliance transaction."""

    _require(
        type(require_post_reset_composed_usd_readback) is bool,
        "mimic compliance post-reset requirement",
    )
    source = validate_mimic_joint_contract(dict(source_contract))
    _require(
        isinstance(value, dict) and set(value) == MIMIC_COMPLIANCE_CONTRACT_FIELDS,
        "mimic compliance contract schema",
    )
    exact = {
        "profile": EEF_GRIPPER_MIMIC_COMPLIANCE_PROFILE,
        "enabled": True,
        "scope": EEF_GRIPPER_MIMIC_COMPLIANCE_SCOPE,
        "timing": EEF_GRIPPER_MIMIC_COMPLIANCE_TIMING,
        "setter": EEF_GRIPPER_MIMIC_COMPLIANCE_SETTER,
        "live_root_profile": EEF_GRIPPER_MIMIC_COMPLIANCE_LIVE_ROOT_PROFILE,
        "live_root_path": EEF_GRIPPER_MIMIC_COMPLIANCE_EXPECTED_LIVE_ROOT,
        "original_spawn_func": EEF_GRIPPER_MIMIC_COMPLIANCE_ORIGINAL_SPAWN_IDENTITY,
        "overlay_func": EEF_GRIPPER_MIMIC_COMPLIANCE_OVERLAY_IDENTITY,
        "original_spawn_call_count": 1,
        "overlay_call_count": 1,
        "follower_count": EEF_GRIPPER_MIMIC_COMPLIANCE_FOLLOWER_COUNT,
        "natural_frequency_write_count": (EEF_GRIPPER_MIMIC_COMPLIANCE_FOLLOWER_COUNT),
        "damping_ratio_write_count": EEF_GRIPPER_MIMIC_COMPLIANCE_FOLLOWER_COUNT,
        "total_write_count": EEF_GRIPPER_MIMIC_COMPLIANCE_TOTAL_WRITE_COUNT,
        "source_usd_sha256": EXPECTED_ROBOT_USD_SHA256,
        "source_usd_unchanged_after_spawn_overlay": True,
    }
    for name, expected in exact.items():
        _require(
            _typed_equal(value.get(name), expected),
            f"mimic compliance {name} drift",
        )
    for name, expected in {
        "physics_hz": EEF_GRIPPER_MIMIC_COMPLIANCE_PHYSICS_HZ,
        "physics_dt": EEF_GRIPPER_MIMIC_COMPLIANCE_PHYSICS_DT,
        "frequency_timestep_product": (
            EEF_GRIPPER_MIMIC_COMPLIANCE_FREQUENCY_TIMESTEP_PRODUCT
        ),
    }.items():
        actual = value.get(name)
        _require(
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isfinite(float(actual))
            and float(actual) == float(expected),
            f"mimic compliance {name} drift",
        )
    _require(
        _same_float32(
            value.get("target_natural_frequency_rad_s"),
            EEF_GRIPPER_MIMIC_COMPLIANCE_NATURAL_FREQUENCY_RAD_S_FLOAT32,
        )
        and _same_float32(
            value.get("target_damping_ratio"),
            EEF_GRIPPER_MIMIC_COMPLIANCE_DAMPING_RATIO_FLOAT32,
        ),
        "mimic compliance target float32 drift",
    )
    _require(
        value["physics_dt"] * value["target_natural_frequency_rad_s"]
        == value["frequency_timestep_product"],
        "mimic compliance frequency/timestep product drift",
    )
    followers = value.get("followers")
    _require(
        isinstance(followers, list)
        and len(followers) == EEF_GRIPPER_MIMIC_COMPLIANCE_FOLLOWER_COUNT,
        "mimic compliance follower list",
    )
    for index, follower in enumerate(followers):
        _require(
            isinstance(follower, dict)
            and set(follower) == MIMIC_COMPLIANCE_FOLLOWER_FIELDS,
            f"mimic compliance follower {index} schema",
        )
        expected_source = source["followers"][index]
        frequency_name, damping_name = _mimic_attribute_names(
            expected_source["mimic_axis"]
        )
        expected_live_path = (
            EEF_GRIPPER_MIMIC_COMPLIANCE_EXPECTED_LIVE_ROOT
            + expected_source["prim_path"].removeprefix("/panda")
        )
        expected_identity = {
            "joint_name": expected_source["joint_name"],
            "joint_index": expected_source["joint_index"],
            "live_prim_path": expected_live_path,
            "mimic_axis": expected_source["mimic_axis"],
            "natural_frequency_attribute": frequency_name,
            "damping_ratio_attribute": damping_name,
            "natural_frequency_write_count": 1,
            "damping_ratio_write_count": 1,
        }
        for name, expected in expected_identity.items():
            _require(
                type(follower.get(name)) is type(expected)
                and follower.get(name) == expected,
                f"mimic compliance follower {index} {name} drift",
            )
        _validate_mimic_compliance_snapshot(
            follower.get("source"),
            expected_natural_frequency=expected_source["natural_frequency_hz"],
            expected_damping_ratio=expected_source["damping_ratio"],
            field=f"mimic compliance follower {index} source",
        )
        _validate_mimic_compliance_snapshot(
            follower.get("before_spawn_write"),
            expected_natural_frequency=expected_source["natural_frequency_hz"],
            expected_damping_ratio=expected_source["damping_ratio"],
            field=f"mimic compliance follower {index} before",
        )
        before_structure = _validate_mimic_compliance_structure(
            follower.get("before_spawn_structure"),
            source=expected_source,
            field=f"mimic compliance follower {index} before structure",
        )
        _validate_mimic_compliance_snapshot(
            follower.get("after_spawn_write"),
            expected_natural_frequency=(
                EEF_GRIPPER_MIMIC_COMPLIANCE_NATURAL_FREQUENCY_RAD_S_FLOAT32
            ),
            expected_damping_ratio=(EEF_GRIPPER_MIMIC_COMPLIANCE_DAMPING_RATIO_FLOAT32),
            field=f"mimic compliance follower {index} after",
        )
        after_structure = _validate_mimic_compliance_structure(
            follower.get("after_spawn_structure"),
            source=expected_source,
            field=f"mimic compliance follower {index} after structure",
        )
        _require(
            _typed_equal(before_structure, after_structure),
            f"mimic compliance follower {index} spawn structure changed",
        )
        post_reset = follower.get("post_reset_composed_usd_readback")
        post_reset_composed_usd_structure = follower.get(
            "post_reset_composed_usd_structure"
        )
        if require_post_reset_composed_usd_readback:
            _validate_mimic_compliance_snapshot(
                post_reset,
                expected_natural_frequency=(
                    EEF_GRIPPER_MIMIC_COMPLIANCE_NATURAL_FREQUENCY_RAD_S_FLOAT32
                ),
                expected_damping_ratio=(
                    EEF_GRIPPER_MIMIC_COMPLIANCE_DAMPING_RATIO_FLOAT32
                ),
                field=f"mimic compliance follower {index} post-reset",
            )
            validated_post_reset_composed_usd_structure = _validate_mimic_compliance_structure(
                post_reset_composed_usd_structure,
                source=expected_source,
                field=f"mimic compliance follower {index} post-reset composed-USD structure",
            )
            _require(
                _typed_equal(
                    validated_post_reset_composed_usd_structure,
                    before_structure,
                ),
                f"mimic compliance follower {index} post-reset composed-USD structure changed",
            )
        else:
            _require(
                post_reset is None and post_reset_composed_usd_structure is None,
                f"mimic compliance follower {index} premature post-reset composed-USD readback",
            )
    return dict(value)


def configure_eef_gripper_mimic_compliance_spawn_overlay(
    spawn_cfg: Any,
    *,
    target_slew_profile: str,
) -> Any:
    """Install the sole candidate overlay while leaving baseline config untouched."""

    profile = eef_gripper_target_slew_profile(target_slew_profile)
    original_func = getattr(spawn_cfg, "func", None)
    original_identity = _validate_original_spawn_func(original_func)
    if profile.profile == EEF_GRIPPER_TARGET_SLEW_PROFILE:
        return original_func
    _require(
        profile.profile == EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE,
        "mimic compliance target-slew profile binding",
    )

    def overlay(
        prim_path: str,
        cfg: Any,
        translation: tuple[float, float, float] | None = None,
        orientation: tuple[float, float, float, float] | None = None,
        **kwargs: Any,
    ) -> Any:
        _require(
            getattr(overlay, "_eef_mimic_compliance_overlay_call_count") == 0,
            "mimic compliance spawn overlay called more than once",
        )
        overlay._eef_mimic_compliance_overlay_call_count = 1
        source_before = _capture_mimic_joint_contract(Path(cfg.usd_path))
        result = original_func(
            prim_path,
            cfg,
            translation=translation,
            orientation=orientation,
            **kwargs,
        )
        overlay._eef_mimic_compliance_original_spawn_call_count = 1
        stage, roots = _current_live_stage_and_robot_roots(prim_path)
        live_root, followers = _write_spawned_mimic_compliance(
            stage=stage,
            roots=roots,
            source_contract=source_before,
        )
        source_after = _capture_mimic_joint_contract(Path(cfg.usd_path))
        _require(
            _typed_equal(source_before, source_after),
            "mimic compliance source USD changed during live overlay",
        )
        evidence = {
            "profile": EEF_GRIPPER_MIMIC_COMPLIANCE_PROFILE,
            "enabled": True,
            "scope": EEF_GRIPPER_MIMIC_COMPLIANCE_SCOPE,
            "timing": EEF_GRIPPER_MIMIC_COMPLIANCE_TIMING,
            "setter": EEF_GRIPPER_MIMIC_COMPLIANCE_SETTER,
            "live_root_profile": EEF_GRIPPER_MIMIC_COMPLIANCE_LIVE_ROOT_PROFILE,
            "live_root_path": live_root,
            "original_spawn_func": original_identity,
            "overlay_func": _callable_identity(overlay),
            "original_spawn_call_count": 1,
            "overlay_call_count": 1,
            "physics_hz": EEF_GRIPPER_MIMIC_COMPLIANCE_PHYSICS_HZ,
            "physics_dt": EEF_GRIPPER_MIMIC_COMPLIANCE_PHYSICS_DT,
            "target_natural_frequency_rad_s": (
                EEF_GRIPPER_MIMIC_COMPLIANCE_NATURAL_FREQUENCY_RAD_S_FLOAT32
            ),
            "target_damping_ratio": (
                EEF_GRIPPER_MIMIC_COMPLIANCE_DAMPING_RATIO_FLOAT32
            ),
            "frequency_timestep_product": (
                EEF_GRIPPER_MIMIC_COMPLIANCE_FREQUENCY_TIMESTEP_PRODUCT
            ),
            "follower_count": EEF_GRIPPER_MIMIC_COMPLIANCE_FOLLOWER_COUNT,
            "natural_frequency_write_count": (
                EEF_GRIPPER_MIMIC_COMPLIANCE_FOLLOWER_COUNT
            ),
            "damping_ratio_write_count": (EEF_GRIPPER_MIMIC_COMPLIANCE_FOLLOWER_COUNT),
            "total_write_count": EEF_GRIPPER_MIMIC_COMPLIANCE_TOTAL_WRITE_COUNT,
            "source_usd_sha256": EXPECTED_ROBOT_USD_SHA256,
            "source_usd_unchanged_after_spawn_overlay": True,
            "followers": followers,
        }
        validate_eef_gripper_mimic_compliance(
            evidence,
            source_contract=source_after,
            require_post_reset_composed_usd_readback=False,
        )
        overlay._eef_mimic_compliance_source_before = copy.deepcopy(source_before)
        overlay._eef_mimic_compliance_source_after = copy.deepcopy(source_after)
        overlay._eef_mimic_compliance_spawn_evidence = copy.deepcopy(evidence)
        return result

    overlay.__name__ = EEF_GRIPPER_MIMIC_COMPLIANCE_OVERLAY_IDENTITY["name"]
    overlay.__qualname__ = EEF_GRIPPER_MIMIC_COMPLIANCE_OVERLAY_IDENTITY["qualname"]
    overlay._eef_mimic_compliance_target_slew_profile = profile.profile
    overlay._eef_mimic_compliance_original_func = original_func
    overlay._eef_mimic_compliance_original_spawn_call_count = 0
    overlay._eef_mimic_compliance_overlay_call_count = 0
    overlay._eef_mimic_compliance_source_before = None
    overlay._eef_mimic_compliance_source_after = None
    overlay._eef_mimic_compliance_spawn_evidence = None
    _require(
        _typed_equal(
            _callable_identity(overlay),
            EEF_GRIPPER_MIMIC_COMPLIANCE_OVERLAY_IDENTITY,
        ),
        "mimic compliance overlay callable identity drift",
    )
    spawn_cfg.func = overlay
    _require(spawn_cfg.func is overlay, "mimic compliance overlay installation failed")
    return overlay


def _post_reset_mimic_compliance_contract(
    *, robot: Any, source_contract: Mapping[str, Any]
) -> dict[str, Any]:
    overlay = getattr(getattr(robot.cfg, "spawn", None), "func", None)
    _require(
        _typed_equal(
            _callable_identity(overlay),
            EEF_GRIPPER_MIMIC_COMPLIANCE_OVERLAY_IDENTITY,
        ),
        "mimic compliance installed overlay identity drift",
    )
    original = getattr(overlay, "_eef_mimic_compliance_original_func", None)
    _validate_original_spawn_func(original)
    _require(
        getattr(overlay, "_eef_mimic_compliance_target_slew_profile", None)
        == EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
        and type(
            getattr(overlay, "_eef_mimic_compliance_original_spawn_call_count", None)
        )
        is int
        and getattr(overlay, "_eef_mimic_compliance_original_spawn_call_count") == 1
        and type(getattr(overlay, "_eef_mimic_compliance_overlay_call_count", None))
        is int
        and getattr(overlay, "_eef_mimic_compliance_overlay_call_count") == 1,
        "mimic compliance overlay lifecycle drift",
    )
    source = validate_mimic_joint_contract(dict(source_contract))
    source_before = getattr(overlay, "_eef_mimic_compliance_source_before", None)
    source_after = getattr(overlay, "_eef_mimic_compliance_source_after", None)
    _require(
        _typed_equal(source_before, source) and _typed_equal(source_after, source),
        "mimic compliance source USD identity drift after reset",
    )
    evidence = copy.deepcopy(
        getattr(overlay, "_eef_mimic_compliance_spawn_evidence", None)
    )
    validate_eef_gripper_mimic_compliance(
        evidence,
        source_contract=source,
        require_post_reset_composed_usd_readback=False,
    )
    stage, roots = _current_live_stage_and_robot_roots(robot.cfg.prim_path)
    live_root, snapshots = _read_live_mimic_snapshots(
        stage=stage,
        roots=roots,
        source_contract=source,
    )
    _require(
        live_root == evidence["live_root_path"]
        and len(snapshots) == len(evidence["followers"]),
        "mimic compliance post-reset live identity drift",
    )
    for follower, snapshot in zip(evidence["followers"], snapshots, strict=True):
        identity_fields = {
            "joint_name",
            "joint_index",
            "live_prim_path",
            "mimic_axis",
            "natural_frequency_attribute",
            "damping_ratio_attribute",
        }
        _require(
            all(follower[name] == snapshot[name] for name in identity_fields),
            "mimic compliance post-reset follower identity drift",
        )
        follower["post_reset_composed_usd_readback"] = snapshot["snapshot"]
        follower["post_reset_composed_usd_structure"] = snapshot["structure"]
    return validate_eef_gripper_mimic_compliance(
        evidence,
        source_contract=source,
        require_post_reset_composed_usd_readback=True,
    )


def _validate_post_reset_mimic_compliance_readback(
    *,
    robot: Any,
    source_contract: Mapping[str, Any],
    expected_contract: Mapping[str, Any],
) -> None:
    expected = validate_eef_gripper_mimic_compliance(
        dict(expected_contract),
        source_contract=source_contract,
        require_post_reset_composed_usd_readback=True,
    )
    current = _post_reset_mimic_compliance_contract(
        robot=robot,
        source_contract=source_contract,
    )
    _require(
        _typed_equal(current, expected),
        "mimic compliance later-reset readback drift",
    )


def _direct_static_physx_tensor(robot: Any, getter_name: str) -> Any:
    getter = getattr(robot.root_physx_view, getter_name, None)
    _require(callable(getter), f"missing PhysX getter {getter_name}")
    tensor = getter()
    _require(
        str(getattr(tensor, "device", "missing")) == PINNED_STATIC_PHYSX_DEVICE
        and str(getattr(tensor, "dtype", "missing")) == PINNED_TENSOR_DTYPE
        and list(getattr(tensor, "shape", ())) == [1, len(EXPECTED_DROID_JOINT_NAMES)],
        f"{getter_name} CPU float32 shape drift",
    )
    return tensor.clone() if hasattr(tensor, "clone") else tensor


def _validate_live_ownership(
    robot: Any, arm_term: Any, finger_term: Any
) -> dict[str, Any]:
    joint_names = tuple(robot.joint_names)
    _require(
        joint_names == EXPECTED_DROID_JOINT_NAMES, "live articulation joint order drift"
    )
    arm = _validated_live_joint_owner(
        owner="arm action term",
        joint_names=getattr(arm_term, "_joint_names", None),
        joint_ids=getattr(arm_term, "_joint_ids", None),
        live_joint_names=joint_names,
    )
    finger = _validated_live_joint_owner(
        owner="finger action term",
        joint_names=getattr(finger_term, "_joint_names", None),
        joint_ids=getattr(finger_term, "_joint_ids", None),
        live_joint_names=joint_names,
    )
    _require(
        arm == (EXPECTED_ARM_JOINT_NAMES, EXPECTED_ARM_JOINT_INDICES),
        "arm ownership drift",
    )
    _require(
        finger == ((DRIVEN_GRIPPER_JOINT_NAME,), (DRIVEN_GRIPPER_JOINT_INDEX,)),
        "driver action ownership drift",
    )
    _require(isinstance(robot.actuators, Mapping), "missing actuator mapping")
    ownership: dict[str, Any] = {}
    for name, actuator in robot.actuators.items():
        names, indices = _validated_live_joint_owner(
            owner=f"actuator {name}",
            joint_names=getattr(actuator, "joint_names", None),
            joint_ids=getattr(actuator, "joint_indices", None),
            live_joint_names=joint_names,
        )
        ownership[name] = {"joint_names": list(names), "joint_indices": list(indices)}
    _require(
        set(ownership) == set(EXPECTED_ACTUATOR_JOINT_OWNERSHIP), "actuator key drift"
    )
    for name, expected in EXPECTED_ACTUATOR_JOINT_OWNERSHIP.items():
        _require(
            ownership[name]
            == {"joint_names": list(expected[0]), "joint_indices": list(expected[1])},
            f"actuator ownership drift {name}",
        )
    return ownership


def _cfg_scalar(value: Any) -> float | None:
    if value is None:
        return None
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        "actuator cfg scalar",
    )
    return float(value)


def _capture_driver_actuator(robot: Any) -> dict[str, Any]:
    actuator = robot.actuators["gripper"]
    fields = {
        "cfg_velocity_limit": _cfg_scalar(actuator.cfg.velocity_limit),
        "cfg_velocity_limit_sim": _cfg_scalar(actuator.cfg.velocity_limit_sim),
        "cfg_effort_limit": _cfg_scalar(actuator.cfg.effort_limit),
        "cfg_effort_limit_sim": _cfg_scalar(actuator.cfg.effort_limit_sim),
        "resolved_velocity_limit": tensor_evidence(actuator.velocity_limit),
        "resolved_velocity_limit_sim": tensor_evidence(actuator.velocity_limit_sim),
        "resolved_effort_limit": tensor_evidence(actuator.effort_limit),
        "resolved_effort_limit_sim": tensor_evidence(actuator.effort_limit_sim),
    }
    _require(set(fields) == DRIVER_ACTUATOR_FIELDS, "driver actuator schema")
    for name in DRIVER_ACTUATOR_FIELDS - {
        "cfg_velocity_limit",
        "cfg_velocity_limit_sim",
        "cfg_effort_limit",
        "cfg_effort_limit_sim",
    }:
        expected = 200.0 if "effort" in name else 5.0
        validate_tensor_evidence(
            fields[name],
            field=f"driver actuator {name}",
            shape=(1, 1),
            device=PINNED_ACTUATOR_DEVICE,
            expected=(expected,),
        )
    _require(
        fields["cfg_velocity_limit"] == 5.0
        and fields["cfg_velocity_limit_sim"] == 5.0
        and fields["cfg_effort_limit"] == 200.0
        and fields["cfg_effort_limit_sim"] == 200.0,
        "driver configured limit drift",
    )
    return fields


def validate_eef_gripper_target_slew_static(
    value: Any,
    *,
    expected_profile: str = EEF_GRIPPER_TARGET_SLEW_PROFILE,
) -> dict[str, Any]:
    """Validate the closed EEF-only driver target-slew identity."""

    profile = eef_gripper_target_slew_profile(expected_profile)
    _require(
        isinstance(value, dict) and set(value) == TARGET_SLEW_STATIC_FIELDS,
        "gripper target-slew static schema",
    )
    exact = {
        "profile": profile.profile,
        "scope": "eef_pose_only_native_joint_position_unchanged_v1",
        "action_class": profile.action_class,
        "driver_joint_name": DRIVEN_GRIPPER_JOINT_NAME,
        "driver_joint_index": DRIVEN_GRIPPER_JOINT_INDEX,
        "endpoint_semantics_profile": GRIPPER_THRESHOLD_PROFILE,
        "physical_velocity_limit_source": (
            "live_implicit_actuator_velocity_limit_sim_float32_v1"
        ),
        "target_slew_rate_source": GRIPPER_TARGET_SLEW_RATE_SOURCE,
        "reset_profile": EEF_GRIPPER_TARGET_SLEW_RESET_PROFILE,
        "tensor_dtype": PINNED_TENSOR_DTYPE,
        "tensor_device": PINNED_ACTUATOR_DEVICE,
    }
    for field, expected in exact.items():
        _require(
            type(value.get(field)) is type(expected) and value.get(field) == expected,
            f"gripper target-slew static {field} drift",
        )
    numeric = {
        "open_target_rad": GRIPPER_OPEN_TARGET_FLOAT32,
        "closed_target_rad": GRIPPER_CLOSED_TARGET_FLOAT32,
        "physical_velocity_limit_rad_s": GRIPPER_DRIVER_VELOCITY_LIMIT_FLOAT32,
        "target_slew_rate_factor": profile.rate_factor_float32,
        "target_slew_rate_rad_s": profile.rate_rad_s_float32,
        "physics_hz": GRIPPER_TARGET_SLEW_PHYSICS_HZ,
        "physics_dt": GRIPPER_TARGET_SLEW_PHYSICS_DT,
        "max_target_step_rad": profile.max_target_step_rad_float32,
        "float32_tolerance_rad": GRIPPER_TARGET_SLEW_FLOAT32_TOLERANCE_RAD,
    }
    for field, expected in numeric.items():
        actual = value.get(field)
        _require(
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isfinite(float(actual)),
            f"gripper target-slew static {field} finite scalar",
        )
        if field in {
            "open_target_rad",
            "closed_target_rad",
            "physical_velocity_limit_rad_s",
            "target_slew_rate_factor",
            "target_slew_rate_rad_s",
            "max_target_step_rad",
        }:
            _require(
                _same_float32(actual, expected),
                f"gripper target-slew static {field} float32 drift",
            )
        else:
            _require(
                float(actual) == float(expected),
                f"gripper target-slew static {field} drift",
            )
    recomputed_rate = float(
        np.multiply(
            np.float32(value["physical_velocity_limit_rad_s"]),
            np.float32(value["target_slew_rate_factor"]),
            dtype=np.float32,
        )
    )
    _require(
        _same_float32(recomputed_rate, value["target_slew_rate_rad_s"]),
        "gripper target-slew physical-limit/factor/rate binding drift",
    )
    recomputed_step = float(
        np.multiply(
            np.float32(value["target_slew_rate_rad_s"]),
            np.float32(value["physics_dt"]),
            dtype=np.float32,
        )
    )
    _require(
        _same_float32(recomputed_step, value["max_target_step_rad"]),
        "gripper target-slew rate/cadence cap binding drift",
    )
    return dict(value)


def validate_eef_gripper_target_slew_dynamic(
    value: Any,
    *,
    expected_profile: str = EEF_GRIPPER_TARGET_SLEW_PROFILE,
) -> dict[str, Any]:
    """Validate one reset-isolated target-slew counter/maximum report."""

    profile = eef_gripper_target_slew_profile(expected_profile)
    _require(
        isinstance(value, dict) and set(value) == TARGET_SLEW_DYNAMIC_FIELDS,
        "gripper target-slew dynamic schema",
    )
    _require(
        value.get("profile") == profile.profile,
        "gripper target-slew dynamic profile drift",
    )
    counter_fields = TARGET_SLEW_DYNAMIC_FIELDS - {
        "profile",
        "max_abs_target_step_rad",
        "max_abs_endpoint_error_before_step_rad",
        "max_abs_endpoint_error_after_step_rad",
        "initial_anchor_rad",
        "last_requested_endpoint_rad",
        "last_applied_target_rad",
    }
    for field in counter_fields:
        _require(
            type(value[field]) is int and value[field] >= 0,
            f"gripper target-slew dynamic {field}",
        )
    process_calls = value["process_action_calls"]
    apply_calls = value["apply_calls"]
    _require(
        (process_calls == 0 and apply_calls == 0)
        or (
            process_calls >= 1
            and max(
                (process_calls - 1) * GRIPPER_APPLY_ENTRY_SAMPLES_PER_POLICY_STEP, 0
            )
            <= apply_calls
            <= process_calls * GRIPPER_APPLY_ENTRY_SAMPLES_PER_POLICY_STEP
        ),
        "gripper target-slew process/apply cadence drift",
    )
    _require(
        value["endpoint_change_count"] + value["repeated_endpoint_process_count"]
        == max(process_calls - 1, 0),
        "gripper target-slew process history drift",
    )
    _require(
        value["slew_limited_apply_count"] + value["endpoint_reached_apply_count"]
        == apply_calls,
        "gripper target-slew apply classification drift",
    )
    _require(
        value["live_limit_validation_count"] == apply_calls,
        "gripper target-slew live-limit validation cadence drift",
    )
    maxima = {
        field: value[field]
        for field in (
            "max_abs_target_step_rad",
            "max_abs_endpoint_error_before_step_rad",
            "max_abs_endpoint_error_after_step_rad",
        )
    }
    for field, scalar in maxima.items():
        _require(
            isinstance(scalar, (int, float))
            and not isinstance(scalar, bool)
            and math.isfinite(float(scalar))
            and float(scalar) >= 0.0,
            f"gripper target-slew dynamic {field}",
        )
    _require(
        float(maxima["max_abs_target_step_rad"])
        <= profile.max_target_step_rad_float32
        + GRIPPER_TARGET_SLEW_FLOAT32_TOLERANCE_RAD,
        "gripper target-slew maximum target step exceeds cap",
    )
    _require(
        float(maxima["max_abs_endpoint_error_before_step_rad"])
        <= GRIPPER_TARGET_SLEW_MAX_ANCHOR_FLOAT32
        and float(maxima["max_abs_endpoint_error_after_step_rad"])
        <= float(maxima["max_abs_endpoint_error_before_step_rad"])
        + GRIPPER_TARGET_SLEW_FLOAT32_TOLERANCE_RAD,
        "gripper target-slew endpoint error maxima drift",
    )
    endpoint = value["last_requested_endpoint_rad"]
    if process_calls == 0:
        _require(
            endpoint is None
            and value["endpoint_change_count"] == 0
            and value["repeated_endpoint_process_count"] == 0,
            "empty gripper target-slew process evidence",
        )
    else:
        _require(
            endpoint is not None
            and (
                _same_float32(endpoint, GRIPPER_OPEN_TARGET_FLOAT32)
                or _same_float32(endpoint, GRIPPER_CLOSED_TARGET_FLOAT32)
            ),
            "gripper target-slew final endpoint drift",
        )
    anchor = value["initial_anchor_rad"]
    applied = value["last_applied_target_rad"]
    if apply_calls == 0:
        _require(
            value["initialization_count"] == 0
            and anchor is None
            and applied is None
            and all(_same_float32(scalar, 0.0) for scalar in maxima.values()),
            "empty gripper target-slew apply evidence",
        )
    else:
        _require(
            process_calls >= 1
            and value["initialization_count"] == 1
            and all(
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(float(item))
                for item in (anchor, applied)
            )
            and GRIPPER_TARGET_SLEW_MIN_ANCHOR_FLOAT32
            <= float(anchor)
            <= GRIPPER_TARGET_SLEW_MAX_ANCHOR_FLOAT32
            and GRIPPER_TARGET_SLEW_MIN_ANCHOR_FLOAT32
            <= float(applied)
            <= GRIPPER_TARGET_SLEW_MAX_ANCHOR_FLOAT32,
            "gripper target-slew initialized state drift",
        )
    return dict(value)


def validate_eef_gripper_static_contract(
    value: Any,
    *,
    expected_target_slew_profile: str = EEF_GRIPPER_TARGET_SLEW_PROFILE,
) -> dict[str, Any]:
    target_slew_spec = eef_gripper_target_slew_profile(expected_target_slew_profile)
    expected_fields = set(STATIC_CONTRACT_FIELDS)
    if target_slew_spec.profile == EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE:
        expected_fields.add(MIMIC_COMPLIANCE_STATIC_FIELD)
    _require(
        isinstance(value, dict) and set(value) == expected_fields,
        "gripper static schema",
    )
    exact = {
        "profile": EEF_GRIPPER_RUNTIME_PROFILE,
        "joint_names": list(EXPECTED_DROID_JOINT_NAMES),
        "gripper_joint_names": list(GRIPPER_JOINT_NAMES),
        "gripper_joint_indices": list(GRIPPER_JOINT_INDICES),
        "driver_joint_name": DRIVEN_GRIPPER_JOINT_NAME,
        "driver_joint_index": DRIVEN_GRIPPER_JOINT_INDEX,
        "follower_joint_names": list(GRIPPER_FOLLOWER_JOINT_NAMES),
        "follower_joint_indices": list(GRIPPER_FOLLOWER_JOINT_INDICES),
        "device_partition": {
            "profile": EEF_GRIPPER_DEVICE_PARTITION_PROFILE,
            "dynamic_articulation": PINNED_DYNAMIC_DEVICE,
            "implicit_actuator": PINNED_ACTUATOR_DEVICE,
            "static_physx": PINNED_STATIC_PHYSX_DEVICE,
            "dtype": PINNED_TENSOR_DTYPE,
        },
        "measured_velocity_is_hard_bounded_by_limit": False,
    }
    for field, expected in exact.items():
        _require(
            _typed_equal(value.get(field), expected), f"gripper static {field} drift"
        )
    expected_ownership = {
        name: {"joint_names": list(joints), "joint_indices": list(indices)}
        for name, (joints, indices) in EXPECTED_ACTUATOR_JOINT_OWNERSHIP.items()
    }
    _require(
        _typed_equal(value.get("actuator_joint_ownership"), expected_ownership),
        "actuator ownership drift",
    )
    _require(isinstance(value.get("driver_actuator"), dict), "driver actuator missing")
    driver = value["driver_actuator"]
    _require(set(driver) == DRIVER_ACTUATOR_FIELDS, "driver actuator fields")
    for name in DRIVER_ACTUATOR_FIELDS - {
        "cfg_velocity_limit",
        "cfg_velocity_limit_sim",
        "cfg_effort_limit",
        "cfg_effort_limit_sim",
    }:
        validate_tensor_evidence(
            driver[name],
            field=f"driver {name}",
            shape=(1, 1),
            device=PINNED_ACTUATOR_DEVICE,
            expected=((200.0 if "effort" in name else 5.0),),
        )
    _require(
        driver["cfg_velocity_limit"] == driver["cfg_velocity_limit_sim"] == 5.0
        and driver["cfg_effort_limit"] == driver["cfg_effort_limit_sim"] == 200.0,
        "driver cfg evidence drift",
    )
    mimic_contract = validate_mimic_joint_contract(value.get("mimic_joint_contract"))
    if target_slew_spec.profile == EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE:
        validate_eef_gripper_mimic_compliance(
            value.get(MIMIC_COMPLIANCE_STATIC_FIELD),
            source_contract=mimic_contract,
            require_post_reset_composed_usd_readback=True,
        )
    before = validate_tensor_evidence(
        value.get("velocity_limits_before_write"),
        field="velocity limits before write",
        shape=(1, 13),
        device=PINNED_STATIC_PHYSX_DEVICE,
        expected=EXPECTED_FULL_VELOCITY_LIMITS_BEFORE_WRITE,
    )
    after = validate_tensor_evidence(
        value.get("velocity_limits_after_write"),
        field="velocity limits after write",
        shape=(1, 13),
        device=PINNED_STATIC_PHYSX_DEVICE,
        expected=EXPECTED_FULL_VELOCITY_LIMITS_AFTER_WRITE,
    )
    write = value.get("velocity_limit_write_contract")
    _require(
        isinstance(write, dict) and set(write) == WRITE_CONTRACT_FIELDS,
        "write contract schema",
    )
    _require(
        write["profile"] == EEF_GRIPPER_VELOCITY_WRITE_PROFILE
        and write["setter"] == EEF_GRIPPER_VELOCITY_WRITE_SETTER
        and write["timing"] == EEF_GRIPPER_VELOCITY_WRITE_TIMING
        and type(write["call_count"]) is int
        and write["call_count"] == 1
        and write["articulation_indices"] == [0],
        "write identity drift",
    )
    full_input = validate_tensor_evidence(
        write["full_input"],
        field="velocity setter input",
        shape=(1, 13),
        device=PINNED_STATIC_PHYSX_DEVICE,
        expected=EXPECTED_FULL_VELOCITY_LIMITS_AFTER_WRITE,
    )
    _require(_typed_equal(full_input, after), "setter input/readback drift")
    _require(
        all(
            _same_float32(before["values"][index], after["values"][index])
            for index in range(8)
        )
        and all(
            _same_float32(after["values"][index], 5.0)
            for index in GRIPPER_FOLLOWER_JOINT_INDICES
        ),
        "write changed a nonfollower or missed a follower",
    )
    validate_eef_gripper_target_slew_static(
        value.get("driver_target_slew"),
        expected_profile=target_slew_spec.profile,
    )
    return dict(value)


def validate_eef_gripper_dynamic_evidence(
    value: Any,
    *,
    expected_target_slew_profile: str = EEF_GRIPPER_TARGET_SLEW_PROFILE,
) -> dict[str, Any]:
    _require(
        isinstance(value, dict) and set(value) == DYNAMIC_EVIDENCE_FIELDS,
        "gripper dynamic schema",
    )
    _require(
        value["profile"] == EEF_GRIPPER_RUNTIME_PROFILE
        and value["joint_names"] == list(GRIPPER_JOINT_NAMES)
        and value["joint_indices"] == list(GRIPPER_JOINT_INDICES),
        "gripper dynamic identity",
    )
    validate_eef_gripper_target_slew_dynamic(
        value.get("driver_target_slew"),
        expected_profile=expected_target_slew_profile,
    )
    for field in (
        "apply_entry_samples",
        "post_policy_step_samples",
        "nonfinite_samples",
        "dropped_diagnostics",
    ):
        _require(
            type(value[field]) is int and value[field] >= 0, f"gripper dynamic {field}"
        )
    _require(value["dropped_diagnostics"] == 0, "incomplete gripper evidence")
    for field in ("max_abs_joint_velocity_rad_s", "max_abs_joint_acceleration_rad_s2"):
        vector = value[field]
        _require(
            isinstance(vector, list)
            and len(vector) == 6
            and all(
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(float(item))
                and item >= 0
                for item in vector
            ),
            f"gripper dynamic {field}",
        )
    diagnostic = value["max_velocity_diagnostic"]
    total_samples = value["apply_entry_samples"] + value["post_policy_step_samples"]
    _require(
        value["nonfinite_samples"] <= total_samples,
        "gripper nonfinite sample cadence",
    )
    finite_samples = total_samples - value["nonfinite_samples"]
    if total_samples == 0:
        _require(
            diagnostic is None
            and value["terminal_state"] is None
            and all(
                _same_float32(item, 0.0)
                for field in (
                    "max_abs_joint_velocity_rad_s",
                    "max_abs_joint_acceleration_rad_s2",
                )
                for item in value[field]
            ),
            "empty gripper evidence",
        )
        return dict(value)
    if finite_samples == 0:
        _require(
            diagnostic is None
            and value["terminal_state"] is None
            and all(
                _same_float32(item, 0.0)
                for field in (
                    "max_abs_joint_velocity_rad_s",
                    "max_abs_joint_acceleration_rad_s2",
                )
                for item in value[field]
            ),
            "all-nonfinite gripper evidence",
        )
        return dict(value)
    _require(
        isinstance(diagnostic, dict)
        and set(diagnostic) == MAX_VELOCITY_DIAGNOSTIC_FIELDS,
        "max velocity diagnostic",
    )
    _require(
        diagnostic["sample_phase"] in {"apply_entry", "post_policy_step"}
        and type(diagnostic["sample_index"]) is int
        and 0 <= diagnostic["sample_index"] < total_samples,
        "max velocity diagnostic identity",
    )
    for field in MAX_VELOCITY_DIAGNOSTIC_FIELDS - {"sample_phase", "sample_index"}:
        vector = diagnostic[field]
        _require(
            isinstance(vector, list)
            and len(vector) == 6
            and all(
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(float(item))
                for item in vector
            ),
            f"max velocity diagnostic {field}",
        )
    diagnostic_velocity_max = max(
        abs(float(item)) for item in diagnostic["joint_velocity_rad_s"]
    )
    aggregate_velocity_max = max(value["max_abs_joint_velocity_rad_s"])
    _require(
        _same_float32(diagnostic_velocity_max, aggregate_velocity_max),
        "max velocity diagnostic/aggregate drift",
    )
    terminal = value["terminal_state"]
    if value["post_policy_step_samples"] == 0:
        _require(terminal is None, "terminal state without post-step sample")
    else:
        _require(
            isinstance(terminal, dict) and set(terminal) == TERMINAL_STATE_FIELDS,
            "terminal gripper schema",
        )
        expected_terminal_sample_index = (
            value["post_policy_step_samples"]
            * GRIPPER_INTERLEAVED_SAMPLES_PER_POLICY_STEP
            - 1
        )
        _require(
            type(terminal["sample_index"]) is int
            and terminal["sample_index"] == expected_terminal_sample_index
            and terminal["sample_index"] < total_samples,
            "terminal sample index",
        )
        for field in TERMINAL_STATE_FIELDS - {"sample_index"}:
            vector = terminal[field]
            _require(
                isinstance(vector, list)
                and len(vector) == 6
                and all(
                    isinstance(item, (int, float))
                    and not isinstance(item, bool)
                    and math.isfinite(float(item))
                    for item in vector
                ),
                f"terminal gripper {field}",
            )
    return dict(value)


def install_eef_gripper_runtime(env: Any, *, robot_usd_path: Path) -> dict[str, Any]:
    """Perform the sole full-tensor follower write after the first reset."""

    import torch  # noqa: PLC0415

    runtime = getattr(env, "unwrapped", env)
    robot = runtime.scene["robot"]
    terms = runtime.action_manager._terms
    _require(list(terms) == ["arm", "finger_joint"], "EEF action order drift")
    arm_term = terms["arm"]
    finger_term = terms["finger_joint"]
    _require(
        type(finger_term).__name__ == EEF_GRIPPER_TARGET_SLEW_ACTION_CLASS,
        "EEF target-slew action class drift",
    )
    target_slew_reporter = getattr(
        finger_term, "gripper_target_slew_static_contract", None
    )
    target_slew_installer = getattr(
        finger_term, "install_gripper_target_slew_contract", None
    )
    _require(
        callable(target_slew_reporter) and callable(target_slew_installer),
        "EEF finger action lacks target-slew runtime methods",
    )
    target_slew_contract = target_slew_reporter()
    _require(
        isinstance(target_slew_contract, dict),
        "EEF finger action returned no target-slew contract",
    )
    target_slew_profile = target_slew_contract.get("profile")
    target_slew_spec = eef_gripper_target_slew_profile(target_slew_profile)
    validate_eef_gripper_target_slew_static(
        target_slew_contract,
        expected_profile=target_slew_spec.profile,
    )
    ownership = _validate_live_ownership(robot, arm_term, finger_term)
    driver = _capture_driver_actuator(robot)
    mimic = _capture_mimic_joint_contract(Path(robot_usd_path))
    mimic_compliance = None
    if target_slew_spec.profile == EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE:
        mimic_compliance = _post_reset_mimic_compliance_contract(
            robot=robot,
            source_contract=mimic,
        )
    before_tensor = _direct_static_physx_tensor(robot, "get_dof_max_velocities")
    before = validate_tensor_evidence(
        tensor_evidence(before_tensor),
        field="pre-write PhysX velocity limits",
        shape=(1, 13),
        device=PINNED_STATIC_PHYSX_DEVICE,
        expected=EXPECTED_FULL_VELOCITY_LIMITS_BEFORE_WRITE,
    )
    replacement = before_tensor.clone()
    replacement[:, list(GRIPPER_FOLLOWER_JOINT_INDICES)] = (
        GRIPPER_FOLLOWER_VELOCITY_LIMIT_FLOAT32
    )
    indices = torch.arange(
        replacement.shape[0], dtype=torch.int32, device=replacement.device
    )
    _require(indices.tolist() == [0], "gripper write articulation identity")
    setter = getattr(robot.root_physx_view, "set_dof_max_velocities", None)
    _require(callable(setter), "missing PhysX velocity setter")
    setter(replacement, indices)
    after_tensor = _direct_static_physx_tensor(robot, "get_dof_max_velocities")
    after = validate_tensor_evidence(
        tensor_evidence(after_tensor),
        field="post-write PhysX velocity limits",
        shape=(1, 13),
        device=PINNED_STATIC_PHYSX_DEVICE,
        expected=EXPECTED_FULL_VELOCITY_LIMITS_AFTER_WRITE,
    )
    contract = {
        "profile": EEF_GRIPPER_RUNTIME_PROFILE,
        "joint_names": list(EXPECTED_DROID_JOINT_NAMES),
        "gripper_joint_names": list(GRIPPER_JOINT_NAMES),
        "gripper_joint_indices": list(GRIPPER_JOINT_INDICES),
        "driver_joint_name": DRIVEN_GRIPPER_JOINT_NAME,
        "driver_joint_index": DRIVEN_GRIPPER_JOINT_INDEX,
        "follower_joint_names": list(GRIPPER_FOLLOWER_JOINT_NAMES),
        "follower_joint_indices": list(GRIPPER_FOLLOWER_JOINT_INDICES),
        "actuator_joint_ownership": ownership,
        "device_partition": {
            "profile": EEF_GRIPPER_DEVICE_PARTITION_PROFILE,
            "dynamic_articulation": PINNED_DYNAMIC_DEVICE,
            "implicit_actuator": PINNED_ACTUATOR_DEVICE,
            "static_physx": PINNED_STATIC_PHYSX_DEVICE,
            "dtype": PINNED_TENSOR_DTYPE,
        },
        "driver_actuator": driver,
        "mimic_joint_contract": mimic,
        "velocity_limits_before_write": before,
        "velocity_limits_after_write": after,
        "velocity_limit_write_contract": {
            "profile": EEF_GRIPPER_VELOCITY_WRITE_PROFILE,
            "setter": EEF_GRIPPER_VELOCITY_WRITE_SETTER,
            "timing": EEF_GRIPPER_VELOCITY_WRITE_TIMING,
            "call_count": 1,
            "articulation_indices": [0],
            "full_input": tensor_evidence(replacement),
        },
        "driver_target_slew": target_slew_contract,
        "measured_velocity_is_hard_bounded_by_limit": False,
    }
    if mimic_compliance is not None:
        contract[MIMIC_COMPLIANCE_STATIC_FIELD] = mimic_compliance
    contract = validate_eef_gripper_static_contract(
        contract,
        expected_target_slew_profile=target_slew_spec.profile,
    )
    target_slew_installer(contract["driver_target_slew"])
    installer = getattr(arm_term, "install_gripper_runtime_contract", None)
    _require(
        callable(installer), "EEF arm term cannot install gripper runtime evidence"
    )
    installer(contract, finger_term=finger_term)
    return contract


def validate_eef_gripper_post_reset(
    env: Any, expected_contract: Mapping[str, Any]
) -> None:
    """Verify that a later reset retained the installed full PhysX tensor."""

    target_slew = expected_contract.get("driver_target_slew")
    _require(isinstance(target_slew, Mapping), "missing expected target-slew profile")
    target_slew_profile = target_slew.get("profile")
    target_slew_spec = eef_gripper_target_slew_profile(target_slew_profile)
    validate_eef_gripper_static_contract(
        dict(expected_contract),
        expected_target_slew_profile=target_slew_spec.profile,
    )
    runtime = getattr(env, "unwrapped", env)
    robot = runtime.scene["robot"]
    if target_slew_spec.profile == EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE:
        _validate_post_reset_mimic_compliance_readback(
            robot=robot,
            source_contract=expected_contract["mimic_joint_contract"],
            expected_contract=expected_contract[MIMIC_COMPLIANCE_STATIC_FIELD],
        )
    current = tensor_evidence(
        _direct_static_physx_tensor(robot, "get_dof_max_velocities")
    )
    validate_tensor_evidence(
        current,
        field="post-reset PhysX velocity limits",
        shape=(1, 13),
        device=PINNED_STATIC_PHYSX_DEVICE,
        expected=EXPECTED_FULL_VELOCITY_LIMITS_AFTER_WRITE,
    )
    terms = runtime.action_manager._terms
    _require(list(terms) == ["arm", "finger_joint"], "EEF action order drift")
    finger_term = terms["finger_joint"]
    static_reporter = getattr(finger_term, "gripper_target_slew_static_contract", None)
    dynamic_reporter = getattr(finger_term, "gripper_target_slew_dynamic_report", None)
    _require(
        callable(static_reporter) and callable(dynamic_reporter),
        "EEF finger action lacks target-slew reset evidence",
    )
    _require(
        static_reporter() == expected_contract["driver_target_slew"],
        "post-reset gripper target-slew static drift",
    )
    dynamic = validate_eef_gripper_target_slew_dynamic(
        dynamic_reporter(),
        expected_profile=target_slew_spec.profile,
    )
    _require(
        dynamic["process_action_calls"] == 0 and dynamic["apply_calls"] == 0,
        "post-reset gripper target-slew state was not cleared",
    )


def record_eef_gripper_post_policy_step(env: Any) -> None:
    runtime = getattr(env, "unwrapped", env)
    recorder = getattr(
        runtime.action_manager._terms["arm"], "record_gripper_post_policy_step", None
    )
    _require(
        callable(recorder), "EEF arm term cannot record gripper post-step evidence"
    )
    recorder()
