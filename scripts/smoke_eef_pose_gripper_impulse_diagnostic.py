#!/usr/bin/env python3
"""Isolate the FoodBussing close-command dynamics at the EEF failure boundary.

This opt-in diagnostic replays the immutable official-LAP action fixture without
loading a model.  It changes no production controller or environment default.
The ``delay_first_close_one_step`` mode changes exactly action[115][7] from
closed-positive 1 to 0; the original close at step 116 is left untouched.

Host-only parsing and validation deliberately import no Isaac modules so a
cluster wrapper can validate the immutable capture after Kit exits.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import stat
import struct
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Mapping, Sequence

BOUNDARY_HELPER_PATH = (
    Path(__file__).resolve().parent / "smoke_eef_pose_boundary_replay.py"
)
_BOOTSTRAP_BOUNDARY_SIZE_BYTES = 108504
_BOOTSTRAP_BOUNDARY_SHA256 = (
    "a63f2a8ab9c42ea872da9d6e1913d43e0a89b0382c01d88071af19bdf2731d97"
)
_boundary_lstat = os.lstat(BOUNDARY_HELPER_PATH)
if (
    not stat.S_ISREG(_boundary_lstat.st_mode)
    or _boundary_lstat.st_nlink != 1
    or _boundary_lstat.st_size != _BOOTSTRAP_BOUNDARY_SIZE_BYTES
):
    raise RuntimeError("PolaRiS boundary helper bootstrap file identity drift")
_boundary_bootstrap_bytes = BOUNDARY_HELPER_PATH.read_bytes()
if (
    len(_boundary_bootstrap_bytes) != _BOOTSTRAP_BOUNDARY_SIZE_BYTES
    or hashlib.sha256(_boundary_bootstrap_bytes).hexdigest()
    != _BOOTSTRAP_BOUNDARY_SHA256
):
    raise RuntimeError("PolaRiS boundary helper bootstrap digest drift")
_BOUNDARY_SPEC = importlib.util.spec_from_file_location(
    "polaris_gripper_impulse_boundary_helper", BOUNDARY_HELPER_PATH
)
if _BOUNDARY_SPEC is None or _BOUNDARY_SPEC.loader is None:
    raise RuntimeError(f"Cannot load boundary helper at {BOUNDARY_HELPER_PATH}")
boundary = importlib.util.module_from_spec(_BOUNDARY_SPEC)
exec(  # noqa: S102 - execute only the bootstrap-verified exact source bytes.
    compile(
        _boundary_bootstrap_bytes,
        str(BOUNDARY_HELPER_PATH),
        "exec",
        dont_inherit=True,
    ),
    boundary.__dict__,
)


ENVIRONMENT = boundary.ENVIRONMENT
FIXTURE_PROFILE = boundary.FIXTURE_PROFILE
DIAGNOSTIC_PROFILE = "foodbussing_gripper_close_impulse_exact_delay1_v4"
FINGER_TRACE_PROFILE = "gripper_apply_causal_tail_all_links_device_partition_v3"
ACTION_PLAN_PROFILE = "foodbussing_first_close_exact_or_delay1_v1"
VIDEO_PROFILE = "lap_model_view_external_then_rot180_wrist_224x448_rational_cadence_v2"
GRIPPER_DRIVE_PROFILE = "implicit_gripper_effort200_cuda_actuator_cpu_static_physx_v3"
GRIPPER_VELOCITY_LIMIT_CANDIDATE_DRIVE_PROFILE = (
    "implicit_gripper_physx_velocity_limit5_cuda_actuator_cpu_static_physx_v1"
)
GRIPPER_DRIVE_PROFILES = (
    GRIPPER_DRIVE_PROFILE,
    GRIPPER_VELOCITY_LIMIT_CANDIDATE_DRIVE_PROFILE,
)
SOLVER_CHANGE_PROFILE = "eef_pose_solver_velocity_iterations_0_to_1_v1"
MODES = ("exact", "delay_first_close_one_step")
SOURCE_CLOSE_POLICY_STEP = 115
HORIZON_POLICY_STEP = 117
# Retain one complete policy step before the closed failure-timing window so a
# guard on the first substep of policy step 113 still has a terminal finger
# state that can be mutually bound to the arm trace.
RELEVANT_POLICY_STEP_START = 112
FAILURE_POLICY_STEP_START = 113
ACTION_WIDTH = 8
GRIPPER_ACTION_INDEX = 7
DECIMATION = 8
TRACE_CAPACITY = 48
VIDEO_FPS = 15
VIDEO_HEIGHT = 224
VIDEO_WIDTH = 448
FFPROBE_DURATION_DECIMAL_PLACES = 6
MP4_CONTAINER_DURATION_TICKS_PER_SECOND = 1000
GRIPPER_OPEN_TARGET_RAD = 0.0
GRIPPER_CLOSED_TARGET_RAD = math.pi / 4
REFERENCE_EXACT_FAILURE_POLICY_STEP = 115
REFERENCE_EXACT_FAILURE_PHYSICS_SUBSTEP = 2
RESET_SEED = 0
INITIAL_CONDITION_INDEX = 0
COMPLETE_FAILURE_POLICY_STEPS = list(
    range(FAILURE_POLICY_STEP_START, HORIZON_POLICY_STEP + 1)
)
ALLOWED_FAILURE_POLICY_STEPS = {
    mode: list(COMPLETE_FAILURE_POLICY_STEPS) for mode in MODES
}
EXPECTED_DROID_JOINT_NAMES = [
    *[f"panda_joint{index}" for index in range(1, 8)],
    "finger_joint",
    "right_outer_knuckle_joint",
    "left_inner_finger_joint",
    "right_inner_finger_joint",
    "left_inner_finger_knuckle_joint",
    "right_inner_finger_knuckle_joint",
]
# Authoritative live Isaac Lab/PhysX order from the pinned nvidia_droid USD,
# captured by the immutable L40S articulation-name probe (Slurm job 1098158).
EXPECTED_DROID_BODY_NAMES = [
    *[f"panda_link{index}" for index in range(9)],
    "base_link",
    "left_outer_knuckle",
    "right_outer_knuckle",
    "left_outer_finger",
    "right_outer_finger",
    "left_inner_finger",
    "right_inner_finger",
    "left_inner_knuckle",
    "right_inner_knuckle",
]
TIMESTAMP_DT_SECONDS = 1.0 / 120.0
TIMESTAMP_ABS_TOLERANCE_SECONDS = 1e-12
TIMESTAMP_CONTRACT = {
    "profile": "articulation_sim_timestamp_float64_one_physx_step_v1",
    "timestamp_type": "python_float_exact_not_bool_v1",
    "post_minus_pre_seconds": TIMESTAMP_DT_SECONDS,
    "relative_tolerance": 0.0,
    "absolute_tolerance_seconds": TIMESTAMP_ABS_TOLERANCE_SECONDS,
}
PINNED_CACHED_DEVICE = "cuda:0"
PINNED_DYNAMIC_PHYSX_DEVICE = "cuda:0"
PINNED_STATIC_PHYSX_DEVICE = "cpu"
PINNED_ACTUATOR_DEVICE = "cuda:0"
PINNED_TENSOR_DTYPE = "torch.float32"
DEVICE_PROBE_EVIDENCE = {
    "profile": "nvidia_droid_l40s_root_getter_device_partition_v1",
    "slurm_job_id": 1098162,
    "result_size_bytes": 11403,
    "result_sha256": (
        "d3c8ccfcb16cd523f084f5c7c82f41a03c1c2ab0f58487f45ff4c2a59066283c"
    ),
    "saved_wrapper_sha256": (
        "7bf346c05b676d16db0f102990efba9c481be01e2fb57ea96115313c200d48d1"
    ),
}
DEVICE_PROBE_EVIDENCE_FIELDS = {
    "profile",
    "slurm_job_id",
    "result_size_bytes",
    "result_sha256",
    "saved_wrapper_sha256",
}
PROBED_GRIPPER_DRIVE_FLOAT32_VALUES = {
    "velocity_limit_rad_s": 8.726646423339844,
    "effort_limit_nm": 200.0,
    "stiffness_nm_per_rad": 5729.578125,
    "damping_nm_s_per_rad": 0.011459155939519405,
}
GRIPPER_VELOCITY_LIMIT_CANDIDATE_FLOAT32_VALUES = {
    **PROBED_GRIPPER_DRIVE_FLOAT32_VALUES,
    "velocity_limit_rad_s": 5.0,
}
EXPECTED_BOUNDARY_HELPER_SIZE_BYTES = _BOOTSTRAP_BOUNDARY_SIZE_BYTES
EXPECTED_BOUNDARY_HELPER_SHA256 = _BOOTSTRAP_BOUNDARY_SHA256
EXPECTED_ROBOT_USD_SIZE_BYTES = 14156155
EXPECTED_ROBOT_USD_SHA256 = (
    "d8379925b103963dbf3e7c85bcc4ae101b81b7c1d7dabe7d2e964f41d069ec44"
)
EXPECTED_POLARIS_HUB_REVISION = "8c7e4103e266ef83d8b1ad2e9a63116edd5f155b"

ACTION_PLAN_FIELDS = {
    "schema_version",
    "profile",
    "mode",
    "fixture_profile",
    "fixture_action_count",
    "action_width",
    "horizon_policy_step",
    "planned_action_count",
    "source_gripper_transitions",
    "effective_gripper_transitions",
    "override_profile",
    "overrides",
    "original_prefix_float32_sha256",
    "effective_prefix_float32_sha256",
    "arm_prefix_float32_sha256",
}
OVERRIDE_FIELDS = {
    "policy_step",
    "action_index",
    "field",
    "original_float32",
    "effective_float32",
    "reason",
    "arm_dimensions_bitwise_unchanged",
}
TRANSITION_FIELDS = {"policy_step", "from_float32", "to_float32"}
TENSOR_EVIDENCE_FIELDS = {
    "shape",
    "dtype",
    "device",
    "values",
    "finite_mask",
    "finite_count",
    "nonfinite",
}
NONFINITE_FIELDS = {"flat_index", "kind"}
SNAPSHOT_FIELDS = {
    "articulation_data_sim_timestamp",
    "joint_names",
    "joint_position_rad",
    "joint_velocity_rad_s",
    "joint_acceleration_rad_s2",
    "joint_position_target_rad",
    "joint_velocity_target_rad_s",
    "joint_effort_target_nm",
    "approximate_pd_computed_torque_nm",
    "approximate_pd_applied_torque_nm",
    "physx_joint_position_rad",
    "physx_joint_velocity_rad_s",
    "physx_projected_joint_force_nm",
    "physx_joint_velocity_limit_rad_s",
    "physx_joint_effort_limit_nm",
    "physx_joint_stiffness_nm_per_rad",
    "physx_joint_damping_nm_s_per_rad",
    "body_names",
    "body_com_velocity_world",
    "body_com_acceleration_world",
    "physx_link_incoming_joint_wrench_child_joint_frame",
    "incoming_joint_wrench_semantics",
}
CACHED_ARTICULATION_TENSOR_FIELDS = (
    "joint_position_rad",
    "joint_velocity_rad_s",
    "joint_acceleration_rad_s2",
    "joint_position_target_rad",
    "joint_velocity_target_rad_s",
    "joint_effort_target_nm",
    "approximate_pd_computed_torque_nm",
    "approximate_pd_applied_torque_nm",
)
DYNAMIC_PHYSX_TENSOR_FIELDS = (
    "physx_joint_position_rad",
    "physx_joint_velocity_rad_s",
    "physx_projected_joint_force_nm",
    "body_com_velocity_world",
    "body_com_acceleration_world",
    "physx_link_incoming_joint_wrench_child_joint_frame",
)
STATIC_PHYSX_TENSOR_FIELDS = (
    "physx_joint_velocity_limit_rad_s",
    "physx_joint_effort_limit_nm",
    "physx_joint_stiffness_nm_per_rad",
    "physx_joint_damping_nm_s_per_rad",
)
JOINT_SNAPSHOT_TENSOR_FIELDS = (
    *CACHED_ARTICULATION_TENSOR_FIELDS,
    *DYNAMIC_PHYSX_TENSOR_FIELDS[:3],
    *STATIC_PHYSX_TENSOR_FIELDS,
)
BODY_SNAPSHOT_TENSOR_FIELDS = DYNAMIC_PHYSX_TENSOR_FIELDS[3:]
EXPECTED_DIRECT_PHYSX_GETTER_NAMES = {
    "get_dof_positions",
    "get_dof_velocities",
    "get_dof_projected_joint_forces",
    "get_dof_max_velocities",
    "get_dof_max_forces",
    "get_dof_stiffnesses",
    "get_dof_dampings",
    "get_link_velocities",
    "get_link_accelerations",
    "get_link_incoming_joint_force",
}
DIRECT_PHYSX_GETTER_CONTRACT = {
    "get_dof_positions": {
        "snapshot_field": "physx_joint_position_rad",
        "device": PINNED_DYNAMIC_PHYSX_DEVICE,
        "shape": [1, len(EXPECTED_DROID_JOINT_NAMES)],
    },
    "get_dof_velocities": {
        "snapshot_field": "physx_joint_velocity_rad_s",
        "device": PINNED_DYNAMIC_PHYSX_DEVICE,
        "shape": [1, len(EXPECTED_DROID_JOINT_NAMES)],
    },
    "get_dof_projected_joint_forces": {
        "snapshot_field": "physx_projected_joint_force_nm",
        "device": PINNED_DYNAMIC_PHYSX_DEVICE,
        "shape": [1, len(EXPECTED_DROID_JOINT_NAMES)],
    },
    "get_dof_max_velocities": {
        "snapshot_field": "physx_joint_velocity_limit_rad_s",
        "device": PINNED_STATIC_PHYSX_DEVICE,
        "shape": [1, len(EXPECTED_DROID_JOINT_NAMES)],
    },
    "get_dof_max_forces": {
        "snapshot_field": "physx_joint_effort_limit_nm",
        "device": PINNED_STATIC_PHYSX_DEVICE,
        "shape": [1, len(EXPECTED_DROID_JOINT_NAMES)],
    },
    "get_dof_stiffnesses": {
        "snapshot_field": "physx_joint_stiffness_nm_per_rad",
        "device": PINNED_STATIC_PHYSX_DEVICE,
        "shape": [1, len(EXPECTED_DROID_JOINT_NAMES)],
    },
    "get_dof_dampings": {
        "snapshot_field": "physx_joint_damping_nm_s_per_rad",
        "device": PINNED_STATIC_PHYSX_DEVICE,
        "shape": [1, len(EXPECTED_DROID_JOINT_NAMES)],
    },
    "get_link_velocities": {
        "snapshot_field": "body_com_velocity_world",
        "device": PINNED_DYNAMIC_PHYSX_DEVICE,
        "shape": [1, len(EXPECTED_DROID_BODY_NAMES), 6],
    },
    "get_link_accelerations": {
        "snapshot_field": "body_com_acceleration_world",
        "device": PINNED_DYNAMIC_PHYSX_DEVICE,
        "shape": [1, len(EXPECTED_DROID_BODY_NAMES), 6],
    },
    "get_link_incoming_joint_force": {
        "snapshot_field": "physx_link_incoming_joint_wrench_child_joint_frame",
        "device": PINNED_DYNAMIC_PHYSX_DEVICE,
        "shape": [1, len(EXPECTED_DROID_BODY_NAMES), 6],
    },
}
TRACE_ENTRY_FIELDS = {
    "apply_index",
    "policy_step",
    "physics_substep",
    "original_gripper_closed_action",
    "effective_gripper_closed_action",
    "raw_action_at_stage",
    "processed_target_at_stage_rad",
    "pre_apply",
    "target_after_setter_rad",
    "post_physics",
    "finalization_reason",
}
FINGER_TRACE_FIELDS = {
    "schema_version",
    "profile",
    "capacity",
    "relevant_policy_step_start",
    "relevant_policy_step_end",
    "total_staged_apply_count",
    "total_finalized_apply_count",
    "pending_apply_count",
    "dropped_relevant_entry_count",
    "tensor_capture_contract",
    "timestamp_contract",
    "entries",
}
TENSOR_CAPTURE_CONTRACT_FIELDS = {
    "profile",
    "cached_articulation_fields",
    "dynamic_physx_fields",
    "static_physx_fields",
    "cached_articulation_device",
    "dynamic_physx_device",
    "static_physx_device",
    "tensor_dtype",
    "authoritative_device_probe",
}
VIDEO_IDENTITY_FIELDS = {
    "path",
    "size_bytes",
    "sha256",
    "mode",
    "nlink",
    "profile",
    "fps",
    "frame_count",
    "height",
    "width",
}
SOLVER_CONTRACT_FIELDS = {
    "profile",
    "configured_solver_velocity_iterations_before_eef_setup",
    "configured_solver_velocity_iterations_after_eef_setup",
    "live_solver_velocity_iterations",
    "live_solver_position_iterations",
    "live_physx_solver_type",
}
GRIPPER_DRIVE_CONTRACT_FIELDS = {
    "profile",
    "actuator_name",
    "joint_names",
    "joint_indices",
    "action_term_joint_names",
    "action_term_joint_indices",
    "actuator_joint_names",
    "actuator_joint_indices",
    "configured_before_articulation_build",
    "live_actuator",
    "live_physx_readback",
    "legacy_velocity_limit_behavior",
    "effort_limit_behavior",
    "incoming_joint_wrench_semantics",
    "computed_applied_torque_semantics",
    "authoritative_device_probe",
}
CONFIGURED_GRIPPER_FIELDS = {
    "legacy_velocity_limit_rad_s",
    "velocity_limit_sim_rad_s",
    "legacy_effort_limit_nm",
    "effort_limit_sim_nm",
    "stiffness",
    "damping",
}
LIVE_ACTUATOR_FIELDS = {
    "cfg_velocity_limit",
    "cfg_velocity_limit_sim",
    "cfg_effort_limit",
    "cfg_effort_limit_sim",
    "cfg_stiffness",
    "cfg_damping",
    "resolved_velocity_limit_rad_s",
    "resolved_velocity_limit_sim_rad_s",
    "resolved_effort_limit_nm",
    "resolved_effort_limit_sim_nm",
    "resolved_stiffness_nm_per_rad",
    "resolved_damping_nm_s_per_rad",
}
LIVE_PHYSX_GRIPPER_FIELDS = {
    "velocity_limit_rad_s",
    "effort_limit_nm",
    "stiffness_nm_per_rad",
    "damping_nm_s_per_rad",
}
OUTCOME_FIELDS = {
    "kind",
    "mode",
    "reference_exact_failure_policy_step",
    "reference_exact_failure_physics_substep",
    "allowed_failure_policy_steps",
    "failure_policy_step",
    "failure_physics_substep",
    "failure_apply_index",
    "last_attempted_policy_step",
    "completed_horizon_policy_step",
    "controller_failure",
    "causal_interpretation",
    "timing_classification",
}
PAYLOAD_FIELDS = {
    "schema_version",
    "diagnostic_profile",
    "fixture_profile",
    "finalized",
    "capture_valid",
    "stage",
    "exit_code",
    "environment",
    "mode",
    "diagnostic_source",
    "runtime_exit_contract",
    "fixture",
    "boundary_helper_source",
    "assets",
    "action_plan",
    "runtime_protocol",
    "runtime_frame",
    "solver_contract",
    "gripper_drive_contract",
    "outcome",
    "video_phase_contract",
    "video",
    "finger_trace",
    "arm_safety",
    "arm_substep_trace",
    "arm_failure_runtime_evidence",
    "close_failures",
}
ASSETS_FIELDS = {
    "foodbussing",
    "robot_usd",
    "robot_usd_revision_metadata",
}
FILE_IDENTITY_FIELDS = {"path", "size_bytes", "sha256", "mode", "nlink"}
BOUNDARY_FILE_IDENTITY_FIELDS = {"path", "size_bytes", "sha256", "mode"}
FOODBUSSING_ASSET_FIELDS = {
    "scene",
    "initial_conditions",
    "polaris_hub_revision",
    "revision_metadata",
    "initial_condition_index",
}
ROBOT_METADATA_FIELDS = {"identity", "revision", "recorded_sha256"}
FIXTURE_IDENTITY_FIELDS = {
    "path",
    "size_bytes",
    "sha256",
    "mode",
    "fixture_profile",
    "source_trace_sha256",
    "action_float32_sha256",
    "action_count",
}
FAILURE_RUNTIME_EVIDENCE_FIELDS = {
    "policy_step",
    "arm_joint_names",
    "articulation_data_sim_timestamp",
    "arm_joint_pos_rad",
    "arm_joint_vel_rad_s",
    "arm_joint_target_rad",
    "arm_joint_velocity_target_rad_s",
    "arm_joint_effort_target_nm",
    "physx_arm_joint_pos_rad",
    "physx_arm_joint_vel_rad_s",
    "cached_minus_physx_arm_joint_pos_rad",
    "cached_minus_physx_arm_joint_vel_rad_s",
    "physx_arm_velocity_limits_rad_s",
    "physx_arm_effort_limits",
    "physx_arm_projected_joint_force_generalized_si",
    "physx_arm_stiffness_nm_per_rad",
    "physx_arm_damping_nm_s_per_rad",
    "arm_computed_torque",
    "arm_applied_torque",
    "ik_safety",
    "controller_substep_trace",
    "controller_substep_trace_error",
}
EXCEPTION_EVIDENCE_FIELDS = {"type", "message", "traceback"}
DIAGNOSTIC_SOURCE_FIELDS = {"scheme", "actual", "launch_expected"}
EXPECTED_SOURCE_FIELDS = {"size_bytes", "sha256"}
VIDEO_PHASE_FIELDS = {
    "profile",
    "pre_action_frame_count",
    "terminal_frame_count",
    "total_frame_count",
    "terminal_frame_index",
    "terminal_frame_phase",
    "terminal_policy_step",
    "physics_advanced_for_terminal_frame",
}
READY_MARKER_PROFILE = "gripper_impulse_raw_video_ready_before_sim_app_close_v1"
RUNTIME_EXIT_PROFILE = "stdlib_parent_pipe_wait_group_drain_validation_v2"
RUNTIME_EXIT_CONTRACT_FIELDS = {
    "profile",
    "path",
    "success_bytes_utf8",
    "mode",
    "nlink",
    "publisher",
    "reconciliation",
}
READY_MARKER_FIELDS = {
    "schema_version",
    "profile",
    "stage",
    "mode",
    "raw_result",
    "video",
    "diagnostic_source",
    "runtime_exit_contract",
}
CHILD_PROCESS_ENV = "POLARIS_GRIPPER_IMPULSE_KIT_CHILD"
CHILD_RESULT_FD_ENV = "POLARIS_GRIPPER_IMPULSE_CHILD_RESULT_FD"
CHILD_TIMEOUT_SECONDS = 900
CHILD_TERMINATE_GRACE_SECONDS = 10
CHILD_GROUP_DRAIN_SECONDS = 10
CHILD_PIPE_DRAIN_SECONDS = 10
PARENT_CLEANUP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


class GripperImpulseDiagnosticError(ValueError):
    """The diagnostic input or artifact violates its closed contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GripperImpulseDiagnosticError(message)


def _exact_int(value: Any, expected: int, *, field: str) -> int:
    _require(type(value) is int and value == expected, f"{field} exact integer")
    return value


def _bounded_int(
    value: Any, *, field: str, minimum: int, maximum: int | None = None
) -> int:
    _require(type(value) is int and value >= minimum, f"{field} integer")
    if maximum is not None:
        _require(value <= maximum, f"{field} integer range")
    return value


def _finite_number(value: Any, *, field: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{field} finite numeric",
    )
    return float(value)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _typed_equal(left: Any, right: Any) -> bool:
    """Compare JSON-like values without Python's bool/int equivalence."""

    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _typed_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _typed_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return bool(left == right)


def _strict_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _same_float32(left: float, right: float) -> bool:
    if (
        not isinstance(left, (int, float))
        or isinstance(left, bool)
        or not isinstance(right, (int, float))
        or isinstance(right, bool)
    ):
        return False
    return struct.pack("<f", left) == struct.pack("<f", right)


def _float32_actions_sha256(actions: Sequence[Sequence[float]]) -> str:
    return _sha256_bytes(
        b"".join(struct.pack("<f", value) for action in actions for value in action)
    )


def _float32_arm_sha256(actions: Sequence[Sequence[float]]) -> str:
    return _sha256_bytes(
        b"".join(struct.pack("<f", value) for action in actions for value in action[:7])
    )


def _gripper_transitions(actions: Sequence[Sequence[float]]) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    previous = actions[0][GRIPPER_ACTION_INDEX]
    for policy_step, action in enumerate(actions[1:], start=1):
        current = action[GRIPPER_ACTION_INDEX]
        if not _same_float32(current, previous):
            transitions.append(
                {
                    "policy_step": policy_step,
                    "from_float32": float(previous),
                    "to_float32": float(current),
                }
            )
        previous = current
    return transitions


def build_action_plan(
    actions: Sequence[Sequence[float]], *, mode: str
) -> tuple[dict[str, Any], list[list[float]]]:
    """Validate the fixture transition and build the one allowed intervention."""

    _require(mode in MODES, f"unsupported diagnostic mode: {mode!r}")
    _require(len(actions) == 378, "fixture action count drift")
    _require(
        all(len(action) == ACTION_WIDTH for action in actions),
        "fixture action width drift",
    )
    _require(
        all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for action in actions
            for value in action
        ),
        "fixture actions must be finite non-boolean numerics",
    )
    source_transitions = _gripper_transitions(actions)
    expected_transition = [{"policy_step": 115, "from_float32": 0.0, "to_float32": 1.0}]
    _require(
        source_transitions == expected_transition,
        "fixture must contain exactly one gripper 0->1 transition at step 115",
    )
    prefix_count = HORIZON_POLICY_STEP + 1
    original_prefix = [list(action) for action in actions[:prefix_count]]
    effective = [list(action) for action in original_prefix]
    overrides: list[dict[str, Any]] = []
    override_profile = "none"
    if mode == "delay_first_close_one_step":
        original = effective[SOURCE_CLOSE_POLICY_STEP][GRIPPER_ACTION_INDEX]
        _require(_same_float32(original, 1.0), "source close command drift")
        effective[SOURCE_CLOSE_POLICY_STEP][GRIPPER_ACTION_INDEX] = 0.0
        override_profile = "delay_exact_first_close_by_one_policy_step_v1"
        overrides.append(
            {
                "policy_step": SOURCE_CLOSE_POLICY_STEP,
                "action_index": GRIPPER_ACTION_INDEX,
                "field": "gripper_closed_positive",
                "original_float32": 1.0,
                "effective_float32": 0.0,
                "reason": "isolate_first_close_command_impulse",
                "arm_dimensions_bitwise_unchanged": True,
            }
        )

    _require(
        _float32_arm_sha256(original_prefix) == _float32_arm_sha256(effective),
        "diagnostic changed an arm action bit",
    )
    effective_transitions = _gripper_transitions(effective)
    wanted_effective = (
        expected_transition
        if mode == "exact"
        else [{"policy_step": 116, "from_float32": 0.0, "to_float32": 1.0}]
    )
    _require(
        effective_transitions == wanted_effective,
        "effective gripper transition contract drift",
    )
    plan = {
        "schema_version": 1,
        "profile": ACTION_PLAN_PROFILE,
        "mode": mode,
        "fixture_profile": FIXTURE_PROFILE,
        "fixture_action_count": len(actions),
        "action_width": ACTION_WIDTH,
        "horizon_policy_step": HORIZON_POLICY_STEP,
        "planned_action_count": prefix_count,
        "source_gripper_transitions": source_transitions,
        "effective_gripper_transitions": effective_transitions,
        "override_profile": override_profile,
        "overrides": overrides,
        "original_prefix_float32_sha256": _float32_actions_sha256(original_prefix),
        "effective_prefix_float32_sha256": _float32_actions_sha256(effective),
        "arm_prefix_float32_sha256": _float32_arm_sha256(effective),
    }
    validate_action_plan(plan)
    return plan, effective


def validate_action_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    _require(isinstance(plan, Mapping), "action plan must be an object")
    _require(set(plan) == ACTION_PLAN_FIELDS, "action plan schema drift")
    _exact_int(plan.get("schema_version"), 1, field="action plan schema version")
    _require(plan.get("profile") == ACTION_PLAN_PROFILE, "action plan profile")
    mode = plan.get("mode")
    _require(mode in MODES, "action plan mode")
    _require(plan.get("fixture_profile") == FIXTURE_PROFILE, "fixture profile")
    for field, expected in (
        ("fixture_action_count", 378),
        ("action_width", ACTION_WIDTH),
        ("horizon_policy_step", HORIZON_POLICY_STEP),
        ("planned_action_count", HORIZON_POLICY_STEP + 1),
    ):
        _exact_int(plan.get(field), expected, field=f"action plan {field}")
    source = plan.get("source_gripper_transitions")
    effective = plan.get("effective_gripper_transitions")
    for field, transitions in (("source", source), ("effective", effective)):
        _require(isinstance(transitions, list), f"{field} transitions")
        _require(
            all(
                isinstance(item, dict) and set(item) == TRANSITION_FIELDS
                for item in transitions
            ),
            f"{field} transition schema",
        )
        for index, item in enumerate(transitions):
            _require(
                type(item.get("policy_step")) is int
                and _same_float32(item.get("from_float32"), item["from_float32"])
                and _same_float32(item.get("to_float32"), item["to_float32"]),
                f"{field} transition {index} scalar types",
            )
    _require(
        _typed_equal(
            source,
            [{"policy_step": 115, "from_float32": 0.0, "to_float32": 1.0}],
        ),
        "source transition drift",
    )
    overrides = plan.get("overrides")
    _require(isinstance(overrides, list), "action overrides")
    if mode == "exact":
        _require(plan.get("override_profile") == "none", "exact override profile")
        _require(overrides == [], "exact mode may not override actions")
        _require(_typed_equal(effective, source), "exact effective transition")
    else:
        _require(
            plan.get("override_profile")
            == "delay_exact_first_close_by_one_policy_step_v1",
            "delay override profile",
        )
        _require(len(overrides) == 1, "delay override count")
        override = overrides[0]
        _require(
            isinstance(override, dict) and set(override) == OVERRIDE_FIELDS,
            "delay override schema",
        )
        _require(
            _typed_equal(
                override,
                {
                    "policy_step": 115,
                    "action_index": 7,
                    "field": "gripper_closed_positive",
                    "original_float32": 1.0,
                    "effective_float32": 0.0,
                    "reason": "isolate_first_close_command_impulse",
                    "arm_dimensions_bitwise_unchanged": True,
                },
            ),
            "delay override drift",
        )
        _require(
            _typed_equal(
                effective,
                [{"policy_step": 116, "from_float32": 0.0, "to_float32": 1.0}],
            ),
            "delayed transition drift",
        )
    for field in (
        "original_prefix_float32_sha256",
        "effective_prefix_float32_sha256",
        "arm_prefix_float32_sha256",
    ):
        value = plan.get(field)
        _require(
            isinstance(value, str)
            and len(value) == 64
            and all(char in "0123456789abcdef" for char in value),
            f"{field} digest",
        )
    return dict(plan)


def _flatten_nested(values: Any) -> tuple[list[int], list[Any]]:
    shape = list(getattr(values, "shape", ()))
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "tolist"):
        nested = values.tolist()
        shape = list(getattr(values, "shape", shape))
    else:
        nested = values
    flat: list[Any] = []

    def visit(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            flat.append(item)

    visit(nested)
    if not shape:
        shape = [len(flat)]
    return shape, flat


def tensor_evidence(values: Any) -> dict[str, Any]:
    """Serialize a tensor losslessly for finite values and explicitly mark infinities."""

    dtype = str(getattr(values, "dtype", "python_float64"))
    device = str(getattr(values, "device", "host"))
    shape, flat = _flatten_nested(values)
    serialized: list[float] = []
    finite_mask: list[bool] = []
    nonfinite: list[dict[str, Any]] = []
    for index, item in enumerate(flat):
        value = float(item)
        finite = math.isfinite(value)
        finite_mask.append(finite)
        serialized.append(value if finite else 0.0)
        if not finite:
            kind = "nan"
            if math.isinf(value):
                kind = "positive_infinity" if value > 0 else "negative_infinity"
            nonfinite.append({"flat_index": index, "kind": kind})
    return {
        "shape": shape,
        "dtype": dtype,
        "device": device,
        "values": serialized,
        "finite_mask": finite_mask,
        "finite_count": sum(finite_mask),
        "nonfinite": nonfinite,
    }


def validate_tensor_evidence(value: Any, *, field: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{field} must be an object")
    _require(set(value) == TENSOR_EVIDENCE_FIELDS, f"{field} schema")
    shape = value.get("shape")
    values = value.get("values")
    mask = value.get("finite_mask")
    _require(
        isinstance(shape, list)
        and shape
        and all(type(item) is int and item >= 0 for item in shape),
        f"{field} shape",
    )
    _require(
        isinstance(value.get("dtype"), str)
        and value["dtype"]
        and isinstance(value.get("device"), str)
        and value["device"],
        f"{field} dtype/device",
    )
    size = math.prod(shape)
    _require(isinstance(values, list) and len(values) == size, f"{field} values")
    _require(
        all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            for item in values
        ),
        f"{field} serialized values",
    )
    _require(
        isinstance(mask, list)
        and len(mask) == size
        and all(type(item) is bool for item in mask),
        f"{field} finite mask",
    )
    _require(
        type(value.get("finite_count")) is int
        and value.get("finite_count") == sum(mask),
        f"{field} finite count",
    )
    nonfinite = value.get("nonfinite")
    _require(isinstance(nonfinite, list), f"{field} nonfinite")
    expected_indices = [index for index, finite in enumerate(mask) if not finite]
    actual_indices: list[int] = []
    for item in nonfinite:
        _require(
            isinstance(item, dict) and set(item) == NONFINITE_FIELDS,
            f"{field} nonfinite schema",
        )
        _require(
            type(item.get("flat_index")) is int,
            f"{field} nonfinite flat index",
        )
        actual_indices.append(item.get("flat_index"))
        _require(
            item.get("kind") in {"nan", "positive_infinity", "negative_infinity"},
            f"{field} nonfinite kind",
        )
    _require(actual_indices == expected_indices, f"{field} nonfinite indices")
    return dict(value)


def _file_identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    _require(path.is_file(), f"missing artifact: {path}")
    info = path.stat()
    return {
        "path": str(path),
        "size_bytes": info.st_size,
        "sha256": _sha256_file(path),
        "mode": f"{stat.S_IMODE(info.st_mode):04o}",
        "nlink": info.st_nlink,
    }


def _validate_sha256(value: Any, *, field: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{field} SHA-256",
    )
    return value


def _validate_diagnostic_source(value: Any) -> dict[str, Any]:
    _require(isinstance(value, dict), "diagnostic source contract")
    _require(
        set(value) == DIAGNOSTIC_SOURCE_FIELDS,
        "diagnostic source contract schema",
    )
    _require(
        value.get("scheme") == "launch_provided_sha256_size_live_rehash_v1",
        "diagnostic source identity scheme",
    )
    actual = value.get("actual")
    expected = value.get("launch_expected")
    _require(
        isinstance(actual, dict) and set(actual) == FILE_IDENTITY_FIELDS,
        "diagnostic source actual identity schema",
    )
    _require(
        isinstance(expected, dict) and set(expected) == EXPECTED_SOURCE_FIELDS,
        "diagnostic source expected identity schema",
    )
    _validate_sha256(actual.get("sha256"), field="diagnostic source actual")
    _validate_sha256(expected.get("sha256"), field="diagnostic source expected")
    _require(
        type(actual.get("size_bytes")) is int and actual["size_bytes"] > 0,
        "diagnostic source actual size",
    )
    _require(
        isinstance(actual.get("path"), str)
        and isinstance(actual.get("mode"), str)
        and len(actual["mode"]) == 4
        and all(character in "01234567" for character in actual["mode"])
        and type(actual.get("nlink")) is int
        and actual["nlink"] == 1,
        "diagnostic source actual file identity",
    )
    _require(
        type(expected.get("size_bytes")) is int and expected["size_bytes"] > 0,
        "diagnostic source expected size",
    )
    _require(
        Path(actual.get("path", "")).resolve() == Path(__file__).resolve(),
        "diagnostic source path",
    )
    _require(
        actual["size_bytes"] == expected["size_bytes"]
        and actual["sha256"] == expected["sha256"],
        "diagnostic source does not match launch-provided identity",
    )
    return dict(value)


def capture_diagnostic_source(
    *, expected_sha256: str, expected_size_bytes: int
) -> dict[str, Any]:
    """Bind this script without embedding a self-referential hash in itself."""

    contract = {
        "scheme": "launch_provided_sha256_size_live_rehash_v1",
        "actual": _file_identity(Path(__file__)),
        "launch_expected": {
            "size_bytes": expected_size_bytes,
            "sha256": expected_sha256,
        },
    }
    return _validate_diagnostic_source(contract)


def publish_immutable_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    serialized = _strict_json_bytes(payload)
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        path.chmod(0o444)
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    _require(path.read_bytes() == serialized, "published JSON changed on reread")
    identity = _file_identity(path)
    _require(identity["mode"] == "0444", "published JSON mode")
    _require(identity["nlink"] == 1, "published JSON link count")
    return identity


def publish_immutable_exit_status(path: Path, exit_code: int) -> dict[str, Any]:
    """Atomically publish one exact process-exit record as a 0444 file."""

    _require(type(exit_code) is int and exit_code in {0, 1}, "runtime exit code")
    serialized = f"{exit_code}\n".encode("ascii")
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not os.path.lexists(path), f"runtime exit status already exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        path.chmod(0o444)
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    _require(path.read_bytes() == serialized, "published runtime exit status drift")
    identity = _file_identity(path)
    _require(identity["mode"] == "0444", "runtime exit status mode")
    _require(identity["nlink"] == 1, "runtime exit status link count")
    return identity


def _publish_parent_exit_and_terminate(path: Path, exit_code: int) -> None:
    publish_immutable_exit_status(path, exit_code)
    os._exit(exit_code)


def _prepare_child_result_descriptor() -> int:
    """Validate the parent's inherited pipe before importing Kit."""

    descriptor_text = os.environ.pop(CHILD_RESULT_FD_ENV, None)
    _require(descriptor_text is not None, f"missing {CHILD_RESULT_FD_ENV}")
    try:
        descriptor = int(descriptor_text)
    except (TypeError, ValueError) as error:
        raise GripperImpulseDiagnosticError(
            "invalid child result descriptor"
        ) from error
    _require(descriptor >= 3, "unsafe child result descriptor")
    metadata = os.fstat(descriptor)
    _require(stat.S_ISFIFO(metadata.st_mode), "child result descriptor is not a pipe")
    access_mode = fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE
    _require(access_mode == os.O_WRONLY, "child result descriptor is not write-only")
    os.set_inheritable(descriptor, False)
    return descriptor


def _write_child_result_byte(exit_code: int, descriptor: int) -> None:
    """Commit exactly one pre-close result byte and close the pipe endpoint."""

    _require(type(exit_code) is int and exit_code in {0, 1}, "child result code")
    try:
        _require(
            os.write(descriptor, bytes((exit_code,))) == 1,
            "short child result write",
        )
    except BaseException:
        try:
            os.close(descriptor)
        except BaseException:
            pass
        raise
    try:
        os.close(descriptor)
    except BaseException:
        pass


def _resolve_child_result(payload: bytes, process_return_code: int) -> int:
    """Accept only an exact one-byte+EOF report matching normal child exit."""

    if payload not in {b"\x00", b"\x01"}:
        return 1
    reported = payload[0]
    if type(process_return_code) is not int or process_return_code not in {0, 1}:
        return 1
    if reported != process_return_code:
        return 1
    return reported


def build_runtime_exit_contract(path: Path) -> dict[str, Any]:
    return {
        "profile": RUNTIME_EXIT_PROFILE,
        "path": str(path.resolve()),
        "success_bytes_utf8": "0\n",
        "mode": "0444",
        "nlink": 1,
        "publisher": "same_script_stdlib_parent_after_child_reap_v1",
        "reconciliation": (
            "exact_one_byte_eof_equals_normal_child_wait_status_after_"
            "empty_process_group_then_host_artifact_revalidation_v2"
        ),
    }


def validate_runtime_exit_contract(value: Any) -> dict[str, Any]:
    _require(isinstance(value, dict), "runtime exit contract")
    _require(
        set(value) == RUNTIME_EXIT_CONTRACT_FIELDS,
        "runtime exit contract schema",
    )
    path = value.get("path")
    _require(
        value.get("profile") == RUNTIME_EXIT_PROFILE
        and isinstance(path, str)
        and bool(path)
        and Path(path).is_absolute()
        and value.get("success_bytes_utf8") == "0\n"
        and value.get("mode") == "0444"
        and type(value.get("nlink")) is int
        and value.get("nlink") == 1,
        "runtime exit contract identity",
    )
    _require(
        value.get("publisher") == "same_script_stdlib_parent_after_child_reap_v1"
        and value.get("reconciliation")
        == (
            "exact_one_byte_eof_equals_normal_child_wait_status_after_"
            "empty_process_group_then_host_artifact_revalidation_v2"
        ),
        "runtime exit reconciliation identity",
    )
    return dict(value)


def publish_immutable_video(
    path: Path,
    frames: Sequence[Any],
    *,
    writer: Callable[..., Any] | None = None,
    probe: Callable[[Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Encode, fully decode-check, and non-overwriting publish one model-view MP4."""

    _require(bool(frames), "cannot publish an empty diagnostic video")
    path = path.resolve()
    _require(path.suffix.lower() == ".mp4", "diagnostic video must be MP4")
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.exists(), f"refusing to overwrite diagnostic video: {path}")
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.mp4")
    _require(not temporary.exists(), f"stale video temporary exists: {temporary}")
    if writer is None:
        import mediapy  # noqa: PLC0415

        writer = mediapy.write_video
    if probe is None:
        from polaris.eval_artifacts import probe_episode_video  # noqa: PLC0415

        probe = probe_episode_video
    try:
        writer(temporary, frames, fps=VIDEO_FPS)
        _require(temporary.is_file() and temporary.stat().st_size > 0, "empty video")
        observed = dict(probe(temporary))
        expected = {
            "frame_count": len(frames),
            "height": VIDEO_HEIGHT,
            "width": VIDEO_WIDTH,
        }
        _require(
            _typed_equal(observed, expected),
            f"diagnostic video shape drift: {observed!r}",
        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, path)
        path.chmod(0o444)
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    identity = _file_identity(path)
    _require(identity["mode"] == "0444", "published video mode")
    _require(identity["nlink"] == 1, "published video link count")
    return {
        **identity,
        "profile": VIDEO_PROFILE,
        "fps": VIDEO_FPS,
        "frame_count": len(frames),
        "height": VIDEO_HEIGHT,
        "width": VIDEO_WIDTH,
    }


def validate_ready_marker(
    marker: Any,
    *,
    mode: str,
    raw_identity: Mapping[str, Any],
    video_identity: Mapping[str, Any],
    diagnostic_source: Mapping[str, Any],
    runtime_exit_contract: Mapping[str, Any],
) -> dict[str, Any]:
    _require(isinstance(marker, dict), "ready marker")
    _require(set(marker) == READY_MARKER_FIELDS, "ready marker schema")
    _require(
        type(marker.get("schema_version")) is int
        and marker.get("schema_version") == 1
        and marker.get("profile") == READY_MARKER_PROFILE
        and marker.get("stage") == "simulation_app_close_pending"
        and marker.get("mode") == mode,
        "ready marker identity",
    )
    _require(
        _typed_equal(marker.get("raw_result"), raw_identity),
        "ready marker raw identity",
    )
    _require(
        _typed_equal(marker.get("video"), video_identity),
        "ready marker video identity",
    )
    _require(
        _typed_equal(marker.get("diagnostic_source"), diagnostic_source),
        "ready marker source identity",
    )
    validated_exit = validate_runtime_exit_contract(marker.get("runtime_exit_contract"))
    _require(
        _typed_equal(validated_exit, dict(runtime_exit_contract)),
        "ready marker runtime-exit identity",
    )
    return dict(marker)


def _expected_tensor_capture_contract() -> dict[str, Any]:
    return {
        "profile": "field_partitioned_device_clone_per_substep_v3",
        "cached_articulation_fields": list(CACHED_ARTICULATION_TENSOR_FIELDS),
        "dynamic_physx_fields": list(DYNAMIC_PHYSX_TENSOR_FIELDS),
        "static_physx_fields": list(STATIC_PHYSX_TENSOR_FIELDS),
        "cached_articulation_device": PINNED_CACHED_DEVICE,
        "dynamic_physx_device": PINNED_DYNAMIC_PHYSX_DEVICE,
        "static_physx_device": PINNED_STATIC_PHYSX_DEVICE,
        "tensor_dtype": PINNED_TENSOR_DTYPE,
        "authoritative_device_probe": dict(DEVICE_PROBE_EVIDENCE),
    }


def _validate_tensor_capture_contract(value: Any) -> dict[str, Any]:
    _require(isinstance(value, dict), "finger tensor capture contract")
    _require(
        set(value) == TENSOR_CAPTURE_CONTRACT_FIELDS,
        "finger tensor capture contract schema",
    )
    _require(
        _typed_equal(value, _expected_tensor_capture_contract()),
        "finger tensor capture field/device partition",
    )
    partitions = (
        set(CACHED_ARTICULATION_TENSOR_FIELDS),
        set(DYNAMIC_PHYSX_TENSOR_FIELDS),
        set(STATIC_PHYSX_TENSOR_FIELDS),
    )
    expected_snapshot_tensors = SNAPSHOT_FIELDS - {
        "articulation_data_sim_timestamp",
        "joint_names",
        "body_names",
        "incoming_joint_wrench_semantics",
    }
    _require(
        set.union(*partitions) == expected_snapshot_tensors
        and all(
            left.isdisjoint(right)
            for index, left in enumerate(partitions)
            for right in partitions[index + 1 :]
        )
        and {
            contract["snapshot_field"]
            for contract in DIRECT_PHYSX_GETTER_CONTRACT.values()
        }
        == partitions[1] | partitions[2],
        "finger tensor capture closed field partition",
    )
    _require(
        set(DIRECT_PHYSX_GETTER_CONTRACT) == EXPECTED_DIRECT_PHYSX_GETTER_NAMES,
        "finger tensor capture exact direct PhysX getter set",
    )
    probe = value["authoritative_device_probe"]
    _require(
        set(probe) == DEVICE_PROBE_EVIDENCE_FIELDS
        and type(probe["slurm_job_id"]) is int
        and type(probe["result_size_bytes"]) is int,
        "finger tensor capture authoritative probe",
    )
    return dict(value)


def _snapshot_tensor_expected_device(
    tensor_field: str, tensor_contract: Mapping[str, Any]
) -> str:
    if tensor_field in CACHED_ARTICULATION_TENSOR_FIELDS:
        return tensor_contract["cached_articulation_device"]
    if tensor_field in DYNAMIC_PHYSX_TENSOR_FIELDS:
        return tensor_contract["dynamic_physx_device"]
    if tensor_field in STATIC_PHYSX_TENSOR_FIELDS:
        return tensor_contract["static_physx_device"]
    raise GripperImpulseDiagnosticError(
        f"snapshot tensor field has no device classification: {tensor_field}"
    )


def _validate_snapshot(
    value: Any,
    *,
    field: str,
    tensor_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{field} must be an object")
    _require(set(value) == SNAPSHOT_FIELDS, f"{field} schema")
    timestamp = value.get("articulation_data_sim_timestamp")
    _require(
        type(timestamp) is float and math.isfinite(timestamp),
        f"{field} timestamp",
    )
    joint_names = value.get("joint_names")
    body_names = value.get("body_names")
    _require(
        isinstance(joint_names, list) and joint_names == EXPECTED_DROID_JOINT_NAMES,
        f"{field} joint names",
    )
    _require(
        isinstance(body_names, list) and body_names == EXPECTED_DROID_BODY_NAMES,
        f"{field} body names",
    )
    contract = _validate_tensor_capture_contract(
        _expected_tensor_capture_contract()
        if tensor_contract is None
        else dict(tensor_contract)
    )
    tensors: dict[str, dict[str, Any]] = {}
    for tensor_field in JOINT_SNAPSHOT_TENSOR_FIELDS:
        tensor = validate_tensor_evidence(
            value.get(tensor_field), field=f"{field}.{tensor_field}"
        )
        tensors[tensor_field] = tensor
        _require(
            tensor["shape"] == [len(joint_names)]
            and tensor["finite_count"] == len(joint_names)
            and tensor["finite_mask"] == [True] * len(joint_names)
            and tensor["nonfinite"] == [],
            f"{field}.{tensor_field} shape",
        )
        _require(
            tensor["device"] == _snapshot_tensor_expected_device(tensor_field, contract)
            and tensor["dtype"] == contract["tensor_dtype"],
            f"{field}.{tensor_field} dtype/device contract",
        )
    for tensor_field in BODY_SNAPSHOT_TENSOR_FIELDS:
        tensor = validate_tensor_evidence(
            value.get(tensor_field), field=f"{field}.{tensor_field}"
        )
        tensors[tensor_field] = tensor
        _require(
            tensor["shape"] == [len(body_names), 6]
            and tensor["finite_count"] == len(body_names) * 6
            and tensor["finite_mask"] == [True] * (len(body_names) * 6)
            and tensor["nonfinite"] == [],
            f"{field}.{tensor_field} shape",
        )
        _require(
            tensor["device"] == _snapshot_tensor_expected_device(tensor_field, contract)
            and tensor["dtype"] == contract["tensor_dtype"],
            f"{field}.{tensor_field} dtype/device contract",
        )
    _require(
        value.get("incoming_joint_wrench_semantics")
        == "physx_link_incoming_joint_total_6d_wrench_child_joint_frame_v1",
        f"{field} incoming-wrench semantics",
    )
    _require(
        _typed_equal(tensors["joint_position_rad"], tensors["physx_joint_position_rad"])
        and _typed_equal(
            tensors["joint_velocity_rad_s"],
            tensors["physx_joint_velocity_rad_s"],
        ),
        f"{field} cached/direct PhysX q/dq exact identity",
    )
    return dict(value)


def _validate_snapshot_gripper_binding(
    snapshot: Mapping[str, Any],
    *,
    gripper_drive: Mapping[str, Any],
    field: str,
) -> None:
    joint_index = gripper_drive["joint_indices"][0]
    _require(
        snapshot["joint_names"] == EXPECTED_DROID_JOINT_NAMES
        and joint_index == EXPECTED_DROID_JOINT_NAMES.index("finger_joint")
        and gripper_drive["joint_names"] == ["finger_joint"],
        f"{field} gripper joint identity",
    )
    for snapshot_field, drive_field in (
        ("physx_joint_velocity_limit_rad_s", "velocity_limit_rad_s"),
        ("physx_joint_effort_limit_nm", "effort_limit_nm"),
        ("physx_joint_stiffness_nm_per_rad", "stiffness_nm_per_rad"),
        ("physx_joint_damping_nm_s_per_rad", "damping_nm_s_per_rad"),
    ):
        source = snapshot[snapshot_field]
        selected = {
            "shape": [1, 1],
            "dtype": source["dtype"],
            "device": source["device"],
            "values": [source["values"][joint_index]],
            "finite_mask": [source["finite_mask"][joint_index]],
            "finite_count": 1,
            "nonfinite": [],
        }
        _require(
            _typed_equal(selected, gripper_drive["live_physx_readback"][drive_field]),
            f"{field} gripper {drive_field} live-drive identity",
        )


def _expected_gripper_value(mode: str, policy_step: int, *, original: bool) -> float:
    if original:
        return float(policy_step >= SOURCE_CLOSE_POLICY_STEP)
    if mode == "delay_first_close_one_step":
        return float(policy_step >= SOURCE_CLOSE_POLICY_STEP + 1)
    return float(policy_step >= SOURCE_CLOSE_POLICY_STEP)


def validate_finger_trace(
    trace: Any,
    *,
    action_plan: Mapping[str, Any],
    outcome: Mapping[str, Any],
    arm_safety: Mapping[str, Any],
    gripper_drive: Mapping[str, Any],
) -> dict[str, Any]:
    _require(isinstance(trace, dict), "finger trace must be an object")
    _require(set(trace) == FINGER_TRACE_FIELDS, "finger trace schema")
    _exact_int(trace.get("schema_version"), 1, field="finger trace schema version")
    _require(trace.get("profile") == FINGER_TRACE_PROFILE, "finger trace profile")
    _exact_int(trace.get("capacity"), TRACE_CAPACITY, field="finger trace capacity")
    _require(
        type(trace.get("relevant_policy_step_start")) is int
        and trace.get("relevant_policy_step_start") == RELEVANT_POLICY_STEP_START
        and type(trace.get("relevant_policy_step_end")) is int
        and trace.get("relevant_policy_step_end") == HORIZON_POLICY_STEP,
        "finger trace relevant range",
    )
    _exact_int(trace.get("pending_apply_count"), 0, field="finger trace pending apply")
    _require(
        type(trace.get("dropped_relevant_entry_count")) is int
        and trace.get("dropped_relevant_entry_count") == 0,
        "finger trace dropped relevant entry",
    )
    _require(
        _typed_equal(trace.get("timestamp_contract"), TIMESTAMP_CONTRACT),
        "finger trace timestamp contract",
    )
    tensor_contract = _validate_tensor_capture_contract(
        trace.get("tensor_capture_contract")
    )
    mode = action_plan["mode"]
    kind = outcome["kind"]
    is_failure = kind == "allowed_velocity_guard_failure"
    if is_failure:
        failure_step = outcome["failure_policy_step"]
        _require(
            failure_step in ALLOWED_FAILURE_POLICY_STEPS[mode],
            "finger trace allowed failure step",
        )
        failure_apply_index = outcome["failure_apply_index"]
        expected_arm_apply_calls = failure_apply_index + 1
        expected_finger_apply_calls = failure_apply_index
        expected_last_apply = expected_finger_apply_calls - 1
        expected_finalization_reason = "arm_guard_exception"
    else:
        _require(kind == "diagnostic_horizon_reached", "finger trace outcome kind")
        expected_arm_apply_calls = (HORIZON_POLICY_STEP + 1) * DECIMATION
        expected_finger_apply_calls = expected_arm_apply_calls
        expected_last_apply = expected_finger_apply_calls - 1
        expected_finalization_reason = "diagnostic_horizon"
    counters = arm_safety.get("counters")
    _require(isinstance(counters, dict), "arm safety counters")
    _require(
        type(counters.get("apply_calls")) is int
        and counters.get("apply_calls") == expected_arm_apply_calls,
        "arm/finger apply-count contract",
    )
    _require(
        type(trace.get("total_staged_apply_count")) is int
        and trace.get("total_staged_apply_count") == expected_finger_apply_calls
        and type(trace.get("total_finalized_apply_count")) is int
        and trace.get("total_finalized_apply_count") == expected_finger_apply_calls,
        "finger staged/finalized apply counts",
    )
    entries = trace.get("entries")
    _require(isinstance(entries, list), "finger trace entries")
    first_apply = RELEVANT_POLICY_STEP_START * DECIMATION
    expected_indices = list(range(first_apply, expected_last_apply + 1))
    _require(len(entries) == len(expected_indices), "finger relevant entry count")
    previous: dict[str, Any] | None = None
    for offset, (entry, apply_index) in enumerate(
        zip(entries, expected_indices, strict=True)
    ):
        field = f"finger trace entry {offset}"
        _require(
            isinstance(entry, dict) and set(entry) == TRACE_ENTRY_FIELDS,
            f"{field} schema",
        )
        policy_step, physics_substep = divmod(apply_index, DECIMATION)
        _require(
            type(entry.get("apply_index")) is int
            and entry.get("apply_index") == apply_index,
            f"{field} apply index",
        )
        _require(
            type(entry.get("policy_step")) is int
            and entry.get("policy_step") == policy_step,
            f"{field} policy step",
        )
        _require(
            type(entry.get("physics_substep")) is int
            and entry.get("physics_substep") == physics_substep,
            f"{field} physics substep",
        )
        original = _expected_gripper_value(mode, policy_step, original=True)
        effective = _expected_gripper_value(mode, policy_step, original=False)
        _require(
            _same_float32(entry.get("original_gripper_closed_action"), original)
            and _same_float32(entry.get("effective_gripper_closed_action"), effective)
            and _same_float32(entry.get("raw_action_at_stage"), effective),
            f"{field} staged action copy",
        )
        target = GRIPPER_CLOSED_TARGET_RAD if effective == 1.0 else 0.0
        target_after = validate_tensor_evidence(
            entry.get("target_after_setter_rad"),
            field=f"{field}.target_after_setter_rad",
        )
        _require(
            _same_float32(entry.get("processed_target_at_stage_rad"), target)
            and target_after["shape"] == [1]
            and target_after["finite_mask"] == [True]
            and target_after["device"] == tensor_contract["cached_articulation_device"]
            and target_after["dtype"] == tensor_contract["tensor_dtype"]
            and _same_float32(target_after["values"][0], target),
            f"{field} staged target copy",
        )
        pre_snapshot = _validate_snapshot(
            entry.get("pre_apply"),
            field=f"{field}.pre_apply",
            tensor_contract=tensor_contract,
        )
        post_snapshot = _validate_snapshot(
            entry.get("post_physics"),
            field=f"{field}.post_physics",
            tensor_contract=tensor_contract,
        )
        _validate_snapshot_gripper_binding(
            pre_snapshot,
            gripper_drive=gripper_drive,
            field=f"{field}.pre_apply",
        )
        _validate_snapshot_gripper_binding(
            post_snapshot,
            gripper_drive=gripper_drive,
            field=f"{field}.post_physics",
        )
        gripper_index = gripper_drive["joint_indices"][0]
        post_target = post_snapshot["joint_position_target_rad"]
        post_target_selected = {
            "shape": [1],
            "dtype": post_target["dtype"],
            "device": post_target["device"],
            "values": [post_target["values"][gripper_index]],
            "finite_mask": [post_target["finite_mask"][gripper_index]],
            "finite_count": 1,
            "nonfinite": [],
        }
        previous_policy_step = (apply_index - 1) // DECIMATION
        previous_effective = _expected_gripper_value(
            mode, previous_policy_step, original=False
        )
        previous_target = (
            GRIPPER_CLOSED_TARGET_RAD if previous_effective == 1.0 else 0.0
        )
        _require(
            _typed_equal(target_after, post_target_selected)
            and _same_float32(post_target_selected["values"][0], target)
            and _same_float32(
                pre_snapshot["joint_position_target_rad"]["values"][gripper_index],
                previous_target,
            ),
            f"{field} setter/post-target/pre-target causal identity",
        )
        q = pre_snapshot["joint_position_rad"]["values"][gripper_index]
        dq = pre_snapshot["joint_velocity_rad_s"]["values"][gripper_index]
        q_target = post_snapshot["joint_position_target_rad"]["values"][gripper_index]
        dq_target = post_snapshot["joint_velocity_target_rad_s"]["values"][
            gripper_index
        ]
        effort_target = post_snapshot["joint_effort_target_nm"]["values"][gripper_index]
        stiffness = post_snapshot["physx_joint_stiffness_nm_per_rad"]["values"][
            gripper_index
        ]
        damping = post_snapshot["physx_joint_damping_nm_s_per_rad"]["values"][
            gripper_index
        ]
        expected_computed = boundary._float32_add(  # noqa: SLF001
            boundary._float32_add(  # noqa: SLF001
                boundary._float32_multiply(  # noqa: SLF001
                    stiffness,
                    boundary._float32_subtract(q_target, q),  # noqa: SLF001
                ),
                boundary._float32_multiply(  # noqa: SLF001
                    damping,
                    boundary._float32_subtract(dq_target, dq),  # noqa: SLF001
                ),
            ),
            effort_target,
        )
        effort_limit = post_snapshot["physx_joint_effort_limit_nm"]["values"][
            gripper_index
        ]
        expected_applied = min(max(expected_computed, -effort_limit), effort_limit)
        _require(
            _same_float32(
                post_snapshot["approximate_pd_computed_torque_nm"]["values"][
                    gripper_index
                ],
                expected_computed,
            )
            and _same_float32(
                post_snapshot["approximate_pd_applied_torque_nm"]["values"][
                    gripper_index
                ],
                expected_applied,
            ),
            f"{field} gripper state/target/gain/torque causal identity",
        )
        pre_timestamp = pre_snapshot["articulation_data_sim_timestamp"]
        post_timestamp = post_snapshot["articulation_data_sim_timestamp"]
        _require(
            math.isclose(
                post_timestamp - pre_timestamp,
                TIMESTAMP_DT_SECONDS,
                rel_tol=TIMESTAMP_CONTRACT["relative_tolerance"],
                abs_tol=TIMESTAMP_CONTRACT["absolute_tolerance_seconds"],
            ),
            f"{field} timestamp one-PhysX-step cadence",
        )
        wanted_reason = (
            expected_finalization_reason
            if apply_index == expected_last_apply
            else "next_gripper_apply"
        )
        _require(
            entry.get("finalization_reason") == wanted_reason,
            f"{field} finalization reason",
        )
        if previous is not None:
            _require(
                _typed_equal(previous["post_physics"], entry["pre_apply"]),
                f"{field} causal snapshot continuity",
            )
        previous = entry
    return dict(trace)


def _validate_solver_contract(value: Any) -> dict[str, Any]:
    _require(isinstance(value, dict), "solver contract")
    _require(set(value) == SOLVER_CONTRACT_FIELDS, "solver contract schema")
    _require(value.get("profile") == SOLVER_CHANGE_PROFILE, "solver profile")
    for field, expected in (
        ("configured_solver_velocity_iterations_before_eef_setup", 0),
        ("configured_solver_velocity_iterations_after_eef_setup", 1),
        ("live_solver_velocity_iterations", 1),
        ("live_solver_position_iterations", 64),
        ("live_physx_solver_type", 1),
    ):
        _exact_int(value.get(field), expected, field=f"solver contract {field}")
    return dict(value)


def _validate_assets(value: Any) -> dict[str, Any]:
    _require(isinstance(value, dict), "capture assets")
    _require(set(value) == ASSETS_FIELDS, "capture assets schema")
    foodbussing = value.get("foodbussing")
    _require(
        isinstance(foodbussing, dict) and set(foodbussing) == FOODBUSSING_ASSET_FIELDS,
        "FoodBussing assets schema",
    )
    _require(
        foodbussing.get("polaris_hub_revision") == EXPECTED_POLARIS_HUB_REVISION
        and type(foodbussing.get("initial_condition_index")) is int
        and foodbussing.get("initial_condition_index") == INITIAL_CONDITION_INDEX,
        "FoodBussing asset revision/index",
    )
    revision_metadata = foodbussing.get("revision_metadata")
    _require(
        isinstance(revision_metadata, dict)
        and set(revision_metadata) == {"initial_conditions.json", "scene.usda"},
        "FoodBussing revision metadata schema",
    )
    for filename, identity in revision_metadata.items():
        _require(
            isinstance(identity, dict)
            and set(identity) == BOUNDARY_FILE_IDENTITY_FIELDS | {"revision"}
            and identity.get("revision") == EXPECTED_POLARIS_HUB_REVISION
            and isinstance(identity.get("path"), str)
            and type(identity.get("size_bytes")) is int
            and identity["size_bytes"] > 0
            and isinstance(identity.get("mode"), str),
            f"FoodBussing revision metadata {filename}",
        )
    scene = foodbussing.get("scene")
    conditions = foodbussing.get("initial_conditions")
    _require(
        isinstance(scene, dict)
        and set(scene) == BOUNDARY_FILE_IDENTITY_FIELDS
        and type(scene.get("size_bytes")) is int
        and scene["size_bytes"] > 0
        and scene.get("sha256") == boundary.EXPECTED_ASSET_CONTRACT["scene_sha256"]
        and isinstance(conditions, dict)
        and set(conditions) == BOUNDARY_FILE_IDENTITY_FIELDS
        and type(conditions.get("size_bytes")) is int
        and conditions["size_bytes"] > 0
        and conditions.get("sha256")
        == boundary.EXPECTED_ASSET_CONTRACT["initial_conditions_sha256"],
        "FoodBussing scene/initial-condition identity",
    )
    robot = value.get("robot_usd")
    _require(
        isinstance(robot, dict) and set(robot) == FILE_IDENTITY_FIELDS,
        "robot USD identity schema",
    )
    _require(
        Path(robot.get("path", "")).name == "noninstanceable.usd"
        and Path(robot.get("path", "")).parent.name == "nvidia_droid"
        and type(robot.get("size_bytes")) is int
        and robot.get("size_bytes") == EXPECTED_ROBOT_USD_SIZE_BYTES
        and type(robot.get("nlink")) is int
        and robot.get("nlink") == 1
        and robot.get("sha256") == EXPECTED_ROBOT_USD_SHA256,
        "robot USD identity drift",
    )
    metadata = value.get("robot_usd_revision_metadata")
    _require(
        isinstance(metadata, dict) and set(metadata) == ROBOT_METADATA_FIELDS,
        "robot USD revision metadata schema",
    )
    _require(
        metadata.get("revision") == EXPECTED_POLARIS_HUB_REVISION
        and metadata.get("recorded_sha256") == EXPECTED_ROBOT_USD_SHA256
        and isinstance(metadata.get("identity"), dict)
        and set(metadata["identity"]) == FILE_IDENTITY_FIELDS,
        "robot USD metadata revision/hash",
    )
    metadata_identity = metadata["identity"]
    _require(
        type(metadata_identity.get("size_bytes")) is int
        and metadata_identity["size_bytes"] > 0
        and type(metadata_identity.get("nlink")) is int
        and metadata_identity["nlink"] == 1,
        "robot USD metadata file identity",
    )
    return dict(value)


def _capture_assets(*, scene_path: Path, robot_usd_path: Path) -> dict[str, Any]:
    foodbussing = boundary.validate_asset_contract(scene_path)
    robot_usd_path = robot_usd_path.resolve()
    robot_identity = _file_identity(robot_usd_path)
    data_root = robot_usd_path.parent.parent
    metadata_path = (
        data_root
        / ".cache"
        / "huggingface"
        / "download"
        / "nvidia_droid"
        / "noninstanceable.usd.metadata"
    )
    metadata_identity = _file_identity(metadata_path)
    metadata_lines = metadata_path.read_text(encoding="utf-8").splitlines()
    _require(len(metadata_lines) >= 2, "robot USD metadata lines")
    assets = {
        "foodbussing": foodbussing,
        "robot_usd": robot_identity,
        "robot_usd_revision_metadata": {
            "identity": metadata_identity,
            "revision": metadata_lines[0].strip(),
            "recorded_sha256": metadata_lines[1].strip(),
        },
    }
    return _validate_assets(assets)


def _tensor_evidence_equal_excluding_device(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    return _typed_equal(
        {field: left[field] for field in TENSOR_EVIDENCE_FIELDS if field != "device"},
        {field: right[field] for field in TENSOR_EVIDENCE_FIELDS if field != "device"},
    )


def _selected_gripper_drive_profile(candidate_enabled: Any) -> str:
    _require(type(candidate_enabled) is bool, "gripper candidate flag type")
    return (
        GRIPPER_VELOCITY_LIMIT_CANDIDATE_DRIVE_PROFILE
        if candidate_enabled
        else GRIPPER_DRIVE_PROFILE
    )


def _gripper_drive_expectations(profile: Any) -> dict[str, Any]:
    if profile == GRIPPER_DRIVE_PROFILE:
        return {
            "configured": {
                "legacy_velocity_limit_rad_s": 5.0,
                "velocity_limit_sim_rad_s": None,
                "legacy_effort_limit_nm": 200.0,
                "effort_limit_sim_nm": None,
                "stiffness": None,
                "damping": None,
            },
            "live_cfg_velocity_limit_sim": None,
            "values": PROBED_GRIPPER_DRIVE_FLOAT32_VALUES,
            "velocity_behavior": (
                "isaaclab_2p3_implicit_legacy_velocity_limit_5_ignored_"
                "velocity_limit_sim_unset_v1"
            ),
            "effort_behavior": (
                "implicit_legacy_effort_limit_200_promoted_to_effort_limit_sim_"
                "and_enforced_v1"
            ),
        }
    if profile == GRIPPER_VELOCITY_LIMIT_CANDIDATE_DRIVE_PROFILE:
        return {
            "configured": {
                "legacy_velocity_limit_rad_s": 5.0,
                "velocity_limit_sim_rad_s": 5.0,
                "legacy_effort_limit_nm": 200.0,
                "effort_limit_sim_nm": 200.0,
                "stiffness": None,
                "damping": None,
            },
            "live_cfg_velocity_limit_sim": 5.0,
            "values": GRIPPER_VELOCITY_LIMIT_CANDIDATE_FLOAT32_VALUES,
            "velocity_behavior": (
                "isaaclab_2p3_explicit_velocity_limit_sim_5_enforced_"
                "eef_diagnostic_only_v1"
            ),
            "effort_behavior": (
                "implicit_equal_legacy_and_sim_effort_limit_200_enforced_v1"
            ),
        }
    raise GripperImpulseDiagnosticError(f"unknown gripper drive profile: {profile!r}")


def _validate_gripper_drive_contract(
    value: Any, *, expected_profile: str | None = None
) -> dict[str, Any]:
    _require(isinstance(value, dict), "gripper drive contract")
    _require(
        set(value) == GRIPPER_DRIVE_CONTRACT_FIELDS,
        "gripper drive contract schema",
    )
    expectations = _gripper_drive_expectations(value.get("profile"))
    if expected_profile is not None:
        _require(
            expected_profile in GRIPPER_DRIVE_PROFILES,
            "expected gripper drive profile",
        )
        _require(
            value.get("profile") == expected_profile,
            "gripper drive profile does not match independent expectation",
        )
    _require(
        _typed_equal(value.get("authoritative_device_probe"), DEVICE_PROBE_EVIDENCE),
        "gripper authoritative device probe",
    )
    _require(value.get("actuator_name") == "gripper", "gripper actuator name")
    expected_joint_names = ["finger_joint"]
    expected_joint_indices = [EXPECTED_DROID_JOINT_NAMES.index("finger_joint")]
    for field in ("joint_names", "action_term_joint_names", "actuator_joint_names"):
        _require(
            _typed_equal(value.get(field), expected_joint_names),
            f"gripper {field} cross-binding",
        )
    for field in (
        "joint_indices",
        "action_term_joint_indices",
        "actuator_joint_indices",
    ):
        _require(
            _typed_equal(value.get(field), expected_joint_indices),
            f"gripper {field} cross-binding",
        )
    configured = value.get("configured_before_articulation_build")
    _require(
        isinstance(configured, dict) and set(configured) == CONFIGURED_GRIPPER_FIELDS,
        "configured gripper schema",
    )
    _require(
        _typed_equal(configured, expectations["configured"]),
        "configured gripper drive drift",
    )
    live = value.get("live_actuator")
    _require(
        isinstance(live, dict) and set(live) == LIVE_ACTUATOR_FIELDS,
        "live gripper actuator schema",
    )
    validated_live: dict[str, dict[str, Any]] = {}
    for field in LIVE_ACTUATOR_FIELDS - {
        "cfg_velocity_limit",
        "cfg_velocity_limit_sim",
        "cfg_effort_limit",
        "cfg_effort_limit_sim",
        "cfg_stiffness",
        "cfg_damping",
    }:
        tensor = validate_tensor_evidence(
            live.get(field), field=f"live gripper {field}"
        )
        _require(
            tensor["shape"] == [1, 1]
            and tensor["finite_mask"] == [True]
            and tensor["finite_count"] == 1
            and tensor["nonfinite"] == [],
            f"live gripper {field} finite scalar",
        )
        _require(
            tensor["dtype"] == PINNED_TENSOR_DTYPE
            and tensor["device"] == PINNED_ACTUATOR_DEVICE,
            f"live gripper {field} CUDA device/dtype contract",
        )
        validated_live[field] = tensor
    _require(
        live.get("cfg_velocity_limit") is None
        and _typed_equal(
            live.get("cfg_velocity_limit_sim"),
            expectations["live_cfg_velocity_limit_sim"],
        )
        and type(live.get("cfg_effort_limit")) is float
        and live.get("cfg_effort_limit") == 200.0
        and type(live.get("cfg_effort_limit_sim")) is float
        and live.get("cfg_effort_limit_sim") == 200.0
        and live.get("cfg_stiffness") is None
        and live.get("cfg_damping") is None,
        "live gripper implicit-actuator cfg behavior",
    )
    physx = value.get("live_physx_readback")
    _require(
        isinstance(physx, dict) and set(physx) == LIVE_PHYSX_GRIPPER_FIELDS,
        "live gripper PhysX schema",
    )
    validated_physx: dict[str, dict[str, Any]] = {}
    for field, tensor in physx.items():
        validated = validate_tensor_evidence(tensor, field=f"gripper PhysX {field}")
        _require(
            validated["shape"] == [1, 1]
            and validated["finite_mask"] == [True]
            and validated["finite_count"] == 1
            and validated["nonfinite"] == [],
            f"gripper PhysX {field} finite scalar",
        )
        _require(
            validated["dtype"] == PINNED_TENSOR_DTYPE
            and validated["device"] == PINNED_STATIC_PHYSX_DEVICE,
            f"gripper PhysX {field} CPU device/dtype contract",
        )
        validated_physx[field] = validated
    for field in (
        "resolved_velocity_limit_rad_s",
        "resolved_velocity_limit_sim_rad_s",
        "resolved_effort_limit_nm",
        "resolved_effort_limit_sim_nm",
    ):
        _require(
            validated_live[field]["values"][0] > 0.0,
            f"live gripper {field} must be positive",
        )
    for field in (
        "resolved_stiffness_nm_per_rad",
        "resolved_damping_nm_s_per_rad",
    ):
        _require(
            validated_live[field]["values"][0] >= 0.0,
            f"live gripper {field} must be nonnegative",
        )
    for field, tensor in validated_physx.items():
        lower_bound = (
            0.0
            if field
            in {
                "stiffness_nm_per_rad",
                "damping_nm_s_per_rad",
            }
            else math.nextafter(0.0, 1.0)
        )
        _require(
            tensor["values"][0] >= lower_bound,
            f"gripper PhysX {field} invalid sign",
        )
    for live_field, physx_field in (
        ("resolved_velocity_limit_rad_s", "velocity_limit_rad_s"),
        ("resolved_velocity_limit_sim_rad_s", "velocity_limit_rad_s"),
        ("resolved_effort_limit_nm", "effort_limit_nm"),
        ("resolved_effort_limit_sim_nm", "effort_limit_nm"),
        ("resolved_stiffness_nm_per_rad", "stiffness_nm_per_rad"),
        ("resolved_damping_nm_s_per_rad", "damping_nm_s_per_rad"),
    ):
        _require(
            _tensor_evidence_equal_excluding_device(
                validated_live[live_field], validated_physx[physx_field]
            ),
            f"gripper actuator/PhysX {live_field} mirror drift excluding device only",
        )
    for live_field, probed_field in (
        ("resolved_velocity_limit_rad_s", "velocity_limit_rad_s"),
        ("resolved_velocity_limit_sim_rad_s", "velocity_limit_rad_s"),
        ("resolved_effort_limit_nm", "effort_limit_nm"),
        ("resolved_effort_limit_sim_nm", "effort_limit_nm"),
        ("resolved_stiffness_nm_per_rad", "stiffness_nm_per_rad"),
        ("resolved_damping_nm_s_per_rad", "damping_nm_s_per_rad"),
    ):
        _require(
            _same_float32(
                validated_live[live_field]["values"][0],
                expectations["values"][probed_field],
            ),
            f"gripper job1098162/profile resolved actuator value drift: {live_field}",
        )
    for physx_field, expected in expectations["values"].items():
        _require(
            _same_float32(validated_physx[physx_field]["values"][0], expected),
            f"gripper job1098162/profile static PhysX value drift: {physx_field}",
        )
    effort = physx["effort_limit_nm"]
    _require(
        effort["finite_mask"] == [True] and _same_float32(effort["values"][0], 200.0),
        "gripper PhysX effort limit is not enforced at 200 Nm",
    )
    _require(
        value.get("legacy_velocity_limit_behavior") == expectations["velocity_behavior"]
        and value.get("effort_limit_behavior") == expectations["effort_behavior"],
        "gripper limit semantics",
    )
    _require(
        value.get("incoming_joint_wrench_semantics")
        == "physx_total_incoming_joint_wrench_not_contact_force_child_joint_frame_v1",
        "gripper wrench semantics",
    )
    _require(
        value.get("computed_applied_torque_semantics")
        == "isaaclab_implicit_actuator_approximate_pd_preclip_and_effortlimit_clipped_v1",
        "gripper torque semantics",
    )
    return dict(value)


def _timing_classification(
    mode: str,
    failure_step: int | None,
    failure_substep: int | None,
) -> str:
    if failure_step is None:
        return (
            "reference_not_reproduced_through_horizon"
            if mode == "exact"
            else "delayed_close_survived_through_horizon"
        )
    timing = (failure_step, failure_substep)
    if mode == "exact" and timing == (
        REFERENCE_EXACT_FAILURE_POLICY_STEP,
        REFERENCE_EXACT_FAILURE_PHYSICS_SUBSTEP,
    ):
        return "reference_reproduced"
    if mode == "delay_first_close_one_step" and timing == (
        REFERENCE_EXACT_FAILURE_POLICY_STEP + 1,
        REFERENCE_EXACT_FAILURE_PHYSICS_SUBSTEP,
    ):
        return "delayed_close_shifted_reference_one_policy_step"
    if mode == "delay_first_close_one_step" and timing == (
        REFERENCE_EXACT_FAILURE_POLICY_STEP,
        REFERENCE_EXACT_FAILURE_PHYSICS_SUBSTEP,
    ):
        return "delayed_close_did_not_shift_reference"
    return "unexpected_complete_failure_timing_inconclusive"


def _causal_interpretation(
    mode: str,
    failure_step: int | None,
    failure_substep: int | None = None,
) -> str:
    classification = _timing_classification(mode, failure_step, failure_substep)
    return {
        "reference_reproduced": (
            "exact_reference_velocity_guard_reproduced_at_policy_115_substep_2"
        ),
        "delayed_close_shifted_reference_one_policy_step": (
            "delayed_close_shifted_velocity_guard_one_policy_step"
        ),
        "delayed_close_did_not_shift_reference": (
            "delayed_close_did_not_shift_reference_velocity_guard"
        ),
        "unexpected_complete_failure_timing_inconclusive": (
            "complete_relevant_window_failure_has_unexpected_timing_inconclusive"
        ),
        "reference_not_reproduced_through_horizon": (
            "exact_fixture_survived_through_horizon_without_reference_reproduction"
        ),
        "delayed_close_survived_through_horizon": (
            "delayed_close_survived_through_horizon"
        ),
    }[classification]


def _validate_outcome(value: Any, *, mode: str) -> dict[str, Any]:
    _require(isinstance(value, dict), "diagnostic outcome")
    _require(set(value) == OUTCOME_FIELDS, "diagnostic outcome schema")
    _require(value.get("mode") == mode, "diagnostic outcome mode")
    _exact_int(
        value.get("reference_exact_failure_policy_step"),
        REFERENCE_EXACT_FAILURE_POLICY_STEP,
        field="diagnostic reference failure policy step",
    )
    _exact_int(
        value.get("reference_exact_failure_physics_substep"),
        REFERENCE_EXACT_FAILURE_PHYSICS_SUBSTEP,
        field="diagnostic reference failure physics substep",
    )
    allowed = value.get("allowed_failure_policy_steps")
    _require(
        isinstance(allowed, list)
        and all(type(item) is int for item in allowed)
        and allowed == ALLOWED_FAILURE_POLICY_STEPS[mode],
        "diagnostic allowed failure steps",
    )
    kind = value.get("kind")
    if kind == "allowed_velocity_guard_failure":
        failure = value.get("controller_failure")
        failure_step = value.get("failure_policy_step")
        failure_substep = value.get("failure_physics_substep")
        failure_apply_index = value.get("failure_apply_index")
        _require(
            type(failure_step) is int
            and failure_step in ALLOWED_FAILURE_POLICY_STEPS[mode]
            and type(failure_substep) is int
            and 0 <= failure_substep < DECIMATION
            and type(failure_apply_index) is int
            and failure_apply_index == failure_step * DECIMATION + failure_substep
            and type(value.get("last_attempted_policy_step")) is int
            and value.get("last_attempted_policy_step") == failure_step
            and value.get("completed_horizon_policy_step") is None,
            "diagnostic failure outcome timing",
        )
        _require(
            isinstance(failure, dict)
            and set(failure) == EXCEPTION_EVIDENCE_FIELDS
            and failure.get("type", "").endswith("DifferentialIKInvariantError")
            and "joint velocity exceeds" in failure.get("message", ""),
            "diagnostic failure type/message",
        )
    elif kind == "diagnostic_horizon_reached":
        _require(
            value.get("failure_policy_step") is None
            and value.get("failure_physics_substep") is None
            and value.get("failure_apply_index") is None
            and type(value.get("last_attempted_policy_step")) is int
            and value.get("last_attempted_policy_step") == HORIZON_POLICY_STEP
            and type(value.get("completed_horizon_policy_step")) is int
            and value.get("completed_horizon_policy_step") == HORIZON_POLICY_STEP
            and value.get("controller_failure") is None,
            "diagnostic horizon outcome",
        )
    else:
        raise GripperImpulseDiagnosticError(f"unexpected diagnostic outcome: {kind!r}")
    _require(
        value.get("causal_interpretation")
        == _causal_interpretation(
            mode,
            value.get("failure_policy_step"),
            value.get("failure_physics_substep"),
        )
        and value.get("timing_classification")
        == _timing_classification(
            mode,
            value.get("failure_policy_step"),
            value.get("failure_physics_substep"),
        ),
        "diagnostic causal interpretation",
    )
    return dict(value)


def build_video_phase_contract(outcome: Mapping[str, Any]) -> dict[str, Any]:
    failure = outcome["kind"] == "allowed_velocity_guard_failure"
    terminal_policy_step = (
        outcome["failure_policy_step"] if failure else HORIZON_POLICY_STEP
    )
    pre_action_count = terminal_policy_step + 1
    return {
        "profile": "pre_action_each_attempt_plus_terminal_no_advance_state_v1",
        "pre_action_frame_count": pre_action_count,
        "terminal_frame_count": 1,
        "total_frame_count": pre_action_count + 1,
        "terminal_frame_index": pre_action_count,
        "terminal_frame_phase": (
            "post_guard_bound_state_no_physics_advance"
            if failure
            else "post_horizon_bound_state_no_physics_advance"
        ),
        "terminal_policy_step": terminal_policy_step,
        "physics_advanced_for_terminal_frame": False,
    }


def validate_video_phase_contract(
    value: Any, *, outcome: Mapping[str, Any]
) -> dict[str, Any]:
    _require(isinstance(value, dict), "video phase contract")
    _require(set(value) == VIDEO_PHASE_FIELDS, "video phase contract schema")
    for field in (
        "pre_action_frame_count",
        "terminal_frame_count",
        "total_frame_count",
        "terminal_frame_index",
        "terminal_policy_step",
    ):
        _require(type(value.get(field)) is int, f"video phase {field} type")
    _require(
        type(value.get("physics_advanced_for_terminal_frame")) is bool,
        "video phase physics-advance type",
    )
    _require(
        value == build_video_phase_contract(outcome),
        "video phase/count contract drift",
    )
    return dict(value)


def _validate_runtime_identity(
    protocol: Any,
    frame: Any,
    *,
    arm_safety: Mapping[str, Any],
) -> None:
    _require(isinstance(protocol, dict), "runtime protocol")
    _require(
        type(protocol.get("reset_seed")) is int
        and type(protocol.get("initial_condition_index")) is int,
        "runtime reset/initial-condition integer identity",
    )
    _require(
        _typed_equal(
            protocol,
            {
                "episode_steps": 450,
                "policy_hz": 15.0,
                "step_dt": 1.0 / 15.0,
                "physics_hz": 120.0,
                "physics_dt": 1.0 / 120.0,
                "decimation": 8,
                "reset_seed": RESET_SEED,
                "initial_condition_index": INITIAL_CONDITION_INDEX,
            },
        ),
        "runtime protocol drift",
    )
    _require(
        isinstance(frame, dict) and set(frame) == boundary.RUNTIME_FRAME_FIELDS,
        "runtime frame schema",
    )
    for field, expected in (
        ("eef_frame", "panda_link8"),
        ("reference_frame", "panda_link0"),
        ("controlled_body", "panda_link8"),
        ("body_offset", "identity"),
        ("command_type", "pose"),
        ("use_relative_mode", False),
        ("ik_method", "dls"),
        ("dls_damping", 0.01),
        ("arm_scale", 1.0),
        ("action_dim", 7),
        ("ik_safety_profile", arm_safety.get("profile")),
        (
            "gripper_threshold_profile",
            "closed_positive_ge_0p5_inverse_open_gt_0p5_v1",
        ),
    ):
        _require(_typed_equal(frame.get(field), expected), f"runtime frame {field}")
    _require(
        frame.get("arm_joint_names")
        == [f"panda_joint{index}" for index in range(1, 8)],
        "runtime frame arm joint names",
    )
    for field, tolerance in (
        ("position_error_m", 1e-5),
        ("rotation_error_rad", math.radians(0.01)),
    ):
        value = frame.get(field)
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and 0.0 <= float(value) <= tolerance,
            f"runtime frame {field}",
        )


def _validate_arm_trace_vector(value: Any, *, field: str, width: int) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{field} vector")
    _require(
        set(value) == {"values", "finite_mask", "finite_count"},
        f"{field} vector schema",
    )
    values = value.get("values")
    mask = value.get("finite_mask")
    _require(
        isinstance(values, list)
        and len(values) == width
        and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            for item in values
        ),
        f"{field} vector values",
    )
    _require(
        isinstance(mask, list)
        and len(mask) == width
        and all(type(item) is bool for item in mask)
        and mask == [True] * width
        and type(value.get("finite_count")) is int
        and value.get("finite_count") == width,
        f"{field} vector finiteness",
    )
    return dict(value)


def _validate_closed_arm_substep_trace(
    trace: Any, *, expected_total_completed: int
) -> list[dict[str, Any]]:
    """Validate every retained arm command without assuming a failure guard."""

    _require(isinstance(trace, dict), "arm substep trace")
    _require(
        set(trace) == boundary.FAILURE_SUBSTEP_TRACE_FIELDS,
        "arm substep trace schema",
    )
    exact_header = {
        "schema_version": 1,
        "profile": boundary.FAILURE_SUBSTEP_TRACE_PROFILE,
        "episode_index": 0,
        "capacity": boundary.FAILURE_SUBSTEP_TRACE_CAPACITY,
        "policy_step_capacity": boundary.FAILURE_SUBSTEP_TRACE_CAPACITY // DECIMATION,
        "decimation": DECIMATION,
        "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
        "effort_semantics": boundary.FAILURE_SUBSTEP_TRACE_EFFORT_SEMANTICS,
        "phase_contract": boundary.FAILURE_SUBSTEP_TRACE_PHASE_CONTRACT,
    }
    for field, expected in exact_header.items():
        _require(trace.get(field) == expected, f"arm trace header {field}")
    for field in (
        "schema_version",
        "episode_index",
        "capacity",
        "policy_step_capacity",
        "decimation",
    ):
        _require(type(trace.get(field)) is int, f"arm trace header {field} type")
    for field, expected in (
        ("joint_drive_stiffness", boundary.EXPECTED_JOINT_DRIVE_STIFFNESS),
        ("joint_drive_damping", boundary.EXPECTED_JOINT_DRIVE_DAMPING),
        ("joint_effort_limits", boundary.EXPECTED_EFFORT_LIMITS),
    ):
        values = trace.get(field)
        _require(
            isinstance(values, list)
            and len(values) == 7
            and all(
                isinstance(actual, (int, float))
                and not isinstance(actual, bool)
                and math.isfinite(float(actual))
                and _same_float32(actual, wanted)
                for actual, wanted in zip(values, expected, strict=True)
            ),
            f"arm trace header {field}",
        )
    _require(
        type(expected_total_completed) is int and expected_total_completed > 0,
        "arm trace expected total",
    )
    expected_retained = min(
        expected_total_completed, boundary.FAILURE_SUBSTEP_TRACE_CAPACITY
    )
    expected_dropped = expected_total_completed - expected_retained
    for field in (
        "total_completed_entry_count",
        "completed_entry_count",
        "dropped_prefix_entry_count",
        "pending_entry_count",
    ):
        _require(type(trace.get(field)) is int, f"arm trace lifecycle {field} type")
    _require(
        trace.get("total_completed_entry_count") == expected_total_completed
        and trace.get("completed_entry_count") == expected_retained
        and trace.get("dropped_prefix_entry_count") == expected_dropped
        and trace.get("pending_entry_count") == 0
        and trace.get("pending_apply_index") is None,
        "arm trace lifecycle/counts",
    )
    entries = trace.get("entries")
    _require(
        isinstance(entries, list) and len(entries) == expected_retained,
        "arm trace retained entries",
    )
    expected_indices = list(range(expected_dropped, expected_total_completed))
    validated: list[dict[str, Any]] = []
    for offset, (entry, apply_index) in enumerate(
        zip(entries, expected_indices, strict=True)
    ):
        field = f"arm trace entry {offset}"
        _require(
            isinstance(entry, dict)
            and set(entry) == boundary.FAILURE_SUBSTEP_TRACE_ENTRY_FIELDS,
            f"{field} schema",
        )
        _require(
            all(
                type(entry.get(index_field)) is int
                for index_field in ("apply_index", "policy_step", "physics_substep")
            ),
            f"{field} index types",
        )
        _require(
            entry.get("apply_index") == apply_index
            and entry.get("policy_step") == apply_index // DECIMATION
            and entry.get("physics_substep") == apply_index % DECIMATION,
            f"{field} apply identity",
        )
        for vector_name, width in boundary.FAILURE_SUBSTEP_TRACE_VECTOR_WIDTHS.items():
            _validate_arm_trace_vector(
                entry.get(vector_name),
                field=f"{field}.{vector_name}",
                width=width,
            )
        _require(
            entry["new_joint_vel_target_rad_s"]["values"] == [0.0] * 7
            and entry["new_joint_effort_target_nm"]["values"] == [0.0] * 7,
            f"{field} zero velocity/effort targets",
        )
        for prefix, unit in (("joint_pos", "rad"), ("joint_vel", "rad_s")):
            pre = entry[f"{prefix}_{unit}"]["values"]
            post = entry[f"post_{prefix}_{unit}"]["values"]
            delta = entry[f"delta_{prefix}_{unit}"]["values"]
            expected_delta = [
                boundary._float32_subtract(after, before)  # noqa: SLF001
                for before, after in zip(pre, post, strict=True)
            ]
            _require(
                all(
                    _same_float32(actual, wanted)
                    for actual, wanted in zip(delta, expected_delta, strict=True)
                ),
                f"{field} {prefix} delta semantics",
            )
        q = entry["joint_pos_rad"]["values"]
        dq = entry["joint_vel_rad_s"]["values"]
        q_target = entry["new_joint_pos_target_rad"]["values"]
        dq_target = entry["new_joint_vel_target_rad_s"]["values"]
        effort_target = entry["new_joint_effort_target_nm"]["values"]
        preclip = entry["approximate_pd_effort_preclip_nm"]["values"]
        postclip = entry["approximate_pd_effort_postclip_nm"]["values"]
        for joint_index in range(7):
            position_term = boundary._float32_multiply(  # noqa: SLF001
                boundary.EXPECTED_JOINT_DRIVE_STIFFNESS[joint_index],
                boundary._float32_subtract(  # noqa: SLF001
                    q_target[joint_index], q[joint_index]
                ),
            )
            velocity_term = boundary._float32_multiply(  # noqa: SLF001
                boundary.EXPECTED_JOINT_DRIVE_DAMPING[joint_index],
                boundary._float32_subtract(  # noqa: SLF001
                    dq_target[joint_index], dq[joint_index]
                ),
            )
            expected_preclip = boundary._float32_add(  # noqa: SLF001
                boundary._float32_add(position_term, velocity_term),  # noqa: SLF001
                effort_target[joint_index],
            )
            effort_limit = boundary.EXPECTED_EFFORT_LIMITS[joint_index]
            expected_postclip = min(max(expected_preclip, -effort_limit), effort_limit)
            _require(
                _same_float32(preclip[joint_index], expected_preclip)
                and _same_float32(postclip[joint_index], expected_postclip),
                f"{field} approximate PD effort semantics",
            )
        for quaternion_field in (
            "current_eef_quaternion_wxyz",
            "desired_eef_quaternion_wxyz",
        ):
            quaternion = entry[quaternion_field]["values"]
            norm = math.sqrt(sum(component * component for component in quaternion))
            _require(
                abs(norm - 1.0) <= 1e-3,
                f"{field} {quaternion_field} unit norm",
            )
        validated.append(entry)
    for previous, current in zip(validated, validated[1:], strict=False):
        _require(
            previous["post_joint_pos_rad"] == current["joint_pos_rad"]
            and previous["post_joint_vel_rad_s"] == current["joint_vel_rad_s"]
            and previous["new_joint_pos_target_rad"]
            == current["previous_joint_pos_target_rad"],
            "arm trace causal entry continuity",
        )
    return validated


def _validate_arm_substep_trace_terminal(
    trace: Any,
    *,
    outcome: Mapping[str, Any],
    arm_safety: Mapping[str, Any],
    finger_trace: Mapping[str, Any],
    failure_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    arm_apply_calls = arm_safety["counters"]["apply_calls"]
    _require(type(arm_apply_calls) is int and arm_apply_calls > 0, "arm apply count")
    expected_completed = arm_apply_calls - int(
        outcome["kind"] == "allowed_velocity_guard_failure"
    )
    entries = _validate_closed_arm_substep_trace(
        trace, expected_total_completed=expected_completed
    )
    final_arm = entries[-1]
    _require(
        final_arm.get("apply_index") == expected_completed - 1,
        "arm terminal completed apply index",
    )
    finger_entries = finger_trace.get("entries")
    _require(
        isinstance(finger_entries, list) and finger_entries,
        "finger terminal trace entries",
    )
    final_finger = finger_entries[-1]
    _require(
        final_finger.get("apply_index") == final_arm.get("apply_index"),
        "arm/finger terminal apply identity",
    )
    snapshot = final_finger["post_physics"]
    joint_names = snapshot["joint_names"]
    arm_names = [f"panda_joint{index}" for index in range(1, 8)]
    _require(trace.get("joint_names") == arm_names, "arm trace joint names")
    _require(
        all(name in joint_names for name in arm_names),
        "finger terminal snapshot is missing arm joints",
    )
    arm_indices = [joint_names.index(name) for name in arm_names]
    arm_by_apply = {entry["apply_index"]: entry for entry in entries}
    for finger_offset, finger_entry in enumerate(finger_entries):
        apply_index = finger_entry["apply_index"]
        _require(
            apply_index in arm_by_apply, "finger entry missing overlapping arm trace"
        )
        arm_entry = arm_by_apply[apply_index]
        pre_snapshot = finger_entry["pre_apply"]
        post_snapshot = finger_entry["post_physics"]
        for arm_field, snapshot_phase, snapshot_field in (
            ("joint_pos_rad", pre_snapshot, "joint_position_rad"),
            ("joint_vel_rad_s", pre_snapshot, "joint_velocity_rad_s"),
            ("new_joint_pos_target_rad", pre_snapshot, "joint_position_target_rad"),
            (
                "new_joint_vel_target_rad_s",
                pre_snapshot,
                "joint_velocity_target_rad_s",
            ),
            ("new_joint_effort_target_nm", pre_snapshot, "joint_effort_target_nm"),
            ("post_joint_pos_rad", post_snapshot, "joint_position_rad"),
            ("post_joint_vel_rad_s", post_snapshot, "joint_velocity_rad_s"),
            (
                "approximate_pd_effort_preclip_nm",
                post_snapshot,
                "approximate_pd_computed_torque_nm",
            ),
            (
                "approximate_pd_effort_postclip_nm",
                post_snapshot,
                "approximate_pd_applied_torque_nm",
            ),
        ):
            arm_vector = arm_entry[arm_field]
            snapshot_vector = snapshot_phase[snapshot_field]
            selected = [snapshot_vector["values"][index] for index in arm_indices]
            selected_mask = [
                snapshot_vector["finite_mask"][index] for index in arm_indices
            ]
            _require(
                arm_vector["finite_mask"] == selected_mask == [True] * 7
                and arm_vector["values"] == selected,
                f"arm/finger overlapping entry {finger_offset} {arm_field} identity",
            )
    for cached_field, direct_field in (
        ("joint_position_rad", "physx_joint_position_rad"),
        ("joint_velocity_rad_s", "physx_joint_velocity_rad_s"),
    ):
        cached = snapshot[cached_field]
        direct = snapshot[direct_field]
        cached_values = [cached["values"][index] for index in arm_indices]
        direct_values = [direct["values"][index] for index in arm_indices]
        cached_mask = [cached["finite_mask"][index] for index in arm_indices]
        direct_mask = [direct["finite_mask"][index] for index in arm_indices]
        _require(
            cached_mask == direct_mask == [True] * 7 and cached_values == direct_values,
            f"finger terminal cached/direct PhysX arm {cached_field} identity",
        )
    is_failure = outcome["kind"] == "allowed_velocity_guard_failure"
    if is_failure:
        _require(
            isinstance(failure_evidence, Mapping),
            "failure terminal PhysX evidence",
        )
        for direct_field, failure_field in (
            ("physx_joint_position_rad", "physx_arm_joint_pos_rad"),
            ("physx_joint_velocity_rad_s", "physx_arm_joint_vel_rad_s"),
        ):
            direct = snapshot[direct_field]
            failure_vector = failure_evidence.get(failure_field)
            direct_values = [direct["values"][index] for index in arm_indices]
            direct_mask = [direct["finite_mask"][index] for index in arm_indices]
            _require(
                isinstance(failure_vector, dict)
                and failure_vector.get("finite_mask") == direct_mask == [True] * 7
                and failure_vector.get("finite_count") == 7
                and failure_vector.get("values") == direct_values,
                f"finger/failure terminal PhysX arm {failure_field} identity",
            )
    else:
        _require(
            failure_evidence is None,
            "horizon terminal unexpectedly has failure evidence",
        )
    return dict(trace)


def _select_tensor_evidence(
    value: Mapping[str, Any], indices: Sequence[int], *, field: str
) -> dict[str, Any]:
    tensor = validate_tensor_evidence(value, field=field)
    _require(
        len(tensor["shape"]) == 1
        and all(
            type(index) is int and 0 <= index < tensor["shape"][0] for index in indices
        ),
        f"{field} selection",
    )
    selected_mask = [tensor["finite_mask"][index] for index in indices]
    _require(selected_mask == [True] * len(indices), f"{field} selected finiteness")
    return {
        "shape": [len(indices)],
        "dtype": tensor["dtype"],
        "device": tensor["device"],
        "values": [tensor["values"][index] for index in indices],
        "finite_mask": selected_mask,
        "finite_count": len(indices),
        "nonfinite": [],
    }


def _validate_failure_runtime_evidence(
    evidence: Any,
    *,
    outcome: Mapping[str, Any],
    arm_safety: Mapping[str, Any],
    arm_substep_trace: Mapping[str, Any],
    finger_trace: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        isinstance(evidence, dict) and set(evidence) == FAILURE_RUNTIME_EVIDENCE_FIELDS,
        "arm failure runtime evidence schema",
    )
    failure_step = outcome["failure_policy_step"]
    failure_apply_index = outcome["failure_apply_index"]
    _require(
        type(evidence.get("policy_step")) is int
        and evidence.get("policy_step") == failure_step
        and type(arm_safety["counters"]["apply_calls"]) is int
        and arm_safety["counters"]["apply_calls"] == failure_apply_index + 1,
        "arm failure policy/apply identity",
    )
    arm_names = [f"panda_joint{index}" for index in range(1, 8)]
    _require(evidence.get("arm_joint_names") == arm_names, "arm failure joint names")
    final_entry = finger_trace["entries"][-1]
    snapshot = final_entry["post_physics"]
    timestamp = evidence.get("articulation_data_sim_timestamp")
    _require(
        type(timestamp) is float
        and math.isfinite(timestamp)
        and timestamp == snapshot["articulation_data_sim_timestamp"],
        "arm failure articulation timestamp",
    )
    joint_names = snapshot["joint_names"]
    _require(
        all(name in joint_names for name in arm_names), "arm failure snapshot joints"
    )
    arm_indices = [joint_names.index(name) for name in arm_names]
    vector_map = {
        "arm_joint_pos_rad": "joint_position_rad",
        "arm_joint_vel_rad_s": "joint_velocity_rad_s",
        "arm_joint_target_rad": "joint_position_target_rad",
        "arm_joint_velocity_target_rad_s": "joint_velocity_target_rad_s",
        "arm_joint_effort_target_nm": "joint_effort_target_nm",
        "physx_arm_joint_pos_rad": "physx_joint_position_rad",
        "physx_arm_joint_vel_rad_s": "physx_joint_velocity_rad_s",
        "arm_computed_torque": "approximate_pd_computed_torque_nm",
        "arm_applied_torque": "approximate_pd_applied_torque_nm",
    }
    validated_vectors: dict[str, dict[str, Any]] = {}
    for evidence_field, snapshot_field in vector_map.items():
        vector = _validate_arm_trace_vector(
            evidence.get(evidence_field),
            field=f"failure evidence {evidence_field}",
            width=7,
        )
        selected = [snapshot[snapshot_field]["values"][index] for index in arm_indices]
        selected_mask = [
            snapshot[snapshot_field]["finite_mask"][index] for index in arm_indices
        ]
        _require(
            selected_mask == [True] * 7 and vector["values"] == selected,
            f"failure evidence {evidence_field} terminal identity",
        )
        validated_vectors[evidence_field] = vector
    for delta_field, cached_field, direct_field in (
        (
            "cached_minus_physx_arm_joint_pos_rad",
            "arm_joint_pos_rad",
            "physx_arm_joint_pos_rad",
        ),
        (
            "cached_minus_physx_arm_joint_vel_rad_s",
            "arm_joint_vel_rad_s",
            "physx_arm_joint_vel_rad_s",
        ),
    ):
        delta = _validate_arm_trace_vector(
            evidence.get(delta_field), field=f"failure evidence {delta_field}", width=7
        )
        expected = [
            boundary._float32_subtract(cached, direct)  # noqa: SLF001
            for cached, direct in zip(
                validated_vectors[cached_field]["values"],
                validated_vectors[direct_field]["values"],
                strict=True,
            )
        ]
        _require(
            all(
                _same_float32(actual, wanted)
                for actual, wanted in zip(delta["values"], expected, strict=True)
            ),
            f"failure evidence {delta_field} arithmetic",
        )
    for field, expected in (
        ("physx_arm_velocity_limits_rad_s", boundary.EXPECTED_VELOCITY_LIMITS_RAD_S),
        ("physx_arm_effort_limits", boundary.EXPECTED_EFFORT_LIMITS),
    ):
        vector = _validate_arm_trace_vector(
            evidence.get(field), field=f"failure evidence {field}", width=7
        )
        _require(
            all(
                _same_float32(actual, wanted)
                for actual, wanted in zip(vector["values"], expected, strict=True)
            ),
            f"failure evidence {field} pinned values",
        )
    for evidence_field, snapshot_field, expected in (
        (
            "physx_arm_projected_joint_force_generalized_si",
            "physx_projected_joint_force_nm",
            None,
        ),
        (
            "physx_arm_stiffness_nm_per_rad",
            "physx_joint_stiffness_nm_per_rad",
            boundary.EXPECTED_JOINT_DRIVE_STIFFNESS,
        ),
        (
            "physx_arm_damping_nm_s_per_rad",
            "physx_joint_damping_nm_s_per_rad",
            boundary.EXPECTED_JOINT_DRIVE_DAMPING,
        ),
    ):
        selected = _select_tensor_evidence(
            snapshot[snapshot_field], arm_indices, field=f"snapshot {snapshot_field}"
        )
        actual = validate_tensor_evidence(
            evidence.get(evidence_field), field=f"failure evidence {evidence_field}"
        )
        _require(
            actual == selected, f"failure evidence {evidence_field} readback identity"
        )
        if expected is not None:
            _require(
                all(
                    _same_float32(observed, wanted)
                    for observed, wanted in zip(actual["values"], expected, strict=True)
                ),
                f"failure evidence {evidence_field} pinned values",
            )
    _require(
        _typed_equal(evidence.get("ik_safety"), arm_safety),
        "arm failure safety identity",
    )
    _require(
        _typed_equal(evidence.get("controller_substep_trace"), arm_substep_trace)
        and evidence.get("controller_substep_trace_error") is None,
        "arm failure trace identity",
    )
    return dict(evidence)


def validate_capture_payload(
    payload: Any,
    *,
    expected_mode: str,
    expected_gripper_drive_profile: str | None = None,
) -> dict[str, Any]:
    _require(expected_mode in MODES, "expected validation mode")
    if expected_gripper_drive_profile is not None:
        _require(
            expected_gripper_drive_profile in GRIPPER_DRIVE_PROFILES,
            "expected gripper drive profile",
        )
    _require(isinstance(payload, dict), "capture must be an object")
    _require(set(payload) == PAYLOAD_FIELDS, "capture top-level schema drift")
    _exact_int(payload.get("schema_version"), 1, field="capture schema version")
    _require(payload.get("diagnostic_profile") == DIAGNOSTIC_PROFILE, "capture profile")
    _require(payload.get("fixture_profile") == FIXTURE_PROFILE, "capture fixture")
    _require(payload.get("finalized") is False, "capture finalized flag")
    _require(
        payload.get("capture_valid") is True,
        "capture-valid flag (not a controller pass/fail claim)",
    )
    _require(
        payload.get("stage") == "simulation_app_close_pending"
        and type(payload.get("exit_code")) is int
        and payload.get("exit_code") == 0
        and payload.get("environment") == ENVIRONMENT
        and payload.get("mode") == expected_mode,
        "capture terminal identity",
    )
    _require(payload.get("close_failures") == [], "capture close failures")
    _validate_diagnostic_source(payload.get("diagnostic_source"))
    validate_runtime_exit_contract(payload.get("runtime_exit_contract"))
    fixture = payload.get("fixture")
    _require(
        isinstance(fixture, dict)
        and set(fixture) == FIXTURE_IDENTITY_FIELDS
        and fixture.get("fixture_profile") == FIXTURE_PROFILE
        and isinstance(fixture.get("path"), str)
        and type(fixture.get("size_bytes")) is int
        and fixture.get("size_bytes") == boundary.EXPECTED_FIXTURE_SIZE_BYTES
        and fixture.get("sha256") == boundary.EXPECTED_FIXTURE_SHA256
        and isinstance(fixture.get("mode"), str)
        and type(fixture.get("action_count")) is int
        and fixture.get("action_count") == 378,
        "capture fixture identity",
    )
    helper = payload.get("boundary_helper_source")
    _require(
        isinstance(helper, dict)
        and set(helper) == FILE_IDENTITY_FIELDS
        and type(helper.get("size_bytes")) is int
        and helper.get("size_bytes") == EXPECTED_BOUNDARY_HELPER_SIZE_BYTES
        and type(helper.get("nlink")) is int
        and helper.get("nlink") == 1
        and helper.get("sha256") == EXPECTED_BOUNDARY_HELPER_SHA256,
        "boundary helper identity",
    )
    plan = validate_action_plan(payload.get("action_plan"))
    _require(plan["mode"] == expected_mode, "capture action-plan mode")
    _validate_solver_contract(payload.get("solver_contract"))
    gripper_drive = _validate_gripper_drive_contract(
        payload.get("gripper_drive_contract"),
        expected_profile=expected_gripper_drive_profile,
    )
    outcome = _validate_outcome(payload.get("outcome"), mode=expected_mode)
    video_phase = validate_video_phase_contract(
        payload.get("video_phase_contract"), outcome=outcome
    )
    video = payload.get("video")
    _require(
        isinstance(video, dict) and set(video) == VIDEO_IDENTITY_FIELDS, "video schema"
    )
    expected_frames = video_phase["total_frame_count"]
    _require(
        video.get("profile") == VIDEO_PROFILE
        and type(video.get("fps")) is int
        and video.get("fps") == VIDEO_FPS
        and type(video.get("frame_count")) is int
        and video.get("frame_count") == expected_frames
        and type(video.get("height")) is int
        and video.get("height") == VIDEO_HEIGHT
        and type(video.get("width")) is int
        and video.get("width") == VIDEO_WIDTH
        and type(video.get("size_bytes")) is int
        and video.get("size_bytes") > 0
        and video.get("mode") == "0444"
        and type(video.get("nlink")) is int
        and video.get("nlink") == 1,
        "video identity contract",
    )
    arm_safety = payload.get("arm_safety")
    _require(isinstance(arm_safety, dict), "arm safety capture")
    _validate_runtime_identity(
        payload.get("runtime_protocol"),
        payload.get("runtime_frame"),
        arm_safety=arm_safety,
    )
    finger_trace = validate_finger_trace(
        payload.get("finger_trace"),
        action_plan=plan,
        outcome=outcome,
        arm_safety=arm_safety,
        gripper_drive=gripper_drive,
    )
    first_snapshot = finger_trace["entries"][0]["pre_apply"]
    gripper_index = gripper_drive["joint_indices"][0]
    _require(
        gripper_index < len(first_snapshot["joint_names"])
        and first_snapshot["joint_names"][gripper_index] == "finger_joint",
        "gripper drive/trace joint identity",
    )
    failure_evidence = payload.get("arm_failure_runtime_evidence")
    if outcome["kind"] == "allowed_velocity_guard_failure":
        _require(
            isinstance(failure_evidence, dict)
            and set(failure_evidence) == FAILURE_RUNTIME_EVIDENCE_FIELDS,
            "arm failure runtime evidence schema",
        )
    else:
        _require(failure_evidence is None, "unexpected horizon failure evidence")
    arm_substep_trace = _validate_arm_substep_trace_terminal(
        payload.get("arm_substep_trace"),
        outcome=outcome,
        arm_safety=arm_safety,
        finger_trace=finger_trace,
        failure_evidence=failure_evidence,
    )
    if outcome["kind"] == "allowed_velocity_guard_failure":
        failure_evidence = _validate_failure_runtime_evidence(
            failure_evidence,
            outcome=outcome,
            arm_safety=arm_safety,
            arm_substep_trace=arm_substep_trace,
            finger_trace=finger_trace,
        )
        boundary.validate_failure_substep_trace(
            arm_substep_trace,
            safety=arm_safety,
            failure_policy_step=outcome["failure_policy_step"],
            current_joint_pos=failure_evidence["arm_joint_pos_rad"],
            current_joint_vel=failure_evidence["arm_joint_vel_rad_s"],
            current_joint_pos_target=failure_evidence["arm_joint_target_rad"],
            current_joint_vel_target=failure_evidence[
                "arm_joint_velocity_target_rad_s"
            ],
            current_joint_effort_target=failure_evidence["arm_joint_effort_target_nm"],
            current_approximate_pd_effort_preclip=failure_evidence[
                "arm_computed_torque"
            ],
            current_approximate_pd_effort_postclip=failure_evidence[
                "arm_applied_torque"
            ],
            physx_joint_pos=failure_evidence["physx_arm_joint_pos_rad"],
            physx_joint_vel=failure_evidence["physx_arm_joint_vel_rad_s"],
        )
        safety_static_probe = copy.deepcopy(arm_safety)
        safety_static_probe["maxima"]["abs_joint_vel_rad_s"] = [
            min(value, limit)
            for value, limit in zip(
                arm_safety["maxima"]["abs_joint_vel_rad_s"],
                boundary.EXPECTED_VELOCITY_LIMITS_RAD_S,
                strict=True,
            )
        ]
        boundary.validate_safety_static(safety_static_probe, episode_index=0)
    else:
        counters = arm_safety.get("counters", {})
        _require(
            all(
                type(counters.get(field)) is int and counters.get(field) == 0
                for field in boundary.ZERO_COUNTERS
            ),
            "horizon arm safety",
        )
        boundary.validate_safety_static(arm_safety, episode_index=0)
    _validate_assets(payload.get("assets"))
    return dict(payload)


def validate_capture_artifacts(
    capture_path: Path,
    video_path: Path,
    *,
    expected_mode: str,
    expected_gripper_drive_profile: str | None = None,
    probe: Callable[[Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Host-only post-Kit validator for the immutable JSON and MP4 pair."""

    capture_path = capture_path.resolve()
    video_path = video_path.resolve()
    payload = boundary.strict_json_loads(
        capture_path.read_bytes(), field="gripper impulse capture"
    )
    validated = validate_capture_payload(
        payload,
        expected_mode=expected_mode,
        expected_gripper_drive_profile=expected_gripper_drive_profile,
    )
    capture_identity = _file_identity(capture_path)
    _require(
        capture_identity["mode"] == "0444" and capture_identity["nlink"] == 1,
        "capture file immutability",
    )
    actual_video = _file_identity(video_path)
    recorded_video = validated["video"]
    for field in ("path", "size_bytes", "sha256", "mode", "nlink"):
        _require(
            _typed_equal(actual_video[field], recorded_video[field]),
            f"video file identity {field}",
        )
    if probe is None:
        from polaris.eval_artifacts import probe_episode_video  # noqa: PLC0415

        probe = probe_episode_video
    observed = dict(probe(video_path))
    _require(
        _typed_equal(
            observed,
            {
                "frame_count": recorded_video["frame_count"],
                "height": VIDEO_HEIGHT,
                "width": VIDEO_WIDTH,
            },
        ),
        "post-Kit video decode",
    )
    helper_identity = _file_identity(BOUNDARY_HELPER_PATH)
    _require(
        helper_identity["size_bytes"] == EXPECTED_BOUNDARY_HELPER_SIZE_BYTES
        and helper_identity["sha256"] == EXPECTED_BOUNDARY_HELPER_SHA256,
        "live boundary helper drift",
    )
    _require(
        _typed_equal(validated["boundary_helper_source"], helper_identity),
        "recorded/live boundary helper identity drift",
    )
    diagnostic_source = validated["diagnostic_source"]
    live_diagnostic_source = _file_identity(Path(__file__))
    _require(
        _typed_equal(diagnostic_source["actual"], live_diagnostic_source),
        "recorded/live diagnostic source identity drift",
    )
    fixture_identity, fixture_actions = boundary.load_replay_fixture()
    _require(
        _typed_equal(validated["fixture"], fixture_identity),
        "live replay fixture identity drift",
    )
    expected_plan, _ = build_action_plan(fixture_actions, mode=expected_mode)
    _require(
        _typed_equal(validated["action_plan"], expected_plan),
        "live action plan/mutation manifest drift",
    )
    assets = validated["assets"]
    for field in ("scene", "initial_conditions"):
        recorded = assets["foodbussing"][field]
        actual = _file_identity(Path(recorded["path"]))
        for identity_field in BOUNDARY_FILE_IDENTITY_FIELDS:
            _require(
                _typed_equal(actual[identity_field], recorded[identity_field]),
                f"live FoodBussing {field} identity {identity_field}",
            )
    for filename, recorded in assets["foodbussing"]["revision_metadata"].items():
        actual = _file_identity(Path(recorded["path"]))
        for identity_field in BOUNDARY_FILE_IDENTITY_FIELDS:
            _require(
                _typed_equal(actual[identity_field], recorded[identity_field]),
                f"live FoodBussing metadata {filename} identity {identity_field}",
            )
    for recorded in (
        assets["robot_usd"],
        assets["robot_usd_revision_metadata"]["identity"],
    ):
        actual = _file_identity(Path(recorded["path"]))
        _require(
            _typed_equal(actual, recorded),
            f"live asset identity drift: {recorded['path']}",
        )
    return validated


def _probe_video_stdlib(path: Path) -> dict[str, int]:
    """Fully decode and ffprobe the single-stream 15-fps model-view MP4."""

    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            (
                "stream=index,codec_type,width,height,nb_read_frames,"
                "avg_frame_rate,r_frame_rate,duration,duration_ts,time_base:"
                "format=duration"
            ),
            "-of",
            "json",
            str(path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    decoded = json.loads(completed.stdout)
    streams = decoded.get("streams") if isinstance(decoded, dict) else None
    _require(isinstance(streams, list) and len(streams) == 1, "ffprobe video stream")
    stream = streams[0]
    _require(isinstance(stream, dict), "ffprobe video stream schema")
    _require(
        set(stream)
        == {
            "index",
            "codec_type",
            "width",
            "height",
            "nb_read_frames",
            "avg_frame_rate",
            "r_frame_rate",
            "duration",
            "duration_ts",
            "time_base",
        },
        "ffprobe video stream closed schema",
    )
    _require(
        type(stream.get("index")) is int
        and stream.get("index") == 0
        and stream.get("codec_type") == "video",
        "ffprobe exactly one video stream",
    )
    try:
        frame_text = stream["nb_read_frames"]
        _require(
            isinstance(frame_text, str) and frame_text.isdecimal(),
            "ffprobe decoded frame count",
        )
        result = {
            "frame_count": int(frame_text),
            "height": stream["height"],
            "width": stream["width"],
        }
    except (KeyError, TypeError, ValueError) as error:
        raise GripperImpulseDiagnosticError("ffprobe video values") from error
    _require(
        all(type(value) is int and value > 0 for value in result.values()),
        "ffprobe video dimensions/count",
    )
    for field in ("avg_frame_rate", "r_frame_rate"):
        try:
            frame_rate = Fraction(stream[field])
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
            raise GripperImpulseDiagnosticError(f"ffprobe video {field}") from error
        _require(frame_rate == Fraction(VIDEO_FPS, 1), f"ffprobe video {field}")
    format_record = decoded.get("format")
    _require(
        isinstance(format_record, dict) and set(format_record) == {"duration"},
        "ffprobe format closed schema",
    )
    try:
        time_base_text = stream["time_base"]
        _require(
            isinstance(time_base_text, str)
            and re.fullmatch(r"[1-9][0-9]*/[1-9][0-9]*", time_base_text) is not None,
            "ffprobe video time_base grammar",
        )
        time_base = Fraction(time_base_text)
        duration_ts = stream["duration_ts"]
        _require(
            type(duration_ts) is int and duration_ts > 0,
            "ffprobe video duration_ts",
        )
        decimal_pattern = (
            rf"(?:0|[1-9][0-9]*)\.[0-9]{{{FFPROBE_DURATION_DECIMAL_PLACES}}}"
        )
        stream_duration_text = stream["duration"]
        format_duration_text = format_record["duration"]
        _require(
            isinstance(stream_duration_text, str)
            and re.fullmatch(decimal_pattern, stream_duration_text) is not None,
            "ffprobe video stream duration grammar",
        )
        _require(
            isinstance(format_duration_text, str)
            and re.fullmatch(decimal_pattern, format_duration_text) is not None,
            "ffprobe video format duration grammar",
        )
        stream_duration = Fraction(stream_duration_text)
        format_duration = Fraction(format_duration_text)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise GripperImpulseDiagnosticError(
            "ffprobe video rational duration"
        ) from error
    expected_duration = Fraction(result["frame_count"], VIDEO_FPS)
    decoded_tick_duration = duration_ts * time_base
    stream_decimal_half_quantum = Fraction(1, 2 * (10**FFPROBE_DURATION_DECIMAL_PLACES))
    expected_container_duration = Fraction(
        (
            expected_duration.numerator * MP4_CONTAINER_DURATION_TICKS_PER_SECOND
            + expected_duration.denominator
            - 1
        )
        // expected_duration.denominator,
        MP4_CONTAINER_DURATION_TICKS_PER_SECOND,
    )
    _require(
        decoded_tick_duration == expected_duration,
        "ffprobe video frame-count/duration_ts/time_base cadence",
    )
    _require(
        stream_duration > 0
        and abs(stream_duration - expected_duration) <= stream_decimal_half_quantum,
        "ffprobe video rational stream duration cadence",
    )
    _require(
        format_duration in {stream_duration, expected_container_duration},
        "ffprobe video canonical stream-or-millisecond-ceiling container duration",
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result


def _host_revalidate_runtime_artifacts(args_cli: argparse.Namespace) -> None:
    """Revalidate the child products in the stdlib-only parent before status."""

    capture = validate_capture_artifacts(
        args_cli.output_json,
        args_cli.output_video,
        expected_mode=args_cli.mode,
        expected_gripper_drive_profile=_selected_gripper_drive_profile(
            args_cli.enable_gripper_velocity_limit_candidate
        ),
        probe=_probe_video_stdlib,
    )
    raw_identity = _file_identity(args_cli.output_json)
    ready_identity = _file_identity(args_cli.output_ready_marker)
    _require(
        ready_identity["mode"] == "0444" and ready_identity["nlink"] == 1,
        "parent ready-marker immutability",
    )
    ready = boundary.strict_json_loads(
        args_cli.output_ready_marker.read_bytes(),
        field="parent gripper impulse ready marker",
    )
    validate_ready_marker(
        ready,
        mode=args_cli.mode,
        raw_identity=raw_identity,
        video_identity=capture["video"],
        diagnostic_source=args_cli.diagnostic_source,
        runtime_exit_contract=args_cli.runtime_exit_contract,
    )
    _require(
        _typed_equal(capture["diagnostic_source"], args_cli.diagnostic_source),
        "child/parent diagnostic source identity",
    )
    _require(
        _typed_equal(capture["runtime_exit_contract"], args_cli.runtime_exit_contract),
        "child/parent runtime-exit contract",
    )
    _require(
        not os.path.lexists(args_cli.runtime_exit),
        "runtime exit status exists before parent publication",
    )


def _load_finalizer_module() -> Any:
    finalizer_path = (
        Path(__file__)
        .resolve()
        .with_name("finalize_eef_pose_gripper_impulse_diagnostic.py")
    )
    spec = importlib.util.spec_from_file_location(
        "polaris_gripper_impulse_finalizer_cli", finalizer_path
    )
    _require(spec is not None and spec.loader is not None, "cannot load finalizer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_validation_parser() -> argparse.ArgumentParser:
    return _load_finalizer_module().build_parser()


def _validation_main(argv: Sequence[str]) -> int:
    module = _load_finalizer_module()
    return module.main(argv)


def _exception_evidence(error: BaseException) -> dict[str, str]:
    return {
        "type": f"{type(error).__module__}.{type(error).__qualname__}",
        "message": str(error),
        "traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
    }


def _scalar_cfg_value(value: Any) -> float | None:
    if value is None:
        return None
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"expected scalar config value, got {value!r}",
    )
    result = float(value)
    _require(math.isfinite(result), "config scalar must be finite")
    return result


def _joint_indices_list(joint_ids: Any, *, joint_count: int) -> list[int]:
    if isinstance(joint_ids, slice):
        return list(range(joint_count))[joint_ids]
    if hasattr(joint_ids, "detach"):
        joint_ids = joint_ids.detach()
    if hasattr(joint_ids, "cpu"):
        joint_ids = joint_ids.cpu()
    if hasattr(joint_ids, "tolist"):
        joint_ids = joint_ids.tolist()
    return [int(value) for value in joint_ids]


def _direct_physx_tensor(robot: Any, getter_name: str) -> Any:
    contract = DIRECT_PHYSX_GETTER_CONTRACT.get(getter_name)
    _require(contract is not None, f"unclassified PhysX getter: {getter_name}")
    getter = getattr(robot.root_physx_view, getter_name, None)
    _require(callable(getter), f"PhysX articulation is missing {getter_name}()")
    tensor = getter()
    _require(
        str(getattr(tensor, "device", "missing")) == contract["device"]
        and str(getattr(tensor, "dtype", "missing")) == PINNED_TENSOR_DTYPE
        and list(getattr(tensor, "shape", ())) == contract["shape"],
        f"{getter_name} field-specific tensor contract drift: "
        f"device={getattr(tensor, 'device', None)!r}, "
        f"dtype={getattr(tensor, 'dtype', None)!r}, "
        f"shape={getattr(tensor, 'shape', None)!r}, expected={contract!r}",
    )
    if hasattr(tensor, "clone"):
        tensor = tensor.clone()
    return tensor


def _capture_articulation_snapshot(robot: Any) -> dict[str, Any]:
    """Clone one snapshot while preserving the pinned CUDA/CPU field partition."""

    sim_timestamp = getattr(robot.data, "_sim_timestamp", None)
    _require(
        isinstance(sim_timestamp, (int, float))
        and not isinstance(sim_timestamp, bool)
        and math.isfinite(float(sim_timestamp)),
        "articulation simulation timestamp",
    )
    joint_names = list(robot.joint_names)
    body_names = list(robot.body_names)
    _require(
        joint_names == EXPECTED_DROID_JOINT_NAMES,
        f"live articulation joint order drift: {joint_names!r}",
    )
    _require(
        body_names == EXPECTED_DROID_BODY_NAMES,
        f"live articulation body order drift: {body_names!r}",
    )
    direct = {
        contract["snapshot_field"]: _direct_physx_tensor(robot, getter_name)[0]
        for getter_name, contract in DIRECT_PHYSX_GETTER_CONTRACT.items()
    }
    cached = {
        "joint_position_rad": robot.data.joint_pos[0],
        "joint_velocity_rad_s": robot.data.joint_vel[0],
        "joint_acceleration_rad_s2": robot.data.joint_acc[0],
        "joint_position_target_rad": robot.data.joint_pos_target[0],
        "joint_velocity_target_rad_s": robot.data.joint_vel_target[0],
        "joint_effort_target_nm": robot.data.joint_effort_target[0],
        "approximate_pd_computed_torque_nm": robot.data.computed_torque[0],
        "approximate_pd_applied_torque_nm": robot.data.applied_torque[0],
    }
    for cached_field, tensor in cached.items():
        _require(
            str(getattr(tensor, "device", "missing")) == PINNED_CACHED_DEVICE
            and str(getattr(tensor, "dtype", "missing")) == PINNED_TENSOR_DTYPE
            and list(getattr(tensor, "shape", ())) == [len(EXPECTED_DROID_JOINT_NAMES)],
            f"cached articulation field contract drift: {cached_field}",
        )
    snapshot: dict[str, Any] = {
        "articulation_data_sim_timestamp": float(sim_timestamp),
        "joint_names": joint_names,
        **{
            name: value.clone() if hasattr(value, "clone") else value
            for name, value in cached.items()
        },
        **direct,
        "body_names": body_names,
        "incoming_joint_wrench_semantics": (
            "physx_link_incoming_joint_total_6d_wrench_child_joint_frame_v1"
        ),
    }
    return snapshot


def _serialize_articulation_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    serialized = {
        "articulation_data_sim_timestamp": snapshot["articulation_data_sim_timestamp"],
        "joint_names": list(snapshot["joint_names"]),
        **{
            field: tensor_evidence(snapshot[field])
            for field in SNAPSHOT_FIELDS
            if field
            not in {
                "articulation_data_sim_timestamp",
                "joint_names",
                "body_names",
                "incoming_joint_wrench_semantics",
            }
        },
        "body_names": list(snapshot["body_names"]),
        "incoming_joint_wrench_semantics": snapshot["incoming_joint_wrench_semantics"],
    }
    return _validate_snapshot(serialized, field="serialized articulation snapshot")


def _make_diagnostic_gripper_class(base_class: type) -> type:
    """Build an ActionTerm used only by this script's private environment cfg."""

    class DiagnosticGripperAction(base_class):
        diagnostic_profile = FINGER_TRACE_PROFILE

        def __init__(self, cfg: Any, env: Any) -> None:
            super().__init__(cfg, env)
            _require(self.num_envs == 1, "gripper diagnostic requires one environment")
            self._diagnostic_context: dict[str, Any] | None = None
            self._diagnostic_raw_action: float | None = None
            self._diagnostic_processed_target: float | None = None
            self._diagnostic_policy_substep = 0
            self._diagnostic_total_staged = 0
            self._diagnostic_total_finalized = 0
            self._diagnostic_pending: dict[str, Any] | None = None
            self._diagnostic_entries: list[dict[str, Any]] = []
            self._diagnostic_dropped = 0
            self._diagnostic_tensor_contract = _expected_tensor_capture_contract()
            _require(
                str(self._asset.data.joint_pos.device) == PINNED_CACHED_DEVICE
                and str(self._asset.data.joint_pos.dtype) == PINNED_TENSOR_DTYPE
                and list(self._asset.data.joint_pos.shape)
                == [1, len(EXPECTED_DROID_JOINT_NAMES)],
                "diagnostic cached articulation tensor contract",
            )

        def reset(self, env_ids: Any = None) -> None:
            super().reset(env_ids=env_ids)
            if not hasattr(self, "_diagnostic_entries"):
                return
            self._diagnostic_context = None
            self._diagnostic_raw_action = None
            self._diagnostic_processed_target = None
            self._diagnostic_policy_substep = 0
            self._diagnostic_total_staged = 0
            self._diagnostic_total_finalized = 0
            self._diagnostic_pending = None
            self._diagnostic_entries.clear()
            self._diagnostic_dropped = 0

        def begin_policy_step(
            self,
            *,
            policy_step: int,
            original_gripper_closed_action: float,
            effective_gripper_closed_action: float,
        ) -> None:
            _require(type(policy_step) is int and policy_step >= 0, "policy step")
            if self._diagnostic_context is not None:
                _require(
                    self._diagnostic_policy_substep == DECIMATION,
                    "previous policy step did not stage eight finger applies",
                )
            self._diagnostic_context = {
                "policy_step": policy_step,
                "original": float(original_gripper_closed_action),
                "effective": float(effective_gripper_closed_action),
            }
            self._diagnostic_policy_substep = 0
            self._diagnostic_raw_action = None
            self._diagnostic_processed_target = None

        def process_actions(self, actions: Any) -> None:
            _require(self._diagnostic_context is not None, "missing policy context")
            super().process_actions(actions)
            raw = float(self._raw_actions[0, 0].detach().cpu().item())
            processed = float(self._processed_actions[0, 0].detach().cpu().item())
            _require(
                _same_float32(raw, self._diagnostic_context["effective"]),
                "processed finger action differs from effective plan",
            )
            wanted = GRIPPER_CLOSED_TARGET_RAD if raw == 1.0 else 0.0
            _require(
                _same_float32(processed, wanted),
                "processed finger target differs from binary plan",
            )
            # These owned scalar copies survive the next policy step's
            # process_actions overwrite before the prior substep is finalized.
            self._diagnostic_raw_action = raw
            self._diagnostic_processed_target = processed

        def _finalize_pending(self, reason: str) -> dict[str, Any] | None:
            pending = self._diagnostic_pending
            _require(pending is not None, "no pending finger apply to finalize")
            entry = pending.get("entry")
            post_snapshot = None
            if entry is not None:
                post_snapshot = _capture_articulation_snapshot(self._asset)
                entry["post_physics"] = post_snapshot
                entry["finalization_reason"] = reason
                self._diagnostic_entries.append(entry)
                if len(self._diagnostic_entries) > TRACE_CAPACITY:
                    self._diagnostic_entries.pop(0)
                    self._diagnostic_dropped += 1
            self._diagnostic_total_finalized += 1
            self._diagnostic_pending = None
            return post_snapshot

        def finalize_pending(self, *, reason: str) -> None:
            self._finalize_pending(reason)

        def has_pending(self) -> bool:
            return self._diagnostic_pending is not None

        def apply_actions(self) -> None:
            current_snapshot = None
            if self._diagnostic_pending is not None:
                current_snapshot = self._finalize_pending("next_gripper_apply")
            context = self._diagnostic_context
            _require(context is not None, "missing finger apply policy context")
            _require(
                self._diagnostic_raw_action is not None
                and self._diagnostic_processed_target is not None,
                "finger apply has no owned processed-action copy",
            )
            policy_step = context["policy_step"]
            physics_substep = self._diagnostic_policy_substep
            _require(physics_substep < DECIMATION, "too many finger policy substeps")
            apply_index = self._diagnostic_total_staged
            _require(
                apply_index == policy_step * DECIMATION + physics_substep,
                "finger apply index/cadence drift",
            )
            entry = None
            if RELEVANT_POLICY_STEP_START <= policy_step <= HORIZON_POLICY_STEP:
                if current_snapshot is None:
                    current_snapshot = _capture_articulation_snapshot(self._asset)
                entry = {
                    "apply_index": apply_index,
                    "policy_step": policy_step,
                    "physics_substep": physics_substep,
                    "original_gripper_closed_action": context["original"],
                    "effective_gripper_closed_action": context["effective"],
                    "raw_action_at_stage": self._diagnostic_raw_action,
                    "processed_target_at_stage_rad": (
                        self._diagnostic_processed_target
                    ),
                    "pre_apply": current_snapshot,
                    "target_after_setter_rad": None,
                    "post_physics": None,
                    "finalization_reason": None,
                }
            super().apply_actions()
            target_after = self._asset.data.joint_pos_target[0, self._joint_ids].clone()
            if entry is not None:
                entry["target_after_setter_rad"] = target_after
            self._diagnostic_pending = {
                "apply_index": apply_index,
                # Pending owns its raw/processed/context copies through
                # process_actions for the next policy step.
                "raw_action_at_stage": self._diagnostic_raw_action,
                "processed_target_at_stage_rad": self._diagnostic_processed_target,
                "context": dict(context),
                "entry": entry,
            }
            self._diagnostic_total_staged += 1
            self._diagnostic_policy_substep += 1

        def diagnostic_trace(self) -> dict[str, Any]:
            entries: list[dict[str, Any]] = []
            for raw_entry in self._diagnostic_entries:
                entry = dict(raw_entry)
                entry["pre_apply"] = _serialize_articulation_snapshot(
                    raw_entry["pre_apply"]
                )
                entry["post_physics"] = _serialize_articulation_snapshot(
                    raw_entry["post_physics"]
                )
                entry["target_after_setter_rad"] = tensor_evidence(
                    raw_entry["target_after_setter_rad"]
                )
                entries.append(entry)
            return {
                "schema_version": 1,
                "profile": FINGER_TRACE_PROFILE,
                "capacity": TRACE_CAPACITY,
                "relevant_policy_step_start": RELEVANT_POLICY_STEP_START,
                "relevant_policy_step_end": HORIZON_POLICY_STEP,
                "total_staged_apply_count": self._diagnostic_total_staged,
                "total_finalized_apply_count": self._diagnostic_total_finalized,
                "pending_apply_count": int(self._diagnostic_pending is not None),
                "dropped_relevant_entry_count": self._diagnostic_dropped,
                "tensor_capture_contract": dict(self._diagnostic_tensor_contract),
                "timestamp_contract": dict(TIMESTAMP_CONTRACT),
                "entries": entries,
            }

    DiagnosticGripperAction.__name__ = "DiagnosticGripperImpulseAction"
    DiagnosticGripperAction.__qualname__ = "DiagnosticGripperImpulseAction"
    return DiagnosticGripperAction


def _capture_gripper_drive_contract(
    robot: Any,
    finger_term: Any,
    *,
    configured_before_build: Mapping[str, Any],
    candidate_enabled: bool,
) -> dict[str, Any]:
    profile = _selected_gripper_drive_profile(candidate_enabled)
    expectations = _gripper_drive_expectations(profile)
    actuator = robot.actuators.get("gripper")
    _require(actuator is not None, "live robot has no gripper actuator")
    action_term_joint_names = list(finger_term._joint_names)
    action_term_joint_indices = _joint_indices_list(
        finger_term._joint_ids, joint_count=len(robot.joint_names)
    )
    actuator_joint_names = list(actuator.joint_names)
    actuator_joint_indices = _joint_indices_list(
        actuator.joint_indices, joint_count=len(robot.joint_names)
    )
    _require(len(action_term_joint_indices) == 1, "gripper diagnostic joint count")
    joint_index = action_term_joint_indices[0]

    def select(tensor: Any) -> Any:
        return tensor[:, [joint_index]]

    live_actuator = {
        "cfg_velocity_limit": _scalar_cfg_value(actuator.cfg.velocity_limit),
        "cfg_velocity_limit_sim": _scalar_cfg_value(actuator.cfg.velocity_limit_sim),
        "cfg_effort_limit": _scalar_cfg_value(actuator.cfg.effort_limit),
        "cfg_effort_limit_sim": _scalar_cfg_value(actuator.cfg.effort_limit_sim),
        "cfg_stiffness": _scalar_cfg_value(actuator.cfg.stiffness),
        "cfg_damping": _scalar_cfg_value(actuator.cfg.damping),
        "resolved_velocity_limit_rad_s": tensor_evidence(actuator.velocity_limit),
        "resolved_velocity_limit_sim_rad_s": tensor_evidence(
            actuator.velocity_limit_sim
        ),
        "resolved_effort_limit_nm": tensor_evidence(actuator.effort_limit),
        "resolved_effort_limit_sim_nm": tensor_evidence(actuator.effort_limit_sim),
        "resolved_stiffness_nm_per_rad": tensor_evidence(actuator.stiffness),
        "resolved_damping_nm_s_per_rad": tensor_evidence(actuator.damping),
    }
    contract = {
        "profile": profile,
        "actuator_name": "gripper",
        "joint_names": action_term_joint_names,
        "joint_indices": action_term_joint_indices,
        "action_term_joint_names": action_term_joint_names,
        "action_term_joint_indices": action_term_joint_indices,
        "actuator_joint_names": actuator_joint_names,
        "actuator_joint_indices": actuator_joint_indices,
        "authoritative_device_probe": dict(DEVICE_PROBE_EVIDENCE),
        "configured_before_articulation_build": dict(configured_before_build),
        "live_actuator": live_actuator,
        "live_physx_readback": {
            "velocity_limit_rad_s": tensor_evidence(
                select(_direct_physx_tensor(robot, "get_dof_max_velocities"))
            ),
            "effort_limit_nm": tensor_evidence(
                select(_direct_physx_tensor(robot, "get_dof_max_forces"))
            ),
            "stiffness_nm_per_rad": tensor_evidence(
                select(_direct_physx_tensor(robot, "get_dof_stiffnesses"))
            ),
            "damping_nm_s_per_rad": tensor_evidence(
                select(_direct_physx_tensor(robot, "get_dof_dampings"))
            ),
        },
        "legacy_velocity_limit_behavior": expectations["velocity_behavior"],
        "effort_limit_behavior": expectations["effort_behavior"],
        "incoming_joint_wrench_semantics": (
            "physx_total_incoming_joint_wrench_not_contact_force_child_joint_frame_v1"
        ),
        "computed_applied_torque_semantics": (
            "isaaclab_implicit_actuator_approximate_pd_preclip_and_"
            "effortlimit_clipped_v1"
        ),
    }
    return _validate_gripper_drive_contract(contract)


def _model_view_frame(observation: Mapping[str, Any]) -> Any:
    import numpy as np  # noqa: PLC0415
    from polaris.policy.lap_eef_pose_client import (  # noqa: PLC0415
        preprocess_lap_wrist_image,
        resize_lap_image,
    )

    try:
        external = observation["splat"]["external_cam"]
        wrist = observation["splat"]["wrist_cam"]
    except KeyError as error:
        raise GripperImpulseDiagnosticError(
            f"missing model-view camera observation: {error}"
        ) from error
    external_model = resize_lap_image(external)
    wrist_model = preprocess_lap_wrist_image(wrist, rotate_180=True)
    frame = np.concatenate([external_model, wrist_model], axis=1)
    _require(
        frame.shape == (VIDEO_HEIGHT, VIDEO_WIDTH, 3) and str(frame.dtype) == "uint8",
        f"model-view frame contract drift: shape={frame.shape}, dtype={frame.dtype}",
    )
    return frame


def _terminal_model_view_frame(env: Any) -> Any:
    """Render the trace-bound terminal state without advancing physics."""

    runtime = env.unwrapped
    runtime.sim.render()
    terminal_observation = {"splat": runtime.custom_render(False)}
    return _model_view_frame(terminal_observation)


def _run_live_diagnostic(args_cli: argparse.Namespace, state: dict[str, Any]):
    import gymnasium as gym  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: PLC0415

    import polaris.environments  # noqa: F401, PLC0415
    from polaris.eef_runtime_contract import (  # noqa: PLC0415
        begin_eef_safety_episode,
        eef_episode_safety_report,
        validate_eef_runtime_frame,
        validate_eef_runtime_safety,
        validate_ego_lap_runtime_protocol,
    )
    from polaris.environments.droid_cfg import (  # noqa: PLC0415
        BinaryJointPositionZeroToOneAction,
        EefPoseActionCfg,
    )
    from polaris.environments.robot_cfg import (  # noqa: PLC0415
        configure_eef_pose_joint_safety,
    )
    from polaris.robust_differential_ik import (  # noqa: PLC0415
        DifferentialIKInvariantError,
    )
    from polaris.utils import load_eval_initial_conditions  # noqa: PLC0415

    state["stage"] = "load_fixture"
    fixture_identity, source_actions = boundary.load_replay_fixture()
    action_plan, effective_actions = build_action_plan(
        source_actions, mode=args_cli.mode
    )
    helper_identity = _file_identity(BOUNDARY_HELPER_PATH)
    _require(
        helper_identity["size_bytes"] == EXPECTED_BOUNDARY_HELPER_SIZE_BYTES
        and helper_identity["sha256"] == EXPECTED_BOUNDARY_HELPER_SHA256,
        "boundary helper source drift",
    )

    state["stage"] = "build_environment"
    env_cfg = parse_env_cfg(
        ENVIRONMENT,
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )
    env_cfg.actions = EefPoseActionCfg()
    env_cfg.actions.arm.enable_failure_substep_trace = True
    env_cfg.actions.arm.enable_wrist_energy_brake = False
    diagnostic_gripper_class = _make_diagnostic_gripper_class(
        BinaryJointPositionZeroToOneAction
    )
    env_cfg.actions.finger_joint.class_type = diagnostic_gripper_class
    articulation_props = env_cfg.scene.robot.spawn.articulation_props
    _require(articulation_props is not None, "missing articulation properties")
    solver_velocity_before = articulation_props.solver_velocity_iteration_count
    configure_eef_pose_joint_safety(
        env_cfg.scene.robot,
        physx_cfg=env_cfg.sim.physx,
        enable_gripper_velocity_limit=(
            args_cli.enable_gripper_velocity_limit_candidate
        ),
    )
    gripper_cfg = env_cfg.scene.robot.actuators["gripper"]
    configured_gripper = {
        "legacy_velocity_limit_rad_s": _scalar_cfg_value(gripper_cfg.velocity_limit),
        "velocity_limit_sim_rad_s": _scalar_cfg_value(gripper_cfg.velocity_limit_sim),
        "legacy_effort_limit_nm": _scalar_cfg_value(gripper_cfg.effort_limit),
        "effort_limit_sim_nm": _scalar_cfg_value(gripper_cfg.effort_limit_sim),
        "stiffness": _scalar_cfg_value(gripper_cfg.stiffness),
        "damping": _scalar_cfg_value(gripper_cfg.damping),
    }
    solver_velocity_after = articulation_props.solver_velocity_iteration_count
    env = gym.make(ENVIRONMENT, cfg=env_cfg)
    state["env"] = env
    runtime_protocol = {
        **validate_ego_lap_runtime_protocol(env),
        "reset_seed": RESET_SEED,
        "initial_condition_index": INITIAL_CONDITION_INDEX,
    }

    state["stage"] = "validate_assets"
    assets = _capture_assets(
        scene_path=Path(env.unwrapped.usd_file),
        robot_usd_path=Path(env_cfg.scene.robot.spawn.usd_path),
    )
    _, initial_conditions = load_eval_initial_conditions(
        usd=env.unwrapped.usd_file,
        rollouts=1,
    )
    _require(
        isinstance(initial_conditions, list)
        and len(initial_conditions) == 1
        and isinstance(initial_conditions[INITIAL_CONDITION_INDEX], dict),
        "FoodBussing initial-condition loader drift",
    )
    state["stage"] = "reset_exact_initial_condition"
    observation, _ = env.reset(
        seed=RESET_SEED,
        object_positions=initial_conditions[INITIAL_CONDITION_INDEX],
        expensive=False,
    )
    runtime_frame = validate_eef_runtime_frame(env, observation)
    begin_eef_safety_episode(env, 0)
    initial_safety = validate_eef_runtime_safety(env)

    action_terms = env.unwrapped.action_manager._terms
    _require(
        list(action_terms) == ["arm", "finger_joint"],
        "ActionManager must apply arm before finger for causal trace",
    )
    arm_term = action_terms["arm"]
    finger_term = action_terms["finger_joint"]
    _require(
        isinstance(finger_term, diagnostic_gripper_class),
        "diagnostic finger ActionTerm was not installed",
    )
    _require(
        getattr(arm_term, "_failure_substep_trace_enabled", False) is True,
        "arm failure substep trace is not enabled",
    )
    robot = env.unwrapped.scene["robot"]
    gripper_drive_contract = _capture_gripper_drive_contract(
        robot,
        finger_term,
        configured_before_build=configured_gripper,
        candidate_enabled=args_cli.enable_gripper_velocity_limit_candidate,
    )
    solver_contract = {
        "profile": SOLVER_CHANGE_PROFILE,
        "configured_solver_velocity_iterations_before_eef_setup": (
            solver_velocity_before
        ),
        "configured_solver_velocity_iterations_after_eef_setup": (
            solver_velocity_after
        ),
        "live_solver_velocity_iterations": initial_safety[
            "solver_velocity_iteration_count"
        ],
        "live_solver_position_iterations": initial_safety[
            "solver_position_iteration_count"
        ],
        "live_physx_solver_type": initial_safety["physx_solver_type"],
    }
    _validate_solver_contract(solver_contract)

    frames: list[Any] = []
    outcome: dict[str, Any] | None = None
    arm_failure_runtime_evidence = None
    arm_substep_trace = None
    arm_safety = None
    allowed_failure_steps = ALLOWED_FAILURE_POLICY_STEPS[args_cli.mode]
    state["stage"] = "replay_gripper_impulse_boundary"
    for policy_step, effective_action in enumerate(effective_actions):
        state["policy_step"] = policy_step
        frames.append(_model_view_frame(observation))
        finger_term.begin_policy_step(
            policy_step=policy_step,
            original_gripper_closed_action=source_actions[policy_step][7],
            effective_gripper_closed_action=effective_action[7],
        )
        action_tensor = torch.tensor(
            effective_action,
            dtype=torch.float32,
            device=env.device,
        ).reshape(1, -1)
        try:
            observation, _, terminated, truncated, _ = env.step(
                action_tensor,
                expensive=False,
            )
        except DifferentialIKInvariantError as error:
            # Arm applies before finger. Finalize the prior successful finger
            # command from the live post-physics state, but stage no entry for
            # the arm's aborted apply.
            finger_term.finalize_pending(reason="arm_guard_exception")
            arm_failure_runtime_evidence = boundary._capture_failure_runtime_evidence(  # noqa: SLF001
                env,
                policy_step=policy_step,
            )
            arm_safety = arm_failure_runtime_evidence["ik_safety"]
            arm_substep_trace = arm_failure_runtime_evidence["controller_substep_trace"]
            apply_calls = arm_safety.get("counters", {}).get("apply_calls")
            _require(
                type(apply_calls) is int and apply_calls > 0,
                "velocity guard failure apply count",
            )
            failure_apply_index = apply_calls - 1
            failure_policy_step, failure_physics_substep = divmod(
                failure_apply_index, DECIMATION
            )
            _require(
                policy_step == failure_policy_step
                and failure_policy_step in allowed_failure_steps,
                "velocity guard failed outside the mode-specific allowed steps",
            )
            outcome = {
                "kind": "allowed_velocity_guard_failure",
                "mode": args_cli.mode,
                "reference_exact_failure_policy_step": (
                    REFERENCE_EXACT_FAILURE_POLICY_STEP
                ),
                "reference_exact_failure_physics_substep": (
                    REFERENCE_EXACT_FAILURE_PHYSICS_SUBSTEP
                ),
                "allowed_failure_policy_steps": list(allowed_failure_steps),
                "failure_policy_step": failure_policy_step,
                "failure_physics_substep": failure_physics_substep,
                "failure_apply_index": failure_apply_index,
                "last_attempted_policy_step": failure_policy_step,
                "completed_horizon_policy_step": None,
                "controller_failure": _exception_evidence(error),
                "causal_interpretation": _causal_interpretation(
                    args_cli.mode, failure_policy_step, failure_physics_substep
                ),
                "timing_classification": _timing_classification(
                    args_cli.mode, failure_policy_step, failure_physics_substep
                ),
            }
            break
        except BaseException:
            if finger_term.has_pending():
                finger_term.finalize_pending(reason="unexpected_exception")
            raise
        _require(
            not bool(terminated[0]) and not bool(truncated[0]),
            f"diagnostic episode ended unexpectedly at policy step {policy_step}",
        )
    if outcome is None:
        _require(
            state["policy_step"] == HORIZON_POLICY_STEP,
            "diagnostic did not reach its explicit horizon",
        )
        finger_term.finalize_pending(reason="diagnostic_horizon")
        finalize_arm_trace = getattr(
            arm_term, "_finalize_pending_failure_substep_trace", None
        )
        _require(callable(finalize_arm_trace), "arm trace has no horizon finalizer")
        finalize_arm_trace(
            post_joint_pos=robot.data.joint_pos[:, arm_term._joint_ids],
            post_joint_vel=robot.data.joint_vel[:, arm_term._joint_ids],
        )
        arm_substep_trace = arm_term.failure_substep_trace(0)
        arm_safety = eef_episode_safety_report(env, 0)
        outcome = {
            "kind": "diagnostic_horizon_reached",
            "mode": args_cli.mode,
            "reference_exact_failure_policy_step": (
                REFERENCE_EXACT_FAILURE_POLICY_STEP
            ),
            "reference_exact_failure_physics_substep": (
                REFERENCE_EXACT_FAILURE_PHYSICS_SUBSTEP
            ),
            "allowed_failure_policy_steps": list(allowed_failure_steps),
            "failure_policy_step": None,
            "failure_physics_substep": None,
            "failure_apply_index": None,
            "last_attempted_policy_step": HORIZON_POLICY_STEP,
            "completed_horizon_policy_step": HORIZON_POLICY_STEP,
            "controller_failure": None,
            "causal_interpretation": _causal_interpretation(args_cli.mode, None, None),
            "timing_classification": _timing_classification(args_cli.mode, None, None),
        }
    _require(arm_safety is not None, "missing terminal arm safety")
    _require(arm_substep_trace is not None, "missing terminal arm substep trace")
    frames.append(_terminal_model_view_frame(env))
    video_phase_contract = build_video_phase_contract(outcome)
    _require(
        len(frames) == video_phase_contract["total_frame_count"],
        "terminal video frame count",
    )
    finger_trace = finger_term.diagnostic_trace()
    if arm_failure_runtime_evidence is not None:
        terminal_snapshot = finger_trace["entries"][-1]["post_physics"]
        terminal_joint_names = terminal_snapshot["joint_names"]
        terminal_arm_indices = [
            terminal_joint_names.index(f"panda_joint{index}") for index in range(1, 8)
        ]
        arm_failure_runtime_evidence.update(
            {
                evidence_field: _select_tensor_evidence(
                    terminal_snapshot[snapshot_field],
                    terminal_arm_indices,
                    field=f"terminal {snapshot_field}",
                )
                for evidence_field, snapshot_field in (
                    (
                        "physx_arm_projected_joint_force_generalized_si",
                        "physx_projected_joint_force_nm",
                    ),
                    (
                        "physx_arm_stiffness_nm_per_rad",
                        "physx_joint_stiffness_nm_per_rad",
                    ),
                    (
                        "physx_arm_damping_nm_s_per_rad",
                        "physx_joint_damping_nm_s_per_rad",
                    ),
                )
            }
        )
    validate_finger_trace(
        finger_trace,
        action_plan=action_plan,
        outcome=outcome,
        arm_safety=arm_safety,
        gripper_drive=gripper_drive_contract,
    )
    _validate_arm_substep_trace_terminal(
        arm_substep_trace,
        outcome=outcome,
        arm_safety=arm_safety,
        finger_trace=finger_trace,
        failure_evidence=arm_failure_runtime_evidence,
    )
    _validate_outcome(outcome, mode=args_cli.mode)
    state["policy_step"] = None
    return {
        "fixture": fixture_identity,
        "boundary_helper_source": helper_identity,
        "assets": assets,
        "action_plan": action_plan,
        "runtime_protocol": runtime_protocol,
        "runtime_frame": runtime_frame,
        "solver_contract": solver_contract,
        "gripper_drive_contract": gripper_drive_contract,
        "outcome": outcome,
        "video_phase_contract": video_phase_contract,
        "finger_trace": finger_trace,
        "arm_safety": arm_safety,
        "arm_substep_trace": arm_substep_trace,
        "arm_failure_runtime_evidence": arm_failure_runtime_evidence,
    }, frames


def build_runtime_parser(
    add_app_launcher_args: Callable[[argparse.ArgumentParser], None] | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--output-ready-marker", type=Path, required=True)
    parser.add_argument("--runtime-exit", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument(
        "--enable-gripper-velocity-limit-candidate",
        action="store_true",
        help="Enable the isolated EEF-only 5 rad/s PhysX gripper canary.",
    )
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-source-size-bytes", type=int, required=True)
    if add_app_launcher_args is not None:
        add_app_launcher_args(parser)
    return parser


def _validate_runtime_output_paths(args_cli: argparse.Namespace) -> None:
    json_path = args_cli.output_json.resolve()
    video_path = args_cli.output_video.resolve()
    marker_path = args_cli.output_ready_marker.resolve()
    runtime_exit_path = args_cli.runtime_exit.resolve()
    _require(
        len({json_path, video_path, marker_path, runtime_exit_path}) == 4,
        "JSON, video, ready-marker, and runtime-exit paths must differ",
    )
    for field, path in (
        ("output JSON", json_path),
        ("output video", video_path),
        ("output ready marker", marker_path),
        ("runtime exit", runtime_exit_path),
    ):
        _require(not os.path.lexists(path), f"{field} already exists: {path}")


def _parse_runtime_args(argv: Sequence[str]) -> tuple[argparse.Namespace, Any]:
    from isaaclab.app import AppLauncher  # noqa: PLC0415

    parser = build_runtime_parser(AppLauncher.add_app_launcher_args)
    args_cli = parser.parse_args(list(argv))
    args_cli.enable_cameras = True
    args_cli.headless = True
    _validate_runtime_output_paths(args_cli)
    args_cli.diagnostic_source = capture_diagnostic_source(
        expected_sha256=args_cli.expected_source_sha256,
        expected_size_bytes=args_cli.expected_source_size_bytes,
    )
    args_cli.runtime_exit_contract = build_runtime_exit_contract(args_cli.runtime_exit)
    return args_cli, AppLauncher


def _publish_failure_capture(
    path: Path,
    *,
    mode: str,
    state: Mapping[str, Any],
    failure: BaseException,
    close_failures: Sequence[Mapping[str, Any]],
) -> None:
    payload = {
        "schema_version": 1,
        "diagnostic_profile": DIAGNOSTIC_PROFILE,
        "capture_valid": False,
        "stage": "failed",
        "exit_code": 1,
        "environment": ENVIRONMENT,
        "mode": mode,
        "policy_step": state.get("policy_step"),
        "failure": _exception_evidence(failure),
        "close_failures": list(close_failures),
    }
    identity = publish_immutable_json(path, payload)
    print(
        "POLARIS_GRIPPER_IMPULSE_INVALID_CAPTURE="
        f"{identity['path']};sha256={identity['sha256']};mode={mode}",
        flush=True,
    )


def _child_runtime_main(argv: Sequence[str]) -> None:
    _unblock_child_cleanup_signals()
    result_descriptor = _prepare_child_result_descriptor()
    os.environ.pop(CHILD_PROCESS_ENV, None)
    args_cli, app_launcher_type = _parse_runtime_args(argv)
    state: dict[str, Any] = {
        "stage": "launch_simulation_app",
        "policy_step": None,
        "env": None,
    }
    simulation_app = None
    failure: BaseException | None = None
    close_failures: list[dict[str, str]] = []
    evidence = None
    frames = None
    try:
        app_launcher = app_launcher_type(args_cli)
        simulation_app = app_launcher.app
        evidence, frames = _run_live_diagnostic(args_cli, state)
    except BaseException as error:
        failure = error
        traceback.print_exception(
            type(error), error, error.__traceback__, file=sys.stderr
        )

    env = state.get("env")
    if env is not None:
        state["stage"] = "close_environment"
        try:
            env.close()
        except BaseException as error:
            close_failures.append(_exception_evidence(error))
            if failure is None:
                failure = error
            traceback.print_exception(
                type(error), error, error.__traceback__, file=sys.stderr
            )

    successful_artifacts: tuple[dict[str, Any], dict[str, Any]] | None = None
    if (
        failure is None
        and evidence is not None
        and frames is not None
        and not close_failures
    ):
        try:
            state["stage"] = "publish_immutable_video"
            video_identity = publish_immutable_video(
                args_cli.output_video,
                frames,
                probe=_probe_video_stdlib,
            )
            state["stage"] = "simulation_app_close_pending"
            payload = {
                "schema_version": 1,
                "diagnostic_profile": DIAGNOSTIC_PROFILE,
                "fixture_profile": FIXTURE_PROFILE,
                "finalized": False,
                "capture_valid": True,
                "stage": state["stage"],
                "exit_code": 0,
                "environment": ENVIRONMENT,
                "mode": args_cli.mode,
                "diagnostic_source": args_cli.diagnostic_source,
                "runtime_exit_contract": args_cli.runtime_exit_contract,
                **evidence,
                "video": video_identity,
                "close_failures": [],
            }
            validate_capture_payload(
                payload,
                expected_mode=args_cli.mode,
                expected_gripper_drive_profile=_selected_gripper_drive_profile(
                    args_cli.enable_gripper_velocity_limit_candidate
                ),
            )
            json_identity = publish_immutable_json(args_cli.output_json, payload)
            ready_payload = {
                "schema_version": 1,
                "profile": READY_MARKER_PROFILE,
                "stage": "simulation_app_close_pending",
                "mode": args_cli.mode,
                "raw_result": json_identity,
                "video": video_identity,
                "diagnostic_source": args_cli.diagnostic_source,
                "runtime_exit_contract": args_cli.runtime_exit_contract,
            }
            validate_ready_marker(
                ready_payload,
                mode=args_cli.mode,
                raw_identity=json_identity,
                video_identity=video_identity,
                diagnostic_source=args_cli.diagnostic_source,
                runtime_exit_contract=args_cli.runtime_exit_contract,
            )
            _require(simulation_app is not None, "missing SimulationApp before close")
            successful_artifacts = (payload, ready_payload)
        except BaseException as error:
            failure = error
            traceback.print_exception(
                type(error), error, error.__traceback__, file=sys.stderr
            )

    if successful_artifacts is not None:
        payload, ready_payload = successful_artifacts
        ready_bytes = _strict_json_bytes(ready_payload)
        print(
            "POLARIS_GRIPPER_IMPULSE_CAPTURE_VALID="
            f"mode={args_cli.mode};outcome={payload['outcome']['kind']};"
            f"json={json_identity['path']};json_sha256={json_identity['sha256']};"
            f"video={video_identity['path']};video_sha256={video_identity['sha256']};"
            f"ready_pending={args_cli.output_ready_marker.resolve()};"
            f"ready_sha256={_sha256_bytes(ready_bytes)};"
            f"runtime_exit_pending={args_cli.runtime_exit.resolve()}",
            flush=True,
        )
        sys.stderr.flush()
        try:
            # The byte is only a pre-close intent.  The stdlib parent accepts
            # success only after normal wait status, process-group drain, pipe
            # EOF, immutable ready-marker validation, and artifact revalidation.
            _write_child_result_byte(0, result_descriptor)
            publish_immutable_json(args_cli.output_ready_marker, ready_payload)
            simulation_app.close()
        except BaseException:
            os._exit(1)
        os._exit(0)

    if failure is None:
        failure = RuntimeError("diagnostic ended without a closed capture")
    try:
        if not args_cli.output_json.exists():
            _publish_failure_capture(
                args_cli.output_json,
                mode=args_cli.mode,
                state=state,
                failure=failure,
                close_failures=close_failures,
            )
    except BaseException as persistence_error:
        traceback.print_exception(
            type(persistence_error),
            persistence_error,
            persistence_error.__traceback__,
            file=sys.stderr,
        )
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except BaseException:
        pass
    _write_child_result_byte(1, result_descriptor)
    if simulation_app is not None:
        try:
            simulation_app.close()
        except BaseException:
            os._exit(1)
    os._exit(1)


def _parse_parent_runtime_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = build_runtime_parser()
    args_cli, _ = parser.parse_known_args(list(argv))
    _validate_runtime_output_paths(args_cli)
    args_cli.diagnostic_source = capture_diagnostic_source(
        expected_sha256=args_cli.expected_source_sha256,
        expected_size_bytes=args_cli.expected_source_size_bytes,
    )
    args_cli.runtime_exit_contract = build_runtime_exit_contract(args_cli.runtime_exit)
    return args_cli


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(process_group_id: int, *, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while _process_group_exists(process_group_id):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


def _kill_process_group(process_group_id: int, signal_number: int) -> None:
    try:
        os.killpg(process_group_id, signal_number)
    except ProcessLookupError:
        pass


def _terminate_and_reap_kit_child(process: Any) -> None:
    process_group_id = process.pid
    _kill_process_group(process_group_id, signal.SIGTERM)
    if process.poll() is None:
        try:
            process.wait(timeout=CHILD_TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            pass
    if _process_group_exists(process_group_id):
        _kill_process_group(process_group_id, signal.SIGKILL)
    if process.poll() is None:
        try:
            process.wait(timeout=CHILD_GROUP_DRAIN_SECONDS)
        except subprocess.TimeoutExpired:
            _kill_process_group(process_group_id, signal.SIGKILL)
            process.wait(timeout=CHILD_GROUP_DRAIN_SECONDS)
    else:
        process.wait()
    _require(
        _wait_for_process_group_exit(
            process_group_id, timeout=CHILD_GROUP_DRAIN_SECONDS
        ),
        "Kit child process group did not drain after termination",
    )


def _reject_and_kill_surviving_process_group(process_group_id: int) -> None:
    if not _process_group_exists(process_group_id):
        return
    _kill_process_group(process_group_id, signal.SIGKILL)
    _require(
        _wait_for_process_group_exit(
            process_group_id, timeout=CHILD_GROUP_DRAIN_SECONDS
        ),
        "Kit child leader exited but its process group did not drain",
    )
    raise GripperImpulseDiagnosticError(
        "Kit child leader exited with surviving process-group members"
    )


def _read_exact_child_result_byte_and_eof(read_descriptor: int) -> bytes:
    os.set_blocking(read_descriptor, False)
    deadline = time.monotonic() + CHILD_PIPE_DRAIN_SECONDS
    payload = bytearray()
    while True:
        remaining = deadline - time.monotonic()
        _require(remaining > 0.0, "timed out waiting for child result pipe EOF")
        readable, _, _ = select.select([read_descriptor], [], [], remaining)
        _require(bool(readable), "timed out waiting for child result pipe EOF")
        try:
            chunk = os.read(read_descriptor, 2)
        except BlockingIOError:
            continue
        if chunk == b"":
            break
        payload.extend(chunk)
        _require(len(payload) <= 1, "child result pipe contained multiple bytes")
    result = bytes(payload)
    _require(result in {b"\x00", b"\x01"}, "child result pipe exact byte")
    return result


class _ParentSignalInterrupt(RuntimeError):
    pass


def _raise_parent_signal(signal_number: int, _frame: Any) -> None:
    raise _ParentSignalInterrupt(f"parent received signal {signal_number}")


def _install_parent_cleanup_signal_handlers() -> dict[int, Any]:
    previous: dict[int, Any] = {}
    for signal_number in PARENT_CLEANUP_SIGNALS:
        previous[signal_number] = signal.getsignal(signal_number)
        signal.signal(signal_number, _raise_parent_signal)
    return previous


def _restore_parent_signal_handlers(previous: Mapping[int, Any]) -> None:
    for signal_number, handler in previous.items():
        signal.signal(signal_number, handler)


def _ignore_parent_cleanup_signals() -> None:
    for signal_number in PARENT_CLEANUP_SIGNALS:
        signal.signal(signal_number, signal.SIG_IGN)


def _unblock_child_cleanup_signals() -> None:
    signal.pthread_sigmask(signal.SIG_UNBLOCK, PARENT_CLEANUP_SIGNALS)
    current_mask = signal.pthread_sigmask(signal.SIG_BLOCK, ())
    _require(
        not set(PARENT_CLEANUP_SIGNALS) & set(current_mask),
        "Kit child inherited blocked parent cleanup signals",
    )


def _run_kit_child(argv: Sequence[str]) -> tuple[bytes, int]:
    read_descriptor, write_descriptor = os.pipe()
    environment = os.environ.copy()
    environment[CHILD_PROCESS_ENV] = "1"
    environment[CHILD_RESULT_FD_ENV] = str(write_descriptor)
    process = None
    previous_signal_handlers = _install_parent_cleanup_signal_handlers()
    try:
        previous_signal_mask = signal.pthread_sigmask(
            signal.SIG_BLOCK, PARENT_CLEANUP_SIGNALS
        )
        try:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), *argv],
                env=environment,
                pass_fds=(write_descriptor,),
                start_new_session=True,
            )
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_signal_mask)
        os.close(write_descriptor)
        write_descriptor = -1
        process_return_code = process.wait(timeout=CHILD_TIMEOUT_SECONDS)
        _require(
            type(process_return_code) is int,
            "Kit child wait status must be an exact integer",
        )
        _reject_and_kill_surviving_process_group(process.pid)
        payload = _read_exact_child_result_byte_and_eof(read_descriptor)
        return payload, process_return_code
    except BaseException:
        _ignore_parent_cleanup_signals()
        if process is not None:
            _terminate_and_reap_kit_child(process)
        raise
    finally:
        _restore_parent_signal_handlers(previous_signal_handlers)
        if write_descriptor >= 0:
            try:
                os.close(write_descriptor)
            except BaseException:
                pass
        try:
            os.close(read_descriptor)
        except BaseException:
            pass


def _parent_runtime_main(argv: Sequence[str]) -> None:
    args_cli = _parse_parent_runtime_args(argv)
    exit_code = 1
    try:
        payload, process_return_code = _run_kit_child(argv)
        exit_code = _resolve_child_result(payload, process_return_code)
        if exit_code == 0:
            _host_revalidate_runtime_artifacts(args_cli)
        print(
            "POLARIS_GRIPPER_IMPULSE_CHILD_RESULT="
            f"payload={payload.hex()!r};process={process_return_code};"
            f"resolved={exit_code}",
            flush=True,
        )
    except BaseException as error:
        exit_code = 1
        traceback.print_exception(
            type(error), error, error.__traceback__, file=sys.stderr
        )
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except BaseException:
        exit_code = 1
    try:
        _publish_parent_exit_and_terminate(args_cli.runtime_exit, exit_code)
    except BaseException as error:
        traceback.print_exception(
            type(error), error, error.__traceback__, file=sys.stderr
        )
        try:
            sys.stderr.flush()
        except BaseException:
            pass
        os._exit(1)
    os._exit(1)


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--validate-capture" in argv:
        return _validation_main(argv)
    if os.environ.get(CHILD_PROCESS_ENV) == "1":
        _child_runtime_main(argv)
    else:
        _parent_runtime_main(argv)
    raise RuntimeError("runtime process returned after os._exit")


if __name__ == "__main__":
    raise SystemExit(main())
