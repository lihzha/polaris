"""Runtime assertions for the canonical Ego-LAP PolaRiS protocol."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from polaris.config import LAP_EEF_FRAME
from polaris.eef_ik_safety import ARM_VELOCITY_TARGET_PROFILE
from polaris.eef_ik_safety import CURRENT_JOINT_SOFT_LIMIT_TOLERANCE_RAD
from polaris.eef_ik_safety import EEF_IK_APPLY_CADENCE
from polaris.eef_ik_safety import EEF_IK_SAFETY_PROFILE
from polaris.eef_ik_safety import EEF_QUATERNION_UNIT_NORM_TOLERANCE
from polaris.eef_ik_safety import JOINT_SLEW_FLOAT32_TOLERANCE_RAD
from polaris.eef_ik_safety import JOINT_VELOCITY_LIMIT_TOLERANCE_RAD_S
from polaris.eef_ik_safety import PANDA_EEF_JOINT_EFFORT_LIMITS
from polaris.eef_ik_safety import PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S
from polaris.eef_ik_safety import PANDA_PHYSX_HARD_JOINT_POS_LIMITS_FLOAT32_SHA256
from polaris.eef_ik_safety import PANDA_SOFT_JOINT_POS_LIMITS_FLOAT32_SHA256
from polaris.eef_ik_safety import PANDA_SOFT_JOINT_POS_LIMITS_RAD
from polaris.eef_ik_safety import PANDA_TARGET_JOINT_POS_LIMITS_FLOAT32_SHA256
from polaris.eef_ik_safety import PHYSX_HARD_LIMIT_PROFILE
from polaris.eef_ik_safety import TARGET_SOFT_LIMIT_GUARD_BAND_PROFILE
from polaris.gripper_semantics import GRIPPER_THRESHOLD_PROFILE
from polaris.eval_artifacts import EVAL_RESULT_COLUMNS
from polaris.eval_artifacts import canonical_episode_result
from polaris.eval_artifacts import probe_episode_video
from polaris.eval_artifacts import validate_episode_artifact_identity


CANONICAL_EPISODE_STEPS = 450
CANONICAL_POLICY_HZ = 15.0
CANONICAL_PHYSICS_HZ = 120.0
CANONICAL_PHYSICS_DT = 1.0 / CANONICAL_PHYSICS_HZ
CANONICAL_DECIMATION = 8
CANONICAL_IK_METHOD = "dls"
CANONICAL_DLS_DAMPING = 0.01
CANONICAL_ARM_SCALE = 1.0
CANONICAL_ARM_JOINTS = tuple(f"panda_joint{index}" for index in range(1, 8))
SAFETY_COUNTER_FIELDS = {
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
SAFETY_MAXIMA_FIELDS = {
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
SAFETY_DIAGNOSTIC_FIELDS = {
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
SAFETY_DIAGNOSTIC_COUNTERS = {
    "current_joint_limit_abort": "current_joint_limit_aborts",
    "current_joint_velocity_limit_abort": "invariant_aborts",
    "post_clamp_position_invariant_abort": "invariant_aborts",
    "post_clamp_slew_invariant_abort": "invariant_aborts",
    "current_eef_quaternion_invariant_abort": "invariant_aborts",
    "desired_eef_quaternion_invariant_abort": "invariant_aborts",
    "nonfinite_abort": "nonfinite_aborts",
    "dls_pseudoinverse_fallback": "dls_fallbacks",
}
SAFETY_STATIC_FIELDS = (
    "profile",
    "apply_actions_cadence",
    "physics_dt",
    "control_dt",
    "decimation",
    "current_joint_soft_limit_tolerance_rad",
    "target_soft_limit_guard_band_profile",
    "physx_hard_limit_profile",
    "physx_hard_limit_write_count",
    "arm_velocity_target_profile",
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
    "arm_velocity_target_rad_s",
    "soft_joint_pos_limits_rad",
    "soft_joint_pos_limits_float32_sha256",
)
EPISODE_SAFETY_FIELDS = {
    "episode_index",
    *SAFETY_STATIC_FIELDS,
    "counters",
    "maxima",
    "guard_diagnostics",
    "max_raw_delta_diagnostic",
}
SAFETY_SIDECAR_FIELDS = {
    "schema_version",
    "transaction_state",
    "episode_index",
    "episode_result",
    "artifact_identity",
    "cadence_evidence",
    "safety",
}
ARTIFACT_IDENTITY_FIELDS = {"video", "terminal_trace"}
VIDEO_IDENTITY_FIELDS = {
    "filename",
    "size_bytes",
    "sha256",
    "frame_count",
    "height",
    "width",
}
TRACE_IDENTITY_FIELDS = {
    "filename",
    "size_bytes",
    "sha256",
    "episode_result",
}
RUNTIME_EPISODE_FIELDS = {
    "episode_index",
    "episode_result",
    "artifact_identity",
    "cadence_evidence",
    "counters",
    "maxima",
    "guard_diagnostics",
    "max_raw_delta_diagnostic",
    "sidecar_path",
    "sidecar_sha256",
}
AGGREGATE_SAFETY_FIELDS = {
    *SAFETY_STATIC_FIELDS,
    "episodes_completed",
    "counters",
    "maxima",
    "episodes",
}


def _unwrapped(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def validate_ego_lap_runtime_protocol(env: Any) -> dict[str, float | int]:
    """Fail unless the live simulator is exactly 450 policy steps at 15 Hz."""

    runtime = _unwrapped(env)
    horizon = int(getattr(env, "max_episode_length", runtime.max_episode_length))
    step_dt = getattr(runtime, "step_dt", None)
    if step_dt is None:
        cfg = runtime.cfg
        step_dt = float(cfg.sim.dt) * int(cfg.decimation)
    step_dt = float(step_dt)
    physics_dt = float(getattr(runtime, "physics_dt", runtime.cfg.sim.dt))
    decimation = int(getattr(runtime.cfg, "decimation", round(step_dt / physics_dt)))
    expected_dt = 1.0 / CANONICAL_POLICY_HZ
    if horizon != CANONICAL_EPISODE_STEPS:
        raise ValueError(
            "Canonical Ego-LAP/PolaRiS evaluation requires exactly "
            f"{CANONICAL_EPISODE_STEPS} policy steps; live environment has {horizon}"
        )
    if not math.isclose(step_dt, expected_dt, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(
            "Canonical Ego-LAP/PolaRiS evaluation requires 15 Hz control; "
            f"live step_dt={step_dt!r} ({1.0 / step_dt if step_dt > 0 else math.inf:g} Hz)"
        )
    if (
        not math.isclose(physics_dt, CANONICAL_PHYSICS_DT, rel_tol=0.0, abs_tol=1e-12)
        or decimation != CANONICAL_DECIMATION
    ):
        raise ValueError(
            "Canonical Ego-LAP/PolaRiS EEF safety requires apply_actions at "
            f"120 Hz with decimation 8; live physics_dt={physics_dt!r}, "
            f"decimation={decimation!r}"
        )
    if not math.isclose(physics_dt * decimation, step_dt, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "Live physics/control cadence is inconsistent: "
            f"physics_dt={physics_dt!r}, decimation={decimation!r}, "
            f"step_dt={step_dt!r}"
        )
    return {
        "episode_steps": horizon,
        "policy_hz": CANONICAL_POLICY_HZ,
        "step_dt": step_dt,
        "physics_hz": CANONICAL_PHYSICS_HZ,
        "physics_dt": physics_dt,
        "decimation": decimation,
    }


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _single_vector(value: Any, *, size: int, field: str) -> np.ndarray:
    array = _numpy(value).astype(np.float64)
    while array.ndim > 1 and array.shape[0] == 1:
        array = array[0]
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{field} must be one finite {size}-vector; got {array.shape}")
    return array


def _rotation_wxyz(value: Any, *, field: str) -> Rotation:
    quaternion = _single_vector(value, size=4, field=field)
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8:
        raise ValueError(f"{field} has near-zero norm")
    quaternion /= norm
    return Rotation.from_quat(quaternion[[1, 2, 3, 0]])


def _identity_offset(offset: Any) -> bool:
    if offset is None:
        return False
    position = tuple(float(value) for value in offset.pos)
    rotation = tuple(float(value) for value in offset.rot)
    return position == (0.0, 0.0, 0.0) and rotation == (1.0, 0.0, 0.0, 0.0)


def _arm_action_term(runtime: Any) -> Any:
    action_manager = getattr(runtime, "action_manager", None)
    terms = getattr(action_manager, "_terms", None)
    if not isinstance(terms, Mapping) or "arm" not in terms:
        raise ValueError("Live Ego-LAP environment has no installed arm action term")
    return terms["arm"]


def _finger_action_term(runtime: Any) -> Any:
    action_manager = getattr(runtime, "action_manager", None)
    terms = getattr(action_manager, "_terms", None)
    if not isinstance(terms, Mapping) or "finger_joint" not in terms:
        raise ValueError("Live Ego-LAP environment has no installed finger action term")
    return terms["finger_joint"]


def _validate_guard_diagnostic(
    diagnostic: Any,
    *,
    episode_index: int | None,
    field: str,
    allowed_kinds: set[str],
) -> None:
    """Require bounded diagnostics to remain strict-JSON finite-or-null."""

    if not isinstance(diagnostic, Mapping):
        raise ValueError(f"EEF IK {field} diagnostic is not an object")
    if set(diagnostic) != SAFETY_DIAGNOSTIC_FIELDS:
        raise ValueError(f"EEF IK {field} diagnostic schema drift")
    if diagnostic.get("kind") not in allowed_kinds:
        raise ValueError(
            f"EEF IK {field} diagnostic kind is not allowed: {diagnostic.get('kind')!r}"
        )
    if diagnostic.get("episode_index") != episode_index:
        raise ValueError(f"EEF IK {field} diagnostic episode identity drift")
    policy_step = diagnostic.get("policy_step")
    physics_substep = diagnostic.get("physics_substep")
    if type(policy_step) is not int or policy_step < 0:
        raise ValueError(f"EEF IK {field} diagnostic has invalid policy step")
    if (
        type(physics_substep) is not int
        or not 0 <= physics_substep < CANONICAL_DECIMATION
    ):
        raise ValueError(f"EEF IK {field} diagnostic has invalid physics substep")
    vector_fields = (
        "joint_pos_rad",
        "raw_delta_joint_pos_rad",
        "raw_joint_pos_target_rad",
        "safe_joint_pos_target_rad",
    )
    for vector_field in vector_fields:
        vector = diagnostic.get(vector_field)
        if vector is None:
            continue
        if not isinstance(vector, Mapping):
            raise ValueError(f"EEF IK diagnostic {vector_field} is invalid")
        if set(vector) != {"values", "finite_mask", "finite_count"}:
            raise ValueError(f"EEF IK diagnostic {vector_field} schema drift")
        values = vector.get("values")
        finite_mask = vector.get("finite_mask")
        finite_count = vector.get("finite_count")
        if (
            not isinstance(values, list)
            or len(values) != 7
            or not isinstance(finite_mask, list)
            or len(finite_mask) != 7
            or any(type(value) is not bool for value in finite_mask)
            or type(finite_count) is not int
            or finite_count != sum(finite_mask)
        ):
            raise ValueError(f"EEF IK diagnostic {vector_field} mask is invalid")
        for value, finite in zip(values, finite_mask, strict=True):
            if finite:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(
                        f"EEF IK diagnostic {vector_field} finite value is invalid"
                    )
            elif value is not None:
                raise ValueError(
                    f"EEF IK diagnostic {vector_field} nonfinite value must be null"
                )
    for scalar_field in (
        "pose_error_norm",
        "jacobian_max_abs",
        "eef_quaternion_norm",
    ):
        value = diagnostic.get(scalar_field)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"EEF IK diagnostic {scalar_field} is non-finite")
        if value is not None and value < 0:
            raise ValueError(f"EEF IK diagnostic {scalar_field} is negative")
    jacobian_finite = diagnostic.get("jacobian_finite")
    if jacobian_finite is not None and type(jacobian_finite) is not bool:
        raise ValueError("EEF IK diagnostic jacobian_finite must be bool or null")


def validate_eef_runtime_safety(env: Any) -> dict[str, Any]:
    """Validate and return cumulative live EEF IK safety evidence."""

    runtime = _unwrapped(env)
    arm_term = _arm_action_term(runtime)
    reporter = getattr(arm_term, "safety_report", None)
    if not callable(reporter):
        raise ValueError("Live Ego-LAP EEF action has no IK safety reporter")
    report = reporter()
    if not isinstance(report, dict):
        raise ValueError("Live Ego-LAP EEF IK safety reporter returned no object")
    if set(report) != EPISODE_SAFETY_FIELDS:
        raise ValueError(
            "Live EEF IK safety report schema drift: "
            f"expected={sorted(EPISODE_SAFETY_FIELDS)!r}, "
            f"actual={sorted(report)!r}"
        )
    exact_fields = {
        "profile": EEF_IK_SAFETY_PROFILE,
        "apply_actions_cadence": EEF_IK_APPLY_CADENCE,
        "target_soft_limit_guard_band_profile": TARGET_SOFT_LIMIT_GUARD_BAND_PROFILE,
        "physx_hard_limit_profile": PHYSX_HARD_LIMIT_PROFILE,
        "physx_hard_limit_write_count": 1,
        "arm_velocity_target_profile": ARM_VELOCITY_TARGET_PROFILE,
        "decimation": CANONICAL_DECIMATION,
        "joint_names": list(CANONICAL_ARM_JOINTS),
    }
    for field, expected in exact_fields.items():
        if report.get(field) != expected:
            raise ValueError(
                f"Live EEF IK safety {field} mismatch: "
                f"expected={expected!r}, actual={report.get(field)!r}"
            )
    physics_dt = float(report.get("physics_dt", math.nan))
    control_dt = float(report.get("control_dt", math.nan))
    current_limit_tolerance = float(
        report.get("current_joint_soft_limit_tolerance_rad", math.nan)
    )
    slew_tolerance = float(report.get("joint_slew_float32_tolerance_rad", math.nan))
    velocity_tolerance = float(
        report.get("joint_velocity_limit_tolerance_rad_s", math.nan)
    )
    quaternion_tolerance = float(
        report.get("eef_quaternion_unit_norm_tolerance", math.nan)
    )
    episode_index = report.get("episode_index")
    if episode_index is not None and (
        type(episode_index) is not int or episode_index < 0
    ):
        raise ValueError(
            f"Live EEF IK safety episode index is invalid: {episode_index!r}"
        )
    if report.get("soft_joint_pos_limit_factor") != 1.0:
        raise ValueError("Live EEF IK safety requires soft_joint_pos_limit_factor=1")
    if (
        not math.isclose(physics_dt, CANONICAL_PHYSICS_DT, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(
            control_dt, 1.0 / CANONICAL_POLICY_HZ, rel_tol=0.0, abs_tol=1e-12
        )
        or not math.isclose(
            current_limit_tolerance,
            CURRENT_JOINT_SOFT_LIMIT_TOLERANCE_RAD,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            slew_tolerance,
            JOINT_SLEW_FLOAT32_TOLERANCE_RAD,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            velocity_tolerance,
            JOINT_VELOCITY_LIMIT_TOLERANCE_RAD_S,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            quaternion_tolerance,
            EEF_QUATERNION_UNIT_NORM_TOLERANCE,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ValueError(
            "Live EEF IK safety cadence mismatch: "
            f"physics_dt={physics_dt!r}, control_dt={control_dt!r}, "
            f"current_limit_tolerance={current_limit_tolerance!r}, "
            f"slew_tolerance={slew_tolerance!r}, "
            f"velocity_tolerance={velocity_tolerance!r}, "
            f"quaternion_tolerance={quaternion_tolerance!r}"
        )

    vector_fields = {
        "joint_velocity_limits_rad_s": PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S,
        "joint_effort_limits": PANDA_EEF_JOINT_EFFORT_LIMITS,
        "max_delta_joint_pos_rad": tuple(
            value * CANONICAL_PHYSICS_DT
            for value in PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S
        ),
        "target_soft_limit_margin_rad": tuple(
            value * CANONICAL_PHYSICS_DT
            for value in PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S
        ),
    }
    for field, expected in vector_fields.items():
        actual = _numpy(report.get(field)).astype(np.float64)
        if (
            actual.shape != (7,)
            or not np.isfinite(actual).all()
            or not np.allclose(
                actual,
                np.asarray(expected),
                rtol=0.0,
                atol=JOINT_SLEW_FLOAT32_TOLERANCE_RAD,
            )
        ):
            raise ValueError(
                f"Live EEF IK safety {field} mismatch: "
                f"expected={expected!r}, actual={actual.tolist()!r}"
            )
    max_delta_report = report.get("max_delta_joint_pos_rad")
    target_margin_report = report.get("target_soft_limit_margin_rad")
    canonical_max_delta = np.asarray(
        PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S, dtype=np.float32
    ) * np.float32(CANONICAL_PHYSICS_DT)
    if target_margin_report != max_delta_report or not np.array_equal(
        np.asarray(max_delta_report, dtype=np.float32),
        canonical_max_delta,
    ):
        raise ValueError(
            "Live EEF IK target margin must exactly equal the canonical "
            "float32 physics-substep slew vector"
        )
    soft_limits = _numpy(report.get("soft_joint_pos_limits_rad")).astype(np.float64)
    if (
        soft_limits.shape != (7, 2)
        or not np.isfinite(soft_limits).all()
        or not np.all(soft_limits[:, 0] < soft_limits[:, 1])
    ):
        raise ValueError(
            "Live EEF IK safety has invalid soft joint position limits: "
            f"{soft_limits.tolist()!r}"
        )
    soft_limit_sha256 = report.get("soft_joint_pos_limits_float32_sha256")
    computed_soft_limit_sha256 = hashlib.sha256(
        soft_limits.astype("<f4", copy=False).tobytes()
    ).hexdigest()
    expected_soft_limits = np.asarray(PANDA_SOFT_JOINT_POS_LIMITS_RAD, dtype="<f4")
    if not np.array_equal(soft_limits.astype("<f4", copy=False), expected_soft_limits):
        raise ValueError(
            "Live EEF IK safety soft limits do not match the canonical Panda "
            f"float32 values: expected={expected_soft_limits.tolist()!r}, "
            f"actual={soft_limits.tolist()!r}"
        )
    if (
        soft_limit_sha256 != computed_soft_limit_sha256
        or soft_limit_sha256 != PANDA_SOFT_JOINT_POS_LIMITS_FLOAT32_SHA256
    ):
        raise ValueError(
            "Live EEF IK safety soft-limit digest mismatch: "
            f"expected={PANDA_SOFT_JOINT_POS_LIMITS_FLOAT32_SHA256!r}, "
            f"recorded={soft_limit_sha256!r}, computed={computed_soft_limit_sha256!r}"
        )

    target_limits = _numpy(report.get("target_joint_pos_limits_rad")).astype(np.float64)
    max_delta_float32 = np.asarray(max_delta_report, dtype=np.float32)
    expected_target_limits = np.stack(
        (
            soft_limits.astype(np.float32)[:, 0] + max_delta_float32,
            soft_limits.astype(np.float32)[:, 1] - max_delta_float32,
        ),
        axis=-1,
    )
    if (
        target_limits.shape != (7, 2)
        or not np.isfinite(target_limits).all()
        or not np.array_equal(target_limits.astype(np.float32), expected_target_limits)
        or not np.all(target_limits[:, 0] < target_limits[:, 1])
    ):
        raise ValueError(
            "Live EEF IK target guard-band limits do not match one physics "
            "substep of the velocity bounds: "
            f"expected={expected_target_limits.tolist()!r}, "
            f"actual={target_limits.tolist()!r}"
        )
    target_limit_sha256 = report.get("target_joint_pos_limits_float32_sha256")
    computed_target_limit_sha256 = hashlib.sha256(
        target_limits.astype("<f4", copy=False).tobytes()
    ).hexdigest()
    expected_target_limit_sha256 = hashlib.sha256(
        expected_target_limits.astype("<f4", copy=False).tobytes()
    ).hexdigest()
    if (
        target_limit_sha256 != computed_target_limit_sha256
        or target_limit_sha256 != expected_target_limit_sha256
        or target_limit_sha256 != PANDA_TARGET_JOINT_POS_LIMITS_FLOAT32_SHA256
    ):
        raise ValueError(
            "Live EEF IK target guard-band digest mismatch: "
            f"expected={expected_target_limit_sha256!r}, "
            "canonical="
            f"{PANDA_TARGET_JOINT_POS_LIMITS_FLOAT32_SHA256!r}, "
            f"recorded={target_limit_sha256!r}, "
            f"computed={computed_target_limit_sha256!r}"
        )
    physx_hard_limits = _numpy(report.get("physx_hard_joint_pos_limits_rad")).astype(
        np.float64
    )
    physx_hard_limit_sha256 = report.get("physx_hard_joint_pos_limits_float32_sha256")
    computed_physx_hard_limit_sha256 = hashlib.sha256(
        physx_hard_limits.astype("<f4", copy=False).tobytes()
    ).hexdigest()
    if (
        physx_hard_limits.shape != (7, 2)
        or not np.array_equal(
            physx_hard_limits.astype(np.float32),
            expected_target_limits,
        )
        or physx_hard_limit_sha256 != computed_physx_hard_limit_sha256
        or physx_hard_limit_sha256 != PANDA_PHYSX_HARD_JOINT_POS_LIMITS_FLOAT32_SHA256
    ):
        raise ValueError(
            "Live EEF IK PhysX hard-limit readback does not match the exact "
            "target guard-band envelope"
        )
    arm_velocity_target = _numpy(report.get("arm_velocity_target_rad_s")).astype(
        np.float64
    )
    if arm_velocity_target.shape != (7,) or not np.array_equal(
        arm_velocity_target,
        np.zeros(7, dtype=np.float64),
    ):
        raise ValueError("Live EEF IK arm velocity target must be exactly zero")
    counters = report.get("counters")
    maxima = report.get("maxima")
    if not isinstance(counters, dict) or not isinstance(maxima, dict):
        raise ValueError("Live EEF IK safety requires counters and maxima objects")
    expected_counters = SAFETY_COUNTER_FIELDS
    if set(counters) != expected_counters or any(
        type(counters[field]) is not int or counters[field] < 0
        for field in expected_counters
    ):
        raise ValueError(f"Live EEF IK safety counters are invalid: {counters!r}")
    if counters["environment_substeps"] != counters["apply_calls"]:
        raise ValueError(
            "Single-environment EEF safety substep count must equal apply calls"
        )
    expected_maxima = SAFETY_MAXIMA_FIELDS
    if set(maxima) != expected_maxima:
        raise ValueError(f"Live EEF IK safety maxima are invalid: {maxima!r}")
    for field in expected_maxima:
        values = _numpy(maxima[field]).astype(np.float64)
        if (
            values.shape != (7,)
            or not np.isfinite(values).all()
            or (field != "minimum_outer_joint_clearance_rad" and np.any(values < 0.0))
        ):
            raise ValueError(
                f"Live EEF IK safety maximum {field} is invalid: {values.tolist()!r}"
            )
    applied = np.asarray(maxima["applied_delta_joint_pos_rad"], dtype=np.float64)
    max_delta = np.asarray(report["max_delta_joint_pos_rad"], dtype=np.float64)
    if np.any(applied > max_delta + JOINT_SLEW_FLOAT32_TOLERANCE_RAD):
        raise ValueError(
            "Live EEF IK applied joint delta exceeds its physics-substep bound: "
            f"applied={applied.tolist()!r}, bound={max_delta.tolist()!r}"
        )
    max_abs_velocity = np.asarray(maxima["abs_joint_vel_rad_s"], dtype=np.float64)
    velocity_limits = np.asarray(
        report["joint_velocity_limits_rad_s"], dtype=np.float64
    )
    if np.any(
        max_abs_velocity > velocity_limits + JOINT_VELOCITY_LIMIT_TOLERANCE_RAD_S
    ):
        raise ValueError("Live EEF IK joint velocity exceeds its configured bound")
    hard_violation = np.asarray(
        maxima["current_physx_hard_limit_violation_rad"], dtype=np.float64
    )
    if np.any(hard_violation > max_delta + CURRENT_JOINT_SOFT_LIMIT_TOLERANCE_RAD):
        raise ValueError(
            "Live EEF IK PhysX hard-limit slop consumed the canonical outer envelope"
        )
    outer_violation = np.asarray(
        maxima["current_joint_soft_limit_violation_rad"], dtype=np.float64
    )
    minimum_outer_clearance = np.asarray(
        maxima["minimum_outer_joint_clearance_rad"], dtype=np.float64
    )
    if not np.allclose(
        outer_violation,
        np.maximum(-minimum_outer_clearance, 0.0),
        rtol=0.0,
        atol=JOINT_SLEW_FLOAT32_TOLERANCE_RAD,
    ):
        raise ValueError("Live EEF IK outer violation/clearance evidence disagrees")
    guard_diagnostics = report.get("guard_diagnostics")
    if not isinstance(guard_diagnostics, list) or len(guard_diagnostics) > 32:
        raise ValueError("Live EEF IK safety guard diagnostics are not bounded")
    if counters["guard_diagnostics_dropped"] != 0:
        raise ValueError("Live EEF IK safety dropped durable guard diagnostics")
    mapped_counts = {
        "current_joint_limit_aborts": 0,
        "invariant_aborts": 0,
        "nonfinite_aborts": 0,
        "dls_fallbacks": 0,
    }
    for diagnostic in guard_diagnostics:
        _validate_guard_diagnostic(
            diagnostic,
            episode_index=episode_index,
            field="guard",
            allowed_kinds=set(SAFETY_DIAGNOSTIC_COUNTERS),
        )
        mapped_counts[SAFETY_DIAGNOSTIC_COUNTERS[diagnostic["kind"]]] += 1
    for counter, diagnostic_count in mapped_counts.items():
        if counters[counter] != diagnostic_count:
            raise ValueError(
                "Live EEF IK safety counter/diagnostic mapping drift for "
                f"{counter}: counter={counters[counter]}, "
                f"diagnostics={diagnostic_count}"
            )
    max_raw_diagnostic = report.get("max_raw_delta_diagnostic")
    if max_raw_diagnostic is not None and not isinstance(max_raw_diagnostic, dict):
        raise ValueError("Live EEF IK safety max-raw-delta diagnostic is invalid")
    if max_raw_diagnostic is not None:
        _validate_guard_diagnostic(
            max_raw_diagnostic,
            episode_index=episode_index,
            field="max-raw-delta",
            allowed_kinds={"max_raw_delta"},
        )
    return report


def validate_eef_runtime_frame(
    env: Any,
    observation: Mapping[str, Any],
    *,
    position_tolerance: float = 1e-5,
    rotation_tolerance_radians: float = math.radians(0.01),
) -> dict[str, bool | float | int | str]:
    """Verify observed and controlled Cartesian frames on the live articulation."""

    runtime = _unwrapped(env)
    robot = runtime.scene["robot"]
    body_names = list(robot.data.body_names)
    try:
        link0_index = body_names.index("panda_link0")
        link8_index = body_names.index(LAP_EEF_FRAME)
    except ValueError as error:
        raise ValueError(
            "Live DROID articulation is missing panda_link0 or panda_link8"
        ) from error

    body_positions = _numpy(robot.data.body_pos_w)
    body_quaternions = _numpy(robot.data.body_quat_w)
    if body_positions.ndim != 3 or body_positions.shape[0] != 1:
        raise ValueError(
            f"Ego-LAP runtime requires one articulation environment; got {body_positions.shape}"
        )
    link0_position = _single_vector(
        body_positions[0, link0_index], size=3, field="panda_link0 position"
    )
    link8_position = _single_vector(
        body_positions[0, link8_index], size=3, field="panda_link8 position"
    )
    link0_rotation = _rotation_wxyz(
        body_quaternions[0, link0_index], field="panda_link0 quaternion"
    )
    link8_rotation = _rotation_wxyz(
        body_quaternions[0, link8_index], field="panda_link8 quaternion"
    )
    direct_position = link0_rotation.inv().apply(link8_position - link0_position)
    direct_rotation = link0_rotation.inv() * link8_rotation

    try:
        policy_observation = observation["policy"]
        observed_position = _single_vector(
            policy_observation["eef_pos"], size=3, field="observed EEF position"
        )
        observed_rotation = _rotation_wxyz(
            policy_observation["eef_quat"], field="observed EEF quaternion"
        )
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Live observation is missing the Ego-LAP EEF state: {error}"
        ) from error

    position_error = float(np.linalg.norm(observed_position - direct_position))
    rotation_error = float((direct_rotation.inv() * observed_rotation).magnitude())
    if (
        position_error > position_tolerance
        or rotation_error > rotation_tolerance_radians
    ):
        raise ValueError(
            "Live Ego-LAP observation is not the direct panda_link0->panda_link8 pose: "
            f"position_error={position_error:g}, rotation_error={rotation_error:g}"
        )

    arm_term = _arm_action_term(runtime)
    arm_cfg = getattr(arm_term, "cfg", None)
    if arm_cfg is None or getattr(arm_cfg, "body_name", None) != LAP_EEF_FRAME:
        raise ValueError(
            "Live Ego-LAP controller does not control physical panda_link8: "
            f"{getattr(arm_cfg, 'body_name', None)!r}"
        )
    if not _identity_offset(getattr(arm_cfg, "body_offset", None)):
        raise ValueError("Live Ego-LAP controller body offset is not identity")
    controller_cfg = getattr(arm_cfg, "controller", None)
    if (
        controller_cfg is None
        or getattr(controller_cfg, "command_type", None) != "pose"
        or bool(getattr(controller_cfg, "use_relative_mode", True))
    ):
        raise ValueError("Live Ego-LAP controller is not absolute pose differential IK")
    ik_method = getattr(controller_cfg, "ik_method", None)
    if ik_method != CANONICAL_IK_METHOD:
        raise ValueError(
            "Live Ego-LAP controller must use damped least-squares IK; "
            f"got {ik_method!r}"
        )
    ik_params = getattr(controller_cfg, "ik_params", None)
    damping = ik_params.get("lambda_val") if isinstance(ik_params, Mapping) else None
    if damping is None or not math.isclose(
        float(damping), CANONICAL_DLS_DAMPING, rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError(
            "Live Ego-LAP DLS damping must be exactly "
            f"{CANONICAL_DLS_DAMPING}; got {damping!r}"
        )
    arm_scale = getattr(arm_cfg, "scale", None)
    if (
        not isinstance(arm_scale, (int, float))
        or float(arm_scale) != CANONICAL_ARM_SCALE
    ):
        raise ValueError(
            f"Live Ego-LAP arm action scale must be {CANONICAL_ARM_SCALE}; got {arm_scale!r}"
        )
    resolved_joint_names = tuple(getattr(arm_term, "_joint_names", ()))
    if resolved_joint_names != CANONICAL_ARM_JOINTS:
        raise ValueError(
            "Live Ego-LAP controller joint order must be panda_joint1..panda_joint7; "
            f"got {resolved_joint_names!r}"
        )
    action_dim = getattr(arm_term, "action_dim", 7)
    if int(action_dim) != 7:
        raise ValueError(
            f"Live Ego-LAP arm action dimension must be 7; got {action_dim!r}"
        )
    body_index = getattr(arm_term, "_body_idx", None)
    if body_index is not None:
        body_index_array = np.asarray(body_index).reshape(-1)
        if body_index_array.size != 1 or int(body_index_array[0]) != link8_index:
            raise ValueError(
                "Live Ego-LAP controller resolved a body index other than panda_link8: "
                f"{body_index!r}"
            )
    finger_term = _finger_action_term(runtime)
    gripper_threshold_profile = getattr(finger_term, "gripper_threshold_profile", None)
    if gripper_threshold_profile != GRIPPER_THRESHOLD_PROFILE:
        raise ValueError(
            "Live Ego-LAP gripper threshold semantics do not match training: "
            f"got {gripper_threshold_profile!r}"
        )

    return {
        "eef_frame": LAP_EEF_FRAME,
        "reference_frame": "panda_link0",
        "position_error_m": position_error,
        "rotation_error_rad": rotation_error,
        "controlled_body": LAP_EEF_FRAME,
        "body_offset": "identity",
        "command_type": "pose",
        "use_relative_mode": False,
        "ik_method": CANONICAL_IK_METHOD,
        "dls_damping": CANONICAL_DLS_DAMPING,
        "arm_scale": CANONICAL_ARM_SCALE,
        "arm_joint_names": list(CANONICAL_ARM_JOINTS),
        "gripper_threshold_profile": GRIPPER_THRESHOLD_PROFILE,
        "ik_safety_profile": EEF_IK_SAFETY_PROFILE,
        "action_dim": 7,
    }


def begin_eef_safety_episode(env: Any, episode_index: int) -> None:
    """Reset the live action-term counters for one rollout."""

    arm_term = _arm_action_term(_unwrapped(env))
    begin = getattr(arm_term, "begin_safety_episode", None)
    if not callable(begin):
        raise ValueError("Live Ego-LAP EEF action cannot begin safety accounting")
    begin(episode_index)


def eef_episode_safety_report(env: Any, episode_index: int) -> dict[str, Any]:
    """Return one completed rollout's live action-term safety report."""

    arm_term = _arm_action_term(_unwrapped(env))
    reporter = getattr(arm_term, "episode_safety_report", None)
    if not callable(reporter):
        raise ValueError("Live Ego-LAP EEF action has no episode safety reporter")
    report = reporter(episode_index)
    validate_eef_runtime_safety(env)
    return report


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _load_strict_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(), parse_constant=_reject_json_constant)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preserve_orphan(path: Path, *, archive_directory: Path) -> Path:
    """Move an uncommitted artifact to a content-addressed evidence archive."""

    digest = _sha256(path)
    archive_directory.mkdir(parents=True, exist_ok=True)
    safe_name = path.name.replace(os.sep, "_")
    destination = archive_directory / f"{safe_name}.sha256-{digest}"
    duplicate = 0
    while destination.exists():
        duplicate += 1
        destination = archive_directory / (
            f"{safe_name}.sha256-{digest}.duplicate-{duplicate}"
        )
    os.replace(path, destination)
    return destination


def _preserve_uncommitted_episode_artifacts(
    *,
    run_folder: Path,
    trace_dir: Path,
    committed_count: int,
    prepared_episode: int | None,
) -> list[Path]:
    """Archive rollout artifacts that have no recoverable prepared sidecar."""

    candidates: list[tuple[int, Path, bool]] = []
    for path in run_folder.glob("episode_*.mp4"):
        suffix = path.stem.removeprefix("episode_")
        if suffix.isdigit():
            candidates.append((int(suffix), path, True))
    for path in trace_dir.glob("episode_*.jsonl") if trace_dir.exists() else ():
        suffix = path.stem.removeprefix("episode_")
        if suffix.isdigit():
            candidates.append((int(suffix), path, True))
    # Hard termination can leave stable hidden temporary files. They never
    # authorize a row, but they are still evidence and must not be deleted.
    for path in (
        *run_folder.glob(".episode_*.tmp.mp4"),
        *(trace_dir.glob(".episode_*.jsonl.tmp") if trace_dir.exists() else ()),
        *(run_folder / "ik_safety").glob(".episode_*.json.*.tmp"),
        *run_folder.glob(".eval_results.tmp.csv"),
        *run_folder.glob(".polaris_runtime_contract.json.*.tmp"),
    ):
        candidates.append((committed_count, path, False))

    preserved: list[Path] = []
    for episode_index, path, protected_by_prepared_sidecar in candidates:
        if episode_index < committed_count or (
            protected_by_prepared_sidecar and episode_index == prepared_episode
        ):
            continue
        preserved.append(
            _preserve_orphan(
                path,
                archive_directory=(
                    run_folder / "recovery_orphans" / f"episode_{episode_index:06d}"
                ),
            )
        )
    return preserved


def _validate_artifact_identity_schema(identity: Mapping[str, Any]) -> None:
    if set(identity) != ARTIFACT_IDENTITY_FIELDS:
        raise ValueError("Episode artifact identity schema drift")
    video = identity.get("video")
    trace = identity.get("terminal_trace")
    if not isinstance(video, Mapping) or set(video) != VIDEO_IDENTITY_FIELDS:
        raise ValueError("Episode video identity schema drift")
    if not isinstance(trace, Mapping) or set(trace) != TRACE_IDENTITY_FIELDS:
        raise ValueError("Episode terminal-trace identity schema drift")
    canonical_episode_result(trace.get("episode_result", {}))


def atomic_write_episode_safety(
    path: Path,
    *,
    episode_index: int,
    episode_result: Mapping[str, Any],
    safety: Mapping[str, Any],
    artifact_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically persist an immutable, CSV-recoverable episode transaction."""

    result = canonical_episode_result(episode_result)
    if result["episode"] != episode_index:
        raise ValueError(
            "Episode transaction result index mismatch: "
            f"expected={episode_index}, actual={result['episode']!r}"
        )
    if safety.get("episode_index") != episode_index:
        raise ValueError(
            "Episode safety index mismatch: "
            f"expected={episode_index}, actual={safety.get('episode_index')!r}"
        )
    _validate_artifact_identity_schema(artifact_identity)
    cadence_evidence = validate_episode_safety_cadence(
        safety=safety,
        episode_result=result,
    )
    payload = {
        "schema_version": 2,
        "transaction_state": "prepared",
        "episode_index": episode_index,
        "episode_result": result,
        "artifact_identity": dict(artifact_identity),
        "cadence_evidence": cadence_evidence,
        "safety": dict(safety),
    }
    if path.exists():
        existing = _load_strict_json(path)
        if existing != payload:
            raise ValueError(
                f"Refusing to overwrite drifted episode safety sidecar: {path}"
            )
        return existing
    _atomic_write_json(path, payload)
    return payload


def _validate_episode_safety_evidence_shape(
    safety: Mapping[str, Any], *, episode_index: int
) -> None:
    """Validate the exact durable per-episode safety schema and counter mapping."""

    if set(safety) != EPISODE_SAFETY_FIELDS:
        raise ValueError("Episode safety report schema drift")
    counters = safety.get("counters")
    maxima = safety.get("maxima")
    if not isinstance(counters, Mapping) or set(counters) != SAFETY_COUNTER_FIELDS:
        raise ValueError("Episode safety counter schema drift")
    if any(type(value) is not int or value < 0 for value in counters.values()):
        raise ValueError("Episode safety counters must be nonnegative integers")
    if counters["environment_substeps"] != counters["apply_calls"]:
        raise ValueError("Episode safety environment/apply substeps disagree")
    apply_calls = counters["apply_calls"]
    for event_name, joint_name in (
        ("slew_limit_events", "slew_limited_joints"),
        ("position_limit_events", "position_limited_joints"),
    ):
        events = counters[event_name]
        joints = counters[joint_name]
        if events > apply_calls or not events <= joints <= 7 * apply_calls:
            raise ValueError(
                f"Episode safety {event_name}/{joint_name} history is impossible"
            )
    if (
        counters["post_clamp_target_violations"] > apply_calls
        or counters["dls_fallbacks"] > apply_calls
        or any(
            counters[name] > 1
            for name in (
                "current_joint_limit_aborts",
                "invariant_aborts",
                "nonfinite_aborts",
            )
        )
    ):
        raise ValueError("Episode safety counter history is impossible")
    if counters["guard_diagnostics_dropped"] != 0:
        raise ValueError("Episode safety dropped durable guard diagnostics")
    if not isinstance(maxima, Mapping) or set(maxima) != SAFETY_MAXIMA_FIELDS:
        raise ValueError("Episode safety maxima schema drift")
    for name, vector in maxima.items():
        if (
            not isinstance(vector, list)
            or len(vector) != 7
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
                for value in vector
            )
        ):
            raise ValueError(f"Episode safety maximum {name} is invalid")
    diagnostics = safety.get("guard_diagnostics")
    if not isinstance(diagnostics, list) or len(diagnostics) > 32:
        raise ValueError("Episode safety guard diagnostics are invalid")
    mapped_counts = {
        "current_joint_limit_aborts": 0,
        "invariant_aborts": 0,
        "nonfinite_aborts": 0,
        "dls_fallbacks": 0,
    }
    diagnostic_indices = []
    for diagnostic in diagnostics:
        _validate_guard_diagnostic(
            diagnostic,
            episode_index=episode_index,
            field="guard",
            allowed_kinds=set(SAFETY_DIAGNOSTIC_COUNTERS),
        )
        flattened_index = (
            diagnostic["policy_step"] * CANONICAL_DECIMATION
            + diagnostic["physics_substep"]
        )
        if flattened_index >= apply_calls:
            raise ValueError("Episode safety guard diagnostic is out of cadence")
        diagnostic_indices.append(flattened_index)
        mapped_counts[SAFETY_DIAGNOSTIC_COUNTERS[diagnostic["kind"]]] += 1
    if diagnostic_indices != sorted(diagnostic_indices):
        raise ValueError("Episode safety guard diagnostics are out of order")
    for name, count in mapped_counts.items():
        if counters[name] != count:
            raise ValueError(
                "Episode safety counter/diagnostic mapping drift for "
                f"{name}: counter={counters[name]}, diagnostics={count}"
            )
    max_raw = safety.get("max_raw_delta_diagnostic")
    if max_raw is not None:
        _validate_guard_diagnostic(
            max_raw,
            episode_index=episode_index,
            field="max-raw-delta",
            allowed_kinds={"max_raw_delta"},
        )


def validate_episode_safety_cadence(
    *,
    safety: Mapping[str, Any],
    episode_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-check one rollout result against exact controller substep counts."""

    result = canonical_episode_result(episode_result)
    if safety.get("episode_index") != result["episode"]:
        raise ValueError(
            "Episode safety/result identity mismatch: "
            f"safety={safety.get('episode_index')!r}, result={result['episode']!r}"
        )
    _validate_episode_safety_evidence_shape(safety, episode_index=result["episode"])
    counters = safety.get("counters")
    if not isinstance(counters, Mapping):
        raise ValueError("Episode safety cadence requires a counters object")
    required_counters = {
        "apply_calls",
        "environment_substeps",
        "current_joint_limit_aborts",
        "invariant_aborts",
        "nonfinite_aborts",
    }
    if not required_counters.issubset(counters) or any(
        type(counters[field]) is not int or counters[field] < 0
        for field in required_counters
    ):
        raise ValueError(f"Episode safety cadence counters are invalid: {counters!r}")
    apply_calls = counters["apply_calls"]
    if counters["environment_substeps"] != apply_calls:
        raise ValueError(
            "Single-environment episode substeps must equal controller apply calls"
        )
    episode_length = result["episode_length"]
    upper = episode_length * CANONICAL_DECIMATION
    abort_count = sum(
        counters[field]
        for field in (
            "current_joint_limit_aborts",
            "invariant_aborts",
            "nonfinite_aborts",
        )
    )
    numerical_failure = result["numerical_failure"]
    if abort_count and not numerical_failure:
        raise ValueError(
            "Controller abort counters require numerical_failure=true: "
            f"abort_count={abort_count}"
        )
    if numerical_failure:
        if abort_count != 1:
            raise ValueError(
                "Numerical failure must have exactly one terminal controller abort"
            )
        lower = (episode_length - 1) * CANONICAL_DECIMATION
        if not lower < apply_calls <= upper:
            raise ValueError(
                "Numerical-failure controller cadence mismatch: "
                f"required={lower}<apply_calls<={upper}, actual={apply_calls}"
            )
        failed_policy_step = episode_length - 1
        failed_physics_substep = (apply_calls - 1) % CANONICAL_DECIMATION
        diagnostics = safety.get("guard_diagnostics")
        if not isinstance(diagnostics, list):
            raise ValueError("Numerical failure has no guard diagnostics list")
        matching_abort = [
            diagnostic
            for diagnostic in diagnostics
            if isinstance(diagnostic, Mapping)
            and str(diagnostic.get("kind", "")).endswith("abort")
            and diagnostic.get("policy_step") == failed_policy_step
            and diagnostic.get("physics_substep") == failed_physics_substep
        ]
        if not matching_abort:
            raise ValueError(
                "Numerical failure lacks an exact controller abort substep diagnostic: "
                f"policy_step={failed_policy_step}, "
                f"physics_substep={failed_physics_substep}"
            )
    else:
        if apply_calls != upper:
            raise ValueError(
                "Completed episode controller cadence mismatch: "
                f"expected={upper}, actual={apply_calls}"
            )
        if abort_count != 0:
            raise ValueError("Completed episode has controller abort counters")
        diagnostics = safety["guard_diagnostics"]
        if any(str(item["kind"]).endswith("abort") for item in diagnostics):
            raise ValueError("Completed episode has an abort diagnostic")
        if counters["post_clamp_target_violations"] != 0:
            raise ValueError("Completed episode has a post-clamp target violation")
        if any(
            value != 0.0
            for value in safety["maxima"]["post_clamp_target_soft_limit_violation_rad"]
        ):
            raise ValueError("Completed episode has post-clamp violation maxima")
        if any(
            value > CURRENT_JOINT_SOFT_LIMIT_TOLERANCE_RAD
            for value in safety["maxima"]["post_clamp_target_guard_band_violation_rad"]
        ):
            raise ValueError(
                "Completed episode exceeded the target guard-band recovery tolerance"
            )
        if any(
            guard > current + JOINT_SLEW_FLOAT32_TOLERANCE_RAD
            for guard, current in zip(
                safety["maxima"]["post_clamp_target_guard_band_violation_rad"],
                safety["maxima"]["current_joint_soft_limit_violation_rad"],
                strict=True,
            )
        ):
            raise ValueError(
                "Completed episode target guard-band recovery is not attributable "
                "to a tolerated current-state limit excursion"
            )
        if any(
            value > CURRENT_JOINT_SOFT_LIMIT_TOLERANCE_RAD
            for value in safety["maxima"]["current_joint_soft_limit_violation_rad"]
        ):
            raise ValueError("Completed episode current-position maximum is unsafe")
        max_raw = safety.get("max_raw_delta_diagnostic")
        if not isinstance(max_raw, Mapping):
            raise ValueError("Completed episode lacks max-raw-delta diagnostic")
        flattened_index = (
            max_raw["policy_step"] * CANONICAL_DECIMATION + max_raw["physics_substep"]
        )
        if flattened_index >= apply_calls:
            raise ValueError("Completed episode max-raw diagnostic is out of cadence")
        for vector_name in (
            "joint_pos_rad",
            "raw_delta_joint_pos_rad",
            "raw_joint_pos_target_rad",
            "safe_joint_pos_target_rad",
        ):
            vector = max_raw.get(vector_name)
            if not isinstance(vector, Mapping) or vector.get("finite_count") != 7:
                raise ValueError("Completed episode max-raw diagnostic is non-finite")
        safe_values = max_raw["safe_joint_pos_target_rad"]["values"]
        target_limits = safety["target_joint_pos_limits_rad"]
        if any(
            value < lower - CURRENT_JOINT_SOFT_LIMIT_TOLERANCE_RAD
            or value > upper + CURRENT_JOINT_SOFT_LIMIT_TOLERANCE_RAD
            for value, (lower, upper) in zip(safe_values, target_limits, strict=True)
        ):
            raise ValueError(
                "Completed episode max-raw safe target exceeded the target "
                "guard-band recovery tolerance"
            )
        if max_raw.get("jacobian_finite") is not True or any(
            not isinstance(max_raw.get(name), (int, float))
            or isinstance(max_raw.get(name), bool)
            or not math.isfinite(float(max_raw[name]))
            or max_raw[name] < 0
            for name in ("pose_error_norm", "jacobian_max_abs")
        ):
            raise ValueError("Completed episode max-raw scalar evidence is invalid")
        raw_vector = max_raw["raw_delta_joint_pos_rad"]
        diagnostic_max = max(abs(value) for value in raw_vector["values"])
        aggregate_max = max(safety["maxima"]["raw_delta_joint_pos_rad"])
        if not math.isclose(
            diagnostic_max,
            aggregate_max,
            rel_tol=0.0,
            abs_tol=JOINT_SLEW_FLOAT32_TOLERANCE_RAD,
        ):
            raise ValueError(
                "Completed episode max-raw diagnostic disagrees with maxima"
            )
        failed_policy_step = None
        failed_physics_substep = None
    return {
        "apply_calls": apply_calls,
        "expected_decimation": CANONICAL_DECIMATION,
        "failed_policy_step": failed_policy_step,
        "failed_physics_substep": failed_physics_substep,
        "abort_count": abort_count,
    }


def load_episode_safety_sidecars(
    directory: Path, committed_episode_indices: list[int]
) -> list[dict[str, Any]]:
    """Load the exact sidecar set corresponding to committed CSV episodes."""

    if committed_episode_indices != list(range(len(committed_episode_indices))):
        raise ValueError(
            "Committed safety episode indices must be a contiguous ordered prefix: "
            f"{committed_episode_indices!r}"
        )
    expected = {
        directory / f"episode_{episode_index:06d}.json"
        for episode_index in committed_episode_indices
    }
    actual = set(directory.glob("episode_*.json")) if directory.exists() else set()
    if actual != expected:
        missing = sorted(str(path) for path in expected - actual)
        orphaned = sorted(str(path) for path in actual - expected)
        raise ValueError(
            "Episode safety sidecars are not transactionally aligned with CSV: "
            f"missing={missing[:5]}, orphaned={orphaned[:5]}"
        )
    payloads = []
    for episode_index in committed_episode_indices:
        path = directory / f"episode_{episode_index:06d}.json"
        payload = _load_strict_json(path)
        if set(payload) != SAFETY_SIDECAR_FIELDS:
            raise ValueError(f"Episode safety sidecar schema drift: {path}")
        if (
            payload.get("schema_version") != 2
            or payload.get("transaction_state") != "prepared"
            or payload.get("episode_index") != episode_index
        ):
            raise ValueError(f"Invalid episode safety sidecar identity: {path}")
        result = canonical_episode_result(payload.get("episode_result", {}))
        if result["episode"] != episode_index:
            raise ValueError(f"Drifted episode result identity in sidecar: {path}")
        cadence = validate_episode_safety_cadence(
            safety=payload.get("safety", {}),
            episode_result=result,
        )
        if payload.get("cadence_evidence") != cadence:
            raise ValueError(f"Drifted cadence evidence in sidecar: {path}")
        if not isinstance(payload.get("artifact_identity"), Mapping):
            raise ValueError(f"Missing artifact identity in sidecar: {path}")
        _validate_artifact_identity_schema(payload["artifact_identity"])
        payload["path"] = str(path)
        payload["sha256"] = _sha256(path)
        payloads.append(payload)
    return payloads


def reconcile_episode_safety_transactions(
    frame: pd.DataFrame,
    *,
    directory: Path,
    run_folder: Path,
    trace_dir: Path,
    expected_rollouts: int,
    expected_horizon: int,
    video_probe_fn: Callable[[Path], Mapping[str, Any]] = probe_episode_video,
) -> tuple[pd.DataFrame, bool]:
    """Validate committed sidecars and recover one prepared row after a crash.

    The immutable sidecar is written only after its video and finalized trace.
    Therefore the sole valid difference between CSV and sidecars is one prepared
    sidecar at the next contiguous episode.  Its exact row is safely replayed
    into the CSV; no evidence is deleted or overwritten.
    """

    rows = [
        canonical_episode_result(row)
        for row in frame.loc[:, list(EVAL_RESULT_COLUMNS)].to_dict(orient="records")
    ]
    committed_indices = [row["episode"] for row in rows]
    if committed_indices != list(range(len(rows))):
        raise ValueError(
            "CSV episode identities are not a contiguous prefix during recovery: "
            f"{committed_indices!r}"
        )
    if len(rows) > expected_rollouts:
        raise ValueError(
            f"CSV has {len(rows)} episodes for only {expected_rollouts} rollouts"
        )
    actual_paths = (
        sorted(directory.glob("episode_*.json")) if directory.exists() else []
    )
    actual_indices: list[int] = []
    for path in actual_paths:
        expected_prefix = "episode_"
        try:
            suffix = path.stem.removeprefix(expected_prefix)
            episode_index = int(suffix)
        except ValueError as error:
            raise ValueError(f"Invalid safety sidecar filename: {path}") from error
        if path.name != f"episode_{episode_index:06d}.json":
            raise ValueError(f"Noncanonical safety sidecar filename: {path}")
        actual_indices.append(episode_index)
    if actual_indices != sorted(set(actual_indices)):
        raise ValueError(f"Duplicate or unordered safety sidecars: {actual_indices!r}")
    allowed_indices = list(range(len(rows)))
    if len(rows) < expected_rollouts:
        allowed_indices.append(len(rows))
    if actual_indices not in (list(range(len(rows))), allowed_indices):
        raise ValueError(
            "Safety transactions must equal the CSV prefix or add exactly its "
            f"next episode: csv={committed_indices!r}, sidecars={actual_indices!r}"
        )
    prepared_episode = (
        len(rows)
        if actual_indices == allowed_indices and len(actual_indices) > len(rows)
        else None
    )
    _preserve_uncommitted_episode_artifacts(
        run_folder=run_folder,
        trace_dir=trace_dir,
        committed_count=len(rows),
        prepared_episode=prepared_episode,
    )

    recovered = False
    for episode_index in actual_indices:
        path = directory / f"episode_{episode_index:06d}.json"
        payload = _load_strict_json(path)
        if set(payload) != SAFETY_SIDECAR_FIELDS:
            raise ValueError(f"Prepared safety sidecar schema drift: {path}")
        if (
            payload.get("schema_version") != 2
            or payload.get("transaction_state") != "prepared"
            or payload.get("episode_index") != episode_index
        ):
            raise ValueError(f"Invalid prepared episode safety transaction: {path}")
        result = canonical_episode_result(payload.get("episode_result", {}))
        if result["episode"] != episode_index:
            raise ValueError(f"Sidecar result identity mismatch: {path}")
        if not 1 <= result["episode_length"] <= expected_horizon:
            raise ValueError(
                f"Sidecar episode length exceeds horizon {expected_horizon}: {path}"
            )
        cadence = validate_episode_safety_cadence(
            safety=payload.get("safety", {}), episode_result=result
        )
        if payload.get("cadence_evidence") != cadence:
            raise ValueError(f"Sidecar cadence identity mismatch: {path}")
        artifact_identity = payload.get("artifact_identity")
        if not isinstance(artifact_identity, Mapping):
            raise ValueError(f"Sidecar has no artifact identity: {path}")
        _validate_artifact_identity_schema(artifact_identity)
        validate_episode_artifact_identity(
            artifact_identity,
            run_folder=run_folder,
            trace_dir=trace_dir,
            episode_result=result,
            video_probe_fn=video_probe_fn,
        )
        if episode_index < len(rows):
            if rows[episode_index] != result:
                raise ValueError(
                    "Committed CSV row differs from immutable episode sidecar: "
                    f"csv={rows[episode_index]!r}, sidecar={result!r}"
                )
        else:
            rows.append(result)
            recovered = True

    reconciled = (
        pd.DataFrame(rows, columns=EVAL_RESULT_COLUMNS)
        if rows
        else pd.DataFrame(columns=EVAL_RESULT_COLUMNS)
    )
    return reconciled, recovered


def aggregate_episode_safety(
    live_template: Mapping[str, Any],
    sidecars: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge immutable per-episode reports without losing resume history."""

    if set(live_template) != EPISODE_SAFETY_FIELDS:
        raise ValueError("Live safety template schema drift")
    static = {field: live_template[field] for field in SAFETY_STATIC_FIELDS}
    counter_names = set(live_template["counters"])
    maxima_names = set(live_template["maxima"])
    counters = {field: 0 for field in counter_names}
    maxima = {field: [0.0] * 7 for field in maxima_names}
    episodes = []
    for sidecar in sidecars:
        safety = sidecar.get("safety")
        if not isinstance(safety, Mapping):
            raise ValueError("Episode safety sidecar has no safety object")
        for field, expected in static.items():
            if safety.get(field) != expected:
                raise ValueError(
                    f"Episode safety static field drift for {field}: "
                    f"expected={expected!r}, actual={safety.get(field)!r}"
                )
        if (
            set(safety.get("counters", {})) != counter_names
            or set(safety.get("maxima", {})) != maxima_names
        ):
            raise ValueError("Episode safety counter/maximum schema drift")
        for field in counter_names:
            counters[field] += int(safety["counters"][field])
        for field in maxima_names:
            maxima[field] = [
                max(previous, float(current))
                for previous, current in zip(
                    maxima[field], safety["maxima"][field], strict=True
                )
            ]
        episodes.append(
            {
                "episode_index": sidecar["episode_index"],
                "episode_result": sidecar["episode_result"],
                "artifact_identity": sidecar["artifact_identity"],
                "cadence_evidence": sidecar["cadence_evidence"],
                "counters": safety["counters"],
                "maxima": safety["maxima"],
                "guard_diagnostics": safety["guard_diagnostics"],
                "max_raw_delta_diagnostic": safety["max_raw_delta_diagnostic"],
                "sidecar_path": sidecar["path"],
                "sidecar_sha256": sidecar["sha256"],
            }
        )
    aggregate = {
        **static,
        "episodes_completed": len(episodes),
        "counters": counters,
        "maxima": maxima,
        "episodes": episodes,
    }
    if set(aggregate) != AGGREGATE_SAFETY_FIELDS or any(
        set(episode) != RUNTIME_EPISODE_FIELDS for episode in episodes
    ):
        raise ValueError("Aggregate EEF IK safety schema drift")
    return aggregate


def atomic_write_runtime_contract(
    path: Path,
    *,
    protocol: Mapping[str, Any],
    frame: Mapping[str, Any],
    ik_safety: Mapping[str, Any],
) -> None:
    """Atomically persist the live simulator/controller contract for this attempt."""

    if set(ik_safety) != AGGREGATE_SAFETY_FIELDS:
        raise ValueError("Runtime aggregate EEF IK safety schema drift")
    episodes = ik_safety.get("episodes")
    if not isinstance(episodes, list) or any(
        not isinstance(episode, Mapping) or set(episode) != RUNTIME_EPISODE_FIELDS
        for episode in episodes
    ):
        raise ValueError("Runtime aggregate episode safety schema drift")
    payload = {
        "schema_version": 2,
        "protocol": dict(protocol),
        "frame": dict(frame),
        "ik_safety": dict(ik_safety),
    }
    _atomic_write_json(path, payload)
