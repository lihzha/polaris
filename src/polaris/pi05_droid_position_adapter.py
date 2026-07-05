"""Pure official-DROID command adapter and closed execution evidence.

The released ``pi05_droid`` policy emits normalized DROID joint-velocity
commands.  The DROID robot stack does not send those values to a velocity
drive.  At every 15 Hz command boundary it clips the command, converts it to a
0.2-radian joint delta, adds that delta to a *fresh* joint measurement, and
sends the resulting absolute joint-position target to the impedance
controller.  This module keeps that boundary independent from Isaac Lab so it
can be golden-tested on CPU.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import numpy as np


PI05_DROID_POSITION_ADAPTER_PROFILE = (
    "openpi_pi05_droid_fresh_jointdelta_position_v1"
)
PI05_DROID_POSITION_TRACE_SCHEMA_VERSION = 1
PI05_DROID_ARM_COMMAND_SCALE_RAD = 0.2
PI05_DROID_POLICY_FREQUENCY_HZ = 15
PI05_DROID_PHYSICS_FREQUENCY_HZ = 120
PI05_DROID_PHYSICS_SUBSTEPS = 8
PI05_DROID_RESPONSE_HORIZON = 15
PI05_DROID_EXECUTION_HORIZON = 8
PI05_DROID_ACTION_DIM = 8
PI05_DROID_ARM_DIM = 7
PI05_DROID_POSITION_LIMIT_CONTRACT_PROFILE = (
    "openpi_pi05_droid_position_limits_factor1_hard_soft_intersection_v1"
)
PI05_DROID_SOFT_JOINT_POSITION_LIMIT_FACTOR = 1.0
PI05_DROID_TARGET_GUARD_INSET_RAD = 0.0
PI05_DROID_PHYSX_HARD_LIMITS_LE_F32_SHA256 = (
    "d7ec7ea6108d670f910c43a9fba370e5023c7a5b9aa31df06b89ffc172529e00"
)
PI05_DROID_ISAACLAB_SOFT_LIMITS_LE_F32_SHA256 = (
    "fbf7535901c042fea5d901812ecd02c5fd81ade06c23c1499c32d66a859104de"
)
PI05_DROID_TARGET_GUARD_LIMITS_LE_F32_SHA256 = (
    "558f0c01f992abe1e6c60559665047d115b4891d25cd627c7bd68d9e9cbcfedb"
)
PI05_DROID_PHYSX_HARD_LIMITS_RAD = (
    (-2.8973000049591064, 2.8973000049591064),
    (-1.7627999782562256, 1.7627999782562256),
    (-2.8973000049591064, 2.8973000049591064),
    (-3.0717999935150146, -0.0697999969124794),
    (-2.8973000049591064, 2.8973000049591064),
    (-0.017500000074505806, 3.752500057220459),
    (-2.8973000049591064, 2.8973000049591064),
)
PI05_DROID_ISAACLAB_SOFT_LIMITS_RAD = (
    (-2.8973000049591064, 2.8973000049591064),
    (-1.7627999782562256, 1.7627999782562256),
    (-2.8973000049591064, 2.8973000049591064),
    (-3.0717999935150146, -0.06979990005493164),
    (-2.8973000049591064, 2.8973000049591064),
    (-0.017499923706054688, 3.752500057220459),
    (-2.8973000049591064, 2.8973000049591064),
)
PI05_DROID_TARGET_GUARD_LIMITS_RAD = (
    (-2.8973000049591064, 2.8973000049591064),
    (-1.7627999782562256, 1.7627999782562256),
    (-2.8973000049591064, 2.8973000049591064),
    (-3.0717999935150146, -0.0697999969124794),
    (-2.8973000049591064, 2.8973000049591064),
    (-0.017499923706054688, 3.752500057220459),
    (-2.8973000049591064, 2.8973000049591064),
)

OFFICIAL_DROID_REPOSITORY = "https://github.com/droid-dataset/droid.git"
OFFICIAL_DROID_COMMIT = "33ae6a67274f36d2e29525b86f23a56616ef43a7"
OFFICIAL_DROID_CONTROL_BLOB_BASE_COMMIT = (
    "c5737e40a6b18859b5b78dbcdbf1e3b3f5e461be"
)
OFFICIAL_DROID_CONTROL_SOURCES = {
    "droid/franka/robot.py": {
        "sha256": (
            "25f2edf095b13f590371a4c53c8fbf0b8948d5c08b42600542a579917442ec38"
        ),
        "git_blob_sha1": "e4f02202542ffcc050b089aa9ff0fde16323d289",
    },
    "droid/robot_env.py": {
        "sha256": (
            "41cff898b9e3c3b465c3465fd4b9889db70edd9da4c083af1789ea335ad6e116"
        ),
        "git_blob_sha1": "f555b501b5b91fb53f82d54c43684078cef15892",
    },
    "droid/robot_ik/robot_ik_solver.py": {
        "sha256": (
            "c32df1ea7e8c56fc32b8560c0a057892c505d1a05fa0a850540b50b6e964c57d"
        ),
        "git_blob_sha1": "a073699c20b4bef3e5941454c0d6a9aaf7b05534",
    },
}


class PositionActionTargetLimitError(ValueError):
    """Action-term target was rejected before the PhysX setter."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode strict canonical JSON used by trace and contract identities."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _one_env_float32_limits(value: Any, *, field: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape == (7, 2):
        array = array[None, ...]
    if (
        array.shape != (1, 7, 2)
        or array.dtype != np.float32
        or not np.isfinite(array).all()
        or bool((array[:, :, 0] >= array[:, :, 1]).any())
    ):
        raise ValueError(f"{field} must be finite float32 [1,7,2] lower/upper limits")
    return np.ascontiguousarray(array)


def position_limits_le_float32_sha256(value: Any) -> str:
    limits = _one_env_float32_limits(
        np.asarray(value, dtype=np.float32), field="position limits"
    )
    return hashlib.sha256(limits.astype("<f4", copy=False).tobytes()).hexdigest()


def derive_isaaclab_factor1_soft_limits(hard_limits: Any) -> np.ndarray:
    """Reproduce Isaac Lab's float32 center/range soft-limit arithmetic."""

    hard = _one_env_float32_limits(hard_limits, field="PhysX hard limits")
    center = (hard[:, :, 0] + hard[:, :, 1]) / np.float32(2.0)
    extent = hard[:, :, 1] - hard[:, :, 0]
    soft_extent = np.float32(0.5) * extent
    soft_extent = soft_extent * np.float32(
        PI05_DROID_SOFT_JOINT_POSITION_LIMIT_FACTOR
    )
    return np.ascontiguousarray(
        np.stack(
            [center - soft_extent, center + soft_extent], axis=-1
        ),
        dtype=np.float32,
    )


def exact_position_target_guard_from_live_limits(
    live_hard_limits: Any, live_soft_limits: Any
) -> np.ndarray:
    """Return the inclusive hard/soft intersection used by both guards."""

    live_hard = _one_env_float32_limits(
        live_hard_limits, field="live buffered PhysX hard limits"
    )
    live_soft = _one_env_float32_limits(
        live_soft_limits, field="live Isaac Lab soft limits"
    )
    expected_hard = np.asarray(
        [PI05_DROID_PHYSX_HARD_LIMITS_RAD], dtype=np.float32
    )
    expected_soft = np.asarray(
        [PI05_DROID_ISAACLAB_SOFT_LIMITS_RAD], dtype=np.float32
    )
    if not np.array_equal(live_hard, expected_hard):
        raise ValueError("live buffered hard limits differ from exact PhysX contract")
    if not np.array_equal(live_soft, expected_soft):
        raise ValueError("live Isaac Lab soft limits differ from exact factor-1 contract")
    guard = np.stack(
        [
            np.maximum(live_hard[:, :, 0], live_soft[:, :, 0])
            + np.float32(PI05_DROID_TARGET_GUARD_INSET_RAD),
            np.minimum(live_hard[:, :, 1], live_soft[:, :, 1])
            - np.float32(PI05_DROID_TARGET_GUARD_INSET_RAD),
        ],
        axis=-1,
    )
    expected_guard = np.asarray(
        [PI05_DROID_TARGET_GUARD_LIMITS_RAD], dtype=np.float32
    )
    if not np.array_equal(guard, expected_guard):
        raise ValueError("hard/soft position target intersection drifted")
    return np.ascontiguousarray(guard, dtype=np.float32)


def evaluate_position_target_guard(
    target: Any, live_hard_limits: Any, live_soft_limits: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cast as the action buffer does and apply one inclusive exact guard."""

    target_array = np.asarray(target)
    if target_array.shape == (7,):
        target_array = target_array[None, :]
    if target_array.shape != (1, 7) or not np.issubdtype(
        target_array.dtype, np.number
    ):
        raise ValueError("position target guard requires one numeric [1,7] target")
    target_float32 = np.ascontiguousarray(target_array, dtype=np.float32)
    if not np.isfinite(target_float32).all():
        raise ValueError("position target is non-finite after float32 actuator cast")
    guard = exact_position_target_guard_from_live_limits(
        live_hard_limits, live_soft_limits
    )
    violation = (target_float32 < guard[:, :, 0]) | (
        target_float32 > guard[:, :, 1]
    )
    return target_float32, guard, violation


def expected_position_limit_contract() -> dict[str, Any]:
    hard = np.asarray([PI05_DROID_PHYSX_HARD_LIMITS_RAD], dtype=np.float32)
    derived_soft = derive_isaaclab_factor1_soft_limits(hard)
    expected_soft = np.asarray(
        [PI05_DROID_ISAACLAB_SOFT_LIMITS_RAD], dtype=np.float32
    )
    if not np.array_equal(derived_soft, expected_soft):
        raise RuntimeError("position-limit constants do not reproduce Isaac Lab")
    guard = exact_position_target_guard_from_live_limits(hard, derived_soft)
    value = {
        "schema_version": 1,
        "profile": PI05_DROID_POSITION_LIMIT_CONTRACT_PROFILE,
        "hard_limits": {
            "source": "Articulation.root_physx_view.get_dof_limits",
            "shape": [1, 7, 2],
            "dtype": "float32",
            "values_rad": hard.tolist(),
            "little_endian_float32_sha256": (
                PI05_DROID_PHYSX_HARD_LIMITS_LE_F32_SHA256
            ),
        },
        "isaaclab_soft_limits": {
            "source": "Articulation.data.soft_joint_pos_limits",
            "shape": [1, 7, 2],
            "dtype": "float32",
            "soft_joint_pos_limit_factor": (
                PI05_DROID_SOFT_JOINT_POSITION_LIMIT_FACTOR
            ),
            "derivation": (
                "float32_center_equals_lower_plus_upper_over_2_then_"
                "center_plus_or_minus_0p5_times_range_times_factor_v1"
            ),
            "rounding_note": (
                "factor_1_is_not_raw_copy_and_shifts_joint4_upper_and_"
                "joint6_lower_by_float32_center_range_rounding"
            ),
            "values_rad": derived_soft.tolist(),
            "little_endian_float32_sha256": (
                PI05_DROID_ISAACLAB_SOFT_LIMITS_LE_F32_SHA256
            ),
        },
        "target_guard": {
            "guard_source": (
                "intersection(live_joint_pos_limits,live_soft_joint_pos_limits)"
            ),
            "guard_inset_rad": PI05_DROID_TARGET_GUARD_INSET_RAD,
            "derivation": "lower_max_and_upper_min_float32_v1",
            "comparison": "inclusive_elementwise_zero_tolerance_no_allclose",
            "shape": [1, 7, 2],
            "dtype": "float32",
            "values_rad": guard.tolist(),
            "little_endian_float32_sha256": (
                PI05_DROID_TARGET_GUARD_LIMITS_LE_F32_SHA256
            ),
        },
    }
    value["contract_sha256"] = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return value


def validate_position_limit_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("position-limit contract must be an object")
    expected = expected_position_limit_contract()
    if value != expected:
        raise ValueError("position-limit contract mismatch")
    for field, expected_digest in (
        ("hard_limits", PI05_DROID_PHYSX_HARD_LIMITS_LE_F32_SHA256),
        ("isaaclab_soft_limits", PI05_DROID_ISAACLAB_SOFT_LIMITS_LE_F32_SHA256),
        ("target_guard", PI05_DROID_TARGET_GUARD_LIMITS_LE_F32_SHA256),
    ):
        digest = position_limits_le_float32_sha256(value[field]["values_rad"])
        if digest != expected_digest:
            raise ValueError(f"position-limit {field} digest mismatch")
    return copy.deepcopy(value)


def _numeric_vector(value: Any, *, size: int, field: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (size,) or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{field} must be one numeric {size}-vector")
    if not np.isfinite(array).all():
        raise ValueError(f"{field} contains non-finite values")
    return array


def adapt_official_droid_action(
    raw_action: Any, measured_joint_position: Any
) -> dict[str, Any]:
    """Convert one model command into one absolute PolaRiS target.

    The fresh measurement is deliberately supplied for every call.  No query-
    time or previous-target anchor is accepted by this API.
    """

    raw = _numeric_vector(
        raw_action, size=PI05_DROID_ACTION_DIM, field="raw DROID action"
    )
    measured = _numeric_vector(
        measured_joint_position,
        size=PI05_DROID_ARM_DIM,
        field="fresh measured joint position",
    )

    # Match the official OpenPI DROID client exactly: binarize the absolute
    # closed-positive gripper first, then clip all eight command dimensions.
    binary = np.concatenate(
        [raw[:-1], np.ones((1,)) if raw[-1].item() > 0.5 else np.zeros((1,))]
    )
    clipped = np.clip(binary, -1.0, 1.0)
    measured_float64 = np.asarray(measured, dtype=np.float64)
    arm_delta = np.asarray(clipped[:7], dtype=np.float64) * np.float64(
        PI05_DROID_ARM_COMMAND_SCALE_RAD
    )
    absolute_target = measured_float64 + arm_delta
    emitted = np.concatenate([absolute_target, clipped[-1:]])

    evidence = {
        "schema_version": PI05_DROID_POSITION_TRACE_SCHEMA_VERSION,
        "profile": PI05_DROID_POSITION_ADAPTER_PROFILE,
        "anchor": "fresh_measured_joint_position_at_executed_step",
        "query_time_anchor": False,
        "raw_action": raw.tolist(),
        "raw_action_dtype": str(raw.dtype),
        "binary_action": binary.tolist(),
        "clipped_action": clipped.tolist(),
        "clip": [-1.0, 1.0],
        "measured_joint_position": measured_float64.tolist(),
        "measured_joint_position_source_dtype": str(measured.dtype),
        "arm_command_scale_rad": PI05_DROID_ARM_COMMAND_SCALE_RAD,
        "arm_delta_rad": arm_delta.tolist(),
        "absolute_joint_position_target_rad": absolute_target.tolist(),
        "absolute_closed_positive_gripper": float(clipped[-1]),
        "emitted_absolute_action": emitted.tolist(),
        "reference_arithmetic_dtype": "numpy_float64",
        "actuator_command_dtype": "torch_float32_after_env_input_copy",
    }
    return validate_position_adapter_evidence(evidence)


def validate_position_adapter_evidence(value: Any) -> dict[str, Any]:
    """Validate one closed-schema command record and recompute all math."""

    if not isinstance(value, dict):
        raise ValueError("position-adapter evidence must be an object")
    required = {
        "schema_version",
        "profile",
        "anchor",
        "query_time_anchor",
        "raw_action",
        "raw_action_dtype",
        "binary_action",
        "clipped_action",
        "clip",
        "measured_joint_position",
        "measured_joint_position_source_dtype",
        "arm_command_scale_rad",
        "arm_delta_rad",
        "absolute_joint_position_target_rad",
        "absolute_closed_positive_gripper",
        "emitted_absolute_action",
        "reference_arithmetic_dtype",
        "actuator_command_dtype",
    }
    if set(value) != required:
        raise ValueError("position-adapter evidence schema mismatch")
    expected_scalars = {
        "schema_version": PI05_DROID_POSITION_TRACE_SCHEMA_VERSION,
        "profile": PI05_DROID_POSITION_ADAPTER_PROFILE,
        "anchor": "fresh_measured_joint_position_at_executed_step",
        "query_time_anchor": False,
        "clip": [-1.0, 1.0],
        "arm_command_scale_rad": PI05_DROID_ARM_COMMAND_SCALE_RAD,
        "reference_arithmetic_dtype": "numpy_float64",
        "actuator_command_dtype": "torch_float32_after_env_input_copy",
    }
    for field, expected in expected_scalars.items():
        if value[field] != expected:
            raise ValueError(f"position-adapter {field} mismatch")
    if not isinstance(value["raw_action_dtype"], str) or not isinstance(
        value["measured_joint_position_source_dtype"], str
    ):
        raise ValueError("position-adapter source dtypes must be strings")

    raw = _numeric_vector(value["raw_action"], size=8, field="raw_action")
    measured = _numeric_vector(
        value["measured_joint_position"],
        size=7,
        field="measured_joint_position",
    )
    recomputed = adapt_official_droid_action_unvalidated(raw, measured)
    for field in (
        "binary_action",
        "clipped_action",
        "arm_delta_rad",
        "absolute_joint_position_target_rad",
        "emitted_absolute_action",
    ):
        actual = _numeric_vector(
            value[field], size=8 if field in {"binary_action", "clipped_action", "emitted_absolute_action"} else 7, field=field
        )
        if not np.array_equal(actual, recomputed[field]):
            raise ValueError(f"position-adapter {field} math mismatch")
    gripper = value["absolute_closed_positive_gripper"]
    if type(gripper) not in (int, float) or isinstance(gripper, bool):
        raise ValueError("position-adapter gripper must be numeric")
    if float(gripper) != float(recomputed["clipped_action"][-1]) or gripper not in (
        0.0,
        1.0,
    ):
        raise ValueError("position-adapter gripper is not absolute binary")
    return copy.deepcopy(value)


def adapt_official_droid_action_unvalidated(
    raw_action: Any, measured_joint_position: Any
) -> dict[str, np.ndarray]:
    """Internal numeric oracle used to avoid recursive evidence validation."""

    raw = _numeric_vector(raw_action, size=8, field="raw_action")
    measured = _numeric_vector(
        measured_joint_position, size=7, field="measured_joint_position"
    )
    binary = np.concatenate(
        [raw[:-1], np.ones((1,)) if raw[-1].item() > 0.5 else np.zeros((1,))]
    )
    clipped = np.clip(binary, -1.0, 1.0)
    measured64 = np.asarray(measured, dtype=np.float64)
    delta = np.asarray(clipped[:7], dtype=np.float64) * np.float64(0.2)
    target = measured64 + delta
    return {
        "binary_action": binary,
        "clipped_action": clipped,
        "arm_delta_rad": delta,
        "absolute_joint_position_target_rad": target,
        "emitted_absolute_action": np.concatenate([target, clipped[-1:]]),
    }


def _target_array(value: Any, *, field: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape == (7,):
        array = array[None, :]
    if array.shape != (1, 7) or array.dtype != np.float32:
        raise ValueError(f"{field} must be one float32 [1,7] target")
    if not np.isfinite(array).all():
        raise ValueError(f"{field} contains non-finite values")
    return np.ascontiguousarray(array)


class PositionTargetHoldRecorder:
    """Prove one absolute target is applied on exactly eight physics substeps."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._expected: np.ndarray | None = None
        self._applied: list[np.ndarray] = []

    def begin_policy_step(self, target: Any) -> None:
        if self._expected is not None:
            raise RuntimeError("previous position-target hold was not consumed")
        self._expected = _target_array(target, field="policy-step target").copy()
        self._applied = []

    def record_physics_substep(self, target: Any) -> None:
        if self._expected is None:
            raise RuntimeError("position target applied without a policy step")
        applied = _target_array(target, field="applied position target")
        if not np.array_equal(applied, self._expected):
            raise ValueError("absolute position target changed within policy step")
        self._applied.append(applied.copy())

    def finish_policy_step(self) -> dict[str, Any]:
        if self._expected is None:
            raise RuntimeError("no active position-target hold")
        if len(self._applied) != PI05_DROID_PHYSICS_SUBSTEPS:
            raise ValueError(
                "position target must be held for exactly eight physics substeps"
            )
        digest = hashlib.sha256(self._expected.astype("<f4", copy=False).tobytes()).hexdigest()
        report = {
            "schema_version": 1,
            "profile": PI05_DROID_POSITION_ADAPTER_PROFILE,
            "policy_frequency_hz": PI05_DROID_POLICY_FREQUENCY_HZ,
            "physics_frequency_hz": PI05_DROID_PHYSICS_FREQUENCY_HZ,
            "physics_substeps": PI05_DROID_PHYSICS_SUBSTEPS,
            "apply_calls": len(self._applied),
            "target_shape": [1, 7],
            "target_dtype": "float32",
            "target_little_endian_float32_sha256": digest,
            "unique_applied_target_count": 1,
            "absolute_joint_position_target_rad": self._expected[0].tolist(),
            "setter": "Articulation.set_joint_position_target",
        }
        self.reset()
        return validate_position_target_hold_report(report)


def validate_position_target_hold_report(value: Any) -> dict[str, Any]:
    """Validate closed hold evidence and its target digest."""

    if not isinstance(value, dict):
        raise ValueError("position-target hold report must be an object")
    required = {
        "schema_version",
        "profile",
        "policy_frequency_hz",
        "physics_frequency_hz",
        "physics_substeps",
        "apply_calls",
        "target_shape",
        "target_dtype",
        "target_little_endian_float32_sha256",
        "unique_applied_target_count",
        "absolute_joint_position_target_rad",
        "setter",
    }
    if set(value) != required:
        raise ValueError("position-target hold report schema mismatch")
    expected = {
        "schema_version": 1,
        "profile": PI05_DROID_POSITION_ADAPTER_PROFILE,
        "policy_frequency_hz": 15,
        "physics_frequency_hz": 120,
        "physics_substeps": 8,
        "apply_calls": 8,
        "target_shape": [1, 7],
        "target_dtype": "float32",
        "unique_applied_target_count": 1,
        "setter": "Articulation.set_joint_position_target",
    }
    for field, expected_value in expected.items():
        if value[field] != expected_value:
            raise ValueError(f"position-target hold {field} mismatch")
    target = np.asarray(value["absolute_joint_position_target_rad"])
    if target.shape != (7,) or not np.isfinite(target).all():
        raise ValueError("position-target hold target is invalid")
    digest = hashlib.sha256(
        np.asarray(target, dtype="<f4").reshape(1, 7).tobytes()
    ).hexdigest()
    if value["target_little_endian_float32_sha256"] != digest:
        raise ValueError("position-target hold target digest mismatch")
    return copy.deepcopy(value)


def official_droid_source_contract() -> dict[str, Any]:
    """Return the immutable primary-source identity behind the adapter."""

    return {
        "repository": OFFICIAL_DROID_REPOSITORY,
        "revision": OFFICIAL_DROID_COMMIT,
        "control_blobs_unchanged_since": OFFICIAL_DROID_CONTROL_BLOB_BASE_COMMIT,
        "control_sources": copy.deepcopy(OFFICIAL_DROID_CONTROL_SOURCES),
        "robot_env_action_space": "joint_velocity",
        "robot_env_gripper_action_space": "position",
        "normalized_arm_command_clip": [-1.0, 1.0],
        "joint_velocity_to_delta_scale_rad": 0.2,
        "joint_target_anchor": "fresh_robot_state_joint_positions_per_command",
        "joint_controller_command": "absolute_joint_position",
        "gripper_command": "absolute_closed_positive_position",
    }
