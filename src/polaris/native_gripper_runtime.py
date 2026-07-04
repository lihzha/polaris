"""Fail-closed all-six Robotiq runtime for native DROID control.

The NVIDIA DROID articulation has one actuator-owned driver and five passive
PhysX mimic followers.  Isaac Lab's actuator configuration updates the driver
only, so every native reset must update both the CUDA articulation buffer and
the CPU PhysX full-DOF tensor for the five followers.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np


NATIVE_GRIPPER_ALL_SIX_PROFILE = (
    "implicit_gripper_physx_velocity_limit5_followers5_every_reset_"
    "cuda_actuator_cpu_static_physx_v1"
)
NATIVE_GRIPPER_RESET_WRITE_PROFILE = (
    "native_reset_event_public_articulation_full13_five_mimic_dofs_velocity_limit5_v1"
)
NATIVE_GRIPPER_RESET_WRITE_TIMING = (
    "after_reset_scene_to_default_on_every_native_reset_v1"
)
NATIVE_GRIPPER_RESET_WRITE_SETTER = "Articulation.write_joint_velocity_limit_to_sim"
NATIVE_GRIPPER_MIMIC_PROFILE = "robotiq_2f85_source_usd_physx_mimic_joint_v1"
NATIVE_GRIPPER_DYNAMIC_PROFILE = (
    "native_jointvelocity_all13_apply_entry_plus_post_policy_step_v1"
)
NATIVE_ALL_JOINT_VELOCITY_FAILURE_PROFILE = (
    "native_jointvelocity_all13_measured_velocity_limit_failure_v1"
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
ARM_JOINT_INDICES = tuple(range(7))
DRIVER_JOINT_INDEX = 7
GRIPPER_JOINT_INDICES = tuple(range(7, 13))
FOLLOWER_JOINT_INDICES = tuple(range(8, 13))
GRIPPER_JOINT_NAMES = EXPECTED_DROID_JOINT_NAMES[7:]
FOLLOWER_JOINT_NAMES = EXPECTED_DROID_JOINT_NAMES[8:]
EXPECTED_ARM_VELOCITY_LIMITS_FLOAT32 = (
    2.174999952316284,
    2.174999952316284,
    2.174999952316284,
    2.174999952316284,
    2.609999895095825,
    2.609999895095825,
    2.609999895095825,
)
DRIVER_VELOCITY_LIMIT_FLOAT32 = 5.0
FOLLOWER_USD_VELOCITY_LIMIT_FLOAT32 = 174.53292846679688
FOLLOWER_VELOCITY_LIMIT_FLOAT32 = 5.0
EXPECTED_FULL_LIMITS_UNCAPPED = (
    *EXPECTED_ARM_VELOCITY_LIMITS_FLOAT32,
    DRIVER_VELOCITY_LIMIT_FLOAT32,
    *([FOLLOWER_USD_VELOCITY_LIMIT_FLOAT32] * 5),
)
EXPECTED_FULL_LIMITS_CAPPED = (
    *EXPECTED_ARM_VELOCITY_LIMITS_FLOAT32,
    *([DRIVER_VELOCITY_LIMIT_FLOAT32] * 6),
)
EXPECTED_ROBOT_USD_SHA256 = (
    "d8379925b103963dbf3e7c85bcc4ae101b81b7c1d7dabe7d2e964f41d069ec44"
)
EXPECTED_ROBOT_USD_SIZE = 14_156_155
EXPECTED_MIMIC_SPECS = (
    ("right_outer_knuckle_joint", 8, "rotZ", -1.0, 1_000_000.0, 0.0),
    ("left_inner_finger_joint", 9, "rotX", 1.0, 1_000.0, 0.05000000074505806),
    ("right_inner_finger_joint", 10, "rotX", -1.0, 1_000.0, 0.05000000074505806),
    (
        "left_inner_finger_knuckle_joint",
        11,
        "rotX",
        1.0,
        1_000.0,
        0.05000000074505806,
    ),
    (
        "right_inner_finger_knuckle_joint",
        12,
        "rotX",
        1.0,
        1_000.0,
        0.05000000074505806,
    ),
)
STATIC_STATE_ATTRIBUTE = "_polaris_native_gripper_all_six_reset_state"
# A real L40S/PhysX capture with a nominal 5 rad/s driver limit measured
# 5.000018119812012 rad/s.  Keep a narrow, named float/solver allowance while
# remaining orders of magnitude below the observed uncapped 55.622322 rad/s.
PHYSX_VELOCITY_LIMIT_ABSOLUTE_TOLERANCE_RAD_S = 5e-5
PHYSICS_DT = 1.0 / 120.0


class NativeAllJointVelocityLimitError(FloatingPointError):
    """Typed terminal failure carrying a durably published 13-DOF incident."""

    def __init__(
        self,
        evidence: dict[str, Any],
        incident_artifact: dict[str, Any] | None,
    ) -> None:
        validated = validate_native_all_joint_velocity_failure(evidence)
        violating = ",".join(validated["violating_joint_names"])
        super().__init__(
            "measured all-joint velocity exceeded the live physical limit: "
            f"policy_step={validated['policy_step_index']} "
            f"sample_kind={validated['sample_kind']} "
            f"physics_substep={validated['physics_substep_index']} "
            f"joints={violating}"
        )
        self.evidence = validated
        self.incident_artifact = copy.deepcopy(incident_artifact)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _tensor_numpy(value: Any, *, field: str) -> tuple[np.ndarray, str, str]:
    dtype = str(getattr(value, "dtype", ""))
    device = str(getattr(value, "device", ""))
    try:
        array = value.detach().cpu().numpy()
    except AttributeError:
        array = np.asarray(value)
    array = np.asarray(array)
    _require(
        np.issubdtype(array.dtype, np.number)
        and not np.issubdtype(array.dtype, np.bool_)
        and np.isfinite(array).all(),
        f"{field} must be finite and numeric",
    )
    return array, dtype, device


def tensor_report(value: Any, *, field: str) -> dict[str, Any]:
    array, dtype, device = _tensor_numpy(value, field=field)
    _require(
        dtype == "torch.float32" and array.dtype == np.float32, f"{field} dtype drift"
    )
    return {
        "shape": list(array.shape),
        "dtype": dtype,
        "device": device,
        "values": array.tolist(),
    }


def _validate_tensor_report(
    value: Any,
    *,
    field: str,
    shape: tuple[int, ...],
    device: str,
    expected: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    _require(
        isinstance(value, dict)
        and set(value) == {"shape", "dtype", "device", "values"},
        f"{field} schema drift",
    )
    _require(
        value["shape"] == list(shape)
        and value["dtype"] == "torch.float32"
        and value["device"] == device,
        f"{field} tensor identity drift",
    )
    array = np.asarray(value["values"])
    _require(
        array.shape == shape
        and np.issubdtype(array.dtype, np.number)
        and not np.issubdtype(array.dtype, np.bool_)
        and np.isfinite(array).all(),
        f"{field} values invalid",
    )
    if expected is not None:
        expected_array = np.asarray(expected, dtype=np.float32).reshape(shape)
        _require(
            np.array_equal(array.astype(np.float32), expected_array),
            f"{field} values drift",
        )
    return copy.deepcopy(value)


def _joint_names(robot: Any) -> tuple[str, ...]:
    names = tuple(getattr(robot, "joint_names", ()))
    if not names:
        names = tuple(getattr(getattr(robot, "data", None), "joint_names", ()))
    _require(names == EXPECTED_DROID_JOINT_NAMES, "live DROID joint order drift")
    return names


def _normalize_env_ids(env_ids: Any) -> list[int]:
    if env_ids is None:
        return [0]
    try:
        values = env_ids.detach().cpu().tolist()
    except AttributeError:
        values = list(env_ids)
    _require(values == [0], "native all-six reset requires exactly environment zero")
    return values


def _exact_live_limits(robot: Any, *, field: str) -> tuple[Any, Any]:
    buffered = robot.data.joint_vel_limits
    direct = robot.root_physx_view.get_dof_max_velocities()
    buffered_report = tensor_report(buffered, field=f"{field} buffered")
    direct_report = tensor_report(direct, field=f"{field} direct PhysX")
    _validate_tensor_report(
        buffered_report,
        field=f"{field} buffered",
        shape=(1, 13),
        device="cuda:0",
    )
    _validate_tensor_report(
        direct_report,
        field=f"{field} direct PhysX",
        shape=(1, 13),
        device="cpu",
    )
    buffered_array = np.asarray(buffered_report["values"], dtype=np.float32)
    direct_array = np.asarray(direct_report["values"], dtype=np.float32)
    _require(
        np.array_equal(buffered_array, direct_array),
        f"{field} buffered/direct velocity limits disagree",
    )
    return buffered_report, direct_report


def apply_native_gripper_all_six_velocity_limits(env: Any, env_ids: Any) -> None:
    """Apply one full-tensor follower cap after every native environment reset."""

    import torch  # noqa: PLC0415

    runtime = getattr(env, "unwrapped", env)
    robot = runtime.scene["robot"]
    _joint_names(robot)
    normalized_env_ids = _normalize_env_ids(env_ids)
    before_buffered, before_direct = _exact_live_limits(robot, field="pre-reset-write")
    before_values = np.asarray(before_direct["values"], dtype=np.float32)
    allowed_before = (
        np.asarray(EXPECTED_FULL_LIMITS_UNCAPPED, dtype=np.float32),
        np.asarray(EXPECTED_FULL_LIMITS_CAPPED, dtype=np.float32),
    )
    _require(
        any(
            np.array_equal(before_values[0], candidate) for candidate in allowed_before
        ),
        "pre-write full velocity-limit tensor drift",
    )

    replacement = robot.data.joint_vel_limits.clone()
    preserved_before, _, _ = _tensor_numpy(
        replacement[:, : DRIVER_JOINT_INDEX + 1],
        field="preserved arm and driver velocity limits",
    )
    replacement[:, list(FOLLOWER_JOINT_INDICES)] = FOLLOWER_VELOCITY_LIMIT_FLOAT32
    preserved_after, _, _ = _tensor_numpy(
        replacement[:, : DRIVER_JOINT_INDEX + 1],
        field="post-edit arm and driver velocity limits",
    )
    _require(
        np.array_equal(preserved_after, preserved_before),
        "native follower cap changed an arm or driver value",
    )
    writer = getattr(robot, "write_joint_velocity_limit_to_sim", None)
    _require(callable(writer), "missing public articulation velocity-limit writer")
    writer(
        replacement,
        joint_ids=None,
        env_ids=torch.tensor(
            normalized_env_ids,
            dtype=torch.long,
            device=getattr(robot, "device", replacement.device),
        ),
    )

    after_buffered, after_direct = _exact_live_limits(robot, field="post-reset-write")
    for label, report in (
        ("post-write buffered", after_buffered),
        ("post-write direct PhysX", after_direct),
    ):
        values = np.asarray(report["values"], dtype=np.float32)
        _require(
            np.array_equal(
                values[0], np.asarray(EXPECTED_FULL_LIMITS_CAPPED, dtype=np.float32)
            ),
            f"{label} all-six cap drift",
        )
        _require(
            np.array_equal(values[:, :8], before_values[:, :8]),
            f"{label} changed arm or driver values",
        )

    state = getattr(runtime, STATIC_STATE_ATTRIBUTE, None)
    if state is None:
        state = {
            "schema_version": 1,
            "profile": NATIVE_GRIPPER_ALL_SIX_PROFILE,
            "write_profile": NATIVE_GRIPPER_RESET_WRITE_PROFILE,
            "setter": NATIVE_GRIPPER_RESET_WRITE_SETTER,
            "timing": NATIVE_GRIPPER_RESET_WRITE_TIMING,
            "reset_count": 0,
            "write_count": 0,
            "initial_before_buffered": before_buffered,
            "initial_before_direct_physx": before_direct,
        }
    _require(
        isinstance(state, dict)
        and state.get("profile") == NATIVE_GRIPPER_ALL_SIX_PROFILE,
        "native all-six reset state drift",
    )
    state["reset_count"] += 1
    state["write_count"] += 1
    state["latest_env_ids"] = normalized_env_ids
    state["latest_before_buffered"] = before_buffered
    state["latest_before_direct_physx"] = before_direct
    state["latest_full_input"] = tensor_report(
        replacement, field="latest full velocity-limit input"
    )
    state["latest_after_buffered"] = after_buffered
    state["latest_after_direct_physx"] = after_direct
    setattr(runtime, STATIC_STATE_ATTRIBUTE, state)


def native_gripper_reset_report(env: Any) -> dict[str, Any]:
    runtime = getattr(env, "unwrapped", env)
    value = getattr(runtime, STATIC_STATE_ATTRIBUTE, None)
    return validate_native_gripper_reset_report(copy.deepcopy(value))


def validate_native_gripper_reset_report(value: Any) -> dict[str, Any]:
    required = {
        "schema_version",
        "profile",
        "write_profile",
        "setter",
        "timing",
        "reset_count",
        "write_count",
        "initial_before_buffered",
        "initial_before_direct_physx",
        "latest_env_ids",
        "latest_before_buffered",
        "latest_before_direct_physx",
        "latest_full_input",
        "latest_after_buffered",
        "latest_after_direct_physx",
    }
    _require(
        isinstance(value, dict) and set(value) == required, "reset report schema drift"
    )
    exact = {
        "schema_version": 1,
        "profile": NATIVE_GRIPPER_ALL_SIX_PROFILE,
        "write_profile": NATIVE_GRIPPER_RESET_WRITE_PROFILE,
        "setter": NATIVE_GRIPPER_RESET_WRITE_SETTER,
        "timing": NATIVE_GRIPPER_RESET_WRITE_TIMING,
        "latest_env_ids": [0],
    }
    for field, expected in exact.items():
        _require(
            value[field] == expected and type(value[field]) is type(expected),
            f"reset report {field} drift",
        )
    _require(
        type(value["reset_count"]) is int
        and value["reset_count"] >= 1
        and value["write_count"] == value["reset_count"]
        and type(value["write_count"]) is int,
        "reset/write count drift",
    )
    initial_expected = np.asarray(EXPECTED_FULL_LIMITS_UNCAPPED, dtype=np.float32)
    for field, device in (
        ("initial_before_buffered", "cuda:0"),
        ("initial_before_direct_physx", "cpu"),
    ):
        _validate_tensor_report(
            value[field],
            field=field,
            shape=(1, 13),
            device=device,
            expected=tuple(initial_expected.tolist()),
        )
    before_buffered = _validate_tensor_report(
        value["latest_before_buffered"],
        field="latest before buffered",
        shape=(1, 13),
        device="cuda:0",
    )
    before_direct = _validate_tensor_report(
        value["latest_before_direct_physx"],
        field="latest before direct",
        shape=(1, 13),
        device="cpu",
    )
    _require(
        np.array_equal(
            np.asarray(before_buffered["values"], dtype=np.float32),
            np.asarray(before_direct["values"], dtype=np.float32),
        ),
        "latest pre-write buffer/direct drift",
    )
    for field, device in (
        ("latest_full_input", "cuda:0"),
        ("latest_after_buffered", "cuda:0"),
        ("latest_after_direct_physx", "cpu"),
    ):
        _validate_tensor_report(
            value[field],
            field=field,
            shape=(1, 13),
            device=device,
            expected=EXPECTED_FULL_LIMITS_CAPPED,
        )
    return copy.deepcopy(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_native_gripper_mimic_contract(robot_usd_path: Path) -> dict[str, Any]:
    from pxr import Usd  # noqa: PLC0415

    path = Path(robot_usd_path).resolve()
    _require(
        path.stat().st_size == EXPECTED_ROBOT_USD_SIZE
        and _file_sha256(path) == EXPECTED_ROBOT_USD_SHA256,
        "native gripper robot USD identity drift",
    )
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadNone)
    _require(stage is not None, "cannot open native gripper robot USD")
    root = "/panda/Gripper/Robotiq_2F_85/Joints"
    driver_path = f"{root}/finger_joint"
    driver = stage.GetPrimAtPath(driver_path)
    _require(driver and driver.IsValid(), "missing native gripper driver joint")
    result = {
        "profile": NATIVE_GRIPPER_MIMIC_PROFILE,
        "robot_usd_sha256": EXPECTED_ROBOT_USD_SHA256,
        "driver_joint_name": "finger_joint",
        "driver_joint_index": DRIVER_JOINT_INDEX,
        "driver_joint_prim_path": str(driver.GetPath()),
        "driver_physics_joint_type": driver.GetTypeName(),
        "driver_exclude_from_articulation": driver.GetAttribute(
            "physics:excludeFromArticulation"
        ).Get(),
        "followers": [],
    }
    for name, index, axis, gearing, frequency, damping in EXPECTED_MIMIC_SPECS:
        prim_path = f"{root}/{name}"
        prim = stage.GetPrimAtPath(prim_path)
        _require(prim and prim.IsValid(), f"missing native mimic joint {name}")
        namespace = f"physxMimicJoint:{axis}"
        references = prim.GetRelationship(f"{namespace}:referenceJoint").GetTargets()
        _require(len(references) == 1, f"native mimic reference drift {name}")
        result["followers"].append(
            {
                "joint_name": name,
                "joint_index": index,
                "prim_path": str(prim.GetPath()),
                "physics_joint_type": prim.GetTypeName(),
                "exclude_from_articulation": prim.GetAttribute(
                    "physics:excludeFromArticulation"
                ).Get(),
                "mimic_axis": axis,
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
    return validate_native_gripper_mimic_contract(result)


def validate_native_gripper_mimic_contract(value: Any) -> dict[str, Any]:
    expected = native_gripper_mimic_reference_contract()
    _require(value == expected, "native gripper source-USD mimic contract drift")
    return copy.deepcopy(value)


def native_gripper_mimic_reference_contract() -> dict[str, Any]:
    """Return the closed source-USD mimic identity for host-side validators."""

    root = "/panda/Gripper/Robotiq_2F_85/Joints"
    driver_path = f"{root}/finger_joint"
    return {
        "profile": NATIVE_GRIPPER_MIMIC_PROFILE,
        "robot_usd_sha256": EXPECTED_ROBOT_USD_SHA256,
        "driver_joint_name": "finger_joint",
        "driver_joint_index": DRIVER_JOINT_INDEX,
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
            for name, index, axis, gearing, frequency, damping in EXPECTED_MIMIC_SPECS
        ],
    }


def _failure_vector(value: Any, *, field: str) -> np.ndarray:
    _require(
        isinstance(value, list)
        and len(value) == 13
        and all(
            type(item) in (int, float)
            and not isinstance(item, bool)
            and math.isfinite(item)
            for item in value
        ),
        f"{field} must contain 13 finite numbers",
    )
    return np.asarray(value, dtype=np.float64)


def validate_native_all_joint_velocity_failure(value: Any) -> dict[str, Any]:
    """Validate and independently recompute one terminal all-DOF incident."""

    required = {
        "schema_version",
        "profile",
        "reason",
        "sample_kind",
        "joint_names",
        "joint_indices",
        "policy_step_index",
        "physics_substep_index",
        "failed_sample_index",
        "completed_apply_calls",
        "completed_post_policy_step_samples",
        "outer_step_physics_complete",
        "joint_position",
        "joint_velocity",
        "joint_acceleration",
        "joint_velocity_target",
        "joint_position_target",
        "absolute_joint_velocity",
        "expected_joint_velocity_limit",
        "live_joint_velocity_limit",
        "absolute_tolerance_rad_s",
        "effective_joint_velocity_threshold",
        "excess_mask",
        "excess_rad_s",
        "violating_joint_indices",
        "violating_joint_names",
    }
    _require(
        isinstance(value, dict) and set(value) == required,
        "all-joint velocity failure schema drift",
    )
    _require(
        value["schema_version"] == 2
        and value["profile"] == NATIVE_ALL_JOINT_VELOCITY_FAILURE_PROFILE
        and value["reason"] == "measured_all_joint_velocity_limit_exceeded"
        and value["sample_kind"] in {"apply_entry", "post_policy_step"}
        and value["joint_names"] == list(EXPECTED_DROID_JOINT_NAMES)
        and value["joint_indices"] == list(range(13)),
        "all-joint velocity failure identity drift",
    )
    indices = (
        value["policy_step_index"],
        value["physics_substep_index"],
        value["failed_sample_index"],
        value["completed_apply_calls"],
        value["completed_post_policy_step_samples"],
    )
    _require(
        all(type(item) is int and item >= 0 for item in indices),
        "all-joint velocity failure index drift",
    )
    sample_kind = value["sample_kind"]
    completed_apply = value["completed_apply_calls"]
    completed_post = value["completed_post_policy_step_samples"]
    _require(
        type(value["outer_step_physics_complete"]) is bool
        and value["failed_sample_index"] == completed_apply + completed_post
        and value["policy_step_index"] == completed_post,
        "all-joint velocity failure cadence drift",
    )
    if sample_kind == "apply_entry":
        _require(
            value["physics_substep_index"] in range(8)
            and value["outer_step_physics_complete"] is False
            and completed_apply == completed_post * 8 + value["physics_substep_index"],
            "all-joint apply-entry velocity failure cadence drift",
        )
    else:
        _require(
            value["physics_substep_index"] == 8
            and value["outer_step_physics_complete"] is True
            and completed_apply == (completed_post + 1) * 8,
            "all-joint post-policy velocity failure cadence drift",
        )
    vectors = {
        field: _failure_vector(value[field], field=field)
        for field in (
            "joint_position",
            "joint_velocity",
            "joint_acceleration",
            "joint_velocity_target",
            "joint_position_target",
            "absolute_joint_velocity",
            "expected_joint_velocity_limit",
            "live_joint_velocity_limit",
            "effective_joint_velocity_threshold",
            "excess_rad_s",
        )
    }
    expected_limits = np.asarray(EXPECTED_FULL_LIMITS_CAPPED, dtype=np.float32).astype(
        np.float64
    )
    _require(
        np.array_equal(vectors["expected_joint_velocity_limit"], expected_limits)
        and np.array_equal(vectors["live_joint_velocity_limit"], expected_limits),
        "all-joint velocity failure limit drift",
    )
    tolerance = value["absolute_tolerance_rad_s"]
    _require(
        type(tolerance) is float
        and tolerance == PHYSX_VELOCITY_LIMIT_ABSOLUTE_TOLERANCE_RAD_S,
        "all-joint velocity failure tolerance drift",
    )
    absolute_velocity = np.abs(vectors["joint_velocity"])
    thresholds = expected_limits + tolerance
    excess = np.maximum(absolute_velocity - thresholds, 0.0)
    mask = absolute_velocity > thresholds
    _require(
        np.array_equal(vectors["absolute_joint_velocity"], absolute_velocity)
        and np.array_equal(vectors["effective_joint_velocity_threshold"], thresholds)
        and np.array_equal(vectors["excess_rad_s"], excess),
        "all-joint velocity failure arithmetic drift",
    )
    _require(
        isinstance(value["excess_mask"], list)
        and len(value["excess_mask"]) == 13
        and all(type(item) is bool for item in value["excess_mask"])
        and value["excess_mask"] == mask.tolist()
        and bool(mask.any()),
        "all-joint velocity failure mask drift",
    )
    violating_indices = np.flatnonzero(mask).tolist()
    violating_names = [EXPECTED_DROID_JOINT_NAMES[index] for index in violating_indices]
    _require(
        value["violating_joint_indices"] == violating_indices
        and value["violating_joint_names"] == violating_names,
        "all-joint velocity failure joint identity drift",
    )
    return copy.deepcopy(value)


class NativeAllJointDynamicRecorder:
    """Record all 13 articulation DOFs at each apply entry and policy boundary."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.samples: list[dict[str, Any]] = []
        self.apply_calls = 0
        self.post_policy_step_samples = 0
        self._previous_velocity: np.ndarray | None = None
        self.max_abs_velocity = np.zeros(13, dtype=np.float64)
        self.max_abs_acceleration = np.zeros(13, dtype=np.float64)
        self.terminal_velocity_failure: dict[str, Any] | None = None
        self._failure_path: Path | None = None

    def bind_failure_path(self, path: Path) -> None:
        """Bind the non-overwriting incident destination after rollout reset."""

        _require(
            not self.samples
            and self.apply_calls == 0
            and self.post_policy_step_samples == 0
            and self.terminal_velocity_failure is None,
            "native all-joint failure path must be bound before sampling",
        )
        path = Path(path)
        _require(
            not path.exists() and not path.is_symlink(),
            "native all-joint failure path already exists",
        )
        self._failure_path = path

    def _sample(self, asset: Any, *, kind: str) -> dict[str, Any]:
        _require(
            kind in {"apply_entry", "post_policy_step"},
            "native all-joint sample kind drift",
        )
        _joint_names(asset)
        position, _, _ = _tensor_numpy(asset.data.joint_pos, field="all-joint position")
        velocity, _, _ = _tensor_numpy(asset.data.joint_vel, field="all-joint velocity")
        velocity_target, _, _ = _tensor_numpy(
            asset.data.joint_vel_target, field="all-joint velocity target"
        )
        position_target, _, _ = _tensor_numpy(
            asset.data.joint_pos_target, field="all-joint position target"
        )
        for field, array in (
            ("position", position),
            ("velocity", velocity),
            ("velocity target", velocity_target),
            ("position target", position_target),
        ):
            _require(array.shape == (1, 13), f"all-joint {field} shape drift")
        current_velocity = velocity[0].astype(np.float64)
        limits = np.asarray(EXPECTED_FULL_LIMITS_CAPPED, dtype=np.float64)
        live_limits, _, _ = _tensor_numpy(
            asset.data.joint_vel_limits, field="live all-joint velocity limit"
        )
        _require(
            live_limits.shape == (1, 13)
            and np.array_equal(
                live_limits[0],
                np.asarray(EXPECTED_FULL_LIMITS_CAPPED, dtype=np.float32),
            ),
            "live all-joint velocity limit drift",
        )
        acceleration = np.zeros(13, dtype=np.float64)
        if self._previous_velocity is not None:
            acceleration = (current_velocity - self._previous_velocity) / PHYSICS_DT
            _require(
                np.isfinite(acceleration).all(), "non-finite all-joint acceleration"
            )
        absolute_velocity = np.abs(current_velocity)
        thresholds = limits + PHYSX_VELOCITY_LIMIT_ABSOLUTE_TOLERANCE_RAD_S
        excess_mask = absolute_velocity > thresholds
        if bool(excess_mask.any()):
            _require(
                self.terminal_velocity_failure is None,
                "multiple all-joint terminal velocity failures",
            )
            violating_indices = np.flatnonzero(excess_mask).tolist()
            evidence = validate_native_all_joint_velocity_failure(
                {
                    "schema_version": 2,
                    "profile": NATIVE_ALL_JOINT_VELOCITY_FAILURE_PROFILE,
                    "reason": "measured_all_joint_velocity_limit_exceeded",
                    "sample_kind": kind,
                    "joint_names": list(EXPECTED_DROID_JOINT_NAMES),
                    "joint_indices": list(range(13)),
                    "policy_step_index": self.post_policy_step_samples,
                    "physics_substep_index": (
                        self.apply_calls % 8 if kind == "apply_entry" else 8
                    ),
                    "failed_sample_index": len(self.samples),
                    "completed_apply_calls": self.apply_calls,
                    "completed_post_policy_step_samples": (
                        self.post_policy_step_samples
                    ),
                    "outer_step_physics_complete": kind == "post_policy_step",
                    "joint_position": position[0].tolist(),
                    "joint_velocity": velocity[0].tolist(),
                    "joint_acceleration": acceleration.tolist(),
                    "joint_velocity_target": velocity_target[0].tolist(),
                    "joint_position_target": position_target[0].tolist(),
                    "absolute_joint_velocity": absolute_velocity.tolist(),
                    "expected_joint_velocity_limit": limits.tolist(),
                    "live_joint_velocity_limit": live_limits[0]
                    .astype(np.float64)
                    .tolist(),
                    "absolute_tolerance_rad_s": (
                        PHYSX_VELOCITY_LIMIT_ABSOLUTE_TOLERANCE_RAD_S
                    ),
                    "effective_joint_velocity_threshold": thresholds.tolist(),
                    "excess_mask": excess_mask.tolist(),
                    "excess_rad_s": np.maximum(
                        absolute_velocity - thresholds, 0.0
                    ).tolist(),
                    "violating_joint_indices": violating_indices,
                    "violating_joint_names": [
                        EXPECTED_DROID_JOINT_NAMES[index] for index in violating_indices
                    ],
                }
            )
            # Retain the closed object before any I/O or exception construction.
            self.terminal_velocity_failure = evidence
            incident_artifact = None
            if self._failure_path is not None:
                from polaris.pi05_droid_native_eval_contract import (  # noqa: PLC0415
                    publish_immutable_json,
                )

                incident_artifact = publish_immutable_json(self._failure_path, evidence)
                incident_artifact = {
                    key: incident_artifact[key]
                    for key in ("path", "size", "sha256", "mode", "nlink")
                }
            raise NativeAllJointVelocityLimitError(evidence, incident_artifact)
        self.max_abs_velocity = np.maximum(
            self.max_abs_velocity, np.abs(current_velocity)
        )
        if self._previous_velocity is not None:
            self.max_abs_acceleration = np.maximum(
                self.max_abs_acceleration, np.abs(acceleration)
            )
        self._previous_velocity = current_velocity.copy()
        sample = {
            "sample_index": len(self.samples),
            "kind": kind,
            "policy_step_index": self.post_policy_step_samples,
            "physics_substep_index": self.apply_calls % 8
            if kind == "apply_entry"
            else 8,
            "joint_position": position[0].tolist(),
            "joint_velocity": velocity[0].tolist(),
            "joint_acceleration": acceleration.tolist(),
            "joint_velocity_target": velocity_target[0].tolist(),
            "joint_position_target": position_target[0].tolist(),
        }
        self.samples.append(sample)
        return sample

    def record_apply_entry(self, asset: Any) -> None:
        self._sample(asset, kind="apply_entry")
        self.apply_calls += 1

    def record_post_policy_step(self, asset: Any) -> dict[str, Any]:
        _require(
            self.apply_calls == (self.post_policy_step_samples + 1) * 8,
            "native all-six apply/policy cadence drift",
        )
        sample = self._sample(asset, kind="post_policy_step")
        self.post_policy_step_samples += 1
        return sample

    def report(self, *, include_samples: bool) -> dict[str, Any]:
        expected_samples = self.apply_calls + self.post_policy_step_samples
        _require(len(self.samples) == expected_samples, "dynamic sample count drift")
        value = {
            "schema_version": 3,
            "profile": NATIVE_GRIPPER_DYNAMIC_PROFILE,
            "joint_names": list(EXPECTED_DROID_JOINT_NAMES),
            "joint_indices": list(range(13)),
            "apply_calls": self.apply_calls,
            "post_policy_step_samples": self.post_policy_step_samples,
            "sample_count": len(self.samples),
            "max_abs_joint_velocity_rad_s": self.max_abs_velocity.tolist(),
            "max_abs_joint_acceleration_rad_s2": self.max_abs_acceleration.tolist(),
            "terminal_velocity_failure": copy.deepcopy(self.terminal_velocity_failure),
            "samples": copy.deepcopy(self.samples) if include_samples else None,
        }
        return validate_native_all_joint_dynamic_report(
            value, require_samples=include_samples
        )


def validate_native_all_joint_dynamic_report(
    value: Any, *, require_samples: bool
) -> dict[str, Any]:
    required = {
        "schema_version",
        "profile",
        "joint_names",
        "joint_indices",
        "apply_calls",
        "post_policy_step_samples",
        "sample_count",
        "max_abs_joint_velocity_rad_s",
        "max_abs_joint_acceleration_rad_s2",
        "terminal_velocity_failure",
        "samples",
    }
    _require(
        isinstance(value, dict) and set(value) == required,
        "dynamic report schema drift",
    )
    _require(
        value["schema_version"] == 3
        and value["profile"] == NATIVE_GRIPPER_DYNAMIC_PROFILE
        and value["joint_names"] == list(EXPECTED_DROID_JOINT_NAMES)
        and value["joint_indices"] == list(range(13)),
        "dynamic report identity drift",
    )
    apply_calls = value["apply_calls"]
    policy_samples = value["post_policy_step_samples"]
    sample_count = value["sample_count"]
    _require(
        all(
            type(item) is int and item >= 0
            for item in (apply_calls, policy_samples, sample_count)
        )
        and sample_count == apply_calls + policy_samples,
        "dynamic report cadence drift",
    )
    terminal_failure = value["terminal_velocity_failure"]
    if terminal_failure is None:
        _require(
            apply_calls == policy_samples * 8,
            "healthy dynamic report cadence drift",
        )
    else:
        failure = validate_native_all_joint_velocity_failure(terminal_failure)
        _require(
            failure["completed_apply_calls"] == apply_calls
            and failure["completed_post_policy_step_samples"] == policy_samples,
            "failed dynamic report cadence drift",
        )
        if failure["sample_kind"] == "apply_entry":
            _require(
                apply_calls == policy_samples * 8 + failure["physics_substep_index"],
                "failed apply-entry dynamic report cadence drift",
            )
        else:
            _require(
                apply_calls == (policy_samples + 1) * 8,
                "failed post-policy dynamic report cadence drift",
            )
    for field in (
        "max_abs_joint_velocity_rad_s",
        "max_abs_joint_acceleration_rad_s2",
    ):
        vector = value[field]
        _require(
            isinstance(vector, list)
            and len(vector) == 13
            and all(
                type(item) in (int, float)
                and not isinstance(item, bool)
                and math.isfinite(item)
                and item >= 0.0
                for item in vector
            ),
            f"dynamic report {field} drift",
        )
    limits = EXPECTED_FULL_LIMITS_CAPPED
    _require(
        all(
            measured <= limit + PHYSX_VELOCITY_LIMIT_ABSOLUTE_TOLERANCE_RAD_S
            for measured, limit in zip(
                value["max_abs_joint_velocity_rad_s"], limits, strict=True
            )
        ),
        "dynamic report exceeded all-joint velocity limits",
    )
    samples = value["samples"]
    if not require_samples:
        _require(
            samples is None, "aggregate dynamic report unexpectedly contains samples"
        )
        return copy.deepcopy(value)
    _require(
        isinstance(samples, list) and len(samples) == sample_count,
        "dynamic sample set drift",
    )
    expected_cadence = []
    for policy_step_index in range(policy_samples):
        expected_cadence.extend(
            ("apply_entry", policy_step_index, physics_substep_index)
            for physics_substep_index in range(8)
        )
        expected_cadence.append(("post_policy_step", policy_step_index, 8))
    if terminal_failure is not None:
        trailing_apply_samples = (
            terminal_failure["physics_substep_index"]
            if terminal_failure["sample_kind"] == "apply_entry"
            else 8
        )
        expected_cadence.extend(
            ("apply_entry", policy_samples, physics_substep_index)
            for physics_substep_index in range(trailing_apply_samples)
        )
    _require(
        len(expected_cadence) == sample_count,
        "dynamic exact sample cadence length drift",
    )
    previous_policy = 0
    for index, (sample, expected_sample_cadence) in enumerate(
        zip(samples, expected_cadence, strict=True)
    ):
        _require(
            isinstance(sample, dict)
            and set(sample)
            == {
                "sample_index",
                "kind",
                "policy_step_index",
                "physics_substep_index",
                "joint_position",
                "joint_velocity",
                "joint_acceleration",
                "joint_velocity_target",
                "joint_position_target",
            },
            f"dynamic sample {index} schema drift",
        )
        _require(sample["sample_index"] == index, "dynamic sample index drift")
        kind = sample["kind"]
        _require(
            kind in {"apply_entry", "post_policy_step"}, "dynamic sample kind drift"
        )
        _require(
            (
                kind,
                sample["policy_step_index"],
                sample["physics_substep_index"],
            )
            == expected_sample_cadence,
            "dynamic exact sample cadence drift",
        )
        if kind == "apply_entry":
            _require(
                sample["physics_substep_index"] in range(8),
                "dynamic apply substep drift",
            )
        else:
            _require(
                sample["physics_substep_index"] == 8, "dynamic post-step index drift"
            )
            previous_policy += 1
        _require(
            sample["policy_step_index"]
            == previous_policy - (kind == "post_policy_step"),
            "dynamic sample policy index drift",
        )
        for field in (
            "joint_position",
            "joint_velocity",
            "joint_acceleration",
            "joint_velocity_target",
            "joint_position_target",
        ):
            vector = sample[field]
            _require(
                isinstance(vector, list)
                and len(vector) == 13
                and all(
                    type(item) in (int, float)
                    and not isinstance(item, bool)
                    and math.isfinite(item)
                    for item in vector
                ),
                f"dynamic sample {index} {field} drift",
            )
    return copy.deepcopy(value)


def validate_actuator_ownership(robot: Any) -> dict[str, Any]:
    _joint_names(robot)
    expected = {
        "panda_shoulder": ([f"panda_joint{i}" for i in range(1, 5)], [0, 1, 2, 3]),
        "panda_forearm": ([f"panda_joint{i}" for i in range(5, 8)], [4, 5, 6]),
        "gripper": (["finger_joint"], [7]),
    }
    _require(isinstance(robot.actuators, Mapping), "actuator mapping missing")
    _require(set(robot.actuators) == set(expected), "actuator set drift")
    report = {}
    for name, (expected_names, expected_indices) in expected.items():
        actuator = robot.actuators[name]
        names = list(getattr(actuator, "joint_names", ()))
        indices_value = getattr(actuator, "joint_indices", ())
        try:
            indices = indices_value.detach().cpu().tolist()
        except AttributeError:
            indices = list(indices_value)
        _require(
            names == expected_names and indices == expected_indices,
            f"actuator ownership drift: {name}",
        )
        report[name] = {"joint_names": names, "joint_indices": indices}
    return report
