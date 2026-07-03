"""Closed runtime evidence for the production EEF Robotiq gripper profile.

The physical articulation contains one driven ``finger_joint`` and five
passive PhysX mimic joints.  Isaac Lab exposes static PhysX drive properties on
CPU while cached articulation state and implicit-actuator tensors live on CUDA.
This module preserves that distinction and performs the one production write
needed by the passive followers.  It deliberately does not claim that a PhysX
maximum-velocity setting is a hard bound on measured passive-joint velocity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
from pathlib import Path
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


def validate_eef_gripper_target_slew_static(value: Any) -> dict[str, Any]:
    """Validate the closed EEF-only driver target-slew identity."""

    _require(
        isinstance(value, dict) and set(value) == TARGET_SLEW_STATIC_FIELDS,
        "gripper target-slew static schema",
    )
    exact = {
        "profile": EEF_GRIPPER_TARGET_SLEW_PROFILE,
        "scope": "eef_pose_only_native_joint_position_unchanged_v1",
        "action_class": EEF_GRIPPER_TARGET_SLEW_ACTION_CLASS,
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
        "target_slew_rate_factor": GRIPPER_TARGET_SLEW_RATE_FACTOR_FLOAT32,
        "target_slew_rate_rad_s": GRIPPER_TARGET_SLEW_RATE_RAD_S_FLOAT32,
        "physics_hz": GRIPPER_TARGET_SLEW_PHYSICS_HZ,
        "physics_dt": GRIPPER_TARGET_SLEW_PHYSICS_DT,
        "max_target_step_rad": GRIPPER_MAX_TARGET_STEP_FLOAT32,
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


def validate_eef_gripper_target_slew_dynamic(value: Any) -> dict[str, Any]:
    """Validate one reset-isolated target-slew counter/maximum report."""

    _require(
        isinstance(value, dict) and set(value) == TARGET_SLEW_DYNAMIC_FIELDS,
        "gripper target-slew dynamic schema",
    )
    _require(
        value.get("profile") == EEF_GRIPPER_TARGET_SLEW_PROFILE,
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
        <= GRIPPER_MAX_TARGET_STEP_FLOAT32 + GRIPPER_TARGET_SLEW_FLOAT32_TOLERANCE_RAD,
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


def validate_eef_gripper_static_contract(value: Any) -> dict[str, Any]:
    _require(
        isinstance(value, dict) and set(value) == STATIC_CONTRACT_FIELDS,
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
    validate_mimic_joint_contract(value.get("mimic_joint_contract"))
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
    validate_eef_gripper_target_slew_static(value.get("driver_target_slew"))
    return dict(value)


def validate_eef_gripper_dynamic_evidence(value: Any) -> dict[str, Any]:
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
    validate_eef_gripper_target_slew_dynamic(value.get("driver_target_slew"))
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
    ownership = _validate_live_ownership(robot, arm_term, finger_term)
    driver = _capture_driver_actuator(robot)
    mimic = _capture_mimic_joint_contract(Path(robot_usd_path))
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
        "driver_target_slew": target_slew_reporter(),
        "measured_velocity_is_hard_bounded_by_limit": False,
    }
    contract = validate_eef_gripper_static_contract(contract)
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

    validate_eef_gripper_static_contract(dict(expected_contract))
    runtime = getattr(env, "unwrapped", env)
    robot = runtime.scene["robot"]
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
    dynamic = validate_eef_gripper_target_slew_dynamic(dynamic_reporter())
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
