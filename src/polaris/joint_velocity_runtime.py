"""Live Isaac-Lab attestation for the native DROID velocity controller."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np

from polaris.pi05_droid_jointvelocity_contract import (
    PANDA_ARM_EFFORT_LIMITS,
    PANDA_ARM_JOINT_NAMES,
    PANDA_ARM_VELOCITY_DRIVE_DAMPING,
    PANDA_ARM_VELOCITY_DRIVE_STIFFNESS,
    PANDA_ARM_VELOCITY_LIMITS,
    PI05_DROID_ISAACLAB_SOURCE_SHA256,
    PI05_DROID_ISAACLAB_VERSION,
    PI05_DROID_JOINTVELOCITY_PROFILE,
    PI05_DROID_POLARIS_RUNTIME_SOURCE_SHA256,
)


JOINT_VELOCITY_RUNTIME_MARKER = "POLARIS_JOINT_VELOCITY_RUNTIME="
_JOINT_VELOCITY_ACTION_CLASS = (
    "isaaclab.envs.mdp.actions.joint_actions.JointVelocityAction"
)
_JOINT_VELOCITY_CFG_CLASS = (
    "isaaclab.envs.mdp.actions.actions_cfg.JointVelocityActionCfg"
)
_IMPLICIT_ACTUATOR_CLASS = "isaaclab.actuators.actuator_pd.ImplicitActuator"
_BINARY_GRIPPER_ACTION_CLASS = (
    "polaris.environments.droid_cfg.BinaryJointPositionZeroToOneAction"
)


def _class_path(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _to_numpy(value: Any, *, field: str) -> tuple[np.ndarray, str, str]:
    dtype = str(getattr(value, "dtype", ""))
    device = str(getattr(value, "device", ""))
    try:
        value = value.detach().cpu().numpy()
    except AttributeError:
        value = np.asarray(value)
    result = np.asarray(value)
    if not np.issubdtype(result.dtype, np.number) or not np.isfinite(result).all():
        raise ValueError(f"{field} must be finite and numeric")
    return result, dtype, device


def _require_float32_array(
    value: Any, expected: np.ndarray, *, field: str
) -> dict[str, Any]:
    actual, source_dtype, source_device = _to_numpy(value, field=field)
    if source_dtype != "torch.float32" or actual.dtype != np.float32:
        raise ValueError(
            f"{field} must be a live torch.float32 tensor, "
            f"got source={source_dtype}, numpy={actual.dtype}"
        )
    expected = np.asarray(expected, dtype=np.float32)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise ValueError(
            f"{field} mismatch: expected {expected.tolist()}, got {actual.tolist()}"
        )
    return {
        "shape": list(actual.shape),
        "dtype": source_dtype,
        "device": source_device,
        "values": actual.tolist(),
    }


def _canonical_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _validate_array_report(report: Any, expected: np.ndarray, *, field: str) -> None:
    if not isinstance(report, dict) or set(report) != {
        "shape",
        "dtype",
        "device",
        "values",
    }:
        raise ValueError(f"{field} report schema mismatch")
    expected = np.asarray(expected, dtype=np.float32)
    if report["shape"] != list(expected.shape):
        raise ValueError(f"{field} report shape mismatch")
    if report["dtype"] != "torch.float32":
        raise ValueError(f"{field} report must attest torch.float32")
    if not isinstance(report["device"], str) or not report["device"]:
        raise ValueError(f"{field} report device is missing")
    actual = np.asarray(report["values"])
    if (
        actual.shape != expected.shape
        or not np.issubdtype(actual.dtype, np.number)
        or np.issubdtype(actual.dtype, np.bool_)
        or not np.isfinite(actual).all()
    ):
        raise ValueError(f"{field} report values are invalid")
    if not np.array_equal(actual.astype(np.float32), expected):
        raise ValueError(f"{field} report values mismatch")


def validate_joint_velocity_runtime_report(report: Any) -> dict[str, Any]:
    """Recompute and validate the complete live runtime contract document."""

    if not isinstance(report, dict):
        raise ValueError("Joint-velocity runtime contract must be an object")
    required_keys = {
        "schema_version",
        "profile",
        "status",
        "isaaclab_version",
        "isaaclab_source_sha256",
        "polaris_runtime_source_sha256",
        "policy_frequency_hz",
        "physics_frequency_hz",
        "decimation",
        "joint_names",
        "action_term_class",
        "action_cfg_class",
        "scale",
        "offset",
        "use_default_offset",
        "clip",
        "position_integration",
        "velocity_drive",
        "gripper",
        "runtime_sha256",
    }
    if set(report) != required_keys:
        raise ValueError("Joint-velocity runtime contract schema mismatch")
    payload = copy.deepcopy(report)
    claimed_digest = payload.pop("runtime_sha256")
    if (
        not isinstance(claimed_digest, str)
        or len(claimed_digest) != 64
        or any(character not in "0123456789abcdef" for character in claimed_digest)
    ):
        raise ValueError("Joint-velocity runtime SHA-256 is malformed")
    if claimed_digest != _canonical_sha256(payload):
        raise ValueError("Joint-velocity runtime contract SHA-256 mismatch")
    expected_scalars = {
        "schema_version": 1,
        "profile": PI05_DROID_JOINTVELOCITY_PROFILE,
        "status": "pass",
        "isaaclab_version": PI05_DROID_ISAACLAB_VERSION,
        "policy_frequency_hz": 15,
        "physics_frequency_hz": 120,
        "decimation": 8,
        "joint_names": list(PANDA_ARM_JOINT_NAMES),
        "action_term_class": _JOINT_VELOCITY_ACTION_CLASS,
        "action_cfg_class": _JOINT_VELOCITY_CFG_CLASS,
        "scale": 1.0,
        "offset": 0.0,
        "use_default_offset": False,
        "position_integration": "absent_by_exact_action_class",
    }
    for key, expected in expected_scalars.items():
        if report[key] != expected or type(report[key]) is not type(expected):
            raise ValueError(f"Joint-velocity runtime contract {key} mismatch")
    if report["isaaclab_source_sha256"] != PI05_DROID_ISAACLAB_SOURCE_SHA256:
        raise ValueError("Joint-velocity runtime Isaac source manifest mismatch")
    if (
        report["polaris_runtime_source_sha256"]
        != PI05_DROID_POLARIS_RUNTIME_SOURCE_SHA256
    ):
        raise ValueError("Joint-velocity runtime PolaRiS source manifest mismatch")
    _validate_array_report(
        report["clip"],
        np.broadcast_to(np.asarray([-1.0, 1.0], dtype=np.float32), (1, 7, 2)),
        field="action clip",
    )
    velocity_drive = report["velocity_drive"]
    if not isinstance(velocity_drive, dict) or set(velocity_drive) != {
        "position_stiffness",
        "velocity_damping",
        "buffered",
        "direct_physx",
    }:
        raise ValueError("Joint-velocity drive report schema mismatch")
    if (
        velocity_drive["position_stiffness"] != PANDA_ARM_VELOCITY_DRIVE_STIFFNESS
        or type(velocity_drive["position_stiffness"]) is not float
    ):
        raise ValueError("Joint-velocity drive stiffness mismatch")
    if (
        velocity_drive["velocity_damping"] != PANDA_ARM_VELOCITY_DRIVE_DAMPING
        or type(velocity_drive["velocity_damping"]) is not float
    ):
        raise ValueError("Joint-velocity drive damping mismatch")
    expected_arrays = {
        "stiffness": np.zeros((1, 7), dtype=np.float32),
        "damping": np.full((1, 7), PANDA_ARM_VELOCITY_DRIVE_DAMPING, dtype=np.float32),
        "effort_limit": np.asarray([PANDA_ARM_EFFORT_LIMITS], dtype=np.float32),
        "velocity_limit": np.asarray([PANDA_ARM_VELOCITY_LIMITS], dtype=np.float32),
    }
    for surface in ("buffered", "direct_physx"):
        surface_report = velocity_drive[surface]
        if not isinstance(surface_report, dict) or set(surface_report) != set(
            expected_arrays
        ):
            raise ValueError(f"Joint-velocity {surface} report schema mismatch")
        for name, expected in expected_arrays.items():
            _validate_array_report(
                surface_report[name], expected, field=f"{surface} {name}"
            )
    gripper = report["gripper"]
    if not isinstance(gripper, dict) or set(gripper) != {
        "action_class",
        "joint_name",
        "threshold",
        "open_command",
        "closed_command",
    }:
        raise ValueError("Joint-velocity gripper report schema mismatch")
    if gripper["action_class"] != _BINARY_GRIPPER_ACTION_CLASS:
        raise ValueError("Joint-velocity gripper action class mismatch")
    if gripper["joint_name"] != "finger_joint":
        raise ValueError("Joint-velocity gripper joint mismatch")
    if gripper["threshold"] != "closed_if_gt_0p5_else_open":
        raise ValueError("Joint-velocity gripper threshold mismatch")
    _validate_array_report(
        gripper["open_command"],
        np.zeros((1, 1), dtype=np.float32),
        field="gripper open command",
    )
    _validate_array_report(
        gripper["closed_command"],
        np.full((1, 1), np.pi / 4.0, dtype=np.float32),
        field="gripper closed command",
    )
    return copy.deepcopy(report)


def _installed_isaaclab_version() -> str:
    return importlib.metadata.version("isaaclab")


def _verify_isaaclab_sources(
    *, arm_term: Any, robot: Any, actuator: Any
) -> dict[str, str]:
    objects = {
        "actions_cfg.py": arm_term.cfg,
        "joint_actions.py": arm_term,
        "actuator_cfg.py": robot.cfg.actuators["panda_shoulder"],
        "actuator_pd.py": actuator,
        "articulation.py": robot,
    }
    actual: dict[str, str] = {}
    for source_name, value in objects.items():
        source_path_string = inspect.getsourcefile(type(value))
        if source_path_string is None:
            raise ValueError(f"Cannot resolve Isaac Lab source for {source_name}")
        source_path = Path(source_path_string)
        if not source_path.is_file() or source_path.is_symlink():
            raise ValueError(f"Isaac Lab source is not a regular file: {source_path}")
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        expected = PI05_DROID_ISAACLAB_SOURCE_SHA256[source_name]
        if digest != expected:
            raise ValueError(
                f"Isaac Lab 2.3 source mismatch for {source_name}: {digest}"
            )
        actual[source_name] = digest
    return actual


def _verify_polaris_sources(*, finger_term: Any) -> dict[str, str]:
    source_path_string = inspect.getsourcefile(type(finger_term))
    if source_path_string is None:
        raise ValueError("Cannot resolve PolaRiS gripper action source")
    source_path = Path(source_path_string)
    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError(f"PolaRiS runtime source is not regular: {source_path}")
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    expected = PI05_DROID_POLARIS_RUNTIME_SOURCE_SHA256["droid_cfg.py"]
    if digest != expected:
        raise ValueError(f"PolaRiS DROID action source mismatch: {digest}")
    return {"droid_cfg.py": digest}


def validate_joint_velocity_runtime(env: Any) -> dict[str, Any]:
    """Fail closed unless action and PhysX drive semantics match the profile."""

    root_env = getattr(env, "unwrapped", env)
    isaaclab_version = _installed_isaaclab_version()
    if isaaclab_version != PI05_DROID_ISAACLAB_VERSION:
        raise ValueError(
            f"Native DROID velocity control requires Isaac Lab "
            f"{PI05_DROID_ISAACLAB_VERSION}; got {isaaclab_version}"
        )
    if root_env.cfg.decimation != 8 or root_env.cfg.sim.dt != 1.0 / 120.0:
        raise ValueError(
            "Native DROID velocity control requires decimation=8 and dt=1/120"
        )

    arm_term = root_env.action_manager._terms["arm"]
    finger_term = root_env.action_manager._terms["finger_joint"]
    if _class_path(arm_term) != _JOINT_VELOCITY_ACTION_CLASS:
        raise ValueError(
            "Arm action term must be Isaac Lab JointVelocityAction; "
            f"got {_class_path(arm_term)}"
        )
    if _class_path(arm_term.cfg) != _JOINT_VELOCITY_CFG_CLASS:
        raise ValueError(
            "Arm action config must be JointVelocityActionCfg; "
            f"got {_class_path(arm_term.cfg)}"
        )
    if tuple(arm_term._joint_names) != PANDA_ARM_JOINT_NAMES:
        raise ValueError(f"Action-term joint order mismatch: {arm_term._joint_names}")
    if arm_term.cfg.preserve_order is not True:
        raise ValueError("JointVelocityActionCfg.preserve_order must be true")
    if arm_term.cfg.use_default_offset is not False:
        raise ValueError("JointVelocityActionCfg.use_default_offset must be false")
    if arm_term._scale != 1.0 or arm_term._offset != 0.0:
        raise ValueError(
            f"Velocity action affine mismatch: scale={arm_term._scale}, offset={arm_term._offset}"
        )
    expected_clip = np.broadcast_to(
        np.asarray([-1.0, 1.0], dtype=np.float32), (1, 7, 2)
    )
    clip_report = _require_float32_array(
        arm_term._clip, expected_clip, field="action term clip"
    )
    if _class_path(finger_term) != _BINARY_GRIPPER_ACTION_CLASS:
        raise ValueError(
            "Gripper action term must preserve closed-positive binary semantics; "
            f"got {_class_path(finger_term)}"
        )
    if tuple(finger_term._joint_names) != ("finger_joint",):
        raise ValueError(f"Gripper joint mismatch: {finger_term._joint_names}")
    gripper_open = _require_float32_array(
        finger_term._open_command,
        np.zeros((1, 1), dtype=np.float32),
        field="gripper open command",
    )
    gripper_closed = _require_float32_array(
        finger_term._close_command,
        np.full((1, 1), np.pi / 4.0, dtype=np.float32),
        field="gripper closed command",
    )

    robot = root_env.scene["robot"]
    joint_ids, joint_names = robot.find_joints(
        list(PANDA_ARM_JOINT_NAMES), preserve_order=True
    )
    if tuple(joint_names) != PANDA_ARM_JOINT_NAMES:
        raise ValueError(f"Articulation joint order mismatch: {joint_names}")

    for actuator_name in ("panda_shoulder", "panda_forearm"):
        actuator = robot.actuators[actuator_name]
        if _class_path(actuator) != _IMPLICIT_ACTUATOR_CLASS:
            raise ValueError(
                f"{actuator_name} must use ImplicitActuator; got {_class_path(actuator)}"
            )
    isaaclab_source_sha256 = _verify_isaaclab_sources(
        arm_term=arm_term,
        robot=robot,
        actuator=robot.actuators["panda_shoulder"],
    )
    polaris_runtime_source_sha256 = _verify_polaris_sources(finger_term=finger_term)

    expected_stiffness = np.zeros((1, 7), dtype=np.float32)
    expected_damping = np.full(
        (1, 7), PANDA_ARM_VELOCITY_DRIVE_DAMPING, dtype=np.float32
    )
    expected_effort = np.asarray([PANDA_ARM_EFFORT_LIMITS], dtype=np.float32)
    expected_velocity = np.asarray([PANDA_ARM_VELOCITY_LIMITS], dtype=np.float32)
    data = robot.data
    buffered = {
        "stiffness": _require_float32_array(
            data.joint_stiffness[:, joint_ids],
            expected_stiffness,
            field="buffered joint stiffness",
        ),
        "damping": _require_float32_array(
            data.joint_damping[:, joint_ids],
            expected_damping,
            field="buffered joint damping",
        ),
        "effort_limit": _require_float32_array(
            data.joint_effort_limits[:, joint_ids],
            expected_effort,
            field="buffered joint effort limits",
        ),
        "velocity_limit": _require_float32_array(
            data.joint_vel_limits[:, joint_ids],
            expected_velocity,
            field="buffered joint velocity limits",
        ),
    }

    view = robot.root_physx_view
    direct = {
        "stiffness": _require_float32_array(
            view.get_dof_stiffnesses()[:, joint_ids],
            expected_stiffness,
            field="direct PhysX joint stiffness",
        ),
        "damping": _require_float32_array(
            view.get_dof_dampings()[:, joint_ids],
            expected_damping,
            field="direct PhysX joint damping",
        ),
        "effort_limit": _require_float32_array(
            view.get_dof_max_forces()[:, joint_ids],
            expected_effort,
            field="direct PhysX joint effort limits",
        ),
        "velocity_limit": _require_float32_array(
            view.get_dof_max_velocities()[:, joint_ids],
            expected_velocity,
            field="direct PhysX joint velocity limits",
        ),
    }

    report: dict[str, Any] = {
        "schema_version": 1,
        "profile": PI05_DROID_JOINTVELOCITY_PROFILE,
        "status": "pass",
        "isaaclab_version": isaaclab_version,
        "isaaclab_source_sha256": isaaclab_source_sha256,
        "polaris_runtime_source_sha256": polaris_runtime_source_sha256,
        "policy_frequency_hz": 15,
        "physics_frequency_hz": 120,
        "decimation": 8,
        "joint_names": list(joint_names),
        "action_term_class": _class_path(arm_term),
        "action_cfg_class": _class_path(arm_term.cfg),
        "scale": 1.0,
        "offset": 0.0,
        "use_default_offset": False,
        "clip": clip_report,
        "position_integration": "absent_by_exact_action_class",
        "velocity_drive": {
            "position_stiffness": PANDA_ARM_VELOCITY_DRIVE_STIFFNESS,
            "velocity_damping": PANDA_ARM_VELOCITY_DRIVE_DAMPING,
            "buffered": buffered,
            "direct_physx": direct,
        },
        "gripper": {
            "action_class": _class_path(finger_term),
            "joint_name": "finger_joint",
            "threshold": "closed_if_gt_0p5_else_open",
            "open_command": gripper_open,
            "closed_command": gripper_closed,
        },
    }
    report["runtime_sha256"] = _canonical_sha256(report)
    return validate_joint_velocity_runtime_report(report)


def print_joint_velocity_runtime(report: dict[str, Any]) -> None:
    print(
        JOINT_VELOCITY_RUNTIME_MARKER
        + json.dumps(report, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
