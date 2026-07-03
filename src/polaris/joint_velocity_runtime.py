"""Live Isaac-Lab attestation for the native DROID velocity controller."""

from __future__ import annotations

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
    if source_dtype not in {"torch.float32", "float32"} or actual.dtype != np.float32:
        raise ValueError(
            f"{field} must be live float32, got source={source_dtype}, numpy={actual.dtype}"
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
    return report


def print_joint_velocity_runtime(report: dict[str, Any]) -> None:
    print(
        JOINT_VELOCITY_RUNTIME_MARKER
        + json.dumps(report, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
