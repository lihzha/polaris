"""Live Isaac-Lab attestation for the native DROID velocity controller."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from polaris.native_gripper_runtime import (
    EXPECTED_DROID_JOINT_NAMES,
    EXPECTED_FULL_LIMITS_CAPPED,
    GRIPPER_JOINT_INDICES,
    GRIPPER_JOINT_NAMES,
    NATIVE_GRIPPER_ALL_SIX_PROFILE,
    capture_native_gripper_mimic_contract,
    native_gripper_reset_report,
    validate_actuator_ownership,
    validate_native_gripper_mimic_contract,
    validate_native_gripper_reset_report,
)
from polaris.pi05_droid_jointvelocity_contract import (
    NATIVE_GRIPPER_DAMPING,
    NATIVE_GRIPPER_DRIVE_PROFILE,
    NATIVE_GRIPPER_EFFORT_LIMIT,
    NATIVE_GRIPPER_STIFFNESS,
    NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S,
    PANDA_ARM_EFFORT_LIMITS,
    PANDA_ARM_JOINT_NAMES,
    PANDA_ARM_VELOCITY_DRIVE_DAMPING,
    PANDA_ARM_VELOCITY_DRIVE_STIFFNESS,
    PANDA_ARM_VELOCITY_LIMITS,
    PI05_DROID_GRIPPER_OBSERVATION_CONTRACT,
    PI05_DROID_ISAACLAB_SOURCE_SHA256,
    PI05_DROID_ISAACLAB_VERSION,
    PI05_DROID_JOINTVELOCITY_PROFILE,
    PI05_DROID_POLARIS_RUNTIME_SOURCE_SHA256,
)


JOINT_VELOCITY_RUNTIME_MARKER = "POLARIS_JOINT_VELOCITY_RUNTIME="
_JOINT_VELOCITY_ACTION_CLASS = (
    "polaris.environments.droid_cfg.AuditedDroidJointVelocityAction"
)
_JOINT_VELOCITY_CFG_CLASS = (
    "polaris.environments.droid_cfg.AuditedDroidJointVelocityActionCfg"
)
_JOINT_VELOCITY_BASE_CLASS = (
    "isaaclab.envs.mdp.actions.joint_actions.JointVelocityAction"
)
_JOINT_VELOCITY_CFG_BASE_CLASS = (
    "isaaclab.envs.mdp.actions.actions_cfg.JointVelocityActionCfg"
)
_MANAGER_BASED_RL_ENV_BASE_CLASS = (
    "isaaclab.envs.manager_based_rl_env.ManagerBasedRLEnv"
)
_IMPLICIT_ACTUATOR_CLASS = "isaaclab.actuators.actuator_pd.ImplicitActuator"
_BINARY_GRIPPER_ACTION_CLASS = (
    "polaris.environments.droid_cfg.BinaryJointPositionZeroToOneAction"
)
_BINARY_GRIPPER_BASE_CLASS = (
    "isaaclab.envs.mdp.actions.binary_joint_actions.BinaryJointPositionAction"
)
_CUDA_DEVICE = "cuda:0"
_CPU_DEVICE = "cpu"


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
    value: Any, expected: np.ndarray, *, field: str, expected_device: str
) -> dict[str, Any]:
    actual, source_dtype, source_device = _to_numpy(value, field=field)
    if source_dtype != "torch.float32" or actual.dtype != np.float32:
        raise ValueError(
            f"{field} must be a live torch.float32 tensor, "
            f"got source={source_dtype}, numpy={actual.dtype}"
        )
    if source_device != expected_device:
        raise ValueError(
            f"{field} must be on {expected_device}, got {source_device or '<missing>'}"
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


def _validate_array_report(
    report: Any, expected: np.ndarray, *, field: str, expected_device: str
) -> None:
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
    if report["device"] != expected_device:
        raise ValueError(f"{field} report must attest device {expected_device}")
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
        "gripper_observation",
        "reset_event_order",
        "joint_names",
        "action_term_class",
        "action_cfg_class",
        "action_cfg_base_class",
        "scale",
        "offset",
        "use_default_offset",
        "clip",
        "action_buffers",
        "position_integration",
        "velocity_drive",
        "gripper",
        "all_six_gripper",
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
        "reset_event_order": ["reset_all", "cap_gripper_followers"],
        "joint_names": list(PANDA_ARM_JOINT_NAMES),
        "action_term_class": _JOINT_VELOCITY_ACTION_CLASS,
        "action_cfg_class": _JOINT_VELOCITY_CFG_CLASS,
        "action_cfg_base_class": _JOINT_VELOCITY_CFG_BASE_CLASS,
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
    if report["gripper_observation"] != PI05_DROID_GRIPPER_OBSERVATION_CONTRACT:
        raise ValueError("Joint-velocity gripper observation contract mismatch")
    _validate_array_report(
        report["clip"],
        np.broadcast_to(np.asarray([-1.0, 1.0], dtype=np.float32), (1, 7, 2)),
        field="action clip",
        expected_device=_CUDA_DEVICE,
    )
    action_buffers = report["action_buffers"]
    if not isinstance(action_buffers, dict) or set(action_buffers) != {
        "raw_action",
        "processed_action",
    }:
        raise ValueError("Joint-velocity arm action-buffer schema mismatch")
    for name in ("raw_action", "processed_action"):
        _validate_array_report(
            action_buffers[name],
            np.zeros((1, 7), dtype=np.float32),
            field=f"arm {name}",
            expected_device=_CUDA_DEVICE,
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
    for surface, expected_device in (
        ("buffered", _CUDA_DEVICE),
        ("direct_physx", _CPU_DEVICE),
    ):
        surface_report = velocity_drive[surface]
        if not isinstance(surface_report, dict) or set(surface_report) != set(
            expected_arrays
        ):
            raise ValueError(f"Joint-velocity {surface} report schema mismatch")
        for name, expected in expected_arrays.items():
            _validate_array_report(
                surface_report[name],
                expected,
                field=f"{surface} {name}",
                expected_device=expected_device,
            )
    gripper = report["gripper"]
    if not isinstance(gripper, dict) or set(gripper) != {
        "action_class",
        "joint_name",
        "threshold",
        "open_command",
        "closed_command",
        "raw_action",
        "processed_action",
        "drive",
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
        np.zeros((1,), dtype=np.float32),
        field="gripper open command",
        expected_device=_CUDA_DEVICE,
    )
    _validate_array_report(
        gripper["closed_command"],
        np.full((1,), np.pi / 4.0, dtype=np.float32),
        field="gripper closed command",
        expected_device=_CUDA_DEVICE,
    )
    for name in ("raw_action", "processed_action"):
        _validate_array_report(
            gripper[name],
            np.zeros((1, 1), dtype=np.float32),
            field=f"gripper {name}",
            expected_device=_CUDA_DEVICE,
        )
    drive = gripper["drive"]
    if not isinstance(drive, dict) or set(drive) != {
        "profile",
        "configured",
        "actuator",
        "direct_physx",
    }:
        raise ValueError("Joint-velocity gripper drive schema mismatch")
    if drive["profile"] != NATIVE_GRIPPER_DRIVE_PROFILE:
        raise ValueError("Joint-velocity gripper drive profile mismatch")
    configured = drive["configured"]
    expected_configured = {
        "joint_names_expr": ["finger_joint"],
        "stiffness": None,
        "damping": None,
        "effort_limit": NATIVE_GRIPPER_EFFORT_LIMIT,
        "effort_limit_sim": NATIVE_GRIPPER_EFFORT_LIMIT,
        "velocity_limit": NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S,
        "velocity_limit_sim": NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S,
    }
    if configured != expected_configured or any(
        type(configured[key]) is not type(value)
        for key, value in expected_configured.items()
    ):
        raise ValueError("Joint-velocity gripper configured drive mismatch")
    expected_live = {
        "stiffness": np.asarray([[NATIVE_GRIPPER_STIFFNESS]], dtype=np.float32),
        "damping": np.asarray([[NATIVE_GRIPPER_DAMPING]], dtype=np.float32),
        "effort_limit": np.asarray([[NATIVE_GRIPPER_EFFORT_LIMIT]], dtype=np.float32),
        "velocity_limit": np.asarray(
            [[NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S]], dtype=np.float32
        ),
    }
    expected_actuator = {
        **expected_live,
        "effort_limit_sim": expected_live["effort_limit"],
        "velocity_limit_sim": expected_live["velocity_limit"],
    }
    for surface, expected_arrays, expected_device in (
        ("actuator", expected_actuator, _CUDA_DEVICE),
        ("direct_physx", expected_live, _CPU_DEVICE),
    ):
        surface_report = drive[surface]
        if not isinstance(surface_report, dict) or set(surface_report) != set(
            expected_arrays
        ):
            raise ValueError(f"Joint-velocity gripper {surface} schema mismatch")
        for name, expected in expected_arrays.items():
            _validate_array_report(
                surface_report[name],
                expected,
                field=f"gripper {surface} {name}",
                expected_device=expected_device,
            )
    all_six = report["all_six_gripper"]
    if not isinstance(all_six, dict) or set(all_six) != {
        "profile",
        "joint_names",
        "joint_indices",
        "actuator_ownership",
        "reset_write",
        "mimic_joint_contract",
        "buffered_velocity_limit",
        "direct_physx_velocity_limit",
    }:
        raise ValueError("Joint-velocity all-six gripper schema mismatch")
    if (
        all_six["profile"] != NATIVE_GRIPPER_ALL_SIX_PROFILE
        or all_six["joint_names"] != list(GRIPPER_JOINT_NAMES)
        or all_six["joint_indices"] != list(GRIPPER_JOINT_INDICES)
    ):
        raise ValueError("Joint-velocity all-six gripper identity mismatch")
    expected_ownership = {
        "panda_shoulder": {
            "joint_names": [f"panda_joint{index}" for index in range(1, 5)],
            "joint_indices": [0, 1, 2, 3],
        },
        "panda_forearm": {
            "joint_names": [f"panda_joint{index}" for index in range(5, 8)],
            "joint_indices": [4, 5, 6],
        },
        "gripper": {"joint_names": ["finger_joint"], "joint_indices": [7]},
    }
    if all_six["actuator_ownership"] != expected_ownership:
        raise ValueError("Joint-velocity all-six actuator ownership mismatch")
    validate_native_gripper_reset_report(all_six["reset_write"])
    validate_native_gripper_mimic_contract(all_six["mimic_joint_contract"])
    _validate_array_report(
        all_six["buffered_velocity_limit"],
        np.asarray([EXPECTED_FULL_LIMITS_CAPPED], dtype=np.float32),
        field="all-six buffered velocity limit",
        expected_device=_CUDA_DEVICE,
    )
    _validate_array_report(
        all_six["direct_physx_velocity_limit"],
        np.asarray([EXPECTED_FULL_LIMITS_CAPPED], dtype=np.float32),
        field="all-six direct PhysX velocity limit",
        expected_device=_CPU_DEVICE,
    )
    return copy.deepcopy(report)


def _installed_isaaclab_version() -> str:
    return importlib.metadata.version("isaaclab")


def _resolve_joint_velocity_cfg_base(arm_term: Any) -> type:
    resolved = next(
        (
            candidate
            for candidate in type(arm_term.cfg).__mro__
            if f"{candidate.__module__}.{candidate.__qualname__}"
            == _JOINT_VELOCITY_CFG_BASE_CLASS
        ),
        None,
    )
    if resolved is None:
        raise ValueError("Cannot resolve pinned Isaac Lab velocity action config base")
    return resolved


def _verify_isaaclab_sources(
    *, root_env: Any, arm_term: Any, finger_term: Any, robot: Any, actuator: Any
) -> dict[str, str]:
    from isaaclab.envs.manager_based_env import ManagerBasedEnv

    binary_base = next(
        (
            cls
            for cls in type(finger_term).__mro__
            if f"{cls.__module__}.{cls.__qualname__}" == _BINARY_GRIPPER_BASE_CLASS
        ),
        None,
    )
    if binary_base is None:
        raise ValueError("Cannot resolve pinned Isaac Lab binary gripper base class")
    velocity_base = next(
        (
            cls
            for cls in type(arm_term).__mro__
            if f"{cls.__module__}.{cls.__qualname__}" == _JOINT_VELOCITY_BASE_CLASS
        ),
        None,
    )
    if velocity_base is None:
        raise ValueError("Cannot resolve pinned Isaac Lab velocity-action base class")
    velocity_cfg_base = _resolve_joint_velocity_cfg_base(arm_term)
    rl_env_base = next(
        (
            cls
            for cls in type(root_env).__mro__
            if f"{cls.__module__}.{cls.__qualname__}"
            == _MANAGER_BASED_RL_ENV_BASE_CLASS
        ),
        None,
    )
    if rl_env_base is None:
        raise ValueError("Cannot resolve pinned Isaac Lab manager-based RL base class")
    objects = {
        "actions_cfg.py": velocity_cfg_base,
        "joint_actions.py": velocity_base,
        "binary_joint_actions.py": binary_base,
        "actuator_cfg.py": robot.cfg.actuators["panda_shoulder"],
        "actuator_pd.py": actuator,
        "articulation.py": robot,
        "action_manager.py": root_env.action_manager,
        "event_manager.py": root_env.event_manager,
        "manager_based_env.py": ManagerBasedEnv,
        "manager_based_rl_env.py": rl_env_base,
    }
    actual: dict[str, str] = {}
    for source_name, value in objects.items():
        source_object = value if inspect.isclass(value) else type(value)
        source_path_string = inspect.getsourcefile(source_object)
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


def _verify_polaris_sources(
    *, root_env: Any, arm_term: Any, finger_term: Any
) -> dict[str, str]:
    from polaris import native_gripper_runtime
    from polaris.environments import robot_cfg

    droid_cfg_objects = (type(arm_term), type(arm_term.cfg), type(finger_term))
    droid_cfg_paths = {inspect.getsourcefile(value) for value in droid_cfg_objects}
    if None in droid_cfg_paths or len(droid_cfg_paths) != 1:
        raise ValueError("PolaRiS DROID action/config source identity mismatch")
    objects = {
        "droid_cfg.py": type(arm_term.cfg),
        "robot_cfg.py": robot_cfg.make_nvidia_droid_joint_velocity_cfg,
        "native_gripper_runtime.py": (
            native_gripper_runtime.apply_native_gripper_all_six_velocity_limits
        ),
        "manager_based_rl_splat_environment.py": type(root_env),
    }
    actual: dict[str, str] = {}
    for source_name, source_object in objects.items():
        source_path_string = inspect.getsourcefile(source_object)
        if source_path_string is None:
            raise ValueError(f"Cannot resolve PolaRiS runtime source {source_name}")
        source_path = Path(source_path_string)
        if source_path.is_symlink() or not source_path.is_file():
            raise ValueError(f"PolaRiS runtime source is not regular: {source_path}")
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        expected = PI05_DROID_POLARIS_RUNTIME_SOURCE_SHA256[source_name]
        if digest != expected:
            raise ValueError(
                f"PolaRiS runtime source mismatch for {source_name}: {digest}"
            )
        actual[source_name] = digest
    return actual


def _validate_gripper_observation_config(root_env: Any) -> dict[str, Any]:
    """Bind the raw official-DROID closed-fraction observation semantics."""

    policy_cfg = getattr(getattr(root_env.cfg, "observations", None), "policy", None)
    gripper_cfg = getattr(policy_cfg, "gripper_pos", None)
    function = getattr(gripper_cfg, "func", None)
    function_path = (
        f"{getattr(function, '__module__', '')}.{getattr(function, '__name__', '')}"
    )
    if (
        function_path
        != PI05_DROID_GRIPPER_OBSERVATION_CONTRACT["polaris_observation_term"]
        or getattr(gripper_cfg, "clip", object()) is not None
        or getattr(policy_cfg, "enable_corruption", None) is not False
        or getattr(policy_cfg, "concatenate_terms", None) is not False
    ):
        raise ValueError("Native DROID gripper observation configuration mismatch")
    return copy.deepcopy(PI05_DROID_GRIPPER_OBSERVATION_CONTRACT)


def validate_joint_velocity_runtime(
    env: Any, *, expected_gripper_drive_profile: str
) -> dict[str, Any]:
    """Fail closed unless action and PhysX drive semantics match the profile."""

    if expected_gripper_drive_profile != NATIVE_GRIPPER_DRIVE_PROFILE:
        raise ValueError("Expected native gripper drive profile mismatch")
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
    gripper_observation = _validate_gripper_observation_config(root_env)

    reset_event_order = list(root_env.event_manager.active_terms.get("reset", ()))
    if reset_event_order != ["reset_all", "cap_gripper_followers"]:
        raise ValueError("Native reset events must run scene reset then all-six cap")
    if list(root_env.action_manager._terms) != ["arm", "finger_joint"]:
        raise ValueError("Native action-term order must be arm then finger_joint")
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
    if (
        type(arm_term._scale) is not float
        or arm_term._scale != 1.0
        or type(arm_term._offset) is not float
        or arm_term._offset != 0.0
    ):
        raise ValueError(
            f"Velocity action affine mismatch: scale={arm_term._scale}, offset={arm_term._offset}"
        )
    expected_clip = np.broadcast_to(
        np.asarray([-1.0, 1.0], dtype=np.float32), (1, 7, 2)
    )
    clip_report = _require_float32_array(
        arm_term._clip,
        expected_clip,
        field="action term clip",
        expected_device=_CUDA_DEVICE,
    )
    action_buffers = {
        "raw_action": _require_float32_array(
            arm_term.raw_actions,
            np.zeros((1, 7), dtype=np.float32),
            field="arm raw action",
            expected_device=_CUDA_DEVICE,
        ),
        "processed_action": _require_float32_array(
            arm_term.processed_actions,
            np.zeros((1, 7), dtype=np.float32),
            field="arm processed action",
            expected_device=_CUDA_DEVICE,
        ),
    }
    if _class_path(finger_term) != _BINARY_GRIPPER_ACTION_CLASS:
        raise ValueError(
            "Gripper action term must preserve closed-positive binary semantics; "
            f"got {_class_path(finger_term)}"
        )
    if tuple(finger_term._joint_names) != ("finger_joint",):
        raise ValueError(f"Gripper joint mismatch: {finger_term._joint_names}")
    gripper_open = _require_float32_array(
        finger_term._open_command,
        np.zeros((1,), dtype=np.float32),
        field="gripper open command",
        expected_device=_CUDA_DEVICE,
    )
    gripper_closed = _require_float32_array(
        finger_term._close_command,
        np.full((1,), np.pi / 4.0, dtype=np.float32),
        field="gripper closed command",
        expected_device=_CUDA_DEVICE,
    )
    gripper_raw = _require_float32_array(
        finger_term.raw_actions,
        np.zeros((1, 1), dtype=np.float32),
        field="gripper raw action",
        expected_device=_CUDA_DEVICE,
    )
    gripper_processed = _require_float32_array(
        finger_term.processed_actions,
        np.zeros((1, 1), dtype=np.float32),
        field="gripper processed action",
        expected_device=_CUDA_DEVICE,
    )

    robot = root_env.scene["robot"]
    if tuple(robot.joint_names) != EXPECTED_DROID_JOINT_NAMES:
        raise ValueError(f"Full DROID articulation order mismatch: {robot.joint_names}")
    joint_ids, joint_names = robot.find_joints(
        list(PANDA_ARM_JOINT_NAMES), preserve_order=True
    )
    if tuple(joint_names) != PANDA_ARM_JOINT_NAMES:
        raise ValueError(f"Articulation joint order mismatch: {joint_names}")
    finger_ids, finger_names = robot.find_joints(["finger_joint"], preserve_order=True)
    if finger_names != ["finger_joint"] or len(finger_ids) != 1:
        raise ValueError(f"Articulation gripper joint mismatch: {finger_names}")

    for actuator_name in ("panda_shoulder", "panda_forearm", "gripper"):
        actuator = robot.actuators[actuator_name]
        if _class_path(actuator) != _IMPLICIT_ACTUATOR_CLASS:
            raise ValueError(
                f"{actuator_name} must use ImplicitActuator; got {_class_path(actuator)}"
            )
    gripper_cfg = robot.cfg.actuators["gripper"]
    configured_gripper = {
        "joint_names_expr": list(gripper_cfg.joint_names_expr),
        "stiffness": gripper_cfg.stiffness,
        "damping": gripper_cfg.damping,
        "effort_limit": gripper_cfg.effort_limit,
        "effort_limit_sim": gripper_cfg.effort_limit_sim,
        "velocity_limit": gripper_cfg.velocity_limit,
        "velocity_limit_sim": gripper_cfg.velocity_limit_sim,
    }
    expected_configured_gripper = {
        "joint_names_expr": ["finger_joint"],
        "stiffness": None,
        "damping": None,
        "effort_limit": NATIVE_GRIPPER_EFFORT_LIMIT,
        "effort_limit_sim": NATIVE_GRIPPER_EFFORT_LIMIT,
        "velocity_limit": NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S,
        "velocity_limit_sim": NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S,
    }
    if configured_gripper != expected_configured_gripper or any(
        type(configured_gripper[key]) is not type(value)
        for key, value in expected_configured_gripper.items()
    ):
        raise ValueError("Configured native gripper drive mismatch")
    isaaclab_source_sha256 = _verify_isaaclab_sources(
        root_env=root_env,
        arm_term=arm_term,
        finger_term=finger_term,
        robot=robot,
        actuator=robot.actuators["panda_shoulder"],
    )
    polaris_runtime_source_sha256 = _verify_polaris_sources(
        root_env=root_env, arm_term=arm_term, finger_term=finger_term
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
            expected_device=_CUDA_DEVICE,
        ),
        "damping": _require_float32_array(
            data.joint_damping[:, joint_ids],
            expected_damping,
            field="buffered joint damping",
            expected_device=_CUDA_DEVICE,
        ),
        "effort_limit": _require_float32_array(
            data.joint_effort_limits[:, joint_ids],
            expected_effort,
            field="buffered joint effort limits",
            expected_device=_CUDA_DEVICE,
        ),
        "velocity_limit": _require_float32_array(
            data.joint_vel_limits[:, joint_ids],
            expected_velocity,
            field="buffered joint velocity limits",
            expected_device=_CUDA_DEVICE,
        ),
    }

    view = robot.root_physx_view
    direct = {
        "stiffness": _require_float32_array(
            view.get_dof_stiffnesses()[:, joint_ids],
            expected_stiffness,
            field="direct PhysX joint stiffness",
            expected_device=_CPU_DEVICE,
        ),
        "damping": _require_float32_array(
            view.get_dof_dampings()[:, joint_ids],
            expected_damping,
            field="direct PhysX joint damping",
            expected_device=_CPU_DEVICE,
        ),
        "effort_limit": _require_float32_array(
            view.get_dof_max_forces()[:, joint_ids],
            expected_effort,
            field="direct PhysX joint effort limits",
            expected_device=_CPU_DEVICE,
        ),
        "velocity_limit": _require_float32_array(
            view.get_dof_max_velocities()[:, joint_ids],
            expected_velocity,
            field="direct PhysX joint velocity limits",
            expected_device=_CPU_DEVICE,
        ),
    }
    gripper_actuator = robot.actuators["gripper"]
    expected_gripper_stiffness = np.asarray(
        [[NATIVE_GRIPPER_STIFFNESS]], dtype=np.float32
    )
    expected_gripper_damping = np.asarray([[NATIVE_GRIPPER_DAMPING]], dtype=np.float32)
    expected_gripper_effort = np.asarray(
        [[NATIVE_GRIPPER_EFFORT_LIMIT]], dtype=np.float32
    )
    expected_gripper_velocity = np.asarray(
        [[NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S]], dtype=np.float32
    )
    gripper_actuator_drive = {
        "stiffness": _require_float32_array(
            gripper_actuator.stiffness,
            expected_gripper_stiffness,
            field="gripper actuator stiffness",
            expected_device=_CUDA_DEVICE,
        ),
        "damping": _require_float32_array(
            gripper_actuator.damping,
            expected_gripper_damping,
            field="gripper actuator damping",
            expected_device=_CUDA_DEVICE,
        ),
        "effort_limit": _require_float32_array(
            gripper_actuator.effort_limit,
            expected_gripper_effort,
            field="gripper actuator effort limit",
            expected_device=_CUDA_DEVICE,
        ),
        "effort_limit_sim": _require_float32_array(
            gripper_actuator.effort_limit_sim,
            expected_gripper_effort,
            field="gripper actuator simulation effort limit",
            expected_device=_CUDA_DEVICE,
        ),
        "velocity_limit": _require_float32_array(
            gripper_actuator.velocity_limit,
            expected_gripper_velocity,
            field="gripper actuator velocity limit",
            expected_device=_CUDA_DEVICE,
        ),
        "velocity_limit_sim": _require_float32_array(
            gripper_actuator.velocity_limit_sim,
            expected_gripper_velocity,
            field="gripper actuator simulation velocity limit",
            expected_device=_CUDA_DEVICE,
        ),
    }
    gripper_direct_drive = {
        "stiffness": _require_float32_array(
            view.get_dof_stiffnesses()[:, finger_ids],
            expected_gripper_stiffness,
            field="direct PhysX gripper stiffness",
            expected_device=_CPU_DEVICE,
        ),
        "damping": _require_float32_array(
            view.get_dof_dampings()[:, finger_ids],
            expected_gripper_damping,
            field="direct PhysX gripper damping",
            expected_device=_CPU_DEVICE,
        ),
        "effort_limit": _require_float32_array(
            view.get_dof_max_forces()[:, finger_ids],
            expected_gripper_effort,
            field="direct PhysX gripper effort limit",
            expected_device=_CPU_DEVICE,
        ),
        "velocity_limit": _require_float32_array(
            view.get_dof_max_velocities()[:, finger_ids],
            expected_gripper_velocity,
            field="direct PhysX gripper velocity limit",
            expected_device=_CPU_DEVICE,
        ),
    }
    all_six_buffered = _require_float32_array(
        data.joint_vel_limits,
        np.asarray([EXPECTED_FULL_LIMITS_CAPPED], dtype=np.float32),
        field="all-six buffered joint velocity limits",
        expected_device=_CUDA_DEVICE,
    )
    all_six_direct = _require_float32_array(
        view.get_dof_max_velocities(),
        np.asarray([EXPECTED_FULL_LIMITS_CAPPED], dtype=np.float32),
        field="all-six direct PhysX joint velocity limits",
        expected_device=_CPU_DEVICE,
    )
    reset_write = native_gripper_reset_report(root_env)
    ownership = validate_actuator_ownership(robot)
    data_path = Path(
        os.environ.get(
            "POLARIS_DATA_PATH",
            str(Path(__file__).resolve().parents[2] / "PolaRiS-Hub"),
        )
    ).resolve()
    mimic_contract = capture_native_gripper_mimic_contract(
        data_path / "nvidia_droid/noninstanceable.usd"
    )

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
        "gripper_observation": gripper_observation,
        "reset_event_order": reset_event_order,
        "joint_names": list(joint_names),
        "action_term_class": _class_path(arm_term),
        "action_cfg_class": _class_path(arm_term.cfg),
        "action_cfg_base_class": _JOINT_VELOCITY_CFG_BASE_CLASS,
        "scale": 1.0,
        "offset": 0.0,
        "use_default_offset": False,
        "clip": clip_report,
        "action_buffers": action_buffers,
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
            "raw_action": gripper_raw,
            "processed_action": gripper_processed,
            "drive": {
                "profile": expected_gripper_drive_profile,
                "configured": configured_gripper,
                "actuator": gripper_actuator_drive,
                "direct_physx": gripper_direct_drive,
            },
        },
        "all_six_gripper": {
            "profile": NATIVE_GRIPPER_ALL_SIX_PROFILE,
            "joint_names": list(GRIPPER_JOINT_NAMES),
            "joint_indices": list(GRIPPER_JOINT_INDICES),
            "actuator_ownership": ownership,
            "reset_write": reset_write,
            "mimic_joint_contract": mimic_contract,
            "buffered_velocity_limit": all_six_buffered,
            "direct_physx_velocity_limit": all_six_direct,
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
