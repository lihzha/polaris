"""Portable live-runtime evidence for native absolute joint-position evaluation.

This module deliberately has no Isaac Lab imports.  The audited action term lives in
``polaris.environments.pi05_droid_jointpos_cfg`` and calls the pure recorder below;
the remaining helpers inspect the live environment through its public/runtime
surfaces.  None of the checks alter, clip, or reject a policy target.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from typing import Any

import numpy as np


PANDA_ARM_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
PI05_DROID_JOINTPOS_PROFILE = "openpi_pi05_droid_native_joint_position_v1"
PI05_DROID_JOINTPOS_TRACE_SCHEMA_VERSION = 4
PI05_DROID_JOINTPOS_RUNTIME_MARKER = "POLARIS_PI05_DROID_JOINTPOS_RUNTIME="
PI05_DROID_JOINTPOS_OUTER_STEPS = 450
PI05_DROID_JOINTPOS_INTERNAL_MAX_EPISODE_STEPS = 451
PI05_DROID_JOINTPOS_DECIMATION = 8
PI05_DROID_JOINTPOS_PHYSICS_HZ = 120
PI05_DROID_JOINTPOS_POLICY_HZ = 15
PI05_DROID_JOINTPOS_SENSOR_NAMES = ("external_cam", "wrist_cam")
PI05_DROID_JOINTPOS_NATIVE_IMAGE_SHAPE = (720, 1280, 3)
PI05_DROID_JOINTPOS_BOUNDARY_PROFILE = "outer450_internal451_no_autoreset"

_ACTION_TERM_CLASS = (
    "polaris.environments.pi05_droid_jointpos_cfg.AuditedDroidJointPositionAction"
)
_ACTION_CFG_CLASS = (
    "polaris.environments.pi05_droid_jointpos_cfg.AuditedDroidJointPositionActionCfg"
)
_ACTION_BASE_CLASS = "isaaclab.envs.mdp.actions.joint_actions.JointPositionAction"
_ACTION_CFG_BASE_CLASS = "isaaclab.envs.mdp.actions.actions_cfg.JointPositionActionCfg"
_GRIPPER_ACTION_CLASS = (
    "polaris.environments.droid_cfg.BinaryJointPositionZeroToOneAction"
)
_EXPECTED_STIFFNESS = np.full((1, 7), 400.0, dtype=np.float32)
_EXPECTED_DAMPING = np.full((1, 7), 80.0, dtype=np.float32)
_EXPECTED_EFFORT = np.asarray([[87.0] * 4 + [12.0] * 3], dtype=np.float32)
_EXPECTED_VELOCITY = np.asarray([[2.175] * 4 + [2.61] * 3], dtype=np.float32)
_EXPECTED_HARD_LIMITS = np.asarray(
    [
        [
            (-2.8973000049591064, 2.8973000049591064),
            (-1.7627999782562256, 1.7627999782562256),
            (-2.8973000049591064, 2.8973000049591064),
            (-3.0717999935150146, -0.0697999969124794),
            (-2.8973000049591064, 2.8973000049591064),
            (-0.017500000074505806, 3.752500057220459),
            (-2.8973000049591064, 2.8973000049591064),
        ]
    ],
    dtype=np.float32,
)
_EXPECTED_SOFT_LIMITS = np.asarray(
    [
        [
            (-2.8973000049591064, 2.8973000049591064),
            (-1.7627999782562256, 1.7627999782562256),
            (-2.8973000049591064, 2.8973000049591064),
            (-3.0717999935150146, -0.06979990005493164),
            (-2.8973000049591064, 2.8973000049591064),
            (-0.017499923706054688, 3.752500057220459),
            (-2.8973000049591064, 2.8973000049591064),
        ]
    ],
    dtype=np.float32,
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _class_path(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _numpy(value: Any, *, field: str) -> np.ndarray:
    try:
        value = value.detach().cpu().numpy()
    except AttributeError:
        value = np.asarray(value)
    result = np.asarray(value)
    if (
        not np.issubdtype(result.dtype, np.number)
        or np.issubdtype(result.dtype, np.bool_)
        or not np.isfinite(result).all()
    ):
        raise ValueError(f"{field} must be finite numeric data")
    return result


def _float32_values(value: Any, expected: np.ndarray, *, field: str) -> list[Any]:
    actual = _numpy(value, field=field)
    expected = np.asarray(expected, dtype=np.float32)
    if actual.dtype != np.float32 or not np.array_equal(actual, expected):
        raise ValueError(
            f"{field} mismatch: expected {expected.tolist()}, got "
            f"dtype={actual.dtype} values={actual.tolist()}"
        )
    return actual.tolist()


def _one_nonnegative_integer(value: Any, *, field: str) -> int:
    actual = _numpy(value, field=field)
    if actual.shape != (1,) or not np.issubdtype(actual.dtype, np.integer):
        raise ValueError(f"{field} must be one integer tensor")
    result = int(actual[0])
    if result < 0:
        raise ValueError(f"{field} must be nonnegative")
    return result


class JointPositionExecutionRecorder:
    """Observe upstream processing and all eight target-buffer setter calls."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._raw: np.ndarray | None = None
        self._processed: np.ndarray | None = None
        self._holds: list[np.ndarray] = []

    def begin_policy_step(self, raw: Any, processed: Any) -> None:
        if self._raw is not None:
            raise RuntimeError("prior joint-position execution report was not consumed")
        raw_array = _numpy(raw, field="raw joint-position action")
        processed_array = _numpy(processed, field="processed joint-position action")
        if (
            raw_array.shape != (1, 7)
            or processed_array.shape != (1, 7)
            or raw_array.dtype != np.float32
            or processed_array.dtype != np.float32
        ):
            raise ValueError("joint-position action buffers must be float32 [1,7]")
        self._raw = raw_array.copy()
        self._processed = processed_array.copy()
        self._holds = []

    def record_apply_target(self, target: Any) -> None:
        if self._raw is None or self._processed is None:
            raise RuntimeError("joint-position setter ran without process_actions")
        target_array = _numpy(target, field="joint-position apply target")
        if target_array.shape != (1, 7) or target_array.dtype != np.float32:
            raise ValueError("joint-position apply target must be float32 [1,7]")
        if len(self._holds) >= PI05_DROID_JOINTPOS_DECIMATION:
            raise ValueError("more than eight joint-position setter calls in one step")
        self._holds.append(target_array.copy())

    def finish_policy_step(self, post_step_target: Any) -> dict[str, Any]:
        if self._raw is None or self._processed is None:
            raise RuntimeError("no pending joint-position execution report")
        post = _numpy(post_step_target, field="post-step articulation target")
        if post.shape != (1, 7) or post.dtype != np.float32:
            raise ValueError("post-step articulation target must be float32 [1,7]")
        if len(self._holds) != PI05_DROID_JOINTPOS_DECIMATION:
            raise ValueError(
                "joint-position action was not held for exactly eight physics substeps"
            )
        if not np.array_equal(self._raw, self._processed):
            raise ValueError("scale-one offset-zero processing changed the raw action")
        if any(not np.array_equal(hold, self._processed) for hold in self._holds):
            raise ValueError(
                "articulation target buffer drifted during the eight holds"
            )
        if not np.array_equal(post, self._processed):
            raise ValueError(
                "post-step articulation target differs from processed target"
            )
        result = {
            "schema_version": 1,
            "processing": "upstream_joint_position_action_scale1_offset0_no_clip",
            "raw_action_buffer": self._raw[0].tolist(),
            "processed_action_buffer": self._processed[0].tolist(),
            "apply_target_holds": [hold[0].tolist() for hold in self._holds],
            "apply_target_hold_count": len(self._holds),
            "post_step_articulation_target": post[0].tolist(),
        }
        self.reset()
        return result


def configure_jointpos_timeout(env_cfg: Any) -> float:
    """Leave one internal step beyond the explicit 450-step scoring horizon."""

    seconds = (
        PI05_DROID_JOINTPOS_INTERNAL_MAX_EPISODE_STEPS / PI05_DROID_JOINTPOS_POLICY_HZ
    )
    env_cfg.episode_length_s = seconds
    return seconds


def capture_jointpos_environment_state(env: Any) -> dict[str, Any]:
    root = getattr(env, "unwrapped", env)
    if root.max_episode_length != PI05_DROID_JOINTPOS_INTERNAL_MAX_EPISODE_STEPS:
        raise ValueError("live joint-position max_episode_length must be 451")
    if type(root._sim_step_counter) is not int or root._sim_step_counter < 0:
        raise ValueError("live simulation step counter is invalid")
    if type(root.common_step_counter) is not int or root.common_step_counter < 0:
        raise ValueError("live common step counter is invalid")
    sensors = getattr(root.scene, "sensors", None)
    if not isinstance(sensors, dict):
        raise ValueError("live environment has no closed camera mapping")
    counters = {}
    for name in PI05_DROID_JOINTPOS_SENSOR_NAMES:
        if name not in sensors:
            raise ValueError(f"missing live camera sensor {name}")
        counters[name] = _one_nonnegative_integer(
            sensors[name].frame, field=f"{name} camera frame counter"
        )
    return {
        "boundary_profile": PI05_DROID_JOINTPOS_BOUNDARY_PROFILE,
        "live_max_episode_length": root.max_episode_length,
        "episode_length": _one_nonnegative_integer(
            root.episode_length_buf, field="episode length buffer"
        ),
        "sim_step_counter": root._sim_step_counter,
        "common_step_counter": root.common_step_counter,
        "sensor_frame_counters": counters,
    }


def _validate_native_observation(obs: Any) -> dict[str, Any]:
    if not isinstance(obs, dict) or not isinstance(obs.get("splat"), dict):
        raise ValueError("joint-position observation has no splat camera mapping")
    report = {}
    for name in PI05_DROID_JOINTPOS_SENSOR_NAMES:
        image = np.asarray(obs["splat"].get(name))
        if image.shape != PI05_DROID_JOINTPOS_NATIVE_IMAGE_SHAPE:
            raise ValueError(f"{name} native image shape mismatch: {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"{name} native image dtype must be uint8")
        report[name] = {"shape": list(image.shape), "dtype": str(image.dtype)}
    return report


def capture_jointpos_runtime(env: Any, obs: Any) -> dict[str, Any]:
    """Fail closed over the live native position-control execution surface."""

    root = getattr(env, "unwrapped", env)
    if root.cfg.sim.dt != 1.0 / PI05_DROID_JOINTPOS_PHYSICS_HZ:
        raise ValueError("native joint-position physics dt must be 1/120")
    if root.cfg.decimation != PI05_DROID_JOINTPOS_DECIMATION:
        raise ValueError("native joint-position decimation must be 8")
    capture_jointpos_environment_state(root)
    if list(root.action_manager._terms) != ["arm", "finger_joint"]:
        raise ValueError("joint-position action order must be arm then finger_joint")
    arm = root.action_manager._terms["arm"]
    finger = root.action_manager._terms["finger_joint"]
    if _class_path(arm) != _ACTION_TERM_CLASS:
        raise ValueError(f"joint-position action class mismatch: {_class_path(arm)}")
    if _class_path(arm.cfg) != _ACTION_CFG_CLASS:
        raise ValueError(
            f"joint-position action config mismatch: {_class_path(arm.cfg)}"
        )
    if not any(
        f"{base.__module__}.{base.__qualname__}" == _ACTION_BASE_CLASS
        for base in type(arm).__mro__
    ):
        raise ValueError("audited action does not preserve JointPositionAction")
    if not any(
        f"{base.__module__}.{base.__qualname__}" == _ACTION_CFG_BASE_CLASS
        for base in type(arm.cfg).__mro__
    ):
        raise ValueError("audited config does not preserve JointPositionActionCfg")
    if (
        tuple(arm._joint_names) != PANDA_ARM_JOINT_NAMES
        or arm.cfg.preserve_order is not True
        or arm.cfg.use_default_offset is not False
        or arm.cfg.clip is not None
        or type(arm._scale) is not float
        or arm._scale != 1.0
        or type(arm._offset) is not float
        or arm._offset != 0.0
    ):
        raise ValueError("live joint-position action affine/order contract mismatch")
    if _class_path(finger) != _GRIPPER_ACTION_CLASS:
        raise ValueError("live gripper action class does not preserve closed-positive")
    if tuple(finger._joint_names) != ("finger_joint",):
        raise ValueError("live gripper action joint mismatch")
    open_command = _float32_values(
        finger._open_command,
        np.zeros((1,), dtype=np.float32),
        field="gripper open command",
    )
    closed_command = _float32_values(
        finger._close_command,
        np.full((1,), np.pi / 4.0, dtype=np.float32),
        field="gripper closed command",
    )

    robot = root.scene["robot"]
    joint_ids, joint_names = robot.find_joints(
        list(PANDA_ARM_JOINT_NAMES), preserve_order=True
    )
    if tuple(joint_names) != PANDA_ARM_JOINT_NAMES or list(joint_ids) != list(range(7)):
        raise ValueError(f"live articulation joint order mismatch: {joint_names}")
    finger_ids, finger_names = robot.find_joints(["finger_joint"], preserve_order=True)
    if finger_names != ["finger_joint"] or list(finger_ids) != [7]:
        raise ValueError("live historical gripper observation index mismatch")
    expected_arrays = {
        "joint_stiffness": _EXPECTED_STIFFNESS,
        "joint_damping": _EXPECTED_DAMPING,
        "joint_effort_limits": _EXPECTED_EFFORT,
        "joint_velocity_limits": _EXPECTED_VELOCITY,
        "hard_joint_position_limits": _EXPECTED_HARD_LIMITS,
        "soft_joint_position_limits": _EXPECTED_SOFT_LIMITS,
    }
    live_values = {
        "joint_stiffness": robot.data.joint_stiffness[:, joint_ids],
        "joint_damping": robot.data.joint_damping[:, joint_ids],
        "joint_effort_limits": robot.data.joint_effort_limits[:, joint_ids],
        "joint_velocity_limits": robot.data.joint_vel_limits[:, joint_ids],
        "hard_joint_position_limits": robot.data.joint_pos_limits[:, joint_ids],
        "soft_joint_position_limits": robot.data.soft_joint_pos_limits[:, joint_ids],
    }
    live_actuator = {
        name: _float32_values(live_values[name], expected, field=name)
        for name, expected in expected_arrays.items()
    }
    direct_values = {
        "joint_stiffness": robot.root_physx_view.get_dof_stiffnesses()[:, joint_ids],
        "joint_damping": robot.root_physx_view.get_dof_dampings()[:, joint_ids],
        "joint_effort_limits": robot.root_physx_view.get_dof_max_forces()[:, joint_ids],
        "joint_velocity_limits": robot.root_physx_view.get_dof_max_velocities()[
            :, joint_ids
        ],
        "hard_joint_position_limits": robot.root_physx_view.get_dof_limits()[
            :, joint_ids
        ],
    }
    direct_physx = {
        name: _float32_values(
            direct_values[name], expected_arrays[name], field=f"direct PhysX {name}"
        )
        for name in direct_values
    }
    configured_actuators = {}
    for name, expected in (
        (
            "panda_shoulder",
            {
                "joint_names_expr": ["panda_joint[1-4]"],
                "stiffness": 400.0,
                "damping": 80.0,
                "effort_limit": 87.0,
                "velocity_limit": 2.175,
            },
        ),
        (
            "panda_forearm",
            {
                "joint_names_expr": ["panda_joint[5-7]"],
                "stiffness": 400.0,
                "damping": 80.0,
                "effort_limit": 12.0,
                "velocity_limit": 2.61,
            },
        ),
    ):
        cfg = robot.cfg.actuators[name]
        actual = {
            "joint_names_expr": list(cfg.joint_names_expr),
            "stiffness": cfg.stiffness,
            "damping": cfg.damping,
            "effort_limit": cfg.effort_limit,
            "velocity_limit": cfg.velocity_limit,
        }
        if actual != expected:
            raise ValueError(f"configured {name} actuator mismatch: {actual}")
        configured_actuators[name] = actual

    cameras = _validate_native_observation(obs)
    for name in PI05_DROID_JOINTPOS_SENSOR_NAMES:
        image_shape = tuple(root.scene.sensors[name].image_shape)
        if image_shape != PI05_DROID_JOINTPOS_NATIVE_IMAGE_SHAPE[:2]:
            raise ValueError(f"live {name} camera shape mismatch: {image_shape}")

    policy_cfg = root.cfg.observations.policy
    arm_function = getattr(policy_cfg.arm_joint_pos, "func", None)
    gripper_function = getattr(policy_cfg.gripper_pos, "func", None)
    eef_pos_function = getattr(policy_cfg.eef_pos, "func", None)
    eef_quat_function = getattr(policy_cfg.eef_quat, "func", None)
    gripper_noise = policy_cfg.gripper_pos.noise
    gripper_noise_class = _class_path(gripper_noise)
    if (
        gripper_noise_class != "isaaclab.utils.noise.noise_cfg.GaussianNoiseCfg"
        or type(gripper_noise.mean) is not float
        or gripper_noise.mean != 0.0
        or type(gripper_noise.std) is not float
        or gripper_noise.std != 0.05
    ):
        raise ValueError("historical gripper noise configuration drifted")
    observation = {
        "term_order": ["arm_joint_pos", "gripper_pos", "eef_pos", "eef_quat"],
        "enable_corruption": policy_cfg.enable_corruption,
        "concatenate_terms": policy_cfg.concatenate_terms,
        "state_layout": {
            "arm_joint_indices": list(joint_ids),
            "gripper_joint_index": finger_ids[0],
            "historical_filter_order_equivalent": True,
        },
        "terms": {
            "arm_joint_pos": {
                "function": (
                    f"{getattr(arm_function, '__module__', '')}."
                    f"{getattr(arm_function, '__name__', '')}"
                ),
                "noise": None,
                "clip": None,
            },
            "gripper_pos": {
                "function": (
                    f"{getattr(gripper_function, '__module__', '')}."
                    f"{getattr(gripper_function, '__name__', '')}"
                ),
                "noise": {
                    "class": gripper_noise_class,
                    "mean": gripper_noise.mean,
                    "std": gripper_noise.std,
                    "active": False,
                },
                "clip": list(policy_cfg.gripper_pos.clip),
            },
            "eef_pos": {
                "function": (
                    f"{getattr(eef_pos_function, '__module__', '')}."
                    f"{getattr(eef_pos_function, '__name__', '')}"
                ),
                "noise": None,
                "clip": None,
            },
            "eef_quat": {
                "function": (
                    f"{getattr(eef_quat_function, '__module__', '')}."
                    f"{getattr(eef_quat_function, '__name__', '')}"
                ),
                "noise": None,
                "clip": None,
            },
        },
    }
    expected_observation = {
        "term_order": ["arm_joint_pos", "gripper_pos", "eef_pos", "eef_quat"],
        "enable_corruption": False,
        "concatenate_terms": False,
        "state_layout": {
            "arm_joint_indices": list(range(7)),
            "gripper_joint_index": 7,
            "historical_filter_order_equivalent": True,
        },
        "terms": {
            "arm_joint_pos": {
                "function": (
                    "polaris.environments.pi05_droid_jointpos_cfg."
                    "ordered_arm_joint_position"
                ),
                "noise": None,
                "clip": None,
            },
            "gripper_pos": {
                "function": (
                    "polaris.environments.pi05_droid_jointpos_cfg."
                    "closed_positive_gripper_position"
                ),
                "noise": {
                    "class": "isaaclab.utils.noise.noise_cfg.GaussianNoiseCfg",
                    "mean": 0.0,
                    "std": 0.05,
                    "active": False,
                },
                "clip": [0.0, 1.0],
            },
            "eef_pos": {
                "function": "polaris.environments.droid_cfg.eef_pos",
                "noise": None,
                "clip": None,
            },
            "eef_quat": {
                "function": "polaris.environments.droid_cfg.eef_quat",
                "noise": None,
                "clip": None,
            },
        },
    }
    if observation != expected_observation:
        raise ValueError(f"joint-position observation config mismatch: {observation}")

    report = {
        "schema_version": 1,
        "profile": PI05_DROID_JOINTPOS_PROFILE,
        "status": "pass",
        "boundary": {
            "profile": PI05_DROID_JOINTPOS_BOUNDARY_PROFILE,
            "outer_steps": PI05_DROID_JOINTPOS_OUTER_STEPS,
            "internal_max_episode_steps": (
                PI05_DROID_JOINTPOS_INTERNAL_MAX_EPISODE_STEPS
            ),
            "returned_terminal_flags": "all_false",
            "terminal_rubric_source": "post_action_450_pre_autoreset_info",
        },
        "timing": {
            "physics_dt_seconds": 1.0 / PI05_DROID_JOINTPOS_PHYSICS_HZ,
            "physics_frequency_hz": PI05_DROID_JOINTPOS_PHYSICS_HZ,
            "decimation": PI05_DROID_JOINTPOS_DECIMATION,
            "policy_frequency_hz": PI05_DROID_JOINTPOS_POLICY_HZ,
        },
        "joint_names": list(PANDA_ARM_JOINT_NAMES),
        "action": {
            "term_class": _class_path(arm),
            "cfg_class": _class_path(arm.cfg),
            "base_class": _ACTION_BASE_CLASS,
            "cfg_base_class": _ACTION_CFG_BASE_CLASS,
            "preserve_order": arm.cfg.preserve_order,
            "scale": arm._scale,
            "offset": arm._offset,
            "use_default_offset": arm.cfg.use_default_offset,
            "clip": arm.cfg.clip,
            "semantic": "absolute_joint_position_observation_only_no_guard",
            "setter_calls_per_outer_step": PI05_DROID_JOINTPOS_DECIMATION,
        },
        "observation": observation,
        "configured_actuators": configured_actuators,
        "live_actuator_and_limits": live_actuator,
        "direct_physx_actuator_and_limits": direct_physx,
        "cameras": cameras,
        "gripper": {
            "action_class": _class_path(finger),
            "joint_name": "finger_joint",
            "threshold": "closed_if_gt_0p5_else_open",
            "open_target_rad": open_command[0],
            "closed_target_rad": closed_command[0],
            "observation": "finger_joint_position_divided_by_pi_over_4_closed_positive",
        },
    }
    report["runtime_sha256"] = canonical_sha256(report)
    return validate_jointpos_runtime_report(report)


def validate_jointpos_runtime_report(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("joint-position runtime report must be an object")
    required = {
        "schema_version",
        "profile",
        "status",
        "boundary",
        "timing",
        "joint_names",
        "action",
        "observation",
        "configured_actuators",
        "live_actuator_and_limits",
        "direct_physx_actuator_and_limits",
        "cameras",
        "gripper",
        "runtime_sha256",
    }
    if set(value) != required:
        raise ValueError("joint-position runtime report schema mismatch")
    payload = copy.deepcopy(value)
    digest = payload.pop("runtime_sha256")
    if digest != canonical_sha256(payload):
        raise ValueError("joint-position runtime report SHA-256 mismatch")
    expected_scalars = {
        "schema_version": 1,
        "profile": PI05_DROID_JOINTPOS_PROFILE,
        "status": "pass",
        "joint_names": list(PANDA_ARM_JOINT_NAMES),
    }
    for name, expected in expected_scalars.items():
        if value[name] != expected:
            raise ValueError(f"joint-position runtime {name} mismatch")
    if value["boundary"] != {
        "profile": PI05_DROID_JOINTPOS_BOUNDARY_PROFILE,
        "outer_steps": PI05_DROID_JOINTPOS_OUTER_STEPS,
        "internal_max_episode_steps": PI05_DROID_JOINTPOS_INTERNAL_MAX_EPISODE_STEPS,
        "returned_terminal_flags": "all_false",
        "terminal_rubric_source": "post_action_450_pre_autoreset_info",
    }:
        raise ValueError("joint-position 450/451 boundary contract mismatch")
    if value["timing"] != {
        "physics_dt_seconds": 1.0 / PI05_DROID_JOINTPOS_PHYSICS_HZ,
        "physics_frequency_hz": PI05_DROID_JOINTPOS_PHYSICS_HZ,
        "decimation": PI05_DROID_JOINTPOS_DECIMATION,
        "policy_frequency_hz": PI05_DROID_JOINTPOS_POLICY_HZ,
    }:
        raise ValueError("joint-position runtime timing mismatch")
    if value["action"] != {
        "term_class": _ACTION_TERM_CLASS,
        "cfg_class": _ACTION_CFG_CLASS,
        "base_class": _ACTION_BASE_CLASS,
        "cfg_base_class": _ACTION_CFG_BASE_CLASS,
        "preserve_order": True,
        "scale": 1.0,
        "offset": 0.0,
        "use_default_offset": False,
        "clip": None,
        "semantic": "absolute_joint_position_observation_only_no_guard",
        "setter_calls_per_outer_step": PI05_DROID_JOINTPOS_DECIMATION,
    }:
        raise ValueError("joint-position runtime action mismatch")
    if value["observation"] != {
        "term_order": ["arm_joint_pos", "gripper_pos", "eef_pos", "eef_quat"],
        "enable_corruption": False,
        "concatenate_terms": False,
        "state_layout": {
            "arm_joint_indices": list(range(7)),
            "gripper_joint_index": 7,
            "historical_filter_order_equivalent": True,
        },
        "terms": {
            "arm_joint_pos": {
                "function": (
                    "polaris.environments.pi05_droid_jointpos_cfg."
                    "ordered_arm_joint_position"
                ),
                "noise": None,
                "clip": None,
            },
            "gripper_pos": {
                "function": (
                    "polaris.environments.pi05_droid_jointpos_cfg."
                    "closed_positive_gripper_position"
                ),
                "noise": {
                    "class": "isaaclab.utils.noise.noise_cfg.GaussianNoiseCfg",
                    "mean": 0.0,
                    "std": 0.05,
                    "active": False,
                },
                "clip": [0.0, 1.0],
            },
            "eef_pos": {
                "function": "polaris.environments.droid_cfg.eef_pos",
                "noise": None,
                "clip": None,
            },
            "eef_quat": {
                "function": "polaris.environments.droid_cfg.eef_quat",
                "noise": None,
                "clip": None,
            },
        },
    }:
        raise ValueError("joint-position runtime observation mismatch")
    if value["configured_actuators"] != {
        "panda_shoulder": {
            "joint_names_expr": ["panda_joint[1-4]"],
            "stiffness": 400.0,
            "damping": 80.0,
            "effort_limit": 87.0,
            "velocity_limit": 2.175,
        },
        "panda_forearm": {
            "joint_names_expr": ["panda_joint[5-7]"],
            "stiffness": 400.0,
            "damping": 80.0,
            "effort_limit": 12.0,
            "velocity_limit": 2.61,
        },
    }:
        raise ValueError("joint-position configured actuator mismatch")
    if value["gripper"] != {
        "action_class": _GRIPPER_ACTION_CLASS,
        "joint_name": "finger_joint",
        "threshold": "closed_if_gt_0p5_else_open",
        "open_target_rad": 0.0,
        "closed_target_rad": float(np.float32(np.pi / 4.0)),
        "observation": ("finger_joint_position_divided_by_pi_over_4_closed_positive"),
    }:
        raise ValueError("joint-position gripper runtime mismatch")
    expected_live = {
        "joint_stiffness": _EXPECTED_STIFFNESS,
        "joint_damping": _EXPECTED_DAMPING,
        "joint_effort_limits": _EXPECTED_EFFORT,
        "joint_velocity_limits": _EXPECTED_VELOCITY,
        "hard_joint_position_limits": _EXPECTED_HARD_LIMITS,
        "soft_joint_position_limits": _EXPECTED_SOFT_LIMITS,
    }
    live = value["live_actuator_and_limits"]
    if not isinstance(live, dict) or set(live) != set(expected_live):
        raise ValueError("joint-position live actuator report schema mismatch")
    for name, expected in expected_live.items():
        if not np.array_equal(np.asarray(live[name], dtype=np.float32), expected):
            raise ValueError(f"joint-position live {name} mismatch")
    direct_expected = {
        name: expected
        for name, expected in expected_live.items()
        if name != "soft_joint_position_limits"
    }
    direct = value["direct_physx_actuator_and_limits"]
    if not isinstance(direct, dict) or set(direct) != set(direct_expected):
        raise ValueError("joint-position direct PhysX report schema mismatch")
    for name, expected in direct_expected.items():
        if not np.array_equal(np.asarray(direct[name], dtype=np.float32), expected):
            raise ValueError(f"joint-position direct PhysX {name} mismatch")
    cameras = value["cameras"]
    expected_camera = {
        "shape": list(PI05_DROID_JOINTPOS_NATIVE_IMAGE_SHAPE),
        "dtype": "uint8",
    }
    if cameras != {name: expected_camera for name in PI05_DROID_JOINTPOS_SENSOR_NAMES}:
        raise ValueError("joint-position native camera contract mismatch")
    return copy.deepcopy(value)


def format_jointpos_runtime(report: dict[str, Any]) -> str:
    canonical = validate_jointpos_runtime_report(report)
    return PI05_DROID_JOINTPOS_RUNTIME_MARKER + canonical_json_bytes(canonical).decode(
        "ascii"
    )


def publish_jointpos_runtime(path: Any, report: dict[str, Any]) -> dict[str, Any]:
    """Publish one immutable runtime document without a writable final state."""

    from pathlib import Path  # local keeps the portable import surface minimal

    canonical = validate_jointpos_runtime_report(report)
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"joint-position runtime artifact exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"joint-position runtime temporary exists: {temporary}")
    payload = (
        json.dumps(
            canonical,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        # link(2) is the no-replace publication primitive: a competing final
        # path makes this fail instead of silently replacing evidence.
        os.link(temporary, destination, follow_symlinks=False)
        temporary.unlink()
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
    return validate_jointpos_runtime_artifact(
        destination, expected_runtime_sha256=canonical["runtime_sha256"]
    )


def validate_jointpos_runtime_artifact(
    path: Any, *, expected_runtime_sha256: str | None = None
) -> dict[str, Any]:
    from pathlib import Path

    artifact = Path(path)
    if artifact.is_symlink() or not artifact.is_file():
        raise ValueError("joint-position runtime artifact must be a regular file")
    metadata = artifact.stat()
    if metadata.st_mode & 0o777 != 0o444 or metadata.st_nlink != 1:
        raise ValueError("joint-position runtime artifact must be immutable mode 0444")
    raw = artifact.read_bytes()
    report = validate_jointpos_runtime_report(json.loads(raw))
    if (
        expected_runtime_sha256 is not None
        and report["runtime_sha256"] != expected_runtime_sha256
    ):
        raise ValueError("joint-position runtime artifact identity mismatch")
    return {
        "path": str(artifact.resolve()),
        "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "runtime_sha256": report["runtime_sha256"],
        "mode": "0444",
        "nlink": metadata.st_nlink,
    }
