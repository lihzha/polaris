"""Live runtime attestation for the official-DROID position adapter."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from polaris.pi05_droid_position_adapter import (
    PI05_DROID_PHYSICS_FREQUENCY_HZ,
    PI05_DROID_PHYSICS_SUBSTEPS,
    PI05_DROID_POLICY_FREQUENCY_HZ,
    PI05_DROID_POSITION_ADAPTER_PROFILE,
    canonical_json_bytes,
)
from polaris.pi05_droid_position_contract import (
    NATIVE_GRIPPER_DRIVE_PROFILE,
    NATIVE_GRIPPER_EFFORT_LIMIT,
    NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S,
    PANDA_ARM_EFFORT_LIMITS,
    PANDA_ARM_JOINT_NAMES,
    PANDA_ARM_VELOCITY_LIMITS,
    PI05_DROID_POSITION_DRIVE_DAMPING,
    PI05_DROID_POSITION_DRIVE_STIFFNESS,
    PI05_DROID_ISAACLAB_SOURCE_SHA256,
    PI05_DROID_ISAACLAB_VERSION,
)
from polaris.pi05_droid_jointvelocity_contract import (
    NATIVE_GRIPPER_DAMPING,
    NATIVE_GRIPPER_STIFFNESS,
    PI05_DROID_GRIPPER_OBSERVATION_CONTRACT,
)
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
    validate_native_all_joint_dynamic_report,
)
from polaris.joint_velocity_runtime import (
    _installed_isaaclab_version,
    _require_float32_array,
    _validate_array_report,
    _validate_gripper_observation_config,
)
from polaris.pi05_droid_native_eval_contract import (
    validate_environment_runtime_contract,
)


POSITION_RUNTIME_MARKER = "POLARIS_PI05_DROID_POSITION_RUNTIME="
POSITION_SAFETY_PROFILE = "openpi_pi05_droid_position_all_dof_safety_v1"
POSITION_CLOSE_READY_PROFILE = "openpi_pi05_droid_position_close_ready_v1"
POSITION_FAILURE_CLOSE_READY_PROFILE = (
    "openpi_pi05_droid_position_failure_close_ready_v1"
)


def _class_path(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _function_path(value: Any) -> str:
    return f"{getattr(value, '__module__', '')}.{getattr(value, '__name__', '')}"


def _scalar(value: Any, *, field: str) -> float:
    try:
        value = value.item()
    except AttributeError:
        pass
    if type(value) not in (int, float) or isinstance(value, bool) or not np.isfinite(value):
        raise ValueError(f"{field} must be one finite scalar")
    return float(value)


def _live_float32(value: Any, *, shape: tuple[int, ...], field: str) -> list[Any]:
    try:
        value = value.detach().cpu().numpy()
    except AttributeError:
        value = np.asarray(value)
    array = np.asarray(value)
    if array.dtype != np.float32 or array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{field} must be finite live float32 {shape}")
    return array.tolist()


def _runtime_sha256(report: dict[str, Any]) -> str:
    payload = copy.deepcopy(report)
    payload.pop("runtime_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _source_sha256(source_object: Any, *, field: str) -> str:
    inspected = (
        source_object
        if inspect.isclass(source_object) or inspect.isfunction(source_object)
        else type(source_object)
    )
    source_path_string = inspect.getsourcefile(inspected)
    if source_path_string is None:
        raise ValueError(f"cannot resolve source for {field}")
    path = Path(source_path_string)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{field} source must be one regular file")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _position_source_reports(root: Any, arm: Any, finger: Any, robot: Any):
    from isaaclab.envs.manager_based_env import ManagerBasedEnv
    from polaris import native_gripper_runtime
    from polaris.environments import droid_cfg, pi05_droid_position_robot_cfg

    def mro_class(value: Any, path: str) -> type:
        found = next(
            (
                candidate
                for candidate in type(value).__mro__
                if f"{candidate.__module__}.{candidate.__qualname__}" == path
            ),
            None,
        )
        if found is None:
            raise ValueError(f"cannot resolve runtime base class {path}")
        return found

    objects = {
        "actions_cfg.py": mro_class(
            arm.cfg,
            "isaaclab.envs.mdp.actions.actions_cfg.JointPositionActionCfg",
        ),
        "joint_actions.py": mro_class(
            arm, "isaaclab.envs.mdp.actions.joint_actions.JointPositionAction"
        ),
        "binary_joint_actions.py": mro_class(
            finger,
            "isaaclab.envs.mdp.actions.binary_joint_actions.BinaryJointPositionAction",
        ),
        "actuator_cfg.py": robot.cfg.actuators["panda_shoulder"],
        "actuator_pd.py": robot.actuators["panda_shoulder"],
        "articulation.py": robot,
        "action_manager.py": root.action_manager,
        "event_manager.py": root.event_manager,
        "manager_based_env.py": ManagerBasedEnv,
        "manager_based_rl_env.py": mro_class(
            root, "isaaclab.envs.manager_based_rl_env.ManagerBasedRLEnv"
        ),
    }
    isaac = {name: _source_sha256(value, field=name) for name, value in objects.items()}
    if isaac != PI05_DROID_ISAACLAB_SOURCE_SHA256:
        raise ValueError("position runtime Isaac Lab source identity mismatch")
    polaris_objects = {
        "droid_cfg.py": droid_cfg.ordered_arm_joint_pos,
        "pi05_droid_position_cfg.py": arm,
        "pi05_droid_position_robot_cfg.py": (
            pi05_droid_position_robot_cfg.make_nvidia_droid_position_adapter_cfg
        ),
        "native_gripper_runtime.py": (
            native_gripper_runtime.apply_native_gripper_all_six_velocity_limits
        ),
        "manager_based_rl_splat_environment.py": root,
    }
    polaris = {
        name: _source_sha256(value, field=name)
        for name, value in polaris_objects.items()
    }
    return isaac, polaris


def _capture_policy_observation_contract(root: Any) -> dict[str, Any]:
    policy = getattr(getattr(root.cfg, "observations", None), "policy", None)
    if policy is None:
        raise ValueError("position runtime policy observation group missing")
    expected_functions = {
        "arm_joint_pos": "polaris.environments.droid_cfg.ordered_arm_joint_pos",
        "arm_joint_vel": "polaris.environments.droid_cfg.ordered_arm_joint_vel",
        "gripper_pos": "polaris.environments.droid_cfg.gripper_pos",
    }
    terms = {}
    for name, expected_function in expected_functions.items():
        cfg = getattr(policy, name, None)
        function_path = _function_path(getattr(cfg, "func", None))
        if (
            function_path != expected_function
            or getattr(cfg, "noise", None) is not None
            or getattr(cfg, "clip", None) is not None
        ):
            raise ValueError(f"position runtime policy observation term drift: {name}")
        terms[name] = {
            "function": function_path,
            "noise": None,
            "clip": None,
        }
    if (
        getattr(policy, "enable_corruption", None) is not False
        or getattr(policy, "concatenate_terms", None) is not False
    ):
        raise ValueError("position runtime policy observation group drift")
    return {
        "group_class": _class_path(policy),
        "term_order": list(expected_functions),
        "enable_corruption": False,
        "concatenate_terms": False,
        "terms": terms,
    }


def capture_position_adapter_runtime(env: Any) -> dict[str, Any]:
    """Capture action, cadence, joint order, and configured position drive."""

    root = getattr(env, "unwrapped", env)
    arm = root.action_manager._terms["arm"]
    finger = root.action_manager._terms["finger_joint"]
    robot = root.scene["robot"]
    joint_ids, joint_names = robot.find_joints(
        list(PANDA_ARM_JOINT_NAMES), preserve_order=True
    )
    if tuple(joint_names) != PANDA_ARM_JOINT_NAMES:
        raise ValueError(f"live Panda joint order mismatch: {joint_names}")
    cfg = arm.cfg
    scene_robot_cfg = root.cfg.scene.robot
    shoulder = scene_robot_cfg.actuators["panda_shoulder"]
    forearm = scene_robot_cfg.actuators["panda_forearm"]
    gripper = scene_robot_cfg.actuators["gripper"]
    decimation = root.cfg.decimation
    dt = _scalar(root.cfg.sim.dt, field="simulation dt")
    if _installed_isaaclab_version() != PI05_DROID_ISAACLAB_VERSION:
        raise ValueError("position runtime Isaac Lab version mismatch")
    if tuple(robot.joint_names) != EXPECTED_DROID_JOINT_NAMES:
        raise ValueError("position runtime full articulation order mismatch")
    finger_ids, finger_names = robot.find_joints(
        ["finger_joint"], preserve_order=True
    )
    if finger_names != ["finger_joint"]:
        raise ValueError("position runtime finger joint mismatch")
    reset_event_order = list(root.event_manager.active_terms.get("reset", ()))
    action_term_order = list(root.action_manager._terms)
    if reset_event_order != ["reset_all", "cap_gripper_followers"]:
        raise ValueError("position runtime reset event order mismatch")
    if action_term_order != ["arm", "finger_joint"]:
        raise ValueError("position runtime action term order mismatch")
    gripper_observation = _validate_gripper_observation_config(root)
    policy_observation = _capture_policy_observation_contract(root)
    isaac_sources, polaris_sources = _position_source_reports(
        root, arm, finger, robot
    )
    expected_stiffness = np.full((1, 7), 400.0, dtype=np.float32)
    expected_damping = np.full((1, 7), 80.0, dtype=np.float32)
    expected_effort = np.asarray([PANDA_ARM_EFFORT_LIMITS], dtype=np.float32)
    expected_velocity = np.asarray([PANDA_ARM_VELOCITY_LIMITS], dtype=np.float32)
    view = robot.root_physx_view
    direct_position_drive = {
        "joint_stiffness": _require_float32_array(
            view.get_dof_stiffnesses()[:, joint_ids],
            expected_stiffness,
            field="direct PhysX position stiffness",
            expected_device="cpu",
        ),
        "joint_damping": _require_float32_array(
            view.get_dof_dampings()[:, joint_ids],
            expected_damping,
            field="direct PhysX position damping",
            expected_device="cpu",
        ),
        "joint_effort_limits": _require_float32_array(
            view.get_dof_max_forces()[:, joint_ids],
            expected_effort,
            field="direct PhysX position effort limits",
            expected_device="cpu",
        ),
        "joint_velocity_limits": _require_float32_array(
            view.get_dof_max_velocities()[:, joint_ids],
            expected_velocity,
            field="direct PhysX position velocity limits",
            expected_device="cpu",
        ),
    }
    action_buffers = {
        "raw_action": _require_float32_array(
            arm.raw_actions,
            np.zeros((1, 7), dtype=np.float32),
            field="position arm raw action",
            expected_device="cuda:0",
        ),
        "processed_action": _require_float32_array(
            arm.processed_actions,
            np.zeros((1, 7), dtype=np.float32),
            field="position arm processed action",
            expected_device="cuda:0",
        ),
    }
    buffered_soft_limits = robot.data.soft_joint_pos_limits[:, joint_ids]
    buffered_soft_numpy = buffered_soft_limits.detach().cpu().numpy()
    position_limit_readback = {
        "buffered_soft": _require_float32_array(
            buffered_soft_limits,
            buffered_soft_numpy,
            field="buffered soft joint-position limits",
            expected_device="cuda:0",
        ),
        "direct_physx": _require_float32_array(
            view.get_dof_limits()[:, joint_ids],
            buffered_soft_numpy,
            field="direct PhysX joint-position limits",
            expected_device="cpu",
        ),
    }
    gripper_actuator = robot.actuators["gripper"]
    gripper_expected = {
        "stiffness": np.asarray([[NATIVE_GRIPPER_STIFFNESS]], dtype=np.float32),
        "damping": np.asarray([[NATIVE_GRIPPER_DAMPING]], dtype=np.float32),
        "effort_limit": np.asarray([[NATIVE_GRIPPER_EFFORT_LIMIT]], dtype=np.float32),
        "velocity_limit": np.asarray(
            [[NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S]], dtype=np.float32
        ),
    }
    gripper_live = {
        "stiffness": _require_float32_array(
            gripper_actuator.stiffness,
            gripper_expected["stiffness"],
            field="position gripper actuator stiffness",
            expected_device="cuda:0",
        ),
        "damping": _require_float32_array(
            gripper_actuator.damping,
            gripper_expected["damping"],
            field="position gripper actuator damping",
            expected_device="cuda:0",
        ),
        "effort_limit": _require_float32_array(
            gripper_actuator.effort_limit,
            gripper_expected["effort_limit"],
            field="position gripper actuator effort",
            expected_device="cuda:0",
        ),
        "effort_limit_sim": _require_float32_array(
            gripper_actuator.effort_limit_sim,
            gripper_expected["effort_limit"],
            field="position gripper actuator simulation effort",
            expected_device="cuda:0",
        ),
        "velocity_limit": _require_float32_array(
            gripper_actuator.velocity_limit,
            gripper_expected["velocity_limit"],
            field="position gripper actuator velocity",
            expected_device="cuda:0",
        ),
        "velocity_limit_sim": _require_float32_array(
            gripper_actuator.velocity_limit_sim,
            gripper_expected["velocity_limit"],
            field="position gripper actuator simulation velocity",
            expected_device="cuda:0",
        ),
    }
    gripper_action_buffers = {
        "open_command": _require_float32_array(
            finger._open_command,
            np.zeros((1,), dtype=np.float32),
            field="position gripper open command",
            expected_device="cuda:0",
        ),
        "closed_command": _require_float32_array(
            finger._close_command,
            np.full((1,), np.pi / 4.0, dtype=np.float32),
            field="position gripper closed command",
            expected_device="cuda:0",
        ),
        "raw_action": _require_float32_array(
            finger.raw_actions,
            np.zeros((1, 1), dtype=np.float32),
            field="position gripper raw action",
            expected_device="cuda:0",
        ),
        "processed_action": _require_float32_array(
            finger.processed_actions,
            np.zeros((1, 1), dtype=np.float32),
            field="position gripper processed action",
            expected_device="cuda:0",
        ),
    }
    gripper_direct = {
        "stiffness": _require_float32_array(
            view.get_dof_stiffnesses()[:, finger_ids],
            gripper_expected["stiffness"],
            field="direct PhysX gripper stiffness",
            expected_device="cpu",
        ),
        "damping": _require_float32_array(
            view.get_dof_dampings()[:, finger_ids],
            gripper_expected["damping"],
            field="direct PhysX gripper damping",
            expected_device="cpu",
        ),
        "effort_limit": _require_float32_array(
            view.get_dof_max_forces()[:, finger_ids],
            gripper_expected["effort_limit"],
            field="direct PhysX gripper effort",
            expected_device="cpu",
        ),
        "velocity_limit": _require_float32_array(
            view.get_dof_max_velocities()[:, finger_ids],
            gripper_expected["velocity_limit"],
            field="direct PhysX gripper velocity",
            expected_device="cpu",
        ),
    }
    all_six = {
        "profile": NATIVE_GRIPPER_ALL_SIX_PROFILE,
        "joint_names": list(GRIPPER_JOINT_NAMES),
        "joint_indices": list(GRIPPER_JOINT_INDICES),
        "actuator_ownership": validate_actuator_ownership(robot),
        "reset_write": native_gripper_reset_report(root),
        "mimic_joint_contract": capture_native_gripper_mimic_contract(
            Path(
                os.environ.get(
                    "POLARIS_DATA_PATH",
                    str(Path(__file__).resolve().parents[2] / "PolaRiS-Hub"),
                )
            ).resolve()
            / "nvidia_droid/noninstanceable.usd"
        ),
        "buffered_velocity_limit": _require_float32_array(
            robot.data.joint_vel_limits,
            np.asarray([EXPECTED_FULL_LIMITS_CAPPED], dtype=np.float32),
            field="position all-six buffered velocity limits",
            expected_device="cuda:0",
        ),
        "direct_physx_velocity_limit": _require_float32_array(
            view.get_dof_max_velocities(),
            np.asarray([EXPECTED_FULL_LIMITS_CAPPED], dtype=np.float32),
            field="position all-six direct velocity limits",
            expected_device="cpu",
        ),
    }
    report = {
        "schema_version": 1,
        "profile": PI05_DROID_POSITION_ADAPTER_PROFILE,
        "status": "pass",
        "isaaclab_version": PI05_DROID_ISAACLAB_VERSION,
        "isaaclab_source_sha256": isaac_sources,
        "polaris_runtime_source_sha256": polaris_sources,
        "policy_frequency_hz": int(round(1.0 / (dt * decimation))),
        "physics_frequency_hz": int(round(1.0 / dt)),
        "decimation": decimation,
        "gripper_observation": gripper_observation,
        "policy_observation": policy_observation,
        "reset_event_order": reset_event_order,
        "action_term_order": action_term_order,
        "joint_names": list(joint_names),
        "joint_indices": list(joint_ids),
        "action_term_class": _class_path(arm),
        "action_cfg_class": _class_path(cfg),
        "action_cfg_base_class": "isaaclab.envs.mdp.actions.actions_cfg.JointPositionActionCfg",
        "scale": _scalar(cfg.scale, field="position action scale"),
        "offset": _scalar(cfg.offset, field="position action offset"),
        "use_default_offset": cfg.use_default_offset,
        "preserve_order": cfg.preserve_order,
        "action_manager_clip": cfg.clip,
        "action_buffers": action_buffers,
        "target_mode": "absolute_joint_position",
        "target_hold": {
            "apply_calls_per_policy_step": PI05_DROID_PHYSICS_SUBSTEPS,
            "recorder_method": "consume_position_target_hold_report",
            "setter": "Articulation.set_joint_position_target",
        },
        "position_drive": {
            "semantic_role": (
                "existing_polaris_NVIDIA_DROID_simulator_analogue_of_"
                "hardware_cartesian_impedance_update_desired_joint_positions"
            ),
            "claims_exact_hardware_controller_gains": False,
            "shoulder_joint_names_expr": list(shoulder.joint_names_expr),
            "forearm_joint_names_expr": list(forearm.joint_names_expr),
            "position_stiffness": [
                _scalar(shoulder.stiffness, field="shoulder stiffness"),
                _scalar(forearm.stiffness, field="forearm stiffness"),
            ],
            "velocity_damping": [
                _scalar(shoulder.damping, field="shoulder damping"),
                _scalar(forearm.damping, field="forearm damping"),
            ],
            "effort_limit_sim": [
                _scalar(shoulder.effort_limit_sim, field="shoulder effort limit"),
                _scalar(forearm.effort_limit_sim, field="forearm effort limit"),
            ],
            "velocity_limit_sim": [
                _scalar(shoulder.velocity_limit_sim, field="shoulder velocity limit"),
                _scalar(forearm.velocity_limit_sim, field="forearm velocity limit"),
            ],
        },
        "live_position_drive": {
            "joint_stiffness": _live_float32(
                robot.data.joint_stiffness[:, joint_ids],
                shape=(1, 7),
                field="live joint stiffness",
            ),
            "joint_damping": _live_float32(
                robot.data.joint_damping[:, joint_ids],
                shape=(1, 7),
                field="live joint damping",
            ),
            "joint_effort_limits": _live_float32(
                robot.data.joint_effort_limits[:, joint_ids],
                shape=(1, 7),
                field="live joint effort limits",
            ),
            "joint_velocity_limits": _live_float32(
                robot.data.joint_vel_limits[:, joint_ids],
                shape=(1, 7),
                field="live joint velocity limits",
            ),
            "soft_joint_position_limits": _live_float32(
                robot.data.soft_joint_pos_limits[:, joint_ids],
                shape=(1, 7, 2),
                field="live soft joint-position limits",
            ),
        },
        "direct_position_drive": direct_position_drive,
        "position_limit_readback": position_limit_readback,
        "gripper": {
            "action_term_class": _class_path(finger),
            "semantics": "absolute_closed_positive_gt_0p5",
            "open_target_rad": 0.0,
            "closed_target_rad": float(np.float32(np.pi / 4.0)),
            "drive_profile": NATIVE_GRIPPER_DRIVE_PROFILE,
            "joint_names_expr": list(gripper.joint_names_expr),
            "actuator": gripper_live,
            "direct_physx": gripper_direct,
            "action_buffers": gripper_action_buffers,
        },
        "all_six_gripper": all_six,
    }
    report["runtime_sha256"] = _runtime_sha256(report)
    return validate_position_adapter_runtime_report(report)


def validate_position_adapter_runtime_report(value: Any) -> dict[str, Any]:
    """Validate the closed live runtime report."""

    if not isinstance(value, dict):
        raise ValueError("position runtime report must be an object")
    required = {
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
        "policy_observation",
        "reset_event_order",
        "action_term_order",
        "joint_names",
        "joint_indices",
        "action_term_class",
        "action_cfg_class",
        "action_cfg_base_class",
        "scale",
        "offset",
        "use_default_offset",
        "preserve_order",
        "action_manager_clip",
        "action_buffers",
        "target_mode",
        "target_hold",
        "position_drive",
        "live_position_drive",
        "direct_position_drive",
        "position_limit_readback",
        "gripper",
        "all_six_gripper",
        "runtime_sha256",
    }
    if set(value) != required:
        raise ValueError("position runtime report schema mismatch")
    expected = {
        "schema_version": 1,
        "profile": PI05_DROID_POSITION_ADAPTER_PROFILE,
        "status": "pass",
        "isaaclab_version": PI05_DROID_ISAACLAB_VERSION,
        "isaaclab_source_sha256": PI05_DROID_ISAACLAB_SOURCE_SHA256,
        "policy_frequency_hz": PI05_DROID_POLICY_FREQUENCY_HZ,
        "physics_frequency_hz": PI05_DROID_PHYSICS_FREQUENCY_HZ,
        "decimation": PI05_DROID_PHYSICS_SUBSTEPS,
        "gripper_observation": copy.deepcopy(
            PI05_DROID_GRIPPER_OBSERVATION_CONTRACT
        ),
        "policy_observation": {
            "group_class": (
                "polaris.environments.droid_cfg."
                "DroidJointVelocityObservationCfg.PolicyCfg"
            ),
            "term_order": ["arm_joint_pos", "arm_joint_vel", "gripper_pos"],
            "enable_corruption": False,
            "concatenate_terms": False,
            "terms": {
                "arm_joint_pos": {
                    "function": (
                        "polaris.environments.droid_cfg.ordered_arm_joint_pos"
                    ),
                    "noise": None,
                    "clip": None,
                },
                "arm_joint_vel": {
                    "function": (
                        "polaris.environments.droid_cfg.ordered_arm_joint_vel"
                    ),
                    "noise": None,
                    "clip": None,
                },
                "gripper_pos": {
                    "function": "polaris.environments.droid_cfg.gripper_pos",
                    "noise": None,
                    "clip": None,
                },
            },
        },
        "reset_event_order": ["reset_all", "cap_gripper_followers"],
        "action_term_order": ["arm", "finger_joint"],
        "joint_names": list(PANDA_ARM_JOINT_NAMES),
        "joint_indices": list(range(7)),
        "action_term_class": (
            "polaris.environments.pi05_droid_position_cfg."
            "AuditedDroidDeltaJointPositionAction"
        ),
        "action_cfg_class": (
            "polaris.environments.pi05_droid_position_cfg."
            "AuditedDroidDeltaJointPositionActionCfg"
        ),
        "action_cfg_base_class": (
            "isaaclab.envs.mdp.actions.actions_cfg.JointPositionActionCfg"
        ),
        "scale": 1.0,
        "offset": 0.0,
        "use_default_offset": False,
        "preserve_order": True,
        "action_manager_clip": None,
        "target_mode": "absolute_joint_position",
        "target_hold": {
            "apply_calls_per_policy_step": 8,
            "recorder_method": "consume_position_target_hold_report",
            "setter": "Articulation.set_joint_position_target",
        },
        "position_drive": {
            "semantic_role": (
                "existing_polaris_NVIDIA_DROID_simulator_analogue_of_"
                "hardware_cartesian_impedance_update_desired_joint_positions"
            ),
            "claims_exact_hardware_controller_gains": False,
            "shoulder_joint_names_expr": ["panda_joint[1-4]"],
            "forearm_joint_names_expr": ["panda_joint[5-7]"],
            "position_stiffness": [
                PI05_DROID_POSITION_DRIVE_STIFFNESS,
                PI05_DROID_POSITION_DRIVE_STIFFNESS,
            ],
            "velocity_damping": [
                PI05_DROID_POSITION_DRIVE_DAMPING,
                PI05_DROID_POSITION_DRIVE_DAMPING,
            ],
            "effort_limit_sim": [PANDA_ARM_EFFORT_LIMITS[0], PANDA_ARM_EFFORT_LIMITS[4]],
            "velocity_limit_sim": [
                PANDA_ARM_VELOCITY_LIMITS[0],
                PANDA_ARM_VELOCITY_LIMITS[4],
            ],
        },
    }
    for field, expected_value in expected.items():
        if value[field] != expected_value:
            raise ValueError(f"position runtime {field} mismatch")
    live = value["live_position_drive"]
    if not isinstance(live, dict) or set(live) != {
        "joint_stiffness",
        "joint_damping",
        "joint_effort_limits",
        "joint_velocity_limits",
        "soft_joint_position_limits",
    }:
        raise ValueError("position runtime live drive schema mismatch")
    expected_live = {
        "joint_stiffness": np.full((1, 7), 400.0, dtype=np.float32),
        "joint_damping": np.full((1, 7), 80.0, dtype=np.float32),
        "joint_effort_limits": np.asarray([PANDA_ARM_EFFORT_LIMITS], dtype=np.float32),
        "joint_velocity_limits": np.asarray([PANDA_ARM_VELOCITY_LIMITS], dtype=np.float32),
    }
    for field, expected_array in expected_live.items():
        if not np.array_equal(np.asarray(live[field], dtype=np.float32), expected_array):
            raise ValueError(f"position runtime live {field} mismatch")
    soft_limits = np.asarray(live["soft_joint_position_limits"], dtype=np.float32)
    if soft_limits.shape != (1, 7, 2) or bool(
        (soft_limits[:, :, 0] >= soft_limits[:, :, 1]).any()
    ):
        raise ValueError("position runtime live soft limits mismatch")
    source_report = value["polaris_runtime_source_sha256"]
    if (
        not isinstance(source_report, dict)
        or set(source_report)
        != {
            "pi05_droid_position_cfg.py",
            "droid_cfg.py",
            "pi05_droid_position_robot_cfg.py",
            "native_gripper_runtime.py",
            "manager_based_rl_splat_environment.py",
        }
        or any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in source_report.values()
        )
    ):
        raise ValueError("position runtime PolaRiS source report mismatch")
    action_buffers = value["action_buffers"]
    if not isinstance(action_buffers, dict) or set(action_buffers) != {
        "raw_action",
        "processed_action",
    }:
        raise ValueError("position runtime arm action-buffer schema mismatch")
    for field in action_buffers:
        _validate_array_report(
            action_buffers[field],
            np.zeros((1, 7), dtype=np.float32),
            field=f"position arm {field}",
            expected_device="cuda:0",
        )
    direct = value["direct_position_drive"]
    if not isinstance(direct, dict) or set(direct) != set(expected_live):
        raise ValueError("position runtime direct drive schema mismatch")
    for field, expected_array in expected_live.items():
        _validate_array_report(
            direct[field], expected_array, field=f"direct {field}", expected_device="cpu"
        )
    limits = value["position_limit_readback"]
    if not isinstance(limits, dict) or set(limits) != {"buffered_soft", "direct_physx"}:
        raise ValueError("position runtime limit readback schema mismatch")
    buffered_limit_values = np.asarray(limits["buffered_soft"].get("values"), dtype=np.float32)
    if buffered_limit_values.shape != (1, 7, 2):
        raise ValueError("position runtime buffered limit shape mismatch")
    _validate_array_report(
        limits["buffered_soft"],
        buffered_limit_values,
        field="position buffered soft limits",
        expected_device="cuda:0",
    )
    _validate_array_report(
        limits["direct_physx"],
        buffered_limit_values,
        field="position direct PhysX limits",
        expected_device="cpu",
    )
    gripper = value["gripper"]
    if not isinstance(gripper, dict) or set(gripper) != {
        "action_term_class",
        "semantics",
        "open_target_rad",
        "closed_target_rad",
        "drive_profile",
        "joint_names_expr",
        "actuator",
        "direct_physx",
        "action_buffers",
    }:
        raise ValueError("position runtime gripper schema mismatch")
    if {
        key: gripper[key]
        for key in (
            "action_term_class",
            "semantics",
            "open_target_rad",
            "closed_target_rad",
            "drive_profile",
            "joint_names_expr",
        )
    } != {
        "action_term_class": (
            "polaris.environments.droid_cfg.BinaryJointPositionZeroToOneAction"
        ),
        "semantics": "absolute_closed_positive_gt_0p5",
        "open_target_rad": 0.0,
        "closed_target_rad": float(np.float32(np.pi / 4.0)),
        "drive_profile": NATIVE_GRIPPER_DRIVE_PROFILE,
        "joint_names_expr": ["finger_joint"],
    }:
        raise ValueError("position runtime gripper identity mismatch")
    gripper_expected = {
        "stiffness": np.asarray([[NATIVE_GRIPPER_STIFFNESS]], dtype=np.float32),
        "damping": np.asarray([[NATIVE_GRIPPER_DAMPING]], dtype=np.float32),
        "effort_limit": np.asarray([[NATIVE_GRIPPER_EFFORT_LIMIT]], dtype=np.float32),
        "velocity_limit": np.asarray(
            [[NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S]], dtype=np.float32
        ),
    }
    for location, device in (("actuator", "cuda:0"), ("direct_physx", "cpu")):
        expected_fields = set(gripper_expected)
        if location == "actuator":
            expected_fields |= {"effort_limit_sim", "velocity_limit_sim"}
        if set(gripper[location]) != expected_fields:
            raise ValueError("position runtime gripper drive schema mismatch")
        for field, expected_array in gripper_expected.items():
            _validate_array_report(
                gripper[location][field],
                expected_array,
                field=f"gripper {location} {field}",
                expected_device=device,
            )
        if location == "actuator":
            for field, base_field in (
                ("effort_limit_sim", "effort_limit"),
                ("velocity_limit_sim", "velocity_limit"),
            ):
                _validate_array_report(
                    gripper[location][field],
                    gripper_expected[base_field],
                    field=f"gripper actuator {field}",
                    expected_device=device,
                )
    gripper_buffers = gripper["action_buffers"]
    expected_gripper_buffers = {
        "open_command": np.zeros((1,), dtype=np.float32),
        "closed_command": np.full((1,), np.pi / 4.0, dtype=np.float32),
        "raw_action": np.zeros((1, 1), dtype=np.float32),
        "processed_action": np.zeros((1, 1), dtype=np.float32),
    }
    if not isinstance(gripper_buffers, dict) or set(gripper_buffers) != set(
        expected_gripper_buffers
    ):
        raise ValueError("position runtime gripper action-buffer schema mismatch")
    for field, expected_array in expected_gripper_buffers.items():
        _validate_array_report(
            gripper_buffers[field],
            expected_array,
            field=f"position gripper {field}",
            expected_device="cuda:0",
        )
    all_six = value["all_six_gripper"]
    expected_ownership = {
        "panda_shoulder": {
            "joint_names": [f"panda_joint{i}" for i in range(1, 5)],
            "joint_indices": [0, 1, 2, 3],
        },
        "panda_forearm": {
            "joint_names": [f"panda_joint{i}" for i in range(5, 8)],
            "joint_indices": [4, 5, 6],
        },
        "gripper": {"joint_names": ["finger_joint"], "joint_indices": [7]},
    }
    if (
        not isinstance(all_six, dict)
        or set(all_six)
        != {
            "profile",
            "joint_names",
            "joint_indices",
            "actuator_ownership",
            "reset_write",
            "mimic_joint_contract",
            "buffered_velocity_limit",
            "direct_physx_velocity_limit",
        }
        or all_six["profile"] != NATIVE_GRIPPER_ALL_SIX_PROFILE
        or all_six["joint_names"] != list(GRIPPER_JOINT_NAMES)
        or all_six["joint_indices"] != list(GRIPPER_JOINT_INDICES)
        or all_six["actuator_ownership"] != expected_ownership
    ):
        raise ValueError("position runtime all-six gripper identity mismatch")
    validate_native_gripper_reset_report(all_six["reset_write"])
    validate_native_gripper_mimic_contract(all_six["mimic_joint_contract"])
    _validate_array_report(
        all_six["buffered_velocity_limit"],
        np.asarray([EXPECTED_FULL_LIMITS_CAPPED], dtype=np.float32),
        field="all-six buffered velocity limits",
        expected_device="cuda:0",
    )
    _validate_array_report(
        all_six["direct_physx_velocity_limit"],
        np.asarray([EXPECTED_FULL_LIMITS_CAPPED], dtype=np.float32),
        field="all-six direct velocity limits",
        expected_device="cpu",
    )
    if value["runtime_sha256"] != _runtime_sha256(value):
        raise ValueError("position runtime SHA-256 mismatch")
    return json.loads(canonical_json_bytes(value))


def print_position_adapter_runtime(report: dict[str, Any]) -> None:
    validated = validate_position_adapter_runtime_report(report)
    print(POSITION_RUNTIME_MARKER + json.dumps(validated, sort_keys=True), flush=True)


def make_position_safety_report(dynamic_report: Any, *, outer_steps: int) -> dict[str, Any]:
    """Wrap the generic all-DOF monitor under this distinct controller profile."""

    dynamic = validate_native_all_joint_dynamic_report(
        dynamic_report, require_samples=False
    )
    if (
        type(outer_steps) is not int
        or outer_steps < 0
        or dynamic["apply_calls"] != outer_steps * 8
        or dynamic["post_policy_step_samples"] != outer_steps
        or dynamic["terminal_velocity_failure"] is not None
    ):
        raise ValueError("position-profile all-DOF safety cadence mismatch")
    value = {
        "schema_version": 1,
        "profile": POSITION_SAFETY_PROFILE,
        "controller_profile": PI05_DROID_POSITION_ADAPTER_PROFILE,
        "outer_steps": outer_steps,
        "apply_calls": dynamic["apply_calls"],
        "post_policy_step_samples": dynamic["post_policy_step_samples"],
        "velocity_monitor": dynamic,
    }
    return validate_position_safety_report(value)


def validate_position_safety_report(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "profile",
        "controller_profile",
        "outer_steps",
        "apply_calls",
        "post_policy_step_samples",
        "velocity_monitor",
    }:
        raise ValueError("position safety report schema mismatch")
    if (
        value["schema_version"] != 1
        or value["profile"] != POSITION_SAFETY_PROFILE
        or value["controller_profile"] != PI05_DROID_POSITION_ADAPTER_PROFILE
        or type(value["outer_steps"]) is not int
        or value["outer_steps"] < 0
        or value["apply_calls"] != value["outer_steps"] * 8
        or value["post_policy_step_samples"] != value["outer_steps"]
    ):
        raise ValueError("position safety report identity mismatch")
    dynamic = validate_native_all_joint_dynamic_report(
        value["velocity_monitor"], require_samples=False
    )
    if (
        dynamic["apply_calls"] != value["apply_calls"]
        or dynamic["post_policy_step_samples"]
        != value["post_policy_step_samples"]
        or dynamic["terminal_velocity_failure"] is not None
    ):
        raise ValueError("position safety monitor mismatch")
    return json.loads(canonical_json_bytes(value))


def _artifact_identity(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} artifact must be an object")
    identity = {key: value.get(key) for key in ("path", "size", "sha256", "mode", "nlink")}
    if (
        not isinstance(identity["path"], str)
        or type(identity["size"]) is not int
        or identity["size"] <= 0
        or not isinstance(identity["sha256"], str)
        or len(identity["sha256"]) != 64
        or identity["mode"] != "0444"
        or identity["nlink"] != 1
    ):
        raise ValueError(f"{field} artifact identity mismatch")
    return identity


def _episode_result(value: Any) -> dict[str, Any]:
    required = {
        "episode",
        "episode_length",
        "success",
        "progress",
        "numerical_failure",
        "numerical_failure_reason",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("position episode result schema mismatch")
    if (
        value["episode"] != 0
        or type(value["episode_length"]) is not int
        or value["episode_length"] <= 0
        or type(value["success"]) is not bool
        or type(value["progress"]) not in (int, float)
        or not np.isfinite(value["progress"])
        or type(value["numerical_failure"]) is not bool
        or not isinstance(value["numerical_failure_reason"], str)
    ):
        raise ValueError("position episode result identity mismatch")
    return copy.deepcopy(value)


def make_position_episode_sidecar(
    *,
    episode_result: dict[str, Any],
    environment_runtime_contract: dict[str, Any],
    terminal_rollout: dict[str, Any],
    safety_report: dict[str, Any],
    trace_artifact: dict[str, Any],
    video_artifact: dict[str, Any],
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "profile": "openpi_pi05_droid_position_episode_sidecar_v1",
        "episode_result": _episode_result(episode_result),
        "environment_runtime_contract": validate_environment_runtime_contract(
            environment_runtime_contract
        ),
        "terminal_rollout": copy.deepcopy(terminal_rollout),
        "safety_report": validate_position_safety_report(safety_report),
        "trace_artifact": _artifact_identity(trace_artifact, field="trace"),
        "video_artifact": _artifact_identity(video_artifact, field="video"),
        "incident_artifact": None,
    }
    return validate_position_episode_sidecar(value)


def validate_position_episode_sidecar(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "profile",
        "episode_result",
        "environment_runtime_contract",
        "terminal_rollout",
        "safety_report",
        "trace_artifact",
        "video_artifact",
        "incident_artifact",
    }:
        raise ValueError("position success sidecar schema mismatch")
    result = _episode_result(value["episode_result"])
    runtime = validate_environment_runtime_contract(
        value["environment_runtime_contract"]
    )
    safety = validate_position_safety_report(value["safety_report"])
    terminal = value["terminal_rollout"]
    if (
        value["schema_version"] != 1
        or value["profile"] != "openpi_pi05_droid_position_episode_sidecar_v1"
        or result["episode_length"] != 450
        or result["numerical_failure"]
        or result["numerical_failure_reason"]
        or result["success"] != terminal["rubric"]["success"]
        or float(result["progress"]) != float(terminal["rubric"]["progress"])
        or safety["outer_steps"] != 450
        or terminal["outer_steps_completed"] != 450
        or terminal["environment_after"]["episode_length"] != 450
        or terminal["environment_before"]["live_max_episode_length"]
        != runtime["live_max_episode_length"]
        or value["incident_artifact"] is not None
    ):
        raise ValueError("position success sidecar content mismatch")
    _artifact_identity(value["trace_artifact"], field="trace")
    _artifact_identity(value["video_artifact"], field="video")
    return json.loads(canonical_json_bytes(value))


def make_position_failure_sidecar(
    *,
    episode_result: dict[str, Any],
    environment_runtime_contract: dict[str, Any],
    terminal_failure: dict[str, Any],
    dynamic_report: dict[str, Any],
    trace_artifact: dict[str, Any],
    video_artifact: dict[str, Any],
    incident_artifact: dict[str, Any],
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "profile": "openpi_pi05_droid_position_failure_sidecar_v1",
        "episode_result": _episode_result(episode_result),
        "environment_runtime_contract": validate_environment_runtime_contract(
            environment_runtime_contract
        ),
        "terminal_failure": copy.deepcopy(terminal_failure),
        "dynamic_report": validate_native_all_joint_dynamic_report(
            dynamic_report, require_samples=False
        ),
        "trace_artifact": _artifact_identity(trace_artifact, field="trace"),
        "video_artifact": _artifact_identity(video_artifact, field="video"),
        "incident_artifact": _artifact_identity(incident_artifact, field="incident"),
    }
    return validate_position_failure_sidecar(value)


def validate_position_failure_sidecar(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "profile",
        "episode_result",
        "environment_runtime_contract",
        "terminal_failure",
        "dynamic_report",
        "trace_artifact",
        "video_artifact",
        "incident_artifact",
    }:
        raise ValueError("position failure sidecar schema mismatch")
    result = _episode_result(value["episode_result"])
    validate_environment_runtime_contract(value["environment_runtime_contract"])
    dynamic = validate_native_all_joint_dynamic_report(
        value["dynamic_report"], require_samples=False
    )
    terminal = value["terminal_failure"]
    incident = _artifact_identity(value["incident_artifact"], field="incident")
    if (
        value["schema_version"] != 1
        or value["profile"] != "openpi_pi05_droid_position_failure_sidecar_v1"
        or not result["numerical_failure"]
        or result["success"]
        or float(result["progress"]) != 0.0
        or result["numerical_failure_reason"] != terminal["reason"]
        or terminal["rubric"] != {"success": False, "progress": 0.0}
        or terminal["incident_artifact"] != incident
        or dynamic != terminal["dynamic_report"]
    ):
        raise ValueError("position failure sidecar content mismatch")
    _artifact_identity(value["trace_artifact"], field="trace")
    _artifact_identity(value["video_artifact"], field="video")
    return json.loads(canonical_json_bytes(value))


def make_position_close_ready(
    *,
    runtime_artifact: dict[str, Any],
    trace_artifact: dict[str, Any],
    video_artifact: dict[str, Any],
    metrics_artifact: dict[str, Any],
    sidecar_artifact: dict[str, Any],
    safety_report: dict[str, Any],
    environment_runtime_contract: dict[str, Any],
    terminal_rollout: dict[str, Any],
    outer_steps: int,
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "profile": POSITION_CLOSE_READY_PROFILE,
        "controller_profile": PI05_DROID_POSITION_ADAPTER_PROFILE,
        "lifecycle_stage": "env_close_pending_then_simulation_app_close",
        "outer_steps": outer_steps,
        "runtime_artifact": _artifact_identity(runtime_artifact, field="runtime"),
        "trace_artifact": _artifact_identity(trace_artifact, field="trace"),
        "video_artifact": _artifact_identity(video_artifact, field="video"),
        "metrics_artifact": _artifact_identity(metrics_artifact, field="metrics"),
        "sidecar_artifact": _artifact_identity(sidecar_artifact, field="sidecar"),
        "safety_report": validate_position_safety_report(safety_report),
        "environment_runtime_contract": validate_environment_runtime_contract(
            environment_runtime_contract
        ),
        "terminal_rollout": copy.deepcopy(terminal_rollout),
    }
    return validate_position_close_ready(value)


def validate_position_close_ready(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "profile",
        "controller_profile",
        "lifecycle_stage",
        "outer_steps",
        "runtime_artifact",
        "trace_artifact",
        "video_artifact",
        "metrics_artifact",
        "sidecar_artifact",
        "safety_report",
        "environment_runtime_contract",
        "terminal_rollout",
    }:
        raise ValueError("position close-ready schema mismatch")
    if (
        value["schema_version"] != 1
        or value["profile"] != POSITION_CLOSE_READY_PROFILE
        or value["controller_profile"] != PI05_DROID_POSITION_ADAPTER_PROFILE
        or value["lifecycle_stage"]
        != "env_close_pending_then_simulation_app_close"
        or type(value["outer_steps"]) is not int
        or value["outer_steps"] <= 0
    ):
        raise ValueError("position close-ready identity mismatch")
    for field in ("runtime", "trace", "video", "metrics", "sidecar"):
        _artifact_identity(value[f"{field}_artifact"], field=field)
    safety = validate_position_safety_report(value["safety_report"])
    if safety["outer_steps"] != value["outer_steps"]:
        raise ValueError("position close-ready safety step mismatch")
    runtime = validate_environment_runtime_contract(
        value["environment_runtime_contract"]
    )
    terminal = value["terminal_rollout"]
    if (
        not isinstance(terminal, dict)
        or set(terminal)
        != {
            "environment_before",
            "environment_after",
            "outer_steps_completed",
            "rubric",
        }
        or terminal["outer_steps_completed"] != value["outer_steps"]
        or terminal["environment_before"]["live_max_episode_length"]
        != runtime["live_max_episode_length"]
        or terminal["environment_after"]["episode_length"]
        != value["outer_steps"]
        or terminal["environment_after"]["sim_step_counter"]
        - terminal["environment_before"]["sim_step_counter"]
        != value["outer_steps"] * 8
        or terminal["environment_after"]["common_step_counter"]
        - terminal["environment_before"]["common_step_counter"]
        != value["outer_steps"]
        or any(
            terminal["environment_after"]["sensor_frame_counters"][name]
            - terminal["environment_before"]["sensor_frame_counters"][name]
            != value["outer_steps"]
            for name in ("external_cam", "wrist_cam")
        )
    ):
        raise ValueError("position close-ready terminal cadence mismatch")
    return json.loads(canonical_json_bytes(value))


def make_position_failure_close_ready(
    *,
    runtime_artifact: dict[str, Any],
    trace_artifact: dict[str, Any],
    video_artifact: dict[str, Any],
    metrics_artifact: dict[str, Any],
    sidecar_artifact: dict[str, Any],
    environment_runtime_contract: dict[str, Any],
    terminal_failure: dict[str, Any],
) -> dict[str, Any]:
    if (
        not isinstance(terminal_failure, dict)
        or terminal_failure.get("profile")
        != "openpi_pi05_droid_position_numerical_failure_v1"
        or terminal_failure.get("rubric") != {"success": False, "progress": 0.0}
    ):
        raise ValueError("position failure close-ready terminal mismatch")
    value = {
        "schema_version": 1,
        "profile": POSITION_FAILURE_CLOSE_READY_PROFILE,
        "controller_profile": PI05_DROID_POSITION_ADAPTER_PROFILE,
        "lifecycle_stage": "env_close_pending_then_simulation_app_close",
        "runtime_artifact": _artifact_identity(runtime_artifact, field="runtime"),
        "trace_artifact": _artifact_identity(trace_artifact, field="trace"),
        "video_artifact": _artifact_identity(video_artifact, field="video"),
        "metrics_artifact": _artifact_identity(metrics_artifact, field="metrics"),
        "sidecar_artifact": _artifact_identity(sidecar_artifact, field="sidecar"),
        "environment_runtime_contract": validate_environment_runtime_contract(
            environment_runtime_contract
        ),
        "terminal_failure": copy.deepcopy(terminal_failure),
    }
    return validate_position_failure_close_ready(value)


def validate_position_failure_close_ready(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "profile",
        "controller_profile",
        "lifecycle_stage",
        "runtime_artifact",
        "trace_artifact",
        "video_artifact",
        "metrics_artifact",
        "sidecar_artifact",
        "environment_runtime_contract",
        "terminal_failure",
    }:
        raise ValueError("position failure close-ready schema mismatch")
    if (
        value["schema_version"] != 1
        or value["profile"] != POSITION_FAILURE_CLOSE_READY_PROFILE
        or value["controller_profile"] != PI05_DROID_POSITION_ADAPTER_PROFILE
        or value["lifecycle_stage"]
        != "env_close_pending_then_simulation_app_close"
    ):
        raise ValueError("position failure close-ready identity mismatch")
    for field in ("runtime", "trace", "video", "metrics", "sidecar"):
        _artifact_identity(value[f"{field}_artifact"], field=field)
    validate_environment_runtime_contract(value["environment_runtime_contract"])
    terminal = value["terminal_failure"]
    if (
        not isinstance(terminal, dict)
        or terminal.get("profile")
        != "openpi_pi05_droid_position_numerical_failure_v1"
        or terminal.get("rubric") != {"success": False, "progress": 0.0}
    ):
        raise ValueError("position failure close-ready terminal mismatch")
    return json.loads(canonical_json_bytes(value))


__all__ = [
    "POSITION_RUNTIME_MARKER",
    "POSITION_CLOSE_READY_PROFILE",
    "POSITION_FAILURE_CLOSE_READY_PROFILE",
    "POSITION_SAFETY_PROFILE",
    "capture_position_adapter_runtime",
    "make_position_close_ready",
    "make_position_episode_sidecar",
    "make_position_failure_close_ready",
    "make_position_failure_sidecar",
    "make_position_safety_report",
    "print_position_adapter_runtime",
    "validate_position_close_ready",
    "validate_position_adapter_runtime_report",
    "validate_position_episode_sidecar",
    "validate_position_failure_sidecar",
    "validate_position_failure_close_ready",
    "validate_position_safety_report",
]
