"""Runtime assertions for the canonical Ego-LAP PolaRiS protocol."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from polaris.config import LAP_EEF_FRAME
from polaris.gripper_semantics import GRIPPER_THRESHOLD_PROFILE


CANONICAL_EPISODE_STEPS = 450
CANONICAL_POLICY_HZ = 15.0
CANONICAL_IK_METHOD = "dls"
CANONICAL_DLS_DAMPING = 0.01
CANONICAL_ARM_SCALE = 1.0
CANONICAL_ARM_JOINTS = tuple(f"panda_joint{index}" for index in range(1, 8))


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
    return {
        "episode_steps": horizon,
        "policy_hz": CANONICAL_POLICY_HZ,
        "step_dt": step_dt,
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
        "action_dim": 7,
    }


def atomic_write_runtime_contract(
    path: Path,
    *,
    protocol: Mapping[str, Any],
    frame: Mapping[str, Any],
) -> None:
    """Atomically persist the live simulator/controller contract for this attempt."""

    payload = {
        "schema_version": 1,
        "protocol": dict(protocol),
        "frame": dict(frame),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
