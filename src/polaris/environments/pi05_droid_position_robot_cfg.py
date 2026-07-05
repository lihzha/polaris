"""Private Panda position-drive configuration for official ``pi05_droid``."""

from __future__ import annotations

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from polaris.environments.robot_cfg import NVIDIA_DROID
from polaris.pi05_droid_position_contract import (
    NATIVE_GRIPPER_EFFORT_LIMIT,
    NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S,
    PANDA_ARM_EFFORT_LIMITS,
    PANDA_ARM_VELOCITY_LIMITS,
    PI05_DROID_POSITION_DRIVE_DAMPING,
    PI05_DROID_POSITION_DRIVE_STIFFNESS,
)


def make_nvidia_droid_position_adapter_cfg() -> ArticulationCfg:
    """Copy the shared robot and bind explicit position-drive limits."""

    config = NVIDIA_DROID.copy()
    config.actuators = dict(config.actuators)
    config.actuators["panda_shoulder"] = ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[1-4]"],
        effort_limit_sim=PANDA_ARM_EFFORT_LIMITS[0],
        velocity_limit_sim=PANDA_ARM_VELOCITY_LIMITS[0],
        stiffness=PI05_DROID_POSITION_DRIVE_STIFFNESS,
        damping=PI05_DROID_POSITION_DRIVE_DAMPING,
    )
    config.actuators["panda_forearm"] = ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[5-7]"],
        effort_limit_sim=PANDA_ARM_EFFORT_LIMITS[4],
        velocity_limit_sim=PANDA_ARM_VELOCITY_LIMITS[4],
        stiffness=PI05_DROID_POSITION_DRIVE_STIFFNESS,
        damping=PI05_DROID_POSITION_DRIVE_DAMPING,
    )
    config.actuators["gripper"] = ImplicitActuatorCfg(
        joint_names_expr=["finger_joint"],
        stiffness=None,
        damping=None,
        effort_limit=NATIVE_GRIPPER_EFFORT_LIMIT,
        effort_limit_sim=NATIVE_GRIPPER_EFFORT_LIMIT,
        velocity_limit=NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S,
        velocity_limit_sim=NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S,
    )
    return config


NVIDIA_DROID_POSITION_ADAPTER = make_nvidia_droid_position_adapter_cfg()


__all__ = [
    "NVIDIA_DROID_POSITION_ADAPTER",
    "make_nvidia_droid_position_adapter_cfg",
]
