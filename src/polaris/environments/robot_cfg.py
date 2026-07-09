import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from polaris.utils import DATA_PATH
from polaris.pi05_droid_jointvelocity_contract import (
    NATIVE_GRIPPER_EFFORT_LIMIT,
    NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S,
    PANDA_ARM_VELOCITY_DRIVE_DAMPING,
    PANDA_ARM_VELOCITY_DRIVE_STIFFNESS,
)

NVIDIA_DROID = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(DATA_PATH / "nvidia_droid/noninstanceable.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0),
        rot=(1, 0, 0, 0),
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -1 / 5 * np.pi,
            "panda_joint3": 0.0,
            "panda_joint4": -4 / 5 * np.pi,
            "panda_joint5": 0.0,
            "panda_joint6": 3 / 5 * np.pi,
            "panda_joint7": 0,
            "finger_joint": 0.0,
            "right_outer.*": 0.0,
            "left_inner.*": 0.0,
            "right_inner.*": 0.0,
        },
    ),
    soft_joint_pos_limit_factor=1,
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=400.0,
            damping=80.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=400.0,
            damping=80.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            stiffness=None,
            damping=None,
            effort_limit=200.0,
            velocity_limit=5.0,  # 2.175,
        ),
    },
)


def make_nvidia_droid_joint_velocity_cfg() -> ArticulationCfg:
    """Return the profile-local Panda velocity-drive articulation config.

    The upstream robot uses a position PD drive (stiffness 400).  A velocity
    target sent into that drive would still be pulled toward the stale
    position target.  The native DROID profile therefore has a private robot
    config with explicit zero position stiffness, nonzero velocity damping,
    and solver-enforced effort/velocity limits.  The shared joint-position and
    EEF ``NVIDIA_DROID`` object is not mutated.
    """

    config = NVIDIA_DROID.copy()
    config.actuators = dict(config.actuators)
    config.actuators["panda_shoulder"] = ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[1-4]"],
        effort_limit_sim=87.0,
        velocity_limit_sim=2.175,
        stiffness=PANDA_ARM_VELOCITY_DRIVE_STIFFNESS,
        damping=PANDA_ARM_VELOCITY_DRIVE_DAMPING,
    )
    config.actuators["panda_forearm"] = ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[5-7]"],
        effort_limit_sim=12.0,
        velocity_limit_sim=2.61,
        stiffness=PANDA_ARM_VELOCITY_DRIVE_STIFFNESS,
        damping=PANDA_ARM_VELOCITY_DRIVE_DAMPING,
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
