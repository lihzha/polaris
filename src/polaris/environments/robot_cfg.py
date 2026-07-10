import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from polaris.eef_ik_safety import PANDA_EEF_JOINT_EFFORT_LIMITS
from polaris.eef_ik_safety import PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S
from polaris.eef_ik_safety import PANDA_EEF_PHYSX_SOLVER_TYPE
from polaris.eef_ik_safety import PANDA_EEF_SOLVER_POSITION_ITERATION_COUNT
from polaris.eef_ik_safety import PANDA_EEF_SOLVER_VELOCITY_ITERATION_COUNT
from polaris.eef_ik_safety import GRIPPER_EFFORT_LIMIT
from polaris.eef_ik_safety import GRIPPER_VELOCITY_LIMIT_RAD_S
from polaris.utils import DATA_PATH


def configure_eef_pose_joint_safety(
    robot_cfg: ArticulationCfg,
    *,
    physx_cfg,
    enable_gripper_velocity_limit: bool = False,
) -> ArticulationCfg:
    """Enable EEF PhysX arm limits and an optional gripper velocity cap.

    Isaac Lab 2.3 intentionally ignores the legacy ``velocity_limit`` field on
    implicit actuators unless ``velocity_limit_sim`` is set. Keeping this
    mutation in the EEF setup path preserves native joint-position semantics.
    One articulation velocity iteration is also required so PhysX resolves
    drives and velocity limits instead of relying exclusively on the TGS
    position iterations. The production EEF caller enables the gripper cap
    explicitly. It remains default-off here so the isolated close-impulse
    baseline can reproduce the legacy drive profile.
    """

    if type(enable_gripper_velocity_limit) is not bool:
        raise ValueError("PolaRiS EEF gripper velocity-limit enable flag must be bool")

    limits = {
        "panda_shoulder": {
            "velocity": PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S[0],
            "effort": PANDA_EEF_JOINT_EFFORT_LIMITS[0],
        },
        "panda_forearm": {
            "velocity": PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S[4],
            "effort": PANDA_EEF_JOINT_EFFORT_LIMITS[4],
        },
    }
    for actuator_name, values in limits.items():
        try:
            actuator = robot_cfg.actuators[actuator_name]
        except KeyError as error:
            raise ValueError(
                f"DROID robot config is missing EEF safety actuator {actuator_name!r}"
            ) from error
        actuator.velocity_limit_sim = values["velocity"]
        actuator.effort_limit_sim = values["effort"]

    if enable_gripper_velocity_limit:
        try:
            gripper = robot_cfg.actuators["gripper"]
        except KeyError as error:
            raise ValueError(
                "DROID robot config is missing EEF safety actuator 'gripper'"
            ) from error
        if (
            tuple(getattr(gripper, "joint_names_expr", ())) != ("finger_joint",)
            or getattr(gripper, "stiffness", None) is not None
            or getattr(gripper, "damping", None) is not None
            or getattr(gripper, "velocity_limit", None) != GRIPPER_VELOCITY_LIMIT_RAD_S
            or getattr(gripper, "effort_limit", None) != GRIPPER_EFFORT_LIMIT
        ):
            raise ValueError(
                "DROID gripper configuration does not match the pinned EEF "
                "velocity-limit profile"
            )
        gripper.velocity_limit_sim = GRIPPER_VELOCITY_LIMIT_RAD_S
        gripper.effort_limit_sim = GRIPPER_EFFORT_LIMIT

    articulation_props = robot_cfg.spawn.articulation_props
    if articulation_props is None:
        raise ValueError("DROID robot config has no articulation properties")
    articulation_props.solver_position_iteration_count = (
        PANDA_EEF_SOLVER_POSITION_ITERATION_COUNT
    )
    articulation_props.solver_velocity_iteration_count = (
        PANDA_EEF_SOLVER_VELOCITY_ITERATION_COUNT
    )
    physx_cfg.solver_type = PANDA_EEF_PHYSX_SOLVER_TYPE
    return robot_cfg


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
