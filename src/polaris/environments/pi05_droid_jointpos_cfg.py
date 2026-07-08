"""Audited, behavior-preserving native absolute joint-position configuration."""

from __future__ import annotations

import numpy as np

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass, noise

from polaris.environments.droid_cfg import (
    BinaryJointPositionZeroToOneActionCfg,
    eef_pos,
    eef_quat,
)
from polaris.pi05_droid_jointpos_runtime import (
    JointPositionExecutionRecorder,
    PANDA_ARM_JOINT_NAMES,
)


def _numpy(value):
    try:
        value = value.detach().cpu().numpy()
    except AttributeError:
        value = np.asarray(value)
    return np.asarray(value)


def ordered_arm_joint_position(env: ManagerBasedRLEnv):
    robot = env.scene["robot"]
    joint_ids, joint_names = robot.find_joints(
        list(PANDA_ARM_JOINT_NAMES), preserve_order=True
    )
    if tuple(joint_names) != PANDA_ARM_JOINT_NAMES:
        raise ValueError(f"live Panda state order mismatch: {joint_names}")
    return robot.data.joint_pos[:, joint_ids]


def closed_positive_gripper_position(env: ManagerBasedRLEnv):
    robot = env.scene["robot"]
    joint_ids, joint_names = robot.find_joints(["finger_joint"], preserve_order=True)
    if joint_names != ["finger_joint"]:
        raise ValueError(f"live gripper state order mismatch: {joint_names}")
    return robot.data.joint_pos[:, joint_ids] / (np.pi / 4.0)


class AuditedDroidJointPositionAction(JointPositionAction):
    """Call upstream processing/setter code and observe its exact live effects."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if tuple(self._joint_names) != PANDA_ARM_JOINT_NAMES:
            raise ValueError(f"live Panda action order mismatch: {self._joint_names}")
        self._joint_position_execution = JointPositionExecutionRecorder()

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._joint_position_execution.reset()

    def process_actions(self, actions):
        # Intentionally preserve Isaac Lab's complete processing path.  The
        # recorder is observation-only and neither clips nor guards a target.
        super().process_actions(actions)
        self._joint_position_execution.begin_policy_step(
            _numpy(self.raw_actions), _numpy(self.processed_actions)
        )

    def apply_actions(self):
        # JointPositionAction invokes Articulation.set_joint_position_target.
        # Read the target buffer after every one of the eight upstream calls.
        super().apply_actions()
        self._joint_position_execution.record_apply_target(
            _numpy(self._asset.data.joint_pos_target[:, self._joint_ids])
        )

    def consume_joint_position_execution_report(self):
        return self._joint_position_execution.finish_policy_step(
            _numpy(self._asset.data.joint_pos_target[:, self._joint_ids])
        )


@configclass
class AuditedDroidJointPositionActionCfg(JointPositionActionCfg):
    class_type = AuditedDroidJointPositionAction


@configclass
class DroidJointPositionActionCfg:
    arm = AuditedDroidJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(PANDA_ARM_JOINT_NAMES),
        preserve_order=True,
        scale=1.0,
        offset=0.0,
        use_default_offset=False,
        clip=None,
    )

    finger_joint = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": np.pi / 4.0},
    )


@configclass
class DroidJointPositionObservationCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        arm_joint_pos = ObsTerm(func=ordered_arm_joint_position)
        # Preserve the historical PolaRiS observation semantics exactly.  The
        # group-level corruption switch is false, so the configured Gaussian
        # noise is inert; clipping to [0, 1] remains active.
        gripper_pos = ObsTerm(
            func=closed_positive_gripper_position,
            noise=noise.GaussianNoiseCfg(std=0.05),
            clip=(0.0, 1.0),
        )
        eef_pos = ObsTerm(func=eef_pos)
        eef_quat = ObsTerm(func=eef_quat)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


__all__ = [
    "AuditedDroidJointPositionAction",
    "AuditedDroidJointPositionActionCfg",
    "DroidJointPositionActionCfg",
    "DroidJointPositionObservationCfg",
    "closed_positive_gripper_position",
    "ordered_arm_joint_position",
]
