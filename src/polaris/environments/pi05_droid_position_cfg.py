"""Isaac Lab action configuration for official-DROID position adaptation."""

from __future__ import annotations

from typing import Any

import numpy as np

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass

from polaris.environments.droid_cfg import (
    BinaryJointPositionZeroToOneActionCfg,
    DroidJointVelocityEventCfg,
    DroidJointVelocityObservationCfg,
)
from polaris.native_gripper_runtime import NativeAllJointDynamicRecorder
from polaris.pi05_droid_position_adapter import (
    PositionActionTargetLimitError,
    PositionTargetHoldRecorder,
)
from polaris.pi05_droid_position_contract import PANDA_ARM_JOINT_NAMES


def _numpy_float32(value: Any) -> np.ndarray:
    try:
        value = value.detach().cpu().numpy()
    except AttributeError:
        value = np.asarray(value)
    return np.asarray(value, dtype=np.float32)


class AuditedDroidDeltaJointPositionAction(JointPositionAction):
    """Absolute position action with an exact eight-substep hold witness."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if tuple(self._joint_names) != PANDA_ARM_JOINT_NAMES:
            raise ValueError(f"Live Panda joint order mismatch: {self._joint_names}")
        self._position_target_hold = PositionTargetHoldRecorder()
        self._native_all_joint_recorder = NativeAllJointDynamicRecorder()

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._position_target_hold.reset()
        self._native_all_joint_recorder.reset()

    def process_actions(self, actions):
        super().process_actions(actions)
        processed = _numpy_float32(self.processed_actions)
        limits = _numpy_float32(
            self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        )
        if limits.shape != (1, 7, 2) or bool(
            (processed < limits[:, :, 0]).any()
        ) or bool((processed > limits[:, :, 1]).any()):
            raise PositionActionTargetLimitError(
                "absolute position target exceeds live soft limits before setter"
            )
        self._position_target_hold.begin_policy_step(processed)

    def apply_actions(self):
        # JointPositionAction calls Articulation.set_joint_position_target.
        # Read the articulation buffer back after every setter invocation so
        # the evidence proves the same absolute target was held, rather than
        # merely proving that process_actions saw it once.
        super().apply_actions()
        target = self._asset.data.joint_pos_target[:, self._joint_ids]
        self._position_target_hold.record_physics_substep(_numpy_float32(target))

    def consume_position_target_hold_report(self) -> dict[str, Any]:
        return self._position_target_hold.finish_policy_step()

    def record_native_all_joint_post_policy_step(self):
        return self._native_all_joint_recorder.record_post_policy_step(self._asset)

    def bind_native_all_joint_failure_path(self, path):
        self._native_all_joint_recorder.bind_failure_path(path)

    def native_all_joint_dynamic_report(self, *, include_samples: bool):
        return self._native_all_joint_recorder.report(include_samples=include_samples)

    def reset_native_all_joint_dynamic_report(self):
        self._native_all_joint_recorder.reset()


@configclass
class AuditedDroidDeltaJointPositionActionCfg(JointPositionActionCfg):
    class_type = AuditedDroidDeltaJointPositionAction


@configclass
class DroidPositionAdapterActionCfg:
    """Fresh-measurement targets are already absolute when they enter Isaac."""

    arm = AuditedDroidDeltaJointPositionActionCfg(
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
        close_command_expr={"finger_joint": np.pi / 4},
    )


# The existing native observation and event definitions already provide exact
# panda_joint1..7 ordering, closed-positive gripper state, joint velocity for
# post-step evidence, and the accepted all-six gripper follower cap.  Aliases
# keep those semantics without mutating their previously promoted source.
DroidPositionAdapterObservationCfg = DroidJointVelocityObservationCfg
DroidPositionAdapterEventCfg = DroidJointVelocityEventCfg


__all__ = [
    "AuditedDroidDeltaJointPositionAction",
    "AuditedDroidDeltaJointPositionActionCfg",
    "DroidPositionAdapterActionCfg",
    "DroidPositionAdapterEventCfg",
    "DroidPositionAdapterObservationCfg",
]
