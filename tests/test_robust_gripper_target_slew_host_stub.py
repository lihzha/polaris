from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_robust_action_binds_finger_target_slew_report_in_isolated_host_stub():
    root = Path(__file__).parents[1]
    source = (root / "src" / "polaris" / "robust_differential_ik.py").read_text()
    assert source.index("self._max_delta_joint_pos =") < source.index(
        "self._reset_episode_safety_state(episode_index=None)"
    )
    code = r"""
import sys
import types

import torch


def module(name):
    value = types.ModuleType(name)
    value.__path__ = []
    sys.modules[name] = value
    return value


omni = module("omni")
omni_log = module("omni.log")
omni_usd = module("omni.usd")
omni.log = omni_log
omni.usd = omni_usd
for name in ("info", "warn", "error"):
    setattr(omni_log, name, lambda *_args, **_kwargs: None)

isaaclab = module("isaaclab")
isaaclab.sim = module("isaaclab.sim")
module("isaaclab.controllers")
differential_ik = module("isaaclab.controllers.differential_ik")


class DifferentialIKController:
    pass


differential_ik.DifferentialIKController = DifferentialIKController
module("isaaclab.envs")
module("isaaclab.envs.mdp")
module("isaaclab.envs.mdp.actions")
actions_cfg = module("isaaclab.envs.mdp.actions.actions_cfg")
task_space = module("isaaclab.envs.mdp.actions.task_space_actions")


class OffsetCfg:
    def __init__(self, *_args, **_kwargs):
        self.pos = (0.0, 0.0, 0.0)
        self.rot = (1.0, 0.0, 0.0, 0.0)


class DifferentialInverseKinematicsActionCfg:
    OffsetCfg = OffsetCfg


class DifferentialInverseKinematicsAction:
    @property
    def num_envs(self):
        return self._env.num_envs

    @property
    def device(self):
        return self._env.device

    def reset(self, *_args, **_kwargs):
        env_ids = _args[0] if _args else slice(None)
        self._raw_actions[env_ids] = 0.0


actions_cfg.DifferentialInverseKinematicsActionCfg = (
    DifferentialInverseKinematicsActionCfg
)
task_space.DifferentialInverseKinematicsAction = DifferentialInverseKinematicsAction
utils = module("isaaclab.utils")
utils.configclass = lambda cls: cls
math_module = module("isaaclab.utils.math")
math_module.compute_pose_error = lambda *_args, **_kwargs: None
pxr = module("pxr")
pxr.PhysxSchema = types.SimpleNamespace()
pxr.UsdPhysics = types.SimpleNamespace()

import polaris.robust_differential_ik as robust
from polaris.eef_ik_safety import PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S
from polaris.eef_gripper_runtime import EEF_GRIPPER_TARGET_SLEW_PROFILE
from polaris.eef_gripper_runtime import (
    EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE,
)

robust.PINNED_DYNAMIC_DEVICE = "cpu"
robust.validate_eef_gripper_static_contract = lambda value, **_kwargs: dict(value)
captured_dynamic = []


def validate_dynamic(value, **_kwargs):
    captured_dynamic.append(value)
    return value


robust.validate_eef_gripper_dynamic_evidence = validate_dynamic
static = {
    "driver_target_slew": {
        "profile": EEF_GRIPPER_TARGET_SLEW_PROFILE
    }
}


class Finger:
    def __init__(self):
        # Match the exact live Isaac Lab BinaryJointPositionAction tensor
        # layout: processed endpoints are per-environment ``(N, 1)`` tensors,
        # while the open/close command constants are ``(1,)`` tensors.
        self._gripper_target_slew_endpoint = torch.tensor(
            [[0.0]], dtype=torch.float32
        )
        self._gripper_target_slew_endpoint_seen = True
        self._gripper_target_slew_endpoint_change_count = 0
        self._open_command = torch.tensor([0.0], dtype=torch.float32)
        self._close_command = torch.tensor(
            [torch.pi / 4.0], dtype=torch.float32
        )

    def gripper_target_slew_static_contract(self):
        return {"profile": EEF_GRIPPER_TARGET_SLEW_PROFILE}

    def gripper_target_slew_dynamic_report(self):
        return {"profile": "target-slew-dynamic"}


def bare_action(*, apply_calls=0):
    action = object.__new__(robust.RobustDifferentialInverseKinematicsAction)
    action._env = types.SimpleNamespace(num_envs=1, device="cpu")
    data = types.SimpleNamespace(
        **{
            name: torch.zeros((1, 13), dtype=torch.float32)
            for name in (
                "joint_pos",
                "joint_vel",
                "joint_acc",
                "joint_pos_target",
                "joint_vel_target",
            )
        }
    )
    action._asset = types.SimpleNamespace(
        data=data,
        joint_names=[f"joint_{index}" for index in range(13)],
    )
    action._gripper_runtime_static = None
    action._apply_call_count = apply_calls
    return action


# Exercise the ordinary reset lifecycle without Isaac imports. The real
# constructor creates this exact per-joint tensor before the first lifecycle
# reset; a test object made with ``object.__new__`` must supply it explicitly.
reset_action = bare_action()
reset_action._raw_actions = torch.ones((1, 7), dtype=torch.float32)
reset_action._max_delta_joint_pos = torch.tensor(
    [PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S],
    dtype=torch.float32,
) * (1.0 / 120.0)
reset_action._wrist_energy_brake_enabled = False
reset_action._gripper_close_arm_interlock_anchor = torch.ones(7)
reset_action._gripper_close_arm_interlock_anchor_valid = True
reset_action._arm_release_ramp_phase = "ramp"
reset_action._arm_release_ramp_next_index = 7
reset_action._arm_release_ramp_started_count = 3
reset_action._arm_target_transaction_failed = True
reset_action.reset([0])
assert not reset_action._raw_actions.any()
assert not reset_action._gripper_close_arm_interlock_anchor.any()
assert reset_action._gripper_close_arm_interlock_anchor.shape == (7,)
assert reset_action._gripper_close_arm_interlock_anchor.dtype == torch.float32
assert reset_action._gripper_close_arm_interlock_anchor_valid is False
assert reset_action._gripper_close_arm_interlock_last_activation_apply_index is None
assert reset_action._arm_release_ramp_phase == "release"
assert reset_action._arm_release_ramp_next_index is None
assert reset_action._arm_release_ramp_started_count == 0
assert reset_action._arm_release_ramp_last_target_apply_index is None
assert reset_action._arm_release_ramp_last_index is None
assert not reset_action._arm_release_ramp_max_abs_nominal_to_ramped_target_change.any()
assert reset_action._arm_target_transaction_failed is False

finger = Finger()
action = bare_action()
action.install_gripper_runtime_contract(static, finger_term=finger)
assert action._gripper_target_slew_term is finger
assert action._gripper_runtime_static == static

# Exercise the live tensor shapes, not just the pure countdown transition.
# This is a regression test for the shape-strict torch.equal integration bug
# found by the first official controller-candidate replay.
action._gripper_close_arm_interlock_enabled = True
action._gripper_close_arm_interlock_observed_endpoint_change_count = 0
action._gripper_close_arm_interlock_endpoint_observed = False
action._gripper_close_arm_interlock_remaining = 0
opened = action._next_gripper_close_arm_interlock_transition()
assert opened.active is False
assert opened.endpoint_observed_after_successful_apply is True

finger._gripper_target_slew_endpoint.fill_(torch.pi / 4.0)
closed = action._next_gripper_close_arm_interlock_transition()
assert closed.active is True
assert closed.remaining_after_successful_apply == 47
assert closed.activation_count_delta == 1

finger._gripper_target_slew_endpoint.fill_(0.1)
try:
    action._next_gripper_close_arm_interlock_transition()
except ValueError as error:
    assert "binary endpoint" in str(error)
else:
    raise AssertionError("non-binary gripper endpoint was accepted")

finger._gripper_target_slew_endpoint.fill_(0.0)
valid_open = torch.tensor([0.0], dtype=torch.float32)
valid_close = torch.tensor([torch.pi / 4.0], dtype=torch.float32)
malformed_commands = (
    ("empty close", valid_open, torch.empty(0, dtype=torch.float32)),
    ("scalar open", torch.tensor(0.0, dtype=torch.float32), valid_close),
    ("wider close", valid_open, torch.tensor([0.0, 0.0], dtype=torch.float32)),
    ("wrong close dtype", valid_open, valid_close.to(torch.float64)),
    (
        "wrong close device",
        valid_open,
        torch.empty((1,), dtype=torch.float32, device="meta"),
    ),
    ("non-finite close", valid_open, torch.tensor([float("nan")])),
    ("drifted close", valid_open, torch.tensor([0.2], dtype=torch.float32)),
    ("drifted open", torch.tensor([0.2], dtype=torch.float32), valid_close),
    ("aliased endpoints", valid_open, valid_open.clone()),
)
for label, open_command, close_command in malformed_commands:
    finger._open_command = open_command
    finger._close_command = close_command
    try:
        action._next_gripper_close_arm_interlock_transition()
    except ValueError as error:
        assert "endpoint state drift" in str(error), label
    else:
        raise AssertionError(f"{label} was accepted")
finger._open_command = valid_open
finger._close_command = valid_close


class SetterAsset:
    def __init__(self, failure):
        self.failure = failure
        self.calls = []
        self.data = types.SimpleNamespace(
            joint_vel_target=torch.full((1, 7), -1.0, dtype=torch.float32),
            joint_pos_target=torch.full((1, 7), -1.0, dtype=torch.float32),
        )

    def set_joint_velocity_target(self, target, joint_ids):
        self.calls.append(("velocity", target.clone(), list(joint_ids)))
        if self.failure == "velocity":
            raise RuntimeError("forced velocity setter failure")
        self.data.joint_vel_target[:, joint_ids] = target

    def set_joint_position_target(self, target, joint_ids):
        self.calls.append(("position", target.clone(), list(joint_ids)))
        if self.failure == "position":
            raise RuntimeError("forced position setter failure")
        self.data.joint_pos_target[:, joint_ids] = (
            target + 1.0 if self.failure == "readback" else target
        )


anchor = torch.arange(7, dtype=torch.float32)
zero = torch.zeros(7, dtype=torch.float32)
staged = robust._StagedGripperCloseArmInterlockState(
    anchor=anchor,
    max_abs_current_anchor_residual=torch.ones(7),
    max_abs_target_anchor_residual=torch.full((7,), 0.5),
    max_abs_active_delta=torch.full((7,), 0.25),
    max_abs_released_delta=zero,
    anchor_valid=True,
    anchor_capture_count=1,
    anchor_target_apply_count=1,
    anchor_first_exact_target_count=1,
    anchor_slew_limit_event_count=0,
    anchor_slew_limited_joint_count=0,
    anchor_position_limit_event_count=0,
    anchor_position_limited_joint_count=0,
    anchor_completion_count=0,
    anchor_open_cancel_count=0,
    last_activation_apply_index=920,
    remaining=85,
    observed_endpoint_change_count=1,
    endpoint_observed=True,
    activation_count=1,
    active_apply_count=1,
    released_apply_count=0,
)
staged_release = robust._StagedArmReleaseRampState(
    phase="hold",
    next_index=None,
    release_observed_count=0,
    ramp_started_count=0,
    ramp_completed_count=0,
    ramp_cancelled_by_reactivation_count=0,
    ramp_target_apply_count=0,
    cancelled_ramp_target_apply_count=0,
    ramp_limited_target_apply_count=0,
    ramp_limited_joint_target_count=0,
    last_target_apply_index=None,
    last_ramp_index=None,
    max_abs_nominal_to_ramped_target_change=zero,
)
safe_target = torch.zeros((1, 7), dtype=torch.float32)
for failure, expected_calls in (("velocity", 1), ("position", 2)):
    transactional = object.__new__(
        robust.RobustDifferentialInverseKinematicsAction
    )
    transactional._asset = SetterAsset(failure)
    transactional._zero_joint_velocity_target = zero.unsqueeze(0)
    transactional._joint_ids = list(range(7))
    transactional._arm_release_ramp_enabled = True
    retained_anchor = torch.full((7,), -1.0)
    transactional._gripper_close_arm_interlock_anchor = retained_anchor
    transactional._gripper_close_arm_interlock_anchor_valid = True
    transactional._gripper_close_arm_interlock_anchor_capture_count = 1
    transactional._gripper_close_arm_interlock_anchor_target_apply_count = 48
    transactional._gripper_close_arm_interlock_remaining = 38
    transactional._gripper_close_arm_interlock_activation_count = 1
    transactional._gripper_close_arm_interlock_active_apply_count = 48
    transactional._arm_release_ramp_phase = "release"
    transactional._arm_release_ramp_next_index = None
    transactional._arm_release_observed_count = 0
    transactional._arm_target_transaction_failed = False
    try:
        transactional._set_targets_and_commit_gripper_close_arm_interlock(
            safe_target, staged, staged_release, None
        )
    except RuntimeError as error:
        assert f"forced {failure} setter failure" in str(error)
    else:
        raise AssertionError(f"{failure} setter failure was swallowed")
    assert len(transactional._asset.calls) == expected_calls
    assert transactional._gripper_close_arm_interlock_anchor is retained_anchor
    assert transactional._gripper_close_arm_interlock_anchor_valid is True
    assert transactional._gripper_close_arm_interlock_anchor_capture_count == 1
    assert transactional._gripper_close_arm_interlock_anchor_target_apply_count == 48
    assert transactional._gripper_close_arm_interlock_remaining == 38
    assert transactional._gripper_close_arm_interlock_activation_count == 1
    assert transactional._gripper_close_arm_interlock_active_apply_count == 48
    assert transactional._arm_release_ramp_phase == "release"
    assert transactional._arm_release_ramp_next_index is None
    assert transactional._arm_release_observed_count == 0
    assert transactional._arm_target_transaction_failed is True

for failure in ("readback", "trace"):
    transactional = object.__new__(
        robust.RobustDifferentialInverseKinematicsAction
    )
    transactional._asset = SetterAsset(
        "readback" if failure == "readback" else None
    )
    transactional._zero_joint_velocity_target = zero.unsqueeze(0)
    transactional._joint_ids = list(range(7))
    transactional._arm_release_ramp_enabled = True
    transactional._gripper_close_arm_interlock_anchor = torch.full((7,), -1.0)
    transactional._gripper_close_arm_interlock_anchor_valid = True
    transactional._gripper_close_arm_interlock_anchor_capture_count = 1
    transactional._gripper_close_arm_interlock_anchor_target_apply_count = 48
    transactional._gripper_close_arm_interlock_remaining = 38
    transactional._gripper_close_arm_interlock_activation_count = 1
    transactional._gripper_close_arm_interlock_active_apply_count = 48
    transactional._arm_release_ramp_phase = "release"
    transactional._arm_release_ramp_next_index = None
    transactional._arm_release_observed_count = 0
    transactional._arm_target_transaction_failed = False
    if failure == "trace":
        transactional._stage_failure_substep_trace = lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("forced trace staging failure")
        )
    try:
        transactional._set_targets_and_commit_gripper_close_arm_interlock(
            safe_target,
            staged,
            staged_release,
            {
                "new_joint_pos_target": safe_target,
                "new_joint_vel_target": zero.unsqueeze(0),
            }
            if failure == "trace"
            else None,
        )
    except (RuntimeError, ValueError) as error:
        assert failure in str(error)
    else:
        raise AssertionError(f"{failure} failure was swallowed")
    assert transactional._gripper_close_arm_interlock_remaining == 38
    assert transactional._arm_release_ramp_phase == "release"
    assert transactional._arm_release_ramp_next_index is None
    assert transactional._arm_release_observed_count == 0
    assert transactional._arm_target_transaction_failed is True

transactional = object.__new__(robust.RobustDifferentialInverseKinematicsAction)
transactional._asset = SetterAsset(None)
transactional._zero_joint_velocity_target = zero.unsqueeze(0)
transactional._joint_ids = list(range(7))
transactional._arm_release_ramp_enabled = True
transactional._arm_target_transaction_failed = False
transactional._set_targets_and_commit_gripper_close_arm_interlock(
    safe_target, staged, staged_release, None
)
assert [entry[0] for entry in transactional._asset.calls] == [
    "velocity", "position"
]
assert transactional._gripper_close_arm_interlock_anchor is anchor
assert transactional._gripper_close_arm_interlock_anchor_valid is True
assert transactional._gripper_close_arm_interlock_anchor_capture_count == 1
assert transactional._gripper_close_arm_interlock_anchor_target_apply_count == 1
assert transactional._gripper_close_arm_interlock_anchor_first_exact_target_count == 1
assert transactional._gripper_close_arm_interlock_remaining == 85
assert transactional._gripper_close_arm_interlock_activation_count == 1
assert transactional._gripper_close_arm_interlock_active_apply_count == 1
assert transactional._gripper_close_arm_interlock_last_activation_apply_index == 920
assert transactional._arm_release_ramp_phase == "hold"
assert transactional._arm_release_ramp_next_index is None
assert transactional._arm_release_observed_count == 0
assert transactional._arm_target_transaction_failed is False


class ParentPathAsset:
    def __init__(self):
        self.calls = []

    def set_joint_velocity_target(self, target, joint_ids):
        self.calls.append(("velocity", target.clone(), list(joint_ids)))

    def set_joint_position_target(self, target, joint_ids):
        self.calls.append(("position", target.clone(), list(joint_ids)))


parent_path = object.__new__(robust.RobustDifferentialInverseKinematicsAction)
parent_path._asset = ParentPathAsset()
parent_path._zero_joint_velocity_target = zero.unsqueeze(0)
parent_path._joint_ids = list(range(7))
parent_path._arm_release_ramp_enabled = False
parent_path._set_targets_and_commit_gripper_close_arm_interlock(
    safe_target, staged, None, None
)
assert [entry[0] for entry in parent_path._asset.calls] == [
    "velocity", "position"
]
assert not hasattr(parent_path._asset, "data")
assert parent_path._gripper_close_arm_interlock_remaining == 85
assert not hasattr(parent_path, "_arm_release_ramp_phase")

report = action._gripper_runtime_dynamic_report()
assert report["driver_target_slew"] == {"profile": "target-slew-dynamic"}
assert captured_dynamic[-1] is report

for invalid_finger in (object(), types.SimpleNamespace(
    gripper_target_slew_static_contract=lambda: {"profile": "wrong"},
    gripper_target_slew_dynamic_report=lambda: {},
)):
    try:
        bare_action().install_gripper_runtime_contract(
            static, finger_term=invalid_finger
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid finger target-slew evidence was accepted")

try:
    bare_action(apply_calls=1).install_gripper_runtime_contract(
        static, finger_term=finger
    )
except ValueError:
    pass
else:
    raise AssertionError("post-apply gripper contract installation was accepted")

print("robust_target_slew_host_stub_ok")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    env["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 0, completed.stderr
    assert "robust_target_slew_host_stub_ok" in completed.stdout
