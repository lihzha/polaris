from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_robust_action_binds_finger_target_slew_report_in_isolated_host_stub():
    root = Path(__file__).parents[1]
    code = r"""
import sys
import types
import inspect
import copy

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
        pass


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
            joint_effort_target=torch.zeros((1, 7), dtype=torch.float32),
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

    def set_joint_effort_target(self, target, joint_ids):
        self.calls.append(("effort", target.clone(), list(joint_ids)))
        if self.failure == "effort":
            raise RuntimeError("forced effort setter failure")
        self.data.joint_effort_target[:, joint_ids] = (
            target + 1.0 if self.failure == "effort_readback" else target
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


def recovery_transaction_action(failure=None):
    value = object.__new__(robust.RobustDifferentialInverseKinematicsAction)
    value._env = types.SimpleNamespace(num_envs=1, device="cpu")
    value._asset = SetterAsset(failure)
    value._zero_joint_velocity_target = zero.unsqueeze(0)
    value._zero_joint_effort_target = zero.unsqueeze(0)
    value._joint_ids = list(range(7))
    value._num_joints = 7
    value._joint_names = tuple(f"panda_joint{index}" for index in range(1, 8))
    value._joint_velocity_limits = torch.tensor(
        [[2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]],
        dtype=torch.float32,
    )
    value._current_joint_velocity_recovery_envelopes = torch.tensor(
        [[
            robust.current_joint_velocity_recovery_envelope(item)
            for item in value._joint_velocity_limits[0].tolist()
        ]],
        dtype=torch.float32,
    )
    value._apply_call_count = 1
    value._decimation = 8
    value._active_episode_index = 0
    value._current_joint_velocity_recovery_enabled = True
    value._current_joint_velocity_recovery_phase = "inactive"
    value._current_joint_velocity_recovery_consecutive_active_substeps = 0
    value._current_joint_velocity_recovery_consecutive_clean_samples = 0
    value._current_joint_velocity_recovery_next_release_ramp_index = None
    value._current_joint_velocity_recovery_events = []
    value._current_joint_velocity_recovery_events_count = 0
    value._current_joint_velocity_recovery_active_substeps = 0
    value._current_joint_velocity_recovery_recovered_events = 0
    value._current_joint_velocity_recovery_hold_target_applies = 0
    value._current_joint_velocity_recovery_release_ramp_target_applies = 0
    value._current_joint_velocity_recovery_transaction_aborts = 0
    value._current_joint_velocity_recovery_max_consecutive_substeps = 0
    value._invariant_abort_count = torch.zeros((), dtype=torch.int64)
    value._guard_diagnostics = []
    value._guard_diagnostics_dropped = 0
    value._max_guard_diagnostics = 32
    value._arm_release_ramp_enabled = True
    value._arm_target_transaction_failed = False
    value._gripper_close_arm_interlock_anchor = torch.full((7,), -1.0)
    value._gripper_close_arm_interlock_remaining = 38
    value._arm_release_ramp_phase = "release"
    value._arm_release_ramp_next_index = None
    value._arm_release_observed_count = 0
    value._stage_failure_substep_trace = lambda **_kwargs: None
    return value


inactive_transition = robust.advance_current_joint_velocity_recovery(
    enabled=True,
    phase_before_apply="inactive",
    consecutive_active_substeps_before_apply=0,
    consecutive_clean_samples_before_apply=0,
    next_release_ramp_index_before_apply=None,
    measured_velocity_over_envelope=False,
)
normal_v5 = recovery_transaction_action()
normal_v5._set_targets_and_commit_gripper_close_arm_interlock(
    safe_target,
    staged,
    staged_release,
    None,
    inactive_transition,
    safe_target,
    zero.unsqueeze(0),
    safe_target,
    torch.ones((1, 7), dtype=torch.float32),
    False,
)
assert [entry[0] for entry in normal_v5._asset.calls] == ["velocity", "position"]
assert normal_v5._current_joint_velocity_recovery_events == []

hold_transition = robust.advance_current_joint_velocity_recovery(
    enabled=True,
    phase_before_apply="inactive",
    consecutive_active_substeps_before_apply=0,
    consecutive_clean_samples_before_apply=0,
    next_release_ramp_index_before_apply=None,
    measured_velocity_over_envelope=True,
)
recovery_v5 = recovery_transaction_action()
recovery_trace = {
    "new_joint_pos_target": safe_target,
    "new_joint_vel_target": zero.unsqueeze(0),
    "new_joint_effort_target": zero.unsqueeze(0),
}
recovery_v5._set_targets_and_commit_gripper_close_arm_interlock(
    safe_target,
    staged,
    staged_release,
    recovery_trace,
    hold_transition,
    safe_target,
    torch.tensor([[11.743, 0, 0, 0, 0, 0, 0]], dtype=torch.float32),
    safe_target,
    torch.ones((1, 7), dtype=torch.float32),
    True,
)
assert [entry[0] for entry in recovery_v5._asset.calls] == [
    "position", "velocity", "effort"
]
assert recovery_v5._current_joint_velocity_recovery_phase == "hold"
assert len(recovery_v5._current_joint_velocity_recovery_events) == 1
assert recovery_v5._current_joint_velocity_recovery_events[0]["end_reason"] is None

for failure in ("effort", "effort_readback"):
    failed_v5 = recovery_transaction_action(failure)
    retained_anchor = failed_v5._gripper_close_arm_interlock_anchor
    try:
        failed_v5._set_targets_and_commit_gripper_close_arm_interlock(
            safe_target,
            staged,
            staged_release,
            recovery_trace,
            hold_transition,
            safe_target,
            torch.tensor([[11.743, 0, 0, 0, 0, 0, 0]], dtype=torch.float32),
            safe_target,
            torch.ones((1, 7), dtype=torch.float32),
            True,
        )
    except robust.DifferentialIKInvariantError as error:
        assert "target transaction failed" in str(error)
    else:
        raise AssertionError(f"v5 {failure} failure was swallowed")
    assert [entry[0] for entry in failed_v5._asset.calls] == [
        "position", "velocity", "effort"
    ]
    assert failed_v5._gripper_close_arm_interlock_anchor is retained_anchor
    assert failed_v5._gripper_close_arm_interlock_remaining == 38
    assert failed_v5._current_joint_velocity_recovery_transaction_aborts == 1
    assert failed_v5._current_joint_velocity_recovery_phase == "inactive"
    assert len(failed_v5._current_joint_velocity_recovery_events) == 1
    assert failed_v5._current_joint_velocity_recovery_events[0]["end_reason"] == (
        "transaction_abort"
    )
    terminal_snapshot = failed_v5._current_joint_velocity_recovery_events[0]["last"]
    assert terminal_snapshot["hold_target_rad"] == [0.0] * 7
    assert terminal_snapshot["hold_position_target_readback_rad"] is None
    assert terminal_snapshot["hold_velocity_target_readback_rad_s"] is None
    assert terminal_snapshot["hold_effort_target_readback_nm"] is None


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

# Clean sample 2 is still recovery-owned: it emits release-ramp index zero,
# which is exactly current q, and must not enter either the Jacobian or DLS
# path.  Exercise the pure transition and the live action dispatcher together.
recovery_start = robust.advance_current_joint_velocity_recovery(
    enabled=True,
    phase_before_apply="inactive",
    consecutive_active_substeps_before_apply=0,
    consecutive_clean_samples_before_apply=0,
    next_release_ramp_index_before_apply=None,
    measured_velocity_over_envelope=True,
)
clean_one = robust.advance_current_joint_velocity_recovery(
    enabled=True,
    phase_before_apply=recovery_start.phase_after_successful_apply,
    consecutive_active_substeps_before_apply=(
        recovery_start.consecutive_active_substeps_after_successful_apply
    ),
    consecutive_clean_samples_before_apply=(
        recovery_start.consecutive_clean_samples_after_successful_apply
    ),
    next_release_ramp_index_before_apply=(
        recovery_start.next_release_ramp_index_after_successful_apply
    ),
    measured_velocity_over_envelope=False,
)
clean_two = robust.advance_current_joint_velocity_recovery(
    enabled=True,
    phase_before_apply=clean_one.phase_after_successful_apply,
    consecutive_active_substeps_before_apply=(
        clean_one.consecutive_active_substeps_after_successful_apply
    ),
    consecutive_clean_samples_before_apply=(
        clean_one.consecutive_clean_samples_after_successful_apply
    ),
    next_release_ramp_index_before_apply=(
        clean_one.next_release_ramp_index_after_successful_apply
    ),
    measured_velocity_over_envelope=False,
)
assert clean_two.skip_dls is True
assert clean_two.release_ramp_index_to_apply == 0
dls_probe = object.__new__(robust.RobustDifferentialInverseKinematicsAction)
dls_probe._compute_frame_jacobian = lambda: (_ for _ in ()).throw(
    AssertionError("clean sample 2 entered the Jacobian path")
)
dls_probe._ik_controller = types.SimpleNamespace(
    compute=lambda *_args: (_ for _ in ()).throw(
        AssertionError("clean sample 2 entered DLS")
    )
)
joint_position = torch.arange(7, dtype=torch.float32).unsqueeze(0)
recovery_target, recovery_jacobian = (
    dls_probe._compute_recovery_aware_joint_position_target(
        recovery_transition=clean_two,
        ee_pos_curr=torch.zeros((1, 3), dtype=torch.float32),
        ee_quat_curr=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        joint_pos=joint_position,
    )
)
assert torch.equal(recovery_target, joint_position)
assert recovery_target.data_ptr() != joint_position.data_ptr()
assert recovery_jacobian is None

# Preserve the inherited v3/v4 failure precedence: malformed lower endpoint
# state must be rejected before frame/Jacobian/DLS staging, irrespective of
# whether the lower arm-release ramp is enabled.
for release_ramp_enabled in (False, True):
    legacy = object.__new__(robust.RobustDifferentialInverseKinematicsAction)
    legacy._arm_release_ramp_enabled = release_ramp_enabled
    legacy._arm_target_transaction_failed = False
    legacy._failure_substep_trace_enabled = False
    legacy._apply_call_count = 0
    legacy._gripper_close_arm_interlock_enabled = True
    legacy._current_joint_velocity_recovery_enabled = False
    legacy._validate_arm_release_ramp_state = lambda **_kwargs: None
    legacy._validate_gripper_close_arm_interlock_anchor_state = lambda: None
    legacy._next_gripper_close_arm_interlock_transition = lambda: (
        (_ for _ in ()).throw(ValueError("malformed endpoint wins"))
    )
    legacy._compute_frame_pose = lambda: (_ for _ in ()).throw(
        AssertionError("legacy endpoint validation ran after frame/DLS")
    )
    try:
        legacy.apply_actions()
    except ValueError as error:
        assert str(error) == "malformed endpoint wins"
    else:
        raise AssertionError("legacy malformed endpoint was accepted")

# A lower release ramp frozen at index five can survive an arbitrarily long
# v5-owned interval.  The first inactive apply accepts only the one stale
# target created by the immediately completed recovery event, advances exact
# index five, and restores strict latest-target validation thereafter.
lower = object.__new__(robust.RobustDifferentialInverseKinematicsAction)
lower._env = types.SimpleNamespace(num_envs=1, device="cpu")
lower._num_joints = 7
lower._apply_call_count = 28
lower._arm_release_ramp_enabled = True
lower._arm_release_ramp_phase = "ramp"
lower._arm_release_ramp_next_index = 5
lower._gripper_close_arm_interlock_remaining = 0
lower._gripper_close_arm_interlock_anchor_completion_count = 1
lower._gripper_close_arm_interlock_anchor_open_cancel_count = 0
lower._arm_release_observed_count = 1
lower._arm_release_ramp_started_count = 1
lower._arm_release_ramp_completed_count = 0
lower._arm_release_ramp_cancelled_by_reactivation_count = 0
lower._arm_release_ramp_target_apply_count = 5
lower._arm_release_ramp_cancelled_target_apply_count = 0
lower._arm_release_ramp_limited_target_apply_count = 0
lower._arm_release_ramp_limited_joint_target_count = 0
lower._arm_release_ramp_last_target_apply_index = 10
lower._arm_release_ramp_last_index = 4
lower._arm_release_ramp_max_abs_nominal_to_ramped_target_change = torch.zeros(
    7, dtype=torch.float32
)
lower._nominal_max_delta_joint_pos = torch.full(
    (1, 7), 0.02, dtype=torch.float32
)
lower._current_joint_velocity_recovery_enabled = True
lower._current_joint_velocity_recovery_phase = "inactive"
lower._current_joint_velocity_recovery_recovered_events = 1
lower._current_joint_velocity_recovery_events = [{
    "end_reason": "clean2_release_ramp_complete",
    "end_apply_index": 27,
    "recovery_completed_apply_index": 27,
    "deferred_lower_endpoint_transition_count": None,
    "lower_endpoint_transition_overflow_context": None,
}]
for _ in range(16):
    frozen = robust.suspend_arm_release_ramp(
        phase_before_apply=lower._arm_release_ramp_phase,
        next_ramp_index_before_apply=lower._arm_release_ramp_next_index,
    )
    assert frozen.phase_after_successful_apply == "ramp"
    assert frozen.next_ramp_index_after_successful_apply == 5
    assert frozen.ramp_index_to_apply is None
try:
    lower._validate_arm_release_ramp_state(require_latest_target=True)
except ValueError as error:
    assert "phase/target drift" in str(error)
else:
    raise AssertionError("stale lower-ramp target passed strict validation")
assert lower._current_joint_velocity_recovery_completed_previous_apply() is True
lower._validate_arm_release_ramp_state(
    require_latest_target=(
        not lower._current_joint_velocity_recovery_enabled
        or (
            lower._current_joint_velocity_recovery_phase == "inactive"
            and not lower._current_joint_velocity_recovery_completed_previous_apply()
        )
    )
)
lower_resume = robust.advance_arm_release_ramp(
    enabled=True,
    phase_before_apply=lower._arm_release_ramp_phase,
    next_ramp_index_before_apply=lower._arm_release_ramp_next_index,
    interlock_remaining_before_apply=0,
    interlock_active_this_apply=False,
    interlock_remaining_after_apply=0,
    interlock_activation_count_delta=0,
)
assert lower_resume.ramp_index_to_apply == 5
lower._apply_call_count += 1
lower._arm_release_ramp_phase = lower_resume.phase_after_successful_apply
lower._arm_release_ramp_next_index = (
    lower_resume.next_ramp_index_after_successful_apply
)
lower._arm_release_ramp_target_apply_count += 1
lower._arm_release_ramp_last_target_apply_index = lower._apply_call_count - 1
lower._arm_release_ramp_last_index = lower_resume.ramp_index_to_apply
lower._validate_arm_release_ramp_state(require_latest_target=True)

# One deferred endpoint transition is observed without consuming the lower
# counter, then processed on the first inactive resume.
one_flip_finger = Finger()
one_flip_finger._gripper_target_slew_endpoint.fill_(torch.pi / 4.0)
one_flip_finger._gripper_target_slew_endpoint_change_count = 1
one_flip = bare_action()
one_flip._gripper_target_slew_term = one_flip_finger
one_flip._gripper_close_arm_interlock_enabled = True
one_flip._gripper_close_arm_interlock_configured_substeps = 48
one_flip._gripper_close_arm_interlock_observed_endpoint_change_count = 0
one_flip._gripper_close_arm_interlock_endpoint_observed = True
one_flip._gripper_close_arm_interlock_remaining = 0
assert one_flip._deferred_gripper_endpoint_transition_count() == 1
one_flip._abort_on_deferred_gripper_endpoint_transition_overflow(
    lower_controller_suspended=False,
    recovery_completed_previous_apply=True,
    snapshot={},
)
frozen_interlock = robust.suspend_gripper_close_arm_interlock(
    remaining_before_apply=0,
    observed_endpoint_change_count=0,
    endpoint_observed_before_apply=True,
)
assert frozen_interlock.observed_endpoint_change_count == 0
resumed_interlock = one_flip._next_gripper_close_arm_interlock_transition()
assert resumed_interlock.observed_endpoint_change_count == 1
assert resumed_interlock.activation_count_delta == 1

zero_flip = bare_action()
zero_flip._gripper_target_slew_term = types.SimpleNamespace(
    _gripper_target_slew_endpoint_change_count=0
)
zero_flip._gripper_close_arm_interlock_observed_endpoint_change_count = 0
zero_flip._abort_on_deferred_gripper_endpoint_transition_overflow(
    lower_controller_suspended=False,
    recovery_completed_previous_apply=True,
    snapshot={},
)

# A second deferred endpoint transition terminalizes the active v5 event and
# emits a rollout diagnostic with an event digest.  The call site must remain
# before DLS dispatch and every target setter.
overflow = recovery_transaction_action()
overflow._current_joint_velocity_recovery_lower_endpoint_transition_aborts = 0
overflow._gripper_close_arm_interlock_observed_endpoint_change_count = 0
overflow._gripper_target_slew_term = types.SimpleNamespace(
    _gripper_target_slew_endpoint_change_count=2
)
overflow._current_joint_velocity_recovery_phase = "hold"
overflow._current_joint_velocity_recovery_consecutive_active_substeps = 1
joint_position = torch.tensor(
    [[0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.0]], dtype=torch.float32
)
joint_velocity = torch.tensor(
    [[11.743, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32
)
hard_limits = torch.tensor(
    robust.PANDA_PHYSX_HARD_JOINT_POS_LIMITS_RAD,
    dtype=torch.float32,
).unsqueeze(0)
predicted_position = joint_position + joint_velocity * torch.tensor(
    robust.PANDA_EEF_PHYSICS_DT_FLOAT32, dtype=torch.float32
)
predicted_clearance = torch.minimum(
    predicted_position - hard_limits[..., 0],
    hard_limits[..., 1] - predicted_position,
)
overflow_snapshot = overflow._current_joint_velocity_recovery_snapshot(
    joint_pos=joint_position,
    joint_vel=joint_velocity,
    predicted_joint_pos=predicted_position,
    predicted_hard_limit_clearance=predicted_clearance,
)
overflow._start_current_joint_velocity_recovery_event(
    reason="measured_velocity_above_float32_envelope",
    snapshot=overflow_snapshot,
)
try:
    overflow._abort_on_deferred_gripper_endpoint_transition_overflow(
        lower_controller_suspended=True,
        recovery_completed_previous_apply=False,
        snapshot=overflow_snapshot,
    )
except robust.DifferentialIKInvariantError as error:
    overflow_message = str(error)
    assert "more than one deferred gripper endpoint transition" in overflow_message
    digest = overflow_message.rsplit("evidence_sha256=", 1)[1].rstrip(")")
    assert len(digest) == 64
    int(digest, 16)
else:
    raise AssertionError("two deferred endpoint transitions were accepted")
assert overflow._current_joint_velocity_recovery_phase == "inactive"
assert overflow._current_joint_velocity_recovery_lower_endpoint_transition_aborts == 1
assert overflow._current_joint_velocity_recovery_events[-1]["end_reason"] == (
    "lower_endpoint_transition_overflow_abort"
)
assert overflow._current_joint_velocity_recovery_events[-1][
    "deferred_lower_endpoint_transition_count"
] == 2
assert overflow._current_joint_velocity_recovery_events[-1][
    "lower_endpoint_transition_overflow_context"
] == "active_recovery"
assert overflow._guard_diagnostics[-1]["kind"] == (
    "measured_velocity_recovery_lower_endpoint_transition_abort"
)

# Two flips arriving on the one stale-target resume apply must reclassify the
# immediately completed measured-recovery event, preserve its truthful ramp
# completion/recovered count, and raise the same typed digest-bound guard.
stale_resume = recovery_transaction_action()
stale_resume._apply_call_count = 18
stale_resume._current_joint_velocity_recovery_phase = "inactive"
stale_resume._current_joint_velocity_recovery_recovered_events = 1
stale_resume._current_joint_velocity_recovery_events_count = 1
start_snapshot = copy.deepcopy(overflow_snapshot)
start_snapshot.update({"apply_index": 0, "policy_step": 0, "physics_substep": 0})
start_snapshot["hold_target_rad"] = list(start_snapshot["joint_pos_rad"])
start_snapshot["hold_position_target_readback_rad"] = list(
    start_snapshot["joint_pos_rad"]
)
start_snapshot["hold_velocity_target_readback_rad_s"] = [0.0] * 7
start_snapshot["hold_effort_target_readback_nm"] = [0.0] * 7
completed_snapshot = copy.deepcopy(start_snapshot)
completed_snapshot.update({
    "apply_index": 17,
    "policy_step": 2,
    "physics_substep": 1,
})
stale_resume._current_joint_velocity_recovery_events = [{
    "event_index": 0,
    "start_apply_index": 0,
    "end_apply_index": 17,
    "start_reason": "measured_velocity_above_float32_envelope",
    "end_reason": "clean2_release_ramp_complete",
    "deferred_lower_endpoint_transition_count": None,
    "lower_endpoint_transition_overflow_context": None,
    "recovery_completed_apply_index": 17,
    "start": start_snapshot,
    "last": completed_snapshot,
}]
assert stale_resume._current_joint_velocity_recovery_completed_previous_apply() is True
stale_resume._apply_call_count += 1
stale_resume._current_joint_velocity_recovery_lower_endpoint_transition_aborts = 0
stale_resume._gripper_close_arm_interlock_observed_endpoint_change_count = 0
stale_resume._gripper_target_slew_term = types.SimpleNamespace(
    _gripper_target_slew_endpoint_change_count=2
)
stale_snapshot = stale_resume._current_joint_velocity_recovery_snapshot(
    joint_pos=joint_position,
    joint_vel=torch.zeros_like(joint_velocity),
    predicted_joint_pos=joint_position,
    predicted_hard_limit_clearance=torch.minimum(
        joint_position - hard_limits[..., 0],
        hard_limits[..., 1] - joint_position,
    ),
)
try:
    stale_resume._abort_on_deferred_gripper_endpoint_transition_overflow(
        lower_controller_suspended=False,
        recovery_completed_previous_apply=True,
        snapshot=stale_snapshot,
    )
except robust.DifferentialIKInvariantError as error:
    stale_message = str(error)
    assert "more than one deferred gripper endpoint transition" in stale_message
    stale_digest = stale_message.rsplit("evidence_sha256=", 1)[1].rstrip(")")
    assert len(stale_digest) == 64
    int(stale_digest, 16)
else:
    raise AssertionError("stale-resume endpoint overflow was accepted")
assert stale_resume._current_joint_velocity_recovery_events_count == 1
assert stale_resume._current_joint_velocity_recovery_recovered_events == 1
stale_event = stale_resume._current_joint_velocity_recovery_events[0]
assert stale_event["start_reason"] == "measured_velocity_above_float32_envelope"
assert stale_event["end_reason"] == "lower_endpoint_transition_overflow_abort"
assert stale_event["end_apply_index"] == 18
assert stale_event["recovery_completed_apply_index"] == 17
assert stale_event["deferred_lower_endpoint_transition_count"] == 2
assert stale_event["lower_endpoint_transition_overflow_context"] == (
    "post_recovery_resume"
)
assert stale_resume._guard_diagnostics[-1]["kind"] == (
    "measured_velocity_recovery_lower_endpoint_transition_abort"
)
apply_source = inspect.getsource(
    robust.RobustDifferentialInverseKinematicsAction.apply_actions
)
overflow_guard = apply_source.index(
    "self._abort_on_deferred_gripper_endpoint_transition_overflow("
)
dls_dispatch = apply_source.index(
    "self._compute_recovery_aware_joint_position_target("
)
target_setter = apply_source.index(
    "self._set_targets_and_commit_gripper_close_arm_interlock("
)
assert overflow_guard < dls_dispatch < target_setter
v5_next_gripper = apply_source.index(
    "self._next_gripper_close_arm_interlock_transition()",
    dls_dispatch,
)
assert overflow_guard < dls_dispatch < v5_next_gripper < target_setter

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
