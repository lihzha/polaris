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

robust.PINNED_DYNAMIC_DEVICE = "cpu"
robust.validate_eef_gripper_static_contract = lambda value: dict(value)
captured_dynamic = []


def validate_dynamic(value):
    captured_dynamic.append(value)
    return value


robust.validate_eef_gripper_dynamic_evidence = validate_dynamic
static = {"driver_target_slew": {"profile": "target-slew-static"}}


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
        return {"profile": "target-slew-static"}

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
