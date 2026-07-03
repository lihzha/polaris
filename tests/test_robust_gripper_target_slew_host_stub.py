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
