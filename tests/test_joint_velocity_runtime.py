from types import SimpleNamespace

import numpy as np
import pytest
import torch

import polaris.joint_velocity_runtime as runtime
from polaris.pi05_droid_jointvelocity_contract import (
    PANDA_ARM_EFFORT_LIMITS,
    PANDA_ARM_JOINT_NAMES,
    PANDA_ARM_VELOCITY_LIMITS,
    PI05_DROID_JOINTVELOCITY_PROFILE,
    PI05_DROID_ISAACLAB_SOURCE_SHA256,
    PI05_DROID_POLARIS_RUNTIME_SOURCE_SHA256,
)


def _named_class(module, name):
    return type(name, (), {"__module__": module})


JointVelocityAction = _named_class(
    "isaaclab.envs.mdp.actions.joint_actions", "JointVelocityAction"
)
JointVelocityActionCfg = _named_class(
    "isaaclab.envs.mdp.actions.actions_cfg", "JointVelocityActionCfg"
)
ImplicitActuator = _named_class("isaaclab.actuators.actuator_pd", "ImplicitActuator")
BinaryGripperAction = _named_class(
    "polaris.environments.droid_cfg", "BinaryJointPositionZeroToOneAction"
)


class _View:
    def __init__(self, stiffness, damping, effort, velocity):
        self.stiffness = stiffness
        self.damping = damping
        self.effort = effort
        self.velocity = velocity

    def get_dof_stiffnesses(self):
        return self.stiffness

    def get_dof_dampings(self):
        return self.damping

    def get_dof_max_forces(self):
        return self.effort

    def get_dof_max_velocities(self):
        return self.velocity


class _Robot:
    def __init__(self):
        stiffness = torch.zeros((1, 7), dtype=torch.float32)
        damping = torch.full((1, 7), 80.0, dtype=torch.float32)
        effort = torch.tensor([PANDA_ARM_EFFORT_LIMITS], dtype=torch.float32)
        velocity = torch.tensor([PANDA_ARM_VELOCITY_LIMITS], dtype=torch.float32)
        self.data = SimpleNamespace(
            joint_stiffness=stiffness.clone(),
            joint_damping=damping.clone(),
            joint_effort_limits=effort.clone(),
            joint_vel_limits=velocity.clone(),
        )
        self.root_physx_view = _View(
            stiffness.clone(), damping.clone(), effort.clone(), velocity.clone()
        )
        self.actuators = {
            "panda_shoulder": ImplicitActuator(),
            "panda_forearm": ImplicitActuator(),
        }
        self.cfg = SimpleNamespace(actuators={"panda_shoulder": SimpleNamespace()})

    def find_joints(self, names, preserve_order=False):
        assert preserve_order is True
        return list(range(7)), list(names)


class _Env:
    def __init__(self):
        cfg = JointVelocityActionCfg()
        cfg.preserve_order = True
        cfg.use_default_offset = False
        arm = JointVelocityAction()
        arm.cfg = cfg
        arm._joint_names = list(PANDA_ARM_JOINT_NAMES)
        arm._scale = 1.0
        arm._offset = 0.0
        arm._clip = torch.tensor(
            np.broadcast_to(
                np.asarray([-1.0, 1.0], dtype=np.float32), (1, 7, 2)
            ).copy(),
            dtype=torch.float32,
        )
        finger = BinaryGripperAction()
        finger._joint_names = ["finger_joint"]
        finger._open_command = torch.zeros((1, 1), dtype=torch.float32)
        finger._close_command = torch.full((1, 1), np.pi / 4.0, dtype=torch.float32)
        self.cfg = SimpleNamespace(decimation=8, sim=SimpleNamespace(dt=1.0 / 120.0))
        self.action_manager = SimpleNamespace(
            _terms={"arm": arm, "finger_joint": finger}
        )
        self.scene = {"robot": _Robot()}

    @property
    def unwrapped(self):
        return self


def _stub_isaaclab(monkeypatch):
    monkeypatch.setattr(runtime, "_installed_isaaclab_version", lambda: "2.3.0")
    monkeypatch.setattr(
        runtime,
        "_verify_isaaclab_sources",
        lambda **_: dict(PI05_DROID_ISAACLAB_SOURCE_SHA256),
    )
    monkeypatch.setattr(
        runtime,
        "_verify_polaris_sources",
        lambda **_: dict(PI05_DROID_POLARIS_RUNTIME_SOURCE_SHA256),
    )


def test_runtime_binds_action_class_order_affine_and_direct_physx_drive(monkeypatch):
    _stub_isaaclab(monkeypatch)
    report = runtime.validate_joint_velocity_runtime(_Env())

    assert report["status"] == "pass"
    assert report["profile"] == PI05_DROID_JOINTVELOCITY_PROFILE
    assert report["joint_names"] == list(PANDA_ARM_JOINT_NAMES)
    assert report["position_integration"] == "absent_by_exact_action_class"
    assert report["velocity_drive"]["position_stiffness"] == 0.0
    assert report["velocity_drive"]["velocity_damping"] == 80.0
    assert report["isaaclab_version"] == "2.3.0"
    assert report["isaaclab_source_sha256"] == PI05_DROID_ISAACLAB_SOURCE_SHA256
    assert (
        report["polaris_runtime_source_sha256"]
        == PI05_DROID_POLARIS_RUNTIME_SOURCE_SHA256
    )
    assert len(report["runtime_sha256"]) == 64
    assert runtime.validate_joint_velocity_runtime_report(report) == report


def test_runtime_rejects_position_stiffness_and_wrong_action_type(monkeypatch):
    _stub_isaaclab(monkeypatch)
    env = _Env()
    env.scene["robot"].data.joint_stiffness[0, 0] = 400.0
    with pytest.raises(ValueError, match="buffered joint stiffness"):
        runtime.validate_joint_velocity_runtime(env)

    env = _Env()
    env.action_manager._terms["arm"] = SimpleNamespace()
    with pytest.raises(ValueError, match="must be Isaac Lab JointVelocityAction"):
        runtime.validate_joint_velocity_runtime(env)


def test_runtime_rejects_unpinned_isaaclab(monkeypatch):
    monkeypatch.setattr(runtime, "_installed_isaaclab_version", lambda: "2.3.1")
    with pytest.raises(ValueError, match="requires Isaac Lab 2.3.0"):
        runtime.validate_joint_velocity_runtime(_Env())


def test_runtime_report_recomputes_full_contract_and_rejects_minimal_report(
    monkeypatch,
):
    _stub_isaaclab(monkeypatch)
    report = runtime.validate_joint_velocity_runtime(_Env())
    report["velocity_drive"]["direct_physx"]["damping"]["values"][0][0] = 79.0
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        runtime.validate_joint_velocity_runtime_report(report)

    with pytest.raises(ValueError, match="schema mismatch"):
        runtime.validate_joint_velocity_runtime_report(
            {"status": "pass", "profile": PI05_DROID_JOINTVELOCITY_PROFILE}
        )
