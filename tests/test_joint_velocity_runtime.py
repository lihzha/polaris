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


class _DeviceTensor:
    """CPU-backed tensor double with an explicit source-device contract."""

    def __init__(self, value, *, device):
        self.tensor = torch.as_tensor(value, dtype=torch.float32).clone()
        self.device = device
        self.dtype = self.tensor.dtype

    def __getitem__(self, index):
        return _DeviceTensor(self.tensor[index], device=self.device)

    def __setitem__(self, index, value):
        self.tensor[index] = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.tensor.numpy()


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
            joint_stiffness=_DeviceTensor(stiffness, device="cuda:0"),
            joint_damping=_DeviceTensor(damping, device="cuda:0"),
            joint_effort_limits=_DeviceTensor(effort, device="cuda:0"),
            joint_vel_limits=_DeviceTensor(velocity, device="cuda:0"),
        )
        self.root_physx_view = _View(
            _DeviceTensor(stiffness, device="cpu"),
            _DeviceTensor(damping, device="cpu"),
            _DeviceTensor(effort, device="cpu"),
            _DeviceTensor(velocity, device="cpu"),
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
        arm._clip = _DeviceTensor(
            np.broadcast_to(
                np.asarray([-1.0, 1.0], dtype=np.float32), (1, 7, 2)
            ).copy(),
            device="cuda:0",
        )
        arm._raw_actions = _DeviceTensor(torch.zeros((1, 7)), device="cuda:0")
        arm._processed_actions = _DeviceTensor(torch.zeros((1, 7)), device="cuda:0")
        arm.raw_actions = arm._raw_actions
        arm.processed_actions = arm._processed_actions
        finger = BinaryGripperAction()
        finger._joint_names = ["finger_joint"]
        finger._open_command = _DeviceTensor(torch.zeros((1,)), device="cuda:0")
        finger._close_command = _DeviceTensor(
            torch.full((1,), np.pi / 4.0), device="cuda:0"
        )
        finger._raw_actions = _DeviceTensor(torch.zeros((1, 1)), device="cuda:0")
        finger._processed_actions = _DeviceTensor(torch.zeros((1, 1)), device="cuda:0")
        finger.raw_actions = finger._raw_actions
        finger.processed_actions = finger._processed_actions
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
    assert report["clip"]["device"] == "cuda:0"
    assert report["action_buffers"]["raw_action"]["device"] == "cuda:0"
    assert report["velocity_drive"]["buffered"]["stiffness"]["device"] == "cuda:0"
    assert report["velocity_drive"]["direct_physx"]["stiffness"]["device"] == "cpu"
    assert report["gripper"]["open_command"]["shape"] == [1]
    assert report["gripper"]["raw_action"]["shape"] == [1, 1]
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


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda env: setattr(
                env.action_manager._terms["arm"]._clip, "device", "cpu"
            ),
            "action term clip must be on cuda:0",
        ),
        (
            lambda env: setattr(
                env.action_manager._terms["finger_joint"]._open_command,
                "device",
                "cpu",
            ),
            "gripper open command must be on cuda:0",
        ),
        (
            lambda env: setattr(env.scene["robot"].data.joint_damping, "device", "cpu"),
            "buffered joint damping must be on cuda:0",
        ),
        (
            lambda env: setattr(
                env.scene["robot"].root_physx_view.velocity,
                "device",
                "cuda:0",
            ),
            "direct PhysX joint velocity limits must be on cpu",
        ),
    ],
)
def test_live_runtime_rejects_field_specific_device_mismatch(
    monkeypatch, mutate, message
):
    _stub_isaaclab(monkeypatch)
    env = _Env()
    mutate(env)
    with pytest.raises(ValueError, match=message):
        runtime.validate_joint_velocity_runtime(env)


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


@pytest.mark.parametrize(
    ("surface", "field", "wrong_device", "message"),
    [
        ("buffered", "stiffness", "cpu", "device cuda:0"),
        ("direct_physx", "damping", "cuda:0", "device cpu"),
    ],
)
def test_runtime_report_rejects_field_specific_drive_devices(
    monkeypatch, surface, field, wrong_device, message
):
    _stub_isaaclab(monkeypatch)
    report = runtime.validate_joint_velocity_runtime(_Env())
    report["velocity_drive"][surface][field]["device"] = wrong_device
    report["runtime_sha256"] = runtime._canonical_sha256(
        {key: value for key, value in report.items() if key != "runtime_sha256"}
    )
    with pytest.raises(ValueError, match=message):
        runtime.validate_joint_velocity_runtime_report(report)


@pytest.mark.parametrize(
    ("path", "message"),
    [
        (("clip",), "action clip report must attest device cuda:0"),
        (
            ("action_buffers", "raw_action"),
            "arm raw_action report must attest device cuda:0",
        ),
        (
            ("gripper", "open_command"),
            "gripper open command report must attest device cuda:0",
        ),
        (
            ("gripper", "processed_action"),
            "gripper processed_action report must attest device cuda:0",
        ),
    ],
)
def test_runtime_report_rejects_cpu_action_and_gripper_tensors(
    monkeypatch, path, message
):
    _stub_isaaclab(monkeypatch)
    report = runtime.validate_joint_velocity_runtime(_Env())
    target = report
    for key in path:
        target = target[key]
    target["device"] = "cpu"
    report["runtime_sha256"] = runtime._canonical_sha256(
        {key: value for key, value in report.items() if key != "runtime_sha256"}
    )
    with pytest.raises(ValueError, match=message):
        runtime.validate_joint_velocity_runtime_report(report)
