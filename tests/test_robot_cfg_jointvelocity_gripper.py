import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

from polaris.pi05_droid_jointvelocity_contract import (
    NATIVE_GRIPPER_EFFORT_LIMIT,
    NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S,
)
from polaris.pi05_droid_position_contract import (
    PI05_DROID_POSITION_DRIVE_DAMPING,
    PI05_DROID_POSITION_DRIVE_STIFFNESS,
)


ROOT = Path(__file__).parents[1]


class _Cfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _ArticulationCfg(_Cfg):
    copy_calls = 0

    class InitialStateCfg(_Cfg):
        pass

    def copy(self):
        type(self).copy_calls += 1
        return copy.deepcopy(self)


def _load_robot_cfg_with_isaac_stubs(monkeypatch):
    _ArticulationCfg.copy_calls = 0
    isaaclab = ModuleType("isaaclab")
    sim = ModuleType("isaaclab.sim")
    sim.UsdFileCfg = _Cfg
    sim.RigidBodyPropertiesCfg = _Cfg
    sim.ArticulationRootPropertiesCfg = _Cfg
    actuators = ModuleType("isaaclab.actuators")
    actuators.ImplicitActuatorCfg = _Cfg
    assets = ModuleType("isaaclab.assets")
    assets.ArticulationCfg = _ArticulationCfg
    isaaclab.sim = sim
    isaaclab.actuators = actuators
    isaaclab.assets = assets
    utils = ModuleType("polaris.utils")
    utils.DATA_PATH = Path("/pinned/polaris-hub")
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim)
    monkeypatch.setitem(sys.modules, "isaaclab.actuators", actuators)
    monkeypatch.setitem(sys.modules, "isaaclab.assets", assets)
    monkeypatch.setitem(sys.modules, "polaris.utils", utils)

    path = ROOT / "src/polaris/environments/robot_cfg.py"
    spec = importlib.util.spec_from_file_location("_native_robot_cfg_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_position_robot_cfg_with_isaac_stubs(monkeypatch, robot_cfg):
    monkeypatch.setitem(sys.modules, "polaris.environments.robot_cfg", robot_cfg)
    path = ROOT / "src/polaris/environments/pi05_droid_position_robot_cfg.py"
    spec = importlib.util.spec_from_file_location("_position_robot_cfg_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_gripper_sim_limits_do_not_mutate_joint_position_or_eef_base(
    monkeypatch,
):
    robot_cfg = _load_robot_cfg_with_isaac_stubs(monkeypatch)
    assert _ArticulationCfg.copy_calls == 0
    assert not hasattr(robot_cfg, "NVIDIA_DROID_JOINT_VELOCITY")
    shoulder = robot_cfg.NVIDIA_DROID.actuators["panda_shoulder"]
    forearm = robot_cfg.NVIDIA_DROID.actuators["panda_forearm"]
    shared = robot_cfg.NVIDIA_DROID.actuators["gripper"]
    native_cfg = robot_cfg.make_nvidia_droid_joint_velocity_cfg()
    assert _ArticulationCfg.copy_calls == 1
    native = native_cfg.actuators["gripper"]

    assert shared is not native
    assert shared.joint_names_expr == ["finger_joint"]
    assert shared.velocity_limit == NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S
    assert shared.effort_limit == NATIVE_GRIPPER_EFFORT_LIMIT
    assert not hasattr(shared, "velocity_limit_sim")
    assert not hasattr(shared, "effort_limit_sim")
    assert shoulder.velocity_limit == 2.175
    assert shoulder.effort_limit == 87.0
    assert forearm.velocity_limit == 2.61
    assert forearm.effort_limit == 12.0
    for actuator in (shoulder, forearm):
        assert not hasattr(actuator, "velocity_limit_sim")
        assert not hasattr(actuator, "effort_limit_sim")
    assert native.joint_names_expr == ["finger_joint"]
    assert native.stiffness is None
    assert native.damping is None
    assert native.velocity_limit == NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S
    assert native.velocity_limit_sim == NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S
    assert native.effort_limit == NATIVE_GRIPPER_EFFORT_LIMIT
    assert native.effort_limit_sim == NATIVE_GRIPPER_EFFORT_LIMIT

    shared_before = copy.deepcopy(shared.__dict__)
    rebuilt = robot_cfg.make_nvidia_droid_joint_velocity_cfg()
    assert _ArticulationCfg.copy_calls == 2
    assert shared.__dict__ == shared_before
    assert rebuilt.actuators["gripper"] is not shared

    droid_source = (ROOT / "src/polaris/environments/droid_cfg.py").read_text(
        encoding="utf-8"
    )
    assert "from polaris.environments.robot_cfg import NVIDIA_DROID" in droid_source
    assert "NVIDIA_DROID_JOINT_VELOCITY" not in droid_source


def test_joint_position_runtime_audit_is_read_only_and_spans_gym_construction():
    evaluator = (ROOT / "scripts/eval.py").read_text(encoding="utf-8")
    intent_capture = evaluator.index(
        "jointpos_actuator_intent = capture_jointpos_actuator_intent("
    )
    gym_make = evaluator.index("env: ManagerBasedRLSplatEnv = gym.make(")
    live_capture = evaluator.index("live_jointpos_runtime = capture_jointpos_runtime(")
    assert intent_capture < gym_make < live_capture
    assert "actuator_intent=jointpos_actuator_intent" in evaluator

    jointpos_branch = evaluator[evaluator.index("elif audited_jointpos:") : gym_make]
    assert "velocity_limit" not in jointpos_branch
    assert "effort_limit" not in jointpos_branch

    robot_source = (ROOT / "src/polaris/environments/robot_cfg.py").read_text(
        encoding="utf-8"
    )
    shared_definition = robot_source[
        : robot_source.index("def make_nvidia_droid_joint_velocity_cfg")
    ]
    assert "velocity_limit_sim" not in shared_definition
    assert "effort_limit_sim" not in shared_definition


def test_position_robot_cfg_is_constructed_only_on_factory_call(monkeypatch):
    robot_cfg = _load_robot_cfg_with_isaac_stubs(monkeypatch)
    position_cfg = _load_position_robot_cfg_with_isaac_stubs(monkeypatch, robot_cfg)

    assert _ArticulationCfg.copy_calls == 0
    assert not hasattr(position_cfg, "NVIDIA_DROID_POSITION_ADAPTER")
    configured = position_cfg.make_nvidia_droid_position_adapter_cfg()
    assert _ArticulationCfg.copy_calls == 1
    assert configured is not robot_cfg.NVIDIA_DROID
    assert configured.actuators["panda_shoulder"].stiffness == (
        PI05_DROID_POSITION_DRIVE_STIFFNESS
    )
    assert configured.actuators["panda_shoulder"].damping == (
        PI05_DROID_POSITION_DRIVE_DAMPING
    )
