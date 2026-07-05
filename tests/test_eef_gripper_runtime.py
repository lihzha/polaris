from __future__ import annotations

import copy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from polaris import eef_gripper_runtime as runtime
from polaris.eef_ik_safety import current_joint_velocity_recovery_envelope
from polaris.eef_ik_safety import PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S


def _tensor(values, *, shape, device):
    return {
        "dtype": runtime.PINNED_TENSOR_DTYPE,
        "device": device,
        "shape": list(shape),
        "values": list(values),
        "finite_mask": [True] * len(values),
        "finite_count": len(values),
    }


def _target_slew_static():
    return {
        "profile": runtime.EEF_GRIPPER_TARGET_SLEW_PROFILE,
        "scope": "eef_pose_only_native_joint_position_unchanged_v1",
        "action_class": runtime.EEF_GRIPPER_TARGET_SLEW_ACTION_CLASS,
        "driver_joint_name": runtime.DRIVEN_GRIPPER_JOINT_NAME,
        "driver_joint_index": runtime.DRIVEN_GRIPPER_JOINT_INDEX,
        "endpoint_semantics_profile": runtime.GRIPPER_THRESHOLD_PROFILE,
        "open_target_rad": runtime.GRIPPER_OPEN_TARGET_FLOAT32,
        "closed_target_rad": runtime.GRIPPER_CLOSED_TARGET_FLOAT32,
        "physical_velocity_limit_source": (
            "live_implicit_actuator_velocity_limit_sim_float32_v1"
        ),
        "physical_velocity_limit_rad_s": (
            runtime.GRIPPER_DRIVER_VELOCITY_LIMIT_FLOAT32
        ),
        "target_slew_rate_source": runtime.GRIPPER_TARGET_SLEW_RATE_SOURCE,
        "target_slew_rate_factor": (runtime.GRIPPER_TARGET_SLEW_RATE_FACTOR_FLOAT32),
        "target_slew_rate_rad_s": runtime.GRIPPER_TARGET_SLEW_RATE_RAD_S_FLOAT32,
        "physics_hz": runtime.GRIPPER_TARGET_SLEW_PHYSICS_HZ,
        "physics_dt": runtime.GRIPPER_TARGET_SLEW_PHYSICS_DT,
        "max_target_step_rad": runtime.GRIPPER_MAX_TARGET_STEP_FLOAT32,
        "float32_tolerance_rad": (runtime.GRIPPER_TARGET_SLEW_FLOAT32_TOLERANCE_RAD),
        "reset_profile": runtime.EEF_GRIPPER_TARGET_SLEW_RESET_PROFILE,
        "tensor_dtype": runtime.PINNED_TENSOR_DTYPE,
        "tensor_device": runtime.PINNED_ACTUATOR_DEVICE,
    }


def _candidate_target_slew_static():
    value = _target_slew_static()
    specification = runtime.eef_gripper_target_slew_profile(
        runtime.EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
    )
    value.update(
        {
            "profile": specification.profile,
            "target_slew_rate_factor": specification.rate_factor_float32,
            "target_slew_rate_rad_s": specification.rate_rad_s_float32,
            "max_target_step_rad": specification.max_target_step_rad_float32,
        }
    )
    return value


def _target_slew_dynamic(*, process_calls=1, apply_calls=1):
    return {
        "profile": runtime.EEF_GRIPPER_TARGET_SLEW_PROFILE,
        "process_action_calls": process_calls,
        "apply_calls": apply_calls,
        "initialization_count": int(apply_calls > 0),
        "endpoint_change_count": 0,
        "repeated_endpoint_process_count": max(process_calls - 1, 0),
        "slew_limited_apply_count": apply_calls,
        "endpoint_reached_apply_count": 0,
        "live_limit_validation_count": apply_calls,
        "max_abs_target_step_rad": (
            runtime.GRIPPER_MAX_TARGET_STEP_FLOAT32 if apply_calls else 0.0
        ),
        "max_abs_endpoint_error_before_step_rad": (
            runtime.GRIPPER_CLOSED_TARGET_FLOAT32 if apply_calls else 0.0
        ),
        "max_abs_endpoint_error_after_step_rad": (
            runtime.GRIPPER_CLOSED_TARGET_FLOAT32
            - runtime.GRIPPER_MAX_TARGET_STEP_FLOAT32
            if apply_calls
            else 0.0
        ),
        "initial_anchor_rad": 0.0 if apply_calls else None,
        "last_requested_endpoint_rad": (
            runtime.GRIPPER_CLOSED_TARGET_FLOAT32 if process_calls else None
        ),
        "last_applied_target_rad": (
            runtime.GRIPPER_MAX_TARGET_STEP_FLOAT32 if apply_calls else None
        ),
    }


def _static_contract():
    before = _tensor(
        runtime.EXPECTED_FULL_VELOCITY_LIMITS_BEFORE_WRITE,
        shape=(1, 13),
        device="cpu",
    )
    after = _tensor(
        runtime.EXPECTED_FULL_VELOCITY_LIMITS_AFTER_WRITE,
        shape=(1, 13),
        device="cpu",
    )
    actuator = {
        "cfg_velocity_limit": 5.0,
        "cfg_velocity_limit_sim": 5.0,
        "cfg_effort_limit": 200.0,
        "cfg_effort_limit_sim": 200.0,
        "resolved_velocity_limit": _tensor([5.0], shape=(1, 1), device="cuda:0"),
        "resolved_velocity_limit_sim": _tensor([5.0], shape=(1, 1), device="cuda:0"),
        "resolved_effort_limit": _tensor([200.0], shape=(1, 1), device="cuda:0"),
        "resolved_effort_limit_sim": _tensor([200.0], shape=(1, 1), device="cuda:0"),
    }
    ownership = {
        name: {"joint_names": list(names), "joint_indices": list(indices)}
        for name, (names, indices) in runtime.EXPECTED_ACTUATOR_JOINT_OWNERSHIP.items()
    }
    return {
        "profile": runtime.EEF_GRIPPER_RUNTIME_PROFILE,
        "joint_names": list(runtime.EXPECTED_DROID_JOINT_NAMES),
        "gripper_joint_names": list(runtime.GRIPPER_JOINT_NAMES),
        "gripper_joint_indices": list(runtime.GRIPPER_JOINT_INDICES),
        "driver_joint_name": runtime.DRIVEN_GRIPPER_JOINT_NAME,
        "driver_joint_index": runtime.DRIVEN_GRIPPER_JOINT_INDEX,
        "follower_joint_names": list(runtime.GRIPPER_FOLLOWER_JOINT_NAMES),
        "follower_joint_indices": list(runtime.GRIPPER_FOLLOWER_JOINT_INDICES),
        "actuator_joint_ownership": ownership,
        "device_partition": {
            "profile": runtime.EEF_GRIPPER_DEVICE_PARTITION_PROFILE,
            "dynamic_articulation": "cuda:0",
            "implicit_actuator": "cuda:0",
            "static_physx": "cpu",
            "dtype": "torch.float32",
        },
        "driver_actuator": actuator,
        "mimic_joint_contract": runtime._expected_mimic_joint_contract(),
        "velocity_limits_before_write": before,
        "velocity_limits_after_write": after,
        "velocity_limit_write_contract": {
            "profile": runtime.EEF_GRIPPER_VELOCITY_WRITE_PROFILE,
            "setter": runtime.EEF_GRIPPER_VELOCITY_WRITE_SETTER,
            "timing": runtime.EEF_GRIPPER_VELOCITY_WRITE_TIMING,
            "call_count": 1,
            "articulation_indices": [0],
            "full_input": copy.deepcopy(after),
        },
        "driver_target_slew": _target_slew_static(),
        "measured_velocity_is_hard_bounded_by_limit": False,
    }


def _dynamic_evidence():
    vector = [0.0] * 6
    diagnostic = {
        "sample_phase": "post_policy_step",
        "sample_index": 8,
        "joint_position_rad": vector,
        "joint_velocity_rad_s": [0.0, 7.174964, 0.0, 0.0, 5.00002, 0.0],
        "joint_acceleration_rad_s2": [0.0] * 6,
        "joint_position_target_rad": vector,
        "joint_velocity_target_rad_s": vector,
    }
    terminal = {
        key: copy.deepcopy(value)
        for key, value in diagnostic.items()
        if key != "sample_phase"
    }
    return {
        "profile": runtime.EEF_GRIPPER_RUNTIME_PROFILE,
        "joint_names": list(runtime.GRIPPER_JOINT_NAMES),
        "joint_indices": list(runtime.GRIPPER_JOINT_INDICES),
        "apply_entry_samples": 8,
        "post_policy_step_samples": 1,
        "max_abs_joint_velocity_rad_s": [
            0.0,
            7.174964,
            0.0,
            0.0,
            5.00002,
            0.0,
        ],
        "max_abs_joint_acceleration_rad_s2": [0.0] * 6,
        "max_velocity_diagnostic": diagnostic,
        "terminal_state": terminal,
        "driver_target_slew": _target_slew_dynamic(),
        "nonfinite_samples": 0,
        "dropped_diagnostics": 0,
    }


def _concurrent_open_endpoint_telemetry(*, coupled_failure: bool = False):
    envelopes = [
        current_joint_velocity_recovery_envelope(limit)
        for limit in PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S
    ]
    arm_velocity = [0.0] * 7
    if coupled_failure:
        arm_velocity[0] = envelopes[0] + 0.001
    follower_velocity = [7.174964, 0.0, 0.0, 5.00002, 0.0]
    diagnostic = {
        "sample_phase": "post_policy_step",
        "sample_index": 8,
        "arm_joint_velocity_rad_s": arm_velocity,
        "follower_joint_velocity_rad_s": follower_velocity,
        "follower_joint_acceleration_rad_s2": [0.0] * 5,
        "follower_threshold_crossed": True,
        "arm_recovery_envelope_crossed": coupled_failure,
        "coupled_impulse_failure": coupled_failure,
    }
    return {
        "enabled": True,
        "profile": runtime.OPEN_ENDPOINT_COUPLED_IMPULSE_PROFILE,
        "endpoint": "open",
        "follower_threshold_rad_s_float32": (
            runtime.OPEN_ENDPOINT_FOLLOWER_TELEMETRY_THRESHOLD_RAD_S_FLOAT32
        ),
        "follower_threshold_semantics": (
            "passive_follower_crossing_is_telemetry_only_v1"
        ),
        "arm_threshold_profile": (
            "per_joint_float32_physical_limit_plus_limit_times_float32_1e_4_v1"
        ),
        "arm_velocity_envelopes_rad_s": envelopes,
        "failure_predicate": (
            "open_and_follower_gt_5p001_and_any_arm_gt_its_recovery_envelope_v1"
        ),
        "open_endpoint_samples": 9,
        "nonfinite_open_endpoint_samples": 0,
        "follower_threshold_crossing_samples": 1,
        "coupled_impulse_failure_samples": int(coupled_failure),
        "max_abs_arm_joint_velocity_rad_s": [abs(item) for item in arm_velocity],
        "max_abs_follower_joint_velocity_rad_s": [
            abs(item) for item in follower_velocity
        ],
        "max_abs_follower_joint_acceleration_rad_s2": [0.0] * 5,
        "maximum_follower_diagnostic": diagnostic,
        "first_coupled_impulse_failure_diagnostic": (
            copy.deepcopy(diagnostic) if coupled_failure else None
        ),
        "passed": not coupled_failure,
    }


class _FakeTensor:
    def __init__(self, values, *, dtype="torch.float32", device="cpu"):
        self._values = np.asarray(values, dtype=np.float32)
        self.dtype = dtype
        self.device = device
        self.shape = self._values.shape

    def clone(self):
        return _FakeTensor(self._values.copy(), dtype=self.dtype, device=self.device)

    def tolist(self):
        return self._values.tolist()

    def __setitem__(self, key, value):
        self._values[key] = value


class _FakePhysxView:
    def __init__(self):
        self.velocity_limits = _FakeTensor(
            [runtime.EXPECTED_FULL_VELOCITY_LIMITS_BEFORE_WRITE]
        )
        self.setter_calls = []

    def get_dof_max_velocities(self):
        return self.velocity_limits.clone()

    def set_dof_max_velocities(self, values, indices):
        self.setter_calls.append((values.clone(), indices.tolist()))
        self.velocity_limits = values.clone()


class _FakePath:
    def __init__(self, path):
        self.pathString = path

    def __str__(self):
        return self.pathString


class _FakeAttribute:
    def __init__(self, value, *, type_name="float", set_result=True):
        self.value = value
        self.type_name = type_name
        self.set_result = set_result
        self.set_calls = []

    def __bool__(self):
        return True

    def GetTypeName(self):
        return self.type_name

    def Get(self):
        return self.value

    def Set(self, value):
        self.set_calls.append(value)
        if self.set_result:
            self.value = np.float32(value).item()
        return self.set_result


class _FakeRelationship:
    def __init__(self, targets):
        self.targets = [_FakePath(path) for path in targets]

    def __bool__(self):
        return True

    def GetTargets(self):
        return list(self.targets)


class _FakePrim:
    def __init__(
        self,
        path,
        *,
        type_name,
        attributes=None,
        applied_schemas=None,
        relationships=None,
    ):
        self.path = path
        self.type_name = type_name
        self.attributes = attributes or {}
        self.applied_schemas = list(applied_schemas or [])
        self.relationships = relationships or {}

    def __bool__(self):
        return True

    def IsValid(self):
        return True

    def GetPath(self):
        return _FakePath(self.path)

    def GetTypeName(self):
        return self.type_name

    def GetAttribute(self, name):
        return self.attributes.get(name)

    def GetAppliedSchemas(self):
        return list(self.applied_schemas)

    def GetRelationship(self, name):
        return self.relationships.get(name)


class _FakeStage:
    def __init__(self, prims):
        self.prims = {prim.path: prim for prim in prims}

    def GetPrimAtPath(self, path):
        return self.prims.get(path)


def _fake_live_mimic_stage():
    source = runtime._expected_mimic_joint_contract()
    root_path = runtime.EEF_GRIPPER_MIMIC_COMPLIANCE_EXPECTED_LIVE_ROOT
    root = _FakePrim(root_path, type_name="Xform")
    followers = []
    driver_path = root_path + source["driver_joint_prim_path"].removeprefix("/panda")
    for specification in source["followers"]:
        live_path = root_path + specification["prim_path"].removeprefix("/panda")
        frequency_name, damping_name = runtime._mimic_attribute_names(
            specification["mimic_axis"]
        )
        followers.append(
            _FakePrim(
                live_path,
                type_name="PhysicsRevoluteJoint",
                attributes={
                    frequency_name: _FakeAttribute(
                        specification["natural_frequency_hz"]
                    ),
                    damping_name: _FakeAttribute(specification["damping_ratio"]),
                    f"physxMimicJoint:{specification['mimic_axis']}:gearing": (
                        _FakeAttribute(specification["gearing"])
                    ),
                    f"physxMimicJoint:{specification['mimic_axis']}:offset": (
                        _FakeAttribute(0.0)
                    ),
                    "physics:excludeFromArticulation": _FakeAttribute(
                        specification["exclude_from_articulation"],
                        type_name="bool",
                    ),
                },
                applied_schemas=[f"PhysxMimicJointAPI:{specification['mimic_axis']}"],
                relationships={
                    f"physxMimicJoint:{specification['mimic_axis']}:referenceJoint": (
                        _FakeRelationship([driver_path])
                    )
                },
            )
        )
    stage = _FakeStage([root, *followers])
    return source, stage, root, followers


def _candidate_compliance_contract(followers, *, post_reset=True):
    serialized = copy.deepcopy(followers)
    if post_reset:
        for follower in serialized:
            follower["post_reset_composed_usd_readback"] = copy.deepcopy(
                follower["after_spawn_write"]
            )
            follower["post_reset_composed_usd_structure"] = copy.deepcopy(
                follower["after_spawn_structure"]
            )
    return {
        "profile": runtime.EEF_GRIPPER_MIMIC_COMPLIANCE_PROFILE,
        "enabled": True,
        "scope": runtime.EEF_GRIPPER_MIMIC_COMPLIANCE_SCOPE,
        "timing": runtime.EEF_GRIPPER_MIMIC_COMPLIANCE_TIMING,
        "setter": runtime.EEF_GRIPPER_MIMIC_COMPLIANCE_SETTER,
        "live_root_profile": runtime.EEF_GRIPPER_MIMIC_COMPLIANCE_LIVE_ROOT_PROFILE,
        "live_root_path": runtime.EEF_GRIPPER_MIMIC_COMPLIANCE_EXPECTED_LIVE_ROOT,
        "original_spawn_func": copy.deepcopy(
            runtime.EEF_GRIPPER_MIMIC_COMPLIANCE_ORIGINAL_SPAWN_IDENTITY
        ),
        "overlay_func": copy.deepcopy(
            runtime.EEF_GRIPPER_MIMIC_COMPLIANCE_OVERLAY_IDENTITY
        ),
        "original_spawn_call_count": 1,
        "overlay_call_count": 1,
        "physics_hz": runtime.EEF_GRIPPER_MIMIC_COMPLIANCE_PHYSICS_HZ,
        "physics_dt": runtime.EEF_GRIPPER_MIMIC_COMPLIANCE_PHYSICS_DT,
        "target_natural_frequency_rad_s": (
            runtime.EEF_GRIPPER_MIMIC_COMPLIANCE_NATURAL_FREQUENCY_RAD_S_FLOAT32
        ),
        "target_damping_ratio": (
            runtime.EEF_GRIPPER_MIMIC_COMPLIANCE_DAMPING_RATIO_FLOAT32
        ),
        "frequency_timestep_product": (
            runtime.EEF_GRIPPER_MIMIC_COMPLIANCE_FREQUENCY_TIMESTEP_PRODUCT
        ),
        "follower_count": runtime.EEF_GRIPPER_MIMIC_COMPLIANCE_FOLLOWER_COUNT,
        "natural_frequency_write_count": (
            runtime.EEF_GRIPPER_MIMIC_COMPLIANCE_FOLLOWER_COUNT
        ),
        "damping_ratio_write_count": (
            runtime.EEF_GRIPPER_MIMIC_COMPLIANCE_FOLLOWER_COUNT
        ),
        "total_write_count": (runtime.EEF_GRIPPER_MIMIC_COMPLIANCE_TOTAL_WRITE_COUNT),
        "source_usd_sha256": runtime.EXPECTED_ROBOT_USD_SHA256,
        "source_usd_unchanged_after_spawn_overlay": True,
        "followers": serialized,
    }


class _FakeArmTerm:
    def __init__(self):
        self.installed = None

    def install_gripper_runtime_contract(self, contract, *, finger_term):
        self.installed = contract
        self.finger_term = finger_term


class EefBinaryJointPositionTargetSlewAction:
    def __init__(self, static):
        self.static = static
        self.installed = None

    def gripper_target_slew_static_contract(self):
        return copy.deepcopy(self.static)

    def install_gripper_target_slew_contract(self, contract):
        self.installed = copy.deepcopy(contract)

    def gripper_target_slew_dynamic_report(self):
        return _target_slew_dynamic(process_calls=0, apply_calls=0)


def test_static_contract_binds_one_follower_only_full_tensor_write():
    validated = runtime.validate_eef_gripper_static_contract(_static_contract())
    before = validated["velocity_limits_before_write"]["values"]
    after = validated["velocity_limits_after_write"]["values"]
    assert before[:8] == after[:8]
    assert before[8:] == [runtime.GRIPPER_FOLLOWER_DEFAULT_VELOCITY_LIMIT_FLOAT32] * 5
    assert after[8:] == [5.0] * 5


def test_installer_performs_one_full_cpu_write_and_later_reset_only_reads(
    monkeypatch, tmp_path
):
    view = _FakePhysxView()
    arm_term = _FakeArmTerm()
    finger_term = EefBinaryJointPositionTargetSlewAction(_target_slew_static())
    robot = SimpleNamespace(root_physx_view=view)
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            scene={"robot": robot},
            action_manager=SimpleNamespace(
                _terms={"arm": arm_term, "finger_joint": finger_term}
            ),
        )
    )
    static = _static_contract()
    monkeypatch.setattr(
        runtime,
        "_validate_live_ownership",
        lambda *_args, **_kwargs: static["actuator_joint_ownership"],
    )
    monkeypatch.setattr(
        runtime,
        "_capture_driver_actuator",
        lambda *_args, **_kwargs: static["driver_actuator"],
    )
    monkeypatch.setattr(
        runtime,
        "_capture_mimic_joint_contract",
        lambda *_args, **_kwargs: static["mimic_joint_contract"],
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            int32="torch.int32",
            arange=lambda count, **_kwargs: SimpleNamespace(
                tolist=lambda: list(range(count))
            ),
        ),
    )

    contract = runtime.install_eef_gripper_runtime(
        env, robot_usd_path=tmp_path / "unused.usd"
    )
    assert len(view.setter_calls) == 1
    written, articulation_indices = view.setter_calls[0]
    assert articulation_indices == [0]
    assert written.tolist()[0] == list(
        runtime.EXPECTED_FULL_VELOCITY_LIMITS_AFTER_WRITE
    )
    assert arm_term.installed == contract
    assert arm_term.finger_term is finger_term
    assert finger_term.installed == contract["driver_target_slew"]

    runtime.validate_eef_gripper_post_reset(env, contract)
    assert len(view.setter_calls) == 1


def test_spawn_overlay_writes_exact_ten_float_attributes_and_post_reset_reads_only(
    monkeypatch, tmp_path
):
    source, stage, root, follower_prims = _fake_live_mimic_stage()
    original_calls = []

    def original(prim_path, cfg, **kwargs):
        original_calls.append((prim_path, cfg, kwargs))
        return root

    original.__module__ = runtime.EEF_GRIPPER_MIMIC_COMPLIANCE_ORIGINAL_SPAWN_IDENTITY[
        "module"
    ]
    original.__qualname__ = (
        runtime.EEF_GRIPPER_MIMIC_COMPLIANCE_ORIGINAL_SPAWN_IDENTITY["qualname"]
    )
    original.__name__ = runtime.EEF_GRIPPER_MIMIC_COMPLIANCE_ORIGINAL_SPAWN_IDENTITY[
        "name"
    ]
    monkeypatch.setattr(runtime, "_expected_original_spawn_func", lambda: original)
    monkeypatch.setattr(
        runtime,
        "_capture_mimic_joint_contract",
        lambda _path: copy.deepcopy(source),
    )
    monkeypatch.setattr(
        runtime,
        "_current_live_stage_and_robot_roots",
        lambda _path: (stage, [root]),
    )
    spawn_cfg = SimpleNamespace(
        func=original,
        usd_path=tmp_path / "immutable-source.usd",
    )
    overlay = runtime.configure_eef_gripper_mimic_compliance_spawn_overlay(
        spawn_cfg,
        target_slew_profile=(
            runtime.EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
        ),
    )
    result = overlay(
        runtime.EEF_GRIPPER_MIMIC_COMPLIANCE_EXPECTED_LIVE_ROOT,
        spawn_cfg,
        translation=(0.0, 0.0, 0.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
    )
    assert result is root
    assert len(original_calls) == 1
    attributes = [
        attribute
        for prim in follower_prims
        for name, attribute in prim.attributes.items()
        if name.endswith((":naturalFrequency", ":dampingRatio"))
    ]
    assert len(attributes) == 10
    assert all(len(attribute.set_calls) == 1 for attribute in attributes)

    robot = SimpleNamespace(
        cfg=SimpleNamespace(
            prim_path=runtime.EEF_GRIPPER_MIMIC_COMPLIANCE_EXPECTED_LIVE_ROOT,
            spawn=spawn_cfg,
        )
    )
    contract = runtime._post_reset_mimic_compliance_contract(  # noqa: SLF001
        robot=robot,
        source_contract=source,
    )
    assert contract["total_write_count"] == 10
    assert all(
        follower["post_reset_composed_usd_readback"] == follower["after_spawn_write"]
        for follower in contract["followers"]
    )
    assert all(len(attribute.set_calls) == 1 for attribute in attributes)

    runtime._validate_post_reset_mimic_compliance_readback(  # noqa: SLF001
        robot=robot,
        source_contract=source,
        expected_contract=contract,
    )
    assert all(len(attribute.set_calls) == 1 for attribute in attributes)


def test_baseline_spawn_configuration_is_bitwise_untouched_and_writes_nothing(
    monkeypatch,
):
    source, _, _, follower_prims = _fake_live_mimic_stage()

    def original(*_args, **_kwargs):
        return None

    original.__module__ = runtime.EEF_GRIPPER_MIMIC_COMPLIANCE_ORIGINAL_SPAWN_IDENTITY[
        "module"
    ]
    original.__qualname__ = (
        runtime.EEF_GRIPPER_MIMIC_COMPLIANCE_ORIGINAL_SPAWN_IDENTITY["qualname"]
    )
    original.__name__ = runtime.EEF_GRIPPER_MIMIC_COMPLIANCE_ORIGINAL_SPAWN_IDENTITY[
        "name"
    ]
    monkeypatch.setattr(runtime, "_expected_original_spawn_func", lambda: original)
    spawn_cfg = SimpleNamespace(func=original, usd_path="/unused.usd")
    returned = runtime.configure_eef_gripper_mimic_compliance_spawn_overlay(
        spawn_cfg,
        target_slew_profile=runtime.EEF_GRIPPER_TARGET_SLEW_PROFILE,
    )
    assert returned is original
    assert spawn_cfg.func is original
    assert source == runtime._expected_mimic_joint_contract()  # noqa: SLF001
    assert all(
        attribute.set_calls == []
        for prim in follower_prims
        for attribute in prim.attributes.values()
    )


def test_mimic_compliance_prevalidates_all_attribute_types_before_any_write():
    source, stage, root, follower_prims = _fake_live_mimic_stage()
    bad_attribute = next(iter(follower_prims[-1].attributes.values()))
    bad_attribute.type_name = "double"
    with pytest.raises(ValueError, match="attribute type drift"):
        runtime._write_spawned_mimic_compliance(  # noqa: SLF001
            stage=stage,
            roots=[root],
            source_contract=source,
        )
    assert all(
        attribute.set_calls == []
        for prim in follower_prims
        for attribute in prim.attributes.values()
    )


def test_mimic_compliance_late_source_value_drift_causes_zero_writes():
    source, stage, root, follower_prims = _fake_live_mimic_stage()
    last = follower_prims[-1]
    frequency_name, _ = runtime._mimic_attribute_names(  # noqa: SLF001
        source["followers"][-1]["mimic_axis"]
    )
    last.attributes[frequency_name].value = 999.0
    with pytest.raises(ValueError, match="pre-write/source drift"):
        runtime._write_spawned_mimic_compliance(  # noqa: SLF001
            stage=stage,
            roots=[root],
            source_contract=source,
        )
    assert all(
        attribute.set_calls == []
        for prim in follower_prims
        for attribute in prim.attributes.values()
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda _source, prim: setattr(
                prim, "applied_schemas", ["PhysxMimicJointAPI:rotY"]
            ),
            "applied mimic API",
        ),
        (
            lambda source, prim: prim.relationships.__setitem__(
                f"physxMimicJoint:{source['mimic_axis']}:referenceJoint",
                _FakeRelationship(["/World/wrong"]),
            ),
            "source structure drift",
        ),
        (
            lambda source, prim: setattr(
                prim.attributes[f"physxMimicJoint:{source['mimic_axis']}:gearing"],
                "value",
                2.0,
            ),
            "source structure drift",
        ),
        (
            lambda _source, prim: setattr(
                prim.attributes["physics:excludeFromArticulation"],
                "value",
                True,
            ),
            "source structure drift",
        ),
    ],
)
def test_mimic_compliance_rejects_live_api_reference_gearing_and_exclusion_drift(
    mutation, match
):
    source, stage, root, follower_prims = _fake_live_mimic_stage()
    mutation(source["followers"][-1], follower_prims[-1])
    with pytest.raises(ValueError, match=match):
        runtime._write_spawned_mimic_compliance(  # noqa: SLF001
            stage=stage,
            roots=[root],
            source_contract=source,
        )
    assert all(
        attribute.set_calls == []
        for prim in follower_prims
        for attribute in prim.attributes.values()
    )


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("total_write_count",), True, "total_write_count drift"),
        (
            ("target_damping_ratio",),
            1.0,
            "target float32 drift",
        ),
        (
            ("followers", 0, "live_prim_path"),
            "/World/wrong",
            "live_prim_path drift",
        ),
        (
            ("followers", 0, "post_reset_composed_usd_readback", "damping_ratio"),
            0.0,
            "post-reset values",
        ),
    ],
)
def test_mimic_compliance_contract_rejects_schema_type_and_value_tampering(
    path, value, match
):
    source, stage, root, _ = _fake_live_mimic_stage()
    _, followers = runtime._write_spawned_mimic_compliance(  # noqa: SLF001
        stage=stage,
        roots=[root],
        source_contract=source,
    )
    contract = _candidate_compliance_contract(followers)
    target = contract
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=match):
        runtime.validate_eef_gripper_mimic_compliance(
            contract,
            source_contract=source,
        )


def test_mimic_compliance_contract_schema_is_closed():
    source, stage, root, _ = _fake_live_mimic_stage()
    _, followers = runtime._write_spawned_mimic_compliance(  # noqa: SLF001
        stage=stage,
        roots=[root],
        source_contract=source,
    )
    contract = _candidate_compliance_contract(followers)
    contract["hidden"] = True
    with pytest.raises(ValueError, match="contract schema"):
        runtime.validate_eef_gripper_mimic_compliance(
            contract,
            source_contract=source,
        )


def test_candidate_static_contract_requires_compliance_and_baseline_forbids_it():
    source, stage, root, _ = _fake_live_mimic_stage()
    _, followers = runtime._write_spawned_mimic_compliance(  # noqa: SLF001
        stage=stage,
        roots=[root],
        source_contract=source,
    )
    candidate_contract = _static_contract()
    candidate_contract["driver_target_slew"] = _candidate_target_slew_static()
    candidate_contract["mimic_compliance"] = _candidate_compliance_contract(followers)
    assert runtime.validate_eef_gripper_static_contract(
        candidate_contract,
        expected_target_slew_profile=(
            runtime.EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
        ),
    )
    with pytest.raises(ValueError, match="gripper static schema"):
        runtime.validate_eef_gripper_static_contract(candidate_contract)

    candidate_contract.pop("mimic_compliance")
    with pytest.raises(ValueError, match="gripper static schema"):
        runtime.validate_eef_gripper_static_contract(
            candidate_contract,
            expected_target_slew_profile=(
                runtime.EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
            ),
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("velocity_limit_write_contract", "call_count"), 2),
        (("velocity_limit_write_contract", "timing"), "before_reset"),
        (("device_partition", "static_physx"), "cuda:0"),
        (("mimic_joint_contract", "followers", 0, "mimic_axis"), "rotX"),
        (("actuator_joint_ownership", "gripper", "joint_indices"), [8]),
    ],
)
def test_static_contract_rejects_identity_device_mimic_and_ownership_drift(path, value):
    contract = _static_contract()
    target = contract
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        runtime.validate_eef_gripper_static_contract(contract)


def test_dynamic_contract_records_but_does_not_claim_passive_velocity_is_bounded():
    static = runtime.validate_eef_gripper_static_contract(_static_contract())
    dynamic = runtime.validate_eef_gripper_dynamic_evidence(_dynamic_evidence())
    assert static["measured_velocity_is_hard_bounded_by_limit"] is False
    assert max(dynamic["max_abs_joint_velocity_rad_s"]) == pytest.approx(7.174964)


def test_open_endpoint_follower_crossing_is_telemetry_only_below_arm_envelope():
    evidence = _dynamic_evidence()
    evidence[runtime.OPEN_ENDPOINT_COUPLED_IMPULSE_FIELD] = (
        _concurrent_open_endpoint_telemetry()
    )
    validated = runtime.validate_eef_gripper_dynamic_evidence(
        evidence,
        expect_open_endpoint_coupled_impulse=True,
    )
    telemetry = validated[runtime.OPEN_ENDPOINT_COUPLED_IMPULSE_FIELD]
    assert telemetry["follower_threshold_crossing_samples"] == 1
    assert telemetry["coupled_impulse_failure_samples"] == 0
    assert telemetry["passed"] is True


def test_open_endpoint_coupled_failure_requires_independent_failure_diagnostic():
    evidence = _dynamic_evidence()
    evidence[runtime.OPEN_ENDPOINT_COUPLED_IMPULSE_FIELD] = (
        _concurrent_open_endpoint_telemetry(coupled_failure=True)
    )
    validated = runtime.validate_eef_gripper_dynamic_evidence(
        evidence,
        expect_open_endpoint_coupled_impulse=True,
    )
    assert validated[runtime.OPEN_ENDPOINT_COUPLED_IMPULSE_FIELD]["passed"] is False
    drifted = copy.deepcopy(evidence)
    drifted[runtime.OPEN_ENDPOINT_COUPLED_IMPULSE_FIELD][
        "first_coupled_impulse_failure_diagnostic"
    ] = None
    with pytest.raises(ValueError, match="failure diagnostic"):
        runtime.validate_eef_gripper_dynamic_evidence(
            drifted,
            expect_open_endpoint_coupled_impulse=True,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dropped_diagnostics", 1),
        ("joint_indices", [7, 8, 9, 10, 11, 11]),
        ("max_abs_joint_velocity_rad_s", [0.0] * 5),
    ],
)
def test_dynamic_contract_rejects_incomplete_or_drifted_evidence(field, value):
    evidence = _dynamic_evidence()
    evidence[field] = value
    with pytest.raises(ValueError):
        runtime.validate_eef_gripper_dynamic_evidence(evidence)


def test_dynamic_contract_preserves_one_all_nonfinite_apply_entry_sample():
    evidence = _dynamic_evidence()
    evidence.update(
        {
            "apply_entry_samples": 1,
            "post_policy_step_samples": 0,
            "max_abs_joint_velocity_rad_s": [0.0] * 6,
            "max_abs_joint_acceleration_rad_s2": [0.0] * 6,
            "max_velocity_diagnostic": None,
            "terminal_state": None,
            "nonfinite_samples": 1,
        }
    )
    runtime.validate_eef_gripper_dynamic_evidence(evidence)
    evidence["nonfinite_samples"] = 2
    with pytest.raises(ValueError, match="nonfinite sample cadence"):
        runtime.validate_eef_gripper_dynamic_evidence(evidence)


@pytest.mark.parametrize(
    ("section", "field", "value", "match"),
    [
        ("max_velocity_diagnostic", "sample_index", 9, "diagnostic identity"),
        (
            "max_velocity_diagnostic",
            "joint_velocity_rad_s",
            [0.0] * 6,
            "diagnostic/aggregate drift",
        ),
        ("terminal_state", "sample_index", 7, "terminal sample index"),
    ],
)
def test_dynamic_contract_rejects_stale_diagnostic_and_terminal_cadence(
    section, field, value, match
):
    evidence = _dynamic_evidence()
    evidence[section][field] = value
    with pytest.raises(ValueError, match=match):
        runtime.validate_eef_gripper_dynamic_evidence(evidence)


def test_empty_dynamic_contract_requires_zero_maxima():
    evidence = _dynamic_evidence()
    evidence.update(
        {
            "apply_entry_samples": 0,
            "post_policy_step_samples": 0,
            "max_abs_joint_velocity_rad_s": [0.0] * 6,
            "max_abs_joint_acceleration_rad_s2": [0.0] * 6,
            "max_velocity_diagnostic": None,
            "terminal_state": None,
            "driver_target_slew": _target_slew_dynamic(process_calls=0, apply_calls=0),
        }
    )
    runtime.validate_eef_gripper_dynamic_evidence(evidence)
    evidence["max_abs_joint_velocity_rad_s"][2] = 1.0
    with pytest.raises(ValueError, match="empty gripper evidence"):
        runtime.validate_eef_gripper_dynamic_evidence(evidence)
