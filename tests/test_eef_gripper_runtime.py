from __future__ import annotations

import copy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from polaris import eef_gripper_runtime as runtime


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
        "velocity_limit_source": (
            "live_implicit_actuator_velocity_limit_sim_float32_v1"
        ),
        "velocity_limit_rad_s": runtime.GRIPPER_DRIVER_VELOCITY_LIMIT_FLOAT32,
        "physics_hz": runtime.GRIPPER_TARGET_SLEW_PHYSICS_HZ,
        "physics_dt": runtime.GRIPPER_TARGET_SLEW_PHYSICS_DT,
        "max_target_step_rad": runtime.GRIPPER_MAX_TARGET_STEP_FLOAT32,
        "float32_tolerance_rad": (runtime.GRIPPER_TARGET_SLEW_FLOAT32_TOLERANCE_RAD),
        "reset_profile": runtime.EEF_GRIPPER_TARGET_SLEW_RESET_PROFILE,
        "tensor_dtype": runtime.PINNED_TENSOR_DTYPE,
        "tensor_device": runtime.PINNED_ACTUATOR_DEVICE,
    }


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
