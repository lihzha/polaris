import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from polaris.native_gripper_runtime import (
    EXPECTED_DROID_JOINT_NAMES,
    EXPECTED_FULL_LIMITS_CAPPED,
    EXPECTED_FULL_LIMITS_UNCAPPED,
    NativeAllJointDynamicRecorder,
    NativeAllJointVelocityLimitError,
    apply_native_gripper_all_six_velocity_limits,
    native_gripper_mimic_reference_contract,
    native_gripper_reset_report,
    validate_native_all_joint_dynamic_report,
    validate_native_all_joint_velocity_failure,
    validate_native_gripper_mimic_contract,
    validate_native_gripper_reset_report,
)


class _DeviceTensor:
    def __init__(self, value, *, device):
        self.tensor = torch.as_tensor(value, dtype=torch.float32).clone()
        self.device = device
        self.dtype = self.tensor.dtype

    @property
    def shape(self):
        return self.tensor.shape

    def __getitem__(self, index):
        return _DeviceTensor(self.tensor[index], device=self.device)

    def __setitem__(self, index, value):
        if isinstance(value, _DeviceTensor):
            value = value.tensor
        self.tensor[index] = value

    def clone(self):
        return _DeviceTensor(self.tensor.clone(), device=self.device)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.tensor.numpy()


class _View:
    def __init__(self, limits):
        self.limits = _DeviceTensor(limits, device="cpu")

    def get_dof_max_velocities(self):
        return self.limits


class _Robot:
    def __init__(self):
        self.device = "cpu"
        self.joint_names = list(EXPECTED_DROID_JOINT_NAMES)
        self.data = SimpleNamespace(
            joint_vel_limits=_DeviceTensor(
                [EXPECTED_FULL_LIMITS_UNCAPPED], device="cuda:0"
            )
        )
        self.root_physx_view = _View([EXPECTED_FULL_LIMITS_UNCAPPED])
        self.writer_calls = []

    def write_joint_velocity_limit_to_sim(self, limits, *, joint_ids, env_ids):
        self.writer_calls.append(
            {
                "limits": limits.tensor.clone(),
                "joint_ids": joint_ids,
                "env_ids": env_ids.detach().cpu().tolist(),
            }
        )
        assert joint_ids is None
        self.data.joint_vel_limits = limits.clone()
        self.root_physx_view.limits.tensor[:] = limits.tensor


class _Env:
    def __init__(self):
        self.scene = {"robot": _Robot()}

    @property
    def unwrapped(self):
        return self


def test_every_reset_write_caps_buffer_and_physx_and_preserves_arm_driver():
    env = _Env()
    apply_native_gripper_all_six_velocity_limits(env, torch.tensor([0]))
    report = native_gripper_reset_report(env)
    assert report["reset_count"] == report["write_count"] == 1
    assert len(env.scene["robot"].writer_calls) == 1
    assert env.scene["robot"].writer_calls[0]["joint_ids"] is None
    assert env.scene["robot"].writer_calls[0]["env_ids"] == [0]
    expected = np.asarray([EXPECTED_FULL_LIMITS_CAPPED], dtype=np.float32)
    assert np.array_equal(env.scene["robot"].writer_calls[0]["limits"], expected)
    assert np.array_equal(env.scene["robot"].data.joint_vel_limits.tensor, expected)
    assert np.array_equal(env.scene["robot"].root_physx_view.limits.tensor, expected)

    apply_native_gripper_all_six_velocity_limits(env, torch.tensor([0]))
    report = native_gripper_reset_report(env)
    assert report["reset_count"] == report["write_count"] == 2
    assert len(env.scene["robot"].writer_calls) == 2
    validate_native_gripper_reset_report(report)


def test_reset_write_fails_closed_on_joint_order_and_prewrite_limit_drift():
    env = _Env()
    env.scene["robot"].joint_names[-1] = "wrong_joint"
    with pytest.raises(ValueError, match="joint order"):
        apply_native_gripper_all_six_velocity_limits(env, torch.tensor([0]))

    env = _Env()
    env.scene["robot"].data.joint_vel_limits.tensor[0, 8] = 7.0
    env.scene["robot"].root_physx_view.limits.tensor[0, 8] = 7.0
    with pytest.raises(ValueError, match="pre-write full"):
        apply_native_gripper_all_six_velocity_limits(env, torch.tensor([0]))


def test_reset_report_and_mimic_contract_mutations_fail_closed():
    env = _Env()
    apply_native_gripper_all_six_velocity_limits(env, torch.tensor([0]))
    report = native_gripper_reset_report(env)
    report["write_count"] = 0
    with pytest.raises(ValueError, match="count"):
        validate_native_gripper_reset_report(report)

    mimic = native_gripper_mimic_reference_contract()
    assert validate_native_gripper_mimic_contract(mimic) == mimic
    wrong = copy.deepcopy(mimic)
    wrong["followers"][0]["gearing"] = 1.0
    with pytest.raises(ValueError, match="mimic"):
        validate_native_gripper_mimic_contract(wrong)


class _DynamicAsset:
    def __init__(self):
        self.joint_names = list(EXPECTED_DROID_JOINT_NAMES)
        self.data = SimpleNamespace(
            joint_pos=torch.zeros((1, 13), dtype=torch.float32),
            joint_vel=torch.zeros((1, 13), dtype=torch.float32),
            joint_vel_target=torch.zeros((1, 13), dtype=torch.float32),
            joint_pos_target=torch.zeros((1, 13), dtype=torch.float32),
            joint_vel_limits=torch.tensor(
                [EXPECTED_FULL_LIMITS_CAPPED], dtype=torch.float32
            ),
        )


def test_dynamic_recorder_requires_exact_eight_apply_plus_post_cadence():
    asset = _DynamicAsset()
    recorder = NativeAllJointDynamicRecorder()
    for substep in range(8):
        asset.data.joint_vel[0, 0] = substep * 0.01
        recorder.record_apply_entry(asset)
    recorder.record_post_policy_step(asset)
    full = recorder.report(include_samples=True)
    assert full["apply_calls"] == 8
    assert full["post_policy_step_samples"] == 1
    assert full["sample_count"] == 9
    assert full["schema_version"] == 2
    assert full["terminal_velocity_failure"] is None
    assert validate_native_all_joint_dynamic_report(full, require_samples=True) == full
    aggregate = recorder.report(include_samples=False)
    assert aggregate["samples"] is None
    assert (
        validate_native_all_joint_dynamic_report(aggregate, require_samples=False)
        == aggregate
    )

    recorder = NativeAllJointDynamicRecorder()
    recorder.record_apply_entry(asset)
    with pytest.raises(ValueError, match="cadence"):
        recorder.record_post_policy_step(asset)


def test_dynamic_recorder_persists_typed_arm_or_gripper_velocity_limit_violation(
    tmp_path,
):
    asset = _DynamicAsset()
    recorder = NativeAllJointDynamicRecorder()
    asset.data.joint_vel[0, 7] = 5.00004
    recorder.record_apply_entry(asset)

    asset = _DynamicAsset()
    recorder = NativeAllJointDynamicRecorder()
    asset.data.joint_vel[0, 7] = 5.00006
    first_incident = tmp_path / "driver.json"
    recorder.bind_failure_path(first_incident)
    with pytest.raises(NativeAllJointVelocityLimitError) as captured:
        recorder.record_apply_entry(asset)
    evidence = captured.value.evidence
    assert evidence["physics_substep_index"] == 0
    assert evidence["policy_step_index"] == 0
    assert evidence["excess_mask"] == [False] * 7 + [True] + [False] * 5
    assert evidence["violating_joint_names"] == ["finger_joint"]
    assert captured.value.incident_artifact["path"] == str(first_incident.resolve())
    assert first_incident.stat().st_mode & 0o777 == 0o444
    assert validate_native_all_joint_velocity_failure(evidence) == evidence
    report = recorder.report(include_samples=True)
    assert report["apply_calls"] == 0
    assert report["post_policy_step_samples"] == 0
    assert report["samples"] == []
    assert report["terminal_velocity_failure"] == evidence
    assert (
        validate_native_all_joint_dynamic_report(report, require_samples=True) == report
    )

    asset = _DynamicAsset()
    recorder = NativeAllJointDynamicRecorder()
    asset.data.joint_vel[0, 8] = 5.001
    recorder.bind_failure_path(tmp_path / "follower.json")
    with pytest.raises(NativeAllJointVelocityLimitError):
        recorder.record_apply_entry(asset)

    asset = _DynamicAsset()
    recorder = NativeAllJointDynamicRecorder()
    asset.data.joint_vel[0, 4] = 2.611
    recorder.bind_failure_path(tmp_path / "arm.json")
    with pytest.raises(NativeAllJointVelocityLimitError):
        recorder.record_apply_entry(asset)


def test_dynamic_failure_records_exact_partial_cadence_and_rejects_mutations(tmp_path):
    asset = _DynamicAsset()
    recorder = NativeAllJointDynamicRecorder()
    recorder.bind_failure_path(tmp_path / "incident.json")
    for _ in range(8):
        recorder.record_apply_entry(asset)
    recorder.record_post_policy_step(asset)
    for _ in range(3):
        recorder.record_apply_entry(asset)
    asset.data.joint_vel[0, 12] = 5.25
    with pytest.raises(NativeAllJointVelocityLimitError) as captured:
        recorder.record_apply_entry(asset)
    evidence = captured.value.evidence
    assert evidence["policy_step_index"] == 1
    assert evidence["physics_substep_index"] == 3
    assert evidence["failed_apply_call_index"] == 11
    assert evidence["completed_apply_calls"] == 11
    assert evidence["completed_policy_steps"] == 1
    report = recorder.report(include_samples=True)
    assert report["apply_calls"] == 11
    assert report["post_policy_step_samples"] == 1
    assert report["sample_count"] == 12
    assert report["terminal_velocity_failure"] == evidence

    wrong_cadence = copy.deepcopy(report)
    wrong_cadence["samples"][-1]["physics_substep_index"] = 7
    with pytest.raises(ValueError, match="exact sample cadence"):
        validate_native_all_joint_dynamic_report(wrong_cadence, require_samples=True)

    for field, replacement, match in (
        ("physics_substep_index", 2, "cadence"),
        ("excess_mask", [False] * 13, "mask"),
        ("live_joint_velocity_limit", [9.0] * 13, "limit"),
        ("excess_rad_s", [0.0] * 13, "arithmetic"),
        ("violating_joint_names", ["finger_joint"], "joint identity"),
    ):
        mutated = copy.deepcopy(evidence)
        mutated[field] = replacement
        with pytest.raises(ValueError, match=match):
            validate_native_all_joint_velocity_failure(mutated)


def test_non_velocity_contract_errors_remain_plain_value_errors(tmp_path):
    asset = _DynamicAsset()
    recorder = NativeAllJointDynamicRecorder()
    recorder.bind_failure_path(tmp_path / "unused.json")
    asset.data.joint_vel_limits[0, 0] = 99.0
    with pytest.raises(
        ValueError, match="live all-joint velocity limit drift"
    ) as error:
        recorder.record_apply_entry(asset)
    assert not isinstance(error.value, NativeAllJointVelocityLimitError)
    assert not (tmp_path / "unused.json").exists()
