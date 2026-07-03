from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from polaris import eef_gripper_runtime as runtime
from polaris import eef_gripper_target_slew as target_slew


class _FakeAsset:
    def __init__(self, *, joint_position=0.0, dtype=torch.float32):
        self.data = SimpleNamespace(
            joint_pos=torch.zeros((1, 13), dtype=dtype),
            joint_pos_target=torch.zeros((1, 13), dtype=dtype),
        )
        self.data.joint_pos[0, runtime.DRIVEN_GRIPPER_JOINT_INDEX] = joint_position
        self.data.joint_pos_target[0, runtime.DRIVEN_GRIPPER_JOINT_INDEX] = (
            joint_position
        )
        limit = torch.tensor(
            [[runtime.GRIPPER_DRIVER_VELOCITY_LIMIT_FLOAT32]], dtype=dtype
        )
        self.actuators = {
            "gripper": SimpleNamespace(
                cfg=SimpleNamespace(
                    velocity_limit=runtime.GRIPPER_DRIVER_VELOCITY_LIMIT_FLOAT32,
                    velocity_limit_sim=(runtime.GRIPPER_DRIVER_VELOCITY_LIMIT_FLOAT32),
                ),
                velocity_limit=limit.clone(),
                velocity_limit_sim=limit.clone(),
            )
        }
        self.setter_calls: list[torch.Tensor] = []

    def set_joint_position_target(self, value, joint_ids):
        self.setter_calls.append(value.clone())
        self.data.joint_pos_target[:, joint_ids] = value


class _PinnedBinaryJointPositionActionStub:
    """Behavioral stub of Isaac Lab v2.3 BinaryJointPositionAction."""

    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env
        self.num_envs = 1
        self.device = "cpu"
        self._asset = env.asset
        self._joint_ids = [runtime.DRIVEN_GRIPPER_JOINT_INDEX]
        self._joint_names = [runtime.DRIVEN_GRIPPER_JOINT_NAME]
        self._raw_actions = torch.zeros((1, 1), dtype=torch.float32)
        self._processed_actions = torch.zeros((1, 1), dtype=torch.float32)
        self._open_command = torch.tensor(
            [runtime.GRIPPER_OPEN_TARGET_FLOAT32], dtype=torch.float32
        )
        self._close_command = torch.tensor(
            [runtime.GRIPPER_CLOSED_TARGET_FLOAT32], dtype=torch.float32
        )

    def process_actions(self, actions):
        self._raw_actions[:] = actions
        binary_mask = actions >= 0.5
        self._processed_actions = torch.where(
            binary_mask, self._close_command, self._open_command
        )

    def reset(self, env_ids=None):
        self._raw_actions[env_ids] = 0.0


class _HostEefTargetSlewAction(
    target_slew.EefGripperTargetSlewMixin,
    _PinnedBinaryJointPositionActionStub,
):
    pass


@pytest.fixture
def host_device_contract(monkeypatch):
    # Production is pinned to cuda:0. The host behavioral test changes only
    # the expected device string; all dtype, value, shape, and transition
    # checks remain production-identical.
    monkeypatch.setattr(target_slew, "PINNED_ACTUATOR_DEVICE", "cpu")
    monkeypatch.setattr(runtime, "PINNED_ACTUATOR_DEVICE", "cpu")


def _action(*, joint_position=0.0):
    asset = _FakeAsset(joint_position=joint_position)
    env = SimpleNamespace(
        physics_dt=runtime.GRIPPER_TARGET_SLEW_PHYSICS_DT,
        asset=asset,
    )
    action = _HostEefTargetSlewAction(SimpleNamespace(clip=None), env)
    static = action.gripper_target_slew_static_contract()
    action.install_gripper_target_slew_contract(static)
    return action, asset, static


def _apply_policy_action(action, value, *, substeps=8):
    action.process_actions(torch.tensor([[value]], dtype=torch.float32))
    for _ in range(substeps):
        action.apply_actions()


def _assert_empty_apply_evidence(action, *, process_calls):
    report = action.gripper_target_slew_dynamic_report()
    assert report["process_action_calls"] == process_calls
    assert report["apply_calls"] == 0
    assert report["initialization_count"] == 0
    assert report["live_limit_validation_count"] == 0
    assert report["slew_limited_apply_count"] == 0
    assert report["endpoint_reached_apply_count"] == 0
    assert report["max_abs_target_step_rad"] == 0.0
    assert report["max_abs_endpoint_error_before_step_rad"] == 0.0
    assert report["max_abs_endpoint_error_after_step_rad"] == 0.0
    assert report["initial_anchor_rad"] is None
    assert report["last_applied_target_rad"] is None
    return report


def _adjacent_float32(value, direction):
    return torch.nextafter(
        torch.tensor(value, dtype=torch.float32),
        torch.tensor(direction, dtype=torch.float32),
    ).item()


def test_static_contract_uses_exact_live_float32_five_over_120_cap(
    host_device_contract,
):
    action, _, static = _action()
    assert static["profile"] == runtime.EEF_GRIPPER_TARGET_SLEW_PROFILE
    assert static["scope"] == "eef_pose_only_native_joint_position_unchanged_v1"
    assert static["endpoint_semantics_profile"] == runtime.GRIPPER_THRESHOLD_PROFILE
    assert static["open_target_rad"] == runtime.GRIPPER_OPEN_TARGET_FLOAT32
    assert static["closed_target_rad"] == runtime.GRIPPER_CLOSED_TARGET_FLOAT32
    assert static["velocity_limit_rad_s"] == 5.0
    assert static["physics_hz"] == 120.0
    assert static["max_target_step_rad"] == (runtime.GRIPPER_MAX_TARGET_STEP_FLOAT32)
    assert action._gripper_target_slew_max_step.item() == (
        runtime.GRIPPER_MAX_TARGET_STEP_FLOAT32
    )


def test_open_close_exact_cap_no_overshoot_and_threshold_equality(
    host_device_contract,
):
    action, asset, _ = _action()
    targets = []
    for substeps in (8, 8, 8):
        action.process_actions(torch.tensor([[0.5]], dtype=torch.float32))
        for _ in range(substeps):
            action.apply_actions()
            targets.append(asset.data.joint_pos_target[0, 7].item())

    steps = [targets[0], *[right - left for left, right in zip(targets, targets[1:])]]
    assert all(step <= runtime.GRIPPER_MAX_TARGET_STEP_FLOAT32 + 1e-7 for step in steps)
    assert targets == sorted(targets)
    assert targets[-1] == runtime.GRIPPER_CLOSED_TARGET_FLOAT32
    assert all(value <= runtime.GRIPPER_CLOSED_TARGET_FLOAT32 for value in targets)

    open_targets = []
    for substeps in (8, 8, 8):
        action.process_actions(torch.tensor([[0.49999997]], dtype=torch.float32))
        for _ in range(substeps):
            action.apply_actions()
            open_targets.append(asset.data.joint_pos_target[0, 7].item())
    assert open_targets == sorted(open_targets, reverse=True)
    assert open_targets[-1] == runtime.GRIPPER_OPEN_TARGET_FLOAT32
    assert all(value >= runtime.GRIPPER_OPEN_TARGET_FLOAT32 for value in open_targets)

    report = action.gripper_target_slew_dynamic_report()
    assert report["process_action_calls"] == 6
    assert report["endpoint_change_count"] == 1
    assert report["repeated_endpoint_process_count"] == 4
    assert report["apply_calls"] == 48
    assert report["max_abs_target_step_rad"] == (
        runtime.GRIPPER_MAX_TARGET_STEP_FLOAT32
    )
    assert report["last_requested_endpoint_rad"] == 0.0
    assert report["last_applied_target_rad"] == 0.0


def test_float32_cap_boundary_never_exceeds_cap_and_then_reaches_endpoint(
    host_device_contract,
):
    start = (
        runtime.GRIPPER_CLOSED_TARGET_FLOAT32 - runtime.GRIPPER_MAX_TARGET_STEP_FLOAT32
    )
    action, asset, _ = _action(joint_position=start)
    action.process_actions(torch.tensor([[1.0]], dtype=torch.float32))
    represented_start = asset.data.joint_pos[0, 7].item()
    represented_error = runtime.GRIPPER_CLOSED_TARGET_FLOAT32 - represented_start
    assert represented_error > runtime.GRIPPER_MAX_TARGET_STEP_FLOAT32
    assert represented_error <= (
        runtime.GRIPPER_MAX_TARGET_STEP_FLOAT32
        + runtime.GRIPPER_TARGET_SLEW_FLOAT32_TOLERANCE_RAD
    )
    action.apply_actions()
    first_target = asset.data.joint_pos_target[0, 7].item()
    assert first_target < runtime.GRIPPER_CLOSED_TARGET_FLOAT32
    assert first_target - represented_start <= (runtime.GRIPPER_MAX_TARGET_STEP_FLOAT32)
    action.apply_actions()
    report = action.gripper_target_slew_dynamic_report()

    assert asset.data.joint_pos_target[0, 7].item() == (
        runtime.GRIPPER_CLOSED_TARGET_FLOAT32
    )
    assert report["slew_limited_apply_count"] == 1
    assert report["endpoint_reached_apply_count"] == 1
    assert report["max_abs_target_step_rad"] <= (
        runtime.GRIPPER_MAX_TARGET_STEP_FLOAT32
        + runtime.GRIPPER_TARGET_SLEW_FLOAT32_TOLERANCE_RAD
    )


def test_job1098263_repeated_open_then_persistent_close_sequence_is_slewed(
    host_device_contract,
):
    action, asset, _ = _action()
    # Validated trace: closed_cmd=0 through action 114, flips to 1 at action
    # 115, and remains 1 at actions 116/117. The original arm velocity abort
    # occurred at action 117/substep 3, 19 physics applies after the flip.
    for _policy_step in (112, 113, 114):
        _apply_policy_action(action, 0.0)
    assert asset.data.joint_pos_target[0, 7].item() == 0.0

    close_targets = []
    for _policy_step in (115, 116, 117):
        action.process_actions(torch.tensor([[1.0]], dtype=torch.float32))
        for _ in range(8):
            action.apply_actions()
            close_targets.append(asset.data.joint_pos_target[0, 7].item())

    assert close_targets[17] < runtime.GRIPPER_CLOSED_TARGET_FLOAT32
    assert close_targets[18] == runtime.GRIPPER_CLOSED_TARGET_FLOAT32
    assert close_targets[19] == runtime.GRIPPER_CLOSED_TARGET_FLOAT32
    report = action.gripper_target_slew_dynamic_report()
    assert report["process_action_calls"] == 6
    assert report["endpoint_change_count"] == 1
    assert report["repeated_endpoint_process_count"] == 4
    assert report["apply_calls"] == 48
    assert report["slew_limited_apply_count"] == 18
    assert report["endpoint_reached_apply_count"] == 30


def test_reset_clears_history_and_reanchors_from_live_position(host_device_contract):
    action, asset, _ = _action()
    _apply_policy_action(action, 1.0, substeps=3)
    assert action.gripper_target_slew_dynamic_report()["apply_calls"] == 3

    action.reset()
    asset.data.joint_pos[0, 7] = 0.1
    asset.data.joint_pos_target[0, 7] = 0.7  # stale target must not be the anchor
    empty = action.gripper_target_slew_dynamic_report()
    assert empty["process_action_calls"] == 0
    assert empty["apply_calls"] == 0
    assert empty["initial_anchor_rad"] is None

    action.process_actions(torch.tensor([[0.0]], dtype=torch.float32))
    action.apply_actions()
    report = action.gripper_target_slew_dynamic_report()
    assert report["initial_anchor_rad"] == pytest.approx(0.1)
    assert asset.data.joint_pos_target[0, 7].item() == pytest.approx(
        0.1 - runtime.GRIPPER_MAX_TARGET_STEP_FLOAT32
    )
    assert report["initialization_count"] == 1


@pytest.mark.parametrize(
    ("joint_position", "command"),
    [
        (-0.01, 0.0),
        (runtime.GRIPPER_CLOSED_TARGET_FLOAT32 + 0.01, 1.0),
    ],
)
def test_out_of_profile_initial_anchor_fails_before_write_or_evidence_commit(
    host_device_contract, joint_position, command
):
    action, asset, _ = _action(joint_position=joint_position)
    action.process_actions(torch.tensor([[command]], dtype=torch.float32))

    with pytest.raises(
        target_slew.EefGripperTargetSlewError, match="initial live anchor"
    ):
        action.apply_actions()

    assert asset.setter_calls == []
    _assert_empty_apply_evidence(action, process_calls=1)


@pytest.mark.parametrize(
    ("in_bound_anchor", "out_of_bound_anchor", "command"),
    [
        (
            runtime.GRIPPER_TARGET_SLEW_MIN_ANCHOR_FLOAT32,
            _adjacent_float32(
                runtime.GRIPPER_TARGET_SLEW_MIN_ANCHOR_FLOAT32, -float("inf")
            ),
            1.0,
        ),
        (
            runtime.GRIPPER_TARGET_SLEW_MAX_ANCHOR_FLOAT32,
            _adjacent_float32(
                runtime.GRIPPER_TARGET_SLEW_MAX_ANCHOR_FLOAT32, float("inf")
            ),
            0.0,
        ),
    ],
)
def test_float32_tolerance_boundary_and_adjacent_outside_are_closed(
    host_device_contract, in_bound_anchor, out_of_bound_anchor, command
):
    accepted, accepted_asset, _ = _action(joint_position=in_bound_anchor)
    accepted.process_actions(torch.tensor([[command]], dtype=torch.float32))
    accepted.apply_actions()
    assert len(accepted_asset.setter_calls) == 1
    accepted_report = accepted.gripper_target_slew_dynamic_report()
    assert accepted_report["apply_calls"] == 1
    assert accepted_report["initialization_count"] == 1
    assert accepted_report["initial_anchor_rad"] == in_bound_anchor
    assert accepted_report["max_abs_endpoint_error_before_step_rad"] == (
        runtime.GRIPPER_TARGET_SLEW_MAX_ANCHOR_FLOAT32
    )

    rejected, rejected_asset, _ = _action(joint_position=out_of_bound_anchor)
    rejected.process_actions(torch.tensor([[command]], dtype=torch.float32))
    with pytest.raises(
        target_slew.EefGripperTargetSlewError, match="initial live anchor"
    ):
        rejected.apply_actions()
    assert rejected_asset.setter_calls == []
    _assert_empty_apply_evidence(rejected, process_calls=1)


def test_rapid_mid_slew_endpoint_reversal_remains_capped_and_reaches_open(
    host_device_contract,
):
    action, asset, _ = _action()
    action.process_actions(torch.tensor([[1.0]], dtype=torch.float32))
    close_targets = []
    for _ in range(5):
        action.apply_actions()
        close_targets.append(asset.data.joint_pos_target[0, 7].item())
    assert close_targets == sorted(close_targets)
    assert close_targets[-1] < runtime.GRIPPER_CLOSED_TARGET_FLOAT32

    action.process_actions(torch.tensor([[0.0]], dtype=torch.float32))
    open_targets = []
    for _ in range(20):
        previous = asset.data.joint_pos_target[0, 7].item()
        action.apply_actions()
        current = asset.data.joint_pos_target[0, 7].item()
        open_targets.append(current)
        assert 0.0 <= current <= previous
        assert previous - current <= runtime.GRIPPER_MAX_TARGET_STEP_FLOAT32
        if current == runtime.GRIPPER_OPEN_TARGET_FLOAT32:
            break

    assert open_targets == sorted(open_targets, reverse=True)
    assert open_targets[-1] == runtime.GRIPPER_OPEN_TARGET_FLOAT32
    report = action.gripper_target_slew_dynamic_report()
    assert report["process_action_calls"] == 2
    assert report["endpoint_change_count"] == 1
    assert report["repeated_endpoint_process_count"] == 0
    assert report["apply_calls"] == 5 + len(open_targets)
    assert report["max_abs_target_step_rad"] <= (
        runtime.GRIPPER_MAX_TARGET_STEP_FLOAT32
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_action_fails_before_target_write(host_device_contract, value):
    action, asset, _ = _action()
    with pytest.raises(target_slew.EefGripperTargetSlewError, match="input tensor"):
        action.process_actions(torch.tensor([[value]], dtype=torch.float32))
    assert asset.setter_calls == []
    _assert_empty_apply_evidence(action, process_calls=0)


@pytest.mark.parametrize("field", ["joint_pos", "joint_pos_target"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_live_state_fails_before_target_write(
    host_device_contract, field, value
):
    action, asset, _ = _action()
    action.process_actions(torch.tensor([[1.0]], dtype=torch.float32))
    getattr(asset.data, field)[0, runtime.DRIVEN_GRIPPER_JOINT_INDEX] = value
    with pytest.raises(target_slew.EefGripperTargetSlewError, match="live .* tensor"):
        action.apply_actions()
    assert asset.setter_calls == []
    getattr(asset.data, field)[0, runtime.DRIVEN_GRIPPER_JOINT_INDEX] = 0.0
    _assert_empty_apply_evidence(action, process_calls=1)


@pytest.mark.parametrize("field", ["joint_pos", "joint_pos_target"])
def test_live_state_dtype_drift_fails_before_target_write(host_device_contract, field):
    action, asset, _ = _action()
    action.process_actions(torch.tensor([[1.0]], dtype=torch.float32))
    setattr(asset.data, field, getattr(asset.data, field).to(torch.float64))
    with pytest.raises(target_slew.EefGripperTargetSlewError, match="live .* tensor"):
        action.apply_actions()
    assert asset.setter_calls == []
    setattr(asset.data, field, getattr(asset.data, field).to(torch.float32))
    _assert_empty_apply_evidence(action, process_calls=1)


def test_wrong_action_dtype_fails_before_processing(host_device_contract):
    action, asset, _ = _action()
    with pytest.raises(target_slew.EefGripperTargetSlewError, match="input tensor"):
        action.process_actions(torch.tensor([[1.0]], dtype=torch.float64))
    assert asset.setter_calls == []
    _assert_empty_apply_evidence(action, process_calls=0)


@pytest.mark.parametrize("surface", ["cfg", "legacy", "simulation"])
def test_live_limit_drift_fails_before_next_target_write(host_device_contract, surface):
    action, asset, _ = _action()
    action.process_actions(torch.tensor([[1.0]], dtype=torch.float32))
    before = len(asset.setter_calls)
    actuator = asset.actuators["gripper"]
    if surface == "cfg":
        actuator.cfg.velocity_limit_sim = 4.9
    elif surface == "legacy":
        actuator.velocity_limit.fill_(4.9)
    else:
        actuator.velocity_limit_sim = actuator.velocity_limit_sim.to(torch.float64)
    with pytest.raises(target_slew.EefGripperTargetSlewError, match="limit"):
        action.apply_actions()
    assert len(asset.setter_calls) == before
    actuator.cfg.velocity_limit = runtime.GRIPPER_DRIVER_VELOCITY_LIMIT_FLOAT32
    actuator.cfg.velocity_limit_sim = runtime.GRIPPER_DRIVER_VELOCITY_LIMIT_FLOAT32
    actuator.velocity_limit = torch.full(
        (1, 1), runtime.GRIPPER_DRIVER_VELOCITY_LIMIT_FLOAT32, dtype=torch.float32
    )
    actuator.velocity_limit_sim = actuator.velocity_limit.clone()
    _assert_empty_apply_evidence(action, process_calls=1)


@pytest.mark.parametrize(
    ("surface", "expected_message"),
    [
        ("profile", "profile drift"),
        ("endpoint", "endpoint profile drift"),
        ("action_dtype", "action tensor profile drift"),
        ("ownership", "ownership profile drift"),
    ],
)
def test_profile_surface_drift_fails_before_write(
    host_device_contract, surface, expected_message
):
    action, asset, _ = _action()
    action.process_actions(torch.tensor([[1.0]], dtype=torch.float32))
    action.apply_actions()
    before = len(asset.setter_calls)
    successful_report = action.gripper_target_slew_dynamic_report()
    if surface == "profile":
        action.gripper_target_slew_profile = "drifted"
    elif surface == "endpoint":
        action._close_command.fill_(0.7)
    elif surface == "action_dtype":
        action._open_command = action._open_command.to(torch.float64)
    else:
        action._joint_ids = [runtime.DRIVEN_GRIPPER_JOINT_INDEX + 1]
    with pytest.raises(target_slew.EefGripperTargetSlewError, match=expected_message):
        action.apply_actions()
    assert len(asset.setter_calls) == before
    action.gripper_target_slew_profile = runtime.EEF_GRIPPER_TARGET_SLEW_PROFILE
    action._close_command = torch.tensor(
        [runtime.GRIPPER_CLOSED_TARGET_FLOAT32], dtype=torch.float32
    )
    action._open_command = action._open_command.to(torch.float32)
    action._joint_ids = [runtime.DRIVEN_GRIPPER_JOINT_INDEX]
    assert action.gripper_target_slew_dynamic_report() == successful_report


def test_external_target_drift_fails_before_write(host_device_contract):
    action, asset, _ = _action()
    action.process_actions(torch.tensor([[1.0]], dtype=torch.float32))
    action.apply_actions()
    before = len(asset.setter_calls)
    successful_report = action.gripper_target_slew_dynamic_report()
    asset.data.joint_pos_target[0, 7] += 0.01
    with pytest.raises(target_slew.EefGripperTargetSlewError, match="external target"):
        action.apply_actions()
    assert len(asset.setter_calls) == before
    asset.data.joint_pos_target[:, action._joint_ids] = (
        action._gripper_target_slew_current
    )
    assert action.gripper_target_slew_dynamic_report() == successful_report


def test_setter_readback_drift_is_detected_without_committing_evidence(
    host_device_contract,
):
    action, asset, _ = _action()
    action.process_actions(torch.tensor([[1.0]], dtype=torch.float32))

    def drifting_setter(value, joint_ids):
        asset.setter_calls.append(value.clone())
        asset.data.joint_pos_target[:, joint_ids] = value + 0.001

    asset.set_joint_position_target = drifting_setter
    with pytest.raises(target_slew.EefGripperTargetSlewError, match="setter/readback"):
        action.apply_actions()
    assert len(asset.setter_calls) == 1
    assert action._gripper_target_slew_apply_calls == 0
    assert action._gripper_target_slew_live_limit_validation_count == 0
    assert action._gripper_target_slew_initialization_count == 0
    assert action._gripper_target_slew_initialized is False
    assert action._gripper_target_slew_current.item() == 0.0
    _assert_empty_apply_evidence(action, process_calls=1)


def test_initialized_setter_readback_failure_retains_last_committed_evidence(
    host_device_contract,
):
    action, asset, _ = _action()
    action.process_actions(torch.tensor([[1.0]], dtype=torch.float32))
    action.apply_actions()
    successful_report = action.gripper_target_slew_dynamic_report()

    def drifting_setter(value, joint_ids):
        asset.setter_calls.append(value.clone())
        asset.data.joint_pos_target[:, joint_ids] = value + 0.001

    asset.set_joint_position_target = drifting_setter
    with pytest.raises(target_slew.EefGripperTargetSlewError, match="setter/readback"):
        action.apply_actions()
    assert len(asset.setter_calls) == 2
    asset.data.joint_pos_target[:, action._joint_ids] = (
        action._gripper_target_slew_current
    )
    assert action.gripper_target_slew_dynamic_report() == successful_report


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("profile", "wrong"),
        ("scope", "native_too"),
        ("action_class", "BinaryJointPositionZeroToOneAction"),
        ("velocity_limit_rad_s", 4.0),
        ("physics_hz", 15.0),
        ("max_target_step_rad", 5.0 / 15.0),
        ("tensor_dtype", "torch.float64"),
        ("tensor_device", "cpu"),
    ],
)
def test_static_contract_rejects_profile_limit_cadence_and_tensor_drift(field, value):
    contract = {
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
        "velocity_limit_rad_s": 5.0,
        "physics_hz": 120.0,
        "physics_dt": 1.0 / 120.0,
        "max_target_step_rad": runtime.GRIPPER_MAX_TARGET_STEP_FLOAT32,
        "float32_tolerance_rad": 1e-6,
        "reset_profile": runtime.EEF_GRIPPER_TARGET_SLEW_RESET_PROFILE,
        "tensor_dtype": "torch.float32",
        "tensor_device": "cuda:0",
    }
    contract[field] = value
    with pytest.raises(ValueError):
        runtime.validate_eef_gripper_target_slew_static(contract)


def test_dynamic_contract_rejects_counter_maximum_and_nullability_drift():
    dynamic = {
        "profile": runtime.EEF_GRIPPER_TARGET_SLEW_PROFILE,
        "process_action_calls": 2,
        "apply_calls": 8,
        "initialization_count": 1,
        "endpoint_change_count": 1,
        "repeated_endpoint_process_count": 0,
        "slew_limited_apply_count": 7,
        "endpoint_reached_apply_count": 1,
        "live_limit_validation_count": 8,
        "max_abs_target_step_rad": runtime.GRIPPER_MAX_TARGET_STEP_FLOAT32,
        "max_abs_endpoint_error_before_step_rad": (
            runtime.GRIPPER_CLOSED_TARGET_FLOAT32
        ),
        "max_abs_endpoint_error_after_step_rad": (
            runtime.GRIPPER_CLOSED_TARGET_FLOAT32
            - runtime.GRIPPER_MAX_TARGET_STEP_FLOAT32
        ),
        "initial_anchor_rad": 0.0,
        "last_requested_endpoint_rad": runtime.GRIPPER_CLOSED_TARGET_FLOAT32,
        "last_applied_target_rad": runtime.GRIPPER_MAX_TARGET_STEP_FLOAT32,
    }
    runtime.validate_eef_gripper_target_slew_dynamic(dynamic)
    for field, value in (
        ("live_limit_validation_count", 1),
        ("initialization_count", 0),
        (
            "max_abs_target_step_rad",
            runtime.GRIPPER_MAX_TARGET_STEP_FLOAT32 + 2e-6,
        ),
        ("last_requested_endpoint_rad", 0.25),
        ("last_applied_target_rad", None),
    ):
        candidate = copy.deepcopy(dynamic)
        candidate[field] = value
        with pytest.raises(ValueError):
            runtime.validate_eef_gripper_target_slew_dynamic(candidate)


def _class_ast(source: str, names: set[str]) -> str:
    tree = ast.parse(source)
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in names
    ]
    return ast.unparse(ast.Module(body=selected, type_ignores=[]))


def test_native_joint_position_action_ast_is_unchanged_and_eef_alone_selects_slew():
    path = Path(__file__).parents[1] / "src/polaris/environments/droid_cfg.py"
    source = path.read_text(encoding="utf-8")
    native_ast = _class_ast(
        source,
        {
            "BinaryJointPositionZeroToOneAction",
            "BinaryJointPositionZeroToOneActionCfg",
            "ActionCfg",
        },
    )
    # Snapshot from exact base 95f57bb. New EEF-only classes and imports are
    # intentionally outside this native AST identity.
    assert hashlib.sha256(native_ast.encode()).hexdigest() == (
        "100e854c6728df5665dda3bf4c5fb6f109e1f710e5fbaec49aba982692e50bcd"
    )
    tree = ast.parse(source)
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}

    def finger_constructor(class_name):
        assignment = next(
            node
            for node in classes[class_name].body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "finger_joint"
                for target in node.targets
            )
        )
        assert isinstance(assignment.value, ast.Call)
        assert isinstance(assignment.value.func, ast.Name)
        return assignment.value.func.id

    assert finger_constructor("ActionCfg") == ("BinaryJointPositionZeroToOneActionCfg")
    assert finger_constructor("EefPoseActionCfg") == (
        "BinaryJointPositionZeroToOneActionCfg"
    )
    assert finger_constructor("EgoLapEefPoseActionCfg") == (
        "EefBinaryJointPositionTargetSlewActionCfg"
    )


def test_public_eval_and_controller_smoke_install_candidate_before_first_apply():
    root = Path(__file__).parents[1]
    eval_source = (root / "scripts/eval.py").read_text(encoding="utf-8")
    assert "EgoLapEefPoseActionCfg() if is_ego_lap else EefPoseActionCfg()" in (
        eval_source
    )
    assert "enable_gripper_velocity_limit=is_ego_lap" in eval_source
    assert "install_eef_gripper_runtime(" in eval_source
    assert "validate_eef_gripper_post_reset(" in eval_source
    assert "record_eef_gripper_post_policy_step(env)" in eval_source

    smoke_source = (root / "scripts/smoke_eef_pose_controller.py").read_text(
        encoding="utf-8"
    )
    assert "env_cfg.actions = EgoLapEefPoseActionCfg()" in smoke_source
    assert "enable_gripper_velocity_limit=True" in smoke_source
    assert "gripper_open = 0.0 if case_index == 0 else 1.0" in smoke_source
    assert (
        smoke_source.index("env.reset(expensive=False)")
        < smoke_source.index("install_eef_gripper_runtime(")
        < smoke_source.index("initial_capture = arm_term.safety_report()")
    )
    assert "validate_eef_gripper_post_reset(env, gripper_runtime_contract)" in (
        smoke_source
    )
    assert smoke_source.count("record_eef_gripper_post_policy_step(env)") == 2
