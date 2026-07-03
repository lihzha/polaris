from pathlib import Path

import pytest
import torch

from polaris.eef_controller_repair import advance_gripper_close_arm_interlock
from polaris.eef_controller_repair import bound_joint_position_target
from polaris.eef_controller_repair import (
    DISABLED_GRIPPER_CLOSE_ARM_INTERLOCK_TRANSITION,
)
from polaris.eef_ik_safety import ARM_SLEW_HEADROOM_RATIO
from polaris.eef_ik_safety import GRIPPER_CLOSE_ARM_INTERLOCK_SUBSTEPS


ROOT = Path(__file__).resolve().parents[1]


def test_candidate_constants_are_bounded_and_exact():
    assert ARM_SLEW_HEADROOM_RATIO == 0.95
    assert GRIPPER_CLOSE_ARM_INTERLOCK_SUBSTEPS == 48
    assert 0.0 < ARM_SLEW_HEADROOM_RATIO < 1.0


def test_disabled_interlock_is_a_zero_state_identity():
    assert DISABLED_GRIPPER_CLOSE_ARM_INTERLOCK_TRANSITION.active is False
    transition = advance_gripper_close_arm_interlock(
        enabled=False,
        previous_endpoint_change_count=0,
        current_endpoint_change_count=0,
        endpoint_observed_before_apply=False,
        endpoint_is_closed=False,
        remaining_before_apply=0,
    )
    assert transition.active is False
    assert transition.remaining_after_successful_apply == 0
    assert transition.observed_endpoint_change_count == 0
    assert transition.endpoint_observed_after_successful_apply is False
    assert transition.activation_count_delta == 0


def test_initial_close_starts_bounded_countdown_without_refresh():
    transition = advance_gripper_close_arm_interlock(
        enabled=True,
        previous_endpoint_change_count=0,
        current_endpoint_change_count=0,
        endpoint_observed_before_apply=False,
        endpoint_is_closed=True,
        remaining_before_apply=0,
    )
    assert transition.active is True
    assert transition.remaining_after_successful_apply == 47
    assert transition.activation_count_delta == 1
    assert transition.endpoint_observed_after_successful_apply is True

    for expected_remaining in range(46, -1, -1):
        transition = advance_gripper_close_arm_interlock(
            enabled=True,
            previous_endpoint_change_count=0,
            current_endpoint_change_count=0,
            endpoint_observed_before_apply=True,
            endpoint_is_closed=True,
            remaining_before_apply=transition.remaining_after_successful_apply,
        )
        assert transition.remaining_after_successful_apply == expected_remaining
        assert transition.activation_count_delta == 0
    assert transition.active is True

    released = advance_gripper_close_arm_interlock(
        enabled=True,
        previous_endpoint_change_count=0,
        current_endpoint_change_count=0,
        endpoint_observed_before_apply=True,
        endpoint_is_closed=True,
        remaining_before_apply=0,
    )
    assert released.active is False
    assert released.remaining_after_successful_apply == 0


def test_initial_open_establishes_baseline_then_close_activates():
    initial_open = advance_gripper_close_arm_interlock(
        enabled=True,
        previous_endpoint_change_count=0,
        current_endpoint_change_count=0,
        endpoint_observed_before_apply=False,
        endpoint_is_closed=False,
        remaining_before_apply=0,
    )
    assert initial_open.active is False
    assert initial_open.endpoint_observed_after_successful_apply is True
    assert initial_open.activation_count_delta == 0

    close = advance_gripper_close_arm_interlock(
        enabled=True,
        previous_endpoint_change_count=0,
        current_endpoint_change_count=1,
        endpoint_observed_before_apply=True,
        endpoint_is_closed=True,
        remaining_before_apply=0,
    )
    assert close.active is True
    assert close.remaining_after_successful_apply == 47
    assert close.activation_count_delta == 1


def test_open_transition_cancels_close_countdown_and_reclose_reactivates():
    opened = advance_gripper_close_arm_interlock(
        enabled=True,
        previous_endpoint_change_count=1,
        current_endpoint_change_count=2,
        endpoint_observed_before_apply=True,
        endpoint_is_closed=False,
        remaining_before_apply=31,
    )
    assert opened.active is False
    assert opened.remaining_after_successful_apply == 0
    assert opened.observed_endpoint_change_count == 2

    reclosed = advance_gripper_close_arm_interlock(
        enabled=True,
        previous_endpoint_change_count=2,
        current_endpoint_change_count=3,
        endpoint_observed_before_apply=True,
        endpoint_is_closed=True,
        remaining_before_apply=0,
    )
    assert reclosed.active is True
    assert reclosed.activation_count_delta == 1


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"enabled": 1}, "flags must be bool"),
        ({"endpoint_observed_before_apply": 1}, "flags must be bool"),
        ({"endpoint_is_closed": 1}, "flags must be bool"),
        ({"previous_endpoint_change_count": -1}, "non-negative int"),
        ({"current_endpoint_change_count": -1}, "non-negative int"),
        ({"remaining_before_apply": -1}, "non-negative int"),
        (
            {
                "previous_endpoint_change_count": 2,
                "current_endpoint_change_count": 1,
            },
            "count regressed",
        ),
        ({"current_endpoint_change_count": 2}, "missed a gripper endpoint"),
        (
            {"enabled": False, "remaining_before_apply": 1},
            "retained state",
        ),
        (
            {"enabled": False, "endpoint_observed_before_apply": True},
            "retained state",
        ),
        (
            {
                "endpoint_observed_before_apply": False,
                "current_endpoint_change_count": 1,
            },
            "first observed gripper endpoint",
        ),
    ],
)
def test_interlock_rejects_lifecycle_drift(overrides, match):
    arguments = {
        "enabled": True,
        "previous_endpoint_change_count": 0,
        "current_endpoint_change_count": 0,
        "endpoint_observed_before_apply": True,
        "endpoint_is_closed": False,
        "remaining_before_apply": 0,
    }
    arguments.update(overrides)
    with pytest.raises(ValueError, match=match):
        advance_gripper_close_arm_interlock(**arguments)


def test_nominal_slew_retains_full_physical_target_guard_band():
    joint_pos = torch.tensor([[0.85]], dtype=torch.float32)
    raw_target = torch.tensor([[1.20]], dtype=torch.float32)
    nominal_delta = torch.tensor([[0.095]], dtype=torch.float32)
    physical_delta = torch.tensor([[0.10]], dtype=torch.float32)
    limits = torch.tensor([[[-1.0, 1.0]]], dtype=torch.float32)
    safe, raw_delta, slew_limited, position_limited = bound_joint_position_target(
        joint_pos,
        raw_target,
        nominal_delta,
        limits,
        target_guard_band_delta_joint_pos=physical_delta,
    )
    assert torch.equal(safe, torch.tensor([[0.90]], dtype=torch.float32))
    assert torch.equal(raw_delta, raw_target - joint_pos)
    assert slew_limited.tolist() == [[True]]
    assert position_limited.tolist() == [[True]]


def test_default_off_bound_is_bitwise_identical_to_explicit_physical_guard():
    joint_pos = torch.tensor([[0.125, -0.25]], dtype=torch.float32)
    raw_target = torch.tensor([[0.2, -0.7]], dtype=torch.float32)
    physical_delta = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
    limits = torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]], dtype=torch.float32)
    inherited = bound_joint_position_target(
        joint_pos, raw_target, physical_delta, limits
    )
    explicit = bound_joint_position_target(
        joint_pos,
        raw_target,
        physical_delta,
        limits,
        target_guard_band_delta_joint_pos=physical_delta,
    )
    assert all(torch.equal(left, right) for left, right in zip(inherited, explicit))


def test_controller_candidates_are_explicit_default_off_config_and_wired():
    source = (ROOT / "src/polaris/robust_differential_ik.py").read_text()
    assert source.count("enable_arm_slew_headroom: bool = False") == 1
    assert source.count("enable_gripper_close_arm_interlock: bool = False") == 1
    assert "self._nominal_max_delta_joint_pos" in source
    assert "ARM_SLEW_HEADROOM_RATIO if self._arm_slew_headroom_enabled" in source
    assert source.count("self._nominal_max_delta_joint_pos,") >= 2
    assert "if close_interlock_transition.active:" in source
    assert "joint_pos,\n                joint_pos," in source
    setter_index = source.index("self._asset.set_joint_position_target(safe_target")
    commit_index = source.index(
        "self._gripper_close_arm_interlock_remaining = (", setter_index
    )
    assert setter_index < commit_index
    assert "self._reset_gripper_close_arm_interlock_state()" in source
    assert (
        "if self._gripper_close_arm_interlock_enabled\n"
        "            else DISABLED_GRIPPER_CLOSE_ARM_INTERLOCK_TRANSITION"
    ) in source
