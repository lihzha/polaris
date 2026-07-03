from pathlib import Path

import pytest
import torch

from polaris.eef_controller_repair import (
    advance_gripper_close_arm_interlock as _advance_gripper_close_arm_interlock,
)
from polaris.eef_controller_repair import bound_joint_position_target
from polaris.eef_controller_repair import (
    DISABLED_GRIPPER_CLOSE_ARM_INTERLOCK_TRANSITION,
)
from polaris.eef_ik_safety import ARM_SLEW_HEADROOM_RATIO
from polaris.eef_ik_safety import GRIPPER_CLOSE_ARM_INTERLOCK_SUBSTEPS
from polaris.eef_gripper_runtime import (
    EEF_GRIPPER_TARGET_SLEW_PROFILE,
    EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE,
)
from polaris.eef_gripper_runtime import eef_gripper_target_slew_profile
from polaris.eef_gripper_runtime import (
    validate_eef_gripper_close_arm_interlock_binding,
)


ROOT = Path(__file__).resolve().parents[1]


def advance_gripper_close_arm_interlock(**kwargs):
    return _advance_gripper_close_arm_interlock(
        configured_substeps=GRIPPER_CLOSE_ARM_INTERLOCK_SUBSTEPS,
        **kwargs,
    )


def test_candidate_constants_are_bounded_and_exact():
    assert ARM_SLEW_HEADROOM_RATIO == 0.95
    assert GRIPPER_CLOSE_ARM_INTERLOCK_SUBSTEPS == 48
    assert 0.0 < ARM_SLEW_HEADROOM_RATIO < 1.0
    slow = eef_gripper_target_slew_profile(
        EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
    )
    assert slow.close_transition_applies == 76
    assert slow.close_limited_applies == 75
    assert slow.close_nextafter_corrections == 41
    assert slow.close_interlock_substeps == 86
    assert slow.close_interlock_profile == (
        "eef_gripper_close_fixed_activation_anchor_86_physics_substeps_v2"
    )
    assert slow.fixed_activation_anchor is True
    baseline = eef_gripper_target_slew_profile(EEF_GRIPPER_TARGET_SLEW_PROFILE)
    assert baseline.fixed_activation_anchor is False


def test_pure_interlock_uses_the_explicit_profile_bound_duration():
    transition = _advance_gripper_close_arm_interlock(
        enabled=True,
        previous_endpoint_change_count=0,
        current_endpoint_change_count=0,
        endpoint_observed_before_apply=False,
        endpoint_is_closed=True,
        remaining_before_apply=0,
        configured_substeps=86,
    )
    assert transition.active is True
    assert transition.remaining_after_successful_apply == 85

    for configured in (0, -1, False, 86.0):
        with pytest.raises(ValueError, match="positive int"):
            _advance_gripper_close_arm_interlock(
                enabled=True,
                previous_endpoint_change_count=0,
                current_endpoint_change_count=0,
                endpoint_observed_before_apply=False,
                endpoint_is_closed=True,
                remaining_before_apply=0,
                configured_substeps=configured,
            )


def test_closed_profile_mapping_rejects_crossed_interlock_durations():
    baseline = eef_gripper_target_slew_profile(EEF_GRIPPER_TARGET_SLEW_PROFILE)
    slow = eef_gripper_target_slew_profile(
        EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
    )
    assert (
        validate_eef_gripper_close_arm_interlock_binding(
            target_slew_profile=baseline.profile,
            interlock_profile=baseline.close_interlock_profile,
            configured_substeps=baseline.close_interlock_substeps,
        )
        == baseline
    )
    assert (
        validate_eef_gripper_close_arm_interlock_binding(
            target_slew_profile=slow.profile,
            interlock_profile=slow.close_interlock_profile,
            configured_substeps=slow.close_interlock_substeps,
        )
        == slow
    )
    for target, interlock, substeps in (
        (slow.profile, baseline.close_interlock_profile, 48),
        (baseline.profile, slow.close_interlock_profile, 86),
        (slow.profile, slow.close_interlock_profile, 48),
        (baseline.profile, baseline.close_interlock_profile, 86),
    ):
        with pytest.raises(ValueError, match="profile mismatch"):
            validate_eef_gripper_close_arm_interlock_binding(
                target_slew_profile=target,
                interlock_profile=interlock,
                configured_substeps=substeps,
            )


def test_float32_close_simulation_binds_nextafter_corrections():
    baseline = eef_gripper_target_slew_profile(EEF_GRIPPER_TARGET_SLEW_PROFILE)
    slow = eef_gripper_target_slew_profile(
        EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
    )
    assert (
        baseline.close_transition_applies,
        baseline.close_limited_applies,
        baseline.close_nextafter_corrections,
    ) == (38, 37, 15)
    assert (
        slow.close_transition_applies,
        slow.close_limited_applies,
        slow.close_nextafter_corrections,
    ) == (76, 75, 41)


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
    assert transition.completion_count_delta == 0
    assert transition.open_cancel_count_delta == 0


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
        assert transition.completion_count_delta == int(expected_remaining == 0)
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
    assert opened.open_cancel_count_delta == 1
    assert opened.completion_count_delta == 0

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
    assert reclosed.open_cancel_count_delta == 0


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


def test_job1098476_drift_returns_toward_activation_anchor_without_direct_jump():
    # Exact apply-920 activation q and apply-967 current q from the immutable
    # job-1098476 failure ring.  A direct anchor write exceeds the nominal
    # bound on joints 5 and 7; the production bounding helper must not.
    anchor = torch.tensor(
        [
            [
                -0.20430134236812592,
                -0.24422302842140198,
                -0.245017409324646,
                -2.8719422817230225,
                -0.47118961811065674,
                2.4242470264434814,
                -0.16500917077064514,
            ]
        ],
        dtype=torch.float32,
    )
    joint_pos = torch.tensor(
        [
            [
                -0.20517736673355103,
                -0.2497541606426239,
                -0.24595116078853607,
                -2.866048574447632,
                -0.4345424473285675,
                2.419935464859009,
                -0.1930602639913559,
            ]
        ],
        dtype=torch.float32,
    )
    nominal_delta = torch.tensor(
        [
            [
                0.017218751832842827,
                0.017218751832842827,
                0.017218751832842827,
                0.017218751832842827,
                0.020662499591708183,
                0.020662499591708183,
                0.020662499591708183,
            ]
        ],
        dtype=torch.float32,
    )
    physical_delta = torch.tensor(
        [
            [
                0.018125001341104507,
                0.018125001341104507,
                0.018125001341104507,
                0.018125001341104507,
                0.02174999937415123,
                0.02174999937415123,
                0.02174999937415123,
            ]
        ],
        dtype=torch.float32,
    )
    limits = torch.tensor(
        [
            [
                [-2.8973, 2.8973],
                [-1.7628, 1.7628],
                [-2.8973, 2.8973],
                [-3.0718, -0.0698],
                [-2.8973, 2.8973],
                [-0.0175, 3.7525],
                [-2.8973, 2.8973],
            ]
        ],
        dtype=torch.float32,
    )

    safe, raw_delta, slew_limited, position_limited = bound_joint_position_target(
        joint_pos,
        anchor,
        nominal_delta,
        limits,
        target_guard_band_delta_joint_pos=physical_delta,
    )

    assert (raw_delta.abs() > nominal_delta).tolist() == [
        [False, False, False, False, True, False, True]
    ]
    assert slew_limited.tolist() == [[False, False, False, False, True, False, True]]
    assert not position_limited.any()
    assert not torch.equal(safe, anchor)
    assert ((safe - joint_pos).abs() <= nominal_delta + 1e-6).all()
    assert (safe >= torch.minimum(joint_pos, anchor)).all()
    assert (safe <= torch.maximum(joint_pos, anchor)).all()
    assert ((anchor - safe).abs() <= (anchor - joint_pos).abs()).all()


def test_first_fixed_anchor_target_is_bitwise_exact_when_anchor_is_current_q():
    joint_pos = torch.tensor([[0.4, -0.3]], dtype=torch.float32)
    anchor = joint_pos.detach().clone()
    nominal_delta = torch.tensor([[0.095, 0.095]], dtype=torch.float32)
    physical_delta = torch.tensor([[0.10, 0.10]], dtype=torch.float32)
    limits = torch.tensor([[[-2.0, 2.0], [-2.0, 2.0]]], dtype=torch.float32)

    safe, _, slew_limited, position_limited = bound_joint_position_target(
        joint_pos,
        anchor,
        nominal_delta,
        limits,
        target_guard_band_delta_joint_pos=physical_delta,
    )

    assert torch.equal(safe, anchor)
    assert not slew_limited.any()
    assert not position_limited.any()


def test_controller_candidates_are_explicit_default_off_config_and_wired():
    source = (ROOT / "src/polaris/robust_differential_ik.py").read_text()
    assert source.count("enable_arm_slew_headroom: bool = False") == 1
    assert source.count("enable_gripper_close_arm_interlock: bool = False") == 1
    assert "self._nominal_max_delta_joint_pos" in source
    assert "ARM_SLEW_HEADROOM_RATIO if self._arm_slew_headroom_enabled" in source
    assert source.count("self._nominal_max_delta_joint_pos,") >= 2
    assert "if close_interlock_transition.active:" in source
    assert "fixed_anchor_for_apply.unsqueeze(0)" in source
    assert "fixed_anchor_for_apply = joint_pos[0].detach().clone()" in source
    assert "self._gripper_close_arm_interlock_anchor.detach().clone()" in source
    transaction_start = source.index(
        "def _set_targets_and_commit_gripper_close_arm_interlock("
    )
    velocity_setter_index = source.index(
        "self._asset.set_joint_velocity_target(", transaction_start
    )
    setter_index = source.index(
        "self._asset.set_joint_position_target(safe_target", transaction_start
    )
    anchor_commit_index = source.index(
        "self._gripper_close_arm_interlock_anchor = staged.anchor", setter_index
    )
    commit_index = source.index(
        "self._gripper_close_arm_interlock_remaining = staged.remaining",
        setter_index,
    )
    assert velocity_setter_index < setter_index < anchor_commit_index < commit_index
    assert "_StagedGripperCloseArmInterlockState(" in source
    assert "self._set_targets_and_commit_gripper_close_arm_interlock(" in source
    assert "self._reset_gripper_close_arm_interlock_state()" in source
    assert "self._gripper_close_arm_interlock_anchor_valid = False" in source
    assert "self._gripper_close_arm_interlock_anchor_refresh_count = 0" in source
    assert (
        "if self._gripper_close_arm_interlock_enabled\n"
        "            else DISABLED_GRIPPER_CLOSE_ARM_INTERLOCK_TRANSITION"
    ) in source
