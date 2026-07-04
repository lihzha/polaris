from pathlib import Path

import pytest
import torch

from polaris.eef_controller_repair import (
    advance_arm_release_ramp,
    apply_arm_release_ramp_target,
    ARM_RELEASE_PHASE_HOLD,
    ARM_RELEASE_PHASE_RAMP,
    ARM_RELEASE_PHASE_RELEASE,
    ARM_RELEASE_RAMP_SUBSTEPS,
    arm_release_ramp_fraction,
)
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


def test_release_ramp_fractions_are_closed_float32_endpoints():
    fractions = [
        arm_release_ramp_fraction(index) for index in range(ARM_RELEASE_RAMP_SUBSTEPS)
    ]
    assert fractions[0] == 0.0
    assert fractions[-1] == 1.0
    assert fractions == sorted(fractions)
    assert all(
        float(torch.tensor(value, dtype=torch.float32).item()) == value
        for value in fractions
    )
    for invalid in (-1, ARM_RELEASE_RAMP_SUBSTEPS, True, 0.0):
        with pytest.raises(ValueError, match="index drift"):
            arm_release_ramp_fraction(invalid)


def test_release_ramp_target_preserves_exact_endpoints_and_float32_clamp():
    current = torch.tensor([[0.0, 0.4, -0.4]], dtype=torch.float32)
    nominal = torch.tensor([[0.09, 0.35, -0.31]], dtype=torch.float32)
    maximum = torch.tensor([[0.095, 0.095, 0.095]], dtype=torch.float32)

    first = apply_arm_release_ramp_target(current, nominal, maximum, ramp_index=0)
    middle = apply_arm_release_ramp_target(current, nominal, maximum, ramp_index=8)
    last = apply_arm_release_ramp_target(current, nominal, maximum, ramp_index=15)

    assert torch.equal(first.target, current)
    assert first.limited_joint_mask.tolist() == [[True, True, True]]
    fraction = torch.tensor(8 / 15, dtype=torch.float32)
    expected = current + torch.clamp(
        nominal - current,
        min=-(maximum * fraction),
        max=maximum * fraction,
    )
    assert torch.equal(middle.target, expected)
    assert torch.equal(last.target, nominal)
    assert not last.limited_joint_mask.any()
    assert first.target.data_ptr() != current.data_ptr()
    assert last.target.data_ptr() != nominal.data_ptr()


def test_release_ramp_target_matches_randomized_float32_reference():
    generator = torch.Generator().manual_seed(42)
    maximum = torch.tensor(
        [[0.0172187518] * 4 + [0.0206624996] * 3], dtype=torch.float32
    )
    for _ in range(100):
        current = torch.rand((1, 7), generator=generator, dtype=torch.float32) - 0.5
        raw_delta = (
            torch.rand((1, 7), generator=generator, dtype=torch.float32) - 0.5
        ) * 0.1
        nominal = current + torch.clamp(
            raw_delta,
            min=-maximum,
            max=maximum,
        )
        for index in range(ARM_RELEASE_RAMP_SUBSTEPS):
            result = apply_arm_release_ramp_target(
                current,
                nominal,
                maximum,
                ramp_index=index,
            )
            if index == 0:
                expected = current
            elif index == ARM_RELEASE_RAMP_SUBSTEPS - 1:
                expected = nominal
            else:
                fraction = torch.tensor(index / 15, dtype=torch.float32)
                bound = maximum * fraction
                expected = current + torch.clamp(
                    nominal - current,
                    min=-bound,
                    max=bound,
                )
            assert torch.equal(result.target, expected)
            assert result.target.dtype == torch.float32
            assert torch.isfinite(result.target).all()


def test_release_ramp_natural_completion_starts_on_following_apply():
    transition = advance_arm_release_ramp(
        enabled=True,
        phase_before_apply=ARM_RELEASE_PHASE_HOLD,
        next_ramp_index_before_apply=None,
        interlock_remaining_before_apply=1,
        interlock_active_this_apply=True,
        interlock_remaining_after_apply=0,
        interlock_activation_count_delta=0,
    )
    assert transition.phase_after_successful_apply == ARM_RELEASE_PHASE_RAMP
    assert transition.ramp_index_to_apply is None
    assert transition.next_ramp_index_after_successful_apply == 0
    assert transition.release_observed_delta == 1
    assert transition.ramp_started_delta == 1

    indices = []
    phase = transition.phase_after_successful_apply
    next_index = transition.next_ramp_index_after_successful_apply
    for _ in range(ARM_RELEASE_RAMP_SUBSTEPS):
        transition = advance_arm_release_ramp(
            enabled=True,
            phase_before_apply=phase,
            next_ramp_index_before_apply=next_index,
            interlock_remaining_before_apply=0,
            interlock_active_this_apply=False,
            interlock_remaining_after_apply=0,
            interlock_activation_count_delta=0,
        )
        indices.append(transition.ramp_index_to_apply)
        phase = transition.phase_after_successful_apply
        next_index = transition.next_ramp_index_after_successful_apply
    assert indices == list(range(ARM_RELEASE_RAMP_SUBSTEPS))
    assert phase == ARM_RELEASE_PHASE_RELEASE
    assert next_index is None
    assert transition.ramp_completed_delta == 1


def test_release_ramp_open_cancel_starts_immediately_and_open_is_idempotent():
    opened = advance_arm_release_ramp(
        enabled=True,
        phase_before_apply=ARM_RELEASE_PHASE_HOLD,
        next_ramp_index_before_apply=None,
        interlock_remaining_before_apply=42,
        interlock_active_this_apply=False,
        interlock_remaining_after_apply=0,
        interlock_activation_count_delta=0,
    )
    assert opened.phase_after_successful_apply == ARM_RELEASE_PHASE_RAMP
    assert opened.ramp_index_to_apply == 0
    assert opened.next_ramp_index_after_successful_apply == 1
    assert opened.release_observed_delta == 1

    repeated_open = advance_arm_release_ramp(
        enabled=True,
        phase_before_apply=opened.phase_after_successful_apply,
        next_ramp_index_before_apply=opened.next_ramp_index_after_successful_apply,
        interlock_remaining_before_apply=0,
        interlock_active_this_apply=False,
        interlock_remaining_after_apply=0,
        interlock_activation_count_delta=0,
    )
    assert repeated_open.ramp_index_to_apply == 1
    assert repeated_open.release_observed_delta == 0
    assert repeated_open.ramp_started_delta == 0


def test_release_ramp_close_reactivation_atomically_returns_to_hold():
    transition = advance_arm_release_ramp(
        enabled=True,
        phase_before_apply=ARM_RELEASE_PHASE_RAMP,
        next_ramp_index_before_apply=7,
        interlock_remaining_before_apply=0,
        interlock_active_this_apply=True,
        interlock_remaining_after_apply=85,
        interlock_activation_count_delta=1,
    )
    assert transition.phase_after_successful_apply == ARM_RELEASE_PHASE_HOLD
    assert transition.ramp_index_to_apply is None
    assert transition.next_ramp_index_after_successful_apply is None
    assert transition.ramp_cancelled_by_reactivation_delta == 1


def test_profile_bound_86_hold_then_exact_16_release_targets():
    phase = ARM_RELEASE_PHASE_RELEASE
    next_index = None
    remaining = 0
    endpoint_observed = False
    observed_changes = 0
    for apply_index in range(86):
        interlock = _advance_gripper_close_arm_interlock(
            enabled=True,
            previous_endpoint_change_count=observed_changes,
            current_endpoint_change_count=0,
            endpoint_observed_before_apply=endpoint_observed,
            endpoint_is_closed=True,
            remaining_before_apply=remaining,
            configured_substeps=86,
        )
        ramp = advance_arm_release_ramp(
            enabled=True,
            phase_before_apply=phase,
            next_ramp_index_before_apply=next_index,
            interlock_remaining_before_apply=remaining,
            interlock_active_this_apply=interlock.active,
            interlock_remaining_after_apply=(
                interlock.remaining_after_successful_apply
            ),
            interlock_activation_count_delta=interlock.activation_count_delta,
        )
        assert ramp.ramp_index_to_apply is None
        remaining = interlock.remaining_after_successful_apply
        endpoint_observed = interlock.endpoint_observed_after_successful_apply
        observed_changes = interlock.observed_endpoint_change_count
        phase = ramp.phase_after_successful_apply
        next_index = ramp.next_ramp_index_after_successful_apply
        if apply_index < 85:
            assert phase == ARM_RELEASE_PHASE_HOLD
        else:
            assert phase == ARM_RELEASE_PHASE_RAMP
            assert next_index == 0

    applied_indices = []
    for _ in range(ARM_RELEASE_RAMP_SUBSTEPS):
        interlock = _advance_gripper_close_arm_interlock(
            enabled=True,
            previous_endpoint_change_count=observed_changes,
            current_endpoint_change_count=0,
            endpoint_observed_before_apply=True,
            endpoint_is_closed=True,
            remaining_before_apply=remaining,
            configured_substeps=86,
        )
        ramp = advance_arm_release_ramp(
            enabled=True,
            phase_before_apply=phase,
            next_ramp_index_before_apply=next_index,
            interlock_remaining_before_apply=remaining,
            interlock_active_this_apply=interlock.active,
            interlock_remaining_after_apply=(
                interlock.remaining_after_successful_apply
            ),
            interlock_activation_count_delta=interlock.activation_count_delta,
        )
        applied_indices.append(ramp.ramp_index_to_apply)
        phase = ramp.phase_after_successful_apply
        next_index = ramp.next_ramp_index_after_successful_apply
    assert applied_indices == list(range(16))
    assert phase == ARM_RELEASE_PHASE_RELEASE
    assert next_index is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"phase_before_apply": "unknown"},
        {
            "phase_before_apply": ARM_RELEASE_PHASE_RAMP,
            "next_ramp_index_before_apply": None,
        },
        {
            "phase_before_apply": ARM_RELEASE_PHASE_RELEASE,
            "next_ramp_index_before_apply": 0,
        },
        {
            "phase_before_apply": ARM_RELEASE_PHASE_HOLD,
            "interlock_remaining_before_apply": 0,
        },
        {"interlock_activation_count_delta": 2},
    ],
)
def test_release_ramp_rejects_state_machine_drift(kwargs):
    arguments = {
        "enabled": True,
        "phase_before_apply": ARM_RELEASE_PHASE_RELEASE,
        "next_ramp_index_before_apply": None,
        "interlock_remaining_before_apply": 0,
        "interlock_active_this_apply": False,
        "interlock_remaining_after_apply": 0,
        "interlock_activation_count_delta": 0,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match="release-ramp"):
        advance_arm_release_ramp(**arguments)


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
    assert source.count("enable_arm_release_ramp: bool = False") == 1
    assert source.count("enable_current_joint_velocity_recovery: bool = False") == 1
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
    readback_index = source.index("live_position_target =", setter_index)
    trace_index = source.index("self._stage_failure_substep_trace", readback_index)
    anchor_commit_index = source.index(
        "self._gripper_close_arm_interlock_anchor = staged.anchor", setter_index
    )
    commit_index = source.index(
        "self._gripper_close_arm_interlock_remaining = staged.remaining",
        setter_index,
    )
    assert (
        velocity_setter_index
        < setter_index
        < readback_index
        < trace_index
        < anchor_commit_index
        < commit_index
    )
    assert "_StagedGripperCloseArmInterlockState(" in source
    assert "self._set_targets_and_commit_gripper_close_arm_interlock(" in source
    assert "self._reset_gripper_close_arm_interlock_state()" in source
    assert source.count("self._reset_arm_release_ramp_state()") == 2
    assert (
        "if self._arm_release_ramp_enabled and self._arm_target_transaction_failed:"
        in source
    )
    assert '"required before another apply"' in source
    assert "if recovery_owns_target" in source
    assert "else self._asset.data.joint_effort_target[:, self._joint_ids]" in source
    assert "self._gripper_close_arm_interlock_anchor_valid = False" in source
    assert "self._gripper_close_arm_interlock_anchor_refresh_count = 0" in source
    assert "DISABLED_GRIPPER_CLOSE_ARM_INTERLOCK_TRANSITION" in source
