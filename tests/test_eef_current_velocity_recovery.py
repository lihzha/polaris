from __future__ import annotations

import copy
import hashlib
import math
import struct

import pytest

from polaris.eef_controller_profile import (
    validate_current_joint_velocity_recovery_report,
)
from polaris.eef_controller_repair import ARM_RELEASE_RAMP_SUBSTEPS
from polaris.eef_controller_repair import advance_current_joint_velocity_recovery
from polaris.eef_controller_repair import suspend_arm_release_ramp
from polaris.eef_controller_repair import suspend_gripper_close_arm_interlock
from polaris.eef_ik_safety import classify_current_joint_velocity_for_recovery
from polaris.eef_ik_safety import current_joint_velocity_recovery_envelope
from polaris.eef_ik_safety import CURRENT_JOINT_VELOCITY_RECOVERY_CLEAN_SAMPLES_REQUIRED
from polaris.eef_ik_safety import (
    CURRENT_JOINT_VELOCITY_RECOVERY_ENVELOPE_FORMULA_PROFILE,
)
from polaris.eef_ik_safety import CURRENT_JOINT_VELOCITY_RECOVERY_HOLD_PROFILE
from polaris.eef_ik_safety import (
    CURRENT_JOINT_VELOCITY_RECOVERY_MAXIMUM_ACTIVE_SUBSTEPS,
)
from polaris.eef_ik_safety import CURRENT_JOINT_VELOCITY_RECOVERY_PHASE_HOLD
from polaris.eef_ik_safety import CURRENT_JOINT_VELOCITY_RECOVERY_PHASE_INACTIVE
from polaris.eef_ik_safety import CURRENT_JOINT_VELOCITY_RECOVERY_PHASE_RELEASE_RAMP
from polaris.eef_ik_safety import (
    CURRENT_JOINT_VELOCITY_RECOVERY_PREDICTED_POSITION_PROFILE,
)
from polaris.eef_ik_safety import CURRENT_JOINT_VELOCITY_RECOVERY_PROFILE
from polaris.eef_ik_safety import (
    CURRENT_JOINT_VELOCITY_RECOVERY_RELATIVE_ENVELOPE_FLOAT32,
)
from polaris.eef_ik_safety import CURRENT_JOINT_VELOCITY_RECOVERY_SCHEMA_VERSION
from polaris.eef_ik_safety import CURRENT_JOINT_VELOCITY_RECOVERY_TRANSACTION_PROFILE
from polaris.eef_ik_safety import PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S
from polaris.eef_ik_safety import PANDA_EEF_PHYSICS_DT_FLOAT32
from polaris.eef_ik_safety import PANDA_PHYSX_HARD_JOINT_POS_LIMITS_FLOAT32_SHA256
from polaris.eef_ik_safety import PANDA_PHYSX_HARD_JOINT_POS_LIMITS_RAD
from polaris.eef_ik_safety import PHYSX_HARD_LIMIT_PROFILE
from polaris.eef_ik_safety import predict_joint_position_against_hard_limits


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _next_float32(value: float, *, positive: bool = True) -> float:
    bits = struct.unpack("<I", struct.pack("<f", _float32(value)))[0]
    if value >= 0.0:
        bits += 1 if positive else -1
    else:
        bits += -1 if positive else 1
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _neutral_hard_position() -> list[float]:
    return [0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.0]


@pytest.mark.parametrize("limit", [2.175, 2.61])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_exact_float32_velocity_envelope_is_dls_eligible_and_nextafter_recovers(
    limit: float,
    sign: float,
) -> None:
    envelope = current_joint_velocity_recovery_envelope(limit)
    expected = _float32(
        _float32(limit)
        + _float32(
            _float32(limit) * CURRENT_JOINT_VELOCITY_RECOVERY_RELATIVE_ENVELOPE_FLOAT32
        )
    )
    assert envelope == expected
    exact = classify_current_joint_velocity_for_recovery(sign * envelope, limit)
    assert exact.residual is True
    assert exact.recovery_required is False
    crossed = classify_current_joint_velocity_for_recovery(
        sign * _next_float32(envelope),
        limit,
    )
    assert crossed.residual is True
    assert crossed.recovery_required is True


def test_observed_canary_velocity_fixtures_have_expected_v5_classification() -> None:
    official = classify_current_joint_velocity_for_recovery(2.6102163791656494, 2.61)
    assert official.residual is True
    assert official.recovery_required is False
    reasoning = classify_current_joint_velocity_for_recovery(-11.743, 2.61)
    assert reasoning.residual is True
    assert reasoning.recovery_required is True


@pytest.mark.parametrize(
    ("position", "velocity", "lower", "upper", "inside", "outside"),
    [
        (0.9, 12.0, -1.0, 1.0, 1.0, _next_float32(1.0)),
        (-0.9, -12.0, -1.0, 1.0, -1.0, _next_float32(-1.0, positive=False)),
    ],
)
def test_predicted_hard_limit_exact_boundary_allowed_nextafter_aborts(
    position: float,
    velocity: float,
    lower: float,
    upper: float,
    inside: float,
    outside: float,
) -> None:
    dt_inside = _float32((_float32(inside) - _float32(position)) / velocity)
    # Use zero velocity at the already-materialized boundary to isolate the
    # inclusive hard-envelope comparison from division rounding.
    allowed = predict_joint_position_against_hard_limits(
        inside, 0.0, 1.0 / 120.0, lower, upper
    )
    assert allowed.within_hard_limits is True
    assert allowed.predicted_joint_pos_rad == _float32(inside)
    crossed = predict_joint_position_against_hard_limits(
        outside, 0.0, 1.0 / 120.0, lower, upper
    )
    assert crossed.within_hard_limits is False
    assert dt_inside > 0.0


def test_below_recovery_envelope_still_has_independent_predicted_crossing_guard() -> (
    None
):
    limit = 2.61
    velocity = _float32(2.0)
    classification = classify_current_joint_velocity_for_recovery(velocity, limit)
    assert classification.recovery_required is False
    dt = _float32(1.0 / 120.0)
    upper = _float32(2.0)
    position = _float32(upper - _float32(velocity * dt) / 2.0)
    prediction = predict_joint_position_against_hard_limits(
        position,
        velocity,
        dt,
        -2.0,
        upper,
    )
    assert prediction.within_hard_limits is False
    assert prediction.predicted_joint_pos_rad > upper


def _transition(
    phase: str,
    active: int,
    clean: int,
    index: int | None,
    *,
    over: bool,
):
    return advance_current_joint_velocity_recovery(
        enabled=True,
        phase_before_apply=phase,
        consecutive_active_substeps_before_apply=active,
        consecutive_clean_samples_before_apply=clean,
        next_release_ramp_index_before_apply=index,
        measured_velocity_over_envelope=over,
    )


def test_recovery_state_machine_bounds_clean2_ramp_and_reexceed_lifecycle() -> None:
    transition = _transition(
        CURRENT_JOINT_VELOCITY_RECOVERY_PHASE_INACTIVE,
        0,
        0,
        None,
        over=True,
    )
    assert transition.recovery_event_delta == 1
    assert transition.skip_dls is True
    assert transition.hold_current_position is True
    phase, active, clean, index = (
        transition.phase_after_successful_apply,
        transition.consecutive_active_substeps_after_successful_apply,
        transition.consecutive_clean_samples_after_successful_apply,
        transition.next_release_ramp_index_after_successful_apply,
    )
    for expected_active in range(
        2, CURRENT_JOINT_VELOCITY_RECOVERY_MAXIMUM_ACTIVE_SUBSTEPS + 1
    ):
        transition = _transition(phase, active, clean, index, over=True)
        assert transition.sustained_abort is False
        assert transition.recovery_event_delta == 0
        phase, active, clean, index = (
            transition.phase_after_successful_apply,
            transition.consecutive_active_substeps_after_successful_apply,
            transition.consecutive_clean_samples_after_successful_apply,
            transition.next_release_ramp_index_after_successful_apply,
        )
        assert active == expected_active
    ninth = _transition(phase, active, clean, index, over=True)
    assert ninth.sustained_abort is True

    first_clean = _transition(
        CURRENT_JOINT_VELOCITY_RECOVERY_PHASE_HOLD, 1, 0, None, over=False
    )
    assert first_clean.hold_current_position is True
    second_clean = _transition(
        first_clean.phase_after_successful_apply,
        first_clean.consecutive_active_substeps_after_successful_apply,
        first_clean.consecutive_clean_samples_after_successful_apply,
        first_clean.next_release_ramp_index_after_successful_apply,
        over=False,
    )
    assert second_clean.phase_after_successful_apply == (
        CURRENT_JOINT_VELOCITY_RECOVERY_PHASE_RELEASE_RAMP
    )
    assert second_clean.release_ramp_index_to_apply == 0
    assert second_clean.skip_dls is True
    assert second_clean.recovered_event_delta == 0
    assert second_clean.active_substep_delta == 1
    assert (
        second_clean.hold_current_position
        or second_clean.release_ramp_index_to_apply == 0
    )

    reexceeded = _transition(
        second_clean.phase_after_successful_apply,
        second_clean.consecutive_active_substeps_after_successful_apply,
        second_clean.consecutive_clean_samples_after_successful_apply,
        second_clean.next_release_ramp_index_after_successful_apply,
        over=True,
    )
    assert reexceeded.phase_after_successful_apply == (
        CURRENT_JOINT_VELOCITY_RECOVERY_PHASE_HOLD
    )
    assert reexceeded.recovery_event_delta == 0

    phase = CURRENT_JOINT_VELOCITY_RECOVERY_PHASE_RELEASE_RAMP
    index = 1
    completed = 0
    applied = [0]
    while phase == CURRENT_JOINT_VELOCITY_RECOVERY_PHASE_RELEASE_RAMP:
        ramp = _transition(phase, 0, 0, index, over=False)
        applied.append(ramp.release_ramp_index_to_apply)
        completed += ramp.recovered_event_delta
        phase = ramp.phase_after_successful_apply
        index = ramp.next_release_ramp_index_after_successful_apply
    assert applied == list(range(ARM_RELEASE_RAMP_SUBSTEPS))
    assert completed == 1


@pytest.mark.parametrize("active", [0, 9])
def test_recovery_state_machine_rejects_uncommitted_hold_counter(active: int) -> None:
    with pytest.raises(ValueError, match="active counter drift"):
        _transition(
            CURRENT_JOINT_VELOCITY_RECOVERY_PHASE_HOLD,
            active,
            0,
            None,
            over=True,
        )


def test_recovery_suspends_lower_interlock_and_release_ramp_without_counting() -> None:
    interlock = suspend_gripper_close_arm_interlock(
        remaining_before_apply=42,
        observed_endpoint_change_count=3,
        endpoint_observed_before_apply=True,
    )
    assert interlock.active is False
    assert interlock.remaining_after_successful_apply == 42
    assert interlock.observed_endpoint_change_count == 3
    assert interlock.endpoint_observed_after_successful_apply is True
    assert interlock.activation_count_delta == 0
    assert interlock.completion_count_delta == 0
    assert interlock.open_cancel_count_delta == 0
    ramp = suspend_arm_release_ramp(
        phase_before_apply="ramp",
        next_ramp_index_before_apply=7,
    )
    assert ramp.phase_after_successful_apply == "ramp"
    assert ramp.next_ramp_index_after_successful_apply == 7
    assert ramp.ramp_index_to_apply is None
    assert ramp.release_observed_delta == 0
    assert ramp.ramp_started_delta == 0
    assert ramp.ramp_completed_delta == 0
    assert ramp.ramp_cancelled_by_reactivation_delta == 0


def _recovery_snapshot(
    *,
    apply_index: int,
    joint_position: list[float],
    joint_velocity: list[float],
    committed: bool,
) -> dict:
    limits = [_float32(value) for value in PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S]
    envelopes = [current_joint_velocity_recovery_envelope(value) for value in limits]
    position = [_float32(value) for value in joint_position]
    velocity = [_float32(value) for value in joint_velocity]
    excess = [
        max(_float32(abs(value) - limit), 0.0)
        for value, limit in zip(velocity, limits, strict=True)
    ]
    ratio = [
        _float32(abs(value) / limit)
        for value, limit in zip(velocity, limits, strict=True)
    ]
    predicted = [
        _float32(q + _float32(dq * PANDA_EEF_PHYSICS_DT_FLOAT32))
        for q, dq in zip(position, velocity, strict=True)
    ]
    clearance = [
        min(_float32(q - lower), _float32(upper - q))
        for q, (lower, upper) in zip(
            predicted,
            PANDA_PHYSX_HARD_JOINT_POS_LIMITS_RAD,
            strict=True,
        )
    ]
    return {
        "apply_index": apply_index,
        "policy_step": apply_index // 8,
        "physics_substep": apply_index % 8,
        "joint_pos_rad": position,
        "joint_velocity_rad_s": velocity,
        "joint_velocity_limit_rad_s": limits,
        "joint_velocity_envelope_rad_s": envelopes,
        "joint_velocity_limit_excess_rad_s": excess,
        "velocity_to_limit_ratio": ratio,
        "predicted_joint_pos_rad": predicted,
        "predicted_hard_limit_clearance_rad": clearance,
        "hold_target_rad": position,
        "hold_position_target_readback_rad": (position if committed else None),
        "hold_velocity_target_readback_rad_s": ([0.0] * 7 if committed else None),
        "hold_effort_target_readback_nm": ([0.0] * 7 if committed else None),
    }


def _active_recovery_report() -> dict:
    limits = [_float32(value) for value in PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S]
    envelopes = [current_joint_velocity_recovery_envelope(value) for value in limits]
    velocity = [_float32(11.743), *([0.0] * 6)]
    excess = [_float32(abs(velocity[0]) - limits[0]), *([0.0] * 6)]
    ratio = [_float32(abs(velocity[0]) / limits[0]), *([0.0] * 6)]
    hard_limits = [list(row) for row in PANDA_PHYSX_HARD_JOINT_POS_LIMITS_RAD]
    hard_digest = hashlib.sha256(
        b"".join(struct.pack("<f", value) for row in hard_limits for value in row)
    ).hexdigest()
    snapshot = _recovery_snapshot(
        apply_index=0,
        joint_position=_neutral_hard_position(),
        joint_velocity=velocity,
        committed=True,
    )
    return {
        "contract": {
            "schema_version": CURRENT_JOINT_VELOCITY_RECOVERY_SCHEMA_VERSION,
            "profile": CURRENT_JOINT_VELOCITY_RECOVERY_PROFILE,
            "envelope_formula_profile": (
                CURRENT_JOINT_VELOCITY_RECOVERY_ENVELOPE_FORMULA_PROFILE
            ),
            "relative_envelope_float32": (
                CURRENT_JOINT_VELOCITY_RECOVERY_RELATIVE_ENVELOPE_FLOAT32
            ),
            "maximum_active_substeps": (
                CURRENT_JOINT_VELOCITY_RECOVERY_MAXIMUM_ACTIVE_SUBSTEPS
            ),
            "clean_samples_required": (
                CURRENT_JOINT_VELOCITY_RECOVERY_CLEAN_SAMPLES_REQUIRED
            ),
            "hold_profile": CURRENT_JOINT_VELOCITY_RECOVERY_HOLD_PROFILE,
            "predicted_position_profile": (
                CURRENT_JOINT_VELOCITY_RECOVERY_PREDICTED_POSITION_PROFILE
            ),
            "hard_limit_profile": PHYSX_HARD_LIMIT_PROFILE,
            "release_ramp_profile": "arm_post_interlock_linear_slew_cap_release_ramp16_v3",
            "transaction_profile": CURRENT_JOINT_VELOCITY_RECOVERY_TRANSACTION_PROFILE,
            "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
            "velocity_limits_rad_s": limits,
            "velocity_envelopes_rad_s": envelopes,
            "physics_dt_float32": PANDA_EEF_PHYSICS_DT_FLOAT32,
            "hard_joint_position_limits_rad": hard_limits,
            "hard_joint_position_limits_little_endian_float32_sha256": hard_digest,
        },
        "state": {
            "phase": CURRENT_JOINT_VELOCITY_RECOVERY_PHASE_HOLD,
            "active": True,
            "consecutive_active_substeps": 1,
            "consecutive_clean_samples": 0,
            "release_ramp_next_index": None,
        },
        "counters": {
            "residual_events": 1,
            "residual_joints": 1,
            "recovery_events": 1,
            "recovery_active_substeps": 1,
            "recovered_events": 0,
            "hold_target_applies": 1,
            "release_ramp_target_applies": 0,
            "sustained_aborts": 0,
            "current_hard_limit_aborts": 0,
            "predicted_limit_aborts": 0,
            "transaction_aborts": 0,
            "lower_endpoint_transition_aborts": 0,
        },
        "maxima": {
            "abs_velocity_to_limit_ratio": ratio[0],
            "consecutive_recovery_substeps": 1,
            "abs_velocity_residual_excess_rad_s": excess,
        },
        "events": [
            {
                "event_index": 0,
                "start_apply_index": 0,
                "end_apply_index": None,
                "start_reason": "measured_velocity_above_float32_envelope",
                "end_reason": None,
                "deferred_lower_endpoint_transition_count": None,
                "lower_endpoint_transition_overflow_context": None,
                "recovery_completed_apply_index": None,
                "start": copy.deepcopy(snapshot),
                "last": copy.deepcopy(snapshot),
            }
        ],
    }


def _clean_completed_recovery_report() -> dict:
    report = _active_recovery_report()
    report["state"] = {
        "phase": CURRENT_JOINT_VELOCITY_RECOVERY_PHASE_INACTIVE,
        "active": False,
        "consecutive_active_substeps": 0,
        "consecutive_clean_samples": 0,
        "release_ramp_next_index": None,
    }
    report["counters"].update(
        {
            "recovery_active_substeps": 3,
            "recovered_events": 1,
            "hold_target_applies": 3,
            "release_ramp_target_applies": ARM_RELEASE_RAMP_SUBSTEPS,
        }
    )
    report["maxima"]["consecutive_recovery_substeps"] = 3
    terminal = _recovery_snapshot(
        apply_index=17,
        joint_position=_neutral_hard_position(),
        joint_velocity=[0.0] * 7,
        committed=True,
    )
    report["events"][0].update(
        {
            "end_apply_index": 17,
            "end_reason": "clean2_release_ramp_complete",
            "recovery_completed_apply_index": 17,
            "last": terminal,
        }
    )
    return report


def _sustained_terminal_recovery_report() -> dict:
    report = _active_recovery_report()
    report["state"] = {
        "phase": CURRENT_JOINT_VELOCITY_RECOVERY_PHASE_INACTIVE,
        "active": False,
        "consecutive_active_substeps": 0,
        "consecutive_clean_samples": 0,
        "release_ramp_next_index": None,
    }
    report["counters"].update(
        {
            "residual_events": 9,
            "residual_joints": 9,
            "recovery_active_substeps": 8,
            "hold_target_applies": 8,
            "sustained_aborts": 1,
        }
    )
    report["maxima"]["consecutive_recovery_substeps"] = 8
    terminal = _recovery_snapshot(
        apply_index=8,
        joint_position=_neutral_hard_position(),
        joint_velocity=[_float32(11.743), *([0.0] * 6)],
        committed=False,
    )
    terminal["hold_target_rad"] = None
    report["events"][0].update(
        {
            "end_apply_index": 8,
            "end_reason": "sustained_recovery_abort",
            "last": terminal,
        }
    )
    return report


def _post_recovery_lower_endpoint_terminal_report() -> dict:
    report = _clean_completed_recovery_report()
    terminal = _recovery_snapshot(
        apply_index=18,
        joint_position=_neutral_hard_position(),
        joint_velocity=[0.0] * 7,
        committed=False,
    )
    terminal["hold_target_rad"] = None
    report["counters"]["lower_endpoint_transition_aborts"] = 1
    report["events"][0].update(
        {
            "end_apply_index": 18,
            "end_reason": "lower_endpoint_transition_overflow_abort",
            "deferred_lower_endpoint_transition_count": 2,
            "lower_endpoint_transition_overflow_context": ("post_recovery_resume"),
            "last": terminal,
        }
    )
    return report


def _immediate_current_hard_terminal_report() -> dict:
    report = _active_recovery_report()
    position = _neutral_hard_position()
    position[0] = _next_float32(PANDA_PHYSX_HARD_JOINT_POS_LIMITS_RAD[0][1])
    snapshot = _recovery_snapshot(
        apply_index=0,
        joint_position=position,
        joint_velocity=[0.0] * 7,
        committed=False,
    )
    snapshot["hold_target_rad"] = None
    report["state"] = {
        "phase": CURRENT_JOINT_VELOCITY_RECOVERY_PHASE_INACTIVE,
        "active": False,
        "consecutive_active_substeps": 0,
        "consecutive_clean_samples": 0,
        "release_ramp_next_index": None,
    }
    report["counters"] = {field: 0 for field in report["counters"]}
    report["counters"].update({"recovery_events": 1, "current_hard_limit_aborts": 1})
    report["maxima"] = {
        "abs_velocity_to_limit_ratio": 0.0,
        "consecutive_recovery_substeps": 0,
        "abs_velocity_residual_excess_rad_s": [0.0] * 7,
    }
    report["events"] = [
        {
            "event_index": 0,
            "start_apply_index": 0,
            "end_apply_index": 0,
            "start_reason": "current_hard_limit_violation",
            "end_reason": "current_hard_limit_abort",
            "deferred_lower_endpoint_transition_count": None,
            "lower_endpoint_transition_overflow_context": None,
            "recovery_completed_apply_index": None,
            "start": copy.deepcopy(snapshot),
            "last": copy.deepcopy(snapshot),
        }
    ]
    return report


def _immediate_predicted_terminal_report() -> dict:
    report = _active_recovery_report()
    velocity = [_float32(2.0), *([0.0] * 6)]
    delta = _float32(velocity[0] * PANDA_EEF_PHYSICS_DT_FLOAT32)
    position = _neutral_hard_position()
    position[0] = _float32(
        PANDA_PHYSX_HARD_JOINT_POS_LIMITS_RAD[0][1] - _float32(delta / 2.0)
    )
    snapshot = _recovery_snapshot(
        apply_index=0,
        joint_position=position,
        joint_velocity=velocity,
        committed=False,
    )
    snapshot["hold_target_rad"] = None
    report["state"] = {
        "phase": CURRENT_JOINT_VELOCITY_RECOVERY_PHASE_INACTIVE,
        "active": False,
        "consecutive_active_substeps": 0,
        "consecutive_clean_samples": 0,
        "release_ramp_next_index": None,
    }
    report["counters"] = {field: 0 for field in report["counters"]}
    report["counters"].update({"recovery_events": 1, "predicted_limit_aborts": 1})
    report["maxima"] = {
        "abs_velocity_to_limit_ratio": max(snapshot["velocity_to_limit_ratio"]),
        "consecutive_recovery_substeps": 0,
        "abs_velocity_residual_excess_rad_s": [0.0] * 7,
    }
    report["events"] = [
        {
            "event_index": 0,
            "start_apply_index": 0,
            "end_apply_index": 0,
            "start_reason": "predicted_hard_limit_crossing",
            "end_reason": "predicted_hard_limit_abort",
            "deferred_lower_endpoint_transition_count": None,
            "lower_endpoint_transition_overflow_context": None,
            "recovery_completed_apply_index": None,
            "start": copy.deepcopy(snapshot),
            "last": copy.deepcopy(snapshot),
        }
    ]
    return report


def _move_terminal_snapshot_to_apply(report: dict, apply_index: int) -> None:
    event = report["events"][0]
    snapshot = copy.deepcopy(event["last"])
    snapshot["apply_index"] = apply_index
    snapshot["policy_step"] = apply_index // 8
    snapshot["physics_substep"] = apply_index % 8
    event["end_apply_index"] = apply_index
    event["last"] = snapshot


def _measured_later_terminal_report(end_reason: str) -> dict:
    report = _active_recovery_report()
    report["state"] = {
        "phase": CURRENT_JOINT_VELOCITY_RECOVERY_PHASE_INACTIVE,
        "active": False,
        "consecutive_active_substeps": 0,
        "consecutive_clean_samples": 0,
        "release_ramp_next_index": None,
    }
    terminal_position = _neutral_hard_position()
    terminal_velocity = [0.0] * 7
    if end_reason == "current_hard_limit_abort":
        terminal_position[0] = _next_float32(
            PANDA_PHYSX_HARD_JOINT_POS_LIMITS_RAD[0][1]
        )
    elif end_reason == "predicted_hard_limit_abort":
        terminal_velocity[0] = _float32(2.0)
        delta = _float32(terminal_velocity[0] * PANDA_EEF_PHYSICS_DT_FLOAT32)
        terminal_position[0] = _float32(
            PANDA_PHYSX_HARD_JOINT_POS_LIMITS_RAD[0][1] - _float32(delta / 2.0)
        )
    terminal = _recovery_snapshot(
        apply_index=1,
        joint_position=terminal_position,
        joint_velocity=terminal_velocity,
        committed=False,
    )
    if end_reason != "transaction_abort":
        terminal["hold_target_rad"] = None
    counter_field = {
        "current_hard_limit_abort": "current_hard_limit_aborts",
        "predicted_hard_limit_abort": "predicted_limit_aborts",
        "transaction_abort": "transaction_aborts",
        "lower_endpoint_transition_overflow_abort": (
            "lower_endpoint_transition_aborts"
        ),
    }[end_reason]
    report["counters"][counter_field] = 1
    report["events"][0].update(
        {
            "end_apply_index": 1,
            "end_reason": end_reason,
            "last": terminal,
        }
    )
    if end_reason == "lower_endpoint_transition_overflow_abort":
        report["events"][0].update(
            {
                "deferred_lower_endpoint_transition_count": 2,
                "lower_endpoint_transition_overflow_context": "active_recovery",
            }
        )
    return report


def _hard_limit_collision_report(*, kind: str, post_recovery: bool) -> dict:
    end_reason = {
        "current": "current_hard_limit_abort",
        "predicted": "predicted_hard_limit_abort",
    }[kind]
    if not post_recovery:
        report = _measured_later_terminal_report(end_reason)
        report["events"][0].update(
            {
                "deferred_lower_endpoint_transition_count": 2,
                "lower_endpoint_transition_overflow_context": "active_recovery",
            }
        )
        return report

    report = _clean_completed_recovery_report()
    owner = (
        _immediate_current_hard_terminal_report()
        if kind == "current"
        else _immediate_predicted_terminal_report()
    )["events"][0]
    start = copy.deepcopy(owner["start"])
    start.update({"apply_index": 18, "policy_step": 2, "physics_substep": 2})
    terminal_event = {
        "event_index": 1,
        "start_apply_index": 18,
        "end_apply_index": 18,
        "start_reason": owner["start_reason"],
        "end_reason": end_reason,
        "deferred_lower_endpoint_transition_count": 2,
        "lower_endpoint_transition_overflow_context": "post_recovery_resume",
        "recovery_completed_apply_index": 17,
        "start": copy.deepcopy(start),
        "last": copy.deepcopy(start),
    }
    report["events"].append(terminal_event)
    report["counters"]["recovery_events"] = 2
    report["counters"][
        "current_hard_limit_aborts" if kind == "current" else "predicted_limit_aborts"
    ] = 1
    return report


def test_recovery_report_schema_and_open_event_are_closed() -> None:
    report = _active_recovery_report()
    assert (
        validate_current_joint_velocity_recovery_report(report, apply_calls=1) == report
    )
    for mutation in (
        lambda value: value["contract"].__setitem__("relative_envelope_float32", 1e-4),
        lambda value: value["state"].__setitem__("active", False),
        lambda value: value["counters"].__setitem__("recovery_events", 2),
        lambda value: value["events"].append(copy.deepcopy(value["events"][0])),
    ):
        drifted = copy.deepcopy(report)
        mutation(drifted)
        with pytest.raises(ValueError):
            validate_current_joint_velocity_recovery_report(drifted, apply_calls=1)


def test_clean_and_sustained_terminal_reports_bind_minimum_chronology() -> None:
    clean = _clean_completed_recovery_report()
    sustained = _sustained_terminal_recovery_report()
    assert (
        validate_current_joint_velocity_recovery_report(clean, apply_calls=18) == clean
    )
    assert (
        validate_current_joint_velocity_recovery_report(sustained, apply_calls=9)
        == sustained
    )

    short_clean = copy.deepcopy(clean)
    short_clean["events"][0]["end_apply_index"] = 1
    short_clean["events"][0]["recovery_completed_apply_index"] = 1
    short_clean["events"][0]["last"] = _recovery_snapshot(
        apply_index=1,
        joint_position=_neutral_hard_position(),
        joint_velocity=[0.0] * 7,
        committed=True,
    )
    with pytest.raises(ValueError, match="start/end pairing drift"):
        validate_current_joint_velocity_recovery_report(short_clean, apply_calls=18)

    short_sustained = copy.deepcopy(sustained)
    short_sustained["events"][0]["end_apply_index"] = 0
    terminal = copy.deepcopy(short_sustained["events"][0]["start"])
    for field in (
        "hold_target_rad",
        "hold_position_target_readback_rad",
        "hold_velocity_target_readback_rad_s",
        "hold_effort_target_readback_nm",
    ):
        terminal[field] = None
    short_sustained["events"][0]["last"] = terminal
    with pytest.raises(ValueError, match="start/end pairing drift"):
        validate_current_joint_velocity_recovery_report(
            short_sustained,
            apply_calls=9,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda report: report["counters"].update(
            {"recovery_active_substeps": 2, "hold_target_applies": 2}
        ),
        lambda report: report["counters"].__setitem__(
            "release_ramp_target_applies", ARM_RELEASE_RAMP_SUBSTEPS - 1
        ),
        lambda report: report["maxima"].__setitem__("consecutive_recovery_substeps", 2),
    ],
)
def test_clean_terminal_rejects_each_minimum_history_mutation(mutation) -> None:
    report = _clean_completed_recovery_report()
    mutation(report)
    with pytest.raises(ValueError, match="counter history|event history"):
        validate_current_joint_velocity_recovery_report(report, apply_calls=18)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda report: report["counters"].update(
            {"recovery_active_substeps": 7, "hold_target_applies": 7}
        ),
        lambda report: report["maxima"].__setitem__("consecutive_recovery_substeps", 7),
    ],
)
def test_sustained_terminal_rejects_each_maximum_history_mutation(mutation) -> None:
    report = _sustained_terminal_recovery_report()
    mutation(report)
    with pytest.raises(ValueError, match="event history"):
        validate_current_joint_velocity_recovery_report(report, apply_calls=9)


def test_post_recovery_lower_overflow_retains_truthful_recovered_evidence() -> None:
    report = _post_recovery_lower_endpoint_terminal_report()
    validated = validate_current_joint_velocity_recovery_report(
        report,
        apply_calls=19,
    )
    event = validated["events"][0]
    assert validated["counters"]["recovered_events"] == 1
    assert event["recovery_completed_apply_index"] == 17
    assert event["lower_endpoint_transition_overflow_context"] == (
        "post_recovery_resume"
    )
    assert event["deferred_lower_endpoint_transition_count"] == 2

    for mutation in (
        lambda value: value["events"][0].__setitem__(
            "deferred_lower_endpoint_transition_count", 1
        ),
        lambda value: value["events"][0].__setitem__(
            "lower_endpoint_transition_overflow_context", "active_recovery"
        ),
        lambda value: value["events"][0].__setitem__(
            "recovery_completed_apply_index", 16
        ),
        lambda value: value["counters"].__setitem__("recovered_events", 0),
    ):
        drifted = copy.deepcopy(report)
        mutation(drifted)
        with pytest.raises(ValueError):
            validate_current_joint_velocity_recovery_report(
                drifted,
                apply_calls=19,
            )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value["contract"].__setitem__("physics_dt_float32", 0.01),
            "identity drift",
        ),
        (
            lambda value: value["contract"]["hard_joint_position_limits_rad"][
                0
            ].__setitem__(
                0,
                _next_float32(
                    value["contract"]["hard_joint_position_limits_rad"][0][0]
                ),
            ),
            "binding drift",
        ),
        (
            lambda value: value["contract"].__setitem__(
                "hard_joint_position_limits_little_endian_float32_sha256",
                "0" * 64,
            ),
            "binding drift",
        ),
        (
            lambda value: value["events"][0]["last"]["joint_pos_rad"].__setitem__(
                0,
                _float32(value["events"][0]["last"]["joint_pos_rad"][0] + 0.125),
            ),
            "numeric binding drift",
        ),
        (
            lambda value: value["events"][0]["start"]["joint_pos_rad"].__setitem__(
                0,
                _float32(value["events"][0]["start"]["joint_pos_rad"][0] + 0.125),
            ),
            "numeric binding drift",
        ),
        (
            lambda value: value["events"][0]["last"][
                "joint_velocity_rad_s"
            ].__setitem__(
                0, _next_float32(value["events"][0]["last"]["joint_velocity_rad_s"][0])
            ),
            "numeric binding drift",
        ),
        (
            lambda value: value["events"][0]["start"][
                "joint_velocity_rad_s"
            ].__setitem__(
                0,
                _next_float32(value["events"][0]["start"]["joint_velocity_rad_s"][0]),
            ),
            "numeric binding drift",
        ),
        (
            lambda value: value["events"][0]["last"][
                "predicted_joint_pos_rad"
            ].__setitem__(
                0,
                _next_float32(value["events"][0]["last"]["predicted_joint_pos_rad"][0]),
            ),
            "numeric binding drift",
        ),
        (
            lambda value: value["events"][0]["start"][
                "predicted_joint_pos_rad"
            ].__setitem__(
                0,
                _next_float32(
                    value["events"][0]["start"]["predicted_joint_pos_rad"][0]
                ),
            ),
            "numeric binding drift",
        ),
        (
            lambda value: value["events"][0]["last"][
                "predicted_hard_limit_clearance_rad"
            ].__setitem__(
                0,
                _next_float32(
                    value["events"][0]["last"]["predicted_hard_limit_clearance_rad"][0]
                ),
            ),
            "numeric binding drift",
        ),
        (
            lambda value: value["events"][0]["start"][
                "predicted_hard_limit_clearance_rad"
            ].__setitem__(
                0,
                _next_float32(
                    value["events"][0]["start"]["predicted_hard_limit_clearance_rad"][0]
                ),
            ),
            "numeric binding drift",
        ),
        (
            lambda value: value["events"][0].__setitem__(
                "start_reason", "predicted_hard_limit_crossing"
            ),
            "start/end pairing drift",
        ),
    ],
)
def test_recovery_report_rejects_numeric_and_reason_mutations(
    mutation,
    match: str,
) -> None:
    report = _active_recovery_report()
    assert (
        report["contract"]["hard_joint_position_limits_little_endian_float32_sha256"]
        == PANDA_PHYSX_HARD_JOINT_POS_LIMITS_FLOAT32_SHA256
    )
    mutation(report)
    with pytest.raises(ValueError, match=match):
        validate_current_joint_velocity_recovery_report(report, apply_calls=1)


def test_transaction_abort_snapshot_allows_target_without_readbacks() -> None:
    report = _active_recovery_report()
    report["state"] = {
        "phase": CURRENT_JOINT_VELOCITY_RECOVERY_PHASE_INACTIVE,
        "active": False,
        "consecutive_active_substeps": 0,
        "consecutive_clean_samples": 0,
        "release_ramp_next_index": None,
    }
    report["counters"].update({"recovery_active_substeps": 0, "hold_target_applies": 0})
    report["counters"]["transaction_aborts"] = 1
    report["maxima"]["consecutive_recovery_substeps"] = 0
    event = report["events"][0]
    event["end_apply_index"] = 0
    event["start_reason"] = "target_transaction_failure"
    event["end_reason"] = "transaction_abort"
    for snapshot_field in ("start", "last"):
        snapshot = event[snapshot_field]
        snapshot["hold_position_target_readback_rad"] = None
        snapshot["hold_velocity_target_readback_rad_s"] = None
        snapshot["hold_effort_target_readback_nm"] = None
    assert (
        validate_current_joint_velocity_recovery_report(report, apply_calls=1) == report
    )

    drifted = copy.deepcopy(report)
    drifted["events"][0]["last"]["hold_velocity_target_readback_rad_s"] = [0.0] * 7
    with pytest.raises(ValueError, match="transaction snapshot split"):
        validate_current_joint_velocity_recovery_report(drifted, apply_calls=1)


def test_current_hard_limit_abort_has_an_exact_nested_terminal_counter() -> None:
    report = _immediate_current_hard_terminal_report()
    validated = validate_current_joint_velocity_recovery_report(report, apply_calls=1)
    assert validated["counters"]["current_hard_limit_aborts"] == 1

    drifted = copy.deepcopy(report)
    drifted["counters"]["current_hard_limit_aborts"] = 0
    with pytest.raises(ValueError, match="event history drift"):
        validate_current_joint_velocity_recovery_report(drifted, apply_calls=1)


def test_immediate_predicted_abort_has_a_legal_owning_start() -> None:
    report = _immediate_predicted_terminal_report()
    validated = validate_current_joint_velocity_recovery_report(report, apply_calls=1)
    assert validated["events"][0]["start_reason"] == ("predicted_hard_limit_crossing")


@pytest.mark.parametrize(
    "end_reason",
    [
        "current_hard_limit_abort",
        "predicted_hard_limit_abort",
        "transaction_abort",
        "lower_endpoint_transition_overflow_abort",
    ],
)
def test_measured_start_accepts_each_legal_later_terminal(end_reason: str) -> None:
    report = _measured_later_terminal_report(end_reason)
    validated = validate_current_joint_velocity_recovery_report(report, apply_calls=2)
    assert validated["events"][0]["end_reason"] == end_reason


@pytest.mark.parametrize("kind", ["current", "predicted"])
@pytest.mark.parametrize("post_recovery", [False, True])
def test_hard_limit_collision_binds_deferred_endpoint_context(
    kind: str,
    post_recovery: bool,
) -> None:
    report = _hard_limit_collision_report(kind=kind, post_recovery=post_recovery)
    apply_calls = 19 if post_recovery else 2
    validated = validate_current_joint_velocity_recovery_report(
        report,
        apply_calls=apply_calls,
    )
    event = validated["events"][-1]
    assert event["deferred_lower_endpoint_transition_count"] == 2
    assert event["lower_endpoint_transition_overflow_context"] == (
        "post_recovery_resume" if post_recovery else "active_recovery"
    )

    mutations = [
        lambda value: value["events"][-1].__setitem__(
            "deferred_lower_endpoint_transition_count", 1
        ),
        lambda value: value["events"][-1].__setitem__(
            "lower_endpoint_transition_overflow_context",
            "active_recovery" if post_recovery else "post_recovery_resume",
        ),
    ]
    if post_recovery:
        mutations.append(
            lambda value: value["events"][-1].__setitem__(
                "recovery_completed_apply_index", 16
            )
        )
    for mutation in mutations:
        drifted = copy.deepcopy(report)
        mutation(drifted)
        with pytest.raises(ValueError):
            validate_current_joint_velocity_recovery_report(
                drifted,
                apply_calls=apply_calls,
            )


@pytest.mark.parametrize(
    "case",
    [
        "delayed_current_own",
        "delayed_predicted_own",
        "current_to_transaction",
        "current_to_lower",
        "current_to_open",
        "current_to_clean",
    ],
)
def test_recovery_event_rejects_illegal_start_end_pairing_matrix(case: str) -> None:
    report = (
        _immediate_predicted_terminal_report()
        if case == "delayed_predicted_own"
        else _immediate_current_hard_terminal_report()
    )
    apply_calls = 1
    event = report["events"][0]
    if case.startswith("delayed_"):
        _move_terminal_snapshot_to_apply(report, 1)
        apply_calls = 2
    elif case == "current_to_open":
        event["end_apply_index"] = None
        event["end_reason"] = None
    else:
        event["end_reason"] = {
            "current_to_transaction": "transaction_abort",
            "current_to_lower": "lower_endpoint_transition_overflow_abort",
            "current_to_clean": "clean2_release_ramp_complete",
        }[case]
    with pytest.raises(ValueError, match="start/end pairing drift"):
        validate_current_joint_velocity_recovery_report(
            report,
            apply_calls=apply_calls,
        )


def test_surviving_measured_start_requires_committed_transaction_readbacks() -> None:
    report = _active_recovery_report()
    start = report["events"][0]["start"]
    start["hold_target_rad"] = None
    start["hold_position_target_readback_rad"] = None
    start["hold_velocity_target_readback_rad_s"] = None
    start["hold_effort_target_readback_nm"] = None
    with pytest.raises(ValueError, match="committed start"):
        validate_current_joint_velocity_recovery_report(report, apply_calls=1)


def test_reexceed_completed_report_has_one_closed_event_and_truthful_counts() -> None:
    report = _active_recovery_report()
    report["state"] = {
        "phase": CURRENT_JOINT_VELOCITY_RECOVERY_PHASE_INACTIVE,
        "active": False,
        "consecutive_active_substeps": 0,
        "consecutive_clean_samples": 0,
        "release_ramp_next_index": None,
    }
    report["counters"].update(
        {
            "residual_events": 2,
            "residual_joints": 2,
            "recovery_events": 1,
            "recovery_active_substeps": 6,
            "recovered_events": 1,
            "hold_target_applies": 6,
            "release_ramp_target_applies": 32,
        }
    )
    report["maxima"]["consecutive_recovery_substeps"] = 3
    final_snapshot = _recovery_snapshot(
        apply_index=37,
        joint_position=_neutral_hard_position(),
        joint_velocity=[0.0] * 7,
        committed=True,
    )
    report["events"][0].update(
        {
            "end_apply_index": 37,
            "end_reason": "clean2_release_ramp_complete",
            "recovery_completed_apply_index": 37,
            "last": final_snapshot,
        }
    )
    validated = validate_current_joint_velocity_recovery_report(
        report,
        apply_calls=38,
    )
    assert len(validated["events"]) == 1
    assert validated["events"][0]["end_reason"] == ("clean2_release_ramp_complete")
    assert validated["counters"]["recovered_events"] == 1
    assert (
        validated["counters"]["recovery_active_substeps"]
        == (validated["counters"]["hold_target_applies"])
    )


def test_float32_relative_constant_is_exact_and_finite() -> None:
    assert CURRENT_JOINT_VELOCITY_RECOVERY_RELATIVE_ENVELOPE_FLOAT32 == _float32(1e-4)
    assert math.isfinite(CURRENT_JOINT_VELOCITY_RECOVERY_RELATIVE_ENVELOPE_FLOAT32)
