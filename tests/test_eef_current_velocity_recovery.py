from __future__ import annotations

import copy
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


def _active_recovery_report() -> dict:
    limits = [_float32(value) for value in PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S]
    envelopes = [current_joint_velocity_recovery_envelope(value) for value in limits]
    velocity = [_float32(11.743), *([0.0] * 6)]
    excess = [_float32(abs(velocity[0]) - limits[0]), *([0.0] * 6)]
    ratio = [_float32(abs(velocity[0]) / limits[0]), *([0.0] * 6)]
    snapshot = {
        "apply_index": 0,
        "policy_step": 0,
        "physics_substep": 0,
        "joint_pos_rad": [0.0] * 7,
        "joint_velocity_rad_s": velocity,
        "joint_velocity_limit_rad_s": limits,
        "joint_velocity_envelope_rad_s": envelopes,
        "joint_velocity_limit_excess_rad_s": excess,
        "velocity_to_limit_ratio": ratio,
        "predicted_joint_pos_rad": [0.0] * 7,
        "predicted_hard_limit_clearance_rad": [1.0] * 7,
        "hold_target_rad": [0.0] * 7,
        "hold_position_target_readback_rad": [0.0] * 7,
        "hold_velocity_target_readback_rad_s": [0.0] * 7,
        "hold_effort_target_readback_nm": [0.0] * 7,
    }
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
            "predicted_limit_aborts": 0,
            "transaction_aborts": 0,
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
                "start": copy.deepcopy(snapshot),
                "last": copy.deepcopy(snapshot),
            }
        ],
    }


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
    event = report["events"][0]
    event["end_apply_index"] = 0
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
    final_snapshot = copy.deepcopy(report["events"][0]["last"])
    final_snapshot.update(
        {
            "apply_index": 37,
            "policy_step": 4,
            "physics_substep": 5,
            "joint_velocity_rad_s": [0.0] * 7,
            "joint_velocity_limit_excess_rad_s": [0.0] * 7,
            "velocity_to_limit_ratio": [0.0] * 7,
        }
    )
    report["events"][0].update(
        {
            "end_apply_index": 37,
            "end_reason": "clean2_release_ramp_complete",
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
