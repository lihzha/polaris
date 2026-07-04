"""Lightweight constants for the canonical PolaRiS EEF IK safety profile."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import struct
from typing import Any

from polaris.eef_gripper_runtime import EEF_GRIPPER_TARGET_SLEW_PROFILE
from polaris.eef_gripper_runtime import eef_gripper_target_slew_profile

EEF_IK_SAFETY_PROFILE = "panda_velocity_physxlimit_solveriter1_v4"
EEF_IK_CURRENT_VELOCITY_RECOVERY_CANDIDATE_PROFILE = (
    "panda_velocity_physxlimit_solveriter1_residual_recovery8_clean2_v5"
)
EEF_IK_WRIST_ENERGY_BRAKE_CANDIDATE_PROFILE = (
    "panda_velocity_physxlimit_solveriter1_wristenergybrake_candidate_v1"
)
EEF_IK_APPLY_CADENCE = "physics_substep"
# Default-off controller-repair candidate. The nominal arm target keeps a
# five-percent command margin below the live PhysX velocity limit. Gripper
# target-slew/interlock pairings live in the closed gripper runtime mapping.
ARM_SLEW_HEADROOM_CANDIDATE_PROFILE = "panda_nominal_target_slew_0p95_physical_limit_v1"
ARM_SLEW_HEADROOM_RATIO = 0.95
# Compatibility aliases remain derived from the runtime's closed baseline
# mapping; no independent duration or profile literal lives here.
_BASELINE_GRIPPER_TARGET_SLEW = eef_gripper_target_slew_profile(
    EEF_GRIPPER_TARGET_SLEW_PROFILE
)
GRIPPER_CLOSE_ARM_INTERLOCK_CANDIDATE_PROFILE = (
    _BASELINE_GRIPPER_TARGET_SLEW.close_interlock_profile
)
GRIPPER_CLOSE_ARM_INTERLOCK_SUBSTEPS = (
    _BASELINE_GRIPPER_TARGET_SLEW.close_interlock_substeps
)
CURRENT_JOINT_SOFT_LIMIT_TOLERANCE_RAD = 1e-5
TARGET_SOFT_LIMIT_GUARD_BAND_PROFILE = "eef_physx_inner_hardlimit_one_substep_v2"
PHYSX_HARD_LIMIT_PROFILE = "outer_minus_one_velocity_substep_v1"
PHYSX_DERIVED_SOFT_LIMIT_PROFILE = "isaaclab_midpoint_range_factor1_float32_v1"
ARM_VELOCITY_TARGET_PROFILE = "zero_per_physics_substep_v1"
ARTICULATION_SOLVER_PROFILE = "tgs_position64_velocity1_eef_only_v1"
ARTICULATION_SOLVER_READBACK = "composed_usd_physx_articulation_api_all_env_roots_v1"
PANDA_EEF_SOLVER_POSITION_ITERATION_COUNT = 64
PANDA_EEF_SOLVER_VELOCITY_ITERATION_COUNT = 1
PANDA_EEF_PHYSX_SOLVER_TYPE = 1
# One named allowance for float32 subtraction around the configured per-substep
# slew bound.  Keep this identical in the controller, runtime validation, and
# the downstream Ego-LAP completion contract.
JOINT_SLEW_FLOAT32_TOLERANCE_RAD = 1e-6
JOINT_VELOCITY_LIMIT_TOLERANCE_RAD_S = 1e-5
EEF_QUATERNION_UNIT_NORM_TOLERANCE = 1e-3
CURRENT_JOINT_VELOCITY_ABORT_EVIDENCE_PROFILE = (
    "current_joint_velocity_limit_abort_signed_dq_limit_excess_v1"
)
CURRENT_JOINT_VELOCITY_RECOVERY_SCHEMA_VERSION = 1
CURRENT_JOINT_VELOCITY_RECOVERY_PROFILE = (
    "current_joint_velocity_residual_hold_recovery_v1"
)
CURRENT_JOINT_VELOCITY_RECOVERY_ENVELOPE_FORMULA_PROFILE = (
    "float32_limit_plus_float32_limit_times_float32_1e_4_v1"
)
CURRENT_JOINT_VELOCITY_RECOVERY_RELATIVE_ENVELOPE_FLOAT32 = struct.unpack(
    "<f", struct.pack("<f", 1e-4)
)[0]
CURRENT_JOINT_VELOCITY_RECOVERY_MAXIMUM_ACTIVE_SUBSTEPS = 8
CURRENT_JOINT_VELOCITY_RECOVERY_CLEAN_SAMPLES_REQUIRED = 2
CURRENT_JOINT_VELOCITY_RECOVERY_HOLD_PROFILE = (
    "all_arm_hold_current_q_zero_velocity_zero_effort_v1"
)
CURRENT_JOINT_VELOCITY_RECOVERY_PREDICTED_POSITION_PROFILE = (
    "float32_q_plus_float32_dq_times_physics_dt_v1"
)
CURRENT_JOINT_VELOCITY_RECOVERY_TRANSACTION_PROFILE = (
    "position_velocity_effort_setter_readback_trace_then_state_commit_v1"
)
CURRENT_JOINT_VELOCITY_RECOVERY_PHASE_INACTIVE = "inactive"
CURRENT_JOINT_VELOCITY_RECOVERY_PHASE_HOLD = "hold"
CURRENT_JOINT_VELOCITY_RECOVERY_PHASE_RELEASE_RAMP = "release_ramp"
CURRENT_JOINT_VELOCITY_RECOVERY_PHASES = (
    CURRENT_JOINT_VELOCITY_RECOVERY_PHASE_INACTIVE,
    CURRENT_JOINT_VELOCITY_RECOVERY_PHASE_HOLD,
    CURRENT_JOINT_VELOCITY_RECOVERY_PHASE_RELEASE_RAMP,
)
CURRENT_JOINT_VELOCITY_RECOVERY_START_REASONS = (
    "measured_velocity_above_float32_envelope",
    "current_hard_limit_violation",
    "predicted_hard_limit_crossing",
    "target_transaction_failure",
)
CURRENT_JOINT_VELOCITY_RECOVERY_END_REASONS = (
    "clean2_release_ramp_complete",
    "sustained_recovery_abort",
    "current_hard_limit_abort",
    "predicted_hard_limit_abort",
    "transaction_abort",
)
CURRENT_JOINT_VELOCITY_RECOVERY_ABORT_MESSAGES = {
    "sustained_recovery_abort": (
        "PolaRiS EEF measured-velocity recovery exceeded eight consecutive "
        "physics substeps; aborting before DLS and PhysX"
    ),
    "current_hard_limit_abort": (
        "PolaRiS EEF current joint position crossed the installed PhysX hard "
        "envelope; aborting before DLS and PhysX"
    ),
    "predicted_hard_limit_abort": (
        "PolaRiS EEF measured velocity predicts a one-substep crossing of the "
        "installed PhysX hard envelope; aborting before DLS and PhysX"
    ),
    "transaction_abort": (
        "PolaRiS EEF measured-velocity recovery target transaction failed; "
        "reset is required before another apply"
    ),
}


def _float32(value: float) -> float:
    """Return one IEEE-754 binary32 round, rejecting non-finite inputs."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("PolaRiS EEF float32 helper requires one numeric scalar")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("PolaRiS EEF float32 helper requires a finite scalar")
    try:
        rounded = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("PolaRiS EEF float32 helper overflowed") from error
    if not math.isfinite(rounded):
        raise ValueError("PolaRiS EEF float32 helper produced a non-finite scalar")
    return rounded


def current_joint_velocity_recovery_envelope(limit_rad_s: float) -> float:
    """Compute ``float32(L + float32(L * float32(1e-4)))`` exactly."""

    limit = _float32(limit_rad_s)
    if limit <= 0.0:
        raise ValueError("PolaRiS EEF velocity limit must be positive")
    residual = _float32(
        limit * CURRENT_JOINT_VELOCITY_RECOVERY_RELATIVE_ENVELOPE_FLOAT32
    )
    return _float32(limit + residual)


@dataclass(frozen=True)
class CurrentJointVelocityRecoveryClassification:
    """Pure scalar classification at the v5 measured-velocity boundary."""

    measured_velocity_rad_s: float
    absolute_velocity_rad_s: float
    limit_rad_s: float
    envelope_rad_s: float
    limit_excess_rad_s: float
    velocity_to_limit_ratio: float
    residual: bool
    recovery_required: bool


def classify_current_joint_velocity_for_recovery(
    measured_velocity_rad_s: float,
    limit_rad_s: float,
) -> CurrentJointVelocityRecoveryClassification:
    """Classify one signed float32 velocity without weakening the live limit."""

    measured = _float32(measured_velocity_rad_s)
    limit = _float32(limit_rad_s)
    envelope = current_joint_velocity_recovery_envelope(limit)
    absolute = abs(measured)
    return CurrentJointVelocityRecoveryClassification(
        measured_velocity_rad_s=measured,
        absolute_velocity_rad_s=absolute,
        limit_rad_s=limit,
        envelope_rad_s=envelope,
        limit_excess_rad_s=max(_float32(absolute - limit), 0.0),
        velocity_to_limit_ratio=_float32(absolute / limit),
        residual=absolute > limit,
        recovery_required=absolute > envelope,
    )


@dataclass(frozen=True)
class PredictedJointPositionHardLimit:
    """Pure float32 one-substep hard-position prediction."""

    predicted_joint_pos_rad: float
    signed_lower_clearance_rad: float
    signed_upper_clearance_rad: float
    within_hard_limits: bool


def predict_joint_position_against_hard_limits(
    joint_pos_rad: float,
    joint_velocity_rad_s: float,
    physics_dt: float,
    hard_lower_rad: float,
    hard_upper_rad: float,
) -> PredictedJointPositionHardLimit:
    """Evaluate the exact float32 ``q + float32(dq * dt)`` hard-limit guard."""

    position = _float32(joint_pos_rad)
    velocity = _float32(joint_velocity_rad_s)
    timestep = _float32(physics_dt)
    lower = _float32(hard_lower_rad)
    upper = _float32(hard_upper_rad)
    if timestep <= 0.0 or not lower < upper:
        raise ValueError("PolaRiS EEF predicted hard-limit inputs are invalid")
    predicted = _float32(position + _float32(velocity * timestep))
    lower_clearance = _float32(predicted - lower)
    upper_clearance = _float32(upper - predicted)
    return PredictedJointPositionHardLimit(
        predicted_joint_pos_rad=predicted,
        signed_lower_clearance_rad=lower_clearance,
        signed_upper_clearance_rad=upper_clearance,
        within_hard_limits=lower_clearance >= 0.0 and upper_clearance >= 0.0,
    )


# Opt-in diagnostic candidate for the deterministic coupled wrist transient.
# This is deliberately not part of EEF_IK_SAFETY_PROFILE until the target-
# surface boundary replay and full-horizon canary both pass.  The trigger is a
# near-full-substep reversal of an actually applied wrist target.  While the
# two-substep group latch is active, a wrist spring term that would inject
# energy is replaced by a safely bounded hold-at-current-position target; the
# exact zero velocity target remains unchanged.
WRIST_ENERGY_BRAKE_PROFILE = (
    "panda_j5_j7_applied_target_reversal_group_energy_brake_2substep_v1"
)
WRIST_ENERGY_BRAKE_JOINT_NAMES = (
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
)
WRIST_ENERGY_BRAKE_LATCH_SUBSTEPS = 2
WRIST_ENERGY_BRAKE_TARGET_SHIFT_FRACTION = 0.9

# Default-off EEF-only diagnostic candidate for the deterministic gripper-close
# transient. Isaac Lab 2.3 does not promote the legacy implicit-actuator
# ``velocity_limit`` field into a PhysX limit. The candidate explicitly authors
# the same intended value through ``velocity_limit_sim`` without changing the
# binary gripper action contract or native joint-position configuration.
GRIPPER_VELOCITY_LIMIT_RAD_S = 5.0
GRIPPER_EFFORT_LIMIT = 200.0

PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S = (
    2.175,
    2.175,
    2.175,
    2.175,
    2.61,
    2.61,
    2.61,
)
PANDA_EEF_JOINT_EFFORT_LIMITS = (
    87.0,
    87.0,
    87.0,
    87.0,
    12.0,
    12.0,
    12.0,
)

# Canonical float32 limits expected from the pinned NVIDIA DROID articulation
# with soft_joint_pos_limit_factor=1.  Every safety-profile revision must
# independently recapture these bytes before a standard suite is launched.
PANDA_SOFT_JOINT_POS_LIMITS_RAD = (
    (-2.8973000049591064, 2.8973000049591064),
    (-1.7627999782562256, 1.7627999782562256),
    (-2.8973000049591064, 2.8973000049591064),
    (-3.0717999935150146, -0.06979990005493164),
    (-2.8973000049591064, 2.8973000049591064),
    (-0.017499923706054688, 3.752500057220459),
    (-2.8973000049591064, 2.8973000049591064),
)
PANDA_SOFT_JOINT_POS_LIMITS_FLOAT32_SHA256 = (
    "fbf7535901c042fea5d901812ecd02c5fd81ade06c23c1499c32d66a859104de"
)
PANDA_TARGET_JOINT_POS_LIMITS_FLOAT32_SHA256 = (
    "09b20ab18c35d6dc22a3edbc2beca2edff419e242dd07d74cd1d65df9ce67e0f"
)
PANDA_PHYSX_HARD_JOINT_POS_LIMITS_FLOAT32_SHA256 = (
    PANDA_TARGET_JOINT_POS_LIMITS_FLOAT32_SHA256
)
# Isaac Lab derives its soft buffer from the installed hard limits with
# float32 midpoint/range arithmetic.  Even with factor=1 this is not a bitwise
# identity for two asymmetric Panda limits, so bind that readback separately
# from both the canonical outer envelope and the exact PhysX hard envelope.
PANDA_PHYSX_DERIVED_SOFT_JOINT_POS_LIMITS_RAD = (
    (-2.8791749477386475, 2.8791749477386475),
    (-1.7446749210357666, 1.7446749210357666),
    (-2.8791749477386475, 2.8791749477386475),
    (-3.0536749362945557, -0.08792495727539062),
    (-2.8755500316619873, 2.8755500316619873),
    (0.004250049591064453, 3.73075008392334),
    (-2.8755500316619873, 2.8755500316619873),
)
PANDA_PHYSX_DERIVED_SOFT_JOINT_POS_LIMITS_FLOAT32_SHA256 = (
    "dd7865f59efb23e96d7d4cbb5e129906b04a42b5e5c0941459bfc8866dd7ecd0"
)


def current_joint_velocity_abort_evidence_sha256(
    evidence: Mapping[str, object],
) -> str:
    """Hash the closed abort object with one reproducible JSON encoding."""

    try:
        encoded = json.dumps(
            dict(evidence),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Current-joint-velocity abort evidence is not canonical JSON"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def format_current_joint_velocity_abort_message(
    evidence: Mapping[str, object],
) -> str:
    """Format the controller exception bound to every signed evidence field."""

    mask = evidence.get("exceeded_joint_mask")
    names = evidence.get("joint_names")
    velocities = evidence.get("joint_velocity_rad_s")
    limits = evidence.get("joint_velocity_limit_rad_s")
    excess = evidence.get("joint_velocity_limit_excess_rad_s")
    if not all(
        isinstance(value, list) for value in (mask, names, velocities, limits, excess)
    ):
        raise ValueError("Current-joint-velocity abort vector type drift")
    if not (
        len(mask) == len(names) == len(velocities) == len(limits) == len(excess) == 7
    ):
        raise ValueError("Current-joint-velocity abort vector width drift")
    try:
        first_exceeded = next(
            index for index, value in enumerate(mask) if value is True
        )
    except StopIteration as error:
        raise ValueError(
            "Current-joint-velocity abort has no exceeded joint"
        ) from error
    digest = current_joint_velocity_abort_evidence_sha256(evidence)
    return (
        "PolaRiS EEF IK current joint velocity exceeds the live simulation "
        "limit; aborting before DLS and PhysX "
        f"(joint={names[first_exceeded]!r}, "
        f"velocity_rad_s={velocities[first_exceeded]!r}, "
        f"limit_rad_s={limits[first_exceeded]!r}, "
        f"excess_rad_s={excess[first_exceeded]!r}, "
        f"policy_step={evidence.get('policy_step')!r}, "
        f"physics_substep={evidence.get('physics_substep')!r}, "
        f"evidence_sha256={digest})"
    )


def validate_one_step_adversarial_report(report: dict[str, Any]) -> dict[str, Any]:
    """Validate the dedicated one-policy-step slew-guard smoke evidence."""

    counters = report.get("counters")
    maxima = report.get("maxima")
    bounds = report.get("max_delta_joint_pos_rad")
    if not isinstance(counters, dict) or not isinstance(maxima, dict):
        raise ValueError("Adversarial EEF smoke report lacks counters/maxima")
    if counters.get("apply_calls") != 8 or counters.get("environment_substeps") != 8:
        raise ValueError(
            "Adversarial EEF smoke must execute exactly 8 physics substeps"
        )
    if (
        type(counters.get("slew_limit_events")) is not int
        or counters["slew_limit_events"] < 1
    ):
        raise ValueError("Adversarial EEF smoke did not activate the slew guard")
    abort_count = sum(
        counters.get(name, -1)
        for name in (
            "current_joint_limit_aborts",
            "invariant_aborts",
            "nonfinite_aborts",
        )
    )
    if abort_count != 0 or counters.get("post_clamp_target_violations") != 0:
        raise ValueError("Adversarial EEF smoke triggered an abort/invariant violation")
    applied = maxima.get("applied_delta_joint_pos_rad")
    if (
        not isinstance(applied, list)
        or len(applied) != 7
        or not isinstance(bounds, list)
        or len(bounds) != 7
    ):
        raise ValueError("Adversarial EEF smoke has invalid per-joint slew evidence")
    for actual, bound in zip(applied, bounds, strict=True):
        if (
            isinstance(actual, bool)
            or isinstance(bound, bool)
            or not isinstance(actual, (int, float))
            or not isinstance(bound, (int, float))
            or not math.isfinite(float(actual))
            or not math.isfinite(float(bound))
            or actual > bound + JOINT_SLEW_FLOAT32_TOLERANCE_RAD
        ):
            raise ValueError("Adversarial EEF smoke exceeded its float32 slew bound")
    return {
        "apply_calls": 8,
        "slew_limit_events": counters["slew_limit_events"],
        "abort_count": abort_count,
        "post_clamp_target_violations": 0,
        "applied_within_bounds": True,
    }
