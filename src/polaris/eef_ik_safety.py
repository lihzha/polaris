"""Lightweight constants for the canonical PolaRiS EEF IK safety profile."""

from __future__ import annotations

import math
from typing import Any

EEF_IK_SAFETY_PROFILE = "panda_velocity_physxlimit_solveriter1_v4"
EEF_IK_WRIST_ENERGY_BRAKE_CANDIDATE_PROFILE = (
    "panda_velocity_physxlimit_solveriter1_wristenergybrake_candidate_v1"
)
EEF_IK_APPLY_CADENCE = "physics_substep"
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
