"""Lightweight constants for the canonical PolaRiS EEF IK safety profile."""

EEF_IK_SAFETY_PROFILE = "panda_velocity_softlimit_v1"
EEF_IK_APPLY_CADENCE = "physics_substep"
CURRENT_JOINT_SOFT_LIMIT_TOLERANCE_RAD = 1e-5
# One named allowance for float32 subtraction around the configured per-substep
# slew bound.  Keep this identical in the controller, runtime validation, and
# the downstream Ego-LAP completion contract.
JOINT_SLEW_FLOAT32_TOLERANCE_RAD = 1e-6

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
# with soft_joint_pos_limit_factor=1.  The first live controller smoke must
# independently recapture these bytes before a standard v3 suite is launched.
PANDA_SOFT_JOINT_POS_LIMITS_RAD = (
    (-2.8973000049591064, 2.8973000049591064),
    (-1.7627999782562256, 1.7627999782562256),
    (-2.8973000049591064, 2.8973000049591064),
    (-3.0717999935150146, -0.0697999969124794),
    (-2.8973000049591064, 2.8973000049591064),
    (-0.017500000074505806, 3.752500057220459),
    (-2.8973000049591064, 2.8973000049591064),
)
PANDA_SOFT_JOINT_POS_LIMITS_FLOAT32_SHA256 = (
    "d7ec7ea6108d670f910c43a9fba370e5023c7a5b9aa31df06b89ffc172529e00"
)
