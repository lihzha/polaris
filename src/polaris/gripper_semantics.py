"""Shared gripper threshold semantics for DROID-compatible policies."""

GRIPPER_THRESHOLD_PROFILE = "closed_positive_ge_0p5_inverse_open_gt_0p5_v1"


def closed_positive_gripper_mask(actions):
    """Return the closed mask inverse to training's open-positive ``> 0.5`` rule."""

    return actions >= 0.5
