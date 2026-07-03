"""Pure state transitions for isolated PolaRiS EEF controller candidates."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from polaris.eef_ik_safety import CURRENT_JOINT_SOFT_LIMIT_TOLERANCE_RAD


def bound_joint_position_target(
    joint_pos: torch.Tensor,
    raw_joint_pos_target: torch.Tensor,
    max_delta_joint_pos: torch.Tensor,
    soft_joint_pos_limits: torch.Tensor,
    *,
    target_guard_band_delta_joint_pos: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slew-limit one target with an independently chosen limit guard band."""

    guard_band = (
        max_delta_joint_pos
        if target_guard_band_delta_joint_pos is None
        else target_guard_band_delta_joint_pos
    )

    raw_delta_joint_pos = raw_joint_pos_target - joint_pos
    applied_delta_joint_pos = torch.clamp(
        raw_delta_joint_pos,
        min=-max_delta_joint_pos,
        max=max_delta_joint_pos,
    )
    slew_limited = raw_delta_joint_pos.abs() > max_delta_joint_pos
    bounded_target = joint_pos + applied_delta_joint_pos
    # Preserve inherited targets bit-for-bit when the slew guard is inactive.
    slew_limited_target = torch.where(
        slew_limited, bounded_target, raw_joint_pos_target
    )
    lower = soft_joint_pos_limits[..., 0]
    upper = soft_joint_pos_limits[..., 1]
    target_lower = lower + guard_band
    target_upper = upper - guard_band
    target_lower_effective = target_lower - torch.clamp(
        lower - joint_pos,
        min=0.0,
        max=CURRENT_JOINT_SOFT_LIMIT_TOLERANCE_RAD,
    )
    target_upper_effective = target_upper + torch.clamp(
        joint_pos - upper,
        min=0.0,
        max=CURRENT_JOINT_SOFT_LIMIT_TOLERANCE_RAD,
    )
    position_limited = (slew_limited_target < target_lower_effective) | (
        slew_limited_target > target_upper_effective
    )
    clipped_target = torch.clamp(
        slew_limited_target,
        min=target_lower_effective,
        max=target_upper_effective,
    )
    safe_target = torch.where(position_limited, clipped_target, slew_limited_target)
    return safe_target, raw_delta_joint_pos, slew_limited, position_limited


@dataclass(frozen=True)
class GripperCloseArmInterlockTransition:
    """One close-transition/interlock countdown update."""

    active: bool
    remaining_after_successful_apply: int
    observed_endpoint_change_count: int
    endpoint_observed_after_successful_apply: bool
    activation_count_delta: int


DISABLED_GRIPPER_CLOSE_ARM_INTERLOCK_TRANSITION = GripperCloseArmInterlockTransition(
    active=False,
    remaining_after_successful_apply=0,
    observed_endpoint_change_count=0,
    endpoint_observed_after_successful_apply=False,
    activation_count_delta=0,
)


def advance_gripper_close_arm_interlock(
    *,
    enabled: bool,
    previous_endpoint_change_count: int,
    current_endpoint_change_count: int,
    endpoint_observed_before_apply: bool,
    endpoint_is_closed: bool,
    remaining_before_apply: int,
    configured_substeps: int,
) -> GripperCloseArmInterlockTransition:
    """Resolve the arm hold for one physics apply without mutating state.

    A newly observed close transition starts the caller's profile-bound
    substep window. A newly observed open transition cancels it. Repeated
    endpoint commands do not refresh the countdown, so a policy cannot freeze
    the arm forever by merely holding a binary close action.
    """

    for name, value in (
        ("previous endpoint-change count", previous_endpoint_change_count),
        ("current endpoint-change count", current_endpoint_change_count),
        ("remaining interlock substeps", remaining_before_apply),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"PolaRiS EEF {name} must be a non-negative int")
    if type(configured_substeps) is not int or configured_substeps <= 0:
        raise ValueError(
            "PolaRiS EEF configured interlock substeps must be a positive int"
        )
    if (
        type(enabled) is not bool
        or type(endpoint_observed_before_apply) is not bool
        or type(endpoint_is_closed) is not bool
    ):
        raise ValueError("PolaRiS EEF close-interlock flags must be bool")
    if current_endpoint_change_count < previous_endpoint_change_count:
        raise ValueError("PolaRiS EEF gripper endpoint-change count regressed")
    count_delta = current_endpoint_change_count - previous_endpoint_change_count
    if count_delta > 1:
        raise ValueError("PolaRiS EEF missed a gripper endpoint transition")
    if not enabled:
        if remaining_before_apply != 0 or endpoint_observed_before_apply:
            raise ValueError("Disabled PolaRiS EEF close interlock retained state")
        return GripperCloseArmInterlockTransition(
            active=False,
            remaining_after_successful_apply=0,
            observed_endpoint_change_count=current_endpoint_change_count,
            endpoint_observed_after_successful_apply=False,
            activation_count_delta=0,
        )

    remaining = remaining_before_apply
    activation_count_delta = 0
    if not endpoint_observed_before_apply:
        if current_endpoint_change_count != 0:
            raise ValueError(
                "PolaRiS EEF first observed gripper endpoint has change history"
            )
        if endpoint_is_closed:
            remaining = configured_substeps
            activation_count_delta = 1
    elif count_delta == 1:
        if endpoint_is_closed:
            remaining = configured_substeps
            activation_count_delta = 1
        else:
            remaining = 0
    active = remaining > 0
    return GripperCloseArmInterlockTransition(
        active=active,
        remaining_after_successful_apply=max(remaining - int(active), 0),
        observed_endpoint_change_count=current_endpoint_change_count,
        endpoint_observed_after_successful_apply=True,
        activation_count_delta=activation_count_delta,
    )
