"""Pure state transitions for isolated PolaRiS EEF controller candidates."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from polaris.eef_ik_safety import CURRENT_JOINT_SOFT_LIMIT_TOLERANCE_RAD


ARM_RELEASE_RAMP_SUBSTEPS = 16
ARM_RELEASE_RAMP_PROFILE = "arm_post_interlock_linear_slew_cap_release_ramp16_v3"
ARM_RELEASE_RAMP_STATE_PROFILE = "hold_ramp_release_state_machine_v1"
ARM_RELEASE_RAMP_FRACTION_PROFILE = (
    "inclusive_linear_float32_0_over_15_to_15_over_15_v2"
)
ARM_RELEASE_RAMP_FORMULA_PROFILE = (
    "endpoint_exact_else_float32_clamp_nominal_delta_by_scaled_slew_v1"
)
ARM_RELEASE_RAMP_TRANSACTION_PROFILE = (
    "single_arm_target_setter_readback_trace_then_state_commit_v1"
)
ARM_RELEASE_PHASE_HOLD = "hold"
ARM_RELEASE_PHASE_RAMP = "ramp"
ARM_RELEASE_PHASE_RELEASE = "release"
ARM_RELEASE_PHASES = (
    ARM_RELEASE_PHASE_HOLD,
    ARM_RELEASE_PHASE_RAMP,
    ARM_RELEASE_PHASE_RELEASE,
)


def arm_release_ramp_fraction(index: int) -> float:
    """Return one closed float32-linear fraction from zero through one."""

    if type(index) is not int or not 0 <= index < ARM_RELEASE_RAMP_SUBSTEPS:
        raise ValueError("PolaRiS EEF arm release-ramp index drift")
    return float(
        torch.tensor(
            index / (ARM_RELEASE_RAMP_SUBSTEPS - 1),
            dtype=torch.float32,
        ).item()
    )


@dataclass(frozen=True)
class ArmReleaseRampTarget:
    """One finite float32 arm target under the selected release fraction."""

    target: torch.Tensor
    fraction: float
    limited_joint_mask: torch.Tensor


def apply_arm_release_ramp_target(
    current_joint_pos: torch.Tensor,
    nominal_joint_pos_target: torch.Tensor,
    nominal_max_delta_joint_pos: torch.Tensor,
    *,
    ramp_index: int,
) -> ArmReleaseRampTarget:
    """Tighten a nominal safe target for one inclusive release-ramp index.

    Index zero is bitwise current position and index fifteen is bitwise the
    already-safe nominal target. Intermediate indices clamp the nominal delta
    by the float32 product of the nominal slew bound and the selected fraction.
    """

    fraction = arm_release_ramp_fraction(ramp_index)
    tensors = (
        current_joint_pos,
        nominal_joint_pos_target,
        nominal_max_delta_joint_pos,
    )
    if any(
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or value.shape != current_joint_pos.shape
        or value.device != current_joint_pos.device
        or not bool(torch.isfinite(value).all().item())
        for value in tensors
    ):
        raise ValueError("PolaRiS EEF arm release-ramp tensor contract drift")
    if not bool((nominal_max_delta_joint_pos > 0.0).all().item()):
        raise ValueError("PolaRiS EEF arm release-ramp slew bound is not positive")

    if ramp_index == 0:
        target = current_joint_pos.detach().clone()
    elif ramp_index == ARM_RELEASE_RAMP_SUBSTEPS - 1:
        target = nominal_joint_pos_target.detach().clone()
    else:
        maximum_delta = nominal_max_delta_joint_pos * fraction
        nominal_delta = nominal_joint_pos_target - current_joint_pos
        target = current_joint_pos + torch.clamp(
            nominal_delta,
            min=-maximum_delta,
            max=maximum_delta,
        )
    limited = target != nominal_joint_pos_target
    if not bool(torch.isfinite(target).all().item()):
        raise ValueError("PolaRiS EEF arm release-ramp target became non-finite")
    return ArmReleaseRampTarget(
        target=target,
        fraction=fraction,
        limited_joint_mask=limited,
    )


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
    completion_count_delta: int
    open_cancel_count_delta: int


DISABLED_GRIPPER_CLOSE_ARM_INTERLOCK_TRANSITION = GripperCloseArmInterlockTransition(
    active=False,
    remaining_after_successful_apply=0,
    observed_endpoint_change_count=0,
    endpoint_observed_after_successful_apply=False,
    activation_count_delta=0,
    completion_count_delta=0,
    open_cancel_count_delta=0,
)


@dataclass(frozen=True)
class ArmReleaseRampTransition:
    """One mutation-free HOLD/RAMP/RELEASE lifecycle transition."""

    phase_after_successful_apply: str
    ramp_index_to_apply: int | None
    next_ramp_index_after_successful_apply: int | None
    release_observed_delta: int
    ramp_started_delta: int
    ramp_completed_delta: int
    ramp_cancelled_by_reactivation_delta: int


DISABLED_ARM_RELEASE_RAMP_TRANSITION = ArmReleaseRampTransition(
    phase_after_successful_apply=ARM_RELEASE_PHASE_RELEASE,
    ramp_index_to_apply=None,
    next_ramp_index_after_successful_apply=None,
    release_observed_delta=0,
    ramp_started_delta=0,
    ramp_completed_delta=0,
    ramp_cancelled_by_reactivation_delta=0,
)


def advance_arm_release_ramp(
    *,
    enabled: bool,
    phase_before_apply: str,
    next_ramp_index_before_apply: int | None,
    interlock_remaining_before_apply: int,
    interlock_active_this_apply: bool,
    interlock_remaining_after_apply: int,
    interlock_activation_count_delta: int,
) -> ArmReleaseRampTransition:
    """Stage one explicit HOLD/RAMP/RELEASE transition.

    A naturally completed hold starts index zero on the following physics
    substep because the completion substep still applies the fixed anchor. An
    open-cancel starts index zero immediately because that substep is already
    released. A new close while ramping cancels the ramp and returns to HOLD.
    Additional open commands while ramping do not restart or skip the ramp.
    """

    if type(enabled) is not bool or type(interlock_active_this_apply) is not bool:
        raise ValueError("PolaRiS EEF arm release-ramp flags must be bool")
    if phase_before_apply not in ARM_RELEASE_PHASES:
        raise ValueError("PolaRiS EEF arm release-ramp phase drift")
    if next_ramp_index_before_apply is not None and (
        type(next_ramp_index_before_apply) is not int
        or not 0 <= next_ramp_index_before_apply < ARM_RELEASE_RAMP_SUBSTEPS
    ):
        raise ValueError("PolaRiS EEF arm release-ramp next-index drift")
    for field, value in (
        ("interlock remaining before apply", interlock_remaining_before_apply),
        ("interlock remaining after apply", interlock_remaining_after_apply),
        ("interlock activation delta", interlock_activation_count_delta),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"PolaRiS EEF arm release-ramp {field} drift")
    if interlock_activation_count_delta not in (0, 1):
        raise ValueError("PolaRiS EEF arm release-ramp activation cadence drift")
    if (phase_before_apply == ARM_RELEASE_PHASE_RAMP) is (
        next_ramp_index_before_apply is None
    ):
        raise ValueError("PolaRiS EEF arm release-ramp phase/index binding drift")
    if phase_before_apply == ARM_RELEASE_PHASE_HOLD and (
        interlock_remaining_before_apply <= 0
    ):
        raise ValueError("PolaRiS EEF arm release-ramp HOLD lacks an interlock")
    if phase_before_apply != ARM_RELEASE_PHASE_HOLD and (
        interlock_remaining_before_apply != 0
    ):
        raise ValueError("PolaRiS EEF arm release-ramp non-HOLD retained interlock")

    if not enabled:
        if (
            phase_before_apply != ARM_RELEASE_PHASE_RELEASE
            or next_ramp_index_before_apply is not None
            or interlock_remaining_before_apply != 0
            or interlock_active_this_apply
            or interlock_remaining_after_apply != 0
            or interlock_activation_count_delta != 0
        ):
            raise ValueError("Disabled PolaRiS EEF arm release ramp retained state")
        return DISABLED_ARM_RELEASE_RAMP_TRANSITION

    if interlock_active_this_apply:
        if interlock_activation_count_delta == 1:
            if interlock_remaining_after_apply <= 0:
                raise ValueError(
                    "PolaRiS EEF arm release-ramp activation did not retain HOLD"
                )
        elif phase_before_apply != ARM_RELEASE_PHASE_HOLD:
            raise ValueError("PolaRiS EEF arm release-ramp entered HOLD without close")
        natural_release = interlock_remaining_after_apply == 0
        if natural_release:
            return ArmReleaseRampTransition(
                phase_after_successful_apply=ARM_RELEASE_PHASE_RAMP,
                ramp_index_to_apply=None,
                next_ramp_index_after_successful_apply=0,
                release_observed_delta=1,
                ramp_started_delta=1,
                ramp_completed_delta=0,
                ramp_cancelled_by_reactivation_delta=0,
            )
        return ArmReleaseRampTransition(
            phase_after_successful_apply=ARM_RELEASE_PHASE_HOLD,
            ramp_index_to_apply=None,
            next_ramp_index_after_successful_apply=None,
            release_observed_delta=0,
            ramp_started_delta=0,
            ramp_completed_delta=0,
            ramp_cancelled_by_reactivation_delta=int(
                phase_before_apply == ARM_RELEASE_PHASE_RAMP
            ),
        )

    if interlock_remaining_after_apply != 0:
        raise ValueError(
            "PolaRiS EEF arm release-ramp inactive interlock retained HOLD"
        )
    open_cancel_release = (
        phase_before_apply == ARM_RELEASE_PHASE_HOLD
        and interlock_remaining_before_apply > 0
    )
    if open_cancel_release:
        ramp_index = 0
        release_observed_delta = 1
        ramp_started_delta = 1
    elif phase_before_apply == ARM_RELEASE_PHASE_RAMP:
        ramp_index = next_ramp_index_before_apply
        release_observed_delta = 0
        ramp_started_delta = 0
    else:
        if interlock_activation_count_delta != 0:
            raise ValueError(
                "PolaRiS EEF arm release-ramp close activation was not active"
            )
        return ArmReleaseRampTransition(
            phase_after_successful_apply=ARM_RELEASE_PHASE_RELEASE,
            ramp_index_to_apply=None,
            next_ramp_index_after_successful_apply=None,
            release_observed_delta=0,
            ramp_started_delta=0,
            ramp_completed_delta=0,
            ramp_cancelled_by_reactivation_delta=0,
        )
    if ramp_index is None:
        raise ValueError("PolaRiS EEF arm release-ramp active index is absent")
    completed = ramp_index == ARM_RELEASE_RAMP_SUBSTEPS - 1
    return ArmReleaseRampTransition(
        phase_after_successful_apply=(
            ARM_RELEASE_PHASE_RELEASE if completed else ARM_RELEASE_PHASE_RAMP
        ),
        ramp_index_to_apply=ramp_index,
        next_ramp_index_after_successful_apply=(None if completed else ramp_index + 1),
        release_observed_delta=release_observed_delta,
        ramp_started_delta=ramp_started_delta,
        ramp_completed_delta=int(completed),
        ramp_cancelled_by_reactivation_delta=0,
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
            completion_count_delta=0,
            open_cancel_count_delta=0,
        )

    remaining = remaining_before_apply
    activation_count_delta = 0
    open_cancel_count_delta = 0
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
            open_cancel_count_delta = int(remaining_before_apply > 0)
            remaining = 0
    active = remaining > 0
    return GripperCloseArmInterlockTransition(
        active=active,
        remaining_after_successful_apply=max(remaining - int(active), 0),
        observed_endpoint_change_count=current_endpoint_change_count,
        endpoint_observed_after_successful_apply=True,
        activation_count_delta=activation_count_delta,
        completion_count_delta=int(active and remaining == 1),
        open_cancel_count_delta=open_cancel_count_delta,
    )
