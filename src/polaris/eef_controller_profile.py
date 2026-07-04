"""Closed configuration and evidence binding for PolaRiS EEF controllers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math
import struct
from types import MappingProxyType
from typing import Any

from polaris.config import EEF_CONTROLLER_BASELINE_PROFILE
from polaris.config import EEF_CONTROLLER_MIMIC_COMPLIANCE_CANDIDATE_PROFILE
from polaris.config import EEF_CONTROLLER_PROFILES
from polaris.config import EEF_CONTROLLER_RELEASE_RAMP_CANDIDATE_PROFILE
from polaris.config import EEF_CONTROLLER_VELOCITY_RECOVERY_CANDIDATE_PROFILE
from polaris.eef_controller_repair import ARM_RELEASE_PHASE_HOLD
from polaris.eef_controller_repair import ARM_RELEASE_PHASE_RAMP
from polaris.eef_controller_repair import ARM_RELEASE_PHASE_RELEASE
from polaris.eef_controller_repair import ARM_RELEASE_RAMP_FORMULA_PROFILE
from polaris.eef_controller_repair import ARM_RELEASE_RAMP_FRACTION_PROFILE
from polaris.eef_controller_repair import ARM_RELEASE_RAMP_PROFILE
from polaris.eef_controller_repair import ARM_RELEASE_RAMP_STATE_PROFILE
from polaris.eef_controller_repair import ARM_RELEASE_RAMP_SUBSTEPS
from polaris.eef_controller_repair import ARM_RELEASE_RAMP_TRANSACTION_PROFILE
from polaris.eef_controller_repair import arm_release_ramp_fraction
from polaris.eef_gripper_runtime import EEF_GRIPPER_MIMIC_COMPLIANCE_PROFILE
from polaris.eef_gripper_runtime import EEF_GRIPPER_MIMIC_COMPLIANCE_OVERLAY_IDENTITY
from polaris.eef_gripper_runtime import EEF_GRIPPER_TARGET_SLEW_ACTION_CLASS
from polaris.eef_gripper_runtime import EEF_GRIPPER_TARGET_SLEW_PROFILE
from polaris.eef_gripper_runtime import (
    EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE,
)
from polaris.eef_gripper_runtime import (
    configure_eef_gripper_mimic_compliance_spawn_overlay,
)
from polaris.eef_gripper_runtime import eef_gripper_target_slew_profile
from polaris.eef_gripper_runtime import validate_eef_gripper_static_contract
from polaris.eef_gripper_failure_trace import EEF_ALL_SIX_GRIPPER_TRACE_PROFILE
from polaris.eef_gripper_failure_trace import (
    make_eef_all_six_gripper_failure_trace_class,
)
from polaris.eef_ik_safety import ARM_SLEW_HEADROOM_CANDIDATE_PROFILE
from polaris.eef_ik_safety import ARM_SLEW_HEADROOM_RATIO
from polaris.eef_ik_safety import current_joint_velocity_recovery_envelope
from polaris.eef_ik_safety import CURRENT_JOINT_VELOCITY_RECOVERY_CLEAN_SAMPLES_REQUIRED
from polaris.eef_ik_safety import (
    CURRENT_JOINT_VELOCITY_RECOVERY_ENVELOPE_FORMULA_PROFILE,
)
from polaris.eef_ik_safety import CURRENT_JOINT_VELOCITY_RECOVERY_END_REASONS
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
from polaris.eef_ik_safety import CURRENT_JOINT_VELOCITY_RECOVERY_START_REASONS
from polaris.eef_ik_safety import CURRENT_JOINT_VELOCITY_RECOVERY_TRANSACTION_PROFILE
from polaris.eef_ik_safety import PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S
from polaris.eef_ik_safety import PANDA_EEF_PHYSICS_DT_FLOAT32
from polaris.eef_ik_safety import PANDA_PHYSX_HARD_JOINT_POS_LIMITS_FLOAT32_SHA256
from polaris.eef_ik_safety import PANDA_PHYSX_HARD_JOINT_POS_LIMITS_RAD
from polaris.eef_ik_safety import PHYSX_HARD_LIMIT_PROFILE


ARM_SLEW_HEADROOM_REPORT_FIELDS = {
    "enabled",
    "profile",
    "ratio",
    "physical_max_delta_joint_pos_rad",
    "nominal_max_delta_joint_pos_rad",
}
GRIPPER_CLOSE_ARM_INTERLOCK_REPORT_FIELDS = {
    "enabled",
    "profile",
    "configured_substeps",
    "remaining_substeps",
    "observed_endpoint_change_count",
    "endpoint_observed",
    "activation_count",
    "active_apply_count",
    "anchor_valid",
    "anchor_capture_count",
    "anchor_target_apply_count",
    "anchor_first_exact_target_count",
    "anchor_refresh_count",
    "anchor_slew_limit_event_count",
    "anchor_slew_limited_joint_count",
    "anchor_position_limit_event_count",
    "anchor_position_limited_joint_count",
    "anchor_completion_count",
    "anchor_open_cancel_count",
    "last_activation_apply_index",
    "last_anchor_joint_pos_rad",
    "last_anchor_little_endian_float32_sha256",
    "max_abs_current_anchor_residual_rad",
    "max_abs_target_anchor_residual_rad",
    "max_abs_active_delta_joint_pos_rad",
    "released_apply_count",
    "max_abs_released_delta_joint_pos_rad",
}
GRIPPER_CLOSE_ARM_INTERLOCK_COUNTER_FIELDS = (
    "remaining_substeps",
    "observed_endpoint_change_count",
    "activation_count",
    "active_apply_count",
    "released_apply_count",
    "anchor_capture_count",
    "anchor_target_apply_count",
    "anchor_first_exact_target_count",
    "anchor_refresh_count",
    "anchor_slew_limit_event_count",
    "anchor_slew_limited_joint_count",
    "anchor_position_limit_event_count",
    "anchor_position_limited_joint_count",
    "anchor_completion_count",
    "anchor_open_cancel_count",
)
ARM_RELEASE_RAMP_REPORT_FIELDS = {
    "enabled",
    "profile",
    "state_profile",
    "substeps",
    "fraction_profile",
    "fractions_float32",
    "formula_profile",
    "transaction_profile",
    "open_during_ramp_policy",
    "phase",
    "next_index",
    "release_observed_count",
    "ramp_started_count",
    "ramp_completed_count",
    "ramp_cancelled_by_reactivation_count",
    "ramp_target_apply_count",
    "cancelled_ramp_target_apply_count",
    "ramp_limited_target_apply_count",
    "ramp_limited_joint_target_count",
    "last_target_apply_index",
    "last_ramp_index",
    "max_abs_nominal_to_ramped_target_change_rad",
    "gripper_target_or_state_write_count",
}
ARM_RELEASE_RAMP_COUNTER_FIELDS = (
    "release_observed_count",
    "ramp_started_count",
    "ramp_completed_count",
    "ramp_cancelled_by_reactivation_count",
    "ramp_target_apply_count",
    "cancelled_ramp_target_apply_count",
    "ramp_limited_target_apply_count",
    "ramp_limited_joint_target_count",
)
CURRENT_JOINT_VELOCITY_RECOVERY_FIELDS = {
    "contract",
    "state",
    "counters",
    "maxima",
    "events",
}
CURRENT_JOINT_VELOCITY_RECOVERY_CONTRACT_FIELDS = {
    "schema_version",
    "profile",
    "envelope_formula_profile",
    "relative_envelope_float32",
    "maximum_active_substeps",
    "clean_samples_required",
    "hold_profile",
    "predicted_position_profile",
    "hard_limit_profile",
    "release_ramp_profile",
    "transaction_profile",
    "joint_names",
    "velocity_limits_rad_s",
    "velocity_envelopes_rad_s",
    "physics_dt_float32",
    "hard_joint_position_limits_rad",
    "hard_joint_position_limits_little_endian_float32_sha256",
}
CURRENT_JOINT_VELOCITY_RECOVERY_STATE_FIELDS = {
    "phase",
    "active",
    "consecutive_active_substeps",
    "consecutive_clean_samples",
    "release_ramp_next_index",
}
CURRENT_JOINT_VELOCITY_RECOVERY_COUNTER_FIELDS = {
    "residual_events",
    "residual_joints",
    "recovery_events",
    "recovery_active_substeps",
    "recovered_events",
    "hold_target_applies",
    "release_ramp_target_applies",
    "sustained_aborts",
    "current_hard_limit_aborts",
    "predicted_limit_aborts",
    "transaction_aborts",
    "lower_endpoint_transition_aborts",
}
CURRENT_JOINT_VELOCITY_RECOVERY_MAXIMA_FIELDS = {
    "abs_velocity_to_limit_ratio",
    "consecutive_recovery_substeps",
    "abs_velocity_residual_excess_rad_s",
}
CURRENT_JOINT_VELOCITY_RECOVERY_EVENT_FIELDS = {
    "event_index",
    "start_apply_index",
    "end_apply_index",
    "start_reason",
    "end_reason",
    "start",
    "last",
}
CURRENT_JOINT_VELOCITY_RECOVERY_SNAPSHOT_FIELDS = {
    "apply_index",
    "policy_step",
    "physics_substep",
    "joint_pos_rad",
    "joint_velocity_rad_s",
    "joint_velocity_limit_rad_s",
    "joint_velocity_envelope_rad_s",
    "joint_velocity_limit_excess_rad_s",
    "velocity_to_limit_ratio",
    "predicted_joint_pos_rad",
    "predicted_hard_limit_clearance_rad",
    "hold_target_rad",
    "hold_position_target_readback_rad",
    "hold_velocity_target_readback_rad_s",
    "hold_effort_target_readback_nm",
}


@dataclass(frozen=True)
class EefControllerProfileSpec:
    """Exact component identities selected by one public controller profile."""

    profile: str
    failure_substep_trace_enabled: bool
    all_six_gripper_trace_enabled: bool
    arm_slew_headroom_enabled: bool
    gripper_close_arm_interlock_enabled: bool
    arm_release_ramp_enabled: bool
    current_joint_velocity_recovery_enabled: bool
    target_slew_rate_0p25_enabled: bool
    target_slew_profile: str
    close_interlock_profile: str
    close_interlock_substeps: int
    fixed_activation_anchor: bool
    mimic_compliance_profile: str | None


def _profile_spec(
    *,
    profile: str,
    failure_substep_trace_enabled: bool,
    all_six_gripper_trace_enabled: bool,
    arm_slew_headroom_enabled: bool,
    gripper_close_arm_interlock_enabled: bool,
    arm_release_ramp_enabled: bool,
    current_joint_velocity_recovery_enabled: bool,
    target_slew_rate_0p25_enabled: bool,
    target_slew_profile: str,
    mimic_compliance_profile: str | None,
) -> EefControllerProfileSpec:
    target = eef_gripper_target_slew_profile(target_slew_profile)
    return EefControllerProfileSpec(
        profile=profile,
        failure_substep_trace_enabled=failure_substep_trace_enabled,
        all_six_gripper_trace_enabled=all_six_gripper_trace_enabled,
        arm_slew_headroom_enabled=arm_slew_headroom_enabled,
        gripper_close_arm_interlock_enabled=gripper_close_arm_interlock_enabled,
        arm_release_ramp_enabled=arm_release_ramp_enabled,
        current_joint_velocity_recovery_enabled=(
            current_joint_velocity_recovery_enabled
        ),
        target_slew_rate_0p25_enabled=target_slew_rate_0p25_enabled,
        target_slew_profile=target.profile,
        close_interlock_profile=target.close_interlock_profile,
        close_interlock_substeps=target.close_interlock_substeps,
        fixed_activation_anchor=target.fixed_activation_anchor,
        mimic_compliance_profile=mimic_compliance_profile,
    )


_EEF_CONTROLLER_PROFILE_SPECS = MappingProxyType(
    {
        EEF_CONTROLLER_BASELINE_PROFILE: _profile_spec(
            profile=EEF_CONTROLLER_BASELINE_PROFILE,
            failure_substep_trace_enabled=False,
            all_six_gripper_trace_enabled=False,
            arm_slew_headroom_enabled=False,
            gripper_close_arm_interlock_enabled=False,
            arm_release_ramp_enabled=False,
            current_joint_velocity_recovery_enabled=False,
            target_slew_rate_0p25_enabled=False,
            target_slew_profile=EEF_GRIPPER_TARGET_SLEW_PROFILE,
            mimic_compliance_profile=None,
        ),
        EEF_CONTROLLER_MIMIC_COMPLIANCE_CANDIDATE_PROFILE: _profile_spec(
            profile=EEF_CONTROLLER_MIMIC_COMPLIANCE_CANDIDATE_PROFILE,
            failure_substep_trace_enabled=True,
            all_six_gripper_trace_enabled=True,
            arm_slew_headroom_enabled=True,
            gripper_close_arm_interlock_enabled=True,
            arm_release_ramp_enabled=False,
            current_joint_velocity_recovery_enabled=False,
            target_slew_rate_0p25_enabled=True,
            target_slew_profile=(EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE),
            mimic_compliance_profile=EEF_GRIPPER_MIMIC_COMPLIANCE_PROFILE,
        ),
        EEF_CONTROLLER_RELEASE_RAMP_CANDIDATE_PROFILE: _profile_spec(
            profile=EEF_CONTROLLER_RELEASE_RAMP_CANDIDATE_PROFILE,
            failure_substep_trace_enabled=True,
            all_six_gripper_trace_enabled=True,
            arm_slew_headroom_enabled=True,
            gripper_close_arm_interlock_enabled=True,
            arm_release_ramp_enabled=True,
            current_joint_velocity_recovery_enabled=False,
            target_slew_rate_0p25_enabled=True,
            target_slew_profile=(EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE),
            mimic_compliance_profile=EEF_GRIPPER_MIMIC_COMPLIANCE_PROFILE,
        ),
        EEF_CONTROLLER_VELOCITY_RECOVERY_CANDIDATE_PROFILE: _profile_spec(
            profile=EEF_CONTROLLER_VELOCITY_RECOVERY_CANDIDATE_PROFILE,
            failure_substep_trace_enabled=True,
            all_six_gripper_trace_enabled=True,
            arm_slew_headroom_enabled=True,
            gripper_close_arm_interlock_enabled=True,
            arm_release_ramp_enabled=True,
            current_joint_velocity_recovery_enabled=True,
            target_slew_rate_0p25_enabled=True,
            target_slew_profile=(EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE),
            mimic_compliance_profile=EEF_GRIPPER_MIMIC_COMPLIANCE_PROFILE,
        ),
    }
)


def eef_controller_profile(profile: str) -> EefControllerProfileSpec:
    """Resolve an exact public profile from the closed mapping."""

    if type(profile) is not str or profile not in _EEF_CONTROLLER_PROFILE_SPECS:
        raise ValueError(f"Unknown PolaRiS EEF controller profile: {profile!r}")
    if tuple(_EEF_CONTROLLER_PROFILE_SPECS) != EEF_CONTROLLER_PROFILES:
        raise ValueError("PolaRiS EEF controller profile mapping drift")
    return _EEF_CONTROLLER_PROFILE_SPECS[profile]


def _config_bool(value: Any, *, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"PolaRiS EEF controller config {field} must be bool")
    return value


def _actions(env_cfg: Any) -> tuple[Any, Any, Any]:
    actions = getattr(env_cfg, "actions", None)
    arm = getattr(actions, "arm", None)
    finger = getattr(actions, "finger_joint", None)
    if actions is None or arm is None or finger is None:
        raise ValueError("PolaRiS EEF controller profile requires arm/finger actions")
    return actions, arm, finger


def _expected_arm_action_class() -> type:
    # This module is imported before AppLauncher starts Isaac. Resolve the
    # production class only when the configured environment is ready.
    from polaris.robust_differential_ik import (  # noqa: PLC0415
        RobustDifferentialInverseKinematicsAction,
    )

    return RobustDifferentialInverseKinematicsAction


def _expected_finger_action_class() -> type:
    # See the AppLauncher import boundary documented above.
    from polaris.environments.droid_cfg import (  # noqa: PLC0415
        EefBinaryJointPositionTargetSlewAction,
    )

    return EefBinaryJointPositionTargetSlewAction


def _validate_unmodified_action_classes(arm: Any, finger: Any) -> tuple[type, type]:
    expected_arm = _expected_arm_action_class()
    expected_finger = _expected_finger_action_class()
    if getattr(arm, "class_type", None) is not expected_arm:
        raise ValueError("PolaRiS EEF controller arm action class drift")
    if getattr(finger, "class_type", None) is not expected_finger:
        raise ValueError("PolaRiS EEF controller finger action class drift")
    return expected_arm, expected_finger


def validate_eef_controller_profile_config(
    env_cfg: Any,
    *,
    expected_profile: str,
) -> EefControllerProfileSpec:
    """Validate the complete pre-``gym.make`` controller selection."""

    spec = eef_controller_profile(expected_profile)
    _actions_cfg, arm, finger = _actions(env_cfg)
    expected_arm = _expected_arm_action_class()
    expected_finger = _expected_finger_action_class()
    if getattr(arm, "class_type", None) is not expected_arm:
        raise ValueError("PolaRiS EEF controller arm action class drift")
    actual = {
        "failure_substep_trace": _config_bool(
            getattr(arm, "enable_failure_substep_trace", None),
            field="arm.enable_failure_substep_trace",
        ),
        "wrist_energy_brake": _config_bool(
            getattr(arm, "enable_wrist_energy_brake", None),
            field="arm.enable_wrist_energy_brake",
        ),
        "arm_slew_headroom": _config_bool(
            getattr(arm, "enable_arm_slew_headroom", None),
            field="arm.enable_arm_slew_headroom",
        ),
        "gripper_close_arm_interlock": _config_bool(
            getattr(arm, "enable_gripper_close_arm_interlock", None),
            field="arm.enable_gripper_close_arm_interlock",
        ),
        "arm_release_ramp": _config_bool(
            getattr(arm, "enable_arm_release_ramp", None),
            field="arm.enable_arm_release_ramp",
        ),
        "current_joint_velocity_recovery": _config_bool(
            getattr(arm, "enable_current_joint_velocity_recovery", None),
            field="arm.enable_current_joint_velocity_recovery",
        ),
        "target_slew_rate_0p25": _config_bool(
            getattr(finger, "enable_target_slew_rate_0p25_candidate", None),
            field="finger.enable_target_slew_rate_0p25_candidate",
        ),
    }
    expected = {
        "failure_substep_trace": spec.failure_substep_trace_enabled,
        "wrist_energy_brake": False,
        "arm_slew_headroom": spec.arm_slew_headroom_enabled,
        "gripper_close_arm_interlock": (spec.gripper_close_arm_interlock_enabled),
        "arm_release_ramp": spec.arm_release_ramp_enabled,
        "current_joint_velocity_recovery": (
            spec.current_joint_velocity_recovery_enabled
        ),
        "target_slew_rate_0p25": spec.target_slew_rate_0p25_enabled,
    }
    if actual != expected:
        raise ValueError(
            "PolaRiS EEF controller config/profile mismatch: "
            f"profile={spec.profile!r}, expected={expected!r}, actual={actual!r}"
        )

    target = eef_gripper_target_slew_profile(spec.target_slew_profile)
    if (
        target.close_interlock_profile != spec.close_interlock_profile
        or target.close_interlock_substeps != spec.close_interlock_substeps
        or target.fixed_activation_anchor is not spec.fixed_activation_anchor
    ):
        raise ValueError("PolaRiS EEF controller target/interlock binding drift")

    spawn = getattr(getattr(env_cfg, "scene", None), "robot", None)
    spawn = getattr(spawn, "spawn", None)
    spawn_func = getattr(spawn, "func", None)
    if not callable(spawn_func):
        raise ValueError("PolaRiS EEF robot spawn callable is absent")
    overlay_target = getattr(
        spawn_func, "_eef_mimic_compliance_target_slew_profile", None
    )
    overlay_identity = {
        "module": getattr(spawn_func, "__module__", None),
        "qualname": getattr(spawn_func, "__qualname__", None),
        "name": getattr(spawn_func, "__name__", None),
    }
    if spec.mimic_compliance_profile is None:
        validated_spawn = configure_eef_gripper_mimic_compliance_spawn_overlay(
            spawn,
            target_slew_profile=EEF_GRIPPER_TARGET_SLEW_PROFILE,
        )
        if (
            validated_spawn is not spawn_func
            or overlay_target is not None
            or overlay_identity == EEF_GRIPPER_MIMIC_COMPLIANCE_OVERLAY_IDENTITY
        ):
            raise ValueError(
                "Baseline EEF controller unexpectedly installed an overlay"
            )
    elif (
        overlay_target != spec.target_slew_profile
        or overlay_identity != EEF_GRIPPER_MIMIC_COMPLIANCE_OVERLAY_IDENTITY
        or getattr(spawn_func, "_eef_mimic_compliance_overlay_call_count", None) != 0
        or getattr(
            spawn_func,
            "_eef_mimic_compliance_original_spawn_call_count",
            None,
        )
        != 0
    ):
        raise ValueError("Candidate EEF mimic-compliance overlay config drift")
    finger_class = getattr(finger, "class_type", None)
    trace_profile = getattr(finger_class, "eef_all_six_gripper_trace_profile", None)
    if spec.all_six_gripper_trace_enabled:
        if (
            not isinstance(finger_class, type)
            or finger_class.__bases__ != (expected_finger,)
            or finger_class.__module__ != "polaris.eef_gripper_failure_trace"
            or finger_class.__name__ != EEF_GRIPPER_TARGET_SLEW_ACTION_CLASS
            or finger_class.__qualname__ != EEF_GRIPPER_TARGET_SLEW_ACTION_CLASS
            or trace_profile != EEF_ALL_SIX_GRIPPER_TRACE_PROFILE
        ):
            raise ValueError("Candidate EEF all-six gripper trace config drift")
    elif finger_class is not expected_finger or trace_profile is not None:
        raise ValueError("Baseline EEF controller finger action class drift")
    return spec


def configure_eef_controller_profile(
    env_cfg: Any,
    *,
    profile: str,
) -> EefControllerProfileSpec:
    """Apply the sole accepted candidate before environment construction.

    The default baseline path performs no writes. The candidate first requires
    every opt-in flag to be false, then selects the exact accepted Gate-0 stack
    and installs its pre-articulation mimic-compliance overlay.
    """

    spec = eef_controller_profile(profile)
    _actions_cfg, arm, finger = _actions(env_cfg)
    _validate_unmodified_action_classes(arm, finger)
    spawn_cfg = getattr(
        getattr(getattr(env_cfg, "scene", None), "robot", None), "spawn", None
    )
    original_spawn = configure_eef_gripper_mimic_compliance_spawn_overlay(
        spawn_cfg,
        target_slew_profile=EEF_GRIPPER_TARGET_SLEW_PROFILE,
    )
    if getattr(spawn_cfg, "func", None) is not original_spawn:
        raise ValueError("PolaRiS EEF controller original spawn binding drift")
    candidate_flags = (
        "enable_failure_substep_trace",
        "enable_wrist_energy_brake",
        "enable_arm_slew_headroom",
        "enable_gripper_close_arm_interlock",
        "enable_arm_release_ramp",
        "enable_current_joint_velocity_recovery",
    )
    for field in candidate_flags:
        if _config_bool(getattr(arm, field, None), field=f"arm.{field}"):
            raise ValueError(
                "PolaRiS EEF controller profile requires an unmodified action "
                f"config; {field} was already enabled"
            )
    if _config_bool(
        getattr(finger, "enable_target_slew_rate_0p25_candidate", None),
        field="finger.enable_target_slew_rate_0p25_candidate",
    ):
        raise ValueError(
            "PolaRiS EEF controller profile requires the rate-0.25 flag default-off"
        )

    if spec.profile == EEF_CONTROLLER_BASELINE_PROFILE:
        return validate_eef_controller_profile_config(
            env_cfg, expected_profile=spec.profile
        )

    arm.enable_failure_substep_trace = True
    arm.enable_arm_slew_headroom = True
    arm.enable_gripper_close_arm_interlock = True
    arm.enable_arm_release_ramp = spec.arm_release_ramp_enabled
    arm.enable_current_joint_velocity_recovery = (
        spec.current_joint_velocity_recovery_enabled
    )
    finger.enable_target_slew_rate_0p25_candidate = True
    finger.class_type = make_eef_all_six_gripper_failure_trace_class(finger.class_type)
    configure_eef_gripper_mimic_compliance_spawn_overlay(
        spawn_cfg,
        target_slew_profile=spec.target_slew_profile,
    )
    return validate_eef_controller_profile_config(
        env_cfg, expected_profile=spec.profile
    )


def validate_eef_controller_safety_evidence(
    safety: Mapping[str, Any],
    *,
    expected_profile: str,
    expected_target_slew_profile: str,
) -> EefControllerProfileSpec:
    """Cross-bind a durable all-six safety report to its public profile."""

    spec = eef_controller_profile(expected_profile)
    if expected_target_slew_profile != spec.target_slew_profile:
        raise ValueError(
            "PolaRiS EEF controller/target-slew expectation mismatch: "
            f"controller={spec.profile!r}, target={expected_target_slew_profile!r}"
        )
    static = safety.get("gripper_runtime_static")
    if not isinstance(static, dict):
        if spec.profile == EEF_CONTROLLER_BASELINE_PROFILE:
            return spec
        raise ValueError("PolaRiS EEF controller profile lacks gripper static evidence")
    validated = validate_eef_gripper_static_contract(
        static,
        expected_target_slew_profile=spec.target_slew_profile,
    )
    compliance = validated.get("mimic_compliance")
    if spec.mimic_compliance_profile is None:
        if compliance is not None:
            raise ValueError("Baseline EEF controller has mimic-compliance evidence")
    elif (
        not isinstance(compliance, Mapping)
        or compliance.get("profile") != spec.mimic_compliance_profile
        or compliance.get("enabled") is not True
    ):
        raise ValueError("Candidate EEF mimic-compliance evidence drift")
    return spec


def eef_controller_apply_counts_from_safety(
    safety: Mapping[str, Any],
) -> tuple[int, int]:
    """Return attempted and transactionally committed arm apply counts."""

    counters = safety.get("counters")
    if not isinstance(counters, Mapping):
        raise ValueError("PolaRiS EEF controller safety counters are absent")
    fields = (
        "apply_calls",
        "current_joint_limit_aborts",
        "invariant_aborts",
        "nonfinite_aborts",
    )
    if any(type(counters.get(field)) is not int for field in fields):
        raise ValueError("PolaRiS EEF controller safety counter type drift")
    apply_calls = counters["apply_calls"]
    abort_count = sum(counters[field] for field in fields[1:])
    if (
        apply_calls < 0
        or any(counters[field] < 0 for field in fields[1:])
        or abort_count not in (0, 1)
        or abort_count > apply_calls
    ):
        raise ValueError("PolaRiS EEF controller committed-apply cadence drift")
    return apply_calls, apply_calls - abort_count


def validate_eef_controller_runtime_profile(
    env: Any,
    safety: Mapping[str, Any],
    *,
    expected_profile: str,
    expected_target_slew_profile: str,
) -> EefControllerProfileSpec:
    """Bind live arm/interlock state and durable gripper evidence together."""

    spec = validate_eef_controller_safety_evidence(
        safety,
        expected_profile=expected_profile,
        expected_target_slew_profile=expected_target_slew_profile,
    )
    runtime = getattr(env, "unwrapped", env)
    terms = getattr(getattr(runtime, "action_manager", None), "_terms", None)
    if not isinstance(terms, Mapping) or list(terms) != ["arm", "finger_joint"]:
        raise ValueError("PolaRiS EEF controller runtime action order drift")
    reporter = getattr(terms["arm"], "controller_repair_candidate_report", None)
    if not callable(reporter):
        raise ValueError("PolaRiS EEF controller runtime reporter is absent")
    report = reporter()
    apply_calls, committed_apply_calls = eef_controller_apply_counts_from_safety(safety)
    validate_eef_controller_repair_candidate_report(
        report,
        expected_profile=spec.profile,
        expected_target_slew_profile=spec.target_slew_profile,
        expected_physical_max_delta_joint_pos_rad=safety.get("max_delta_joint_pos_rad"),
        apply_calls=apply_calls,
        committed_apply_calls=committed_apply_calls,
    )
    return spec


def _finite_vector(value: Any, *, field: str) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != 7
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) < 0.0
            for item in value
        )
    ):
        raise ValueError(f"PolaRiS EEF controller report {field} vector drift")
    return [float(item) for item in value]


def _is_exact_float32(value: Any) -> bool:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        return False
    try:
        round_trip = struct.unpack("<f", struct.pack("<f", float(value)))[0]
    except (OverflowError, struct.error):
        return False
    return float(value) == round_trip


def _finite_signed_vector(value: Any, *, field: str) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != 7
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or not _is_exact_float32(item)
            for item in value
        )
    ):
        raise ValueError(f"PolaRiS EEF velocity-recovery {field} vector drift")
    return [float(item) for item in value]


def _finite_hard_limit_matrix(value: Any, *, field: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 7:
        raise ValueError(f"PolaRiS EEF velocity-recovery {field} matrix drift")
    result: list[list[float]] = []
    for row in value:
        if (
            not isinstance(row, list)
            or len(row) != 2
            or any(not _is_exact_float32(item) for item in row)
            or not float(row[0]) < float(row[1])
        ):
            raise ValueError(f"PolaRiS EEF velocity-recovery {field} matrix drift")
        result.append([float(item) for item in row])
    return result


def _float32_round(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def validate_current_joint_velocity_recovery_report(
    value: Any,
    *,
    apply_calls: int,
) -> dict[str, Any]:
    """Validate the complete additive v5 recovery report and event history."""

    if type(apply_calls) is not int or apply_calls < 0:
        raise ValueError("PolaRiS EEF velocity-recovery apply cadence drift")
    if (
        not isinstance(value, dict)
        or set(value) != CURRENT_JOINT_VELOCITY_RECOVERY_FIELDS
    ):
        raise ValueError("PolaRiS EEF velocity-recovery report schema drift")
    contract = value.get("contract")
    if (
        not isinstance(contract, dict)
        or set(contract) != CURRENT_JOINT_VELOCITY_RECOVERY_CONTRACT_FIELDS
    ):
        raise ValueError("PolaRiS EEF velocity-recovery contract schema drift")
    exact_contract = {
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
        "release_ramp_profile": ARM_RELEASE_RAMP_PROFILE,
        "transaction_profile": CURRENT_JOINT_VELOCITY_RECOVERY_TRANSACTION_PROFILE,
        "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
        "physics_dt_float32": PANDA_EEF_PHYSICS_DT_FLOAT32,
    }
    if any(
        contract.get(field) != expected for field, expected in exact_contract.items()
    ):
        raise ValueError("PolaRiS EEF velocity-recovery contract identity drift")
    limits = _finite_vector(
        contract.get("velocity_limits_rad_s"), field="velocity limits"
    )
    envelopes = _finite_vector(
        contract.get("velocity_envelopes_rad_s"), field="velocity envelopes"
    )
    expected_limits = [
        struct.unpack("<f", struct.pack("<f", float(value)))[0]
        for value in PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S
    ]
    expected_envelopes = [
        current_joint_velocity_recovery_envelope(limit) for limit in expected_limits
    ]
    hard_limits = _finite_hard_limit_matrix(
        contract.get("hard_joint_position_limits_rad"),
        field="hard limits",
    )
    expected_hard_limits = [
        [float(lower), float(upper)]
        for lower, upper in PANDA_PHYSX_HARD_JOINT_POS_LIMITS_RAD
    ]
    hard_limit_digest = hashlib.sha256(
        b"".join(struct.pack("<f", item) for row in hard_limits for item in row)
    ).hexdigest()
    if (
        limits != expected_limits
        or envelopes != expected_envelopes
        or hard_limits != expected_hard_limits
        or contract.get("hard_joint_position_limits_little_endian_float32_sha256")
        != hard_limit_digest
        or hard_limit_digest != PANDA_PHYSX_HARD_JOINT_POS_LIMITS_FLOAT32_SHA256
    ):
        raise ValueError("PolaRiS EEF velocity-recovery envelope binding drift")

    state = value.get("state")
    if (
        not isinstance(state, dict)
        or set(state) != CURRENT_JOINT_VELOCITY_RECOVERY_STATE_FIELDS
    ):
        raise ValueError("PolaRiS EEF velocity-recovery state schema drift")
    phase = state.get("phase")
    active = state.get("active")
    consecutive = state.get("consecutive_active_substeps")
    clean = state.get("consecutive_clean_samples")
    next_index = state.get("release_ramp_next_index")
    if (
        phase
        not in (
            CURRENT_JOINT_VELOCITY_RECOVERY_PHASE_INACTIVE,
            CURRENT_JOINT_VELOCITY_RECOVERY_PHASE_HOLD,
            CURRENT_JOINT_VELOCITY_RECOVERY_PHASE_RELEASE_RAMP,
        )
        or type(active) is not bool
        or active is (phase == CURRENT_JOINT_VELOCITY_RECOVERY_PHASE_INACTIVE)
        or type(consecutive) is not int
        or not 0
        <= consecutive
        <= CURRENT_JOINT_VELOCITY_RECOVERY_MAXIMUM_ACTIVE_SUBSTEPS
        or type(clean) is not int
        or not 0 <= clean < CURRENT_JOINT_VELOCITY_RECOVERY_CLEAN_SAMPLES_REQUIRED
        or (phase == CURRENT_JOINT_VELOCITY_RECOVERY_PHASE_HOLD)
        is not (consecutive > 0)
        or (phase == CURRENT_JOINT_VELOCITY_RECOVERY_PHASE_RELEASE_RAMP)
        is (next_index is None)
        or (
            next_index is not None
            and (
                type(next_index) is not int
                or not 0 <= next_index < ARM_RELEASE_RAMP_SUBSTEPS
            )
        )
    ):
        raise ValueError("PolaRiS EEF velocity-recovery lifecycle drift")

    counters = value.get("counters")
    if (
        not isinstance(counters, dict)
        or set(counters) != CURRENT_JOINT_VELOCITY_RECOVERY_COUNTER_FIELDS
        or any(type(item) is not int or item < 0 for item in counters.values())
    ):
        raise ValueError("PolaRiS EEF velocity-recovery counter schema drift")
    if (
        counters["residual_events"] > apply_calls
        or not counters["residual_events"]
        <= counters["residual_joints"]
        <= 7 * counters["residual_events"]
        or counters["recovery_events"] > apply_calls
        or counters["recovery_active_substeps"] != counters["hold_target_applies"]
        or counters["recovery_active_substeps"] > apply_calls
        or counters["release_ramp_target_applies"] > apply_calls
        or counters["release_ramp_target_applies"]
        < ARM_RELEASE_RAMP_SUBSTEPS * counters["recovered_events"]
        or counters["recovered_events"] > counters["recovery_events"]
        or counters["sustained_aborts"]
        + counters["current_hard_limit_aborts"]
        + counters["predicted_limit_aborts"]
        + counters["transaction_aborts"]
        + counters["lower_endpoint_transition_aborts"]
        > counters["recovery_events"]
    ):
        raise ValueError("PolaRiS EEF velocity-recovery counter history drift")
    maxima = value.get("maxima")
    if (
        not isinstance(maxima, dict)
        or set(maxima) != CURRENT_JOINT_VELOCITY_RECOVERY_MAXIMA_FIELDS
        or isinstance(maxima.get("abs_velocity_to_limit_ratio"), bool)
        or not isinstance(maxima.get("abs_velocity_to_limit_ratio"), (int, float))
        or not math.isfinite(float(maxima["abs_velocity_to_limit_ratio"]))
        or not _is_exact_float32(maxima["abs_velocity_to_limit_ratio"])
        or float(maxima["abs_velocity_to_limit_ratio"]) < 0.0
        or type(maxima.get("consecutive_recovery_substeps")) is not int
        or not 0
        <= maxima["consecutive_recovery_substeps"]
        <= CURRENT_JOINT_VELOCITY_RECOVERY_MAXIMUM_ACTIVE_SUBSTEPS
    ):
        raise ValueError("PolaRiS EEF velocity-recovery maxima schema drift")
    residual_maxima = _finite_vector(
        maxima.get("abs_velocity_residual_excess_rad_s"),
        field="residual maximum",
    )
    if (
        any(not _is_exact_float32(item) for item in residual_maxima)
        or maxima["consecutive_recovery_substeps"] < consecutive
        or (counters["residual_events"] == 0)
        is not (
            float(maxima["abs_velocity_to_limit_ratio"]) <= 1.0
            and all(item == 0.0 for item in residual_maxima)
        )
    ):
        raise ValueError("PolaRiS EEF velocity-recovery maxima history drift")

    events = value.get("events")
    if not isinstance(events, list) or len(events) != counters["recovery_events"]:
        raise ValueError("PolaRiS EEF velocity-recovery event count drift")
    active_event_count = 0
    previous_end_index: int | None = None
    end_reason_counts = {
        reason: 0 for reason in CURRENT_JOINT_VELOCITY_RECOVERY_END_REASONS
    }
    observed_event_max_ratio = 0.0
    observed_event_max_excess = [0.0] * 7
    for event_index, event in enumerate(events):
        if (
            not isinstance(event, dict)
            or set(event) != CURRENT_JOINT_VELOCITY_RECOVERY_EVENT_FIELDS
        ):
            raise ValueError("PolaRiS EEF velocity-recovery event schema drift")
        if (
            event.get("event_index") != event_index
            or type(event.get("start_apply_index")) is not int
            or not 0 <= event["start_apply_index"] < apply_calls
            or event.get("start_reason")
            not in CURRENT_JOINT_VELOCITY_RECOVERY_START_REASONS
            or (
                previous_end_index is not None
                and event["start_apply_index"] <= previous_end_index
            )
        ):
            raise ValueError("PolaRiS EEF velocity-recovery event identity drift")
        end_reason = event.get("end_reason")
        end_index = event.get("end_apply_index")
        if end_reason is None:
            active_event_count += 1
            if end_index is not None or event_index != len(events) - 1:
                raise ValueError("PolaRiS EEF active recovery event has an end index")
        elif (
            end_reason not in CURRENT_JOINT_VELOCITY_RECOVERY_END_REASONS
            or type(end_index) is not int
            or not event["start_apply_index"] <= end_index < apply_calls
        ):
            raise ValueError("PolaRiS EEF recovery event termination drift")
        else:
            end_reason_counts[end_reason] += 1
            previous_end_index = end_index
        snapshot_facts: dict[str, dict[str, Any]] = {}
        for snapshot_field in ("start", "last"):
            snapshot = event.get(snapshot_field)
            if (
                not isinstance(snapshot, dict)
                or set(snapshot) != CURRENT_JOINT_VELOCITY_RECOVERY_SNAPSHOT_FIELDS
                or type(snapshot.get("apply_index")) is not int
                or not 0 <= snapshot["apply_index"] < apply_calls
                or snapshot.get("policy_step") != snapshot["apply_index"] // 8
                or snapshot.get("physics_substep") != snapshot["apply_index"] % 8
            ):
                raise ValueError("PolaRiS EEF recovery event snapshot identity drift")
            joint_velocity = _finite_signed_vector(
                snapshot.get("joint_velocity_rad_s"), field="joint_velocity_rad_s"
            )
            joint_position = _finite_signed_vector(
                snapshot.get("joint_pos_rad"), field="joint_pos_rad"
            )
            snapshot_limits = _finite_signed_vector(
                snapshot.get("joint_velocity_limit_rad_s"),
                field="joint_velocity_limit_rad_s",
            )
            snapshot_envelopes = _finite_signed_vector(
                snapshot.get("joint_velocity_envelope_rad_s"),
                field="joint_velocity_envelope_rad_s",
            )
            recorded_excess = _finite_signed_vector(
                snapshot.get("joint_velocity_limit_excess_rad_s"),
                field="joint_velocity_limit_excess_rad_s",
            )
            recorded_ratio = _finite_signed_vector(
                snapshot.get("velocity_to_limit_ratio"),
                field="velocity_to_limit_ratio",
            )
            recorded_predicted_position = _finite_signed_vector(
                snapshot.get("predicted_joint_pos_rad"),
                field="predicted_joint_pos_rad",
            )
            recorded_predicted_clearance = _finite_signed_vector(
                snapshot.get("predicted_hard_limit_clearance_rad"),
                field="predicted_hard_limit_clearance_rad",
            )
            expected_excess = [
                max(
                    struct.unpack("<f", struct.pack("<f", abs(velocity) - limit))[0],
                    0.0,
                )
                for velocity, limit in zip(joint_velocity, snapshot_limits, strict=True)
            ]
            expected_ratio = [
                _float32_round(abs(velocity) / limit)
                for velocity, limit in zip(joint_velocity, snapshot_limits, strict=True)
            ]
            expected_predicted_position = [
                _float32_round(
                    position + _float32_round(velocity * PANDA_EEF_PHYSICS_DT_FLOAT32)
                )
                for position, velocity in zip(
                    joint_position,
                    joint_velocity,
                    strict=True,
                )
            ]
            expected_predicted_clearance = [
                min(
                    _float32_round(predicted - lower),
                    _float32_round(upper - predicted),
                )
                for predicted, (lower, upper) in zip(
                    expected_predicted_position,
                    hard_limits,
                    strict=True,
                )
            ]
            if (
                any(item < 0.0 for item in recorded_excess)
                or any(item < 0.0 for item in recorded_ratio)
                or recorded_excess != expected_excess
                or recorded_ratio != expected_ratio
                or recorded_predicted_position != expected_predicted_position
                or recorded_predicted_clearance != expected_predicted_clearance
            ):
                raise ValueError("PolaRiS EEF recovery snapshot numeric binding drift")
            observed_event_max_ratio = max(
                observed_event_max_ratio, max(recorded_ratio)
            )
            observed_event_max_excess = [
                max(previous, current)
                for previous, current in zip(
                    observed_event_max_excess,
                    recorded_excess,
                    strict=True,
                )
            ]
            hold_target = snapshot.get("hold_target_rad")
            transaction_readbacks = [
                snapshot.get(field)
                for field in (
                    "hold_position_target_readback_rad",
                    "hold_velocity_target_readback_rad_s",
                    "hold_effort_target_readback_nm",
                )
            ]
            # A pre-transaction terminal snapshot may know the target while
            # carrying no readback: the setter/readback failure is precisely
            # why no successful transaction evidence exists.  Once any
            # readback is present, however, all three target surfaces and the
            # target they bind must be closed together.
            if any(item is not None for item in transaction_readbacks) and (
                hold_target is None
                or any(item is None for item in transaction_readbacks)
            ):
                raise ValueError("PolaRiS EEF recovery transaction snapshot split")
            for item in (hold_target, *transaction_readbacks):
                if item is not None:
                    _finite_signed_vector(item, field="transaction readback")
            if all(item is not None for item in transaction_readbacks):
                if (
                    transaction_readbacks[0] != hold_target
                    or any(value != 0.0 for value in transaction_readbacks[1])
                    or any(value != 0.0 for value in transaction_readbacks[2])
                ):
                    raise ValueError(
                        "PolaRiS EEF recovery transaction readback binding drift"
                    )
            if snapshot_limits != limits or snapshot_envelopes != envelopes:
                raise ValueError("PolaRiS EEF recovery snapshot static binding drift")
            current_hard_violation = any(
                position < lower or position > upper
                for position, (lower, upper) in zip(
                    joint_position,
                    hard_limits,
                    strict=True,
                )
            )
            predicted_hard_crossing = any(
                clearance < 0.0 for clearance in expected_predicted_clearance
            )
            velocity_over_envelope = any(
                abs(velocity) > envelope
                for velocity, envelope in zip(
                    joint_velocity,
                    envelopes,
                    strict=True,
                )
            )
            snapshot_facts[snapshot_field] = {
                "current_hard_violation": current_hard_violation,
                "predicted_hard_crossing": predicted_hard_crossing,
                "velocity_over_envelope": velocity_over_envelope,
                "hold_target_present": hold_target is not None,
                "readbacks_present": all(
                    item is not None for item in transaction_readbacks
                ),
            }
        start_snapshot = event["start"]
        last_snapshot = event["last"]
        if (
            start_snapshot["apply_index"] != event["start_apply_index"]
            or last_snapshot["apply_index"] < start_snapshot["apply_index"]
            or (
                end_reason is not None
                and last_snapshot["apply_index"] != event["end_apply_index"]
            )
        ):
            raise ValueError("PolaRiS EEF recovery event snapshot cadence drift")
        start_reason = event["start_reason"]
        start_facts = snapshot_facts["start"]
        last_facts = snapshot_facts["last"]
        start_reason_valid = {
            "measured_velocity_above_float32_envelope": (
                start_facts["velocity_over_envelope"]
                and not start_facts["current_hard_violation"]
            ),
            "current_hard_limit_violation": start_facts["current_hard_violation"],
            "predicted_hard_limit_crossing": (
                not start_facts["current_hard_violation"]
                and start_facts["predicted_hard_crossing"]
                and not start_facts["velocity_over_envelope"]
            ),
            "target_transaction_failure": (
                not start_facts["current_hard_violation"]
                and not start_facts["predicted_hard_crossing"]
                and start_facts["velocity_over_envelope"]
                and start_facts["hold_target_present"]
                and not start_facts["readbacks_present"]
                and end_reason == "transaction_abort"
                and end_index == event["start_apply_index"]
            ),
        }[start_reason]
        end_reason_valid = {
            None: (
                not last_facts["current_hard_violation"]
                and not last_facts["predicted_hard_crossing"]
            ),
            "clean2_release_ramp_complete": (
                not last_facts["current_hard_violation"]
                and not last_facts["predicted_hard_crossing"]
                and not last_facts["velocity_over_envelope"]
            ),
            "sustained_recovery_abort": (
                not last_facts["current_hard_violation"]
                and not last_facts["predicted_hard_crossing"]
            ),
            "current_hard_limit_abort": last_facts["current_hard_violation"],
            "predicted_hard_limit_abort": (
                not last_facts["current_hard_violation"]
                and last_facts["predicted_hard_crossing"]
            ),
            "transaction_abort": (
                not last_facts["current_hard_violation"]
                and not last_facts["predicted_hard_crossing"]
                and last_facts["hold_target_present"]
                and not last_facts["readbacks_present"]
            ),
            "lower_endpoint_transition_overflow_abort": (
                not last_facts["current_hard_violation"]
                and not last_facts["predicted_hard_crossing"]
            ),
        }[end_reason]
        if not start_reason_valid or not end_reason_valid:
            raise ValueError("PolaRiS EEF recovery event reason predicate drift")
        if end_reason in (
            "sustained_recovery_abort",
            "current_hard_limit_abort",
            "predicted_hard_limit_abort",
            "transaction_abort",
            "lower_endpoint_transition_overflow_abort",
        ):
            terminal_readbacks = [
                last_snapshot[field]
                for field in (
                    "hold_position_target_readback_rad",
                    "hold_velocity_target_readback_rad_s",
                    "hold_effort_target_readback_nm",
                )
            ]
            if any(item is not None for item in terminal_readbacks) or (
                end_reason != "transaction_abort"
                and last_snapshot["hold_target_rad"] is not None
            ):
                raise ValueError("PolaRiS EEF terminal recovery retained readbacks")
        elif end_reason == "clean2_release_ramp_complete" or end_reason is None:
            last_readbacks = [
                last_snapshot[field]
                for field in (
                    "hold_position_target_readback_rad",
                    "hold_velocity_target_readback_rad_s",
                    "hold_effort_target_readback_nm",
                )
            ]
            if any(item is None for item in last_readbacks):
                raise ValueError("PolaRiS EEF live/completed recovery lacks readbacks")
    if active_event_count != int(active):
        raise ValueError("PolaRiS EEF velocity-recovery active event drift")
    expected_end_counts = {
        "clean2_release_ramp_complete": counters["recovered_events"],
        "sustained_recovery_abort": counters["sustained_aborts"],
        "current_hard_limit_abort": counters["current_hard_limit_aborts"],
        "predicted_hard_limit_abort": counters["predicted_limit_aborts"],
        "transaction_abort": counters["transaction_aborts"],
        "lower_endpoint_transition_overflow_abort": counters[
            "lower_endpoint_transition_aborts"
        ],
    }
    if any(
        end_reason_counts[reason] != expected
        for reason, expected in expected_end_counts.items()
    ) or (
        float(maxima["abs_velocity_to_limit_ratio"]) < observed_event_max_ratio
        or any(
            maximum < observed
            for maximum, observed in zip(
                residual_maxima, observed_event_max_excess, strict=True
            )
        )
    ):
        raise ValueError("PolaRiS EEF velocity-recovery event history drift")
    return dict(value)


def validate_eef_controller_repair_candidate_report(
    report: Any,
    *,
    expected_profile: str,
    expected_target_slew_profile: str,
    expected_physical_max_delta_joint_pos_rad: Any,
    apply_calls: int,
    committed_apply_calls: int | None = None,
    require_initial_state: bool = False,
) -> dict[str, Any]:
    """Independently validate the accepted live arm/interlock report."""

    spec = eef_controller_profile(expected_profile)
    if expected_target_slew_profile != spec.target_slew_profile:
        raise ValueError("PolaRiS EEF controller report target-slew binding drift")
    if type(apply_calls) is not int or apply_calls < 0:
        raise ValueError("PolaRiS EEF controller report apply cadence drift")
    if committed_apply_calls is None:
        committed_apply_calls = apply_calls
    if (
        type(committed_apply_calls) is not int
        or not 0 <= committed_apply_calls <= apply_calls
        or apply_calls - committed_apply_calls not in (0, 1)
    ):
        raise ValueError("PolaRiS EEF controller committed-apply cadence drift")
    expected_report_fields = {
        "arm_slew_headroom",
        "gripper_close_arm_interlock",
    }
    if spec.arm_release_ramp_enabled:
        expected_report_fields.add("arm_release_ramp")
    if spec.current_joint_velocity_recovery_enabled:
        expected_report_fields.add("current_joint_velocity_recovery")
    if not isinstance(report, dict) or set(report) != expected_report_fields:
        raise ValueError("PolaRiS EEF controller runtime report schema drift")
    arm = report.get("arm_slew_headroom")
    interlock = report.get("gripper_close_arm_interlock")
    if (
        not isinstance(arm, dict)
        or set(arm) != ARM_SLEW_HEADROOM_REPORT_FIELDS
        or arm.get("enabled") is not spec.arm_slew_headroom_enabled
        or arm.get("profile") != ARM_SLEW_HEADROOM_CANDIDATE_PROFILE
        or isinstance(arm.get("ratio"), bool)
        or not isinstance(arm.get("ratio"), (int, float))
        or not math.isfinite(float(arm["ratio"]))
        or not math.isclose(
            float(arm["ratio"]),
            ARM_SLEW_HEADROOM_RATIO,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ValueError("PolaRiS EEF controller arm-slew runtime drift")
    physical = _finite_vector(
        arm.get("physical_max_delta_joint_pos_rad"), field="physical arm slew"
    )
    expected_physical = _finite_vector(
        expected_physical_max_delta_joint_pos_rad,
        field="expected physical arm slew",
    )
    nominal = _finite_vector(
        arm.get("nominal_max_delta_joint_pos_rad"), field="nominal arm slew"
    )
    if physical != expected_physical:
        raise ValueError("PolaRiS EEF controller physical arm-slew evidence drift")
    nominal_ratio = ARM_SLEW_HEADROOM_RATIO if spec.arm_slew_headroom_enabled else 1.0
    if any(
        not math.isclose(
            inner,
            outer * nominal_ratio,
            rel_tol=2e-7,
            abs_tol=0.0,
        )
        for outer, inner in zip(physical, nominal, strict=True)
    ):
        raise ValueError("PolaRiS EEF controller nominal arm-slew evidence drift")
    if (
        not isinstance(interlock, dict)
        or set(interlock) != GRIPPER_CLOSE_ARM_INTERLOCK_REPORT_FIELDS
        or interlock.get("enabled") is not spec.gripper_close_arm_interlock_enabled
        or interlock.get("profile") != spec.close_interlock_profile
        or interlock.get("configured_substeps") != spec.close_interlock_substeps
    ):
        raise ValueError("PolaRiS EEF controller interlock runtime drift")
    for field in GRIPPER_CLOSE_ARM_INTERLOCK_COUNTER_FIELDS:
        if type(interlock.get(field)) is not int or interlock[field] < 0:
            raise ValueError(f"PolaRiS EEF controller interlock {field} drift")
    if (
        type(interlock.get("endpoint_observed")) is not bool
        or type(interlock.get("anchor_valid")) is not bool
    ):
        raise ValueError("PolaRiS EEF controller interlock boolean drift")
    active = _finite_vector(
        interlock.get("max_abs_active_delta_joint_pos_rad"),
        field="active interlock delta",
    )
    released = _finite_vector(
        interlock.get("max_abs_released_delta_joint_pos_rad"),
        field="released interlock delta",
    )
    current_residual = _finite_vector(
        interlock.get("max_abs_current_anchor_residual_rad"),
        field="current anchor residual",
    )
    target_residual = _finite_vector(
        interlock.get("max_abs_target_anchor_residual_rad"),
        field="target anchor residual",
    )
    if (
        any(delta > bound + 1e-6 for delta, bound in zip(active, nominal, strict=True))
        or any(
            delta > bound + 1e-6 for delta, bound in zip(released, nominal, strict=True)
        )
        or any(
            target > current + 1e-6
            for target, current in zip(target_residual, current_residual, strict=True)
        )
    ):
        raise ValueError("PolaRiS EEF controller fixed-anchor bound drift")

    remaining = interlock["remaining_substeps"]
    activation_count = interlock["activation_count"]
    active_count = interlock["active_apply_count"]
    capture_count = interlock["anchor_capture_count"]
    target_count = interlock["anchor_target_apply_count"]
    first_exact_count = interlock["anchor_first_exact_target_count"]
    completion_count = interlock["anchor_completion_count"]
    cancel_count = interlock["anchor_open_cancel_count"]
    anchor_valid = interlock["anchor_valid"]
    if (
        remaining >= spec.close_interlock_substeps
        or active_count < activation_count
        or active_count > activation_count * spec.close_interlock_substeps
        or active_count + interlock["released_apply_count"] > committed_apply_calls
        or interlock["observed_endpoint_change_count"] > committed_apply_calls
        or activation_count > interlock["observed_endpoint_change_count"] + 1
        or (
            spec.gripper_close_arm_interlock_enabled
            and interlock["endpoint_observed"] is not (committed_apply_calls > 0)
        )
        or capture_count != activation_count
        or capture_count > target_count
        or capture_count > active_count
        or target_count != active_count
        or first_exact_count != capture_count
        or interlock["anchor_refresh_count"] != 0
        or anchor_valid != (remaining > 0)
        or completion_count + cancel_count + int(anchor_valid) != capture_count
    ):
        raise ValueError("PolaRiS EEF controller fixed-anchor lifecycle drift")
    current_active_count = (
        spec.close_interlock_substeps - remaining if anchor_valid else 0
    )
    minimum_active_count = (
        spec.close_interlock_substeps * completion_count
        + cancel_count
        + current_active_count
    )
    maximum_active_count = (
        spec.close_interlock_substeps * completion_count
        + (spec.close_interlock_substeps - 1) * cancel_count
        + current_active_count
    )
    if not minimum_active_count <= active_count <= maximum_active_count:
        raise ValueError("PolaRiS EEF controller fixed-anchor countdown drift")
    for prefix in ("slew", "position"):
        events = interlock[f"anchor_{prefix}_limit_event_count"]
        joints = interlock[f"anchor_{prefix}_limited_joint_count"]
        if not events <= joints <= 7 * events or events > target_count:
            raise ValueError(
                f"PolaRiS EEF controller anchor {prefix}-limit counter drift"
            )

    last_index = interlock.get("last_activation_apply_index")
    last_anchor = interlock.get("last_anchor_joint_pos_rad")
    last_digest = interlock.get("last_anchor_little_endian_float32_sha256")
    if capture_count == 0:
        if last_index is not None or last_anchor is not None or last_digest is not None:
            raise ValueError("PolaRiS EEF controller inactive anchor identity drift")
    else:
        if type(last_index) is not int or not 0 <= last_index < committed_apply_calls:
            raise ValueError("PolaRiS EEF controller anchor activation index drift")
        if (
            not isinstance(last_anchor, list)
            or len(last_anchor) != 7
            or any(not _is_exact_float32(value) for value in last_anchor)
        ):
            raise ValueError("PolaRiS EEF controller last anchor vector drift")
        if (
            type(last_digest) is not str
            or len(last_digest) != 64
            or any(character not in "0123456789abcdef" for character in last_digest)
        ):
            raise ValueError("PolaRiS EEF controller last anchor digest drift")
        computed = hashlib.sha256(
            struct.pack("<7f", *(float(value) for value in last_anchor))
        ).hexdigest()
        if last_digest != computed:
            raise ValueError("PolaRiS EEF controller last anchor digest mismatch")

    zero_vectors = all(
        value == 0.0
        for vector in (active, released, current_residual, target_residual)
        for value in vector
    )
    if not spec.gripper_close_arm_interlock_enabled:
        if (
            any(
                interlock[field] != 0
                for field in GRIPPER_CLOSE_ARM_INTERLOCK_COUNTER_FIELDS
            )
            or interlock["endpoint_observed"] is not False
            or interlock["anchor_valid"] is not False
            or not zero_vectors
        ):
            raise ValueError("Disabled PolaRiS EEF interlock retained evidence")
    elif activation_count == 0 and (
        active_count != 0
        or remaining != 0
        or not zero_vectors
        or interlock["released_apply_count"] != 0
    ):
        raise ValueError("Inactive PolaRiS EEF fixed anchor retained evidence")

    ramp = report.get("arm_release_ramp")
    if spec.arm_release_ramp_enabled:
        if (
            not isinstance(ramp, dict)
            or set(ramp) != ARM_RELEASE_RAMP_REPORT_FIELDS
            or ramp.get("enabled") is not True
            or ramp.get("profile") != ARM_RELEASE_RAMP_PROFILE
            or ramp.get("state_profile") != ARM_RELEASE_RAMP_STATE_PROFILE
            or ramp.get("substeps") != ARM_RELEASE_RAMP_SUBSTEPS
            or ramp.get("fraction_profile") != ARM_RELEASE_RAMP_FRACTION_PROFILE
            or ramp.get("formula_profile") != ARM_RELEASE_RAMP_FORMULA_PROFILE
            or ramp.get("transaction_profile") != ARM_RELEASE_RAMP_TRANSACTION_PROFILE
            or ramp.get("open_during_ramp_policy")
            != "continue_current_ramp_without_restart_or_skip_v1"
            or type(ramp.get("gripper_target_or_state_write_count")) is not int
            or ramp.get("gripper_target_or_state_write_count") != 0
        ):
            raise ValueError("PolaRiS EEF controller release-ramp identity drift")
        fractions = ramp.get("fractions_float32")
        expected_fractions = [
            arm_release_ramp_fraction(index)
            for index in range(ARM_RELEASE_RAMP_SUBSTEPS)
        ]
        if (
            not isinstance(fractions, list)
            or len(fractions) != ARM_RELEASE_RAMP_SUBSTEPS
            or fractions != expected_fractions
            or any(not _is_exact_float32(value) for value in fractions)
        ):
            raise ValueError("PolaRiS EEF controller release-ramp fraction drift")
        phase = ramp.get("phase")
        next_index = ramp.get("next_index")
        if (
            phase
            not in (
                ARM_RELEASE_PHASE_HOLD,
                ARM_RELEASE_PHASE_RAMP,
                ARM_RELEASE_PHASE_RELEASE,
            )
            or (phase == ARM_RELEASE_PHASE_RAMP) is (next_index is None)
            or (
                next_index is not None
                and (
                    type(next_index) is not int
                    or not 0 <= next_index < ARM_RELEASE_RAMP_SUBSTEPS
                )
            )
            or (phase == ARM_RELEASE_PHASE_HOLD) is not (remaining > 0)
        ):
            raise ValueError("PolaRiS EEF controller release-ramp phase drift")
        for field in ARM_RELEASE_RAMP_COUNTER_FIELDS:
            if type(ramp.get(field)) is not int or ramp[field] < 0:
                raise ValueError(f"PolaRiS EEF controller release-ramp {field} drift")
        active_ramp_count = int(phase == ARM_RELEASE_PHASE_RAMP)
        current_ramp_target_count = next_index if next_index is not None else 0
        if (
            ramp["release_observed_count"] != ramp["ramp_started_count"]
            or ramp["ramp_started_count"]
            != ramp["ramp_completed_count"]
            + ramp["ramp_cancelled_by_reactivation_count"]
            + active_ramp_count
            or ramp["release_observed_count"] != completion_count + cancel_count
            or ramp["ramp_target_apply_count"]
            != ramp["ramp_completed_count"] * ARM_RELEASE_RAMP_SUBSTEPS
            + ramp["cancelled_ramp_target_apply_count"]
            + current_ramp_target_count
            or ramp["cancelled_ramp_target_apply_count"]
            > ramp["ramp_cancelled_by_reactivation_count"]
            * (ARM_RELEASE_RAMP_SUBSTEPS - 1)
            or ramp["ramp_target_apply_count"] > committed_apply_calls
            or not (
                0
                <= ramp["ramp_limited_target_apply_count"]
                <= ramp["ramp_target_apply_count"] - ramp["ramp_completed_count"]
            )
            or not (
                ramp["ramp_limited_target_apply_count"]
                <= ramp["ramp_limited_joint_target_count"]
                <= 7 * ramp["ramp_limited_target_apply_count"]
            )
        ):
            raise ValueError("PolaRiS EEF controller release-ramp lifecycle drift")
        last_target_apply = ramp.get("last_target_apply_index")
        last_ramp_index = ramp.get("last_ramp_index")
        if (
            (last_target_apply is None) != (last_ramp_index is None)
            or (ramp["ramp_target_apply_count"] == 0) != (last_target_apply is None)
            or (
                last_target_apply is not None
                and (
                    type(last_target_apply) is not int
                    or not 0 <= last_target_apply < committed_apply_calls
                    or type(last_ramp_index) is not int
                    or not 0 <= last_ramp_index < ARM_RELEASE_RAMP_SUBSTEPS
                )
            )
        ):
            raise ValueError("PolaRiS EEF controller release-ramp last-target drift")
        if (
            phase == ARM_RELEASE_PHASE_RAMP
            and next_index is not None
            and next_index > 0
            and (
                last_ramp_index != next_index - 1
                or last_target_apply != committed_apply_calls - 1
            )
        ) or (
            phase == ARM_RELEASE_PHASE_RELEASE
            and ramp["ramp_target_apply_count"] > 0
            and last_ramp_index != ARM_RELEASE_RAMP_SUBSTEPS - 1
        ):
            raise ValueError("PolaRiS EEF controller release-ramp phase/target drift")
        maximum_change = _finite_vector(
            ramp.get("max_abs_nominal_to_ramped_target_change_rad"),
            field="release-ramp maximum target change",
        )
        if any(
            change > bound + 1e-6
            for change, bound in zip(maximum_change, nominal, strict=True)
        ):
            raise ValueError("PolaRiS EEF controller release-ramp maximum drift")
        has_positive_maximum = any(change > 0.0 for change in maximum_change)
        if has_positive_maximum is not (ramp["ramp_limited_target_apply_count"] > 0):
            raise ValueError(
                "PolaRiS EEF controller release-ramp limited/maximum drift"
            )
    elif ramp is not None:
        raise ValueError("Non-ramp PolaRiS EEF controller has release-ramp evidence")
    recovery = report.get("current_joint_velocity_recovery")
    if spec.current_joint_velocity_recovery_enabled:
        validate_current_joint_velocity_recovery_report(
            recovery,
            apply_calls=apply_calls,
        )
    elif recovery is not None:
        raise ValueError("Non-v5 PolaRiS EEF controller has velocity-recovery evidence")
    if require_initial_state and (
        apply_calls != 0
        or committed_apply_calls != 0
        or any(
            interlock[field] != 0
            for field in GRIPPER_CLOSE_ARM_INTERLOCK_COUNTER_FIELDS
        )
        or interlock["endpoint_observed"] is not False
        or interlock["anchor_valid"] is not False
        or not zero_vectors
        or (
            spec.arm_release_ramp_enabled
            and (
                ramp["phase"] != ARM_RELEASE_PHASE_RELEASE
                or ramp["next_index"] is not None
                or any(ramp[field] != 0 for field in ARM_RELEASE_RAMP_COUNTER_FIELDS)
                or ramp["last_target_apply_index"] is not None
                or ramp["last_ramp_index"] is not None
                or any(
                    value != 0.0
                    for value in ramp["max_abs_nominal_to_ramped_target_change_rad"]
                )
            )
        )
        or (
            spec.current_joint_velocity_recovery_enabled
            and (
                recovery["state"]
                != {
                    "phase": CURRENT_JOINT_VELOCITY_RECOVERY_PHASE_INACTIVE,
                    "active": False,
                    "consecutive_active_substeps": 0,
                    "consecutive_clean_samples": 0,
                    "release_ramp_next_index": None,
                }
                or any(recovery["counters"].values())
                or recovery["events"]
            )
        )
    ):
        raise ValueError("Initial PolaRiS EEF controller report is not empty")
    return dict(report)


def capture_eef_controller_repair_candidate_report(
    env: Any,
    safety: Mapping[str, Any],
    *,
    expected_profile: str,
    expected_target_slew_profile: str,
    require_initial_state: bool = False,
) -> dict[str, Any]:
    """Capture and independently close one live report for durable artifacts."""

    validate_eef_controller_safety_evidence(
        safety,
        expected_profile=expected_profile,
        expected_target_slew_profile=expected_target_slew_profile,
    )
    runtime = getattr(env, "unwrapped", env)
    terms = getattr(getattr(runtime, "action_manager", None), "_terms", None)
    if not isinstance(terms, Mapping) or list(terms) != ["arm", "finger_joint"]:
        raise ValueError("PolaRiS EEF controller runtime action order drift")
    reporter = getattr(terms["arm"], "controller_repair_candidate_report", None)
    if not callable(reporter):
        raise ValueError("PolaRiS EEF controller runtime reporter is absent")
    apply_calls, committed_apply_calls = eef_controller_apply_counts_from_safety(safety)
    return validate_eef_controller_repair_candidate_report(
        reporter(),
        expected_profile=expected_profile,
        expected_target_slew_profile=expected_target_slew_profile,
        expected_physical_max_delta_joint_pos_rad=safety.get("max_delta_joint_pos_rad"),
        apply_calls=apply_calls,
        committed_apply_calls=committed_apply_calls,
        require_initial_state=require_initial_state,
    )
