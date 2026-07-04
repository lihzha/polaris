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


@dataclass(frozen=True)
class EefControllerProfileSpec:
    """Exact component identities selected by one public controller profile."""

    profile: str
    failure_substep_trace_enabled: bool
    all_six_gripper_trace_enabled: bool
    arm_slew_headroom_enabled: bool
    gripper_close_arm_interlock_enabled: bool
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
    if not isinstance(report, dict) or set(report) != {
        "arm_slew_headroom",
        "gripper_close_arm_interlock",
    }:
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
