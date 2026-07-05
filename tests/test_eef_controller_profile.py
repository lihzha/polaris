from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

from polaris import eef_controller_profile as controller_profile_module
from polaris.config import EEF_CONTROLLER_BASELINE_PROFILE
from polaris.config import EEF_CONTROLLER_CONCURRENT_ARM_GRIPPER_CANDIDATE_PROFILE
from polaris.config import EEF_CONTROLLER_MIMIC_COMPLIANCE_CANDIDATE_PROFILE
from polaris.config import EEF_CONTROLLER_RELEASE_RAMP_CANDIDATE_PROFILE
from polaris.config import EEF_CONTROLLER_VELOCITY_RECOVERY_CANDIDATE_PROFILE
from polaris.eef_controller_profile import configure_eef_controller_profile
from polaris.eef_controller_profile import CONCURRENT_ARM_NO_CLOSE_INTERLOCK_PROFILE
from polaris.eef_controller_profile import eef_controller_apply_counts_from_safety
from polaris.eef_controller_profile import eef_controller_profile
from polaris.eef_controller_profile import (
    validate_eef_controller_repair_candidate_report,
)
from polaris.eef_controller_profile import validate_eef_controller_profile_config
from polaris.eef_controller_repair import ARM_RELEASE_RAMP_FORMULA_PROFILE
from polaris.eef_controller_repair import ARM_RELEASE_RAMP_FRACTION_PROFILE
from polaris.eef_controller_repair import ARM_RELEASE_RAMP_PROFILE
from polaris.eef_controller_repair import ARM_RELEASE_RAMP_STATE_PROFILE
from polaris.eef_controller_repair import ARM_RELEASE_RAMP_SUBSTEPS
from polaris.eef_controller_repair import ARM_RELEASE_RAMP_TRANSACTION_PROFILE
from polaris.eef_controller_repair import arm_release_ramp_fraction
from polaris.eef_gripper_failure_trace import EEF_ALL_SIX_GRIPPER_TRACE_PROFILE
from polaris.eef_gripper_failure_trace import (
    make_eef_all_six_gripper_failure_trace_class,
)
from polaris.eef_gripper_runtime import EEF_GRIPPER_MIMIC_COMPLIANCE_PROFILE
from polaris.eef_gripper_runtime import EEF_GRIPPER_TARGET_SLEW_PROFILE
from polaris.eef_gripper_runtime import (
    EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE,
)
from polaris.eef_runtime_contract import ARM_FAILURE_SUBSTEP_TRACE_PHASE_CONTRACT
from polaris.eef_runtime_contract import validate_arm_failure_substep_trace


class _BaseArmAction:
    pass


class _BaseFingerAction:
    pass


def _original_spawn(*_args, **_kwargs):
    return None


_original_spawn.__module__ = "isaaclab.sim.spawners.from_files.from_files"
_original_spawn.__qualname__ = "spawn_from_usd"
_original_spawn.__name__ = "spawn_from_usd"


@pytest.fixture(autouse=True)
def _exact_production_objects(monkeypatch) -> None:
    monkeypatch.setattr(
        "polaris.eef_controller_profile._expected_arm_action_class",
        lambda: _BaseArmAction,
    )
    monkeypatch.setattr(
        "polaris.eef_controller_profile._expected_finger_action_class",
        lambda: _BaseFingerAction,
    )
    monkeypatch.setattr(
        "polaris.eef_gripper_runtime._expected_original_spawn_func",
        lambda: _original_spawn,
    )


def _env_cfg() -> SimpleNamespace:
    arm = SimpleNamespace(
        class_type=_BaseArmAction,
        enable_failure_substep_trace=False,
        enable_wrist_energy_brake=False,
        enable_arm_slew_headroom=False,
        enable_gripper_close_arm_interlock=False,
        enable_arm_release_ramp=False,
        enable_current_joint_velocity_recovery=False,
    )
    finger = SimpleNamespace(
        enable_target_slew_rate_0p25_candidate=False,
        class_type=_BaseFingerAction,
    )
    spawn = SimpleNamespace(func=_original_spawn)
    return SimpleNamespace(
        actions=SimpleNamespace(arm=arm, finger_joint=finger),
        scene=SimpleNamespace(robot=SimpleNamespace(spawn=spawn)),
    )


def _snapshot(cfg: SimpleNamespace) -> dict:
    return {
        "arm": copy.deepcopy(vars(cfg.actions.arm)),
        "finger_flag": cfg.actions.finger_joint.enable_target_slew_rate_0p25_candidate,
        "finger_class": cfg.actions.finger_joint.class_type,
        "spawn_func": cfg.scene.robot.spawn.func,
    }


def test_baseline_profile_is_an_exact_config_noop() -> None:
    cfg = _env_cfg()
    before = _snapshot(cfg)

    spec = configure_eef_controller_profile(
        cfg,
        profile=EEF_CONTROLLER_BASELINE_PROFILE,
    )

    assert _snapshot(cfg) == before
    assert spec.profile == EEF_CONTROLLER_BASELINE_PROFILE
    assert spec.target_slew_profile == EEF_GRIPPER_TARGET_SLEW_PROFILE
    assert spec.failure_substep_trace_enabled is False
    assert spec.all_six_gripper_trace_enabled is False
    assert spec.fixed_activation_anchor is False
    assert spec.mimic_compliance_profile is None


def test_candidate_configures_exact_accepted_stack_before_spawn(monkeypatch) -> None:
    cfg = _env_cfg()
    monkeypatch.setattr(
        "polaris.eef_gripper_runtime._expected_original_spawn_func",
        lambda: _original_spawn,
    )

    spec = configure_eef_controller_profile(
        cfg,
        profile=EEF_CONTROLLER_MIMIC_COMPLIANCE_CANDIDATE_PROFILE,
    )

    assert spec.profile == (
        "arm_slew_0p95_gripper_rate0p25_fixed_anchor86_mimic100_damping1p2_v3"
    )
    assert spec.target_slew_profile == (
        EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
    )
    assert spec.close_interlock_profile == (
        "eef_gripper_close_fixed_activation_anchor_86_physics_substeps_v2"
    )
    assert spec.close_interlock_substeps == 86
    assert spec.fixed_activation_anchor is True
    assert spec.mimic_compliance_profile == EEF_GRIPPER_MIMIC_COMPLIANCE_PROFILE
    assert spec.failure_substep_trace_enabled is True
    assert spec.all_six_gripper_trace_enabled is True
    assert cfg.actions.arm.enable_failure_substep_trace is True
    assert cfg.actions.arm.enable_wrist_energy_brake is False
    assert cfg.actions.arm.enable_arm_slew_headroom is True
    assert cfg.actions.arm.enable_gripper_close_arm_interlock is True
    assert cfg.actions.arm.enable_arm_release_ramp is False
    assert cfg.actions.arm.enable_current_joint_velocity_recovery is False
    assert cfg.actions.finger_joint.enable_target_slew_rate_0p25_candidate is True
    assert (
        cfg.actions.finger_joint.class_type.eef_all_six_gripper_trace_profile
        == EEF_ALL_SIX_GRIPPER_TRACE_PROFILE
    )
    overlay = cfg.scene.robot.spawn.func
    assert overlay is not _original_spawn
    assert overlay.__module__ == "polaris.eef_gripper_runtime"
    assert overlay.__qualname__ == "eef_mimic_compliance_spawn_overlay"
    assert overlay._eef_mimic_compliance_target_slew_profile == (
        EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
    )
    assert overlay._eef_mimic_compliance_overlay_call_count == 0
    assert overlay._eef_mimic_compliance_original_spawn_call_count == 0
    validate_eef_controller_profile_config(
        cfg,
        expected_profile=EEF_CONTROLLER_MIMIC_COMPLIANCE_CANDIDATE_PROFILE,
    )


def test_release_ramp_profile_adds_only_the_versioned_v4_flag(monkeypatch) -> None:
    cfg = _env_cfg()
    monkeypatch.setattr(
        "polaris.eef_gripper_runtime._expected_original_spawn_func",
        lambda: _original_spawn,
    )
    spec = configure_eef_controller_profile(
        cfg,
        profile=EEF_CONTROLLER_RELEASE_RAMP_CANDIDATE_PROFILE,
    )
    assert spec.profile == (
        "arm_slew_0p95_gripper_rate0p25_fixed_anchor86_release_ramp16_"
        "mimic100_damping1p2_v4"
    )
    assert spec.arm_release_ramp_enabled is True
    assert cfg.actions.arm.enable_arm_release_ramp is True
    assert spec.current_joint_velocity_recovery_enabled is False
    assert cfg.actions.arm.enable_current_joint_velocity_recovery is False
    assert cfg.actions.arm.enable_failure_substep_trace is True
    assert cfg.actions.arm.enable_arm_slew_headroom is True
    assert cfg.actions.arm.enable_gripper_close_arm_interlock is True
    validate_eef_controller_profile_config(
        cfg,
        expected_profile=EEF_CONTROLLER_RELEASE_RAMP_CANDIDATE_PROFILE,
    )


def test_velocity_recovery_profile_adds_only_the_versioned_v5_flag(
    monkeypatch,
) -> None:
    cfg = _env_cfg()
    monkeypatch.setattr(
        "polaris.eef_gripper_runtime._expected_original_spawn_func",
        lambda: _original_spawn,
    )
    spec = configure_eef_controller_profile(
        cfg,
        profile=EEF_CONTROLLER_VELOCITY_RECOVERY_CANDIDATE_PROFILE,
    )
    assert spec.profile == (
        "arm_slew_0p95_gripper_rate0p25_fixed_anchor86_release_ramp16_"
        "velocity_recovery8_clean2_mimic100_damping1p2_v5"
    )
    assert spec.arm_release_ramp_enabled is True
    assert spec.current_joint_velocity_recovery_enabled is True
    assert cfg.actions.arm.enable_arm_release_ramp is True
    assert cfg.actions.arm.enable_current_joint_velocity_recovery is True
    validate_eef_controller_profile_config(
        cfg,
        expected_profile=EEF_CONTROLLER_VELOCITY_RECOVERY_CANDIDATE_PROFILE,
    )


def test_concurrent_v6_configures_recovery_without_interlock_or_release_ramp(
    monkeypatch,
) -> None:
    cfg = _env_cfg()
    monkeypatch.setattr(
        "polaris.eef_gripper_runtime._expected_original_spawn_func",
        lambda: _original_spawn,
    )
    spec = configure_eef_controller_profile(
        cfg,
        profile=EEF_CONTROLLER_CONCURRENT_ARM_GRIPPER_CANDIDATE_PROFILE,
    )
    assert spec.profile.endswith(
        "concurrent_arm_velocity_recovery8_clean2_mimic100_damping1p2_v6"
    )
    assert spec.concurrent_arm_gripper_enabled is True
    assert spec.open_endpoint_coupled_impulse_telemetry_enabled is True
    assert spec.current_joint_velocity_recovery_enabled is True
    assert spec.gripper_close_arm_interlock_enabled is False
    assert spec.arm_release_ramp_enabled is False
    assert spec.close_interlock_profile == CONCURRENT_ARM_NO_CLOSE_INTERLOCK_PROFILE
    assert spec.close_interlock_substeps == 0
    assert spec.fixed_activation_anchor is False
    assert cfg.actions.arm.enable_gripper_close_arm_interlock is False
    assert cfg.actions.arm.enable_arm_release_ramp is False
    assert cfg.actions.arm.enable_current_joint_velocity_recovery is True
    assert cfg.actions.finger_joint.enable_target_slew_rate_0p25_candidate is True
    validate_eef_controller_profile_config(
        cfg,
        expected_profile=EEF_CONTROLLER_CONCURRENT_ARM_GRIPPER_CANDIDATE_PROFILE,
    )


def test_candidate_fails_closed_on_pre_enabled_or_tampered_components(
    monkeypatch,
) -> None:
    cfg = _env_cfg()
    cfg.actions.arm.enable_wrist_energy_brake = True
    with pytest.raises(ValueError, match="already enabled"):
        configure_eef_controller_profile(
            cfg,
            profile=EEF_CONTROLLER_MIMIC_COMPLIANCE_CANDIDATE_PROFILE,
        )
    assert cfg.scene.robot.spawn.func is _original_spawn

    cfg = _env_cfg()
    monkeypatch.setattr(
        "polaris.eef_gripper_runtime._expected_original_spawn_func",
        lambda: _original_spawn,
    )
    configure_eef_controller_profile(
        cfg,
        profile=EEF_CONTROLLER_MIMIC_COMPLIANCE_CANDIDATE_PROFILE,
    )
    cfg.actions.arm.enable_gripper_close_arm_interlock = False
    with pytest.raises(ValueError, match="config/profile mismatch"):
        validate_eef_controller_profile_config(
            cfg,
            expected_profile=EEF_CONTROLLER_MIMIC_COMPLIANCE_CANDIDATE_PROFILE,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda cfg: setattr(cfg.actions.arm, "class_type", object),
        lambda cfg: setattr(cfg.actions.finger_joint, "class_type", object),
        lambda cfg: setattr(
            cfg.actions.finger_joint,
            "class_type",
            make_eef_all_six_gripper_failure_trace_class(_BaseFingerAction),
        ),
        lambda cfg: setattr(cfg.scene.robot.spawn, "func", lambda *_a, **_k: None),
    ],
)
@pytest.mark.parametrize(
    "profile",
    [
        EEF_CONTROLLER_BASELINE_PROFILE,
        EEF_CONTROLLER_MIMIC_COMPLIANCE_CANDIDATE_PROFILE,
        EEF_CONTROLLER_RELEASE_RAMP_CANDIDATE_PROFILE,
        EEF_CONTROLLER_VELOCITY_RECOVERY_CANDIDATE_PROFILE,
    ],
)
def test_profiles_reject_modified_action_classes_and_spawn_before_writes(
    mutation,
    profile,
) -> None:
    cfg = _env_cfg()
    mutation(cfg)
    before = _snapshot(cfg)
    with pytest.raises(ValueError, match="class drift|spawn"):
        configure_eef_controller_profile(cfg, profile=profile)
    assert _snapshot(cfg) == before


def test_profile_mapping_is_closed() -> None:
    with pytest.raises(ValueError, match="Unknown PolaRiS EEF controller profile"):
        eef_controller_profile("candidate-ish")
    with pytest.raises(ValueError, match="Unknown PolaRiS EEF controller profile"):
        eef_controller_profile(None)  # type: ignore[arg-type]


def test_public_eval_wires_profile_before_gym_and_threads_evidence() -> None:
    source = (Path(__file__).parents[1] / "scripts" / "eval.py").read_text()
    configure = source.index("configure_eef_controller_profile(")
    make = source.index("env: ManagerBasedRLSplatEnv = gym.make(")
    assert configure < make
    assert "profile=eval_args.eef_controller_profile" in source
    assert "expected_eef_controller_profile=controller_profile.profile" in source
    assert source.count("expected_gripper_target_slew_profile=(") >= 6
    assert "controller_repair_candidate=controller_repair_aggregate" in source
    assert "all_six_gripper_trace=all_six_gripper_trace" in source
    assert "arm_failure_substep_trace=arm_failure_substep_trace" in source


def _arm_trace() -> dict:
    widths = {
        "joint_pos_rad": 7,
        "joint_vel_rad_s": 7,
        "post_joint_pos_rad": 7,
        "post_joint_vel_rad_s": 7,
        "delta_joint_pos_rad": 7,
        "delta_joint_vel_rad_s": 7,
        "previous_joint_pos_target_rad": 7,
        "raw_dls_joint_pos_target_rad": 7,
        "new_joint_pos_target_rad": 7,
        "new_joint_vel_target_rad_s": 7,
        "new_joint_effort_target_nm": 7,
        "current_eef_position_m": 3,
        "current_eef_quaternion_wxyz": 4,
        "desired_eef_position_m": 3,
        "desired_eef_quaternion_wxyz": 4,
        "pose_error_position_m_axis_angle_rad": 6,
        "approximate_pd_effort_preclip_nm": 7,
        "approximate_pd_effort_postclip_nm": 7,
    }

    def vector(width: int) -> dict:
        return {
            "values": [0.0] * width,
            "finite_mask": [True] * width,
            "finite_count": width,
        }

    entries = [
        {
            "apply_index": apply_index,
            "policy_step": 0,
            "physics_substep": apply_index,
            **{name: vector(width) for name, width in widths.items()},
        }
        for apply_index in range(2)
    ]
    return {
        "schema_version": 1,
        "profile": "eef_applied_substep_ring_last64_v1",
        "episode_index": 2,
        "capacity": 64,
        "policy_step_capacity": 8,
        "decimation": 8,
        "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
        "joint_drive_stiffness": [400.0] * 7,
        "joint_drive_damping": [80.0] * 7,
        "joint_effort_limits": [87.0] * 4 + [12.0] * 3,
        "effort_semantics": (
            "isaaclab_implicit_actuator_approximate_pd_preclip_and_"
            "effortlimit_clipped_v1"
        ),
        "phase_contract": copy.deepcopy(ARM_FAILURE_SUBSTEP_TRACE_PHASE_CONTRACT),
        "completed_entry_count": 2,
        "total_completed_entry_count": 2,
        "dropped_prefix_entry_count": 0,
        "pending_entry_count": 0,
        "pending_apply_index": None,
        "entries": entries,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda phase: phase.__setitem__("extra", "unreviewed"),
        lambda phase: phase.pop("effort"),
        lambda phase: phase.__setitem__("joint_state", "changed"),
    ],
)
def test_arm_failure_trace_phase_contract_is_closed(mutation) -> None:
    trace = _arm_trace()
    validate_arm_failure_substep_trace(trace, episode_index=2, apply_calls=3)
    drifted = copy.deepcopy(trace)
    mutation(drifted["phase_contract"])
    with pytest.raises(ValueError, match="semantics drift"):
        validate_arm_failure_substep_trace(
            drifted,
            episode_index=2,
            apply_calls=3,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda trace: trace.__setitem__("schema_version", True),
            "schema_version type",
        ),
        (
            lambda trace: trace["entries"][0].__setitem__("apply_index", False),
            "entry cadence",
        ),
        (
            lambda trace: trace["entries"][0].__setitem__("policy_step", False),
            "entry cadence",
        ),
        (
            lambda trace: trace["entries"][0].__setitem__("physics_substep", False),
            "entry cadence",
        ),
        (
            lambda trace: trace["entries"][1]["joint_pos_rad"]["values"].__setitem__(
                0, 1.0
            ),
            "transition continuity",
        ),
        (
            lambda trace: trace["entries"][0]["delta_joint_pos_rad"][
                "values"
            ].__setitem__(0, 1.0),
            "joint_pos delta semantics",
        ),
        (
            lambda trace: trace["entries"][0]["approximate_pd_effort_preclip_nm"][
                "values"
            ].__setitem__(0, 1.0),
            "PD preclip semantics",
        ),
        (
            lambda trace: trace["entries"][0]["approximate_pd_effort_postclip_nm"][
                "values"
            ].__setitem__(0, 1.0),
            "PD postclip semantics",
        ),
    ],
)
def test_arm_failure_trace_causal_semantics_are_closed(mutation, message) -> None:
    trace = _arm_trace()
    mutation(trace)
    with pytest.raises(ValueError, match=message):
        validate_arm_failure_substep_trace(trace, episode_index=2, apply_calls=3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"episode_index": True, "apply_calls": 3},
        {"episode_index": 2, "apply_calls": True},
    ],
)
def test_arm_failure_trace_public_identity_inputs_reject_bool(kwargs) -> None:
    with pytest.raises(ValueError, match="validation input"):
        validate_arm_failure_substep_trace(_arm_trace(), **kwargs)


def test_arm_failure_trace_binds_exact_failed_apply_boundary() -> None:
    first_apply_failure = _arm_trace()
    first_apply_failure.update(
        {
            "completed_entry_count": 0,
            "total_completed_entry_count": 0,
            "dropped_prefix_entry_count": 0,
            "entries": [],
        }
    )
    validate_arm_failure_substep_trace(
        first_apply_failure,
        episode_index=2,
        apply_calls=1,
    )

    with pytest.raises(ValueError, match="retention/cadence"):
        validate_arm_failure_substep_trace(
            first_apply_failure,
            episode_index=2,
            apply_calls=937,
        )

    pending = _arm_trace()
    pending["pending_entry_count"] = 1
    pending["pending_apply_index"] = 2
    with pytest.raises(ValueError, match="retention/cadence"):
        validate_arm_failure_substep_trace(
            pending,
            episode_index=2,
            apply_calls=3,
        )


def _candidate_report() -> dict:
    spec = eef_controller_profile(EEF_CONTROLLER_MIMIC_COMPLIANCE_CANDIDATE_PROFILE)
    physical = [0.01] * 7
    return {
        "arm_slew_headroom": {
            "enabled": True,
            "profile": "panda_nominal_target_slew_0p95_physical_limit_v1",
            "ratio": 0.95,
            "physical_max_delta_joint_pos_rad": physical,
            "nominal_max_delta_joint_pos_rad": [0.0095] * 7,
        },
        "gripper_close_arm_interlock": {
            "enabled": True,
            "profile": spec.close_interlock_profile,
            "configured_substeps": spec.close_interlock_substeps,
            "remaining_substeps": 0,
            "observed_endpoint_change_count": 0,
            "endpoint_observed": False,
            "activation_count": 0,
            "active_apply_count": 0,
            "anchor_valid": False,
            "anchor_capture_count": 0,
            "anchor_target_apply_count": 0,
            "anchor_first_exact_target_count": 0,
            "anchor_refresh_count": 0,
            "anchor_slew_limit_event_count": 0,
            "anchor_slew_limited_joint_count": 0,
            "anchor_position_limit_event_count": 0,
            "anchor_position_limited_joint_count": 0,
            "anchor_completion_count": 0,
            "anchor_open_cancel_count": 0,
            "last_activation_apply_index": None,
            "last_anchor_joint_pos_rad": None,
            "last_anchor_little_endian_float32_sha256": None,
            "max_abs_current_anchor_residual_rad": [0.0] * 7,
            "max_abs_target_anchor_residual_rad": [0.0] * 7,
            "max_abs_active_delta_joint_pos_rad": [0.0] * 7,
            "released_apply_count": 0,
            "max_abs_released_delta_joint_pos_rad": [0.0] * 7,
        },
    }


def _concurrent_v6_report(*, apply_calls: int, endpoint_changes: int = 0) -> dict:
    report = _candidate_report()
    interlock = report["gripper_close_arm_interlock"]
    interlock.update(
        {
            "enabled": False,
            "profile": CONCURRENT_ARM_NO_CLOSE_INTERLOCK_PROFILE,
            "configured_substeps": 0,
            "observed_endpoint_change_count": endpoint_changes,
        }
    )
    report["concurrent_arm_gripper"] = {
        "enabled": True,
        "profile": "fresh_dls_every_normal_apply_no_gripper_target_replay_v1",
        "fresh_dls_target_applies": apply_calls,
        "normal_target_setter_applies": apply_calls,
        "closed_endpoint_fresh_dls_target_applies": apply_calls,
        "closed_endpoint_distinct_desired_pose_count": min(apply_calls, 3),
        "recovery_owned_target_applies": 0,
        "deferred_endpoint_transition_count": 0,
        "stored_target_replay_count": 0,
    }
    report["current_joint_velocity_recovery"] = {}
    return report


def test_concurrent_v6_report_has_explicit_disabled_interlock_and_fresh_dls_cadence(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        controller_profile_module,
        "validate_current_joint_velocity_recovery_report",
        lambda value, *, apply_calls: {
            "counters": {
                "hold_target_applies": 0,
                "recovery_active_substeps": 0,
            }
        },
    )
    report = _concurrent_v6_report(apply_calls=80, endpoint_changes=15)
    validated = validate_eef_controller_repair_candidate_report(
        report,
        expected_profile=EEF_CONTROLLER_CONCURRENT_ARM_GRIPPER_CANDIDATE_PROFILE,
        expected_target_slew_profile=(
            EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
        ),
        expected_physical_max_delta_joint_pos_rad=[0.01] * 7,
        apply_calls=80,
    )
    assert validated["gripper_close_arm_interlock"]["configured_substeps"] == 0
    assert (
        validated["gripper_close_arm_interlock"]["observed_endpoint_change_count"] == 15
    )
    assert "arm_release_ramp" not in validated
    drifted = copy.deepcopy(report)
    drifted["concurrent_arm_gripper"]["stored_target_replay_count"] = 1
    with pytest.raises(ValueError, match="concurrent-arm cadence"):
        validate_eef_controller_repair_candidate_report(
            drifted,
            expected_profile=(EEF_CONTROLLER_CONCURRENT_ARM_GRIPPER_CANDIDATE_PROFILE),
            expected_target_slew_profile=(
                EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
            ),
            expected_physical_max_delta_joint_pos_rad=[0.01] * 7,
            apply_calls=80,
        )


def test_concurrent_v6_report_binds_recovery_owned_targets_to_recovery_counters(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        controller_profile_module,
        "validate_current_joint_velocity_recovery_report",
        lambda value, *, apply_calls: {
            "counters": {
                "hold_target_applies": 3,
                "recovery_active_substeps": 3,
            }
        },
    )
    report = _concurrent_v6_report(apply_calls=80)
    concurrent = report["concurrent_arm_gripper"]
    concurrent["fresh_dls_target_applies"] = 77
    concurrent["normal_target_setter_applies"] = 77
    concurrent["closed_endpoint_fresh_dls_target_applies"] = 77
    concurrent["recovery_owned_target_applies"] = 3
    validate_eef_controller_repair_candidate_report(
        report,
        expected_profile=EEF_CONTROLLER_CONCURRENT_ARM_GRIPPER_CANDIDATE_PROFILE,
        expected_target_slew_profile=(
            EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
        ),
        expected_physical_max_delta_joint_pos_rad=[0.01] * 7,
        apply_calls=80,
    )

    concurrent["recovery_owned_target_applies"] = 2
    concurrent["fresh_dls_target_applies"] = 78
    concurrent["normal_target_setter_applies"] = 78
    with pytest.raises(ValueError, match="target ownership drift"):
        validate_eef_controller_repair_candidate_report(
            report,
            expected_profile=(EEF_CONTROLLER_CONCURRENT_ARM_GRIPPER_CANDIDATE_PROFILE),
            expected_target_slew_profile=(
                EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
            ),
            expected_physical_max_delta_joint_pos_rad=[0.01] * 7,
            apply_calls=80,
        )


def test_controller_smoke_has_v6_moving_close_reopen_target_surface_gate() -> None:
    source = (
        Path(__file__).parents[1] / "scripts" / "smoke_eef_pose_controller.py"
    ).read_text(encoding="utf-8")
    assert '"--eef-controller-profile"' in source
    assert source.index("configure_eef_controller_profile(") < source.index(
        "env = gym.make("
    )
    assert "moving EEF through close and reopen transitions" in source
    assert '"fresh_dls_target_applies"' in source
    assert '"closed_endpoint_distinct_desired_pose_count"' in source
    assert '"stored_target_replay_count"' in source
    assert '"open_endpoint_contact_mimic_impulse"' in source
    assert 'driver_target_slew = concurrent_safety["gripper_runtime_dynamic"]' in source
    assert (
        'interlock["observed_endpoint_change_count"]\n'
        '            == driver_target_slew["endpoint_change_count"]' in source
    )
    assert '"arm_release_ramp" not in concurrent_report' in source
    assert '"eef_open115_then_close5_same_arm_pose_v1"' in source
    assert '"eef_open115_then_close10_same_arm_pose_v2"' in source
    assert "delayed_close_replay_profile = (" in source
    assert "finger_term.begin_eef_policy_step(" in source

    tree = ast.parse(source)
    env_steps_with_context: list[tuple[ast.stmt, ast.stmt | None, list[ast.stmt]]] = []

    def visit_statement_lists(node: ast.AST) -> None:
        for _, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                visit_statement_lists(value)
            elif isinstance(value, list):
                if value and all(isinstance(item, ast.stmt) for item in value):
                    for index, statement in enumerate(value):
                        has_env_step = isinstance(statement, ast.Assign) and any(
                            isinstance(candidate, ast.Call)
                            and isinstance(candidate.func, ast.Attribute)
                            and isinstance(candidate.func.value, ast.Name)
                            and candidate.func.value.id == "env"
                            and candidate.func.attr == "step"
                            for candidate in ast.walk(statement)
                        )
                        if has_env_step:
                            predecessor = value[index - 1] if index else None
                            env_steps_with_context.append(
                                (statement, predecessor, value[index + 1 : index + 3])
                            )
                        visit_statement_lists(statement)
                else:
                    for item in value:
                        if isinstance(item, ast.AST):
                            visit_statement_lists(item)

    visit_statement_lists(tree)
    expected_contexts = [
        ({"episode_index": "case_index", "policy_step": "policy_step"}, None),
        (
            {
                "episode_index": "delayed_close_index",
                "policy_step": "delayed_policy_step",
            },
            "delayed_policy_step",
        ),
        (
            {
                "episode_index": "concurrent_index",
                "policy_step": "concurrent_policy_step",
            },
            "concurrent_policy_step",
        ),
        (
            {
                "episode_index": "concurrent_index",
                "policy_step": "concurrent_policy_step",
            },
            "concurrent_policy_step",
        ),
        ({"episode_index": "adversarial_index", "policy_step": "0"}, None),
    ]
    assert len(env_steps_with_context) == len(expected_contexts)
    for (env_step, predecessor, successors), (
        expected_context,
        expected_counter,
    ) in zip(env_steps_with_context, expected_contexts, strict=True):
        assert isinstance(predecessor, ast.Expr), ast.unparse(env_step)
        begin_call = predecessor.value
        assert isinstance(begin_call, ast.Call), ast.unparse(env_step)
        assert isinstance(begin_call.func, ast.Name), ast.unparse(env_step)
        assert begin_call.func.id == "begin_gripper_trace_policy_step", ast.unparse(
            env_step
        )
        assert {
            keyword.arg: ast.unparse(keyword.value) for keyword in begin_call.keywords
        } == expected_context
        if expected_counter is not None:
            assert len(successors) == 2
            increment = successors[1]
            assert isinstance(increment, ast.AugAssign), ast.unparse(env_step)
            assert isinstance(increment.target, ast.Name), ast.unparse(env_step)
            assert increment.target.id == expected_counter, ast.unparse(env_step)
            assert isinstance(increment.op, ast.Add), ast.unparse(env_step)
            assert isinstance(increment.value, ast.Constant), ast.unparse(env_step)
            assert increment.value.value == 1, ast.unparse(env_step)

    counter_initializers = {
        target.id: statement.value.value
        for statement in ast.walk(tree)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance((target := statement.targets[0]), ast.Name)
        and target.id in {"delayed_policy_step", "concurrent_policy_step"}
        and isinstance(statement.value, ast.Constant)
    }
    assert counter_initializers == {
        "delayed_policy_step": 0,
        "concurrent_policy_step": 0,
    }

    helper_definitions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "begin_gripper_trace_policy_step"
    ]
    assert len(helper_definitions) == 1
    helper_guards = [
        statement
        for statement in helper_definitions[0].body
        if isinstance(statement, ast.If)
    ]
    assert len(helper_guards) == 1
    helper_guard = helper_guards[0]
    assert ast.unparse(helper_guard.test) == (
        "controller_spec.all_six_gripper_trace_enabled"
    )
    assert len(helper_guard.body) == 1
    helper_delegate = helper_guard.body[0]
    assert isinstance(helper_delegate, ast.Expr)
    assert isinstance(helper_delegate.value, ast.Call)
    assert ast.unparse(helper_delegate.value.func) == (
        "finger_term.begin_eef_policy_step"
    )
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in helper_delegate.value.keywords
    } == {"episode_index": "episode_index", "policy_step": "policy_step"}


def _release_ramp_candidate_report() -> dict:
    report = _candidate_report()
    report["arm_release_ramp"] = {
        "enabled": True,
        "profile": ARM_RELEASE_RAMP_PROFILE,
        "state_profile": ARM_RELEASE_RAMP_STATE_PROFILE,
        "substeps": ARM_RELEASE_RAMP_SUBSTEPS,
        "fraction_profile": ARM_RELEASE_RAMP_FRACTION_PROFILE,
        "fractions_float32": [
            arm_release_ramp_fraction(index)
            for index in range(ARM_RELEASE_RAMP_SUBSTEPS)
        ],
        "formula_profile": ARM_RELEASE_RAMP_FORMULA_PROFILE,
        "transaction_profile": ARM_RELEASE_RAMP_TRANSACTION_PROFILE,
        "open_during_ramp_policy": ("continue_current_ramp_without_restart_or_skip_v1"),
        "phase": "release",
        "next_index": None,
        "release_observed_count": 0,
        "ramp_started_count": 0,
        "ramp_completed_count": 0,
        "ramp_cancelled_by_reactivation_count": 0,
        "ramp_target_apply_count": 0,
        "cancelled_ramp_target_apply_count": 0,
        "ramp_limited_target_apply_count": 0,
        "ramp_limited_joint_target_count": 0,
        "last_target_apply_index": None,
        "last_ramp_index": None,
        "max_abs_nominal_to_ramped_target_change_rad": [0.0] * 7,
        "gripper_target_or_state_write_count": 0,
    }
    return report


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _completed_release_ramp_candidate_report() -> dict:
    report = _release_ramp_candidate_report()
    interlock = report["gripper_close_arm_interlock"]
    anchor = [_float32(0.1 * (index + 1)) for index in range(7)]
    interlock.update(
        {
            "observed_endpoint_change_count": 1,
            "endpoint_observed": True,
            "activation_count": 1,
            "active_apply_count": 86,
            "anchor_capture_count": 1,
            "anchor_target_apply_count": 86,
            "anchor_first_exact_target_count": 1,
            "anchor_completion_count": 1,
            "last_activation_apply_index": 0,
            "last_anchor_joint_pos_rad": anchor,
            "last_anchor_little_endian_float32_sha256": hashlib.sha256(
                struct.pack("<7f", *anchor)
            ).hexdigest(),
            "released_apply_count": 16,
        }
    )
    report["arm_release_ramp"].update(
        {
            "release_observed_count": 1,
            "ramp_started_count": 1,
            "ramp_completed_count": 1,
            "ramp_target_apply_count": 16,
            "ramp_limited_target_apply_count": 15,
            "ramp_limited_joint_target_count": 105,
            "last_target_apply_index": 101,
            "last_ramp_index": 15,
            "max_abs_nominal_to_ramped_target_change_rad": [0.0095] * 7,
        }
    )
    return report


def _active_release_ramp_candidate_report() -> dict:
    report = _completed_release_ramp_candidate_report()
    report["gripper_close_arm_interlock"]["released_apply_count"] = 5
    report["arm_release_ramp"].update(
        {
            "phase": "ramp",
            "next_index": 5,
            "ramp_completed_count": 0,
            "ramp_target_apply_count": 5,
            "ramp_limited_target_apply_count": 5,
            "ramp_limited_joint_target_count": 35,
            "last_target_apply_index": 90,
            "last_ramp_index": 4,
        }
    )
    return report


def _multi_activation_candidate_report() -> dict:
    report = _candidate_report()
    anchor = [_float32(0.1 * (index + 1)) for index in range(7)]
    interlock = report["gripper_close_arm_interlock"]
    interlock.update(
        {
            "remaining_substeps": 0,
            "observed_endpoint_change_count": 4,
            "endpoint_observed": True,
            "activation_count": 3,
            "active_apply_count": 100,
            "anchor_valid": False,
            "anchor_capture_count": 3,
            "anchor_target_apply_count": 100,
            "anchor_first_exact_target_count": 3,
            "anchor_slew_limit_event_count": 1,
            "anchor_slew_limited_joint_count": 2,
            "anchor_position_limit_event_count": 1,
            "anchor_position_limited_joint_count": 1,
            "anchor_completion_count": 1,
            "anchor_open_cancel_count": 2,
            "last_activation_apply_index": 150,
            "last_anchor_joint_pos_rad": anchor,
            "last_anchor_little_endian_float32_sha256": hashlib.sha256(
                struct.pack("<7f", *anchor)
            ).hexdigest(),
            "max_abs_current_anchor_residual_rad": [0.002] * 7,
            "max_abs_target_anchor_residual_rad": [0.001] * 7,
            "max_abs_active_delta_joint_pos_rad": [0.009] * 7,
            "released_apply_count": 50,
            "max_abs_released_delta_joint_pos_rad": [0.009] * 7,
        }
    )
    return report


def _make_last_anchor_non_float32(report: dict) -> None:
    anchor = report["gripper_close_arm_interlock"]["last_anchor_joint_pos_rad"]
    anchor[0] += 1e-12
    report["gripper_close_arm_interlock"][
        "last_anchor_little_endian_float32_sha256"
    ] = hashlib.sha256(struct.pack("<7f", *anchor)).hexdigest()


def _active_anchor_candidate_report() -> dict:
    report = _multi_activation_candidate_report()
    report["gripper_close_arm_interlock"].update(
        {
            "remaining_substeps": 50,
            "active_apply_count": 132,
            "anchor_valid": True,
            "anchor_target_apply_count": 132,
            "anchor_completion_count": 1,
            "anchor_open_cancel_count": 1,
            "last_activation_apply_index": 250,
        }
    )
    return report


def _set_impossibly_short_cancel_history(report: dict) -> None:
    report["gripper_close_arm_interlock"].update(
        {"active_apply_count": 87, "anchor_target_apply_count": 87}
    )


def _set_impossibly_long_cancel_history(report: dict) -> None:
    report["gripper_close_arm_interlock"].update(
        {
            "active_apply_count": 257,
            "anchor_target_apply_count": 257,
            "released_apply_count": 0,
        }
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda report: report["arm_slew_headroom"].__setitem__("profile", "unreviewed"),
        lambda report: report["gripper_close_arm_interlock"].__setitem__(
            "anchor_refresh_count", 1
        ),
        lambda report: report["gripper_close_arm_interlock"].__setitem__(
            "anchor_capture_count", 1
        ),
    ],
)
def test_candidate_report_identity_and_anchor_lifecycle_are_closed(mutation) -> None:
    report = _candidate_report()
    validate_eef_controller_repair_candidate_report(
        report,
        expected_profile=EEF_CONTROLLER_MIMIC_COMPLIANCE_CANDIDATE_PROFILE,
        expected_target_slew_profile=(
            EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
        ),
        expected_physical_max_delta_joint_pos_rad=[0.01] * 7,
        apply_calls=0,
        require_initial_state=True,
    )
    drifted = copy.deepcopy(report)
    mutation(drifted)
    with pytest.raises(ValueError):
        validate_eef_controller_repair_candidate_report(
            drifted,
            expected_profile=EEF_CONTROLLER_MIMIC_COMPLIANCE_CANDIDATE_PROFILE,
            expected_target_slew_profile=(
                EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
            ),
            expected_physical_max_delta_joint_pos_rad=[0.01] * 7,
            apply_calls=0,
            require_initial_state=True,
        )


def test_release_ramp_v4_report_is_closed_and_v3_remains_byte_compatible() -> None:
    ramp = _release_ramp_candidate_report()
    validate_eef_controller_repair_candidate_report(
        ramp,
        expected_profile=EEF_CONTROLLER_RELEASE_RAMP_CANDIDATE_PROFILE,
        expected_target_slew_profile=(
            EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
        ),
        expected_physical_max_delta_joint_pos_rad=[0.01] * 7,
        apply_calls=0,
        require_initial_state=True,
    )
    with pytest.raises(ValueError, match="schema drift"):
        validate_eef_controller_repair_candidate_report(
            ramp,
            expected_profile=EEF_CONTROLLER_MIMIC_COMPLIANCE_CANDIDATE_PROFILE,
            expected_target_slew_profile=(
                EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
            ),
            expected_physical_max_delta_joint_pos_rad=[0.01] * 7,
            apply_calls=0,
        )
    with pytest.raises(ValueError, match="schema drift"):
        validate_eef_controller_repair_candidate_report(
            _candidate_report(),
            expected_profile=EEF_CONTROLLER_RELEASE_RAMP_CANDIDATE_PROFILE,
            expected_target_slew_profile=(
                EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
            ),
            expected_physical_max_delta_joint_pos_rad=[0.01] * 7,
            apply_calls=0,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda ramp: ramp.__setitem__("phase", "unknown"),
        lambda ramp: ramp.__setitem__("next_index", 16),
        lambda ramp: ramp.__setitem__("release_observed_count", 1),
        lambda ramp: ramp.__setitem__("ramp_target_apply_count", 1),
        lambda ramp: ramp.__setitem__("open_during_ramp_policy", "restart_on_open"),
        lambda ramp: ramp.__setitem__("gripper_target_or_state_write_count", False),
        lambda ramp: ramp["fractions_float32"].__setitem__(8, 0.5),
        lambda ramp: ramp["max_abs_nominal_to_ramped_target_change_rad"].__setitem__(
            0, 0.02
        ),
    ],
)
def test_release_ramp_v4_report_rejects_tampering(mutation) -> None:
    report = _release_ramp_candidate_report()
    mutation(report["arm_release_ramp"])
    with pytest.raises(ValueError, match="release-ramp"):
        validate_eef_controller_repair_candidate_report(
            report,
            expected_profile=EEF_CONTROLLER_RELEASE_RAMP_CANDIDATE_PROFILE,
            expected_target_slew_profile=(
                EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
            ),
            expected_physical_max_delta_joint_pos_rad=[0.01] * 7,
            apply_calls=0,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda ramp: ramp.__setitem__("last_ramp_index", 0),
        lambda ramp: ramp.__setitem__("ramp_limited_target_apply_count", 16),
        lambda ramp: ramp.__setitem__("ramp_limited_joint_target_count", 112),
        lambda ramp: ramp.__setitem__(
            "max_abs_nominal_to_ramped_target_change_rad", [0.0] * 7
        ),
    ],
)
def test_release_ramp_v4_completed_report_rejects_impossible_evidence(
    mutation,
) -> None:
    report = _completed_release_ramp_candidate_report()
    mutation(report["arm_release_ramp"])
    with pytest.raises(ValueError, match="release-ramp"):
        validate_eef_controller_repair_candidate_report(
            report,
            expected_profile=EEF_CONTROLLER_RELEASE_RAMP_CANDIDATE_PROFILE,
            expected_target_slew_profile=(
                EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
            ),
            expected_physical_max_delta_joint_pos_rad=[0.01] * 7,
            apply_calls=102,
        )


def test_release_ramp_v4_report_accepts_completed_cycle() -> None:
    report = _completed_release_ramp_candidate_report()
    validate_eef_controller_repair_candidate_report(
        report,
        expected_profile=EEF_CONTROLLER_RELEASE_RAMP_CANDIDATE_PROFILE,
        expected_target_slew_profile=(
            EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
        ),
        expected_physical_max_delta_joint_pos_rad=[0.01] * 7,
        apply_calls=102,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("last_ramp_index", 3),
        ("last_target_apply_index", 89),
    ],
)
def test_release_ramp_v4_active_report_binds_latest_index_and_apply(
    field,
    value,
) -> None:
    report = _active_release_ramp_candidate_report()
    validate_eef_controller_repair_candidate_report(
        report,
        expected_profile=EEF_CONTROLLER_RELEASE_RAMP_CANDIDATE_PROFILE,
        expected_target_slew_profile=(
            EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
        ),
        expected_physical_max_delta_joint_pos_rad=[0.01] * 7,
        apply_calls=91,
    )
    report["arm_release_ramp"][field] = value
    with pytest.raises(ValueError, match="phase/target"):
        validate_eef_controller_repair_candidate_report(
            report,
            expected_profile=EEF_CONTROLLER_RELEASE_RAMP_CANDIDATE_PROFILE,
            expected_target_slew_profile=(
                EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
            ),
            expected_physical_max_delta_joint_pos_rad=[0.01] * 7,
            apply_calls=91,
        )


def test_candidate_report_accepts_multi_activation_completion_and_open_cancel() -> None:
    for report in (
        _multi_activation_candidate_report(),
        _active_anchor_candidate_report(),
    ):
        validate_eef_controller_repair_candidate_report(
            report,
            expected_profile=EEF_CONTROLLER_MIMIC_COMPLIANCE_CANDIDATE_PROFILE,
            expected_target_slew_profile=(
                EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
            ),
            expected_physical_max_delta_joint_pos_rad=[0.01] * 7,
            apply_calls=300,
        )


def test_candidate_report_allows_exact_first_apply_transaction_abort() -> None:
    safety = {
        "counters": {
            "apply_calls": 1,
            "current_joint_limit_aborts": 0,
            "invariant_aborts": 1,
            "nonfinite_aborts": 0,
        }
    }
    apply_calls, committed_apply_calls = eef_controller_apply_counts_from_safety(safety)
    assert (apply_calls, committed_apply_calls) == (1, 0)
    validate_eef_controller_repair_candidate_report(
        _candidate_report(),
        expected_profile=EEF_CONTROLLER_MIMIC_COMPLIANCE_CANDIDATE_PROFILE,
        expected_target_slew_profile=(
            EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
        ),
        expected_physical_max_delta_joint_pos_rad=[0.01] * 7,
        apply_calls=apply_calls,
        committed_apply_calls=committed_apply_calls,
    )
    with pytest.raises(ValueError, match="lifecycle"):
        validate_eef_controller_repair_candidate_report(
            _candidate_report(),
            expected_profile=EEF_CONTROLLER_MIMIC_COMPLIANCE_CANDIDATE_PROFILE,
            expected_target_slew_profile=(
                EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
            ),
            expected_physical_max_delta_joint_pos_rad=[0.01] * 7,
            apply_calls=1,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda report: report["arm_slew_headroom"].__setitem__("ratio", True),
        lambda report: report["arm_slew_headroom"].__setitem__("ratio", "0.95"),
        lambda report: report["gripper_close_arm_interlock"].__setitem__(
            "remaining_substeps", 86
        ),
        lambda report: report["gripper_close_arm_interlock"].__setitem__(
            "active_apply_count", 2
        ),
        lambda report: report["gripper_close_arm_interlock"].__setitem__(
            "active_apply_count", 259
        ),
        lambda report: report["gripper_close_arm_interlock"].__setitem__(
            "released_apply_count", 201
        ),
        lambda report: report["gripper_close_arm_interlock"][
            "max_abs_released_delta_joint_pos_rad"
        ].__setitem__(0, 0.01),
        lambda report: report["gripper_close_arm_interlock"].__setitem__(
            "observed_endpoint_change_count", 301
        ),
        lambda report: report["gripper_close_arm_interlock"].__setitem__(
            "observed_endpoint_change_count", 1
        ),
        lambda report: report["gripper_close_arm_interlock"].__setitem__(
            "endpoint_observed", False
        ),
        lambda report: report["gripper_close_arm_interlock"].__setitem__(
            "anchor_target_apply_count", 2
        ),
        _set_impossibly_short_cancel_history,
        _set_impossibly_long_cancel_history,
        _make_last_anchor_non_float32,
    ],
)
def test_candidate_report_rejects_impossible_counter_and_anchor_evidence(
    mutation,
) -> None:
    report = _multi_activation_candidate_report()
    mutation(report)
    with pytest.raises(ValueError):
        validate_eef_controller_repair_candidate_report(
            report,
            expected_profile=EEF_CONTROLLER_MIMIC_COMPLIANCE_CANDIDATE_PROFILE,
            expected_target_slew_profile=(
                EEF_GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
            ),
            expected_physical_max_delta_joint_pos_rad=[0.01] * 7,
            apply_calls=300,
        )
