from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPTS))
    try:
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPTS))
    return module


candidate = _load_script("smoke_eef_pose_canary_controller_candidate")


def _candidate_report(*, official: bool, final: bool):
    physical = [0.018125, 0.018125, 0.018125, 0.018125, 0.02175, 0.02175, 0.02175]
    nominal = [value * 0.95 for value in physical]
    vector = [0.0] * 7
    if official and final:
        vector[4] = nominal[4]
    return {
        "arm_slew_headroom": {
            "enabled": True,
            "profile": "panda_nominal_target_slew_0p95_physical_limit_v1",
            "ratio": 0.95,
            "physical_max_delta_joint_pos_rad": physical,
            "nominal_max_delta_joint_pos_rad": nominal,
        },
        "gripper_close_arm_interlock": {
            "enabled": official,
            "profile": "eef_gripper_close_hold_arm_48_physics_substeps_v1",
            "configured_substeps": 48,
            "remaining_substeps": 0,
            "observed_endpoint_change_count": 1 if official and final else 0,
            "endpoint_observed": official and final,
            "activation_count": 1 if official and final else 0,
            "active_apply_count": 48 if official and final else 0,
            "max_abs_active_delta_joint_pos_rad": [0.0] * 7,
            "released_apply_count": 8 if official and final else 0,
            "max_abs_released_delta_joint_pos_rad": vector,
        },
    }


def _final_safety(*, official: bool):
    physical = [0.018125] * 4 + [0.02175] * 3
    nominal = [value * 0.95 for value in physical]
    expected_changes = 1 if official else 0
    return {
        "counters": {
            "apply_calls": 976,
            "environment_substeps": 976,
            "current_joint_limit_aborts": 0,
            "dls_fallbacks": 0,
            "guard_diagnostics_dropped": 0,
            "invariant_aborts": 0,
            "nonfinite_aborts": 0,
            "position_limit_events": 0,
            "position_limited_joints": 0,
            "post_clamp_target_violations": 0,
            "slew_limit_events": 10,
            "slew_limited_joints": 15,
        },
        "maxima": {
            "abs_joint_vel_rad_s": [1.0] * 7,
            "applied_delta_joint_pos_rad": nominal,
            "current_joint_soft_limit_violation_rad": [0.0] * 7,
            "current_physx_hard_limit_violation_rad": [0.0] * 7,
            "post_clamp_target_guard_band_violation_rad": [0.0] * 7,
            "post_clamp_target_soft_limit_violation_rad": [0.0] * 7,
        },
        "joint_velocity_limits_rad_s": [2.175] * 4 + [2.61] * 3,
        "guard_diagnostics": [],
        "current_joint_velocity_abort": None,
        "gripper_runtime_dynamic": {
            "apply_entry_samples": 976,
            "post_policy_step_samples": 122,
            "nonfinite_samples": 0,
            "dropped_diagnostics": 0,
            "driver_target_slew": {
                "apply_calls": 976,
                "live_limit_validation_count": 976,
                "process_action_calls": 122,
                "initialization_count": 1,
                "endpoint_change_count": expected_changes,
                "repeated_endpoint_process_count": 121 - expected_changes,
            },
        },
    }


@pytest.mark.parametrize("variant", sorted(candidate.CANDIDATE_BY_VARIANT))
def test_initial_candidate_report_is_empty_and_profile_exact(variant):
    report = _candidate_report(official=variant == "official_lap3b", final=False)
    assert candidate.validate_candidate_report(report, variant=variant, final=False)


@pytest.mark.parametrize("variant", sorted(candidate.CANDIDATE_BY_VARIANT))
def test_final_candidate_report_requires_expected_isolation(variant):
    report = _candidate_report(official=variant == "official_lap3b", final=True)
    assert candidate.validate_candidate_report(report, variant=variant, final=True)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("activation_count", 0, "did not activate"),
        ("active_apply_count", 47, "did not activate"),
        ("remaining_substeps", 1, "did not activate"),
        ("released_apply_count", 7, "did not activate"),
        (
            "max_abs_released_delta_joint_pos_rad",
            [0.0] * 7,
            "did not activate",
        ),
        (
            "max_abs_active_delta_joint_pos_rad",
            [0.0] * 6 + [1e-3],
            "did not activate",
        ),
    ],
)
def test_official_final_report_rejects_incomplete_release(field, value, match):
    report = _candidate_report(official=True, final=True)
    report["gripper_close_arm_interlock"][field] = value
    with pytest.raises(candidate.CandidateReplayValidationError, match=match):
        candidate.validate_candidate_report(
            report, variant="official_lap3b", final=True
        )


def test_reasoning_final_report_rejects_interlock_evidence():
    report = _candidate_report(official=False, final=True)
    report["gripper_close_arm_interlock"]["activation_count"] = 1
    with pytest.raises(
        candidate.CandidateReplayValidationError, match="unexpectedly used"
    ):
        candidate.validate_candidate_report(
            report, variant="reasoning_43075", final=True
        )


def test_candidate_report_rejects_nominal_ratio_drift():
    report = _candidate_report(official=False, final=True)
    report["arm_slew_headroom"]["nominal_max_delta_joint_pos_rad"][3] *= 0.9
    with pytest.raises(candidate.CandidateReplayValidationError, match="ratio 3"):
        candidate.validate_candidate_report(
            report, variant="reasoning_43075", final=True
        )


def test_velocity_headroom_uses_live_safety_maxima_and_limits():
    safety = {
        "maxima": {"abs_joint_vel_rad_s": [1.0, 2.0, 0.0, 1.5, 2.5, 2.0, 1.0]},
        "joint_velocity_limits_rad_s": [2.175] * 4 + [2.61] * 3,
    }
    result = candidate.validate_velocity_headroom(safety)
    assert result["passed"] is True
    assert result["maximum_ratio"] == pytest.approx(2.5 / 2.61)


def test_velocity_headroom_rejects_a_physical_limit_crossing():
    safety = {
        "maxima": {"abs_joint_vel_rad_s": [0.0] * 6 + [2.7]},
        "joint_velocity_limits_rad_s": [2.175] * 4 + [2.61] * 3,
    }
    with pytest.raises(candidate.CandidateReplayValidationError, match="bound 6"):
        candidate.validate_velocity_headroom(safety)


def test_runner_binds_exact_fixture_plus_two_action_release_probe():
    assert candidate.FIXTURE_ACTION_COUNT == 120
    assert candidate.POST_FIXTURE_REPEAT_COUNT == 2
    assert candidate.TOTAL_ACTION_COUNT == 122
    source = (SCRIPTS / "smoke_eef_pose_canary_controller_candidate.py").read_text()
    assert source.count("env_cfg.actions.arm.enable_arm_slew_headroom = True") == 1
    assert (
        source.count("env_cfg.actions.arm.enable_gripper_close_arm_interlock = (") == 1
    )
    assert "list(actions) + [list(actions[-1])] * POST_FIXTURE_REPEAT_COUNT" in source
    assert "record_eef_gripper_post_policy_step(env)" in source
    assert "validate_eef_runtime_safety(env, require_gripper_runtime=True)" in source


def test_candidate_report_schema_is_closed():
    report = _candidate_report(official=False, final=False)
    tampered = copy.deepcopy(report)
    tampered["hidden"] = True
    with pytest.raises(candidate.CandidateReplayValidationError, match="schema drift"):
        candidate.validate_candidate_report(
            tampered, variant="reasoning_43075", final=False
        )


def test_nested_candidate_report_schemas_are_closed():
    report = _candidate_report(official=False, final=False)
    for section in ("arm_slew_headroom", "gripper_close_arm_interlock"):
        tampered = copy.deepcopy(report)
        tampered[section]["hidden"] = True
        with pytest.raises(
            candidate.CandidateReplayValidationError, match="schema drift"
        ):
            candidate.validate_candidate_report(
                tampered, variant="reasoning_43075", final=False
            )


@pytest.mark.parametrize("variant", sorted(candidate.CANDIDATE_BY_VARIANT))
def test_candidate_replay_evidence_binds_exact_cadence_and_nominal_slew(variant):
    official = variant == "official_lap3b"
    report = _candidate_report(official=official, final=True)
    summary = candidate.validate_candidate_replay_evidence(
        _final_safety(official=official), report, variant=variant
    )
    assert summary == {
        "profile": "polaris_eef_candidate_exact_cadence_and_nominal_bound_v1",
        "arm_apply_calls": 976,
        "gripper_apply_calls": 976,
        "process_action_calls": 122,
        "post_policy_step_samples": 122,
        "slew_limit_events": 10,
        "dls_fallbacks": 0,
        "abort_count": 0,
        "nominal_applied_delta_bound_passed": True,
        "guard_diagnostics_empty": True,
    }


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda safety: safety["counters"].update(apply_calls=975), "apply cadence"),
        (lambda safety: safety["counters"].update(dls_fallbacks=1), "dls_fallbacks"),
        (
            lambda safety: safety["counters"].update(slew_limit_events=0),
            "did not exercise",
        ),
        (
            lambda safety: safety["maxima"]["applied_delta_joint_pos_rad"].__setitem__(
                0, 0.018125
            ),
            "applied-delta bound 0",
        ),
        (
            lambda safety: safety["gripper_runtime_dynamic"].update(
                post_policy_step_samples=121
            ),
            "sample cadence",
        ),
        (
            lambda safety: safety["gripper_runtime_dynamic"][
                "driver_target_slew"
            ].update(apply_calls=975),
            "target-slew cadence",
        ),
        (lambda safety: safety.update(guard_diagnostics=[{}]), "guard diagnostics"),
    ],
)
def test_candidate_replay_evidence_rejects_tampering(mutation, match):
    safety = _final_safety(official=False)
    mutation(safety)
    with pytest.raises(candidate.CandidateReplayValidationError, match=match):
        candidate.validate_candidate_replay_evidence(
            safety,
            _candidate_report(official=False, final=True),
            variant="reasoning_43075",
        )


def test_container_argument_is_closed_and_exact():
    assert candidate.validate_container_argument(
        "/images/isaac.sqsh", size_bytes=123, sha256="a" * 64
    ) == {
        "profile": "host_regular_nonsymlink_sha256_verified_before_pyxis_v1",
        "path": "/images/isaac.sqsh",
        "size_bytes": 123,
        "sha256": "a" * 64,
    }
    for kwargs in (
        {"image": "relative.sqsh", "size_bytes": 123, "sha256": "a" * 64},
        {"image": "/image", "size_bytes": 0, "sha256": "a" * 64},
        {"image": "/image", "size_bytes": 1, "sha256": "A" * 64},
    ):
        with pytest.raises(candidate.CandidateReplayValidationError):
            candidate.validate_container_argument(**kwargs)
