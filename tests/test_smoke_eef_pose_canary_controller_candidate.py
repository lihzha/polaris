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
            "enabled": True,
            "profile": "eef_gripper_close_hold_arm_86_physics_substeps_v1",
            "configured_substeps": 86,
            "remaining_substeps": 0,
            "observed_endpoint_change_count": 1 if official and final else 0,
            "endpoint_observed": final,
            "activation_count": 1 if official and final else 0,
            "active_apply_count": 86 if official and final else 0,
            "max_abs_active_delta_joint_pos_rad": [0.0] * 7,
            "released_apply_count": 10 if official and final else 0,
            "max_abs_released_delta_joint_pos_rad": vector,
        },
    }


def _failure_context():
    lifecycle = {
        "profile": "slurm_single_task_srun_lifecycle_v1",
        "launch_id": "a" * 64,
        "job_id": 123,
        "step_id": 4,
        "nodelist": "l401",
        "procid": 0,
        "localid": 0,
        "ntasks": 1,
    }
    gripper_contract = {
        "driver_target_slew": {"profile": candidate.CANDIDATE_TARGET_SLEW_PROFILE}
    }
    return {
        "lifecycle": lifecycle,
        "repository": {
            "path": "/repo",
            "commit": "b" * 40,
            "clean_tracked": True,
        },
        "container_image": candidate.validate_container_argument(
            "/container.sqsh", size_bytes=123, sha256="c" * 64
        ),
        "production_eval": {},
        "fixture": {"fixture_action_count": candidate.FIXTURE_ACTION_COUNT},
        "action_plan": {
            "profile": "exact_fixture_then_repeat_final_recorded_action_v1",
            "fixture_action_count": candidate.FIXTURE_ACTION_COUNT,
            "post_fixture_repeat_count": candidate.POST_FIXTURE_REPEAT_COUNT,
            "total_action_count": candidate.TOTAL_ACTION_COUNT,
        },
        "boundary_helper": {},
        "assets": {"contract": candidate.gate0.EXPECTED_ASSET_CONTRACT},
        "runtime_protocol": {
            "decimation": candidate.gate0.DECIMATION,
            "physics_hz": 120.0,
            "policy_hz": 15.0,
        },
        "runtime_frame": {
            "eef_frame": "panda_link8",
            "reference_frame": "panda_link0",
            "controlled_body": "panda_link8",
        },
        "gripper_runtime_contract": gripper_contract,
        "initial_safety": {
            "counters": {"apply_calls": 0},
            "current_joint_velocity_abort": None,
            "gripper_runtime_static": gripper_contract,
        },
        "initial_candidate": _candidate_report(official=True, final=False),
    }


@pytest.mark.parametrize(
    ("field", "boolean_alias"),
    (("procid", False), ("localid", False), ("ntasks", True)),
)
def test_failure_context_rejects_boolean_rank_and_task_aliases(field, boolean_alias):
    context = _failure_context()
    assert (
        candidate.validate_failure_context(context, variant="official_lap3b") == context
    )
    context["lifecycle"][field] = boolean_alias
    with pytest.raises(
        candidate.CandidateReplayValidationError,
        match="failure-context lifecycle drift",
    ):
        candidate.validate_failure_context(context, variant="official_lap3b")


def _final_safety(*, official: bool):
    physical = [0.018125] * 4 + [0.02175] * 3
    nominal = [value * 0.95 for value in physical]
    expected_changes = 1 if official else 0
    return {
        "counters": {
            "apply_calls": 1016,
            "environment_substeps": 1016,
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
            "apply_entry_samples": 1016,
            "post_policy_step_samples": 127,
            "nonfinite_samples": 0,
            "dropped_diagnostics": 0,
            "driver_target_slew": {
                "profile": candidate.CANDIDATE_TARGET_SLEW_PROFILE,
                "apply_calls": 1016,
                "live_limit_validation_count": 1016,
                "process_action_calls": 127,
                "initialization_count": 1,
                "endpoint_change_count": expected_changes,
                "repeated_endpoint_process_count": 126 - expected_changes,
                "slew_limited_apply_count": 75 if official else 0,
                "endpoint_reached_apply_count": 941 if official else 1016,
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
        ("active_apply_count", 85, "did not activate"),
        ("remaining_substeps", 1, "did not activate"),
        ("released_apply_count", 9, "did not activate"),
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


def test_reasoning_final_report_rejects_interlock_activation():
    report = _candidate_report(official=False, final=True)
    report["gripper_close_arm_interlock"]["activation_count"] = 1
    with pytest.raises(
        candidate.CandidateReplayValidationError, match="unexpectedly activated"
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


def test_runner_binds_exact_fixture_plus_seven_action_release_probe():
    assert candidate.FIXTURE_ACTION_COUNT == 120
    assert candidate.POST_FIXTURE_REPEAT_COUNT == 7
    assert candidate.TOTAL_ACTION_COUNT == 127
    source = (SCRIPTS / "smoke_eef_pose_canary_controller_candidate.py").read_text()
    assert source.count("env_cfg.actions.arm.enable_arm_slew_headroom = True") == 1
    assert (
        source.count("env_cfg.actions.arm.enable_gripper_close_arm_interlock = True")
        == 1
    )
    assert (
        source.count(
            "env_cfg.actions.finger_joint.enable_target_slew_rate_0p25_candidate = True"
        )
        == 1
    )
    assert "list(actions) + [list(actions[-1])] * POST_FIXTURE_REPEAT_COUNT" in source
    assert "record_eef_gripper_post_policy_step(env)" in source
    assert source.count("validate_eef_runtime_safety(") == 2
    assert source.count("expected_gripper_target_slew_profile=") == 2


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
        "arm_apply_calls": 1016,
        "gripper_apply_calls": 1016,
        "process_action_calls": 127,
        "post_policy_step_samples": 127,
        "target_slew_profile": candidate.CANDIDATE_TARGET_SLEW_PROFILE,
        "target_slew_limited_apply_count": 75 if official else 0,
        "target_slew_endpoint_reached_apply_count": 941 if official else 1016,
        "slew_limit_events": 10,
        "dls_fallbacks": 0,
        "abort_count": 0,
        "nominal_applied_delta_bound_passed": True,
        "guard_diagnostics_empty": True,
    }


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda safety: safety["counters"].update(apply_calls=1015), "apply cadence"),
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
                post_policy_step_samples=126
            ),
            "sample cadence",
        ),
        (
            lambda safety: safety["gripper_runtime_dynamic"][
                "driver_target_slew"
            ].update(apply_calls=1015),
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


def _abort_capture(monkeypatch):
    message = (
        "joint='panda_joint5', policy_step=117, physics_substep=6, "
        f"evidence_sha256={'a' * 64})"
    )
    target = {"profile": candidate.CANDIDATE_TARGET_SLEW_PROFILE}
    arm = {
        "ik_safety": {
            "current_joint_velocity_abort": {"profile": "captured"},
            "gripper_runtime_dynamic": {"driver_target_slew": target},
        },
        "controller_substep_trace": {"entries": ["validated"]},
    }

    def validate_arm(value, *, expected_failure):
        assert value["controller_substep_trace"] == {"entries": ["validated"]}
        assert expected_failure["policy_step"] == 117
        return value

    def validate_tail(value, *, expected_failure):
        assert value == {"entries": ["all-six-validated"]}
        assert expected_failure["physics_substep"] == 6
        return value

    monkeypatch.setattr(
        candidate.gate0, "_validate_arm_failure_runtime_evidence", validate_arm
    )
    monkeypatch.setattr(candidate.gate0, "validate_gripper_tail", validate_tail)
    return {
        "profile": candidate.CONTROLLER_ABORT_CAPTURE_PROFILE,
        "failure_exception": {
            "type": "polaris.robust_differential_ik.DifferentialIKInvariantError",
            "message": message,
            "traceback": "original traceback",
        },
        "parsed_failure": {
            "joint_name": "panda_joint5",
            "policy_step": 117,
            "physics_substep": 6,
            "evidence_sha256": "a" * 64,
        },
        "arm_failure_runtime_evidence": arm,
        "all_six_gripper_tail": {"entries": ["all-six-validated"]},
        "active_safety": arm["ik_safety"],
        "active_candidate": _candidate_report(official=True, final=False),
        "active_target_slew": target,
    }


def test_transactional_abort_capture_requires_arm_ring_all_six_and_target_state(
    monkeypatch,
):
    capture = _abort_capture(monkeypatch)
    assert (
        candidate.validate_controller_abort_capture(capture, variant="official_lap3b")
        == capture
    )
    for field in (
        "arm_failure_runtime_evidence",
        "all_six_gripper_tail",
        "active_target_slew",
    ):
        tampered = copy.deepcopy(capture)
        tampered.pop(field)
        with pytest.raises(
            candidate.CandidateReplayValidationError, match="schema drift"
        ):
            candidate.validate_controller_abort_capture(
                tampered, variant="official_lap3b"
            )


def test_built_abort_capture_is_retained_before_forced_validation_failure(
    monkeypatch,
):
    state = {"controller_abort_capture": None}
    payload = {"built": "all diagnostics"}

    def reject(value, *, variant):
        assert value is payload
        assert variant == "official_lap3b"
        assert state["controller_abort_capture"] is payload
        raise candidate.CandidateReplayValidationError("forced secondary failure")

    monkeypatch.setattr(candidate, "validate_controller_abort_capture", reject)
    with pytest.raises(
        candidate.CandidateReplayValidationError,
        match="forced secondary failure",
    ):
        candidate._retain_then_validate_controller_abort_capture(
            state,
            payload,
            variant="official_lap3b",
        )
    assert state["controller_abort_capture"] is payload


def test_incomplete_abort_preserves_primary_and_built_capture_without_promotion(
    monkeypatch,
):
    context = {"initial_candidate": _candidate_report(official=True, final=False)}
    monkeypatch.setattr(
        candidate,
        "validate_failure_context",
        lambda value, *, variant: value,
    )
    primary = {
        "type": "polaris.robust_differential_ik.DifferentialIKNumericalError",
        "message": "non-parseable primary numerical failure",
        "traceback": "primary traceback",
    }
    secondary = {
        "type": "smoke_eef_pose_canary_trace_replay.Gate0ReplayValidationError",
        "message": "failure message contract",
        "traceback": "secondary traceback",
    }
    built_capture = {"built": "diagnostics retained before validation"}
    payload = {
        "schema_version": 1,
        "profile": candidate.PROFILE,
        "finalized": False,
        "passed": False,
        "stage": "failed_controller_abort_capture_incomplete",
        "environment": candidate.gate0.ENVIRONMENT,
        "variant": "official_lap3b",
        "candidate": candidate.CANDIDATE_BY_VARIANT["official_lap3b"],
        "policy_step": 7,
        "failure_context": context,
        "failure": primary,
        "controller_abort_capture": built_capture,
        "controller_abort_capture_failure": secondary,
        "close_failures": [],
    }
    validated = candidate.validate_failure_payload(
        payload,
        variant="official_lap3b",
        require_complete_capture=False,
    )
    assert validated["failure"] == primary
    assert validated["controller_abort_capture"] is built_capture


def test_failure_path_has_separate_postjob_verifier_and_no_ready_publication():
    source = (SCRIPTS / "smoke_eef_pose_canary_controller_candidate.py").read_text()
    success_marker = source.index("POLARIS_CONTROLLER_CANDIDATE_READY")
    failure_start = source.index("    except BaseException as error:", success_marker)
    except_source = source[failure_start:]
    assert "validate_failure_payload(" in except_source
    assert "failure_context" in except_source
    assert ".ready.json" not in except_source
    assert "_build_controller_abort_capture(" in source
    assert "_retain_then_validate_controller_abort_capture(" in source
    verifier = SCRIPTS / "verify_eef_pose_canary_controller_candidate_failure.py"
    verifier_source = verifier.read_text()
    assert "validator.validate_failure(args)" in verifier_source
    assert "POLARIS_CONTROLLER_CANDIDATE_FAILURE_VERIFY_PASS" in verifier_source
