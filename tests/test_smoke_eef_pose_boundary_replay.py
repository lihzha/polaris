from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import pytest

from scripts import finalize_eef_pose_boundary_replay as finalizer
from scripts import smoke_eef_pose_boundary_replay as smoke


class _FakeTensor:
    def __init__(self, values):
        self._values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def flatten(self):
        return self

    def tolist(self):
        return self._values


def test_failure_vector_evidence_replaces_nonfinite_values_with_null() -> None:
    evidence = smoke._finite_vector_evidence(  # noqa: SLF001
        _FakeTensor([1.0, float("nan"), float("inf"), -2.0])
    )

    assert evidence == {
        "values": [1.0, None, None, -2.0],
        "finite_mask": [True, False, False, True],
        "finite_count": 2,
    }
    assert b"NaN" not in smoke._strict_json_bytes(evidence)  # noqa: SLF001


def _diagnostic_vector(values):
    return {
        "values": list(values),
        "finite_mask": [True] * 7,
        "finite_count": 7,
    }


def _safety_report(*, episode_index=0, apply_calls=smoke.EXPECTED_APPLY_CALLS):
    active = apply_calls > 0
    counters = {
        "apply_calls": apply_calls,
        "environment_substeps": apply_calls,
        "slew_limit_events": 256 if active else 0,
        "slew_limited_joints": 512 if active else 0,
        "position_limit_events": 128 if active else 0,
        "position_limited_joints": 128 if active else 0,
        "post_clamp_target_violations": 0,
        "current_joint_limit_aborts": 0,
        "invariant_aborts": 0,
        "nonfinite_aborts": 0,
        "dls_fallbacks": 0,
        "guard_diagnostics_dropped": 0,
    }
    raw_delta = [0.05] * 7 if active else [0.0] * 7
    maxima = {
        "raw_delta_joint_pos_rad": raw_delta,
        "applied_delta_joint_pos_rad": (
            list(smoke.EXPECTED_MAX_DELTA_RAD) if active else [0.0] * 7
        ),
        "raw_target_soft_limit_violation_rad": (
            [0.0, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0] if active else [0.0] * 7
        ),
        "post_clamp_target_soft_limit_violation_rad": [0.0] * 7,
        "post_clamp_target_guard_band_violation_rad": [0.0] * 7,
        "current_joint_soft_limit_violation_rad": [0.0] * 7,
    }
    q = [(lower + upper) / 2.0 for lower, upper in smoke.EXPECTED_TARGET_LIMITS_RAD]
    raw_target = [value + delta for value, delta in zip(q, raw_delta, strict=True)]
    max_diagnostic = (
        {
            "kind": "max_raw_delta",
            "episode_index": episode_index,
            "policy_step": 10,
            "physics_substep": 3,
            "joint_pos_rad": _diagnostic_vector(q),
            "raw_delta_joint_pos_rad": _diagnostic_vector(raw_delta),
            "raw_joint_pos_target_rad": _diagnostic_vector(raw_target),
            "safe_joint_pos_target_rad": _diagnostic_vector(q),
            "pose_error_norm": 0.5,
            "jacobian_finite": True,
            "jacobian_max_abs": 1.0,
            "eef_quaternion_norm": None,
        }
        if active
        else None
    )
    return {
        "episode_index": episode_index,
        "profile": "panda_velocity_softlimit_guardband_v2",
        "apply_actions_cadence": "physics_substep",
        "physics_dt": 1.0 / 120.0,
        "control_dt": 1.0 / 15.0,
        "decimation": 8,
        "current_joint_soft_limit_tolerance_rad": 1e-5,
        "target_soft_limit_guard_band_profile": (
            "one_physics_substep_velocity_bound_v1"
        ),
        "eef_quaternion_unit_norm_tolerance": 1e-3,
        "joint_slew_float32_tolerance_rad": 1e-6,
        "soft_joint_pos_limit_factor": 1.0,
        "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
        "joint_velocity_limits_rad_s": list(smoke.EXPECTED_VELOCITY_LIMITS_RAD_S),
        "joint_effort_limits": list(smoke.EXPECTED_EFFORT_LIMITS),
        "max_delta_joint_pos_rad": list(smoke.EXPECTED_MAX_DELTA_RAD),
        "target_soft_limit_margin_rad": list(smoke.EXPECTED_MAX_DELTA_RAD),
        "target_joint_pos_limits_rad": copy.deepcopy(smoke.EXPECTED_TARGET_LIMITS_RAD),
        "target_joint_pos_limits_float32_sha256": smoke.TARGET_LIMIT_DIGEST,
        "soft_joint_pos_limits_rad": copy.deepcopy(smoke.EXPECTED_OUTER_LIMITS_RAD),
        "soft_joint_pos_limits_float32_sha256": smoke.SOFT_LIMIT_DIGEST,
        "counters": counters,
        "maxima": maxima,
        "guard_diagnostics": [],
        "max_raw_delta_diagnostic": max_diagnostic,
    }


def _boundary_result():
    arm_q = [(lower + upper) / 2.0 for lower, upper in smoke.EXPECTED_OUTER_LIMITS_RAD]
    arm_q[smoke.TARGET_JOINT_INDEX] = 2.88
    arm_target = [
        (lower + upper) / 2.0 for lower, upper in smoke.EXPECTED_TARGET_LIMITS_RAD
    ]
    arm_target[smoke.TARGET_JOINT_INDEX] = smoke.INNER_UPPER_LIMIT_RAD
    records = []
    for drive_step in range(smoke.ADAPTIVE_DRIVE_STEPS):
        records.append(
            {
                "drive_step": drive_step,
                "policy_step": smoke.EXPECTED_ACTION_ENCODING["action_count"]
                + drive_step,
                "position_limit_events_delta": 8,
                "joint_pos_rad": arm_q[smoke.TARGET_JOINT_INDEX],
                "joint_vel_rad_s": 0.0,
                "joint_target_rad": smoke.INNER_UPPER_LIMIT_RAD,
                "predicted_outward_joint_delta_rad": 0.1,
                "arm_joint_pos_rad": list(arm_q),
                "arm_joint_vel_rad_s": [0.0] * 7,
                "arm_joint_target_rad": list(arm_target),
                "eef_position_m": [0.4, -0.2, 0.2],
                "eef_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "target_is_inner_limit": True,
                "within_outer_limits": True,
                "all_arm_joints_within_outer_limits": True,
                "eef_state_is_finite": True,
                "state_is_finite": True,
            }
        )
    return {
        "target_joint_name": smoke.TARGET_JOINT_NAME,
        "target_joint_index": smoke.TARGET_JOINT_INDEX,
        "direction": "upper",
        "replay_action_count": smoke.EXPECTED_ACTION_ENCODING["action_count"],
        "adaptive_drive_steps": smoke.ADAPTIVE_DRIVE_STEPS,
        "total_policy_steps": smoke.EXPECTED_TOTAL_POLICY_STEPS,
        "expected_apply_calls": smoke.EXPECTED_APPLY_CALLS,
        "outward_delta_scale": smoke.OUTWARD_DELTA_SCALE,
        "required_consecutive_dwell_policy_steps": (
            smoke.REQUIRED_CONSECUTIVE_DWELL_STEPS
        ),
        "observed_max_consecutive_dwell_policy_steps": smoke.ADAPTIVE_DRIVE_STEPS,
        "joint_outer_lower_limit_rad": smoke.OUTER_LOWER_LIMIT_RAD,
        "joint_outer_upper_limit_rad": smoke.OUTER_UPPER_LIMIT_RAD,
        "joint_inner_target_upper_limit_rad": smoke.INNER_UPPER_LIMIT_RAD,
        "joint_pos_observed_min_rad": 0.0,
        "joint_pos_observed_max_rad": 2.88,
        "terminated": False,
        "truncated": False,
        "state_is_finite": True,
        "dwell_records": records,
    }


def _identity(path: Path, *, forced_sha256=None):
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "size_bytes": len(data),
        "sha256": forced_sha256 or hashlib.sha256(data).hexdigest(),
        "mode": f"{path.stat().st_mode & 0o777:04o}",
    }


def _success_payload():
    fixture, _ = smoke.load_replay_fixture()
    dummy = {
        "path": "/pinned/food_bussing/scene.usda",
        "size_bytes": 1,
        "sha256": smoke.EXPECTED_ASSET_CONTRACT["scene_sha256"],
        "mode": "0444",
    }
    initial = {
        **dummy,
        "path": "/pinned/food_bussing/initial_conditions.json",
        "sha256": smoke.EXPECTED_ASSET_CONTRACT["initial_conditions_sha256"],
    }
    metadata = {
        filename: {
            **dummy,
            "path": f"/pinned/metadata/{filename}.metadata",
            "revision": smoke.EXPECTED_ASSET_CONTRACT["polaris_hub_revision"],
        }
        for filename in ("initial_conditions.json", "scene.usda")
    }
    return {
        "schema_version": 1,
        "fixture_profile": smoke.FIXTURE_PROFILE,
        "smoke_profile": smoke.SMOKE_PROFILE,
        "finalized": False,
        "passed": True,
        "stage": "simulation_app_close_pending",
        "exit_code": 0,
        "environment": smoke.ENVIRONMENT,
        "fixture": fixture,
        "assets": {
            "scene": dummy,
            "initial_conditions": initial,
            "polaris_hub_revision": smoke.EXPECTED_ASSET_CONTRACT[
                "polaris_hub_revision"
            ],
            "revision_metadata": metadata,
            "initial_condition_index": 0,
        },
        "runtime_protocol": {
            "episode_steps": 450,
            "policy_hz": 15.0,
            "step_dt": 1.0 / 15.0,
            "physics_hz": 120.0,
            "physics_dt": 1.0 / 120.0,
            "decimation": 8,
        },
        "runtime_frame": {
            "eef_frame": "panda_link8",
            "reference_frame": "panda_link0",
            "position_error_m": 0.0,
            "rotation_error_rad": 0.0,
            "controlled_body": "panda_link8",
            "body_offset": "identity",
            "command_type": "pose",
            "use_relative_mode": False,
            "ik_method": "dls",
            "dls_damping": 0.01,
            "arm_scale": 1.0,
            "arm_joint_names": [f"panda_joint{index}" for index in range(1, 8)],
            "gripper_threshold_profile": (
                "closed_positive_ge_0p5_inverse_open_gt_0p5_v1"
            ),
            "ik_safety_profile": "panda_velocity_softlimit_guardband_v2",
            "action_dim": 7,
        },
        "initial_ik_safety_capture": _safety_report(episode_index=None, apply_calls=0),
        "boundary": _boundary_result(),
        "ik_safety": _safety_report(),
        "failure": None,
        "close_failures": [],
    }


def test_fixture_exact_identity_and_decoded_action_contract():
    identity, actions = smoke.load_replay_fixture()

    assert identity["size_bytes"] == 15967
    assert identity["sha256"] == smoke.EXPECTED_FIXTURE_SHA256
    assert identity["source_trace_sha256"] == smoke.EXPECTED_SOURCE["trace_sha256"]
    assert (
        identity["action_float32_sha256"]
        == (smoke.EXPECTED_ACTION_ENCODING["uncompressed_sha256"])
    )
    assert len(actions) == 378
    assert actions[0] == pytest.approx(
        [
            0.3568142354488373,
            -0.0012728864094242454,
            0.49005618691444397,
            -0.007359412964433432,
            0.9999475479125977,
            0.005898015107959509,
            -0.003998896572738886,
            0.0,
        ],
        abs=0.0,
    )
    assert actions[-1][-1] == 1.0
    assert all(
        abs(math.sqrt(sum(v * v for v in a[3:7])) - 1.0) <= 1e-3 for a in actions
    )


def test_fixture_parser_rejects_schema_source_encoding_and_payload_mutations():
    fixture = smoke.strict_json_loads(smoke.FIXTURE_PATH.read_bytes(), field="fixture")
    mutations = []

    extra = copy.deepcopy(fixture)
    extra["extra"] = True
    mutations.append(extra)
    source = copy.deepcopy(fixture)
    source["source"]["failed_policy_step"] -= 1
    mutations.append(source)
    encoding = copy.deepcopy(fixture)
    encoding["action_encoding"]["action_count"] -= 1
    mutations.append(encoding)
    payload = copy.deepcopy(fixture)
    payload["actions_zlib_base64_chunks"][0] = (
        "A" + payload["actions_zlib_base64_chunks"][0][1:]
    )
    mutations.append(payload)

    for mutation in mutations:
        with pytest.raises(smoke.BoundaryReplayValidationError):
            smoke.decode_replay_fixture(mutation)


def test_fixture_file_identity_rejects_byte_tamper(tmp_path):
    tampered = tmp_path / smoke.FIXTURE_PATH.name
    tampered.write_bytes(smoke.FIXTURE_PATH.read_bytes() + b"\n")

    with pytest.raises(smoke.BoundaryReplayValidationError, match="file size"):
        smoke.load_replay_fixture(tampered)


def test_strict_json_rejects_duplicates_and_nonfinite_constants():
    with pytest.raises(smoke.BoundaryReplayValidationError, match="duplicate"):
        smoke.strict_json_loads(b'{"a":1,"a":2}', field="test")
    with pytest.raises(smoke.BoundaryReplayValidationError, match="constant"):
        smoke.strict_json_loads(b'{"a":NaN}', field="test")


def test_boundary_evidence_accepts_exact_full_state_dwell():
    summary = smoke.validate_boundary_result(_boundary_result(), _safety_report())

    assert summary["apply_calls"] == 3152
    assert summary["max_consecutive_dwell_policy_steps"] == 16
    assert summary["joint5_raw_outer_violation_rad"] == 0.04


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda boundary, safety: boundary["dwell_records"][-1].__setitem__(
                "position_limit_events_delta", 7
            ),
            "last eight",
        ),
        (
            lambda boundary, safety: boundary["dwell_records"][0][
                "arm_joint_pos_rad"
            ].__setitem__(0, 3.0),
            "all-arm limit",
        ),
        (
            lambda boundary, safety: boundary["dwell_records"][0][
                "arm_joint_target_rad"
            ].__setitem__(0, 3.0),
            "target guard",
        ),
        (
            lambda boundary, safety: boundary["dwell_records"][0][
                "eef_quaternion_wxyz"
            ].__setitem__(0, 0.0),
            "quaternion norm",
        ),
        (
            lambda boundary, safety: safety["counters"].__setitem__(
                "current_joint_limit_aborts", 1
            ),
            "must be zero",
        ),
        (
            lambda boundary, safety: safety["maxima"][
                "raw_target_soft_limit_violation_rad"
            ].__setitem__(4, 0.0),
            "never drove joint5",
        ),
        (
            lambda boundary, safety: safety.__setitem__(
                "target_joint_pos_limits_float32_sha256", "0" * 64
            ),
            "target_joint_pos_limits_float32_sha256",
        ),
    ],
)
def test_boundary_evidence_rejects_mutations(mutation, message):
    boundary = _boundary_result()
    safety = _safety_report()
    mutation(boundary, safety)

    with pytest.raises(smoke.BoundaryReplayValidationError, match=message):
        smoke.validate_boundary_result(boundary, safety)


def test_success_payload_is_closed_and_binds_initial_capture():
    payload = _success_payload()
    smoke.validate_success_payload(payload)

    extra = copy.deepcopy(payload)
    extra["extra"] = True
    with pytest.raises(smoke.BoundaryReplayValidationError, match="schema"):
        smoke.validate_success_payload(extra)

    dirty_initial = copy.deepcopy(payload)
    dirty_initial["initial_ik_safety_capture"]["counters"]["apply_calls"] = 1
    with pytest.raises(smoke.BoundaryReplayValidationError, match="initial"):
        smoke.validate_success_payload(dirty_initial)


def test_immutable_raw_publication_is_nonoverwriting_and_mode_0444(tmp_path):
    path = tmp_path / "result.json"
    identity = smoke._atomic_write_immutable(path, {"schema_version": 1})

    assert identity["mode"] == "0444"
    assert path.stat().st_mode & 0o777 == 0o444
    with pytest.raises(FileExistsError):
        smoke._atomic_write_immutable(path, {"schema_version": 1})


def test_host_finalizer_reconstructs_raw_ready_and_provenance(tmp_path, monkeypatch):
    repo = tmp_path / "PolaRiS"
    (repo / "scripts" / "fixtures").mkdir(parents=True)
    runner = repo / "scripts" / "smoke_eef_pose_boundary_replay.py"
    fixture_path = (
        repo
        / "scripts"
        / "fixtures"
        / "official_lap3b_foodbussing_v3_boundary_actions.json"
    )
    shutil.copy2(Path(smoke.__file__), runner)
    shutil.copy2(smoke.FIXTURE_PATH, fixture_path)
    fixture, _ = smoke.load_replay_fixture(fixture_path)

    payload = _success_payload()
    payload["fixture"] = fixture
    job_id = 12345
    raw_path = tmp_path / f"boundary-replay-smoke-{job_id}.json"
    raw_identity = smoke._atomic_write_immutable(raw_path, payload)
    marker_path = raw_path.with_name(raw_path.name + ".ready.json")
    smoke._atomic_write_immutable(
        marker_path,
        {
            "schema_version": 1,
            "stage": "simulation_app_close_pending",
            "raw_result": raw_identity,
        },
    )
    image = tmp_path / "image.sqsh"
    image.write_bytes(b"image")
    runtime_script = tmp_path / "job.sh"
    saved_script = tmp_path / "job.saved.sh"
    runtime_script.write_text("#!/bin/bash\ntrue\n")
    saved_script.write_bytes(runtime_script.read_bytes())
    runtime_script.chmod(0o444)
    saved_script.chmod(0o444)
    attestation = tmp_path / f"boundary-replay-smoke-{job_id}.attestation.json"
    commit = "a" * 40
    monkeypatch.setenv("SLURM_JOB_ID", str(job_id))
    monkeypatch.setenv("SLURM_NODELIST", "pool0-00005")
    monkeypatch.setattr(
        finalizer,
        "_git",
        lambda _repo, *arguments: (
            "" if arguments == ("status", "--porcelain") else commit
        ),
    )
    monkeypatch.setattr(
        finalizer,
        "_validate_asset_identities",
        lambda raw: {"validated": True},
    )
    monkeypatch.setattr(finalizer.smoke, "__file__", str(runner))
    args = argparse.Namespace(
        raw_result=raw_path,
        attestation=attestation,
        srun_rc=0,
        job_id=job_id,
        runtime_job_script=runtime_script,
        saved_job_script=saved_script,
        polaris_repo=repo,
        expected_polaris_commit=commit,
        expected_runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
        expected_fixture_sha256=smoke.EXPECTED_FIXTURE_SHA256,
        container_image=image,
        expected_image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        expected_finalizer_sha256=hashlib.sha256(
            Path(finalizer.__file__).read_bytes()
        ).hexdigest(),
        expected_saved_job_script_sha256=hashlib.sha256(
            saved_script.read_bytes()
        ).hexdigest(),
    )

    expected = finalizer.build_expected_attestation(args)

    assert expected["passed"] is True
    assert expected["raw_result"]["ready_marker"]["mode"] == "0444"
    assert expected["provenance"]["fixture_source"] == smoke.EXPECTED_SOURCE
    assert expected["provenance"]["slurm"] == {
        "job_id": job_id,
        "nodelist": "pool0-00005",
    }

    cli = [
        "finalize_eef_pose_boundary_replay.py",
        "finalize",
        "--raw-result",
        str(raw_path),
        "--attestation",
        str(attestation),
        "--srun-rc",
        "0",
        "--job-id",
        str(job_id),
        "--runtime-job-script",
        str(runtime_script),
        "--saved-job-script",
        str(saved_script),
        "--polaris-repo",
        str(repo),
        "--expected-polaris-commit",
        commit,
        "--expected-runner-sha256",
        args.expected_runner_sha256,
        "--expected-fixture-sha256",
        args.expected_fixture_sha256,
        "--container-image",
        str(image),
        "--expected-image-sha256",
        args.expected_image_sha256,
        "--expected-finalizer-sha256",
        args.expected_finalizer_sha256,
        "--expected-saved-job-script-sha256",
        args.expected_saved_job_script_sha256,
    ]
    monkeypatch.setattr(sys, "argv", cli)
    assert finalizer.main() == 0
    assert attestation.stat().st_mode & 0o777 == 0o444
    cli[1] = "verify"
    monkeypatch.setattr(sys, "argv", cli)
    assert finalizer.main() == 0

    bad_marker = json.loads(marker_path.read_text())
    bad_marker["raw_result"]["sha256"] = "0" * 64
    marker_path.chmod(0o640)
    marker_path.write_text(json.dumps(bad_marker))
    marker_path.chmod(0o444)
    with pytest.raises(finalizer.FinalizationError, match="ready marker"):
        finalizer.build_expected_attestation(args)
