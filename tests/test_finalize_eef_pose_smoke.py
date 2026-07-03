from argparse import Namespace
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts import finalize_eef_pose_smoke as finalizer


def _vector_evidence(values):
    return {
        "values": values,
        "finite_mask": [True] * 7,
        "finite_count": 7,
        "max_abs": max(map(abs, values)),
    }


def _diagnostic_vector(values):
    return {
        "values": values,
        "finite_mask": [True] * 7,
        "finite_count": 7,
    }


def _safety_report(episode_index, apply_calls, *, adversarial=False):
    counters = {name: 0 for name in finalizer.COUNTER_FIELDS}
    counters["apply_calls"] = apply_calls
    counters["environment_substeps"] = apply_calls
    if adversarial:
        counters["slew_limit_events"] = 1
        counters["slew_limited_joints"] = 1
    maxima = {name: [0.0] * 7 for name in finalizer.MAXIMA_FIELDS}
    q = [(lower + upper) / 2 for lower, upper in finalizer.EXPECTED_LIMITS]
    max_raw = None
    if apply_calls:
        max_raw = {
            "kind": "max_raw_delta",
            "episode_index": episode_index,
            "policy_step": 0,
            "physics_substep": 0,
            "joint_pos_rad": _diagnostic_vector(q),
            "raw_delta_joint_pos_rad": _diagnostic_vector([0.0] * 7),
            "raw_joint_pos_target_rad": _diagnostic_vector(q),
            "safe_joint_pos_target_rad": _diagnostic_vector(q),
            "pose_error_norm": 0.0,
            "jacobian_finite": True,
            "jacobian_max_abs": 1.0,
            "eef_quaternion_norm": None,
        }
    return {
        "episode_index": episode_index,
        "profile": "panda_velocity_softlimit_v1",
        "apply_actions_cadence": "physics_substep",
        "physics_dt": 1.0 / 120.0,
        "control_dt": 1.0 / 15.0,
        "decimation": 8,
        "current_joint_soft_limit_tolerance_rad": 1e-5,
        "eef_quaternion_unit_norm_tolerance": 1e-3,
        "joint_slew_float32_tolerance_rad": 1e-6,
        "soft_joint_pos_limit_factor": 1.0,
        "joint_names": finalizer.EXPECTED_JOINT_NAMES,
        "joint_velocity_limits_rad_s": finalizer.EXPECTED_VELOCITY_LIMITS,
        "joint_effort_limits": finalizer.EXPECTED_EFFORT_LIMITS,
        "max_delta_joint_pos_rad": finalizer.EXPECTED_MAX_DELTA,
        "soft_joint_pos_limits_rad": finalizer.EXPECTED_LIMITS,
        "soft_joint_pos_limits_float32_sha256": finalizer.EXPECTED_DIGEST,
        "counters": counters,
        "maxima": maxima,
        "guard_diagnostics": [],
        "max_raw_delta_diagnostic": max_raw,
    }


def _valid_raw_result():
    q = [(lower + upper) / 2 for lower, upper in finalizer.EXPECTED_LIMITS]
    reports = [_safety_report(index, 360) for index in range(13)]
    adversarial_safety = _safety_report(13, 8, adversarial=True)
    payload = {
        "schema_version": 1,
        "finalized": False,
        "passed": False,
        "stage": "simulation_app_close_pending",
        "case": None,
        "exit_code": 0,
        "failure": None,
        "close_failures": [],
        "persistence_failures": [],
        "environment": "DROID-FoodBussing",
        "eef_frame": "panda_link8",
        "hold_steps": 45,
        "position_delta_m": 0.04,
        "rotation_delta_deg": 15.0,
        "position_tolerance_m": 0.01,
        "rotation_tolerance_deg": 5.0,
        "frame_position_tolerance_m": 1e-5,
        "frame_rotation_tolerance_deg": 0.01,
        "raw_ik_safety_capture": _safety_report(None, 0),
        "results": [
            {
                "case": case,
                "passed": True,
                "position_error_m": 0.0,
                "rotation_error_rad": 0.0,
                "target_position": [0.0, 0.0, 0.0],
                "actual_position": [0.0, 0.0, 0.0],
                "target_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "actual_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "reset_frame_position_error_m": 0.0,
                "reset_frame_rotation_error_rad": 0.0,
                "final_frame_position_error_m": 0.0,
                "final_frame_rotation_error_rad": 0.0,
            }
            for case in finalizer.EXPECTED_CASES
        ],
        "ik_safety_episodes": reports,
        "ik_safety_adversarial": {
            "case": "oversized absolute +x target for one policy step",
            "passed": True,
            "state_is_finite": True,
            "eef_state_is_finite": True,
            "joint_state_is_finite": True,
            "joint_pos_within_captured_soft_limits": True,
            "terminated": False,
            "truncated": False,
            "guard_error": "",
            "joint_state": {
                "joint_names": finalizer.EXPECTED_JOINT_NAMES,
                "position_within_captured_soft_limits": True,
                "soft_limit_tolerance_rad": 1e-5,
                "joint_pos_rad": _vector_evidence(q),
                "joint_vel_rad_s": _vector_evidence([0.0] * 7),
                "soft_limit_violation_rad": _vector_evidence([0.0] * 7),
            },
            "ik_safety": adversarial_safety,
            "guard_evidence": {
                "apply_calls": 8,
                "slew_limit_events": 1,
                "abort_count": 0,
                "post_clamp_target_violations": 0,
                "applied_within_bounds": True,
            },
        },
    }
    return copy.deepcopy(payload)


def _write_immutable_json(path: Path, payload) -> bytes:
    data = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
    path.write_bytes(data)
    path.chmod(0o444)
    return data


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments], text=True
    ).strip()


def _attestation_args(tmp_path: Path) -> Namespace:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    smoke_source = repo / "scripts" / "smoke_eef_pose_controller.py"
    smoke_source.write_text("# reviewed synthetic smoke\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "synthetic"], check=True)

    image = tmp_path / "image.sqsh"
    image.write_bytes(b"synthetic pinned image")
    saved_job = tmp_path / "saved.sbatch"
    runtime_job = tmp_path / "runtime.sbatch"
    saved_job.write_text("#!/bin/bash\n# immutable\n")
    runtime_job.write_bytes(saved_job.read_bytes())

    job_id = 12345
    raw_path = tmp_path / f"smoke-{job_id}.json"
    raw_bytes = _write_immutable_json(raw_path, _valid_raw_result())
    marker_path = raw_path.with_name(raw_path.name + ".ready.json")
    _write_immutable_json(
        marker_path,
        {
            "schema_version": 1,
            "stage": "simulation_app_close_pending",
            "raw_result": {
                "path": str(raw_path),
                "size_bytes": len(raw_bytes),
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "mode": "0444",
            },
        },
    )
    return Namespace(
        raw_result=raw_path,
        attestation=tmp_path / f"smoke-{job_id}.attestation.json",
        srun_rc=0,
        job_id=job_id,
        runtime_job_script=runtime_job,
        saved_job_script=saved_job,
        polaris_repo=repo,
        expected_polaris_commit=_git(repo, "rev-parse", "HEAD"),
        expected_smoke_sha256=hashlib.sha256(smoke_source.read_bytes()).hexdigest(),
        container_image=image,
        expected_image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
    )


def test_raw_smoke_gate_requires_pending_full_evidence():
    summary = finalizer._verify_raw(_valid_raw_result())
    assert summary["ordinary_pass_count"] == 13
    assert summary["adversarial"]["slew_limit_events"] == 1

    raw = _valid_raw_result()
    raw["finalized"] = True
    with pytest.raises(finalizer.VerificationError, match="finalized"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["ik_safety_episodes"][4]["counters"]["apply_calls"] = 359
    with pytest.raises(finalizer.VerificationError, match="apply_calls"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["raw_ik_safety_capture"]["soft_joint_pos_limits_rad"][0][0] += 0.01
    with pytest.raises(finalizer.VerificationError, match="soft_joint_pos_limits"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["unexpected"] = True
    with pytest.raises(finalizer.VerificationError, match="top-level schema"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["results"][2]["position_error_m"] = 0.02
    with pytest.raises(finalizer.VerificationError, match="position error"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["ik_safety_episodes"][3]["max_raw_delta_diagnostic"]["raw_delta_joint_pos_rad"][
        "values"
    ][0] = 0.1
    with pytest.raises(finalizer.VerificationError, match="max-raw/maxima"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["ik_safety_episodes"][5]["counters"]["dls_fallbacks"] = 1
    with pytest.raises(finalizer.VerificationError, match="dls_fallbacks"):
        finalizer._verify_raw(raw)


def test_attestation_is_bound_verified_and_nonoverwriting(tmp_path):
    args = _attestation_args(tmp_path)
    expected = finalizer._build_expected(args)
    finalizer._publish_nonoverwriting(args.attestation, expected)

    attestation, _, _ = finalizer._read_json_once(args.attestation, "attestation")
    assert attestation == finalizer._build_expected(args)
    assert args.attestation.stat().st_mode & 0o777 == 0o444
    with pytest.raises(finalizer.VerificationError, match="already exists"):
        finalizer._publish_nonoverwriting(args.attestation, expected)

    args.srun_rc = 9
    with pytest.raises(finalizer.VerificationError, match="srun_rc"):
        finalizer._build_expected(args)


def test_attestation_rejects_writable_or_mutated_evidence(tmp_path):
    args = _attestation_args(tmp_path)
    args.raw_result.chmod(0o644)
    with pytest.raises(finalizer.VerificationError, match="mode"):
        finalizer._build_expected(args)

    args = _attestation_args(tmp_path / "second")
    marker = args.raw_result.with_name(args.raw_result.name + ".ready.json")
    marker.chmod(0o644)
    with pytest.raises(finalizer.VerificationError, match="ready marker mode"):
        finalizer._build_expected(args)

    args = _attestation_args(tmp_path / "third")
    args.raw_result.chmod(0o644)
    args.raw_result.write_bytes(args.raw_result.read_bytes() + b" ")
    args.raw_result.chmod(0o444)
    with pytest.raises(finalizer.VerificationError, match="ready marker"):
        finalizer._build_expected(args)

    args = _attestation_args(tmp_path / "fourth")
    args.expected_image_sha256 = "0" * 64
    with pytest.raises(finalizer.VerificationError, match="image digest"):
        finalizer._build_expected(args)

    args = _attestation_args(tmp_path / "fifth")
    args.saved_job_script.write_text("tampered\n")
    with pytest.raises(finalizer.VerificationError, match="job script digest"):
        finalizer._build_expected(args)

    args = _attestation_args(tmp_path / "sixth")
    dirty_path = args.polaris_repo / "dirty.txt"
    dirty_path.write_text("dirty\n")
    with pytest.raises(finalizer.VerificationError, match="repo dirty"):
        finalizer._build_expected(args)
