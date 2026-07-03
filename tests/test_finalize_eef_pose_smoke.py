from argparse import Namespace
import copy
import hashlib
import json
import math
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
        "values": list(values),
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
    if adversarial:
        raw_delta = finalizer.EXPECTED_MAX_DELTA[0] + 0.01
        maxima["raw_delta_joint_pos_rad"][0] = raw_delta
        maxima["applied_delta_joint_pos_rad"][0] = finalizer.EXPECTED_MAX_DELTA[0]
        raw_delta_vector = [raw_delta] + [0.0] * 6
        raw_target = [q[0] + raw_delta] + q[1:]
        safe_target = [q[0] + finalizer.EXPECTED_MAX_DELTA[0]] + q[1:]
        max_raw["raw_delta_joint_pos_rad"] = _diagnostic_vector(raw_delta_vector)
        max_raw["raw_joint_pos_target_rad"] = _diagnostic_vector(raw_target)
        max_raw["safe_joint_pos_target_rad"] = _diagnostic_vector(safe_target)
    return {
        "episode_index": episode_index,
        "profile": "panda_velocity_physxlimit_slew0p8_solveriter1_v6",
        "apply_actions_cadence": "physics_substep",
        "physics_dt": 1.0 / 120.0,
        "control_dt": 1.0 / 15.0,
        "decimation": 8,
        "current_joint_soft_limit_tolerance_rad": 1e-5,
        "target_soft_limit_guard_band_profile": (
            "eef_physx_inner_hardlimit_one_substep_v2"
        ),
        "physx_hard_limit_profile": "outer_minus_one_velocity_substep_v1",
        "physx_derived_soft_limit_profile": (
            "isaaclab_midpoint_range_factor1_float32_v1"
        ),
        "physx_hard_limit_write_count": 1,
        "arm_velocity_target_profile": "zero_per_physics_substep_v1",
        "joint_target_slew_profile": (
            "velocity_dt_factor0p8_with_full_inward_hardlimit_recovery_v1"
        ),
        "joint_target_slew_factor": 0.8,
        "articulation_solver_profile": "tgs_position64_velocity1_eef_only_v1",
        "articulation_solver_readback": (
            "composed_usd_physx_articulation_api_all_env_roots_v1"
        ),
        "physx_solver_type": 1,
        "solver_position_iteration_count": 64,
        "solver_velocity_iteration_count": 1,
        "joint_velocity_limit_tolerance_rad_s": 1e-5,
        "eef_quaternion_unit_norm_tolerance": 1e-3,
        "joint_slew_float32_tolerance_rad": 1e-6,
        "soft_joint_pos_limit_factor": 1.0,
        "joint_names": finalizer.EXPECTED_JOINT_NAMES,
        "joint_velocity_limits_rad_s": finalizer.EXPECTED_VELOCITY_LIMITS,
        "joint_effort_limits": finalizer.EXPECTED_EFFORT_LIMITS,
        "joint_drive_stiffness": finalizer.EXPECTED_DRIVE_STIFFNESS,
        "joint_drive_damping": finalizer.EXPECTED_DRIVE_DAMPING,
        "max_delta_joint_pos_rad": finalizer.EXPECTED_MAX_DELTA,
        "target_soft_limit_margin_rad": finalizer.EXPECTED_PHYSICAL_MARGIN,
        "target_joint_pos_limits_rad": finalizer.EXPECTED_TARGET_LIMITS,
        "target_joint_pos_limits_float32_sha256": (finalizer.EXPECTED_TARGET_DIGEST),
        "physx_hard_joint_pos_limits_rad": finalizer.EXPECTED_TARGET_LIMITS,
        "physx_hard_joint_pos_limits_float32_sha256": (
            finalizer.EXPECTED_TARGET_DIGEST
        ),
        "physx_derived_soft_joint_pos_limits_rad": (
            [list(pair) for pair in finalizer.EXPECTED_PHYSX_DERIVED_SOFT_LIMITS]
        ),
        "physx_derived_soft_joint_pos_limits_float32_sha256": (
            finalizer.EXPECTED_PHYSX_DERIVED_SOFT_DIGEST
        ),
        "arm_velocity_target_rad_s": [0.0] * 7,
        "soft_joint_pos_limits_rad": finalizer.EXPECTED_LIMITS,
        "soft_joint_pos_limits_float32_sha256": finalizer.EXPECTED_DIGEST,
        "counters": counters,
        "maxima": maxima,
        "guard_diagnostics": [],
        "max_raw_delta_diagnostic": max_raw,
    }


def _case_results():
    hold_position = [0.3, 0.0, 0.5]
    hold_quaternion = [1.0, 0.0, 0.0, 0.0]
    targets = [(hold_position.copy(), hold_quaternion.copy())]
    for axis, sign in ((0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0), (2, 1.0), (2, -1.0)):
        position = hold_position.copy()
        position[axis] += sign * 0.04
        targets.append((position, hold_quaternion.copy()))
    half_angle = math.radians(15.0) / 2.0
    for axis, sign in ((0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0), (2, 1.0), (2, -1.0)):
        delta = [math.cos(half_angle), 0.0, 0.0, 0.0]
        delta[axis + 1] = sign * math.sin(half_angle)
        targets.append(
            (
                hold_position.copy(),
                finalizer._quaternion_multiply_wxyz(hold_quaternion, delta),
            )
        )
    return [
        {
            "case": case,
            "passed": True,
            "position_error_m": 0.0,
            "rotation_error_rad": 0.0,
            "target_position": position,
            "actual_position": position.copy(),
            "target_quaternion_wxyz": quaternion,
            "actual_quaternion_wxyz": quaternion.copy(),
            "reset_frame_position_error_m": 0.0,
            "reset_frame_rotation_error_rad": 0.0,
            "final_frame_position_error_m": 0.0,
            "final_frame_rotation_error_rad": 0.0,
        }
        for case, (position, quaternion) in zip(
            finalizer.EXPECTED_CASES, targets, strict=True
        )
    ]


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
        "results": _case_results(),
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


def test_finalizer_accepts_coherent_full_inward_hardlimit_recovery():
    exact_threshold = _valid_raw_result()
    exact_threshold["ik_safety_episodes"][0]["maxima"]["applied_delta_joint_pos_rad"][
        0
    ] = finalizer._float32_add(finalizer.EXPECTED_MAX_DELTA[0], 1e-6)
    finalizer._verify_raw(exact_threshold)

    raw = _valid_raw_result()
    safety = raw["ik_safety_episodes"][0]
    safety["counters"]["position_limit_events"] = 1
    safety["counters"]["position_limited_joints"] = 1
    safety["counters"]["hard_limit_inward_recovery_events"] = 1
    safety["counters"]["hard_limit_inward_recovery_joints"] = 1
    safety["maxima"]["applied_delta_joint_pos_rad"][0] = (
        finalizer.EXPECTED_MAX_DELTA[0] + finalizer.EXPECTED_PHYSICAL_MARGIN[0]
    ) / 2.0
    finalizer._verify_raw(raw)

    spurious = _valid_raw_result()
    counters = spurious["ik_safety_episodes"][0]["counters"]
    counters["position_limit_events"] = 1
    counters["position_limited_joints"] = 1
    counters["hard_limit_inward_recovery_events"] = 1
    counters["hard_limit_inward_recovery_joints"] = 1
    with pytest.raises(finalizer.VerificationError, match="activation mismatch"):
        finalizer._verify_raw(spurious)


def test_finalizer_rejects_max_diagnostic_above_recorded_applied_maximum():
    raw = _valid_raw_result()
    safety = raw["ik_safety_episodes"][0]
    diagnostic = safety["max_raw_delta_diagnostic"]
    q = diagnostic["joint_pos_rad"]["values"][0]
    diagnostic["safe_joint_pos_target_rad"]["values"][0] = (
        q
        + (finalizer.EXPECTED_MAX_DELTA[0] + finalizer.EXPECTED_PHYSICAL_MARGIN[0])
        / 2.0
    )
    with pytest.raises(finalizer.VerificationError, match="exceeds applied maximum"):
        finalizer._verify_raw(raw)


def _write_immutable_json(path: Path, payload) -> bytes:
    data = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
    path.write_bytes(data)
    path.chmod(0o444)
    return data


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments], text=True
    ).strip()


def _attestation_args(tmp_path: Path, monkeypatch) -> Namespace:
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
    monkeypatch.setenv("SLURM_JOB_ID", str(job_id))
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
        expected_finalizer_sha256=hashlib.sha256(
            Path(finalizer.__file__).resolve().read_bytes()
        ).hexdigest(),
        expected_saved_job_script_sha256=hashlib.sha256(
            saved_job.read_bytes()
        ).hexdigest(),
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
    with pytest.raises(finalizer.VerificationError, match="limits"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["raw_ik_safety_capture"]["physx_derived_soft_joint_pos_limits_rad"][3][1] = (
        finalizer.EXPECTED_TARGET_LIMITS[3][1]
    )
    with pytest.raises(finalizer.VerificationError, match="physx_derived_soft"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["raw_ik_safety_capture"]["arm_velocity_target_rad_s"][4] = 1e-3
    with pytest.raises(finalizer.VerificationError, match="velocity_target"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["raw_ik_safety_capture"]["solver_velocity_iteration_count"] = 0
    with pytest.raises(finalizer.VerificationError, match="solver_velocity"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["raw_ik_safety_capture"]["profile"] = "panda_velocity_physxlimit_solveriter4_v5"
    with pytest.raises(finalizer.VerificationError, match="profile"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["raw_ik_safety_capture"]["joint_drive_damping"][6] = 79.0
    with pytest.raises(finalizer.VerificationError, match="joint_drive_damping"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    safety = raw["raw_ik_safety_capture"]
    safety["max_delta_joint_pos_rad"], safety["target_soft_limit_margin_rad"] = (
        safety["target_soft_limit_margin_rad"],
        safety["max_delta_joint_pos_rad"],
    )
    with pytest.raises(finalizer.VerificationError, match="max_delta_joint_pos_rad"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["raw_ik_safety_capture"]["physx_solver_type"] = 0
    with pytest.raises(finalizer.VerificationError, match="physx_solver_type"):
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

    raw = _valid_raw_result()
    raw["schema_version"] = True
    with pytest.raises(finalizer.VerificationError, match="schema_version"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["exit_code"] = False
    with pytest.raises(finalizer.VerificationError, match="exit_code"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["ik_safety_episodes"][0]["counters"]["apply_calls"] = True
    with pytest.raises(finalizer.VerificationError, match="counters invalid"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    counters = raw["ik_safety_episodes"][0]["counters"]
    counters["slew_limit_events"] = 1
    counters["slew_limited_joints"] = 100
    with pytest.raises(finalizer.VerificationError, match="impossible"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    counters = raw["ik_safety_episodes"][0]["counters"]
    counters["slew_limit_events"] = 1
    counters["slew_limited_joints"] = 1
    with pytest.raises(finalizer.VerificationError, match="activation mismatch"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    safety = raw["ik_safety_episodes"][0]
    diagnostic = safety["max_raw_delta_diagnostic"]
    q0 = diagnostic["joint_pos_rad"]["values"][0]
    bound = finalizer.EXPECTED_MAX_DELTA[0]
    raw_delta = bound + 0.01
    safety["maxima"]["raw_delta_joint_pos_rad"][0] = raw_delta
    safety["maxima"]["applied_delta_joint_pos_rad"][0] = bound
    diagnostic["raw_delta_joint_pos_rad"]["values"][0] = raw_delta
    diagnostic["raw_joint_pos_target_rad"]["values"][0] = q0 + raw_delta
    diagnostic["safe_joint_pos_target_rad"]["values"][0] = q0 + bound
    with pytest.raises(finalizer.VerificationError, match="activation mismatch"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["ik_safety_episodes"][0]["max_raw_delta_diagnostic"]["pose_error_norm"] = None
    with pytest.raises(finalizer.VerificationError, match="pose_error_norm"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["ik_safety_episodes"][0]["max_raw_delta_diagnostic"]["jacobian_max_abs"] = None
    with pytest.raises(finalizer.VerificationError, match="jacobian_max_abs"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["ik_safety_episodes"][0]["max_raw_delta_diagnostic"]["eef_quaternion_norm"] = (
        1.0
    )
    with pytest.raises(finalizer.VerificationError, match="eef_quaternion_norm"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["results"][0]["actual_quaternion_wxyz"] = [
        math.cos(0.1),
        math.sin(0.1),
        0.0,
        0.0,
    ]
    with pytest.raises(
        finalizer.VerificationError, match="rotation error inconsistent"
    ):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["results"][1]["target_position"] = raw["results"][0]["target_position"].copy()
    raw["results"][1]["actual_position"] = raw["results"][1]["target_position"].copy()
    with pytest.raises(
        finalizer.VerificationError, match="translation target geometry"
    ):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    wrong_sign = raw["results"][8]["target_quaternion_wxyz"].copy()
    raw["results"][7]["target_quaternion_wxyz"] = wrong_sign
    raw["results"][7]["actual_quaternion_wxyz"] = wrong_sign.copy()
    with pytest.raises(finalizer.VerificationError, match="rotation target geometry"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["ik_safety_episodes"][0]["max_raw_delta_diagnostic"][
        "raw_joint_pos_target_rad"
    ]["values"][0] += 0.01
    with pytest.raises(finalizer.VerificationError, match="raw target identity"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["ik_safety_episodes"][0]["max_raw_delta_diagnostic"][
        "safe_joint_pos_target_rad"
    ]["values"][0] += 0.1
    with pytest.raises(finalizer.VerificationError, match="safe slew"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["ik_safety_episodes"][0]["target_joint_pos_limits_float32_sha256"] = "0" * 64
    with pytest.raises(finalizer.VerificationError, match="target_joint_pos_limits"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["ik_safety_episodes"][0]["maxima"][
        "post_clamp_target_guard_band_violation_rad"
    ][0] = 2e-5
    with pytest.raises(finalizer.VerificationError, match="target guard-band maxima"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["ik_safety_episodes"][0]["maxima"][
        "post_clamp_target_guard_band_violation_rad"
    ][0] = 5e-6
    with pytest.raises(
        finalizer.VerificationError,
        match="target guard-band recovery attribution",
    ):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    diagnostic = raw["ik_safety_adversarial"]["ik_safety"]["max_raw_delta_diagnostic"]
    safety = raw["ik_safety_adversarial"]["ik_safety"]
    safety["maxima"]["raw_delta_joint_pos_rad"] = [0.0] * 7
    safety["maxima"]["applied_delta_joint_pos_rad"] = [0.0] * 7
    diagnostic["raw_delta_joint_pos_rad"] = _diagnostic_vector([0.0] * 7)
    diagnostic["raw_joint_pos_target_rad"] = copy.deepcopy(diagnostic["joint_pos_rad"])
    diagnostic["safe_joint_pos_target_rad"] = copy.deepcopy(diagnostic["joint_pos_rad"])
    with pytest.raises(finalizer.VerificationError, match="activation mismatch"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    dq = raw["ik_safety_adversarial"]["joint_state"]["joint_vel_rad_s"]
    dq["values"][0] = finalizer.EXPECTED_VELOCITY_LIMITS[0] + 0.1
    dq["max_abs"] = dq["values"][0]
    with pytest.raises(finalizer.VerificationError, match="terminal dq"):
        finalizer._verify_raw(raw)


def test_attestation_is_bound_verified_and_nonoverwriting(tmp_path, monkeypatch):
    args = _attestation_args(tmp_path, monkeypatch)
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


def test_attestation_rejects_writable_or_mutated_evidence(tmp_path, monkeypatch):
    args = _attestation_args(tmp_path, monkeypatch)
    args.raw_result.chmod(0o644)
    with pytest.raises(finalizer.VerificationError, match="mode"):
        finalizer._build_expected(args)

    args = _attestation_args(tmp_path / "second", monkeypatch)
    marker = args.raw_result.with_name(args.raw_result.name + ".ready.json")
    marker.chmod(0o644)
    with pytest.raises(finalizer.VerificationError, match="ready marker mode"):
        finalizer._build_expected(args)

    args = _attestation_args(tmp_path / "third", monkeypatch)
    args.raw_result.chmod(0o644)
    args.raw_result.write_bytes(args.raw_result.read_bytes() + b" ")
    args.raw_result.chmod(0o444)
    with pytest.raises(finalizer.VerificationError, match="ready marker"):
        finalizer._build_expected(args)

    args = _attestation_args(tmp_path / "fourth", monkeypatch)
    args.expected_image_sha256 = "0" * 64
    with pytest.raises(finalizer.VerificationError, match="image digest"):
        finalizer._build_expected(args)

    args = _attestation_args(tmp_path / "fifth", monkeypatch)
    args.saved_job_script.write_text("tampered\n")
    with pytest.raises(finalizer.VerificationError, match="job script digest"):
        finalizer._build_expected(args)

    args = _attestation_args(tmp_path / "sixth", monkeypatch)
    dirty_path = args.polaris_repo / "dirty.txt"
    dirty_path.write_text("dirty\n")
    with pytest.raises(finalizer.VerificationError, match="repo dirty"):
        finalizer._build_expected(args)

    args = _attestation_args(tmp_path / "seventh", monkeypatch)
    args.expected_finalizer_sha256 = "0" * 64
    with pytest.raises(finalizer.VerificationError, match="finalizer expected digest"):
        finalizer._build_expected(args)

    args = _attestation_args(tmp_path / "eighth", monkeypatch)
    args.expected_saved_job_script_sha256 = "0" * 64
    with pytest.raises(finalizer.VerificationError, match="saved job script expected"):
        finalizer._build_expected(args)

    args = _attestation_args(tmp_path / "ninth", monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", "99999")
    with pytest.raises(finalizer.VerificationError, match="SLURM_JOB_ID"):
        finalizer._build_expected(args)

    args = _attestation_args(tmp_path / "tenth", monkeypatch)
    marker = args.raw_result.with_name(args.raw_result.name + ".ready.json")
    marker_payload = json.loads(marker.read_text())
    marker_payload["schema_version"] = True
    marker.chmod(0o644)
    marker.write_text(json.dumps(marker_payload, indent=2) + "\n")
    marker.chmod(0o444)
    with pytest.raises(finalizer.VerificationError, match="ready marker"):
        finalizer._build_expected(args)

    assert not finalizer._typed_equal({"schema_version": True}, {"schema_version": 1})
