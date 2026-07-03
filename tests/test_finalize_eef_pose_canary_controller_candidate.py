from __future__ import annotations

from argparse import Namespace
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import stat
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
TEST_START_TIME_MARGIN_NS = 1_000_000_000


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


status_writer = _load_script("write_eef_pose_canary_controller_candidate_srun_status")
finalizer = _load_script("finalize_eef_pose_canary_controller_candidate")
validator = finalizer.validator
candidate = finalizer.candidate
gate0 = finalizer.gate0


def _chmod_readonly(path: Path) -> None:
    path.chmod(0o444)
    assert stat.S_IMODE(path.stat().st_mode) == 0o444


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish_raw_pair(
    tmp_path: Path, *, launch_id: str, job_id: int, started_at_ns: int
) -> tuple[Path, Path, dict, int]:
    variant = "official_lap3b"
    namespace = tmp_path / variant / f"job_{job_id}" / f"launch_{launch_id}"
    raw_path = namespace / f"candidate-{variant}.raw.json"
    lifecycle = {
        "profile": "slurm_single_task_srun_lifecycle_v1",
        "launch_id": launch_id,
        "job_id": job_id,
        "step_id": 0,
        "nodelist": "l401",
        "procid": 0,
        "localid": 0,
        "ntasks": 1,
    }
    raw = {field: None for field in validator.RAW_FIELDS}
    raw.update(
        {
            "schema_version": 1,
            "profile": candidate.PROFILE,
            "finalized": False,
            "passed": True,
            "stage": "simulation_app_close_pending",
            "environment": gate0.ENVIRONMENT,
            "variant": variant,
            "candidate": candidate.CANDIDATE_BY_VARIANT[variant],
            "lifecycle": lifecycle,
            "action_plan": validator.ACTION_PLAN,
            "final_safety": {"counters": {"apply_calls": 1016}},
            "fixture": {"source_trace_sha256": "a" * 64},
            "outcome": {
                "status": "candidate_replay_completed_without_controller_abort",
                "actions_completed": 127,
                "original_failure_step_crossed": 117,
                "post_fixture_release_probe_completed": True,
            },
            "close_failures": [],
        }
    )
    raw_identity = gate0._atomic_write_immutable(raw_path, raw)
    ready_path = raw_path.with_name(raw_path.name + ".ready.json")
    gate0._atomic_write_immutable(
        ready_path,
        {
            "schema_version": 1,
            "profile": candidate.PROFILE,
            "stage": "simulation_app_close_pending",
            "raw_result": raw_identity,
        },
    )
    returned_at_ns = time.time_ns()
    assert started_at_ns <= raw_path.stat().st_mtime_ns
    assert ready_path.stat().st_mtime_ns <= returned_at_ns
    return raw_path, ready_path, lifecycle, returned_at_ns


def _publish_status(
    raw_path: Path,
    *,
    launch_id: str,
    job_id: int,
    started_at_ns: int,
    returned_at_ns: int,
) -> Path:
    status_path = raw_path.with_name("candidate-official_lap3b.srun-status.json")
    args = Namespace(
        variant="official_lap3b",
        launch_id=launch_id,
        job_id=job_id,
        srun_rc=0,
        srun_started_at_ns=started_at_ns,
        srun_returned_at_ns=returned_at_ns,
        raw_result=raw_path,
        status=status_path,
    )
    payload = status_writer.build_status(args)
    gate0._atomic_write_immutable(status_path, payload)
    return status_path


def _write_job_artifacts(namespace: Path, *, job_id: int) -> dict[str, Path]:
    artifacts = {
        "gpu_inventory": namespace / "gpu-inventory.txt",
        "job_metadata": namespace / "slurm-job.txt",
        "stdout_log": namespace / "srun.stdout.log",
        "stderr_log": namespace / "srun.stderr.log",
    }
    artifacts["gpu_inventory"].write_text(
        "Product Name                          : NVIDIA L40S\n"
    )
    artifacts["job_metadata"].write_text(
        f"JobId={job_id} JobState=RUNNING NodeList=l401\n"
    )
    artifacts["stdout_log"].write_text("candidate raw published\n")
    artifacts["stderr_log"].write_text("")
    for path in artifacts.values():
        _chmod_readonly(path)
    return artifacts


def _finalizer_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Namespace, dict]:
    launch_id = "1" * 64
    job_id = 123
    monkeypatch.setenv("SLURM_JOB_ID", str(job_id))
    # Some filesystems quantize an immediately following mtime just below the
    # userspace clock sample. Keep the synthetic start unambiguously pre-write.
    started_at_ns = time.time_ns() - TEST_START_TIME_MARGIN_NS
    raw_path, ready_path, lifecycle, returned_at_ns = _publish_raw_pair(
        tmp_path,
        launch_id=launch_id,
        job_id=job_id,
        started_at_ns=started_at_ns,
    )
    status_path = _publish_status(
        raw_path,
        launch_id=launch_id,
        job_id=job_id,
        started_at_ns=started_at_ns,
        returned_at_ns=returned_at_ns,
    )
    namespace = raw_path.parent
    job_artifacts = _write_job_artifacts(namespace, job_id=job_id)

    wrapper = SCRIPTS / "run_eef_pose_canary_controller_candidate_srun.sh"
    saved_wrapper = namespace / "candidate-official_lap3b.job.sh"
    shutil.copyfile(wrapper, saved_wrapper)
    _chmod_readonly(saved_wrapper)
    container = namespace / "polaris.sqsh"
    container.write_bytes(b"immutable candidate test container")
    fixture = gate0._file_identity(
        SCRIPTS / "fixtures" / gate0.EXPECTED_FIXTURES["official_lap3b"]["filename"]
    )

    commit = "2" * 40

    def fake_git(_repo, *arguments):
        if arguments == ("rev-parse", "HEAD"):
            return commit
        assert arguments == ("status", "--porcelain", "--untracked-files=no")
        return ""

    monkeypatch.setattr(finalizer, "_git", fake_git)
    validation = {
        "profile": "polaris_eef_candidate_independent_artifact_validation_v1",
        "variant": "official_lap3b",
        "job_id": job_id,
        "launch_id": launch_id,
        "raw_result": validator._immutable_file(raw_path),
        "ready_marker": validator._immutable_file(ready_path),
        "repository": {"path": str(ROOT), "commit": commit, "clean_tracked": True},
        "container_image": {},
        "sources": {"fixture": {key: fixture[key] for key in fixture if key != "mode"}},
        "initial_candidate": {"stage": "initial"},
        "final_candidate": {"stage": "final"},
        "replay_validation": {"arm_apply_calls": 1016},
        "velocity_headroom": {"maximum_ratio": 0.94, "passed": True},
    }
    monkeypatch.setattr(validator, "validate", lambda _args: validation)

    args = Namespace(
        mode="finalize",
        variant="official_lap3b",
        launch_id=launch_id,
        job_id=job_id,
        raw_result=raw_path,
        srun_status=status_path,
        attestation=namespace / "candidate-official_lap3b.attestation.json",
        polaris_repo=ROOT,
        expected_polaris_commit=commit,
        expected_runner_sha256=_sha256(
            SCRIPTS / "smoke_eef_pose_canary_controller_candidate.py"
        ),
        expected_validator_sha256=_sha256(
            SCRIPTS / "validate_eef_pose_canary_controller_candidate.py"
        ),
        expected_failure_verifier_sha256=_sha256(
            SCRIPTS / "verify_eef_pose_canary_controller_candidate_failure.py"
        ),
        expected_safety_validator_sha256=_sha256(
            SCRIPTS / "finalize_eef_pose_smoke.py"
        ),
        expected_gate0_helper_sha256=_sha256(
            SCRIPTS / "smoke_eef_pose_canary_trace_replay.py"
        ),
        expected_fixture_sha256=fixture["sha256"],
        expected_status_writer_sha256=_sha256(
            SCRIPTS / "write_eef_pose_canary_controller_candidate_srun_status.py"
        ),
        expected_finalizer_sha256=_sha256(
            SCRIPTS / "finalize_eef_pose_canary_controller_candidate.py"
        ),
        container_image=container,
        expected_container_size_bytes=container.stat().st_size,
        expected_container_sha256=_sha256(container),
        runtime_job_script=wrapper,
        saved_job_script=saved_wrapper,
        expected_saved_job_script_sha256=_sha256(wrapper),
        **job_artifacts,
    )
    return args, {"lifecycle": lifecycle, "validation": validation}


def test_status_writer_binds_zero_return_raw_ready_and_strict_types(
    tmp_path, monkeypatch
):
    launch_id = "3" * 64
    job_id = 456
    monkeypatch.setenv("SLURM_JOB_ID", str(job_id))
    started_at_ns = time.time_ns() - TEST_START_TIME_MARGIN_NS
    raw, _, _, returned_at_ns = _publish_raw_pair(
        tmp_path,
        launch_id=launch_id,
        job_id=job_id,
        started_at_ns=started_at_ns,
    )
    status = _publish_status(
        raw,
        launch_id=launch_id,
        job_id=job_id,
        started_at_ns=started_at_ns,
        returned_at_ns=returned_at_ns,
    )
    payload = gate0.strict_json_loads(status.read_bytes(), field="status")
    assert set(payload) == status_writer.STATUS_FIELDS
    assert payload["srun_rc"] == 0
    assert payload["raw_result"]["mode"] == "0444"
    assert payload["ready_marker"]["mode"] == "0444"
    assert stat.S_IMODE(status.stat().st_mode) == 0o444
    assert validator._typed_equal(True, 1) is False
    assert validator._typed_equal(False, 0.0) is False


def test_finalizer_reconstructs_and_nonoverwrites_attestation(tmp_path, monkeypatch):
    args, expected_evidence = _finalizer_args(tmp_path, monkeypatch)
    expected = finalizer.build_attestation(args)
    assert expected["lifecycle"] == expected_evidence["lifecycle"]
    assert expected["validation_summary"]["replay_validation"] == {
        "arm_apply_calls": 1016
    }
    assert expected["provenance"]["job_artifacts"]["gpu_inventory"]["mode"] == "0444"
    failure_verifier = expected["provenance"]["sources"]["failure_verifier"]
    assert failure_verifier["path"] == str(
        SCRIPTS / "verify_eef_pose_canary_controller_candidate_failure.py"
    )
    assert failure_verifier["sha256"] == args.expected_failure_verifier_sha256
    finalizer._publish(args.attestation, expected)
    assert stat.S_IMODE(args.attestation.stat().st_mode) == 0o444
    with pytest.raises(finalizer.CandidateFinalizationError, match="exists"):
        finalizer._publish(args.attestation, expected)

    args.mode = "verify"
    reconstructed = finalizer.build_attestation(args)
    actual = gate0.strict_json_loads(
        args.attestation.read_bytes(), field="candidate attestation"
    )
    assert actual == reconstructed == expected


def test_finalizer_cannot_publish_when_artifact_validator_rejects(
    tmp_path, monkeypatch
):
    args, _ = _finalizer_args(tmp_path, monkeypatch)

    def reject(_args):
        raise validator.CandidateArtifactValidationError("semantic evidence drift")

    monkeypatch.setattr(validator, "validate", reject)
    with pytest.raises(
        validator.CandidateArtifactValidationError,
        match="semantic evidence drift",
    ):
        finalizer.build_attestation(args)
    assert not args.attestation.exists()


def test_finalizer_rejects_failure_verifier_digest_mismatch(tmp_path, monkeypatch):
    args, _ = _finalizer_args(tmp_path, monkeypatch)
    args.expected_failure_verifier_sha256 = "f" * 64
    with pytest.raises(
        finalizer.CandidateFinalizationError,
        match="candidate failure verifier digest mismatch",
    ):
        finalizer.build_attestation(args)
    assert not args.attestation.exists()


def test_finalizer_rejects_numeric_type_alias_in_status_identity(tmp_path, monkeypatch):
    args, _ = _finalizer_args(tmp_path, monkeypatch)
    payload = gate0.strict_json_loads(args.srun_status.read_bytes(), field="status")
    payload["raw_result"]["mtime_ns"] = float(payload["raw_result"]["mtime_ns"])
    args.srun_status.chmod(0o644)
    args.srun_status.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    _chmod_readonly(args.srun_status)
    with pytest.raises(finalizer.CandidateFinalizationError, match="mtime"):
        finalizer.build_attestation(args)
    assert not args.attestation.exists()


def test_wrapper_uses_host_preflight_status_and_separate_finalizer():
    source = (SCRIPTS / "run_eef_pose_canary_controller_candidate_srun.sh").read_text()
    preflight = 'python3 "${CANDIDATE_POLARIS_REPO}/scripts/${host_consumer}" --help'
    assert preflight in source
    assert source.index(preflight) < source.index("srun \\")
    assert "CANDIDATE_STATUS_WRITER_SHA256" in source
    assert "CANDIDATE_FINALIZER_SHA256" in source
    assert "CANDIDATE_SAFETY_VALIDATOR_SHA256" in source
    assert "CANDIDATE_FAILURE_VERIFIER_SHA256" in source
    assert 'scripts/finalize_eef_pose_smoke.py"' in source
    assert (
        '--expected-safety-validator-sha256 "${CANDIDATE_SAFETY_VALIDATOR_SHA256}"'
    ) in source
    finalizer_call = source.index(
        'finalize_eef_pose_canary_controller_candidate.py" finalize'
    )
    assert (
        "--expected-failure-verifier-sha256 "
        '"${CANDIDATE_FAILURE_VERIFIER_SHA256}"' in source[finalizer_call:]
    )
    assert "write_eef_pose_canary_controller_candidate_srun_status.py" in source
    assert 'finalize_eef_pose_canary_controller_candidate.py" finalize' in source
    assert '--srun-status "${srun_status}"' in source
    assert '--attestation "${attestation}"' in source
    assert "POLARIS_CONTROLLER_CANDIDATE_COMPLETE=${attestation}" in source
