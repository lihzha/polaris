#!/usr/bin/env python3
"""Finalize or reconstruct one controller-candidate replay attestation.

This host-only consumer runs after the outer ``srun`` returns.  It independently
validates the immutable raw/ready pair, binds the zero-return status and Slurm
artifacts, rehashes every launch source plus the container, and only then
publishes a separate mode-0444 attestation without overwrite.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any

try:
    from scripts import smoke_eef_pose_canary_controller_candidate as candidate
    from scripts import smoke_eef_pose_canary_trace_replay as gate0
    from scripts import validate_eef_pose_canary_controller_candidate as validator
    from scripts import (
        write_eef_pose_canary_controller_candidate_srun_status as status_writer,
    )
except ImportError:  # Direct ``python scripts/finalize_...py`` execution.
    import smoke_eef_pose_canary_controller_candidate as candidate
    import smoke_eef_pose_canary_trace_replay as gate0
    import validate_eef_pose_canary_controller_candidate as validator
    import write_eef_pose_canary_controller_candidate_srun_status as status_writer


PROFILE = "polaris_eef_controller_candidate_attestation_v1"
ATTESTATION_FIELDS = {
    "schema_version",
    "profile",
    "finalized",
    "passed",
    "stage",
    "variant",
    "lifecycle",
    "raw_result",
    "srun_status",
    "validation_summary",
    "provenance",
}
STATUS_IDENTITY_FIELDS = {*validator.FILE_IDENTITY_FIELDS, "mtime_ns"}


class CandidateFinalizationError(ValueError):
    """Candidate evidence or launch provenance is not promotable."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateFinalizationError(message)


def _mode(path: Path) -> str:
    return f"{stat.S_IMODE(path.stat().st_mode):04o}"


def _identity(path: Path, *, field: str, include_mtime: bool = False) -> dict[str, Any]:
    path = Path(os.path.abspath(path))
    _require(path.is_file() and not path.is_symlink(), f"{field} missing/linked")
    metadata = path.stat()
    _require(metadata.st_nlink == 1, f"{field} must have one hard link")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    result = {
        "path": str(path),
        "size_bytes": size,
        "sha256": digest.hexdigest(),
        "mode": _mode(path),
    }
    if include_mtime:
        result["mtime_ns"] = metadata.st_mtime_ns
    return result


def _read_json(path: Path, *, field: str) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = _identity(path, field=field)
    payload = gate0.strict_json_loads(path.read_bytes(), field=field)
    return payload, identity


def _hex(value: Any, *, field: str, length: int) -> str:
    _require(
        isinstance(value, str)
        and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None,
        f"{field} malformed",
    )
    return value


def _same_core_identity(recorded: Any, current: dict[str, Any], *, field: str) -> None:
    _require(isinstance(recorded, dict), f"{field} recorded identity")
    recorded_path = recorded.get("path")
    current_path = current.get("path")
    try:
        same_file = (
            isinstance(recorded_path, str)
            and isinstance(current_path, str)
            and os.path.samefile(recorded_path, current_path)
        )
    except OSError:
        same_file = False
    _require(same_file, f"{field} path changed")
    for name in ("size_bytes", "sha256", "mode"):
        _require(
            validator._typed_equal(recorded.get(name), current[name]),
            f"{field} {name} value/type changed",
        )


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _source_identity(path: Path, *, field: str, expected_sha256: str) -> dict[str, Any]:
    identity = _identity(path, field=field)
    _require(identity["sha256"] == expected_sha256, f"{field} digest mismatch")
    return identity


def _validate_job_artifacts(
    args: argparse.Namespace, *, lifecycle: dict[str, Any]
) -> dict[str, Any]:
    gpu = _identity(args.gpu_inventory, field="GPU inventory")
    job = _identity(args.job_metadata, field="Slurm job metadata")
    stdout = _identity(args.stdout_log, field="srun stdout")
    stderr = _identity(args.stderr_log, field="srun stderr")
    for name, identity in (
        ("GPU inventory", gpu),
        ("Slurm job metadata", job),
        ("srun stdout", stdout),
        ("srun stderr", stderr),
    ):
        _require(identity["mode"] == "0444", f"{name} mode")
    gpu_text = args.gpu_inventory.read_text(errors="strict")
    _require(
        len(re.findall(r"Product Name\s*:\s*NVIDIA L40S\b", gpu_text)) == 1,
        "GPU inventory is not exactly one NVIDIA L40S",
    )
    job_text = args.job_metadata.read_text(errors="strict")
    _require(
        re.search(rf"\bJobId={args.job_id}\b", job_text) is not None
        and re.search(r"\bJobState=RUNNING\b", job_text) is not None
        and f"NodeList={lifecycle['nodelist']}" in job_text,
        "Slurm job metadata/lifecycle drift",
    )
    return {
        "gpu_inventory": gpu,
        "slurm_job_metadata": job,
        "srun_stdout": stdout,
        "srun_stderr": stderr,
    }


def _candidate_validation_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        variant=args.variant,
        launch_id=args.launch_id,
        job_id=args.job_id,
        raw_result=args.raw_result,
        ready_marker=args.raw_result.with_name(args.raw_result.name + ".ready.json"),
        polaris_repo=args.polaris_repo,
        expected_polaris_commit=args.expected_polaris_commit,
        expected_runner_sha256=args.expected_runner_sha256,
        expected_validator_sha256=args.expected_validator_sha256,
        expected_safety_validator_sha256=args.expected_safety_validator_sha256,
        expected_gate0_helper_sha256=args.expected_gate0_helper_sha256,
        expected_fixture_sha256=args.expected_fixture_sha256,
        container_image=args.container_image,
        expected_container_size_bytes=args.expected_container_size_bytes,
        expected_container_sha256=args.expected_container_sha256,
    )


def build_attestation(args: argparse.Namespace) -> dict[str, Any]:
    _require(args.variant in candidate.CANDIDATE_BY_VARIANT, "unknown variant")
    _require(type(args.job_id) is int and args.job_id > 0, "job ID")
    _hex(args.launch_id, field="launch ID", length=64)
    for value, field, length in (
        (args.expected_polaris_commit, "PolaRiS commit", 40),
        (args.expected_runner_sha256, "runner SHA", 64),
        (args.expected_validator_sha256, "validator SHA", 64),
        (args.expected_failure_verifier_sha256, "failure verifier SHA", 64),
        (args.expected_safety_validator_sha256, "safety validator SHA", 64),
        (args.expected_gate0_helper_sha256, "Gate0 helper SHA", 64),
        (args.expected_fixture_sha256, "fixture SHA", 64),
        (args.expected_status_writer_sha256, "status writer SHA", 64),
        (args.expected_finalizer_sha256, "finalizer SHA", 64),
        (args.expected_container_sha256, "container SHA", 64),
        (args.expected_saved_job_script_sha256, "job script SHA", 64),
    ):
        _hex(value, field=field, length=length)
    _require(
        type(args.expected_container_size_bytes) is int
        and args.expected_container_size_bytes > 0,
        "container size",
    )
    live_job_id = os.environ.get("SLURM_JOB_ID")
    if args.mode == "finalize":
        _require(
            isinstance(live_job_id, str)
            and live_job_id.isdecimal()
            and int(live_job_id) == args.job_id,
            "outer SLURM_JOB_ID mismatch",
        )
    elif live_job_id is not None:
        _require(
            live_job_id.isdecimal() and int(live_job_id) == args.job_id,
            "verify SLURM_JOB_ID mismatch",
        )

    namespace = args.raw_result.resolve().parent
    _require(
        namespace.name == f"launch_{args.launch_id}"
        and namespace.parent.name == f"job_{args.job_id}"
        and namespace.parent.parent.name == args.variant,
        "variant/job/launch namespace",
    )
    expected_names = {
        "raw": f"candidate-{args.variant}.raw.json",
        "status": f"candidate-{args.variant}.srun-status.json",
        "attestation": f"candidate-{args.variant}.attestation.json",
        "saved_job": f"candidate-{args.variant}.job.sh",
        "gpu": "gpu-inventory.txt",
        "job": "slurm-job.txt",
        "stdout": "srun.stdout.log",
        "stderr": "srun.stderr.log",
    }
    paths = {
        "raw": args.raw_result,
        "status": args.srun_status,
        "attestation": args.attestation,
        "saved_job": args.saved_job_script,
        "gpu": args.gpu_inventory,
        "job": args.job_metadata,
        "stdout": args.stdout_log,
        "stderr": args.stderr_log,
    }
    for name, path in paths.items():
        _require(path.name == expected_names[name], f"{name} filename")
        _require(path.resolve().parent == namespace, f"{name} namespace")

    validation = validator.validate(_candidate_validation_args(args))
    raw, raw_identity = _read_json(args.raw_result, field="candidate raw")
    ready_path = args.raw_result.with_name(args.raw_result.name + ".ready.json")
    _, ready_identity = _read_json(ready_path, field="candidate ready")
    _require(
        raw_identity["mode"] == ready_identity["mode"] == "0444",
        "raw/ready mode",
    )
    lifecycle = raw["lifecycle"]
    _require(
        raw["variant"] == args.variant
        and lifecycle["job_id"] == args.job_id
        and lifecycle["launch_id"] == args.launch_id,
        "raw lifecycle mismatch",
    )
    _same_core_identity(validation["raw_result"], raw_identity, field="validated raw")
    _same_core_identity(
        validation["ready_marker"], ready_identity, field="validated ready"
    )

    status, status_identity = _read_json(args.srun_status, field="srun status")
    _require(status_identity["mode"] == "0444", "srun status mode")
    _require(
        set(status) == status_writer.STATUS_FIELDS
        and type(status.get("schema_version")) is int
        and status.get("schema_version") == 1
        and status.get("profile") == status_writer.PROFILE
        and type(status.get("variant")) is str
        and status.get("variant") == args.variant
        and type(status.get("launch_id")) is str
        and status.get("launch_id") == args.launch_id
        and type(status.get("job_id")) is int
        and status.get("job_id") == args.job_id
        and type(status.get("srun_rc")) is int
        and status.get("srun_rc") == 0
        and validator._typed_equal(status.get("raw_lifecycle"), lifecycle),
        "srun status lifecycle",
    )
    for name in (
        "srun_started_at_ns",
        "srun_returned_at_ns",
        "status_published_at_ns",
    ):
        _require(
            type(status.get(name)) is int and status[name] > 0,
            f"srun status {name} type/range",
        )
    _require(
        isinstance(status.get("raw_result"), dict)
        and set(status["raw_result"]) == STATUS_IDENTITY_FIELDS
        and isinstance(status.get("ready_marker"), dict)
        and set(status["ready_marker"]) == STATUS_IDENTITY_FIELDS,
        "srun status evidence identity schema",
    )
    raw_with_mtime = _identity(args.raw_result, field="raw result", include_mtime=True)
    ready_with_mtime = _identity(ready_path, field="ready marker", include_mtime=True)
    _same_core_identity(status.get("raw_result"), raw_with_mtime, field="status raw")
    _same_core_identity(
        status.get("ready_marker"), ready_with_mtime, field="status ready"
    )
    _require(
        validator._typed_equal(
            status["raw_result"].get("mtime_ns"), raw_with_mtime["mtime_ns"]
        )
        and validator._typed_equal(
            status["ready_marker"].get("mtime_ns"), ready_with_mtime["mtime_ns"]
        ),
        "srun status immutable pair mtime",
    )
    _require(
        0
        < status["srun_started_at_ns"]
        <= raw_with_mtime["mtime_ns"]
        <= ready_with_mtime["mtime_ns"]
        <= status["srun_returned_at_ns"]
        <= status["status_published_at_ns"],
        "srun lifecycle timestamps",
    )

    repo = args.polaris_repo.resolve()
    commit = _git(repo, "rev-parse", "HEAD")
    _require(commit == args.expected_polaris_commit, "PolaRiS commit mismatch")
    _require(
        _git(repo, "status", "--porcelain", "--untracked-files=no") == "",
        "PolaRiS tracked worktree dirty",
    )
    sources = {
        "runner": _source_identity(
            repo / "scripts/smoke_eef_pose_canary_controller_candidate.py",
            field="candidate runner",
            expected_sha256=args.expected_runner_sha256,
        ),
        "validator": _source_identity(
            repo / "scripts/validate_eef_pose_canary_controller_candidate.py",
            field="candidate validator",
            expected_sha256=args.expected_validator_sha256,
        ),
        # The verifier is a standalone failure-path subprocess consumer.  Bind
        # its exact repository source here rather than importing it into the
        # successful finalization process.
        "failure_verifier": _source_identity(
            repo / "scripts/verify_eef_pose_canary_controller_candidate_failure.py",
            field="candidate failure verifier",
            expected_sha256=args.expected_failure_verifier_sha256,
        ),
        "safety_validator": _source_identity(
            repo / "scripts/finalize_eef_pose_smoke.py",
            field="production-equivalent host safety validator",
            expected_sha256=args.expected_safety_validator_sha256,
        ),
        "gate0_helper": _source_identity(
            repo / "scripts/smoke_eef_pose_canary_trace_replay.py",
            field="Gate0 helper",
            expected_sha256=args.expected_gate0_helper_sha256,
        ),
        "status_writer": _source_identity(
            repo / "scripts/write_eef_pose_canary_controller_candidate_srun_status.py",
            field="candidate status writer",
            expected_sha256=args.expected_status_writer_sha256,
        ),
        "finalizer": _source_identity(
            Path(__file__),
            field="candidate finalizer",
            expected_sha256=args.expected_finalizer_sha256,
        ),
    }
    imported = {
        "runner": Path(candidate.__file__),
        "validator": Path(validator.__file__),
        "safety_validator": Path(validator.safety_validator.__file__),
        "gate0_helper": Path(gate0.__file__),
        "status_writer": Path(status_writer.__file__),
        "finalizer": Path(__file__),
    }
    for name, path in imported.items():
        _require(
            os.path.samefile(path, sources[name]["path"]),
            f"imported {name} differs from hashed source",
        )

    fixture = validation["sources"]["fixture"]
    _require(fixture["sha256"] == args.expected_fixture_sha256, "fixture digest")
    container = _identity(args.container_image, field="container image")
    _require(
        container["size_bytes"] == args.expected_container_size_bytes
        and container["sha256"] == args.expected_container_sha256,
        "container identity",
    )
    runtime_job_script = _identity(args.runtime_job_script, field="runtime job script")
    saved_job_script = _identity(args.saved_job_script, field="saved job script")
    _require(
        runtime_job_script["sha256"]
        == saved_job_script["sha256"]
        == args.expected_saved_job_script_sha256
        and saved_job_script["mode"] == "0444",
        "runtime/saved job script identity",
    )
    job_script = {
        "profile": "slurm_runtime_content_saved_copy_v1",
        "runtime_content_size_bytes": runtime_job_script["size_bytes"],
        "runtime_content_sha256": runtime_job_script["sha256"],
        "saved_identity": saved_job_script,
    }
    job_artifacts = _validate_job_artifacts(args, lifecycle=lifecycle)

    final_safety = raw["final_safety"]
    result = {
        "schema_version": 1,
        "profile": PROFILE,
        "finalized": True,
        "passed": True,
        "stage": "complete",
        "variant": args.variant,
        "lifecycle": lifecycle,
        "raw_result": {**raw_identity, "ready_marker": ready_identity},
        "srun_status": status_identity,
        "validation_summary": {
            "outcome": raw["outcome"],
            "action_plan": raw["action_plan"],
            "initial_candidate": validation["initial_candidate"],
            "final_candidate": validation["final_candidate"],
            "replay_validation": validation["replay_validation"],
            "velocity_headroom": validation["velocity_headroom"],
            "final_safety_counters": final_safety["counters"],
            "fixture_sha256": fixture["sha256"],
            "fixture_source_trace_sha256": raw["fixture"]["source_trace_sha256"],
        },
        "provenance": {
            "polaris_repo": str(repo),
            "polaris_commit": commit,
            "sources": sources,
            "fixture": fixture,
            "container_image": container,
            "job_script": job_script,
            "job_artifacts": job_artifacts,
        },
    }
    _require(set(result) == ATTESTATION_FIELDS, "attestation schema")
    return result


def _publish(path: Path, payload: dict[str, Any]) -> None:
    _require(not path.exists(), f"attestation exists: {path}")
    gate0._atomic_write_immutable(path, payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("finalize", "verify"))
    parser.add_argument(
        "--variant", choices=sorted(candidate.CANDIDATE_BY_VARIANT), required=True
    )
    parser.add_argument("--launch-id", required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--raw-result", type=Path, required=True)
    parser.add_argument("--srun-status", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--polaris-repo", type=Path, required=True)
    parser.add_argument("--expected-polaris-commit", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-validator-sha256", required=True)
    parser.add_argument("--expected-failure-verifier-sha256", required=True)
    parser.add_argument("--expected-safety-validator-sha256", required=True)
    parser.add_argument("--expected-gate0-helper-sha256", required=True)
    parser.add_argument("--expected-fixture-sha256", required=True)
    parser.add_argument("--expected-status-writer-sha256", required=True)
    parser.add_argument("--expected-finalizer-sha256", required=True)
    parser.add_argument("--container-image", type=Path, required=True)
    parser.add_argument("--expected-container-size-bytes", type=int, required=True)
    parser.add_argument("--expected-container-sha256", required=True)
    parser.add_argument("--runtime-job-script", type=Path, required=True)
    parser.add_argument("--saved-job-script", type=Path, required=True)
    parser.add_argument("--expected-saved-job-script-sha256", required=True)
    parser.add_argument("--gpu-inventory", type=Path, required=True)
    parser.add_argument("--job-metadata", type=Path, required=True)
    parser.add_argument("--stdout-log", type=Path, required=True)
    parser.add_argument("--stderr-log", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        expected = build_attestation(args)
        if args.mode == "finalize":
            _publish(args.attestation, expected)
        actual, identity = _read_json(args.attestation, field="attestation")
        _require(identity["mode"] == "0444", "attestation mode")
        _require(
            validator._typed_equal(actual, expected),
            "attestation reconstruction value/type mismatch",
        )
    except (
        CandidateFinalizationError,
        OSError,
        UnicodeError,
        gate0.Gate0ReplayValidationError,
        status_writer.CandidateSrunStatusError,
        validator.CandidateArtifactValidationError,
        subprocess.CalledProcessError,
    ) as error:
        print(
            f"POLARIS_CONTROLLER_CANDIDATE_ATTESTATION_FAIL={error}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(
        f"POLARIS_CONTROLLER_CANDIDATE_ATTESTATION_PASS={args.attestation.resolve()}",
        flush=True,
    )
    print(
        f"POLARIS_CONTROLLER_CANDIDATE_ATTESTATION_SHA256={identity['sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
