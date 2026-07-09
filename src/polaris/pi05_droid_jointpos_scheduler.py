"""Closed Slurm lifecycle evidence for the official pi0.5 evaluation.

The submission host and worker each persist the controller's own one-line job
record.  A separate post-terminal command then joins those immutable records to
Slurm accounting and to the evaluator's sealed evidence manifest.  This keeps a
successful evaluation from silently spanning a controller requeue/restart.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any

from polaris.pi05_droid_jointpos_immutable import (
    publish_immutable_json,
    validate_immutable_file,
    validate_immutable_json,
)


SCHEDULER_JOB_PROFILE = "openpi_pi05_droid_jointpos_scheduler_job_v1"
SCHEDULER_TERMINAL_PROFILE = "openpi_pi05_droid_jointpos_scheduler_terminal_v1"
SCHEDULER_RUNNING_FILENAME = "pi05_droid_jointpos_scheduler_running.json"
SCHEDULER_TERMINAL_FILENAME = "pi05_droid_jointpos_scheduler_terminal.json"

_TRANSACTION_PATTERN = re.compile(r"pi05-[0-9a-f]{40}")
_TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _identity(artifact: dict[str, Any]) -> dict[str, Any]:
    return {key: artifact[key] for key in ("path", "size", "sha256", "mode", "nlink")}


def _one_scontrol_line(raw: str) -> tuple[str, dict[str, str]]:
    _require(isinstance(raw, str), "scontrol job record must be text")
    _require(
        raw.endswith("\n") and "\r" not in raw, "scontrol job record is unterminated"
    )
    lines = raw.splitlines()
    _require(len(lines) == 1 and lines[0], "scontrol must return exactly one job row")
    line = lines[0]
    _require(
        all(character == " " or 0x21 <= ord(character) <= 0x7E for character in line),
        "scontrol job record is not printable ASCII",
    )
    fields: dict[str, str] = {}
    for token in line.split():
        key, separator, value = token.partition("=")
        if not separator:
            continue
        if key in fields:
            raise ValueError(f"duplicate scontrol field: {key}")
        fields[key] = value
    return line, fields


def parse_scontrol_job_record(
    raw: str,
    *,
    phase: str,
    expected_job_id: int,
    expected_transaction_id: str,
) -> dict[str, Any]:
    """Parse the exact held/running controller record and enforce no requeue."""

    _require(phase in {"held", "running"}, "scheduler phase must be held or running")
    _require(
        type(expected_job_id) is int and expected_job_id > 0,
        "expected Slurm job ID must be positive",
    )
    _require(
        isinstance(expected_transaction_id, str)
        and _TRANSACTION_PATTERN.fullmatch(expected_transaction_id) is not None,
        "scheduler transaction ID is invalid",
    )
    line, fields = _one_scontrol_line(raw)
    required = {"JobId", "JobState", "Reason", "Requeue", "Restarts", "Comment"}
    _require(required.issubset(fields), "scontrol job record omits required fields")
    expected_state = "PENDING" if phase == "held" else "RUNNING"
    _require(fields["JobId"] == str(expected_job_id), "scontrol job ID mismatch")
    _require(fields["JobState"] == expected_state, f"scheduler {phase} state mismatch")
    if phase == "held":
        _require(fields["Reason"] == "JobHeldUser", "held job is not user-held")
    _require(fields["Comment"] == expected_transaction_id, "scheduler comment mismatch")
    _require(fields["Requeue"] == "0", "scheduler job permits requeue")
    _require(fields["Restarts"] == "0", "scheduler job has already restarted")
    return {
        "job_id": expected_job_id,
        "phase": phase,
        "state": fields["JobState"],
        "reason": fields["Reason"],
        "requeue": 0,
        "restarts": 0,
        "transaction_id": expected_transaction_id,
        "raw": raw,
        "raw_sha256": _sha256(raw.encode("ascii")),
        "line": line,
    }


def _scheduler_job_value(
    raw: str,
    *,
    phase: str,
    expected_job_id: int,
    expected_transaction_id: str,
    command: list[str],
) -> dict[str, Any]:
    parsed = parse_scontrol_job_record(
        raw,
        phase=phase,
        expected_job_id=expected_job_id,
        expected_transaction_id=expected_transaction_id,
    )
    return {
        "schema_version": 1,
        "profile": SCHEDULER_JOB_PROFILE,
        "status": f"{phase}_requeue_disabled_restart_count_zero",
        "command": command,
        "job": parsed,
    }


def capture_scheduler_job(
    output_path: Path,
    *,
    phase: str,
    expected_job_id: int,
    expected_transaction_id: str,
) -> dict[str, Any]:
    """Query ``scontrol`` and atomically publish one immutable job record."""

    command = [
        "scontrol",
        "show",
        "job",
        str(expected_job_id),
        "--oneliner",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    _require(
        result.returncode == 0 and result.stderr == "",
        "scontrol job-record query failed",
    )
    value = _scheduler_job_value(
        result.stdout,
        phase=phase,
        expected_job_id=expected_job_id,
        expected_transaction_id=expected_transaction_id,
        command=command,
    )
    return publish_immutable_json(Path(output_path), value)


def validate_persisted_scheduler_job(
    path: Path,
    *,
    phase: str,
    expected_job_id: int,
    expected_transaction_id: str,
) -> dict[str, Any]:
    artifact = validate_immutable_json(Path(path))
    value = artifact["value"]
    expected_command = [
        "scontrol",
        "show",
        "job",
        str(expected_job_id),
        "--oneliner",
    ]
    _require(
        isinstance(value, dict)
        and set(value) == {"schema_version", "profile", "status", "command", "job"}
        and value["schema_version"] == 1
        and value["profile"] == SCHEDULER_JOB_PROFILE
        and value["status"] == f"{phase}_requeue_disabled_restart_count_zero"
        and value["command"] == expected_command
        and isinstance(value["job"], dict),
        "scheduler job artifact schema mismatch",
    )
    parsed = parse_scontrol_job_record(
        value["job"].get("raw", ""),
        phase=phase,
        expected_job_id=expected_job_id,
        expected_transaction_id=expected_transaction_id,
    )
    _require(value["job"] == parsed, "scheduler job artifact is not canonical")
    return artifact


def parse_sacct_terminal_record(raw: str, *, expected_job_id: int) -> dict[str, Any]:
    """Parse one allocation-level accounting row and reject any restart."""

    _require(isinstance(raw, str), "sacct terminal record must be text")
    _require(
        raw.endswith("\n") and "\r" not in raw, "sacct terminal record is unterminated"
    )
    rows = [line for line in raw.splitlines() if line]
    _require(len(rows) == 1, "sacct must return exactly one terminal allocation row")
    fields = rows[0].split("|")
    _require(
        len(fields) == 9,
        "sacct terminal row is not exact parsable2 output",
    )
    job_id, state, exit_code, submitted, started, ended, elapsed, nodes, restarts = (
        fields
    )
    base_state = state.split()[0]
    timestamp = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}")
    _require(job_id == str(expected_job_id), "sacct terminal job ID mismatch")
    _require(base_state in _TERMINAL_STATES, "sacct row is not terminal")
    _require(
        re.fullmatch(r"[0-9]+:[0-9]+", exit_code) is not None,
        "sacct exit code is invalid",
    )
    _require(
        all(
            timestamp.fullmatch(value) is not None
            for value in (submitted, started, ended)
        )
        and submitted <= started <= ended,
        "sacct timestamps are invalid",
    )
    _require(elapsed.isdecimal(), "sacct elapsed duration is invalid")
    _require(bool(nodes) and nodes != "Unknown", "sacct node list is invalid")
    _require(restarts == "0", "sacct proves that the job restarted")
    _require(
        base_state == "COMPLETED" and exit_code == "0:0",
        "Slurm job did not complete cleanly",
    )
    return {
        "job_id": expected_job_id,
        "state": base_state,
        "exit_code": exit_code,
        "submitted_at": submitted,
        "started_at": started,
        "ended_at": ended,
        "elapsed_raw": int(elapsed),
        "nodes": nodes,
        "restarts": 0,
        "raw": raw,
        "raw_sha256": _sha256(raw.encode("utf-8")),
    }


def _read_success_marker(
    path: Path, *, expected_manifest_sha256: str
) -> dict[str, Any]:
    artifact_before = validate_immutable_file(path)
    payload = Path(path).read_text(encoding="utf-8")
    artifact_after = validate_immutable_file(path)
    _require(
        _identity(artifact_before) == _identity(artifact_after),
        "task success marker changed while being read",
    )
    _require(
        payload.endswith("\n") and "\r" not in payload, "task success marker is invalid"
    )
    values: dict[str, str] = {}
    for line in payload.splitlines():
        key, separator, value = line.partition("=")
        _require(
            bool(separator) and key not in values, "task success marker schema mismatch"
        )
        values[key] = value
    _require(
        values.get("status") == "success"
        and values.get("evidence_manifest_sha256") == expected_manifest_sha256,
        "task success marker does not bind the evidence manifest",
    )
    return {"artifact": _identity(artifact_after), "value": values}


def build_terminal_attestation(
    *,
    held_record_path: Path,
    running_record_path: Path,
    evidence_manifest_path: Path,
    task_success_path: Path,
    expected_job_id: int,
    expected_transaction_id: str,
    sacct_raw: str,
    sacct_command: list[str],
) -> dict[str, Any]:
    """Join held, running, evaluator, and accounting evidence after job exit."""

    held = validate_persisted_scheduler_job(
        held_record_path,
        phase="held",
        expected_job_id=expected_job_id,
        expected_transaction_id=expected_transaction_id,
    )
    running = validate_persisted_scheduler_job(
        running_record_path,
        phase="running",
        expected_job_id=expected_job_id,
        expected_transaction_id=expected_transaction_id,
    )
    evidence = validate_immutable_json(evidence_manifest_path)
    evidence_value = evidence["value"]
    _require(
        isinstance(evidence_value, dict)
        and evidence_value.get("status") == "pass"
        and isinstance(evidence_value.get("artifacts"), dict)
        and evidence_value["artifacts"].get("scheduler_running") == _identity(running),
        "evidence manifest does not bind the running scheduler record",
    )
    success = _read_success_marker(
        task_success_path, expected_manifest_sha256=evidence["sha256"]
    )
    terminal = parse_sacct_terminal_record(sacct_raw, expected_job_id=expected_job_id)
    return {
        "schema_version": 1,
        "profile": SCHEDULER_TERMINAL_PROFILE,
        "status": "completed_without_requeue_or_restart",
        "job_id": expected_job_id,
        "transaction_id": expected_transaction_id,
        "held_scheduler_record": _identity(held),
        "running_scheduler_record": _identity(running),
        "evidence_manifest": _identity(evidence),
        "task_success": success,
        "sacct_command": sacct_command,
        "terminal": terminal,
    }


def attest_terminal(
    output_path: Path,
    *,
    held_record_path: Path,
    running_record_path: Path,
    evidence_manifest_path: Path,
    task_success_path: Path,
    expected_job_id: int,
    expected_transaction_id: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    _require(0 < timeout_seconds <= 600, "terminal accounting timeout is invalid")
    command = [
        "sacct",
        "-X",
        "--noheader",
        "--parsable2",
        f"--jobs={expected_job_id}",
        "--format=JobIDRaw,State,ExitCode,Submit,Start,End,ElapsedRaw,NodeList,Restarts",
    ]
    deadline = time.monotonic() + timeout_seconds
    last_error = "sacct did not return a terminal row"
    while True:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode == 0 and result.stderr == "":
            try:
                value = build_terminal_attestation(
                    held_record_path=held_record_path,
                    running_record_path=running_record_path,
                    evidence_manifest_path=evidence_manifest_path,
                    task_success_path=task_success_path,
                    expected_job_id=expected_job_id,
                    expected_transaction_id=expected_transaction_id,
                    sacct_raw=result.stdout,
                    sacct_command=command,
                )
                return publish_immutable_json(output_path, value)
            except ValueError as error:
                last_error = str(error)
        else:
            last_error = "sacct terminal query failed"
        if time.monotonic() >= deadline:
            raise TimeoutError(last_error)
        time.sleep(1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture-job")
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--phase", choices=("held", "running"), required=True)
    capture.add_argument("--job-id", type=int, required=True)
    capture.add_argument("--transaction-id", required=True)
    terminal = subparsers.add_parser("attest-terminal")
    terminal.add_argument("--output", type=Path, required=True)
    terminal.add_argument("--held-record", type=Path, required=True)
    terminal.add_argument("--running-record", type=Path, required=True)
    terminal.add_argument("--evidence-manifest", type=Path, required=True)
    terminal.add_argument("--task-success", type=Path, required=True)
    terminal.add_argument("--job-id", type=int, required=True)
    terminal.add_argument("--transaction-id", required=True)
    terminal.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    if args.command == "capture-job":
        result = capture_scheduler_job(
            args.output,
            phase=args.phase,
            expected_job_id=args.job_id,
            expected_transaction_id=args.transaction_id,
        )
    else:
        result = attest_terminal(
            args.output,
            held_record_path=args.held_record,
            running_record_path=args.running_record,
            evidence_manifest_path=args.evidence_manifest,
            task_success_path=args.task_success,
            expected_job_id=args.job_id,
            expected_transaction_id=args.transaction_id,
            timeout_seconds=args.timeout_seconds,
        )
    print(json.dumps(_identity(result), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()


__all__ = [
    "SCHEDULER_JOB_PROFILE",
    "SCHEDULER_RUNNING_FILENAME",
    "SCHEDULER_TERMINAL_FILENAME",
    "SCHEDULER_TERMINAL_PROFILE",
    "attest_terminal",
    "build_terminal_attestation",
    "capture_scheduler_job",
    "parse_sacct_terminal_record",
    "parse_scontrol_job_record",
    "validate_persisted_scheduler_job",
]
