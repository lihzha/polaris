#!/usr/bin/env python3
"""Audit one official pi0.5-DROID position-adapter episode transaction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import stat
from typing import Any

from polaris.pi05_droid_native_eval_contract import (
    canonical_json_bytes,
    publish_immutable_json,
    validate_immutable_json,
)
from polaris.pi05_droid_position_adapter import PI05_DROID_POSITION_ADAPTER_PROFILE
from polaris.pi05_droid_position_runtime import (
    validate_position_adapter_runtime_report,
    validate_position_close_ready,
    validate_position_episode_sidecar,
    validate_position_failure_close_ready,
    validate_position_failure_sidecar,
)
from polaris.policy.droid_delta_position_client import (
    validate_position_trace,
    validate_position_trace_record,
)


AUDIT_PROFILE = "openpi_pi05_droid_position_episode_transaction_audit_v1"
TASK = "DROID-FoodBussing"
METRIC_COLUMNS = (
    "episode",
    "episode_length",
    "success",
    "progress",
    "numerical_failure",
    "numerical_failure_reason",
)


def _strict_loads(payload: str, *, field: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{field} contains forbidden constant {value}")

    try:
        return json.loads(payload, parse_constant=reject_constant)
    except json.JSONDecodeError as error:
        raise ValueError(f"{field} is not strict JSON") from error


def _regular_identity(
    path: Path, *, field: str, require_mode_0444: bool
) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"{field} must not be a symlink")
    file_stat = path.stat()
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
        raise ValueError(f"{field} must be one regular link")
    if require_mode_0444 and stat.S_IMODE(file_stat.st_mode) != 0o444:
        raise ValueError(f"{field} must have mode 0444")
    payload = path.read_bytes()
    if not payload:
        raise ValueError(f"{field} is empty")
    return {
        "path": str(path.resolve()),
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "mode": format(stat.S_IMODE(file_stat.st_mode), "04o"),
        "nlink": file_stat.st_nlink,
    }


def _require_bound_artifact(
    expected: Any,
    actual: dict[str, Any],
    *,
    field: str,
) -> None:
    if not isinstance(expected, dict) or set(expected) != {
        "path",
        "size",
        "sha256",
        "mode",
        "nlink",
    }:
        raise ValueError(f"{field} artifact schema mismatch")
    if Path(expected["path"]).resolve() != Path(actual["path"]):
        raise ValueError(f"{field} artifact path mismatch")
    if any(expected[key] != actual[key] for key in ("size", "sha256", "mode", "nlink")):
        raise ValueError(f"{field} artifact identity mismatch")


def _parse_bool(value: str, *, field: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{field} must be exactly True or False")


def _parse_metrics(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = _regular_identity(path, field="metrics CSV", require_mode_0444=True)
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != METRIC_COLUMNS:
                raise ValueError("metrics CSV column contract mismatch")
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise ValueError("metrics CSV is unreadable") from error
    if len(rows) != 1 or set(rows[0]) != set(METRIC_COLUMNS):
        raise ValueError("metrics CSV must contain exactly one episode")
    row = rows[0]
    try:
        result = {
            "episode": int(row["episode"]),
            "episode_length": int(row["episode_length"]),
            "success": _parse_bool(row["success"], field="metrics success"),
            "progress": float(row["progress"]),
            "numerical_failure": _parse_bool(
                row["numerical_failure"], field="metrics numerical_failure"
            ),
            "numerical_failure_reason": row["numerical_failure_reason"],
        }
    except ValueError as error:
        raise ValueError("metrics CSV value contract mismatch") from error
    if (
        result["episode"] != 0
        or not 1 <= result["episode_length"] <= 450
        or not math.isfinite(result["progress"])
        or not isinstance(result["numerical_failure_reason"], str)
    ):
        raise ValueError("metrics CSV episode identity mismatch")
    return result, identity


def _read_trace(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    identity = _regular_identity(path, field="policy trace", require_mode_0444=True)
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError("policy trace is unreadable") from error
    if not lines or any(not line for line in lines):
        raise ValueError("policy trace contains an empty or missing record")
    records = [
        validate_position_trace_record(
            _strict_loads(line, field=f"policy trace line {index}")
        )
        for index, line in enumerate(lines, start=1)
    ]
    return records, identity


def _audit_failure_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if (
        len(records) < 2
        or records[0]["record_type"] != "openpi_droid_position_rollout_start"
        or records[-1]["record_type"] != "openpi_droid_position_rollout_failure"
    ):
        raise ValueError("numerical-failure trace boundary mismatch")
    identity_fields = (
        "profile",
        "serving_contract_sha256",
        "serving_contract_artifact_sha256",
        "reset_index",
    )
    identity = {field: records[0][field] for field in identity_fields}
    query_count = 0
    executions = 0
    active_query = -1
    next_chunk_index = 0
    pending_action: dict[str, Any] | None = None
    for record in records[1:-1]:
        if any(record[field] != identity[field] for field in identity_fields):
            raise ValueError("numerical-failure trace identity drift")
        record_type = record["record_type"]
        if record_type == "openpi_droid_position_query":
            if pending_action is not None or next_chunk_index not in (0, 8):
                raise ValueError("numerical-failure query grouping mismatch")
            if record["query_index"] != query_count:
                raise ValueError("numerical-failure query index mismatch")
            active_query = query_count
            query_count += 1
            next_chunk_index = 0
        elif record_type == "openpi_droid_position_action":
            if pending_action is not None or active_query < 0:
                raise ValueError("numerical-failure action ordering mismatch")
            if (
                record["query_index"] != active_query
                or record["chunk_action_index"] != next_chunk_index
            ):
                raise ValueError("numerical-failure action index mismatch")
            pending_action = record
        elif record_type == "openpi_droid_position_execution":
            if pending_action is None:
                raise ValueError("numerical-failure execution has no action")
            if (
                record["query_index"] != pending_action["query_index"]
                or record["chunk_action_index"] != pending_action["chunk_action_index"]
                or record["outer_step_index"] != executions
            ):
                raise ValueError("numerical-failure execution identity mismatch")
            pending_action = None
            executions += 1
            next_chunk_index += 1
        else:
            raise ValueError("unexpected record in numerical-failure trace")
    terminal = records[-1]
    if any(terminal[field] != identity[field] for field in identity_fields):
        raise ValueError("numerical-failure terminal identity drift")
    failure = terminal["terminal_failure"]
    if failure["outer_steps_completed"] != executions:
        raise ValueError("numerical-failure completed-step count mismatch")
    if failure["sample_kind"] == "apply_entry":
        if pending_action is None or failure["actions_attempted"] != executions + 1:
            raise ValueError("apply-entry failure boundary mismatch")
    elif pending_action is not None:
        raise ValueError("numerical failure retained an unexpected pending action")
    if failure["actions_attempted"] not in {executions, executions + 1}:
        raise ValueError("numerical-failure attempted-step count mismatch")
    expected_queries = (failure["actions_attempted"] + 7) // 8
    if query_count != expected_queries:
        raise ValueError("numerical-failure query count mismatch")
    return {
        "status": "numerical_failure",
        "executions": executions,
        "queries": query_count,
        "terminal_failure": failure,
    }


def audit_position_episode(
    *,
    trace_path: Path,
    metrics_path: Path,
    runtime_path: Path,
    close_ready_path: Path,
    sidecar_path: Path,
    video_path: Path,
) -> dict[str, Any]:
    records, trace_identity = _read_trace(trace_path)
    metrics, metrics_identity = _parse_metrics(metrics_path)
    runtime_artifact = validate_immutable_json(runtime_path)
    runtime = validate_position_adapter_runtime_report(runtime_artifact["value"])
    close_artifact = validate_immutable_json(close_ready_path)
    sidecar_artifact = validate_immutable_json(sidecar_path)
    video_identity = _regular_identity(
        video_path, field="rollout video", require_mode_0444=True
    )

    runtime_identity = {
        key: runtime_artifact[key]
        for key in ("path", "size", "sha256", "mode", "nlink")
    }
    close_identity = {
        key: close_artifact[key] for key in ("path", "size", "sha256", "mode", "nlink")
    }
    sidecar_identity = {
        key: sidecar_artifact[key]
        for key in ("path", "size", "sha256", "mode", "nlink")
    }

    if records[0]["profile"] != PI05_DROID_POSITION_ADAPTER_PROFILE:
        raise ValueError("trace uses the wrong controller profile")
    if metrics["numerical_failure"]:
        trace_summary = _audit_failure_records(records)
        close = validate_position_failure_close_ready(close_artifact["value"])
        sidecar = validate_position_failure_sidecar(sidecar_artifact["value"])
        terminal = trace_summary["terminal_failure"]
        if (
            metrics["success"]
            or metrics["progress"] != 0.0
            or metrics["numerical_failure_reason"] != terminal["reason"]
            or metrics["episode_length"] != terminal["actions_attempted"]
            or close["terminal_failure"] != terminal
            or sidecar["terminal_failure"] != terminal
            or sidecar["episode_result"] != metrics
        ):
            raise ValueError("numerical-failure trace/metrics/sidecar mismatch")
        incident_path = Path(sidecar["incident_artifact"]["path"])
        incident_identity = _regular_identity(
            incident_path, field="numerical incident", require_mode_0444=True
        )
        _require_bound_artifact(
            sidecar["incident_artifact"], incident_identity, field="incident"
        )
        _require_bound_artifact(
            terminal["incident_artifact"], incident_identity, field="terminal incident"
        )
        apply_calls = sidecar["dynamic_report"]["apply_calls"]
        scientific_outcome = "numerical_failure"
    else:
        if metrics["episode_length"] != 450 or metrics["numerical_failure_reason"]:
            raise ValueError("nonfailed canary must be one complete 450-step rollout")
        trace_summary = validate_position_trace(records, expected_executions=450)
        if trace_summary != {"executions": 450, "queries": 57, "status": "pass"}:
            raise ValueError("complete canary trace cadence mismatch")
        close = validate_position_close_ready(close_artifact["value"])
        sidecar = validate_position_episode_sidecar(sidecar_artifact["value"])
        terminal = records[-1]["terminal_rollout"]
        if (
            sidecar["episode_result"] != metrics
            or sidecar["terminal_rollout"] != terminal
            or close["terminal_rollout"] != terminal
            or sidecar["safety_report"] != close["safety_report"]
            or sidecar["safety_report"]["outer_steps"] != 450
            or sidecar["safety_report"]["apply_calls"] != 3600
            or sidecar["safety_report"]["post_policy_step_samples"] != 450
        ):
            raise ValueError("complete trace/metrics/sidecar cadence mismatch")
        apply_calls = sidecar["safety_report"]["apply_calls"]
        scientific_outcome = "task_success" if metrics["success"] else "task_failure"

    for field, actual in (
        ("runtime", runtime_identity),
        ("trace", trace_identity),
        ("video", video_identity),
        ("metrics", metrics_identity),
        ("sidecar", sidecar_identity),
    ):
        _require_bound_artifact(
            close[f"{field}_artifact"], actual, field=f"close-ready {field}"
        )
    _require_bound_artifact(
        sidecar["trace_artifact"], trace_identity, field="sidecar trace"
    )
    _require_bound_artifact(
        sidecar["video_artifact"], video_identity, field="sidecar video"
    )
    if sidecar["environment_runtime_contract"] != close["environment_runtime_contract"]:
        raise ValueError("environment runtime contract drifted across transaction")

    return {
        "schema_version": 1,
        "profile": AUDIT_PROFILE,
        "task": TASK,
        "controller_profile": PI05_DROID_POSITION_ADAPTER_PROFILE,
        "scientific_outcome": scientific_outcome,
        "metrics": metrics,
        "trace_summary": trace_summary,
        "execution_cadence": {
            "queries": trace_summary["queries"],
            "executions": trace_summary["executions"],
            "apply_calls": apply_calls,
        },
        "runtime_sha256": runtime["runtime_sha256"],
        "runtime_contract": runtime,
        "serving_contract_sha256": records[0]["serving_contract_sha256"],
        "serving_contract_artifact_sha256": records[0][
            "serving_contract_artifact_sha256"
        ],
        "environment_runtime_contract": sidecar["environment_runtime_contract"],
        "artifacts": {
            "trace": trace_identity,
            "metrics": metrics_identity,
            "runtime": runtime_identity,
            "close_ready": close_identity,
            "sidecar": sidecar_identity,
            "video": video_identity,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--close-ready", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_position_episode(
        trace_path=args.trace,
        metrics_path=args.metrics,
        runtime_path=args.runtime,
        close_ready_path=args.close_ready,
        sidecar_path=args.sidecar,
        video_path=args.video,
    )
    publish_immutable_json(args.output, result)
    print(canonical_json_bytes(result).decode("ascii"), end="")


if __name__ == "__main__":
    main()
