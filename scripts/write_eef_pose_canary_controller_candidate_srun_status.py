#!/usr/bin/env python3
"""Publish immutable post-srun status for one controller-candidate replay."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import stat
import sys
import time
from typing import Any

try:
    from scripts import smoke_eef_pose_canary_controller_candidate as candidate
    from scripts import smoke_eef_pose_canary_trace_replay as gate0
    from scripts import validate_eef_pose_canary_controller_candidate as validator
except ImportError:  # Direct ``python scripts/write_...py`` execution.
    import smoke_eef_pose_canary_controller_candidate as candidate
    import smoke_eef_pose_canary_trace_replay as gate0
    import validate_eef_pose_canary_controller_candidate as validator


PROFILE = "polaris_eef_controller_candidate_srun_status_v1"
STATUS_FIELDS = {
    "schema_version",
    "profile",
    "variant",
    "launch_id",
    "job_id",
    "srun_rc",
    "srun_started_at_ns",
    "srun_returned_at_ns",
    "status_published_at_ns",
    "raw_result",
    "ready_marker",
    "raw_lifecycle",
}


class CandidateSrunStatusError(ValueError):
    """The outer srun cannot be bound to a promotable immutable raw pair."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateSrunStatusError(message)


def _identity(path: Path, *, field: str) -> dict[str, Any]:
    # Preserve the publisher-visible absolute path.  Host and container may
    # expose the same Lustre inode through different mount aliases.
    path = Path(os.path.abspath(path))
    _require(path.is_file() and not path.is_symlink(), f"{field} missing/linked")
    metadata = path.stat()
    _require(metadata.st_nlink == 1, f"{field} must have one hard link")
    data = path.read_bytes()
    return {
        "path": str(path),
        "size_bytes": len(data),
        "sha256": gate0._sha256(data),
        "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
        "mtime_ns": metadata.st_mtime_ns,
    }


def _same_file_identity(recorded: Any, current: dict[str, Any], *, field: str) -> None:
    _require(
        isinstance(recorded, dict) and set(recorded) == validator.FILE_IDENTITY_FIELDS,
        f"{field} identity schema",
    )
    for name in ("size_bytes", "sha256", "mode"):
        _require(
            validator._typed_equal(recorded.get(name), current[name]),
            f"{field} {name} value/type drift",
        )
    try:
        same_file = os.path.samefile(recorded.get("path"), current["path"])
    except (OSError, TypeError):
        same_file = False
    _require(same_file, f"{field} path drift")


def build_status(args: argparse.Namespace) -> dict[str, Any]:
    _require(args.variant in candidate.CANDIDATE_BY_VARIANT, "unknown variant")
    _require(
        isinstance(args.launch_id, str)
        and re.fullmatch(r"[0-9a-f]{64}", args.launch_id) is not None,
        "launch ID",
    )
    _require(type(args.job_id) is int and args.job_id > 0, "job ID")
    _require(args.srun_rc == 0, "only a zero srun may publish promotable status")
    _require(
        type(args.srun_started_at_ns) is int
        and type(args.srun_returned_at_ns) is int
        and 0 < args.srun_started_at_ns <= args.srun_returned_at_ns,
        "srun time interval",
    )
    live_job_id = os.environ.get("SLURM_JOB_ID")
    _require(
        isinstance(live_job_id, str)
        and live_job_id.isdecimal()
        and int(live_job_id) == args.job_id,
        "outer job ID/SLURM_JOB_ID mismatch",
    )

    expected_raw_name = f"candidate-{args.variant}.raw.json"
    expected_status_name = f"candidate-{args.variant}.srun-status.json"
    _require(args.raw_result.name == expected_raw_name, "raw filename")
    _require(args.status.name == expected_status_name, "status filename")
    _require(
        args.status.parent.resolve() == args.raw_result.parent.resolve(),
        "raw/status directories differ",
    )
    _require(
        args.raw_result.parent.name == f"launch_{args.launch_id}"
        and args.raw_result.parent.parent.name == f"job_{args.job_id}"
        and args.raw_result.parent.parent.parent.name == args.variant,
        "variant/job/launch namespace",
    )

    raw_identity = _identity(args.raw_result, field="raw result")
    _require(raw_identity["mode"] == "0444", "raw mode")
    raw = gate0.strict_json_loads(args.raw_result.read_bytes(), field="candidate raw")
    _require(set(raw) == validator.RAW_FIELDS, "candidate raw schema")
    lifecycle = raw.get("lifecycle")
    _require(
        type(raw.get("schema_version")) is int
        and raw.get("schema_version") == 1
        and raw.get("profile") == candidate.PROFILE
        and raw.get("finalized") is False
        and raw.get("passed") is True
        and raw.get("stage") == "simulation_app_close_pending"
        and raw.get("variant") == args.variant
        and isinstance(lifecycle, dict)
        and set(lifecycle) == validator.LIFECYCLE_FIELDS
        and type(lifecycle.get("job_id")) is int
        and lifecycle.get("job_id") == args.job_id
        and type(lifecycle.get("launch_id")) is str
        and lifecycle.get("launch_id") == args.launch_id,
        "raw lifecycle/outer srun mismatch",
    )

    marker_path = args.raw_result.with_name(args.raw_result.name + ".ready.json")
    marker_identity = _identity(marker_path, field="ready marker")
    _require(marker_identity["mode"] == "0444", "ready marker mode")
    marker = gate0.strict_json_loads(marker_path.read_bytes(), field="candidate ready")
    _require(
        isinstance(marker, dict)
        and set(marker) == validator.READY_FIELDS
        and type(marker.get("schema_version")) is int
        and marker.get("schema_version") == 1
        and marker.get("profile") == candidate.PROFILE
        and marker.get("stage") == "simulation_app_close_pending",
        "ready marker schema/identity",
    )
    _same_file_identity(marker.get("raw_result"), raw_identity, field="ready/raw")

    published_at_ns = time.time_ns()
    _require(
        published_at_ns >= args.srun_returned_at_ns,
        "status publication predates srun return",
    )
    result = {
        "schema_version": 1,
        "profile": PROFILE,
        "variant": args.variant,
        "launch_id": args.launch_id,
        "job_id": args.job_id,
        "srun_rc": args.srun_rc,
        "srun_started_at_ns": args.srun_started_at_ns,
        "srun_returned_at_ns": args.srun_returned_at_ns,
        "status_published_at_ns": published_at_ns,
        "raw_result": raw_identity,
        "ready_marker": marker_identity,
        "raw_lifecycle": lifecycle,
    }
    _require(set(result) == STATUS_FIELDS, "status schema")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=sorted(candidate.CANDIDATE_BY_VARIANT), required=True
    )
    parser.add_argument("--launch-id", required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--srun-rc", type=int, required=True)
    parser.add_argument("--srun-started-at-ns", type=int, required=True)
    parser.add_argument("--srun-returned-at-ns", type=int, required=True)
    parser.add_argument("--raw-result", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        payload = build_status(args)
        gate0._atomic_write_immutable(args.status, payload)
    except (
        CandidateSrunStatusError,
        OSError,
        gate0.Gate0ReplayValidationError,
    ) as error:
        print(
            f"POLARIS_CONTROLLER_CANDIDATE_SRUN_STATUS_FAIL={error}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(
        f"POLARIS_CONTROLLER_CANDIDATE_SRUN_STATUS_PASS={args.status.resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
