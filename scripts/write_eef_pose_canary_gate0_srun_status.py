#!/usr/bin/env python3
"""Publish immutable post-srun status for one Gate-0 replay lifecycle."""

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
    from scripts import smoke_eef_pose_canary_trace_replay as replay
except ImportError:  # Direct ``python scripts/write_...py`` execution.
    import smoke_eef_pose_canary_trace_replay as replay


PROFILE = "polaris_eef_canary_gate0_srun_status_v1"
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


class SrunStatusError(ValueError):
    """The outer srun lifecycle cannot be bound to the immutable raw pair."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SrunStatusError(message)


def _identity(path: Path, *, field: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"{field} missing/linked")
    metadata = path.stat()
    _require(metadata.st_nlink == 1, f"{field} must have one hard link")
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "size_bytes": len(data),
        "sha256": replay._sha256(data),
        "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
        "mtime_ns": metadata.st_mtime_ns,
    }


def build_status(args: argparse.Namespace) -> dict[str, Any]:
    _require(args.variant in replay.EXPECTED_FIXTURES, "unknown variant")
    _require(
        re.fullmatch(r"[0-9a-f]{64}", args.launch_id) is not None,
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
    expected_raw_name = f"gate0-{args.variant}.raw.json"
    expected_status_name = f"gate0-{args.variant}.srun-status.json"
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
    raw = replay.strict_json_loads(args.raw_result.read_bytes(), field="raw result")
    replay.validate_capture_payload(raw)
    lifecycle = raw["lifecycle"]
    _require(
        raw["variant"] == args.variant
        and lifecycle["launch_id"] == args.launch_id
        and lifecycle["job_id"] == args.job_id,
        "raw lifecycle/outer srun mismatch",
    )
    marker_path = args.raw_result.with_name(args.raw_result.name + ".ready.json")
    marker_identity = _identity(marker_path, field="ready marker")
    _require(marker_identity["mode"] == "0444", "ready marker mode")
    marker = replay.strict_json_loads(marker_path.read_bytes(), field="ready marker")
    expected_raw_identity = {
        key: raw_identity[key] for key in ("path", "size_bytes", "sha256", "mode")
    }
    _require(
        marker
        == {
            "schema_version": 1,
            "profile": replay.PROFILE,
            "stage": "simulation_app_close_pending",
            "raw_result": expected_raw_identity,
        },
        "ready marker/raw binding",
    )
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
        "--variant", choices=sorted(replay.EXPECTED_FIXTURES), required=True
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
        replay._atomic_write_immutable(args.status, payload)
    except (OSError, replay.Gate0ReplayValidationError, SrunStatusError) as error:
        print(f"POLARIS_GATE0_SRUN_STATUS_FAIL={error}", file=sys.stderr, flush=True)
        return 1
    print(f"POLARIS_GATE0_SRUN_STATUS_PASS={args.status.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
