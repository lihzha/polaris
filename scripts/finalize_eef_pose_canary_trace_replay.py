#!/usr/bin/env python3
"""Finalize or verify one immutable PolaRiS Gate-0 replay attestation.

This host-only finalizer runs after the outer ``srun`` has returned.  It binds
the exact raw/ready pair to its immutable srun-status record, in-srun Slurm
lifecycle, variant-specific fixture, clean PolaRiS commit, saved job script,
container image, runtime assets, and all validator sources before publishing a
mode-0444 attestation without overwrite.
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
    from scripts import smoke_eef_pose_canary_trace_replay as replay
    from scripts import write_eef_pose_canary_gate0_srun_status as status_writer
except ImportError:  # Direct ``python scripts/finalize_...py`` execution.
    import smoke_eef_pose_canary_trace_replay as replay
    import write_eef_pose_canary_gate0_srun_status as status_writer


PROFILE = "polaris_eef_canary_gate0_attestation_v1"
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


class Gate0FinalizationError(ValueError):
    """Gate-0 raw evidence or launch provenance is not promotable."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Gate0FinalizationError(message)


def _mode(path: Path) -> str:
    return f"{stat.S_IMODE(path.stat().st_mode):04o}"


def _identity(path: Path, *, field: str, include_mtime: bool = False) -> dict[str, Any]:
    # Keep the publisher-visible absolute path instead of canonicalizing a
    # cluster mount alias.  Pyxis and the outer host can expose one inode as
    # /lustre/fsw and /lustre/fs11 respectively.
    path = Path(os.path.abspath(path))
    _require(path.is_file() and not path.is_symlink(), f"{field} missing/linked")
    metadata = path.stat()
    _require(metadata.st_nlink == 1, f"{field} must have one hard link")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
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
    payload = replay.strict_json_loads(path.read_bytes(), field=field)
    return payload, identity


def _hex(value: Any, *, field: str, length: int) -> str:
    _require(
        isinstance(value, str)
        and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None,
        f"{field} malformed",
    )
    return value


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


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
        _require(recorded.get(name) == current[name], f"{field} {name} changed")


def _require_live_recorded_identity(recorded: Any, *, field: str) -> dict[str, Any]:
    _require(
        isinstance(recorded, dict) and isinstance(recorded.get("path"), str),
        f"{field} recorded identity",
    )
    current = _identity(Path(recorded["path"]), field=field)
    _same_core_identity(recorded, current, field=field)
    return current


def _validate_live_assets(raw: dict[str, Any]) -> dict[str, Any]:
    assets = raw["assets"]
    scene = assets["scene"]
    result = {
        "scene": _require_live_recorded_identity(scene["scene"], field="scene USD"),
        "initial_conditions": _require_live_recorded_identity(
            scene["initial_conditions"], field="FoodBussing initial conditions"
        ),
        "revision_metadata": {},
        "robot_usd": _require_live_recorded_identity(
            assets["robot_usd"], field="robot USD"
        ),
    }
    for filename in ("initial_conditions.json", "scene.usda"):
        recorded = scene["revision_metadata"][filename]
        identity = _require_live_recorded_identity(
            recorded, field=f"Hub metadata {filename}"
        )
        _require(
            recorded.get("revision")
            == replay.EXPECTED_ASSET_CONTRACT["polaris_hub_revision"],
            f"Hub metadata revision {filename}",
        )
        result["revision_metadata"][filename] = {
            **identity,
            "revision": recorded["revision"],
        }
    _require(
        result["scene"]["sha256"] == replay.EXPECTED_ASSET_CONTRACT["scene_sha256"]
        and result["initial_conditions"]["sha256"]
        == replay.EXPECTED_ASSET_CONTRACT["initial_conditions_sha256"]
        and result["robot_usd"]["sha256"] == replay.EXPECTED_ROBOT_USD_SHA256,
        "live asset digest drift",
    )
    return result


def build_attestation(args: argparse.Namespace) -> dict[str, Any]:
    _require(args.variant in replay.EXPECTED_FIXTURES, "unknown variant")
    _require(type(args.job_id) is int and args.job_id > 0, "job ID")
    _hex(args.launch_id, field="launch ID", length=64)
    for value, field, length in (
        (args.expected_polaris_commit, "PolaRiS commit", 40),
        (args.expected_runner_sha256, "runner SHA", 64),
        (args.expected_fixture_sha256, "fixture SHA", 64),
        (args.expected_generator_sha256, "generator SHA", 64),
        (args.expected_status_writer_sha256, "status writer SHA", 64),
        (args.expected_finalizer_sha256, "finalizer SHA", 64),
        (args.expected_container_sha256, "container SHA", 64),
        (args.expected_saved_job_script_sha256, "saved job script SHA", 64),
    ):
        _hex(value, field=field, length=length)
    expected_fixture = replay.EXPECTED_FIXTURES[args.variant]
    _require(
        args.expected_fixture_sha256 == expected_fixture["sha256"],
        "CLI fixture digest is not protocol-pinned",
    )
    live_job_id = os.environ.get("SLURM_JOB_ID")
    _require(
        isinstance(live_job_id, str)
        and live_job_id.isdecimal()
        and int(live_job_id) == args.job_id,
        "outer SLURM_JOB_ID mismatch",
    )

    namespace = args.raw_result.resolve().parent
    _require(
        namespace.name == f"launch_{args.launch_id}"
        and namespace.parent.name == f"job_{args.job_id}"
        and namespace.parent.parent.name == args.variant,
        "variant/job/launch namespace",
    )
    expected_names = {
        "raw": f"gate0-{args.variant}.raw.json",
        "status": f"gate0-{args.variant}.srun-status.json",
        "attestation": f"gate0-{args.variant}.attestation.json",
    }
    _require(args.raw_result.name == expected_names["raw"], "raw filename")
    _require(args.srun_status.name == expected_names["status"], "status filename")
    _require(
        args.attestation.name == expected_names["attestation"], "attestation filename"
    )
    _require(
        args.srun_status.resolve().parent == namespace
        and args.attestation.resolve().parent == namespace,
        "evidence namespaces differ",
    )

    raw, raw_identity = _read_json(args.raw_result, field="raw result")
    _require(raw_identity["mode"] == "0444", "raw mode")
    replay.validate_capture_payload(raw)
    lifecycle = raw["lifecycle"]
    _require(
        raw["variant"] == args.variant
        and lifecycle["launch_id"] == args.launch_id
        and lifecycle["job_id"] == args.job_id,
        "raw lifecycle mismatch",
    )
    ready_path = args.raw_result.with_name(args.raw_result.name + ".ready.json")
    ready, ready_identity = _read_json(ready_path, field="ready marker")
    _require(ready_identity["mode"] == "0444", "ready marker mode")
    _require(
        ready
        == {
            "schema_version": 1,
            "profile": replay.PROFILE,
            "stage": "simulation_app_close_pending",
            "raw_result": raw_identity,
        },
        "ready marker/raw mismatch",
    )

    status, status_identity = _read_json(args.srun_status, field="srun status")
    _require(status_identity["mode"] == "0444", "srun status mode")
    _require(
        set(status) == status_writer.STATUS_FIELDS
        and status["schema_version"] == 1
        and status["profile"] == status_writer.PROFILE
        and status["variant"] == args.variant
        and status["launch_id"] == args.launch_id
        and status["job_id"] == args.job_id
        and status["srun_rc"] == 0
        and status["raw_lifecycle"] == lifecycle,
        "srun status lifecycle",
    )
    raw_with_mtime = _identity(args.raw_result, field="raw result", include_mtime=True)
    ready_with_mtime = _identity(ready_path, field="ready marker", include_mtime=True)
    _require(
        status["raw_result"] == raw_with_mtime
        and status["ready_marker"] == ready_with_mtime,
        "srun status immutable pair identity",
    )
    _require(
        0
        < status["srun_started_at_ns"]
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
    _require(raw["repository"]["commit"] == commit, "raw/repo commit mismatch")
    sources = {
        "runner": _identity(
            repo / "scripts" / "smoke_eef_pose_canary_trace_replay.py",
            field="Gate 0 runner",
        ),
        "generator": _identity(
            repo / "scripts" / "generate_eef_pose_canary_trace_fixtures.py",
            field="fixture generator",
        ),
        "status_writer": _identity(
            repo / "scripts" / "write_eef_pose_canary_gate0_srun_status.py",
            field="srun status writer",
        ),
        "finalizer": _identity(Path(__file__).resolve(), field="Gate 0 finalizer"),
    }
    production_eval = replay.validate_production_reset_source()
    _same_core_identity(
        raw["production_eval"], production_eval, field="production eval source"
    )
    _require(
        raw["production_eval"] == production_eval,
        "production reset/render source evidence changed",
    )
    sources["production_eval"] = production_eval
    _require(
        Path(replay.__file__).resolve() == Path(sources["runner"]["path"]),
        "imported runner differs from hashed runner",
    )
    for name, wanted in (
        ("runner", args.expected_runner_sha256),
        ("generator", args.expected_generator_sha256),
        ("status_writer", args.expected_status_writer_sha256),
        ("finalizer", args.expected_finalizer_sha256),
    ):
        _require(sources[name]["sha256"] == wanted, f"{name} digest mismatch")

    fixture_identity, fixture_payload, actions = replay.load_replay_fixture(
        args.variant
    )
    _require(
        fixture_identity["sha256"] == args.expected_fixture_sha256
        and len(actions) == 120
        and fixture_payload["source"]["trace_sha256"]
        == expected_fixture["trace_sha256"],
        "live fixture identity/source",
    )
    _same_core_identity(raw["fixture"], fixture_identity, field="raw/live fixture")
    container = _identity(args.container_image, field="container image")
    _require(container["sha256"] == args.expected_container_sha256, "container digest")
    runtime_job_script = _identity(args.runtime_job_script, field="runtime job script")
    saved_job_script = _identity(args.saved_job_script, field="saved job script")
    _require(
        runtime_job_script["sha256"]
        == saved_job_script["sha256"]
        == args.expected_saved_job_script_sha256
        and saved_job_script["mode"] == "0444",
        "runtime/saved job script identity",
    )
    assets = _validate_live_assets(raw)

    tail = raw["all_six_gripper_tail"]
    arm_ring = raw["arm_failure_runtime_evidence"]["controller_substep_trace"]
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
            "arm_failure_ring_capacity": arm_ring["capacity"],
            "arm_failure_ring_entries": len(arm_ring["entries"]),
            "gripper_tail_capacity": tail["capacity"],
            "gripper_tail_entries": len(tail["entries"]),
            "gripper_total_apply_entries": tail["total_apply_entries"],
            "fixture_action_count": len(actions),
            "source_trace_sha256": expected_fixture["trace_sha256"],
        },
        "provenance": {
            "polaris_repo": str(repo),
            "polaris_commit": commit,
            "sources": sources,
            "fixture": fixture_identity,
            "fixture_source": fixture_payload["source"],
            "assets": assets,
            "container_image": container,
            "runtime_job_script": runtime_job_script,
            "saved_job_script": saved_job_script,
        },
    }
    _require(set(result) == ATTESTATION_FIELDS, "attestation schema")
    return result


def _publish(path: Path, payload: dict[str, Any]) -> None:
    _require(not path.exists(), f"attestation exists: {path}")
    replay._atomic_write_immutable(path, payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("finalize", "verify"))
    parser.add_argument(
        "--variant", choices=sorted(replay.EXPECTED_FIXTURES), required=True
    )
    parser.add_argument("--launch-id", required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--raw-result", type=Path, required=True)
    parser.add_argument("--srun-status", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--polaris-repo", type=Path, required=True)
    parser.add_argument("--expected-polaris-commit", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-fixture-sha256", required=True)
    parser.add_argument("--expected-generator-sha256", required=True)
    parser.add_argument("--expected-status-writer-sha256", required=True)
    parser.add_argument("--expected-finalizer-sha256", required=True)
    parser.add_argument("--container-image", type=Path, required=True)
    parser.add_argument("--expected-container-sha256", required=True)
    parser.add_argument("--runtime-job-script", type=Path, required=True)
    parser.add_argument("--saved-job-script", type=Path, required=True)
    parser.add_argument("--expected-saved-job-script-sha256", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        expected = build_attestation(args)
        if args.mode == "finalize":
            _publish(args.attestation, expected)
        actual, identity = _read_json(args.attestation, field="attestation")
        _require(identity["mode"] == "0444", "attestation mode")
        _require(actual == expected, "attestation reconstruction mismatch")
    except (
        Gate0FinalizationError,
        OSError,
        replay.Gate0ReplayValidationError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"POLARIS_GATE0_ATTESTATION_FAIL={error}", file=sys.stderr, flush=True)
        return 1
    print(f"POLARIS_GATE0_ATTESTATION_PASS={args.attestation.resolve()}", flush=True)
    print(f"POLARIS_GATE0_ATTESTATION_SHA256={identity['sha256']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
