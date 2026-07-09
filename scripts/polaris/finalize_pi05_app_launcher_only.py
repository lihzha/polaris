#!/usr/bin/env python3
"""Externally promote one terminal AppLauncher-only Slurm allocation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

from polaris.pi05_droid_jointpos_immutable import validate_immutable_json
from polaris import pi05_droid_jointpos_scheduler as scheduler


APP_MANIFEST_COLUMNS = (
    "job_id",
    "mode",
    "task",
    "rollouts",
    "environment_seed",
    "run_namespace",
    "source_tree_sha256",
    "source_approval_sha256",
    "implementation_commit",
    "openpi_commit",
    "submitted_at",
    "batch_script_sha256",
    "submission_argv_sha256",
    "held_scheduler_record_sha256",
    "provenance_dir",
    "app_runtime_provenance_sha256",
)
APP_MANIFEST_HEADER = "\t".join(APP_MANIFEST_COLUMNS)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _stable_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    path = Path(path)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        _require(
            stat.S_ISREG(before.st_mode)
            and stat.S_IMODE(before.st_mode) == 0o644
            and before.st_nlink == 1,
            "app manifest must be one mode-0644 regular file",
        )
        payload = b""
        while block := os.read(descriptor, 1024 * 1024):
            payload += block
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.stat(path, follow_symlinks=False)
    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_gid",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    _require(
        all(
            getattr(before, name) == getattr(after, name) == getattr(current, name)
            for name in fields
        ),
        "app manifest changed while being read",
    )
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("app manifest is not UTF-8") from error
    _require(text.endswith("\n") and "\r" not in text, "app manifest is unterminated")
    lines = text.splitlines()
    _require(lines and lines[0] == APP_MANIFEST_HEADER, "app manifest header mismatch")
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        values = line.split("\t")
        _require(
            len(values) == len(APP_MANIFEST_COLUMNS), "app manifest row width mismatch"
        )
        rows.append(dict(zip(APP_MANIFEST_COLUMNS, values, strict=True)))
    import hashlib

    return (
        {
            "path": str(path.resolve(strict=True)),
            "mode": "0644",
            "nlink": 1,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        rows,
    )


def _resolve_job(manifest: Path, job_id: int) -> dict[str, Any]:
    manifest_identity, rows = _stable_manifest(manifest)
    matches = [row for row in rows if row["job_id"] == str(job_id)]
    _require(len(matches) == 1, "app manifest must contain exactly one requested job")
    row = matches[0]
    _require(
        row["mode"] == "app-launcher-only"
        and row["task"] == "DROID-FoodBussing"
        and row["rollouts"] == "1"
        and row["environment_seed"] == "0"
        and re.fullmatch(r"[A-Za-z0-9._-]+", row["run_namespace"]) is not None,
        "app manifest row is not the closed diagnostic contract",
    )
    for name, pattern in (
        ("source_tree_sha256", r"[0-9a-f]{64}"),
        ("source_approval_sha256", r"[0-9a-f]{64}"),
        ("implementation_commit", r"[0-9a-f]{40}"),
        ("openpi_commit", r"[0-9a-f]{40}"),
        ("batch_script_sha256", r"[0-9a-f]{64}"),
        ("submission_argv_sha256", r"[0-9a-f]{64}"),
        ("held_scheduler_record_sha256", r"[0-9a-f]{64}"),
        ("app_runtime_provenance_sha256", r"[0-9a-f]{64}"),
    ):
        _require(
            re.fullmatch(pattern, row[name]) is not None, f"manifest {name} malformed"
        )
    provenance = Path(row["provenance_dir"])
    expected_provenance = (
        Path(manifest_identity["path"]).parent
        / "submission_provenance"
        / f"job_{job_id}"
    )
    _require(
        provenance.is_absolute()
        and provenance.resolve(strict=True) == provenance
        and provenance == expected_provenance
        and provenance.name == f"job_{job_id}"
        and provenance.is_dir()
        and not provenance.is_symlink(),
        "app provenance directory is not canonical and job-bound",
    )
    held_path = provenance / "scheduler_held.json"
    held_candidate = validate_immutable_json(held_path)
    transaction_id = held_candidate["value"].get("job", {}).get("transaction_id")
    _require(
        isinstance(transaction_id, str)
        and re.fullmatch(r"pi05-[0-9a-f]{40}", transaction_id) is not None,
        "held scheduler transaction ID is malformed",
    )
    held = scheduler.validate_persisted_scheduler_job(
        held_path,
        phase="held",
        expected_job_id=job_id,
        expected_transaction_id=transaction_id,
    )
    _require(
        scheduler._identity(held_candidate) == scheduler._identity(held)
        and held["sha256"] == row["held_scheduler_record_sha256"],
        "manifest does not bind the held scheduler record",
    )
    approval_path = provenance / "app_runtime_approval.env"
    _, approval_payload = scheduler._stable_payload(
        approval_path, expected_sha256=row["app_runtime_provenance_sha256"]
    )
    approval = scheduler._parse_env_record(
        approval_payload, label="AppLauncher runtime approval"
    )
    provenance_value = scheduler._validate_app_provenance(
        approval_path,
        expected_sha256=row["app_runtime_provenance_sha256"],
        expected_job_id=job_id,
        expected_transaction_id=transaction_id,
        held=held,
    )
    _require(
        approval.get("batch_script_sha256") == row["batch_script_sha256"]
        and approval.get("submission_argv_sha256") == row["submission_argv_sha256"]
        and approval.get("held_scheduler_record_sha256")
        == row["held_scheduler_record_sha256"],
        "manifest and AppLauncher runtime approval disagree",
    )
    exports = provenance_value["exports"]
    source_approval = validate_immutable_json(Path(exports["POLARIS_SOURCE_APPROVAL"]))
    source_value = source_approval["value"]
    _require(
        exports["EXPECTED_POLARIS_SOURCE_TREE_SHA256"] == row["source_tree_sha256"]
        and source_approval["sha256"] == row["source_approval_sha256"]
        and isinstance(source_value, dict)
        and source_value.get("source_tree_sha256") == row["source_tree_sha256"]
        and source_value.get("implementation_commit") == row["implementation_commit"]
        and source_value.get("openpi_commit") == row["openpi_commit"],
        "manifest source identity disagrees with the submission approval",
    )
    namespace_parent = Path(approval["output_namespace_parent"])
    output_root = Path(approval["output_root"])
    _require(
        output_root.is_absolute()
        and output_root.resolve(strict=True) == output_root
        and namespace_parent == output_root / row["run_namespace"]
        and namespace_parent.resolve(strict=True) == namespace_parent
        and namespace_parent.is_dir()
        and not namespace_parent.is_symlink(),
        "AppLauncher output namespace escaped the manifest-bound output root",
    )
    run_name = f"{row['run_namespace']}_public-app-launcher-only_{row['task']}_{job_id}"
    run_dir = namespace_parent / run_name
    preterminal = run_dir / "app_launcher_only" / "preterminal_attestation.json"
    _require(
        preterminal.is_file() and not preterminal.is_symlink(),
        "derived AppLauncher preterminal attestation is missing",
    )
    return {
        "manifest": manifest_identity,
        "row": row,
        "provenance": provenance,
        "held": held_path,
        "transaction_id": transaction_id,
        "approval": approval_path,
        "preterminal": preterminal,
        "promotion": provenance / scheduler.APP_TERMINAL_PROMOTION_FILENAME,
    }


def finalize_manifest_job(
    *,
    manifest: Path,
    job_id: int,
    verify_only: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    resolved = _resolve_job(manifest, job_id)
    row = resolved["row"]
    promotion_path = resolved["promotion"]
    if verify_only:
        return scheduler.validate_app_terminal_promotion(
            promotion_path,
            expected_job_id=job_id,
            expected_transaction_id=resolved["transaction_id"],
            expected_app_runtime_approval_sha256=row["app_runtime_provenance_sha256"],
        )
    _require(
        not promotion_path.exists() and not promotion_path.is_symlink(),
        "authoritative AppLauncher promotion already exists",
    )
    return scheduler._attest_app_terminal(
        promotion_path,
        held_record_path=resolved["held"],
        app_runtime_approval_path=resolved["approval"],
        expected_app_runtime_approval_sha256=row["app_runtime_provenance_sha256"],
        preterminal_path=resolved["preterminal"],
        expected_job_id=job_id,
        expected_transaction_id=resolved["transaction_id"],
        timeout_seconds=timeout_seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    result = finalize_manifest_job(
        manifest=args.manifest,
        job_id=args.job_id,
        verify_only=args.verify_only,
        timeout_seconds=args.timeout_seconds,
    )
    print(
        json.dumps(scheduler._identity(result), sort_keys=True, separators=(",", ":"))
    )


if __name__ == "__main__":
    main()
