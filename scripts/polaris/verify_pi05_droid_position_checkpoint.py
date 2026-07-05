#!/usr/bin/env python3
"""Full-content, pre/post checkpoint attestation for the pi0.5 position canary."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
from typing import Any

from polaris.pi05_droid_native_eval_contract import (
    PI05_DROID_NATIVE_NORM_REFERENCE_PROBES,
    publish_immutable_json,
)
from polaris.pi05_droid_position_contract import (
    PI05_DROID_CHECKPOINT_BYTES,
    PI05_DROID_CHECKPOINT_MANIFEST_SHA256,
    PI05_DROID_CHECKPOINT_OBJECT_COUNT,
    PI05_DROID_CHECKPOINT_URI,
    PI05_DROID_NORM_STATS_SHA256,
    verify_profile_manifest,
)


PROFILE = "openpi_pi05_droid_position_checkpoint_full_md5_attestation_v1"
INTEGRITY_MODE = "strict_pre_post_full_md5_and_stat_identity_v1"
SNAPSHOT_PROFILE = "openpi_pi05_droid_position_run_snapshot_creation_v1"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    try:
        return json.loads(path.read_bytes(), parse_constant=reject_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read strict JSON: {path}") from error


def _stat_identity(value: os.stat_result) -> dict[str, Any]:
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "size": value.st_size,
        "mode": format(stat.S_IMODE(value.st_mode), "04o"),
        "nlink": value.st_nlink,
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


def _read_manifest(path: Path) -> list[tuple[str, int, str]]:
    report = verify_profile_manifest(path)
    if report != {
        "sha256": PI05_DROID_CHECKPOINT_MANIFEST_SHA256,
        "object_count": PI05_DROID_CHECKPOINT_OBJECT_COUNT,
        "total_bytes": PI05_DROID_CHECKPOINT_BYTES,
    }:
        raise ValueError("official pi05_droid manifest identity mismatch")
    prefix = "checkpoints/pi05_droid/"
    entries: list[tuple[str, int, str]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="ascii").splitlines(), start=1
    ):
        fields = line.split("\t")
        if (
            len(fields) != 3
            or not fields[0].startswith(prefix)
            or not fields[0][len(prefix) :]
        ):
            raise ValueError(f"invalid checkpoint manifest line {line_number}")
        relative = fields[0][len(prefix) :]
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"unsafe checkpoint manifest path: {relative}")
        entries.append((relative, int(fields[1]), fields[2]))
    return entries


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _actual_file_set(root: Path) -> set[str]:
    paths: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"checkpoint contains a symlink: {path}")
        if path.is_file():
            paths.add(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise ValueError(f"checkpoint contains a non-file object: {path}")
    return paths


def create_run_snapshot(
    source_dir: Path,
    destination_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Copy one verified source cache into an immutable, non-hardlinked tree."""

    source_requested = Path(source_dir)
    destination = Path(destination_dir)
    if source_requested.is_symlink() or destination.is_symlink():
        raise ValueError("checkpoint source/destination must not be symlinks")
    source = source_requested.resolve()
    if not source.is_dir():
        raise ValueError("checkpoint source cache is missing")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"checkpoint snapshot already exists: {destination}")
    destination_parent = destination.parent.resolve()
    if not destination_parent.is_dir() or destination_parent.is_symlink():
        raise ValueError("checkpoint snapshot parent is invalid")
    destination = destination_parent / destination.name
    temporary = destination_parent / f".{destination.name}.partial-{os.getpid()}"
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"checkpoint snapshot temporary exists: {temporary}")

    entries = _read_manifest(Path(manifest_path))
    expected_paths = {relative for relative, _, _ in entries}
    if _actual_file_set(source) != expected_paths:
        raise ValueError("checkpoint source cache object set mismatch")
    source_root_before = _stat_identity(os.stat(source, follow_symlinks=False))
    copied: list[dict[str, Any]] = []
    try:
        temporary.mkdir(mode=0o700)
        for relative, expected_size, expected_md5 in entries:
            source_path = source / relative
            output_path = temporary / relative
            output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            source_descriptor = os.open(
                source_path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                output_descriptor = os.open(
                    output_path,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o400,
                )
                try:
                    source_before = os.fstat(source_descriptor)
                    if (
                        not stat.S_ISREG(source_before.st_mode)
                        or source_before.st_size != expected_size
                    ):
                        raise ValueError(
                            f"checkpoint source identity mismatch: {relative}"
                        )
                    digest = hashlib.md5(usedforsecurity=False)
                    copied_size = 0
                    while True:
                        block = os.read(source_descriptor, 16 * 1024 * 1024)
                        if not block:
                            break
                        digest.update(block)
                        view = memoryview(block)
                        while view:
                            written = os.write(output_descriptor, view)
                            if written <= 0:
                                raise OSError(
                                    "checkpoint snapshot write made no progress"
                                )
                            view = view[written:]
                        copied_size += len(block)
                    os.fsync(output_descriptor)
                    os.fchmod(output_descriptor, 0o444)
                    os.fsync(output_descriptor)
                    source_after = os.fstat(source_descriptor)
                    output_stat = os.fstat(output_descriptor)
                finally:
                    os.close(output_descriptor)
            finally:
                os.close(source_descriptor)
            md5_base64 = base64.b64encode(digest.digest()).decode("ascii")
            if (
                copied_size != expected_size
                or md5_base64 != expected_md5
                or _stat_identity(source_after) != _stat_identity(source_before)
                or not stat.S_ISREG(output_stat.st_mode)
                or stat.S_IMODE(output_stat.st_mode) != 0o444
                or output_stat.st_nlink != 1
                or output_stat.st_size != expected_size
            ):
                raise ValueError(f"checkpoint snapshot copy mismatch: {relative}")
            copied.append(
                {
                    "relative_path": relative,
                    "size": expected_size,
                    "md5_base64": md5_base64,
                    "source_stat": _stat_identity(source_after),
                    "snapshot_stat": _stat_identity(output_stat),
                }
            )
        for directory in sorted(
            (path for path in temporary.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            directory.chmod(0o555)
            _fsync_directory(directory)
        temporary.chmod(0o555)
        _fsync_directory(temporary)
        os.rename(temporary, destination)
        _fsync_directory(destination_parent)
    except BaseException:
        if temporary.exists() and not temporary.is_symlink():
            for path in temporary.rglob("*"):
                if path.is_dir():
                    path.chmod(0o700)
                elif path.is_file():
                    path.chmod(0o600)
            temporary.chmod(0o700)
            shutil.rmtree(temporary)
        raise
    if _actual_file_set(destination) != expected_paths:
        raise ValueError("published checkpoint snapshot object set mismatch")
    snapshot_root_stat = _stat_identity(os.stat(destination, follow_symlinks=False))
    source_root_after = _stat_identity(os.stat(source, follow_symlinks=False))
    if source_root_after != source_root_before:
        raise ValueError("checkpoint source root changed during snapshot copy")
    return {
        "schema_version": 1,
        "profile": SNAPSHOT_PROFILE,
        "status": "pass",
        "copy_semantics": "independent_byte_copy_no_hardlinks_no_reflinks_v1",
        "checkpoint_uri": PI05_DROID_CHECKPOINT_URI,
        "source_dir": str(source),
        "source_root_stat": source_root_after,
        "snapshot_dir": str(destination.resolve()),
        "snapshot_root_stat": snapshot_root_stat,
        "manifest_sha256": PI05_DROID_CHECKPOINT_MANIFEST_SHA256,
        "object_count": PI05_DROID_CHECKPOINT_OBJECT_COUNT,
        "total_bytes": PI05_DROID_CHECKPOINT_BYTES,
        "objects": copied,
    }


def _normalization_reference(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("global DROID norm stats must be one regular file")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != PI05_DROID_NORM_STATS_SHA256:
        raise ValueError("global DROID norm stats SHA-256 mismatch")
    value = _strict_json(path)
    if not isinstance(value, dict) or set(value) != {"norm_stats"}:
        raise ValueError("global DROID norm stats root schema mismatch")
    groups = value["norm_stats"]
    if not isinstance(groups, dict) or set(groups) != {"actions", "state"}:
        raise ValueError("global DROID norm stats group schema mismatch")
    for group_name in ("actions", "state"):
        group = groups[group_name]
        if not isinstance(group, dict) or set(group) != {"mean", "std", "q01", "q99"}:
            raise ValueError(f"global DROID {group_name} stats schema mismatch")
        for statistic in ("mean", "std", "q01", "q99"):
            vector = group[statistic]
            if (
                not isinstance(vector, list)
                or len(vector) != 32
                or any(
                    type(item) not in (int, float)
                    or isinstance(item, bool)
                    or not math.isfinite(item)
                    for item in vector
                )
            ):
                raise ValueError(
                    f"global DROID {group_name} {statistic} vector mismatch"
                )
    probes = {
        "actions_q01_first8": groups["actions"]["q01"][:8],
        "actions_q99_first8": groups["actions"]["q99"][:8],
        "state_q01_first8": groups["state"]["q01"][:8],
        "state_q99_first8": groups["state"]["q99"][:8],
    }
    if _canonical_json_bytes(probes) != _canonical_json_bytes(
        PI05_DROID_NATIVE_NORM_REFERENCE_PROBES
    ):
        raise ValueError("global DROID normalization probes mismatch")
    return {
        "sha256": PI05_DROID_NORM_STATS_SHA256,
        "path_within_checkpoint": "assets/droid/norm_stats.json",
        "scope": "checkpoint_global_droid",
        "asset_id": "droid",
        "category_override": "forbidden",
        "rejected_category_substitutions": ["single_arm", "single-arm", "single arm"],
        "probes": probes,
        "state_semantics": "panda_joint_position_plus_closed_positive_gripper",
        "model_action_semantics": "normalized_droid_joint_velocity_command",
        "simulator_adapter": (
            "fresh_measured_q_plus_0p2_times_clipped_command_to_absolute_position"
        ),
    }


def attest_checkpoint(
    checkpoint_dir: Path,
    manifest_path: Path,
    *,
    verification_phase: str,
) -> dict[str, Any]:
    if verification_phase not in {"pre_server", "post_server"}:
        raise ValueError("checkpoint verification phase mismatch")
    requested = Path(checkpoint_dir)
    if requested.is_symlink():
        raise ValueError("checkpoint root must not be a symlink")
    root = requested.resolve()
    if not root.is_dir():
        raise ValueError("checkpoint root is missing")
    entries = _read_manifest(Path(manifest_path))
    expected_paths = {relative for relative, _, _ in entries}
    actual_paths = _actual_file_set(root)
    if actual_paths != expected_paths:
        raise ValueError(
            "checkpoint object set mismatch: "
            f"missing={sorted(expected_paths - actual_paths)} "
            f"extra={sorted(actual_paths - expected_paths)}"
        )

    root_before = os.stat(root, follow_symlinks=False)
    if (
        not stat.S_ISDIR(root_before.st_mode)
        or stat.S_IMODE(root_before.st_mode) != 0o555
    ):
        raise ValueError("checkpoint snapshot root must have mode 0555")
    for directory in (path for path in root.rglob("*") if path.is_dir()):
        directory_stat = os.stat(directory, follow_symlinks=False)
        if stat.S_IMODE(directory_stat.st_mode) != 0o555:
            raise ValueError(
                f"checkpoint snapshot directory must have mode 0555: {directory}"
            )
    objects: list[dict[str, Any]] = []
    for relative, expected_size, expected_md5 in entries:
        path = root / relative
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_size != expected_size
                or stat.S_IMODE(before.st_mode) != 0o444
                or before.st_nlink != 1
            ):
                raise ValueError(f"checkpoint object identity mismatch: {relative}")
            digest = hashlib.md5(usedforsecurity=False)
            while True:
                block = os.read(descriptor, 16 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        if _stat_identity(after) != _stat_identity(before):
            raise ValueError(f"checkpoint object changed while hashing: {relative}")
        md5_base64 = base64.b64encode(digest.digest()).decode("ascii")
        if md5_base64 != expected_md5:
            raise ValueError(f"checkpoint MD5 mismatch: {relative}")
        objects.append(
            {
                "relative_path": relative,
                "size": expected_size,
                "md5_base64": md5_base64,
                "stat": _stat_identity(after),
            }
        )
    root_after = os.stat(root, follow_symlinks=False)
    if _stat_identity(root_after) != _stat_identity(root_before):
        raise ValueError("checkpoint root changed during full verification")
    normalization = _normalization_reference(root / "assets/droid/norm_stats.json")
    return {
        "schema_version": 1,
        "profile": PROFILE,
        "status": "pass",
        "verification_phase": verification_phase,
        "integrity_mode": INTEGRITY_MODE,
        "checkpoint_uri": PI05_DROID_CHECKPOINT_URI,
        "checkpoint_dir": str(root),
        "manifest": {
            "path": str(Path(manifest_path).resolve()),
            "sha256": PI05_DROID_CHECKPOINT_MANIFEST_SHA256,
            "object_count": PI05_DROID_CHECKPOINT_OBJECT_COUNT,
            "total_bytes": PI05_DROID_CHECKPOINT_BYTES,
        },
        "root_stat": _stat_identity(root_after),
        "objects": objects,
        "normalization": normalization,
        "full_md5": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot_parser = subparsers.add_parser("snapshot")
    snapshot_parser.add_argument("--source", type=Path, required=True)
    snapshot_parser.add_argument("--destination", type=Path, required=True)
    snapshot_parser.add_argument("--manifest", type=Path, required=True)
    snapshot_parser.add_argument("--output", type=Path, required=True)
    attest_parser = subparsers.add_parser("attest")
    attest_parser.add_argument("checkpoint", type=Path)
    attest_parser.add_argument("manifest", type=Path)
    attest_parser.add_argument(
        "--phase", choices=("pre_server", "post_server"), required=True
    )
    attest_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "snapshot":
        publish_immutable_json(
            args.output,
            create_run_snapshot(args.source, args.destination, args.manifest),
        )
        return
    publish_immutable_json(
        args.output,
        attest_checkpoint(
            args.checkpoint,
            args.manifest,
            verification_phase=args.phase,
        ),
    )


if __name__ == "__main__":
    main()
