#!/usr/bin/env python3
"""Host-only validation and staged attestation for the gripper diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
DIAGNOSTIC_PATH = SCRIPT_PATH.with_name("smoke_eef_pose_gripper_impulse_diagnostic.py")
BOUNDARY_PATH = SCRIPT_PATH.with_name("smoke_eef_pose_boundary_replay.py")
FIXTURE_PATH = (
    SCRIPT_PATH.parent
    / "fixtures"
    / ("official_lap3b_foodbussing_v3_boundary_actions.json")
)
MODES = ("exact", "delay_first_close_one_step")
EXPECTED_BOUNDARY_SHA256 = (
    "a63f2a8ab9c42ea872da9d6e1913d43e0a89b0382c01d88071af19bdf2731d97"
)
EXPECTED_FIXTURE_SHA256 = (
    "640a11df435b6a8d05e924a3781c86e121dec15477211aa67f301423c539d910"
)
diagnostic: Any | None = None


ATTESTATION_PROFILE = "gripper_impulse_post_kit_staged_attestation_v6"
ATTESTATION_FIELDS = {
    "schema_version",
    "profile",
    "stage",
    "mode",
    "intended_attestation_path",
    "outcome",
    "artifacts",
    "runtime_identity",
    "sources",
    "repository",
    "execution",
    "assets",
}
ARTIFACT_FIELDS = {
    "capture",
    "video",
    "ready_marker",
    "runtime_exit",
    "outer_srun_status",
    "validator_status",
}
SOURCE_FIELDS = {
    "imported_diagnostic_file",
    "diagnostic",
    "finalizer",
    "boundary_helper",
    "replay_fixture",
}
EXECUTION_FIELDS = {
    "slurm",
    "container_image",
    "submitted_saved_wrapper",
    "runtime_dollar_zero_snapshot",
    "scontrol_batch_script_snapshot",
}
SLURM_FIELDS = {
    "profile",
    "snapshot",
    "job_id",
    "node_list",
    "node_name",
    "account",
    "partition",
    "job_name",
    "num_nodes",
    "num_cpus",
    "num_tasks",
    "requested_tres",
    "allocated_tres",
    "tres_per_node",
    "gpus_on_node",
    "cpus_per_task",
    "stdout_path_raw",
    "stdout_path",
    "command_raw",
    "command",
    "work_dir_raw",
    "work_dir",
    "batch_host",
    "job_state",
}
SLURM_SNAPSHOT_PROFILE = "immutable_scontrol_show_job_oneliner_v1"
SUBMODULE_STATUS_PROFILE = "exact_pinned_uninitialized_gitlinks_v1"
REPOSITORY_LAYOUT_PROFILE = "standalone_detached_clone_inroot_git_directory_v1"
REPOSITORY_FIELDS = {
    "path",
    "top_level",
    "commit",
    "clean",
    "head_state",
    "layout_profile",
    "git_dir",
    "git_common_dir",
    "submodule_status",
    "submodules",
}
EXPECTED_UNINITIALIZED_SUBMODULE_GITLINKS = (
    (
        "src/diff-surfel-rasterization/third_party/glm",
        "5c46b9c07008ae65cb81ab79cd677ecc1934b903",
    ),
    (
        "third_party/openpi",
        "bd70b8f4011e85b3f3b0f039f12113f78718e7bf",
    ),
)
TRES_ASSIGNMENT_FIELDS = {"raw", "entries"}
TRES_ASSIGNMENT_ENTRY_FIELDS = {"name", "kind", "value"}
TRES_PER_NODE_FIELDS = {"raw", "entries"}
TRES_PER_NODE_ENTRY_FIELDS = {"resource", "type", "count"}


class GripperImpulseFinalizationError(ValueError):
    """The post-Kit evidence set is incomplete, mutable, or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GripperImpulseFinalizationError(message)


def _normalized_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _canonical_target(path: Path) -> Path:
    return _normalized_absolute(path).resolve(strict=False)


def _validate_sha256(value: Any, *, field: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{field} SHA-256",
    )
    return value


def _validate_hex(value: Any, *, field: str, length: int) -> str:
    _require(
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value),
        f"{field} hexadecimal identity",
    )
    return value


def _secure_file(
    path: Path,
    *,
    field: str,
    required_mode: int | None = None,
    read_bytes: bool = False,
) -> tuple[dict[str, Any], bytes | None]:
    """Hash one stable regular single-link file without following its final link."""

    path = _normalized_absolute(path)
    try:
        before = os.lstat(path)
    except FileNotFoundError as error:
        raise GripperImpulseFinalizationError(f"missing {field}: {path}") from error
    _require(stat.S_ISREG(before.st_mode), f"{field} is not a regular file")
    _require(before.st_nlink == 1, f"{field} link count is not one")
    if required_mode is not None:
        _require(
            stat.S_IMODE(before.st_mode) == required_mode,
            f"{field} mode is not {required_mode:04o}",
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise GripperImpulseFinalizationError(
            f"cannot securely open {field}: {path}: {error}"
        ) from error
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if read_bytes else None
    try:
        opened = os.fstat(descriptor)
        _require(
            stat.S_ISREG(opened.st_mode)
            and (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
            f"{field} changed during secure open",
        )
        _require(
            (
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
                opened.st_nlink,
                stat.S_IMODE(opened.st_mode),
            )
            == (
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
                before.st_nlink,
                stat.S_IMODE(before.st_mode),
            ),
            f"{field} metadata changed during secure open",
        )
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(descriptor)
        _require(
            (
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
                after.st_nlink,
                stat.S_IMODE(after.st_mode),
            )
            == (
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
                opened.st_nlink,
                stat.S_IMODE(opened.st_mode),
            ),
            f"{field} changed during read",
        )
        linked = os.lstat(path)
        _require(
            stat.S_ISREG(linked.st_mode)
            and (linked.st_dev, linked.st_ino) == (opened.st_dev, opened.st_ino)
            and linked.st_nlink == 1,
            f"{field} path changed during read",
        )
        _require(
            (
                linked.st_size,
                linked.st_mtime_ns,
                linked.st_ctime_ns,
                stat.S_IMODE(linked.st_mode),
            )
            == (
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
                stat.S_IMODE(after.st_mode),
            ),
            f"{field} path metadata changed during read",
        )
    finally:
        os.close(descriptor)
    data = b"".join(chunks) if chunks is not None else None
    if data is not None:
        _require(len(data) == before.st_size, f"{field} read size drift")
    return (
        {
            "path": str(path),
            "size_bytes": after.st_size,
            "sha256": digest.hexdigest(),
            "mode": f"{stat.S_IMODE(after.st_mode):04o}",
            "nlink": 1,
        },
        data,
    )


def _secure_read_immutable(path: Path, *, field: str) -> tuple[dict[str, Any], bytes]:
    identity, data = _secure_file(
        path, field=field, required_mode=0o444, read_bytes=True
    )
    _require(data is not None, f"{field} was not read")
    return identity, data


def _require_zero_status(path: Path, *, field: str) -> dict[str, Any]:
    identity, data = _secure_read_immutable(path, field=field)
    _require(data == b"0\n", f"{field} must contain exact bytes b'0\\n'")
    return identity


def _with_publication_times(
    identity: Mapping[str, Any], path: Path, *, field: str
) -> dict[str, Any]:
    metadata = os.lstat(_normalized_absolute(path))
    _require(
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_nlink == 1
        and metadata.st_size == identity["size_bytes"]
        and f"{stat.S_IMODE(metadata.st_mode):04o}" == identity["mode"],
        f"{field} publication metadata drift",
    )
    return {
        **identity,
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
        "publication_time_ns": max(metadata.st_mtime_ns, metadata.st_ctime_ns),
    }


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.rstrip("\n")


def _require_diagnostic() -> Any:
    _require(diagnostic is not None, "diagnostic module was not securely bootstrapped")
    return diagnostic


def _validate_standalone_repository_layout(repo: Path) -> dict[str, str]:
    """Reject linked worktrees and any Git metadata outside the mounted repo."""

    repo = _canonical_target(repo)
    git_entry = repo / ".git"
    try:
        git_metadata = os.lstat(git_entry)
    except FileNotFoundError as error:
        raise GripperImpulseFinalizationError(
            f"PolaRiS standalone clone is missing in-root .git directory: {git_entry}"
        ) from error
    _require(
        stat.S_ISDIR(git_metadata.st_mode),
        "PolaRiS .git must be a real in-root directory, not a gitdir file or symlink",
    )
    expected_git_dir = _normalized_absolute(git_entry)
    _require(
        _canonical_target(git_entry) == expected_git_dir,
        "PolaRiS .git directory resolves outside the repository",
    )

    absolute_git_dir_text = _git(repo, "rev-parse", "--absolute-git-dir")
    common_git_dir_text = _git(repo, "rev-parse", "--git-common-dir")
    _require(
        bool(absolute_git_dir_text) and bool(common_git_dir_text),
        "PolaRiS Git directory query returned an empty path",
    )
    absolute_git_dir_path = Path(absolute_git_dir_text)
    _require(
        absolute_git_dir_path.is_absolute(),
        "PolaRiS absolute Git directory query returned a relative path",
    )
    absolute_git_dir = _canonical_target(absolute_git_dir_path)
    common_git_dir_path = Path(common_git_dir_text)
    if not common_git_dir_path.is_absolute():
        common_git_dir_path = repo / common_git_dir_path
    common_git_dir = _canonical_target(common_git_dir_path)
    _require(
        absolute_git_dir == expected_git_dir,
        "PolaRiS Git directory is not the real in-root .git directory",
    )
    _require(
        common_git_dir == expected_git_dir,
        "PolaRiS common Git directory is external to the standalone clone",
    )
    return {
        "layout_profile": REPOSITORY_LAYOUT_PROFILE,
        "git_dir": str(absolute_git_dir),
        "git_common_dir": str(common_git_dir),
    }


def _validate_submodule_status(value: Any) -> dict[str, Any]:
    _require(type(value) is str, "Git submodule status must be text")
    lines = value.splitlines()
    _require(
        len(lines) == len(EXPECTED_UNINITIALIZED_SUBMODULE_GITLINKS),
        "Git submodule status entry count drift",
    )
    entries: list[dict[str, str]] = []
    for index, (line, (expected_path, expected_commit)) in enumerate(
        zip(lines, EXPECTED_UNINITIALIZED_SUBMODULE_GITLINKS, strict=True)
    ):
        match = re.fullmatch(r"-([0-9a-f]{40}) ([^\s()]+)", line)
        _require(match is not None, f"Git submodule status line {index} malformed")
        commit, path = match.groups()
        _require(
            path == expected_path and commit == expected_commit,
            f"Git submodule status line {index} gitlink/path drift",
        )
        entries.append(
            {
                "path": path,
                "gitlink_commit": commit,
                "state": "uninitialized",
            }
        )
    return {
        "profile": SUBMODULE_STATUS_PROFILE,
        "entries": entries,
    }


def _validate_repository(args: argparse.Namespace) -> dict[str, Any]:
    repo = _canonical_target(args.polaris_repo)
    _validate_hex(
        args.expected_polaris_commit,
        field="expected PolaRiS commit",
        length=40,
    )
    layout = _validate_standalone_repository_layout(repo)
    top_level = _canonical_target(Path(_git(repo, "rev-parse", "--show-toplevel")))
    _require(top_level == repo, "PolaRiS repo is not the exact Git top level")
    commit = _git(repo, "rev-parse", "HEAD")
    _require(commit == args.expected_polaris_commit, "PolaRiS commit mismatch")
    head_name = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    _require(head_name == "HEAD", "PolaRiS standalone clone HEAD is not detached")
    status = _git(
        repo,
        "-c",
        "status.showUntrackedFiles=all",
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    _require(status == "", "PolaRiS repo is dirty")
    submodule_status = _git(repo, "submodule", "status", "--recursive")
    submodules = _validate_submodule_status(submodule_status)
    result = {
        "path": str(repo),
        "top_level": str(top_level),
        "commit": commit,
        "clean": True,
        "head_state": "detached",
        **layout,
        "submodule_status": submodule_status,
        "submodules": submodules,
    }
    _require(set(result) == REPOSITORY_FIELDS, "PolaRiS repository schema")
    return result


def _bootstrap_trusted_sources(args: argparse.Namespace) -> Any:
    """Hash all Python inputs and the clean repo before importing diagnostics."""

    repository = _validate_repository(args)
    repo = Path(repository["path"])
    _require(
        (repo / "scripts" / DIAGNOSTIC_PATH.name).resolve() == DIAGNOSTIC_PATH
        and (repo / "scripts" / SCRIPT_PATH.name).resolve() == SCRIPT_PATH
        and (repo / "scripts" / BOUNDARY_PATH.name).resolve() == BOUNDARY_PATH
        and (repo / "scripts" / "fixtures" / FIXTURE_PATH.name).resolve()
        == FIXTURE_PATH,
        "diagnostic source paths are not from the exact PolaRiS repo",
    )
    source_specs = (
        ("finalizer bootstrap", SCRIPT_PATH, args.expected_finalizer_sha256),
        ("diagnostic bootstrap", DIAGNOSTIC_PATH, args.expected_diagnostic_sha256),
        ("boundary bootstrap", BOUNDARY_PATH, args.expected_boundary_sha256),
        ("fixture bootstrap", FIXTURE_PATH, args.expected_fixture_sha256),
    )
    identities: dict[str, dict[str, Any]] = {}
    source_bytes: dict[str, bytes] = {}
    for field, path, expected in source_specs:
        _validate_sha256(expected, field=field)
        identity, data = _secure_file(path, field=field, read_bytes=True)
        _require(identity["sha256"] == expected, f"{field} digest mismatch")
        _require(data is not None, f"{field} bootstrap bytes")
        identities[field] = identity
        source_bytes[field] = data
    _require(
        identities["boundary bootstrap"]["sha256"] == EXPECTED_BOUNDARY_SHA256,
        "bootstrap boundary closed digest mismatch",
    )
    _require(
        identities["fixture bootstrap"]["sha256"] == EXPECTED_FIXTURE_SHA256,
        "bootstrap fixture closed digest mismatch",
    )
    spec = importlib.util.spec_from_file_location(
        "polaris_gripper_impulse_diagnostic_host", DIAGNOSTIC_PATH
    )
    _require(spec is not None and spec.loader is not None, "cannot load diagnostic")
    module = importlib.util.module_from_spec(spec)
    exec(  # noqa: S102 - execute only the securely read, exact-hash source bytes.
        compile(
            source_bytes["diagnostic bootstrap"],
            str(DIAGNOSTIC_PATH),
            "exec",
            dont_inherit=True,
        ),
        module.__dict__,
    )
    _require(
        Path(getattr(module, "__file__", "")).resolve() == DIAGNOSTIC_PATH
        and module.BOUNDARY_HELPER_PATH.resolve() == BOUNDARY_PATH
        and module.boundary.FIXTURE_PATH.resolve() == FIXTURE_PATH,
        "securely imported diagnostic path drift",
    )
    return module


def _parse_scontrol_oneliner(data: bytes) -> dict[str, str]:
    """Strictly parse one immutable ``scontrol show job --oneliner`` record."""

    _require(
        data.endswith(b"\n") and data.count(b"\n") == 1 and b"\r" not in data,
        "Slurm job snapshot must contain exactly one LF-terminated record",
    )
    try:
        text = data[:-1].decode("utf-8").rstrip(" ")
    except UnicodeDecodeError as error:
        raise GripperImpulseFinalizationError(
            "Slurm job snapshot is not strict UTF-8"
        ) from error
    _require(bool(text) and "\t" not in text, "empty or tabbed Slurm job snapshot")
    pattern = re.compile(
        r"(?:^| )([A-Za-z][A-Za-z0-9_/:]*)=(.*?)(?= [A-Za-z][A-Za-z0-9_/:]*=|$)"
    )
    matches = list(pattern.finditer(text))
    _require(
        bool(matches) and "".join(match.group(0) for match in matches) == text,
        "unparseable Slurm job snapshot",
    )
    record: dict[str, str] = {}
    for match in matches:
        key, value = match.group(1), match.group(2)
        _require(key not in record, f"duplicate Slurm job snapshot field {key}")
        record[key] = value
    return record


def _canonical_tres_assignments(value: Any, *, field: str) -> dict[str, Any]:
    """Canonicalize ReqTRES/AllocTRES while preserving the scheduler bytes."""

    _require(type(value) is str and bool(value), f"{field} must be a nonempty string")
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for token in value.split(","):
        _require(token.count("=") == 1, f"{field} assignment syntax")
        name, raw_value = token.split("=", 1)
        _require(
            bool(re.fullmatch(r"[A-Za-z0-9_./:-]+", name))
            and bool(raw_value)
            and not any(character.isspace() for character in raw_value),
            f"{field} assignment value",
        )
        _require(name not in seen, f"duplicate {field} assignment {name}")
        seen.add(name)
        if raw_value.isdecimal():
            _require(
                str(int(raw_value)) == raw_value,
                f"{field} integer assignment is not canonical",
            )
            kind = "integer"
            canonical_value: int | str = int(raw_value)
        else:
            memory_match = re.fullmatch(r"([1-9][0-9]*)([KMGTPE])", raw_value)
            if memory_match is not None:
                exponent = "KMGTPE".index(memory_match.group(2)) + 1
                kind = "bytes_binary"
                canonical_value = int(memory_match.group(1)) * (1024**exponent)
            else:
                kind = "opaque"
                canonical_value = raw_value
        entry = {"name": name, "kind": kind, "value": canonical_value}
        _require(
            set(entry) == TRES_ASSIGNMENT_ENTRY_FIELDS,
            f"{field} assignment schema",
        )
        entries.append(entry)
    entries.sort(key=lambda entry: entry["name"])
    result = {"raw": value, "entries": entries}
    _require(set(result) == TRES_ASSIGNMENT_FIELDS, f"{field} schema")
    return result


def _canonical_tres_per_node(value: Any, *, field: str) -> dict[str, Any]:
    """Canonicalize generic and typed per-node GRES forms."""

    _require(type(value) is str and bool(value), f"{field} must be a nonempty string")
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None]] = set()
    for token in value.split(","):
        parts = token.split(":")
        _require(len(parts) in {2, 3}, f"{field} token syntax")
        resource = parts[0]
        resource_type = None if len(parts) == 2 else parts[1]
        count_text = parts[-1]
        _require(
            bool(re.fullmatch(r"[A-Za-z0-9_./-]+", resource))
            and (
                resource_type is None
                or bool(re.fullmatch(r"[A-Za-z0-9_.+-]+", resource_type))
            )
            and bool(re.fullmatch(r"[0-9]+", count_text)),
            f"{field} token value",
        )
        count = int(count_text)
        _require(count > 0 and str(count) == count_text, f"{field} token count")
        key = (resource, resource_type)
        _require(key not in seen, f"duplicate {field} token {token}")
        seen.add(key)
        entry = {"resource": resource, "type": resource_type, "count": count}
        _require(set(entry) == TRES_PER_NODE_ENTRY_FIELDS, f"{field} entry schema")
        entries.append(entry)
    entries.sort(key=lambda entry: (entry["resource"], entry["type"] or ""))
    result = {"raw": value, "entries": entries}
    _require(set(result) == TRES_PER_NODE_FIELDS, f"{field} schema")
    return result


def _require_exact_positive_int(value: Any, *, field: str) -> int:
    _require(type(value) is int and value > 0, field)
    return value


def _exact_env(name: str, expected: str) -> None:
    _require(os.environ.get(name) == expected, f"live {name} mismatch")


def _capture_slurm_provenance(args: argparse.Namespace) -> dict[str, Any]:
    _require_exact_positive_int(
        args.expected_slurm_job_id, field="expected Slurm job id"
    )
    for field in (
        "expected_slurm_num_nodes",
        "expected_slurm_num_cpus",
        "expected_slurm_num_tasks",
        "expected_slurm_cpus_per_task",
    ):
        _require_exact_positive_int(getattr(args, field), field=field)
    for field in (
        "expected_slurm_node_list",
        "expected_slurm_node_name",
        "expected_slurm_account",
        "expected_slurm_partition",
        "expected_slurm_job_name",
        "expected_slurm_req_tres",
        "expected_slurm_alloc_tres",
        "expected_slurm_tres_per_node",
        "expected_slurm_gpus_on_node",
    ):
        _require(
            type(getattr(args, field)) is str and bool(getattr(args, field)),
            field,
        )
    snapshot_identity, snapshot_bytes = _secure_read_immutable(
        args.slurm_job_oneliner_snapshot,
        field="Slurm show-job oneliner snapshot",
    )
    snapshot_identity = _with_publication_times(
        snapshot_identity,
        args.slurm_job_oneliner_snapshot,
        field="Slurm show-job oneliner snapshot",
    )
    record = _parse_scontrol_oneliner(snapshot_bytes)

    requested_tres = _canonical_tres_assignments(
        record.get("ReqTRES"), field="Slurm ReqTRES"
    )
    expected_requested_tres = _canonical_tres_assignments(
        args.expected_slurm_req_tres, field="expected Slurm ReqTRES"
    )
    allocated_tres = _canonical_tres_assignments(
        record.get("AllocTRES"), field="Slurm AllocTRES"
    )
    expected_allocated_tres = _canonical_tres_assignments(
        args.expected_slurm_alloc_tres, field="expected Slurm AllocTRES"
    )
    tres_per_node = _canonical_tres_per_node(
        record.get("TresPerNode"), field="Slurm TresPerNode"
    )
    expected_tres_per_node = _canonical_tres_per_node(
        args.expected_slurm_tres_per_node,
        field="expected Slurm TresPerNode",
    )
    _require(
        requested_tres["entries"] == expected_requested_tres["entries"],
        "Slurm ReqTRES mismatch",
    )
    _require(
        allocated_tres["entries"] == expected_allocated_tres["entries"],
        "Slurm AllocTRES mismatch",
    )
    _require(
        tres_per_node["entries"] == expected_tres_per_node["entries"],
        "Slurm TresPerNode mismatch",
    )
    _require(
        args.expected_slurm_gpus_on_node.isdecimal()
        and str(int(args.expected_slurm_gpus_on_node))
        == args.expected_slurm_gpus_on_node
        and int(args.expected_slurm_gpus_on_node) > 0,
        "expected Slurm GPUs-on-node integer",
    )
    expected_tres_scalars = {
        "cpu": args.expected_slurm_num_cpus,
        "node": args.expected_slurm_num_nodes,
        "gres/gpu": int(args.expected_slurm_gpus_on_node),
    }
    for label, tres in (
        ("requested", requested_tres),
        ("allocated", allocated_tres),
    ):
        by_name = {entry["name"]: entry for entry in tres["entries"]}
        for name, wanted in expected_tres_scalars.items():
            _require(
                by_name.get(name) == {"name": name, "kind": "integer", "value": wanted},
                f"Slurm {label} TRES {name} scalar mismatch",
            )
        _require(
            by_name.get("mem", {}).get("kind") == "bytes_binary"
            and type(by_name["mem"].get("value")) is int
            and by_name["mem"]["value"] > 0,
            f"Slurm {label} TRES memory canonicalization",
        )
    _require(
        sum(
            entry["count"]
            for entry in tres_per_node["entries"]
            if entry["resource"] == "gres/gpu"
        )
        == int(args.expected_slurm_gpus_on_node),
        "Slurm TresPerNode GPU count mismatch",
    )

    expected_stdout = str(_canonical_target(args.expected_slurm_output))
    expected_command = str(_canonical_target(Path(args.expected_slurm_command)))
    expected_work_dir = str(_canonical_target(Path(args.expected_slurm_work_dir)))
    _require(
        expected_command == str(_canonical_target(args.submitted_saved_wrapper)),
        "expected Slurm command is not the saved wrapper",
    )
    expected = {
        "JobId": str(args.expected_slurm_job_id),
        "NodeList": args.expected_slurm_node_list,
        "Account": args.expected_slurm_account,
        "Partition": args.expected_slurm_partition,
        "JobName": args.expected_slurm_job_name,
        "NumNodes": str(args.expected_slurm_num_nodes),
        "NumCPUs": str(args.expected_slurm_num_cpus),
        "NumTasks": str(args.expected_slurm_num_tasks),
        "CPUs/Task": str(args.expected_slurm_cpus_per_task),
        "BatchHost": args.expected_slurm_node_name,
        "JobState": "RUNNING",
        "BatchFlag": "1",
    }
    for field, wanted in expected.items():
        _require(record.get(field) == wanted, f"Slurm snapshot {field} mismatch")

    raw_stdout = record.get("StdOut")
    raw_command = record.get("Command")
    raw_work_dir = record.get("WorkDir")
    _require(
        all(
            type(value) is str and bool(value)
            for value in (raw_stdout, raw_command, raw_work_dir)
        ),
        "Slurm snapshot path fields",
    )
    canonical_stdout = str(_canonical_target(Path(raw_stdout)))
    canonical_command = str(_canonical_target(Path(raw_command)))
    canonical_work_dir = str(_canonical_target(Path(raw_work_dir)))
    _require(canonical_stdout == expected_stdout, "Slurm snapshot StdOut mismatch")
    _require(canonical_command == expected_command, "Slurm snapshot Command mismatch")
    _require(canonical_work_dir == expected_work_dir, "Slurm snapshot WorkDir mismatch")

    env_expected = {
        "SLURM_JOB_ID": str(args.expected_slurm_job_id),
        "SLURM_JOB_NODELIST": args.expected_slurm_node_list,
        "SLURMD_NODENAME": args.expected_slurm_node_name,
        "SLURM_JOB_PARTITION": args.expected_slurm_partition,
        "SLURM_NTASKS": str(args.expected_slurm_num_tasks),
        "SLURM_GPUS_ON_NODE": args.expected_slurm_gpus_on_node,
        "SLURM_CPUS_PER_TASK": str(args.expected_slurm_cpus_per_task),
    }
    for name, wanted in env_expected.items():
        _exact_env(name, wanted)
    result = {
        "profile": SLURM_SNAPSHOT_PROFILE,
        "snapshot": snapshot_identity,
        "job_id": args.expected_slurm_job_id,
        "node_list": args.expected_slurm_node_list,
        "node_name": args.expected_slurm_node_name,
        "account": args.expected_slurm_account,
        "partition": args.expected_slurm_partition,
        "job_name": args.expected_slurm_job_name,
        "num_nodes": args.expected_slurm_num_nodes,
        "num_cpus": args.expected_slurm_num_cpus,
        "num_tasks": args.expected_slurm_num_tasks,
        "requested_tres": requested_tres,
        "allocated_tres": allocated_tres,
        "tres_per_node": tres_per_node,
        "gpus_on_node": args.expected_slurm_gpus_on_node,
        "cpus_per_task": args.expected_slurm_cpus_per_task,
        "stdout_path_raw": raw_stdout,
        "stdout_path": canonical_stdout,
        "command_raw": raw_command,
        "command": canonical_command,
        "work_dir_raw": raw_work_dir,
        "work_dir": canonical_work_dir,
        "batch_host": args.expected_slurm_node_name,
        "job_state": "RUNNING",
    }
    _require(set(result) == SLURM_FIELDS, "Slurm provenance schema")
    return result


def _capture_source_provenance(
    args: argparse.Namespace, capture: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    module = _require_diagnostic()
    _require(
        Path(getattr(module, "__file__", "")).resolve() == DIAGNOSTIC_PATH,
        "imported diagnostic __file__ mismatch",
    )
    repository = _validate_repository(args)
    repo = Path(repository["path"])
    _require(
        (repo / "scripts" / DIAGNOSTIC_PATH.name).resolve() == DIAGNOSTIC_PATH,
        "diagnostic is not imported from the exact PolaRiS repo",
    )
    _require(
        (repo / "scripts" / SCRIPT_PATH.name).resolve() == SCRIPT_PATH,
        "finalizer is not imported from the exact PolaRiS repo",
    )
    source_specs = (
        (
            "diagnostic",
            DIAGNOSTIC_PATH,
            args.expected_diagnostic_sha256,
            capture["diagnostic_source"]["actual"],
        ),
        (
            "finalizer",
            SCRIPT_PATH,
            args.expected_finalizer_sha256,
            None,
        ),
        (
            "boundary_helper",
            module.BOUNDARY_HELPER_PATH,
            args.expected_boundary_sha256,
            capture["boundary_helper_source"],
        ),
        (
            "replay_fixture",
            module.boundary.FIXTURE_PATH,
            args.expected_fixture_sha256,
            capture["fixture"],
        ),
    )
    sources: dict[str, Any] = {"imported_diagnostic_file": str(DIAGNOSTIC_PATH)}
    for field, path, expected_sha256, recorded in source_specs:
        _validate_sha256(expected_sha256, field=f"expected {field}")
        identity, _ = _secure_file(path, field=field)
        _require(identity["sha256"] == expected_sha256, f"{field} digest mismatch")
        if recorded is not None:
            common_fields = set(identity) & set(recorded)
            _require(
                common_fields >= {"path", "size_bytes", "sha256", "mode"}
                and all(
                    module._typed_equal(  # noqa: SLF001
                        identity[identity_field], recorded[identity_field]
                    )
                    for identity_field in common_fields
                ),
                f"{field} recorded/live identity mismatch",
            )
        sources[field] = identity
    _require(
        sources["boundary_helper"]["sha256"] == module.EXPECTED_BOUNDARY_HELPER_SHA256,
        "boundary helper closed digest mismatch",
    )
    _require(
        sources["replay_fixture"]["sha256"] == module.boundary.EXPECTED_FIXTURE_SHA256,
        "replay fixture closed digest mismatch",
    )
    _require(set(sources) == SOURCE_FIELDS, "source provenance schema")
    return sources, repository


def _capture_execution_provenance(args: argparse.Namespace) -> dict[str, Any]:
    for field in (
        "expected_container_image_sha256",
        "expected_submitted_saved_wrapper_sha256",
        "expected_runtime_dollar_zero_sha256",
        "expected_scontrol_batch_script_sha256",
    ):
        _validate_sha256(getattr(args, field), field=field)
    container, _ = _secure_file(args.container_image, field="container image")
    submitted_saved_wrapper, submitted_bytes = _secure_file(
        args.submitted_saved_wrapper,
        field="submitted saved wrapper",
        required_mode=0o444,
        read_bytes=True,
    )
    runtime_dollar_zero, runtime_bytes = _secure_file(
        args.runtime_dollar_zero_snapshot,
        field="runtime dollar-zero wrapper snapshot",
        required_mode=0o444,
        read_bytes=True,
    )
    scontrol_batch_script, scontrol_bytes = _secure_file(
        args.scontrol_batch_script_snapshot,
        field="scontrol batch-script snapshot",
        required_mode=0o444,
        read_bytes=True,
    )
    _require(
        container["sha256"] == args.expected_container_image_sha256,
        "container image digest mismatch",
    )
    _require(
        submitted_saved_wrapper["sha256"]
        == args.expected_submitted_saved_wrapper_sha256,
        "submitted saved-wrapper digest mismatch",
    )
    _require(
        runtime_dollar_zero["sha256"] == args.expected_runtime_dollar_zero_sha256,
        "runtime dollar-zero wrapper digest mismatch",
    )
    _require(
        scontrol_batch_script["sha256"] == args.expected_scontrol_batch_script_sha256,
        "scontrol batch-script snapshot digest mismatch",
    )
    _require(
        submitted_bytes is not None
        and runtime_bytes is not None
        and scontrol_bytes is not None
        and submitted_bytes == runtime_bytes == scontrol_bytes,
        "submitted/runtime-dollar-zero/scontrol wrapper bytes mismatch",
    )
    container_path_literal = os.fsencode(
        os.fspath(_normalized_absolute(args.container_image))
    )
    _require(
        b"srun" in submitted_bytes and container_path_literal in submitted_bytes,
        "saved wrapper lacks srun and the pinned container path literal",
    )
    _require(
        submitted_saved_wrapper["sha256"]
        == runtime_dollar_zero["sha256"]
        == scontrol_batch_script["sha256"]
        and submitted_saved_wrapper["size_bytes"]
        == runtime_dollar_zero["size_bytes"]
        == scontrol_batch_script["size_bytes"],
        "wrapper identity relation mismatch",
    )
    execution = {
        "slurm": _capture_slurm_provenance(args),
        "container_image": container,
        "submitted_saved_wrapper": submitted_saved_wrapper,
        "runtime_dollar_zero_snapshot": runtime_dollar_zero,
        "scontrol_batch_script_snapshot": scontrol_batch_script,
    }
    _require(set(execution) == EXECUTION_FIELDS, "execution provenance schema")
    return execution


def _require_secure_recorded_file(
    recorded: Mapping[str, Any],
    *,
    field: str,
    read_bytes: bool = False,
) -> tuple[dict[str, Any], bytes | None]:
    _require(isinstance(recorded, Mapping), f"{field} recorded identity")
    actual, data = _secure_file(
        Path(recorded.get("path", "")), field=field, read_bytes=read_bytes
    )
    common = set(actual) & set(recorded)
    _require(
        common >= {"path", "size_bytes", "sha256", "mode"}
        and all(
            finalizer_value == recorded[identity_field]
            and type(finalizer_value) is type(recorded[identity_field])
            for identity_field in common
            for finalizer_value in (actual[identity_field],)
        ),
        f"{field} recorded/live secure identity mismatch",
    )
    return actual, data


def _secure_validate_live_assets(capture: Mapping[str, Any]) -> None:
    """Secure-open every scene, IC, USD, and Hub revision record in the capture."""

    assets = capture["assets"]
    food = assets["foodbussing"]
    scene, _ = _require_secure_recorded_file(food["scene"], field="FoodBussing scene")
    conditions, _ = _require_secure_recorded_file(
        food["initial_conditions"], field="FoodBussing initial conditions"
    )
    scene_path = Path(scene["path"])
    conditions_path = Path(conditions["path"])
    _require(
        scene_path.name == "scene.usda"
        and conditions_path.name == "initial_conditions.json"
        and scene_path.parent == conditions_path.parent
        and scene_path.parent.name == "food_bussing",
        "FoodBussing secure asset paths",
    )
    metadata_root = (
        scene_path.parent.parent
        / ".cache"
        / "huggingface"
        / "download"
        / "food_bussing"
    )
    for filename in ("initial_conditions.json", "scene.usda"):
        recorded = food["revision_metadata"][filename]
        actual, data = _require_secure_recorded_file(
            recorded,
            field=f"FoodBussing metadata {filename}",
            read_bytes=True,
        )
        _require(
            Path(actual["path"]) == metadata_root / f"{filename}.metadata",
            f"FoodBussing metadata {filename} path",
        )
        _require(data is not None, f"FoodBussing metadata {filename} bytes")
        lines = data.decode("utf-8").splitlines()
        _require(
            bool(lines)
            and lines[0].strip() == food["polaris_hub_revision"]
            and recorded["revision"] == food["polaris_hub_revision"],
            f"FoodBussing metadata {filename} revision",
        )

    robot, _ = _require_secure_recorded_file(assets["robot_usd"], field="robot USD")
    robot_path = Path(robot["path"])
    _require(
        robot_path.name == "noninstanceable.usd"
        and robot_path.parent.name == "nvidia_droid",
        "robot USD secure path",
    )
    robot_metadata = assets["robot_usd_revision_metadata"]
    metadata, metadata_bytes = _require_secure_recorded_file(
        robot_metadata["identity"],
        field="robot USD metadata",
        read_bytes=True,
    )
    expected_metadata_path = (
        robot_path.parent.parent
        / ".cache"
        / "huggingface"
        / "download"
        / "nvidia_droid"
        / "noninstanceable.usd.metadata"
    )
    _require(
        Path(metadata["path"]) == expected_metadata_path,
        "robot USD metadata secure path",
    )
    _require(metadata_bytes is not None, "robot USD metadata bytes")
    metadata_lines = metadata_bytes.decode("utf-8").splitlines()
    _require(
        len(metadata_lines) >= 2
        and metadata_lines[0].strip() == robot_metadata["revision"]
        and metadata_lines[1].strip() == robot_metadata["recorded_sha256"]
        and robot_metadata["recorded_sha256"] == robot["sha256"],
        "robot USD metadata content",
    )


def _validate_capture_with_stdlib_probe(
    module: Any, args: argparse.Namespace
) -> dict[str, Any]:
    """Force every host-finalizer pass through the stdlib-only ffprobe path."""

    probe = getattr(module, "_probe_video_stdlib", None)
    _require(callable(probe), "missing diagnostic stdlib video probe")
    return module.validate_capture_artifacts(
        args.validate_capture,
        args.video,
        expected_mode=args.expected_mode,
        probe=probe,
    )


def _validate_common(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    module = _require_diagnostic()
    artifact_paths = [
        _normalized_absolute(args.validate_capture),
        _normalized_absolute(args.video),
        _normalized_absolute(args.ready_marker),
        _normalized_absolute(args.runtime_exit),
        _normalized_absolute(args.outer_srun_status),
    ]
    _require(
        len(set(artifact_paths)) == len(artifact_paths),
        "attempt artifact paths collide",
    )
    capture_identity, _ = _secure_read_immutable(
        args.validate_capture, field="raw capture"
    )
    video_identity, _ = _secure_file(args.video, field="video", required_mode=0o444)
    ready_identity, ready_bytes = _secure_read_immutable(
        args.ready_marker, field="ready marker"
    )
    runtime_exit_identity = _require_zero_status(
        args.runtime_exit, field="parent-reconciled runtime exit status"
    )
    outer_srun_identity = _require_zero_status(
        args.outer_srun_status, field="outer srun status"
    )
    timed_identities = {
        "capture": _with_publication_times(
            capture_identity, args.validate_capture, field="raw capture"
        ),
        "video": _with_publication_times(video_identity, args.video, field="video"),
        "ready_marker": _with_publication_times(
            ready_identity, args.ready_marker, field="ready marker"
        ),
        "runtime_exit": _with_publication_times(
            runtime_exit_identity, args.runtime_exit, field="runtime exit"
        ),
        "outer_srun_status": _with_publication_times(
            outer_srun_identity, args.outer_srun_status, field="outer srun status"
        ),
    }
    _require(
        max(
            timed_identities["capture"]["publication_time_ns"],
            timed_identities["video"]["publication_time_ns"],
        )
        <= timed_identities["ready_marker"]["publication_time_ns"]
        <= timed_identities["runtime_exit"]["publication_time_ns"]
        <= timed_identities["outer_srun_status"]["publication_time_ns"],
        "attempt artifact publication order",
    )
    capture = _validate_capture_with_stdlib_probe(module, args)
    _secure_validate_live_assets(capture)
    capture_after, _ = _secure_read_immutable(
        args.validate_capture, field="raw capture"
    )
    video_after, _ = _secure_file(args.video, field="video", required_mode=0o444)
    _require(capture_after == capture_identity, "raw capture changed during validation")
    _require(video_after == video_identity, "video changed during validation")
    runtime_contract = module.validate_runtime_exit_contract(
        capture["runtime_exit_contract"]
    )
    _require(
        Path(runtime_contract["path"]) == _normalized_absolute(args.runtime_exit),
        "capture runtime-exit path mismatch",
    )
    ready = module.boundary.strict_json_loads(
        ready_bytes, field="gripper impulse ready marker"
    )
    module.validate_ready_marker(
        ready,
        mode=args.expected_mode,
        raw_identity=capture_identity,
        video_identity=capture["video"],
        diagnostic_source=capture["diagnostic_source"],
        runtime_exit_contract=runtime_contract,
    )
    sources, repository = _capture_source_provenance(args, capture)
    execution = _capture_execution_provenance(args)
    _require(
        execution["slurm"]["snapshot"]["publication_time_ns"]
        >= timed_identities["outer_srun_status"]["publication_time_ns"],
        "Slurm job snapshot predates the outer srun status",
    )
    ready_after, ready_bytes_after = _secure_read_immutable(
        args.ready_marker, field="ready marker"
    )
    runtime_exit_after = _require_zero_status(
        args.runtime_exit, field="parent-reconciled runtime exit status"
    )
    outer_srun_after = _require_zero_status(
        args.outer_srun_status, field="outer srun status"
    )
    _require(
        ready_after == ready_identity and ready_bytes_after == ready_bytes,
        "ready marker changed during validation",
    )
    _require(
        runtime_exit_after == runtime_exit_identity,
        "parent-reconciled runtime exit status changed during validation",
    )
    _require(
        outer_srun_after == outer_srun_identity,
        "outer srun status changed during validation",
    )
    context = {
        "identities": {
            **timed_identities,
        },
        "sources": sources,
        "repository": repository,
        "execution": execution,
    }
    return capture, context


def _attestation_payload(
    *,
    capture: Mapping[str, Any],
    context: Mapping[str, Any],
    validator_status: Mapping[str, Any],
    mode: str,
    intended_attestation_path: Path,
) -> dict[str, Any]:
    artifacts = {
        **context["identities"],
        "validator_status": validator_status,
    }
    _require(set(artifacts) == ARTIFACT_FIELDS, "attestation artifact schema")
    payload = {
        "schema_version": 1,
        "profile": ATTESTATION_PROFILE,
        "stage": "post_kit_host_attested",
        "mode": mode,
        "intended_attestation_path": str(_canonical_target(intended_attestation_path)),
        "outcome": capture["outcome"],
        "artifacts": artifacts,
        "runtime_identity": {
            "protocol": capture["runtime_protocol"],
            "frame": capture["runtime_frame"],
            "runtime_exit_contract": capture["runtime_exit_contract"],
        },
        "sources": context["sources"],
        "repository": context["repository"],
        "execution": context["execution"],
        "assets": capture["assets"],
    }
    _require(set(payload) == ATTESTATION_FIELDS, "attestation schema")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action", choices=("validate", "finalize", "verify"), required=True
    )
    parser.add_argument("--validate-capture", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--runtime-exit", type=Path, required=True)
    parser.add_argument("--outer-srun-status", type=Path, required=True)
    parser.add_argument("--validator-status", type=Path)
    parser.add_argument("--intended-attestation-path", type=Path, required=True)
    parser.add_argument("--attestation-staging-output", type=Path)
    parser.add_argument("--attestation-input", type=Path)
    parser.add_argument("--expected-mode", choices=MODES, required=True)
    parser.add_argument("--polaris-repo", type=Path, required=True)
    parser.add_argument("--expected-polaris-commit", required=True)
    parser.add_argument("--expected-diagnostic-sha256", required=True)
    parser.add_argument("--expected-finalizer-sha256", required=True)
    parser.add_argument("--expected-boundary-sha256", required=True)
    parser.add_argument("--expected-fixture-sha256", required=True)
    parser.add_argument("--container-image", type=Path, required=True)
    parser.add_argument("--expected-container-image-sha256", required=True)
    parser.add_argument("--submitted-saved-wrapper", type=Path, required=True)
    parser.add_argument("--expected-submitted-saved-wrapper-sha256", required=True)
    parser.add_argument("--runtime-dollar-zero-snapshot", type=Path, required=True)
    parser.add_argument("--expected-runtime-dollar-zero-sha256", required=True)
    parser.add_argument("--scontrol-batch-script-snapshot", type=Path, required=True)
    parser.add_argument("--expected-scontrol-batch-script-sha256", required=True)
    parser.add_argument("--slurm-job-oneliner-snapshot", type=Path, required=True)
    parser.add_argument("--expected-slurm-job-id", type=int, required=True)
    parser.add_argument("--expected-slurm-node-list", required=True)
    parser.add_argument("--expected-slurm-node-name", required=True)
    parser.add_argument("--expected-slurm-account", required=True)
    parser.add_argument("--expected-slurm-partition", required=True)
    parser.add_argument("--expected-slurm-job-name", required=True)
    parser.add_argument("--expected-slurm-num-nodes", type=int, required=True)
    parser.add_argument("--expected-slurm-num-cpus", type=int, required=True)
    parser.add_argument("--expected-slurm-num-tasks", type=int, required=True)
    parser.add_argument("--expected-slurm-req-tres", required=True)
    parser.add_argument("--expected-slurm-alloc-tres", required=True)
    parser.add_argument("--expected-slurm-tres-per-node", required=True)
    parser.add_argument("--expected-slurm-gpus-on-node", required=True)
    parser.add_argument("--expected-slurm-cpus-per-task", type=int, required=True)
    parser.add_argument("--expected-slurm-output", type=Path, required=True)
    parser.add_argument("--expected-slurm-command", type=Path, required=True)
    parser.add_argument("--expected-slurm-work-dir", type=Path, required=True)
    return parser


def build_source_preflight_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail closed unless PolaRiS is a clean standalone detached clone "
            "whose real .git directory is contained inside the mounted repository."
        )
    )
    parser.add_argument("--source-preflight", action="store_true", required=True)
    parser.add_argument("--polaris-repo", type=Path, required=True)
    parser.add_argument("--expected-polaris-commit", required=True)
    return parser


def _require_action_paths(args: argparse.Namespace) -> None:
    input_list = [
        _normalized_absolute(path)
        for path in (
            args.validate_capture,
            args.video,
            args.ready_marker,
            args.runtime_exit,
            args.outer_srun_status,
            args.container_image,
            args.submitted_saved_wrapper,
            args.runtime_dollar_zero_snapshot,
            args.scontrol_batch_script_snapshot,
            args.slurm_job_oneliner_snapshot,
        )
    ]
    canonical_inputs = {_canonical_target(path) for path in input_list}
    _require(len(canonical_inputs) == len(input_list), "finalizer input paths collide")
    intended = _canonical_target(args.intended_attestation_path)
    _require(
        intended not in canonical_inputs,
        "intended attestation collides with an input",
    )
    if args.action == "validate":
        _require(
            not os.path.lexists(intended),
            "intended attestation already exists before validation",
        )
        _require(
            args.validator_status is None, "validate may not consume validator status"
        )
        _require(
            args.attestation_staging_output is None,
            "validate may not write a staging attestation",
        )
        _require(args.attestation_input is None, "validate may not read attestation")
        return
    _require(
        args.validator_status is not None, f"{args.action} requires validator status"
    )
    validator = _canonical_target(args.validator_status)
    _require(
        validator not in canonical_inputs | {intended},
        "validator status path collision",
    )
    if args.action == "finalize":
        _require(
            not os.path.lexists(intended),
            "intended attestation already exists before finalization",
        )
        _require(
            args.attestation_staging_output is not None,
            "finalize requires staging output",
        )
        _require(
            args.attestation_input is None, "finalize may not read final attestation"
        )
        staging = _canonical_target(args.attestation_staging_output)
        _require(
            staging not in canonical_inputs | {intended, validator},
            "staging attestation path collision",
        )
        _require(
            staging.parent
            not in {path.parent for path in canonical_inputs} | {intended.parent},
            "staging attestation directory is not isolated",
        )
        _require(
            not os.path.lexists(staging), "refusing to overwrite staging attestation"
        )
        return
    _require(
        args.attestation_staging_output is None,
        "verify may not write a staging attestation",
    )
    _require(args.attestation_input is not None, "verify requires attestation input")
    _require(
        _canonical_target(args.attestation_input) == intended,
        "verify input is not the intended final attestation path",
    )


def main(argv: Sequence[str] | None = None) -> int:
    global diagnostic
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if "--source-preflight" in raw_argv:
        args = build_source_preflight_parser().parse_args(raw_argv)
        repository = _validate_repository(args)
        print(
            "POLARIS_GRIPPER_IMPULSE_SOURCE_PREFLIGHT="
            f"profile={repository['layout_profile']};"
            f"repo={repository['path']};git_dir={repository['git_dir']};"
            f"commit={repository['commit']}",
            flush=True,
        )
        return 0
    args = build_parser().parse_args(raw_argv)
    diagnostic = _bootstrap_trusted_sources(args)
    module = _require_diagnostic()
    _require_action_paths(args)
    capture, context = _validate_common(args)
    if args.action == "validate":
        print(
            "POLARIS_GRIPPER_IMPULSE_HOST_VALID="
            f"mode={args.expected_mode};outcome={capture['outcome']['kind']}",
            flush=True,
        )
        return 0
    validator_status = _require_zero_status(
        args.validator_status, field="validator status"
    )
    validator_status = _with_publication_times(
        validator_status, args.validator_status, field="validator status"
    )
    _require(
        validator_status["publication_time_ns"]
        >= context["execution"]["slurm"]["snapshot"]["publication_time_ns"],
        "validator status predates the Slurm job snapshot",
    )
    expected = _attestation_payload(
        capture=capture,
        context=context,
        validator_status=validator_status,
        mode=args.expected_mode,
        intended_attestation_path=args.intended_attestation_path,
    )
    expected_bytes = module._strict_json_bytes(expected)  # noqa: SLF001
    if args.action == "finalize":
        identity = module.publish_immutable_json(
            args.attestation_staging_output, expected
        )
        reread_identity, reread_bytes = _secure_read_immutable(
            args.attestation_staging_output, field="staging attestation"
        )
        _require(identity == reread_identity, "staging attestation identity drift")
        _require(reread_bytes == expected_bytes, "staging attestation byte drift")
        print(
            "POLARIS_GRIPPER_IMPULSE_ATTESTATION_STAGED="
            f"{identity['path']};sha256={identity['sha256']};"
            f"intended={_canonical_target(args.intended_attestation_path)}",
            flush=True,
        )
        return 0
    actual_identity, actual_bytes = _secure_read_immutable(
        args.attestation_input, field="final attestation"
    )
    _require(actual_bytes == expected_bytes, "attestation byte verification drift")
    actual = module.boundary.strict_json_loads(
        actual_bytes, field="gripper impulse attestation"
    )
    _require(
        module._typed_equal(actual, expected),  # noqa: SLF001
        "attestation typed verification drift",
    )
    print(
        "POLARIS_GRIPPER_IMPULSE_ATTESTATION_VERIFIED="
        f"{actual_identity['path']};sha256={actual_identity['sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
