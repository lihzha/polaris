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
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
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
APP_TERMINAL_PROMOTION_PROFILE = "polaris_app_launcher_allocation_promotion_v2"
APP_SACCT_QUERY_RECEIPT_PROFILE = "polaris_app_launcher_live_sacct_query_receipt_v1"
SACCT_PRELAUNCH_RECEIPT_PROFILE = (
    "polaris_external_sacct_runtime_prelaunch_validation_receipt_v2"
)
SCHEDULER_RUNNING_FILENAME = "pi05_droid_jointpos_scheduler_running.json"
SCHEDULER_TERMINAL_FILENAME = "pi05_droid_jointpos_scheduler_terminal.json"
APP_TERMINAL_PROMOTION_FILENAME = "app_launcher_allocation_promotion.json"
APP_SACCT_QUERY_RAW_FILENAME = "app_launcher_sacct.raw"
APP_SACCT_QUERY_RECEIPT_FILENAME = "app_launcher_sacct_query_receipt.json"
SACCT_PRELAUNCH_RECEIPT_FILENAME = "sacct_runtime_prelaunch_validation.json"
PINNED_SACCT_PATH = Path("/cm/local/apps/slurm/24.11/bin/sacct")
PINNED_SLURM_CONFIG_PATH = Path("/cm/shared/apps/slurm/etc/oci-ord-cs-004/slurm.conf")
PINNED_SLURM_LIBRARY_PATH = Path(
    "/cm/local/apps/slurm/24.11/lib64/slurm/libslurmfull.so"
)
SACCT_QUERY_PATH = "/usr/bin:/bin"
SACCT_QUERY_LD_LIBRARY_PATH = (
    "/cm/local/apps/slurm/24.11/lib64:/cm/local/apps/slurm/24.11/lib64/slurm"
)
SACCT_SUBPROCESS_TIMEOUT_SECONDS = 10.0
SACCT_RUNTIME_APPROVAL_PROFILE = "polaris_external_sacct_runtime_approval_v2"
SACCT_RUNTIME_CANDIDATE_PROFILE = "polaris_external_sacct_runtime_candidate_v2"
SACCT_RUNTIME_REVIEW_PROFILE = "polaris_external_sacct_runtime_review_v2"
SACCT_RUNTIME_CAPTURE_TERMINAL_PROFILE = (
    "polaris_external_sacct_runtime_capture_terminal_v1"
)
SACCT_RUNTIME_CAPTURE_PRODUCER_PROFILE = (
    "polaris_external_sacct_runtime_capture_producer_v1"
)
SACCT_RUNTIME_CAPTURE_QUERY_PROFILE = "polaris_external_sacct_runtime_capture_query_v1"
SACCT_RUNTIME_CAPTURE_CLOSURE_PROFILE = (
    "polaris_external_sacct_runtime_capture_closure_v1"
)
SACCT_RUNTIME_DEPENDENCY_CENSUS_PROFILE = "polaris_external_sacct_dependency_census_v1"
SACCT_RUNTIME_CAPTURE_PRODUCER_MODULE = "runtime_trace_v2.finalize_runtime_trace_v2"
SACCT_RUNTIME_REVIEWER_IDENTITY = {
    "principal": "UNPINNED_INDEPENDENT_AGENT_REVIEW_PENDING",
    "profile": "polaris_external_sacct_runtime_reviewer_identity_v2",
    "role": "independent_agent_runtime_approval_reviewer",
}
SACCT_RUNTIME_REVIEW_SCOPE = (
    "sacct_runtime_candidate_and_capture_terminal_non_self_approval_v1"
)
SACCT_RUNTIME_CAPTURE_SCOPE = "l40s_login_host_external_finalizer_after_source_freeze"

_SACCT_REQUIRED_FILE_ROLES = frozenset(
    {
        "approval_bound_configuration",
        "sacct_elf_dependency",
        "sacct_entrypoint",
        "sacct_slurm_plugin",
        "sacct_slurm_runtime",
    }
)
_SACCT_CLOSURE_FIELDS = (
    "capture_scope",
    "execution_surface",
    "query_contract",
    "immutable_files",
    "symlink_bindings",
    "ambient_runtime_dependencies",
    "trust_boundary",
)

_TRANSACTION_PATTERN = re.compile(r"pi05-[0-9a-f]{40}")
_CAPTURE_TRANSACTION_PATTERN = re.compile(
    r"polaris-runtime-trace-v2-[0-9a-f]{8}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
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
_APP_PRETERMINAL_FILENAME = "preterminal_attestation.json"
_APP_ZERO_WORK_COUNTERS = {
    "checkpoint_resolutions": 0,
    "checkpoint_reads": 0,
    "tokenizer_reads": 0,
    "model_loads": 0,
    "policy_client_class_loads": 0,
    "policy_client_instances": 0,
    "policy_requests": 0,
    "action_predictions": 0,
    "environment_imports": 0,
    "environment_constructions": 0,
    "gym_make_calls": 0,
    "environment_resets": 0,
    "environment_steps": 0,
    "episodes_started": 0,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _expected_sacct_query_command_template() -> list[str]:
    return [
        str(PINNED_SACCT_PATH),
        "-X",
        "--noheader",
        "--parsable2",
        "--jobs={job_id}",
        "--format=JobIDRaw,State,ExitCode,Submit,Start,End,ElapsedRaw,NodeList,Restarts",
    ]


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


def _stable_payload(
    path: Path, *, expected_sha256: str | None = None
) -> tuple[dict[str, Any], bytes]:
    path = Path(path)
    before = validate_immutable_file(path)
    payload = path.read_bytes()
    after = validate_immutable_file(path)
    _require(_identity(before) == _identity(after), f"immutable file changed: {path}")
    _require(
        _sha256(payload) == after["sha256"], f"immutable file digest drift: {path}"
    )
    if expected_sha256 is not None:
        _require(
            re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is not None
            and after["sha256"] == expected_sha256,
            f"immutable file approval mismatch: {path}",
        )
    return _identity(after), payload


def _stable_regular_file(
    path: Path, *, expected_mode: int, label: str
) -> dict[str, Any]:
    path = Path(path)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        _require(
            stat.S_ISREG(before.st_mode)
            and stat.S_IMODE(before.st_mode) == expected_mode
            and before.st_nlink == 1,
            f"{label} identity mismatch",
        )
        while block := os.read(descriptor, 16 * 1024 * 1024):
            digest.update(block)
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
        f"{label} changed during validation",
    )
    return {
        "path": str(path.resolve(strict=True)),
        "mode": f"{expected_mode:04o}",
        "nlink": 1,
        "size": before.st_size,
        "sha256": digest.hexdigest(),
    }


def _stable_executable(path: Path) -> dict[str, Any]:
    return _stable_regular_file(path, expected_mode=0o755, label="scheduler executable")


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _strict_json(payload: bytes, *, label: str) -> Any:
    try:
        value = json.loads(
            payload,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not strict JSON") from error
    _require(payload == _canonical_json_bytes(value), f"{label} is not canonical JSON")
    return value


def _runtime_record_schema(
    value: Any, *, label: str, with_roles: bool
) -> dict[str, Any]:
    required = {"path", "mode", "uid", "gid", "nlink", "size", "sha256"}
    if with_roles:
        required.add("roles")
    _require(
        isinstance(value, dict) and set(value) == required, f"{label} schema mismatch"
    )
    path = value["path"]
    _require(
        isinstance(path, str)
        and path.startswith("/")
        and PurePosixPath(path).as_posix() == path
        and ".." not in PurePosixPath(path).parts,
        f"{label} path is invalid",
    )
    _require(
        isinstance(value["mode"], str)
        and re.fullmatch(r"[0-7]{4}", value["mode"]) is not None
        and int(value["mode"], 8) & 0o022 == 0,
        f"{label} mode is unsafe",
    )
    for name in ("uid", "gid", "nlink", "size"):
        _require(
            type(value[name]) is int and value[name] >= 0,
            f"{label} {name} is invalid",
        )
    _require(
        value["nlink"] == 1
        and value["size"] > 0
        and isinstance(value["sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None,
        f"{label} immutable identity is invalid",
    )
    if with_roles:
        roles = value["roles"]
        _require(
            isinstance(roles, list)
            and roles
            and roles == sorted(set(roles))
            and all(
                isinstance(role, str)
                and re.fullmatch(r"[a-z][a-z0-9_]*", role) is not None
                for role in roles
            ),
            f"{label} roles are invalid",
        )
    return dict(value)


def _live_runtime_record(expected: dict[str, Any], *, label: str) -> dict[str, Any]:
    path = Path(expected["path"])
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
        while block := os.read(descriptor, 16 * 1024 * 1024):
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.stat(path, follow_symlinks=False)
    stable_fields = (
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
            for name in stable_fields
        )
        and str(path.resolve(strict=True)) == expected["path"],
        f"{label} changed during live validation",
    )
    observed = {
        "path": expected["path"],
        "mode": f"{stat.S_IMODE(before.st_mode):04o}",
        "uid": before.st_uid,
        "gid": before.st_gid,
        "nlink": before.st_nlink,
        "size": before.st_size,
        "sha256": digest.hexdigest(),
    }
    projection = {name: expected[name] for name in observed}
    _require(observed == projection, f"{label} differs from reviewed approval")
    return observed


def _validate_runtime_identity_list(
    values: Any, *, label: str, with_roles: bool, live: bool
) -> list[dict[str, Any]]:
    _require(isinstance(values, list), f"{label} must be a list")
    records = [
        _runtime_record_schema(item, label=f"{label} record", with_roles=with_roles)
        for item in values
    ]
    paths = [record["path"] for record in records]
    _require(paths == sorted(set(paths)), f"{label} paths are not unique and sorted")
    if live:
        for record in records:
            _live_runtime_record(record, label=f"{label}: {record['path']}")
    return records


def _validate_symlink_bindings(
    values: Any, *, live: bool, closed_regular_paths: set[str]
) -> list[dict[str, str]]:
    _require(isinstance(values, list), "sacct runtime symlink bindings must be a list")
    bindings: list[dict[str, str]] = []
    for value in values:
        _require(
            isinstance(value, dict)
            and set(value) == {"path", "target", "resolved_path"},
            "sacct runtime symlink binding schema mismatch",
        )
        path = value["path"]
        resolved = value["resolved_path"]
        target = value["target"]
        _require(
            all(isinstance(item, str) and item for item in (path, resolved, target))
            and path.startswith("/")
            and resolved.startswith("/")
            and PurePosixPath(path).as_posix() == path
            and PurePosixPath(resolved).as_posix() == resolved,
            "sacct runtime symlink binding is malformed",
        )
        _require(
            resolved in closed_regular_paths,
            "sacct runtime symlink target is outside the reviewed regular-file closure",
        )
        if live:
            before = os.lstat(path)
            observed_target = os.readlink(path)
            observed_resolved = str(Path(path).resolve(strict=True))
            after = os.lstat(path)
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
                stat.S_ISLNK(before.st_mode)
                and observed_target == target
                and observed_resolved == resolved
                and all(
                    getattr(before, name) == getattr(after, name) for name in fields
                ),
                f"sacct runtime symlink binding changed: {path}",
            )
        bindings.append(dict(value))
    paths = [binding["path"] for binding in bindings]
    _require(paths == sorted(set(paths)), "sacct runtime symlink paths are not sorted")
    return bindings


def _validate_ambient_runtime(value: Any, *, live: bool) -> dict[str, Any]:
    required = {
        "regular_files",
        "sockets",
        "external_services",
        "negative_resolution_assertions",
    }
    _require(
        isinstance(value, dict) and set(value) == required,
        "sacct ambient runtime dependency schema mismatch",
    )
    regular = _validate_runtime_identity_list(
        value["regular_files"],
        label="sacct ambient regular files",
        with_roles=False,
        live=live,
    )
    _require(
        isinstance(value["sockets"], list) and value["sockets"],
        "sacct ambient socket census must be nonempty",
    )
    sockets: list[dict[str, Any]] = []
    for item in value["sockets"]:
        _require(
            isinstance(item, dict)
            and set(item) == {"path", "present", "mode", "uid", "gid", "role"}
            and isinstance(item["path"], str)
            and item["path"].startswith("/")
            and PurePosixPath(item["path"]).as_posix() == item["path"]
            and ".." not in PurePosixPath(item["path"]).parts
            and type(item["present"]) is bool
            and isinstance(item["mode"], str)
            and re.fullmatch(r"[0-7]{4}", item["mode"]) is not None
            and type(item["uid"]) is int
            and item["uid"] >= 0
            and type(item["gid"]) is int
            and item["gid"] >= 0
            and isinstance(item["role"], str)
            and bool(item["role"]),
            "sacct ambient socket record is malformed",
        )
        if live:
            if item["present"]:
                before = os.stat(item["path"], follow_symlinks=False)
                after = os.stat(item["path"], follow_symlinks=False)
                _require(
                    stat.S_ISSOCK(before.st_mode)
                    and f"{stat.S_IMODE(before.st_mode):04o}" == item["mode"]
                    and before.st_uid == item["uid"]
                    and before.st_gid == item["gid"]
                    and (before.st_dev, before.st_ino, before.st_mode)
                    == (after.st_dev, after.st_ino, after.st_mode),
                    f"sacct ambient socket changed: {item['path']}",
                )
            else:
                _require(
                    not os.path.lexists(item["path"]),
                    f"sacct ambient absent socket appeared: {item['path']}",
                )
        sockets.append(dict(item))
    socket_paths = [item["path"] for item in sockets]
    _require(
        socket_paths == sorted(set(socket_paths)),
        "sacct ambient socket paths are not sorted",
    )
    services = value["external_services"]
    _require(
        isinstance(services, list)
        and services
        and all(
            isinstance(item, dict)
            and set(item) == {"kind", "endpoint", "role"}
            and all(isinstance(item[key], str) and item[key] for key in item)
            for item in services
        )
        and services
        == sorted(
            services, key=lambda item: (item["kind"], item["endpoint"], item["role"])
        )
        and len(services)
        == len({(item["kind"], item["endpoint"], item["role"]) for item in services}),
        "sacct ambient external-service census is malformed",
    )
    negative = value["negative_resolution_assertions"]
    _require(
        isinstance(negative, list)
        and negative
        and negative == sorted(set(negative))
        and all(
            isinstance(path, str)
            and path.startswith("/")
            and PurePosixPath(path).as_posix() == path
            and ".." not in PurePosixPath(path).parts
            for path in negative
        ),
        "sacct negative resolution assertions are malformed",
    )
    if live:
        for path in negative:
            _require(
                not os.path.lexists(path),
                f"sacct negative resolution assertion changed: {path}",
            )
    return {
        "regular_files": regular,
        "sockets": sockets,
        "external_services": services,
        "negative_resolution_assertions": negative,
    }


def _approval_artifact(value: Any, *, label: str) -> dict[str, Any]:
    _require(
        isinstance(value, dict)
        and set(value) == {"path", "mode", "nlink", "size", "sha256"}
        and value["mode"] == "0444"
        and value["nlink"] == 1
        and type(value["size"]) is int
        and value["size"] > 0
        and isinstance(value["sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None,
        f"{label} identity is malformed",
    )
    observed, payload = _stable_payload(
        Path(value["path"]), expected_sha256=value["sha256"]
    )
    _require(observed == value, f"{label} identity changed")
    return {"identity": observed, "payload": payload}


def _utc_timestamp(value: Any, *, label: str) -> str:
    _require(
        isinstance(value, str)
        and re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
            value,
        )
        is not None,
        f"{label} is malformed",
    )
    return value


def _reviewer_identity(value: Any) -> dict[str, str]:
    _require(
        isinstance(value, dict)
        and set(value) == {"profile", "principal", "role"}
        and value["profile"] == "polaris_external_sacct_runtime_reviewer_identity_v2"
        and value["role"] == "independent_agent_runtime_approval_reviewer"
        and isinstance(value["principal"], str)
        and re.fullmatch(r"codex-agent:/root/[a-z0-9_]+", value["principal"])
        is not None
        and value["principal"] != "UNPINNED_INDEPENDENT_AGENT_REVIEW_PENDING",
        "external sacct reviewer identity is not independently pinned",
    )
    return dict(value)


_DEPENDENCY_LIST_NAMES = (
    "immutable_files",
    "symlink_bindings",
    "ambient_regular_files",
    "ambient_sockets",
    "external_services",
    "negative_resolution_assertions",
)


def _dependency_lists(approval: dict[str, Any]) -> dict[str, list[Any]]:
    ambient = approval["ambient_runtime_dependencies"]
    return {
        "ambient_regular_files": ambient["regular_files"],
        "ambient_sockets": ambient["sockets"],
        "external_services": ambient["external_services"],
        "immutable_files": approval["immutable_files"],
        "negative_resolution_assertions": ambient["negative_resolution_assertions"],
        "symlink_bindings": approval["symlink_bindings"],
    }


def _expected_dependency_census(
    approval: dict[str, Any], *, closure_sha256: str
) -> dict[str, Any]:
    lists = _dependency_lists(approval)
    entries = {
        name: {
            "count": len(lists[name]),
            "sha256": _sha256(_canonical_json_bytes(lists[name])),
        }
        for name in _DEPENDENCY_LIST_NAMES
    }
    return {
        **entries,
        "census_sha256": _sha256(_canonical_json_bytes(lists)),
        "closure_sha256": closure_sha256,
        "profile": SACCT_RUNTIME_DEPENDENCY_CENSUS_PROFILE,
        "total_count": sum(len(value) for value in lists.values()),
    }


def _capture_package_root(value: Any) -> tuple[Path, dict[str, Any]]:
    required = {"path", "device", "inode", "mode", "uid", "gid"}
    _require(
        isinstance(value, dict) and set(value) == required,
        "capture package root schema mismatch",
    )
    path_text = value["path"]
    _require(
        isinstance(path_text, str)
        and path_text.startswith("/")
        and PurePosixPath(path_text).as_posix() == path_text
        and ".." not in PurePosixPath(path_text).parts,
        "capture package root path is invalid",
    )
    path = Path(path_text)
    metadata = os.stat(path, follow_symlinks=False)
    observed = {
        "device": metadata.st_dev,
        "gid": metadata.st_gid,
        "inode": metadata.st_ino,
        "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
        "path": str(path.resolve(strict=True)),
        "uid": metadata.st_uid,
    }
    _require(
        stat.S_ISDIR(metadata.st_mode)
        and not path.is_symlink()
        and value == observed
        and value["mode"] == "0555",
        "capture package root identity mismatch",
    )
    return path, observed


def _validate_capture_package(
    value: Any, *, producer: dict[str, Any]
) -> dict[str, Any]:
    _require(
        isinstance(value, dict)
        and set(value) == {"root", "manifest", "source", "payload_sha256"},
        "capture producer package schema mismatch",
    )
    root, _ = _capture_package_root(value["root"])
    manifest = _approval_artifact(value["manifest"], label="capture package manifest")
    source = _approval_artifact(
        value["source"], label="capture package producer source"
    )
    manifest_path = Path(manifest["identity"]["path"])
    source_path = Path(source["identity"]["path"])
    payload_sha256 = value["payload_sha256"]
    _require(
        manifest_path.parent == root
        and source_path.parent == root
        and manifest_path != source_path
        and isinstance(payload_sha256, dict)
        and bool(payload_sha256)
        and list(payload_sha256) == sorted(payload_sha256)
        and all(
            isinstance(name, str)
            and re.fullmatch(r"[A-Za-z0-9_.-]+", name) is not None
            and isinstance(digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
            for name, digest in payload_sha256.items()
        )
        and source_path.name in payload_sha256
        and payload_sha256[source_path.name] == source["identity"]["sha256"]
        and producer["path"] == str(source_path)
        and producer["sha256"] == source["identity"]["sha256"],
        "capture package source/manifest binding mismatch",
    )
    expected_manifest = "".join(
        f"{payload_sha256[name]}  {name}\n" for name in sorted(payload_sha256)
    ).encode("ascii")
    _require(
        manifest["payload"] == expected_manifest,
        "capture package manifest content mismatch",
    )
    _require(
        {item.name for item in root.iterdir()}
        == set(payload_sha256) | {manifest_path.name},
        "capture package top-level closure mismatch",
    )
    for name, expected_sha256 in payload_sha256.items():
        _, payload = _stable_payload(root / name, expected_sha256=expected_sha256)
        _require(
            _sha256(payload) == expected_sha256,
            "capture package payload digest mismatch",
        )
    return dict(value)


def _parse_capture_root_sacct(
    payload: bytes, *, expected_job_id: int, expected_node: str
) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("capture root sacct output is not UTF-8") from error
    _require(
        text.endswith("\n") and "\r" not in text and len(text.splitlines()) == 1,
        "capture root sacct must contain exactly one row",
    )
    fields = text.rstrip("\n").split("|")
    _require(
        len(fields) == 10 and fields[-1] == "",
        "capture root sacct is not exact ten-field parsable2",
    )
    job_id, state, exit_code, submitted, started, ended, elapsed, node, restarts, _ = (
        fields
    )
    timestamp = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}")
    _require(
        job_id == str(expected_job_id)
        and state == "COMPLETED"
        and exit_code == "0:0"
        and restarts == "0"
        and re.fullmatch(r"0|[1-9][0-9]*", elapsed) is not None
        and node == expected_node
        and all(
            timestamp.fullmatch(item) is not None
            for item in (submitted, started, ended)
        )
        and submitted <= started <= ended,
        "capture root sacct terminal values/order mismatch",
    )
    from datetime import datetime

    _require(
        int(
            (
                datetime.fromisoformat(ended) - datetime.fromisoformat(started)
            ).total_seconds()
        )
        == int(elapsed),
        "capture root sacct elapsed duration mismatch",
    )
    return {
        "elapsed_raw": int(elapsed),
        "end": ended,
        "exit_code": exit_code,
        "job_id": expected_job_id,
        "node": node,
        "raw_sha256": _sha256(payload),
        "restarts": 0,
        "start": started,
        "state": state,
        "submit": submitted,
    }


def _validate_capture_terminal(
    reference: Any,
    *,
    approval: dict[str, Any],
    candidate_identity: dict[str, Any],
    closure_sha256: str,
) -> dict[str, Any]:
    artifact = _approval_artifact(reference, label="external sacct capture terminal")
    value = _strict_json(artifact["payload"], label="external sacct capture terminal")
    required = {
        "schema_version",
        "profile",
        "status",
        "producer",
        "package",
        "job",
        "query",
        "surface",
        "sacct",
        "dependency_census",
        "candidate",
    }
    _require(
        isinstance(value, dict)
        and set(value) == required
        and type(value["schema_version"]) is int
        and value["schema_version"] == 1
        and value["profile"] == SACCT_RUNTIME_CAPTURE_TERMINAL_PROFILE
        and value["status"] == "captured_completed_root_sacct"
        and _canonical_json_bytes(value["candidate"])
        == _canonical_json_bytes(candidate_identity)
        and value["dependency_census"]
        == _expected_dependency_census(approval, closure_sha256=closure_sha256),
        "external sacct capture terminal schema mismatch",
    )
    producer = value["producer"]
    _require(
        isinstance(producer, dict)
        and set(producer) == {"profile", "module", "path", "sha256", "uid", "host"}
        and producer["profile"] == SACCT_RUNTIME_CAPTURE_PRODUCER_PROFILE
        and producer["module"] == SACCT_RUNTIME_CAPTURE_PRODUCER_MODULE
        and isinstance(producer["path"], str)
        and producer["path"].startswith("/")
        and PurePosixPath(producer["path"]).as_posix() == producer["path"]
        and isinstance(producer["sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", producer["sha256"]) is not None
        and type(producer["uid"]) is int
        and producer["uid"] >= 0
        and isinstance(producer["host"], str)
        and bool(producer["host"]),
        "capture terminal producer mismatch",
    )
    package = _validate_capture_package(value["package"], producer=producer)
    job = value["job"]
    _require(
        isinstance(job, dict)
        and set(job) == {"job_id", "compute_node", "transaction_id"}
        and type(job["job_id"]) is int
        and job["job_id"] > 0
        and isinstance(job["compute_node"], str)
        and re.fullmatch(r"pool0-[A-Za-z0-9.-]+", job["compute_node"]) is not None
        and isinstance(job["transaction_id"], str)
        and _CAPTURE_TRANSACTION_PATTERN.fullmatch(job["transaction_id"]) is not None,
        "capture terminal job identity mismatch",
    )
    expected_query = {
        "argv": [
            token.replace("{job_id}", str(job["job_id"]))
            for token in approval["query_contract"]["command_template"]
        ],
        "environment": approval["query_contract"]["environment"],
        "profile": SACCT_RUNTIME_CAPTURE_QUERY_PROFILE,
        "subprocess_timeout_seconds": approval["query_contract"][
            "subprocess_timeout_seconds"
        ],
    }
    _require(
        _canonical_json_bytes(value["query"]) == _canonical_json_bytes(expected_query),
        "capture terminal query contract mismatch",
    )
    surface = value["surface"]
    surface_keys = {
        "hostname",
        "machine_id_sha256",
        "kernel_release",
        "architecture",
        "effective_uid",
        "effective_gid",
    }
    _require(
        isinstance(surface, dict)
        and set(surface)
        == surface_keys | {"capture_started_at", "capture_finished_at"},
        "capture terminal surface schema mismatch",
    )
    started = _utc_timestamp(surface["capture_started_at"], label="capture start")
    finished = _utc_timestamp(surface["capture_finished_at"], label="capture finish")
    _require(
        {key: surface[key] for key in surface_keys} == approval["execution_surface"]
        and finished >= started
        and producer["uid"] == surface["effective_uid"]
        and producer["host"] == surface["hostname"],
        "capture terminal surface/producer mismatch",
    )
    sacct = value["sacct"]
    _require(
        isinstance(sacct, dict)
        and set(sacct)
        == {
            "executable",
            "version",
            "runtime_closure",
            "stdout",
            "stderr",
            "returncode",
            "parsed_root_row",
        },
        "capture terminal sacct schema mismatch",
    )
    executable = next(
        (
            item
            for item in approval["immutable_files"]
            if item["path"] == str(PINNED_SACCT_PATH)
        ),
        None,
    )
    stdout = _approval_artifact(sacct["stdout"], label="capture terminal sacct stdout")
    parsed = _parse_capture_root_sacct(
        stdout["payload"],
        expected_job_id=job["job_id"],
        expected_node=job["compute_node"],
    )
    _require(
        executable is not None
        and sacct["executable"] == executable
        and isinstance(sacct["version"], str)
        and 0 < len(sacct["version"]) <= 256
        and all(0x20 <= ord(character) <= 0x7E for character in sacct["version"])
        and sacct["runtime_closure"]
        == {
            "after_sha256": closure_sha256,
            "before_sha256": closure_sha256,
            "identical": True,
            "profile": SACCT_RUNTIME_CAPTURE_CLOSURE_PROFILE,
        }
        and sacct["stderr"] == ""
        and type(sacct["returncode"]) is int
        and sacct["returncode"] == 0
        and sacct["parsed_root_row"] == parsed
        and stdout["identity"]["path"]
        not in {
            producer["path"],
            candidate_identity["path"],
            artifact["identity"]["path"],
            package["manifest"]["path"],
        }
        and package["source"]["path"] == producer["path"],
        "capture terminal sacct content binding mismatch",
    )
    return {
        "identity": artifact["identity"],
        "path": artifact["identity"]["path"],
        "producer": producer,
        "sha256": artifact["identity"]["sha256"],
        "job": job,
        "finished_at": finished,
    }


def _validate_review_decision(
    reference: Any,
    *,
    candidate: dict[str, Any],
    capture_terminal: dict[str, Any],
    closure_sha256: str,
    approval_path: Path,
    reviewer_identity: dict[str, str],
) -> dict[str, Any]:
    artifact = _approval_artifact(reference, label="external sacct review decision")
    value = _strict_json(artifact["payload"], label="external sacct review decision")
    required = {
        "approval_path",
        "approved_at",
        "candidate_path",
        "candidate_sha256",
        "capture_terminal_path",
        "capture_terminal_sha256",
        "closure_sha256",
        "decision",
        "profile",
        "review_path",
        "reviewer_identity",
        "review_scope",
        "schema_version",
    }
    approved_at = _utc_timestamp(
        value.get("approved_at") if isinstance(value, dict) else None,
        label="external sacct review timestamp",
    )
    _require(
        isinstance(value, dict)
        and set(value) == required
        and value["profile"] == SACCT_RUNTIME_REVIEW_PROFILE
        and type(value["schema_version"]) is int
        and value["schema_version"] == 2
        and value["decision"] == "approve"
        and value["reviewer_identity"] == reviewer_identity
        and value["review_scope"] == SACCT_RUNTIME_REVIEW_SCOPE
        and value["candidate_path"] == candidate["path"]
        and value["capture_terminal_path"] == capture_terminal["path"]
        and value["review_path"] == artifact["identity"]["path"]
        and value["approval_path"] == str(approval_path)
        and value["candidate_sha256"] == candidate["sha256"]
        and value["capture_terminal_sha256"] == capture_terminal["sha256"]
        and value["closure_sha256"] == closure_sha256
        and approved_at >= capture_terminal["finished_at"]
        and reviewer_identity["principal"]
        != (
            f"uid:{capture_terminal['producer']['uid']}@"
            f"{capture_terminal['producer']['host']}"
        ),
        "external sacct review decision mismatch",
    )
    return {
        "identity": artifact["identity"],
        "path": artifact["identity"]["path"],
        "sha256": artifact["identity"]["sha256"],
    }


def validate_sacct_runtime_approval(
    path: Path, *, expected_sha256: str, live: bool
) -> dict[str, Any]:
    path = Path(path)
    _require(
        path.is_absolute()
        and PurePosixPath(str(path)).as_posix() == str(path)
        and path.resolve(strict=True) == path,
        "external sacct runtime approval path is not canonical and physical",
    )
    artifact, payload = _stable_payload(path, expected_sha256=expected_sha256)
    value = _strict_json(payload, label="external sacct runtime approval")
    required = {
        "schema_version",
        "profile",
        "capture_scope",
        "execution_surface",
        "query_contract",
        "immutable_files",
        "symlink_bindings",
        "ambient_runtime_dependencies",
        "trace_evidence",
        "trust_boundary",
    }
    _require(
        isinstance(value, dict)
        and set(value) == required
        and type(value["schema_version"]) is int
        and value["schema_version"] == 2
        and value["profile"] == SACCT_RUNTIME_APPROVAL_PROFILE
        and value["capture_scope"] == SACCT_RUNTIME_CAPTURE_SCOPE,
        "external sacct runtime approval schema mismatch",
    )
    surface = value["execution_surface"]
    _require(
        isinstance(surface, dict)
        and set(surface)
        == {
            "hostname",
            "machine_id_sha256",
            "kernel_release",
            "architecture",
            "effective_uid",
            "effective_gid",
        }
        and isinstance(surface["hostname"], str)
        and bool(surface["hostname"])
        and re.fullmatch(r"[0-9a-f]{64}", surface["machine_id_sha256"]) is not None
        and isinstance(surface["kernel_release"], str)
        and bool(surface["kernel_release"])
        and surface["architecture"] == "x86_64"
        and type(surface["effective_uid"]) is int
        and surface["effective_uid"] >= 0
        and type(surface["effective_gid"]) is int
        and surface["effective_gid"] >= 0,
        "external sacct execution surface is malformed",
    )
    query = value["query_contract"]
    expected_environment = {
        "PATH": SACCT_QUERY_PATH,
        "SLURM_CONF": str(PINNED_SLURM_CONFIG_PATH),
        "LD_LIBRARY_PATH": SACCT_QUERY_LD_LIBRARY_PATH,
    }
    _require(
        isinstance(query, dict)
        and set(query)
        == {"profile", "command_template", "environment", "subprocess_timeout_seconds"}
        and query["profile"] == "polaris_external_sacct_query_v1"
        and query["command_template"] == _expected_sacct_query_command_template()
        and query["environment"] == expected_environment
        and query["subprocess_timeout_seconds"] == SACCT_SUBPROCESS_TIMEOUT_SECONDS,
        "external sacct query contract mismatch",
    )
    immutable_files = _validate_runtime_identity_list(
        value["immutable_files"],
        label="sacct immutable runtime files",
        with_roles=True,
        live=live,
    )
    by_path = {record["path"]: record for record in immutable_files}
    observed_roles = {
        role for record in immutable_files for role in record.get("roles", [])
    }
    _require(
        observed_roles == _SACCT_REQUIRED_FILE_ROLES
        and str(PINNED_SACCT_PATH) in by_path
        and str(PINNED_SLURM_CONFIG_PATH) in by_path
        and str(PINNED_SLURM_LIBRARY_PATH) in by_path
        and "sacct_entrypoint" in by_path[str(PINNED_SACCT_PATH)]["roles"]
        and by_path[str(PINNED_SACCT_PATH)]["mode"] == "0755"
        and by_path[str(PINNED_SACCT_PATH)]["size"] > 0
        and "approval_bound_configuration"
        in by_path[str(PINNED_SLURM_CONFIG_PATH)]["roles"]
        and by_path[str(PINNED_SLURM_CONFIG_PATH)]["mode"] == "0644"
        and by_path[str(PINNED_SLURM_CONFIG_PATH)]["size"] > 0
        and "sacct_slurm_runtime" in by_path[str(PINNED_SLURM_LIBRARY_PATH)]["roles"]
        and by_path[str(PINNED_SLURM_LIBRARY_PATH)]["mode"] == "0644"
        and by_path[str(PINNED_SLURM_LIBRARY_PATH)]["size"] > 0,
        "external sacct runtime closure omits a required role or file",
    )
    ambient = _validate_ambient_runtime(
        value["ambient_runtime_dependencies"], live=live
    )
    immutable_paths = set(by_path)
    ambient_paths = {record["path"] for record in ambient["regular_files"]}
    _require(
        immutable_paths.isdisjoint(ambient_paths),
        "sacct immutable and ambient regular-file closures overlap",
    )
    _validate_symlink_bindings(
        value["symlink_bindings"],
        live=live,
        closed_regular_paths=immutable_paths | ambient_paths,
    )
    machine_record = by_path.get("/etc/machine-id") or next(
        (
            record
            for record in ambient["regular_files"]
            if record["path"] == "/etc/machine-id"
        ),
        None,
    )
    _require(
        machine_record is not None
        and machine_record["sha256"] == surface["machine_id_sha256"],
        "external sacct surface does not bind /etc/machine-id",
    )
    trust = value["trust_boundary"]
    _require(
        isinstance(trust, dict)
        and set(trust)
        == {
            "closed_claim",
            "trusted_but_unclosed",
            "review_requirement",
            "reviewer_identity",
        }
        and trust["closed_claim"]
        == "reviewed_full_sacct_regular_file_symlink_and_declared_ambient_runtime_closure_v1"
        and isinstance(trust["trusted_but_unclosed"], list)
        and bool(trust["trusted_but_unclosed"])
        and trust["trusted_but_unclosed"] == sorted(set(trust["trusted_but_unclosed"]))
        and all(
            isinstance(item, str) and item for item in trust["trusted_but_unclosed"]
        )
        and trust["review_requirement"]
        == "independent_agent_review_of_trace_candidate_and_capture_terminal"
        and isinstance(trust["reviewer_identity"], dict),
        "external sacct runtime trust boundary mismatch",
    )
    reviewer_identity = _reviewer_identity(trust["reviewer_identity"])
    closure_value = {key: value[key] for key in _SACCT_CLOSURE_FIELDS}
    closure_sha256 = _sha256(_canonical_json_bytes(closure_value))
    trace = value["trace_evidence"]
    _require(
        isinstance(trace, dict)
        and set(trace)
        == {"candidate", "capture_terminal", "review_decision", "closure_sha256"}
        and trace["closure_sha256"] == closure_sha256,
        "external sacct trace-evidence schema mismatch",
    )
    candidate_artifact = _approval_artifact(
        trace["candidate"], label="sacct trace candidate"
    )
    candidate_value = _strict_json(
        candidate_artifact["payload"], label="sacct runtime trace candidate"
    )
    _require(
        isinstance(candidate_value, dict)
        and set(candidate_value) == required
        and type(candidate_value["schema_version"]) is int
        and candidate_value["schema_version"] == 2
        and candidate_value["profile"] == SACCT_RUNTIME_CANDIDATE_PROFILE
        and candidate_value["trace_evidence"] == {}
        and _canonical_json_bytes(
            {key: candidate_value[key] for key in _SACCT_CLOSURE_FIELDS}
        )
        == _canonical_json_bytes({key: value[key] for key in _SACCT_CLOSURE_FIELDS}),
        "reviewed sacct approval differs from its trace candidate closure",
    )
    candidate = {
        "identity": candidate_artifact["identity"],
        "path": candidate_artifact["identity"]["path"],
        "sha256": candidate_artifact["identity"]["sha256"],
    }
    capture_terminal = _validate_capture_terminal(
        trace["capture_terminal"],
        approval=value,
        candidate_identity=candidate["identity"],
        closure_sha256=closure_sha256,
    )
    review = _validate_review_decision(
        trace["review_decision"],
        candidate=candidate,
        capture_terminal=capture_terminal,
        closure_sha256=closure_sha256,
        approval_path=path,
        reviewer_identity=reviewer_identity,
    )
    _require(
        len(
            {
                capture_terminal["producer"]["path"],
                candidate["path"],
                capture_terminal["path"],
                review["path"],
                str(path),
            }
        )
        == 5,
        "external sacct approval evidence paths overlap",
    )
    if live:
        uname = os.uname()
        _require(
            surface
            == {
                "hostname": uname.nodename,
                "machine_id_sha256": machine_record["sha256"],
                "kernel_release": uname.release,
                "architecture": uname.machine,
                "effective_uid": os.geteuid(),
                "effective_gid": os.getegid(),
            },
            "external sacct finalizer execution surface mismatch",
        )
    return {
        "artifact": artifact,
        "closure_sha256": closure_sha256,
        "execution_surface": surface,
        "query_contract": query,
        "key_files": {
            "sacct": by_path[str(PINNED_SACCT_PATH)],
            "slurm_config": by_path[str(PINNED_SLURM_CONFIG_PATH)],
            "slurm_library": by_path[str(PINNED_SLURM_LIBRARY_PATH)],
        },
        "trace_evidence": {
            "candidate": candidate["identity"],
            "capture_terminal": capture_terminal["identity"],
            "review_decision": review["identity"],
            "closure_sha256": closure_sha256,
        },
        "capture_producer": dict(capture_terminal["producer"]),
        "capture_job": dict(capture_terminal["job"]),
        "reviewer_identity": reviewer_identity,
    }


def _scheduler_module_producer() -> dict[str, Any]:
    path = Path(__file__).resolve(strict=True)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        _require(
            stat.S_ISREG(before.st_mode)
            and before.st_nlink == 1
            and stat.S_IMODE(before.st_mode) & 0o022 == 0,
            "scheduler validation producer source is mutable or not regular",
        )
        while block := os.read(descriptor, 16 * 1024 * 1024):
            digest.update(block)
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
        "scheduler validation producer source changed during hashing",
    )
    host = os.uname().nodename
    uid = os.geteuid()
    return {
        "profile": "polaris_sacct_prelaunch_validation_producer_v1",
        "module": "polaris.pi05_droid_jointpos_scheduler",
        "path": str(path),
        "sha256": digest.hexdigest(),
        "uid": uid,
        "host": host,
        "principal": f"uid:{uid}@{host}",
    }


def _receipt_evidence_paths(runtime_approval: dict[str, Any]) -> dict[str, str]:
    trace = runtime_approval["trace_evidence"]
    paths = {
        "approval": runtime_approval["artifact"]["path"],
        "candidate": trace["candidate"]["path"],
        "capture_terminal": trace["capture_terminal"]["path"],
        "review_decision": trace["review_decision"]["path"],
    }
    _require(
        len(set(paths.values())) == 4,
        "prelaunch receipt evidence paths are not distinct",
    )
    return paths


def publish_sacct_prelaunch_validation_receipt(
    output_path: Path,
    *,
    approval_path: Path,
    expected_approval_sha256: str,
    source_approval_path: Path,
) -> dict[str, Any]:
    """Fully validate the reviewed sacct closure before any Slurm submission."""

    output_path = Path(output_path)
    _require(
        output_path.is_absolute()
        and output_path.name == SACCT_PRELAUNCH_RECEIPT_FILENAME
        and not output_path.exists()
        and not output_path.is_symlink(),
        "sacct prelaunch receipt must be a new canonical output",
    )
    source_approval = validate_immutable_json(Path(source_approval_path))
    source_value = source_approval["value"]
    producer = _scheduler_module_producer()
    _require(
        isinstance(source_value, dict)
        and isinstance(source_value.get("snapshot_path"), str)
        and isinstance(source_value.get("source_tree_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", source_value["source_tree_sha256"])
        is not None,
        "source approval does not bind a source snapshot",
    )
    snapshot = Path(source_value["snapshot_path"])
    _require(
        snapshot.is_absolute()
        and snapshot.resolve(strict=True) == snapshot
        and Path(producer["path"]).is_relative_to(snapshot)
        and Path(producer["path"])
        == snapshot / "src/polaris/pi05_droid_jointpos_scheduler.py",
        "prelaunch validation producer escaped the approved source snapshot",
    )
    runtime_approval = validate_sacct_runtime_approval(
        Path(approval_path), expected_sha256=expected_approval_sha256, live=True
    )
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    value = {
        "schema_version": 1,
        "profile": SACCT_PRELAUNCH_RECEIPT_PROFILE,
        "status": "full_live_validation_passed_before_sbatch",
        "validated_at": timestamp,
        "producer": producer,
        "source_approval": _identity(source_approval),
        "source_tree_sha256": source_value["source_tree_sha256"],
        "approval": runtime_approval["artifact"],
        "closure_sha256": runtime_approval["closure_sha256"],
        "execution_surface": runtime_approval["execution_surface"],
        "query_contract": runtime_approval["query_contract"],
        "capture_producer": runtime_approval["capture_producer"],
        "reviewer_identity": runtime_approval["reviewer_identity"],
        "evidence_paths": _receipt_evidence_paths(runtime_approval),
    }
    return publish_immutable_json(output_path, value)


def validate_sacct_prelaunch_validation_receipt(
    path: Path,
    *,
    expected_sha256: str,
    expected_approval_path: Path,
    expected_approval_sha256: str,
    live: bool,
) -> dict[str, Any]:
    receipt = validate_immutable_json(Path(path))
    _require(receipt["sha256"] == expected_sha256, "prelaunch receipt digest mismatch")
    value = receipt["value"]
    required = {
        "schema_version",
        "profile",
        "status",
        "validated_at",
        "producer",
        "source_approval",
        "source_tree_sha256",
        "approval",
        "closure_sha256",
        "execution_surface",
        "query_contract",
        "capture_producer",
        "reviewer_identity",
        "evidence_paths",
    }
    _require(
        isinstance(value, dict)
        and set(value) == required
        and type(value["schema_version"]) is int
        and value["schema_version"] == 1
        and value["profile"] == SACCT_PRELAUNCH_RECEIPT_PROFILE
        and value["status"] == "full_live_validation_passed_before_sbatch",
        "prelaunch receipt schema mismatch",
    )
    receipt_reviewer = _reviewer_identity(value["reviewer_identity"])
    _utc_timestamp(value["validated_at"], label="prelaunch receipt timestamp")
    producer = value["producer"]
    _require(
        isinstance(producer, dict)
        and set(producer)
        == {"profile", "module", "path", "sha256", "uid", "host", "principal"}
        and producer["profile"] == "polaris_sacct_prelaunch_validation_producer_v1"
        and producer["module"] == "polaris.pi05_droid_jointpos_scheduler"
        and producer["principal"] == f"uid:{producer['uid']}@{producer['host']}"
        and isinstance(producer["path"], str)
        and producer["path"].startswith("/")
        and re.fullmatch(r"[0-9a-f]{64}", producer["sha256"]) is not None
        and type(producer["uid"]) is int
        and producer["uid"] >= 0
        and isinstance(producer["host"], str)
        and bool(producer["host"]),
        "prelaunch receipt producer mismatch",
    )
    source_approval = validate_immutable_json(Path(value["source_approval"]["path"]))
    source_value = source_approval["value"]
    _require(
        _identity(source_approval) == value["source_approval"]
        and isinstance(source_value, dict)
        and source_value.get("source_tree_sha256") == value["source_tree_sha256"]
        and isinstance(source_value.get("snapshot_path"), str)
        and Path(producer["path"])
        == Path(source_value["snapshot_path"])
        / "src/polaris/pi05_droid_jointpos_scheduler.py",
        "prelaunch receipt source binding mismatch",
    )
    producer_identity, producer_payload = _stable_payload(Path(producer["path"]))
    _require(
        producer_identity["sha256"] == producer["sha256"]
        and _sha256(producer_payload) == producer["sha256"],
        "prelaunch receipt producer source changed",
    )
    runtime_approval = validate_sacct_runtime_approval(
        Path(expected_approval_path),
        expected_sha256=expected_approval_sha256,
        live=live,
    )
    _require(
        value["approval"] == runtime_approval["artifact"]
        and value["closure_sha256"] == runtime_approval["closure_sha256"]
        and value["execution_surface"] == runtime_approval["execution_surface"]
        and value["query_contract"] == runtime_approval["query_contract"]
        and value["capture_producer"] == runtime_approval["capture_producer"]
        and receipt_reviewer == runtime_approval["reviewer_identity"]
        and value["evidence_paths"] == _receipt_evidence_paths(runtime_approval),
        "prelaunch receipt differs from the reviewed live approval",
    )
    if live:
        current = _scheduler_module_producer()
        _require(
            producer == current,
            "prelaunch receipt producer host/source differs at live validation",
        )
    return receipt


def _parse_env_record(payload: bytes, *, label: str) -> dict[str, str]:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8") from error
    _require(text.endswith("\n") and "\r" not in text, f"{label} is unterminated")
    values: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        _require(
            bool(separator)
            and re.fullmatch(r"[a-z][a-z0-9_]*", key) is not None
            and key not in values,
            f"{label} schema is malformed",
        )
        values[key] = value
    return values


def _validate_app_provenance(
    path: Path,
    *,
    expected_sha256: str,
    expected_job_id: int,
    expected_transaction_id: str,
    held: dict[str, Any],
) -> dict[str, Any]:
    artifact, payload = _stable_payload(path, expected_sha256=expected_sha256)
    fields = _parse_env_record(payload, label="AppLauncher runtime approval")
    required = {
        "profile",
        "output_root",
        "output_namespace_parent",
        "output_namespace_parent_identity",
        "runtime_closure_approval",
        "runtime_closure_approval_sha256",
        "sacct_runtime_approval",
        "sacct_runtime_approval_sha256",
        "sacct_prelaunch_validation_receipt",
        "sacct_prelaunch_validation_receipt_sha256",
        "scheduler_query_profile",
        "scheduler_query_path",
        "scheduler_query_slurm_conf",
        "scheduler_query_ld_library_path",
        "scheduler_query_timeout_seconds",
        "expected_slurm_config_path",
        "expected_slurm_config_sha256",
        "expected_slurm_config_size",
        "expected_scontrol_sha256",
        "expected_scontrol_size",
        "expected_slurm_library_path",
        "expected_slurm_library_sha256",
        "expected_slurm_library_size",
        "expected_sacct_path",
        "expected_sacct_sha256",
        "expected_sacct_size",
        "expected_scancel_sha256",
        "expected_scancel_size",
        "expected_srun_sha256",
        "expected_srun_size",
        "approved_batch_script",
        "batch_script_sha256",
        "submission_argv_sha256",
        "held_scheduler_record_sha256",
    }
    _require(
        set(fields) == required
        and fields["profile"] == "polaris_app_launcher_runtime_approval_v6",
        "AppLauncher runtime approval schema mismatch",
    )
    for name in (
        "runtime_closure_approval_sha256",
        "sacct_runtime_approval_sha256",
        "sacct_prelaunch_validation_receipt_sha256",
        "expected_slurm_config_sha256",
        "expected_scontrol_sha256",
        "expected_slurm_library_sha256",
        "expected_sacct_sha256",
        "expected_scancel_sha256",
        "expected_srun_sha256",
        "batch_script_sha256",
        "submission_argv_sha256",
        "held_scheduler_record_sha256",
    ):
        _require(
            re.fullmatch(r"[0-9a-f]{64}", fields[name]) is not None,
            f"AppLauncher runtime approval digest is malformed: {name}",
        )
    for name in (
        "expected_slurm_config_size",
        "expected_scontrol_size",
        "expected_slurm_library_size",
        "expected_sacct_size",
        "expected_scancel_size",
        "expected_srun_size",
    ):
        _require(
            re.fullmatch(r"[1-9][0-9]*", fields[name]) is not None,
            f"AppLauncher runtime approval size is malformed: {name}",
        )
    expected_environment = {
        "PATH": SACCT_QUERY_PATH,
        "SLURM_CONF": str(PINNED_SLURM_CONFIG_PATH),
        "LD_LIBRARY_PATH": SACCT_QUERY_LD_LIBRARY_PATH,
    }
    _require(
        fields["scheduler_query_profile"] == "polaris_app_launcher_sacct_query_v1"
        and fields["scheduler_query_path"] == expected_environment["PATH"]
        and fields["scheduler_query_slurm_conf"] == expected_environment["SLURM_CONF"]
        and fields["scheduler_query_ld_library_path"]
        == expected_environment["LD_LIBRARY_PATH"]
        and fields["scheduler_query_timeout_seconds"]
        == str(int(SACCT_SUBPROCESS_TIMEOUT_SECONDS))
        and fields["expected_slurm_config_path"] == str(PINNED_SLURM_CONFIG_PATH)
        and fields["expected_slurm_library_path"] == str(PINNED_SLURM_LIBRARY_PATH)
        and fields["expected_sacct_path"] == str(PINNED_SACCT_PATH),
        "AppLauncher runtime approval scheduler query environment mismatch",
    )
    _require(
        fields["held_scheduler_record_sha256"] == held["sha256"],
        "AppLauncher runtime approval does not bind the held scheduler record",
    )
    provenance_dir = Path(path).parent
    _require(
        Path(held["path"]).parent == provenance_dir,
        "held scheduler record escaped the approved provenance directory",
    )
    sacct_runtime_approval_path = Path(fields["sacct_runtime_approval"])
    _require(
        sacct_runtime_approval_path.is_absolute()
        and PurePosixPath(fields["sacct_runtime_approval"]).as_posix()
        == fields["sacct_runtime_approval"],
        "external sacct runtime approval path is not canonical",
    )
    sacct_runtime_approval = validate_sacct_runtime_approval(
        sacct_runtime_approval_path,
        expected_sha256=fields["sacct_runtime_approval_sha256"],
        live=False,
    )
    prelaunch_receipt_path = Path(fields["sacct_prelaunch_validation_receipt"])
    _require(
        prelaunch_receipt_path.is_absolute()
        and PurePosixPath(str(prelaunch_receipt_path)).as_posix()
        == str(prelaunch_receipt_path),
        "sacct prelaunch validation receipt path is not canonical",
    )
    prelaunch_receipt = validate_sacct_prelaunch_validation_receipt(
        prelaunch_receipt_path,
        expected_sha256=fields["sacct_prelaunch_validation_receipt_sha256"],
        expected_approval_path=sacct_runtime_approval_path,
        expected_approval_sha256=fields["sacct_runtime_approval_sha256"],
        live=False,
    )
    batch, _ = _stable_payload(
        provenance_dir / "batch_script.sbatch",
        expected_sha256=fields["batch_script_sha256"],
    )
    argv_artifact, argv_payload = _stable_payload(
        provenance_dir / "submission_argv.sh",
        expected_sha256=fields["submission_argv_sha256"],
    )
    try:
        argv_text = argv_payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("submission argv is not UTF-8") from error
    _require(
        argv_text.endswith("\n") and "\r" not in argv_text,
        "submission argv is unterminated",
    )
    tokens = shlex.split(argv_text)
    _require(
        len(tokens) == 10
        and tokens[1:4] == ["--parsable", "--hold", "--no-requeue"]
        and tokens[4] == f"--comment={expected_transaction_id}"
        and tokens[-1] == fields["approved_batch_script"],
        "submission argv does not bind the no-requeue transaction",
    )
    export_tokens = [token for token in tokens if token.startswith("--export=")]
    _require(len(export_tokens) == 1, "submission argv export is not unique")
    exports: dict[str, str] = {}
    for item in export_tokens[0].removeprefix("--export=").split(","):
        key, separator, value = item.partition("=")
        _require(bool(separator) and key not in exports, "submission export malformed")
        exports[key] = value
    _require(
        exports.get("SUBMISSION_TRANSACTION_ID") == expected_transaction_id
        and exports.get("POLARIS_EVAL_MODE") == "app_launcher_only"
        and re.fullmatch(
            r"[0-9a-f]{64}", exports.get("EXPECTED_POLARIS_SOURCE_TREE_SHA256", "")
        )
        is not None
        and re.fullmatch(r"[0-9a-f]{40}", exports.get("EXPECTED_POLARIS_COMMIT", ""))
        is not None,
        "submission export does not bind mode, transaction, and source",
    )
    _require(
        held["value"]["job"]["job_id"] == expected_job_id
        and held["value"]["job"]["transaction_id"] == expected_transaction_id,
        "held scheduler record differs from promotion identity",
    )
    return {
        "artifact": artifact,
        "fields": fields,
        "batch_script": batch,
        "submission_argv": argv_artifact,
        "exports": exports,
        "sacct_runtime_approval": sacct_runtime_approval,
        "sacct_prelaunch_validation_receipt": _identity(prelaunch_receipt),
    }


def _capture_app_scheduler_query_runtime(
    provenance: dict[str, Any], *, expected_job_id: int
) -> dict[str, Any]:
    fields = provenance["fields"]
    prelaunch_receipt = validate_sacct_prelaunch_validation_receipt(
        Path(fields["sacct_prelaunch_validation_receipt"]),
        expected_sha256=fields["sacct_prelaunch_validation_receipt_sha256"],
        expected_approval_path=Path(fields["sacct_runtime_approval"]),
        expected_approval_sha256=fields["sacct_runtime_approval_sha256"],
        live=True,
    )
    runtime_approval = validate_sacct_runtime_approval(
        Path(fields["sacct_runtime_approval"]),
        expected_sha256=fields["sacct_runtime_approval_sha256"],
        live=True,
    )
    key_files = runtime_approval["key_files"]
    sacct_client = _identity(key_files["sacct"])
    slurm_config = _identity(key_files["slurm_config"])
    slurm_library = _identity(key_files["slurm_library"])
    expected_identities = (
        (
            sacct_client,
            fields["expected_sacct_path"],
            fields["expected_sacct_sha256"],
            fields["expected_sacct_size"],
            "sacct client",
        ),
        (
            slurm_config,
            fields["expected_slurm_config_path"],
            fields["expected_slurm_config_sha256"],
            fields["expected_slurm_config_size"],
            "Slurm config",
        ),
        (
            slurm_library,
            fields["expected_slurm_library_path"],
            fields["expected_slurm_library_sha256"],
            fields["expected_slurm_library_size"],
            "Slurm library",
        ),
    )
    for identity, path, sha256, size, label in expected_identities:
        _require(
            identity["path"] == path
            and identity["sha256"] == sha256
            and identity["size"] == int(size),
            f"{label} differs from the AppLauncher runtime approval",
        )
    environment = dict(runtime_approval["query_contract"]["environment"])
    _require(
        environment
        == {
            "PATH": SACCT_QUERY_PATH,
            "SLURM_CONF": str(PINNED_SLURM_CONFIG_PATH),
            "LD_LIBRARY_PATH": SACCT_QUERY_LD_LIBRARY_PATH,
        },
        "AppLauncher sacct environment is not the exact approved environment",
    )
    command = [
        token.replace("{job_id}", str(expected_job_id))
        for token in runtime_approval["query_contract"]["command_template"]
    ]
    return {
        "sacct_prelaunch_validation_receipt": _identity(prelaunch_receipt),
        "sacct_runtime_approval": runtime_approval["artifact"],
        "sacct_runtime_closure_sha256": runtime_approval["closure_sha256"],
        "sacct_execution_surface": runtime_approval["execution_surface"],
        "sacct_trace_evidence": runtime_approval["trace_evidence"],
        "sacct_client": sacct_client,
        "slurm_config": slurm_config,
        "slurm_library": slurm_library,
        "sacct_environment": environment,
        "sacct_command": command,
    }


def _validate_app_preterminal_tree(
    path: Path,
    *,
    expected_job_id: int,
    expected_transaction_id: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    preterminal = validate_immutable_json(Path(path))
    value = preterminal["value"]
    required = {
        "schema_version",
        "profile",
        "status",
        "authoritative_completion",
        "termination_mode",
        "log",
        "pre_seal_worker_exit_codes",
        "pre_seal_exit_claim",
        "final_process_exit",
        "required_external_terminal_contract",
        "artifacts_before_attestation",
        "source_approval",
        "execution_approvals",
        "pyxis_image",
        "zero_work_counters",
        "scientific_result",
        "seal_intent",
    }
    _require(
        isinstance(value, dict)
        and set(value) == required
        and value["schema_version"] == 1
        and value["profile"] == "polaris_public_eval_app_launcher_preterminal_v1"
        and value["status"] == "awaiting_external_allocation_terminal_attestation"
        and value["authoritative_completion"] is False
        and value["required_external_terminal_contract"]
        == "allocation_COMPLETED_exit_0:0_Restarts_0"
        and value["scientific_result"] is None
        and value["zero_work_counters"] == _APP_ZERO_WORK_COUNTERS
        and value["pre_seal_worker_exit_codes"]
        == {"srun_exit_code": 0, "log_exit_code": 0, "helper_exit_code": 0},
        "AppLauncher preterminal attestation schema mismatch",
    )
    task_dir = Path(path).parent
    run_dir = task_dir.parent
    _require(
        Path(path).name == _APP_PRETERMINAL_FILENAME
        and task_dir.name == "app_launcher_only"
        and not task_dir.is_symlink()
        and not run_dir.is_symlink()
        and stat.S_IMODE(task_dir.stat().st_mode) == 0o555
        and stat.S_IMODE(run_dir.stat().st_mode) == 0o555,
        "AppLauncher preterminal directories are not terminally sealed",
    )
    names = set(os.listdir(task_dir))
    evidence = value["artifacts_before_attestation"]
    _require(
        isinstance(evidence, dict)
        and names == set(evidence) | {_APP_PRETERMINAL_FILENAME}
        and "SUCCESS" not in names
        and "FAILED" not in names
        and not (run_dir / "SUCCESS").exists()
        and not (run_dir / "FAILED").exists(),
        "AppLauncher in-job tree contains a success-shaped terminal marker",
    )
    for name, expected in evidence.items():
        observed = validate_immutable_file(task_dir / name)
        _require(
            _identity(observed) == expected, f"AppLauncher evidence changed: {name}"
        )
    preterminal_identity = _identity(preterminal)
    source = value["source_approval"]
    exports = provenance["exports"]
    source_approval_path = Path(exports["POLARIS_SOURCE_APPROVAL"])
    source_approval = validate_immutable_json(source_approval_path)
    source_value = source_approval["value"]
    _require(
        isinstance(source, dict)
        and source
        == {
            "BATCH_VERIFIED_POLARIS_SOURCE_TREE_SHA256": exports[
                "EXPECTED_POLARIS_SOURCE_TREE_SHA256"
            ],
            "POLARIS_IMPLEMENTATION_COMMIT": source_value.get("implementation_commit"),
            "SOURCE_APPROVAL_SHA256": source_approval["sha256"],
        }
        and source_value.get("snapshot_path") == exports["POLARIS_SOURCE_SNAPSHOT"]
        and source_value.get("source_tree_sha256")
        == exports["EXPECTED_POLARIS_SOURCE_TREE_SHA256"]
        and source_value.get("polaris_base_commit")
        == exports["EXPECTED_POLARIS_COMMIT"],
        "AppLauncher preterminal source approval is not submission-bound",
    )
    image = value["pyxis_image"]
    _require(
        isinstance(image, dict)
        and set(image) == {"path", "expected_sha256", "observed"}
        and isinstance(image["observed"], dict),
        "AppLauncher Pyxis image evidence is malformed",
    )
    image_identity = validate_immutable_file(Path(image["path"]))
    _require(
        image["expected_sha256"] == image_identity["sha256"]
        and image["observed"] == _identity(image_identity),
        "AppLauncher Pyxis image changed before external promotion",
    )
    request = validate_immutable_json(task_dir / "scheduler_request.json")
    handoff = validate_immutable_json(task_dir / "scheduler_handoff.json")
    step_terminal = validate_immutable_json(task_dir / "scheduler_terminal.json")
    _require(
        _identity(request) == evidence["scheduler_request.json"]
        and _identity(handoff) == evidence["scheduler_handoff.json"]
        and _identity(step_terminal) == evidence["scheduler_terminal.json"],
        "AppLauncher scheduler evidence identity changed",
    )
    request_value = request["value"]
    handoff_value = handoff["value"]
    step_value = step_terminal["value"]
    _require(
        isinstance(request_value, dict)
        and request_value.get("job_id") == expected_job_id
        and request_value.get("submission_transaction_id") == expected_transaction_id
        and isinstance(handoff_value, dict)
        and handoff_value.get("profile") == "polaris_public_eval_scheduler_handoff_v1"
        and handoff_value.get("status") == "host_scheduler_records_sealed"
        and handoff_value.get("request") == _identity(request)
        and isinstance(handoff_value.get("job_record"), dict),
        "AppLauncher running scheduler handoff is not transaction-bound",
    )
    running_raw = handoff_value["job_record"].get("raw")
    _require(isinstance(running_raw, str), "AppLauncher running job record is missing")
    running = parse_scontrol_job_record(
        running_raw + "\n",
        phase="running",
        expected_job_id=expected_job_id,
        expected_transaction_id=expected_transaction_id,
    )
    _require(
        isinstance(step_value, dict)
        and step_value.get("profile") == "polaris_public_eval_scheduler_terminal_v1"
        and step_value.get("status") == "scheduler_step_terminal_and_cgroup_unpopulated"
        and step_value.get("request") == _identity(request)
        and step_value.get("handoff") == _identity(handoff)
        and step_value.get("scancel_invoked") is False
        and step_value.get("sacct_terminal", {}).get("state") == "COMPLETED"
        and step_value.get("sacct_terminal", {}).get("exit_code") == "0:0"
        and step_value.get("cgroup_events", {}).get("values", {}).get("populated") == 0,
        "AppLauncher step closure is not clean and terminal",
    )
    return {
        "preterminal": preterminal_identity,
        "source_approval": _identity(source_approval),
        "pyxis_image": _identity(image_identity),
        "scheduler_request": _identity(request),
        "scheduler_handoff": _identity(handoff),
        "scheduler_step_terminal": _identity(step_terminal),
        "running_job": running,
    }


def _publish_immutable_payload(path: Path, payload: bytes) -> dict[str, Any]:
    """Durably publish one nonempty payload with no replace authority."""

    path = Path(path)
    _require(
        path.is_absolute()
        and path.parent.resolve(strict=True) == path.parent
        and not path.exists()
        and not path.is_symlink()
        and isinstance(payload, bytes)
        and bool(payload),
        "immutable payload output is invalid",
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            _require(written > 0, "short immutable payload write")
            offset += written
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return validate_immutable_file(path)


def _allocation_projection(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "raw"}


def _validate_app_sacct_query_receipt(
    path: Path,
    *,
    expected_job_id: int,
    expected_transaction_id: str,
    provenance: dict[str, Any],
    scheduler_runtime: dict[str, Any],
) -> dict[str, Any]:
    receipt = validate_immutable_json(Path(path))
    value = receipt["value"]
    required = {
        "schema_version",
        "profile",
        "status",
        "producer",
        "job_id",
        "transaction_id",
        "app_runtime_approval",
        "sacct_prelaunch_validation_receipt",
        "sacct_runtime_approval",
        "sacct_runtime_closure_sha256",
        "sacct_execution_surface",
        "sacct_trace_evidence",
        "sacct_runtime_validation",
        "sacct_client",
        "slurm_config",
        "slurm_library",
        "sacct_environment",
        "sacct_command",
        "sacct_subprocess_timeout_seconds",
        "stdout",
        "stderr",
        "returncode",
        "allocation_terminal",
    }
    _require(
        isinstance(value, dict)
        and set(value) == required
        and type(value["schema_version"]) is int
        and value["schema_version"] == 1
        and value["profile"] == APP_SACCT_QUERY_RECEIPT_PROFILE
        and value["status"] == "live_governed_query_completed"
        and value["job_id"] == expected_job_id
        and value["transaction_id"] == expected_transaction_id,
        "AppLauncher live sacct query receipt schema mismatch",
    )
    _require(
        value["producer"] == _scheduler_module_producer(),
        "AppLauncher live sacct query producer changed",
    )
    runtime_sha256 = _sha256(_canonical_json_bytes(scheduler_runtime))
    expected_runtime_validation = {
        "profile": "polaris_external_sacct_runtime_query_validation_v1",
        "before_sha256": runtime_sha256,
        "after_sha256": runtime_sha256,
        "publication_recheck_sha256": runtime_sha256,
        "identical": True,
    }
    _require(
        value["app_runtime_approval"] == provenance["artifact"]
        and value["sacct_prelaunch_validation_receipt"]
        == scheduler_runtime["sacct_prelaunch_validation_receipt"]
        and value["sacct_runtime_approval"]
        == scheduler_runtime["sacct_runtime_approval"]
        and value["sacct_runtime_closure_sha256"]
        == scheduler_runtime["sacct_runtime_closure_sha256"]
        and value["sacct_execution_surface"]
        == scheduler_runtime["sacct_execution_surface"]
        and value["sacct_trace_evidence"] == scheduler_runtime["sacct_trace_evidence"]
        and value["sacct_runtime_validation"] == expected_runtime_validation
        and value["sacct_client"] == scheduler_runtime["sacct_client"]
        and value["slurm_config"] == scheduler_runtime["slurm_config"]
        and value["slurm_library"] == scheduler_runtime["slurm_library"]
        and value["sacct_environment"] == scheduler_runtime["sacct_environment"]
        and value["sacct_command"] == scheduler_runtime["sacct_command"]
        and value["sacct_subprocess_timeout_seconds"]
        == SACCT_SUBPROCESS_TIMEOUT_SECONDS
        and value["stderr"] == ""
        and type(value["returncode"]) is int
        and value["returncode"] == 0,
        "AppLauncher live sacct query receipt contract mismatch",
    )
    stdout_path = Path(value["stdout"]["path"])
    _require(
        stdout_path.parent == Path(path).parent
        and stdout_path.name == APP_SACCT_QUERY_RAW_FILENAME
        and stdout_path != Path(path),
        "AppLauncher live sacct stdout path escaped provenance",
    )
    stdout_artifact = _approval_artifact(
        value["stdout"], label="AppLauncher live sacct stdout"
    )
    raw = stdout_artifact["payload"]
    try:
        raw_text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("AppLauncher live sacct stdout is not UTF-8") from error
    parsed = parse_sacct_terminal_record(raw_text, expected_job_id=expected_job_id)
    _require(
        value["allocation_terminal"] == _allocation_projection(parsed),
        "AppLauncher live sacct parsed result differs from governed stdout",
    )
    return receipt


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


def _attest_app_terminal(
    output_path: Path,
    *,
    held_record_path: Path,
    app_runtime_approval_path: Path,
    expected_app_runtime_approval_sha256: str,
    preterminal_path: Path,
    expected_job_id: int,
    expected_transaction_id: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Publish the sole authoritative AppLauncher completion artifact."""

    _require(
        SACCT_SUBPROCESS_TIMEOUT_SECONDS <= timeout_seconds <= 600,
        "terminal accounting timeout is invalid",
    )
    output_path = Path(output_path)
    approval_path = Path(app_runtime_approval_path)
    _require(
        output_path.name == APP_TERMINAL_PROMOTION_FILENAME
        and output_path.parent == approval_path.parent
        and not output_path.exists()
        and not output_path.is_symlink(),
        "AppLauncher promotion output must be a new provenance-local artifact",
    )
    held = validate_persisted_scheduler_job(
        held_record_path,
        phase="held",
        expected_job_id=expected_job_id,
        expected_transaction_id=expected_transaction_id,
    )
    provenance = _validate_app_provenance(
        approval_path,
        expected_sha256=expected_app_runtime_approval_sha256,
        expected_job_id=expected_job_id,
        expected_transaction_id=expected_transaction_id,
        held=held,
    )
    app = _validate_app_preterminal_tree(
        preterminal_path,
        expected_job_id=expected_job_id,
        expected_transaction_id=expected_transaction_id,
        provenance=provenance,
    )
    raw_path = output_path.parent / APP_SACCT_QUERY_RAW_FILENAME
    query_receipt_path = output_path.parent / APP_SACCT_QUERY_RECEIPT_FILENAME
    _require(
        not raw_path.exists()
        and not raw_path.is_symlink()
        and not query_receipt_path.exists()
        and not query_receipt_path.is_symlink(),
        "AppLauncher governed sacct query outputs already exist",
    )
    deadline = time.monotonic() + timeout_seconds
    last_error = "sacct did not return a terminal allocation row"
    while True:
        scheduler_runtime_before = _capture_app_scheduler_query_runtime(
            provenance, expected_job_id=expected_job_id
        )
        remaining = deadline - time.monotonic()
        if remaining < SACCT_SUBPROCESS_TIMEOUT_SECONDS:
            raise TimeoutError(last_error)
        command = scheduler_runtime_before["sacct_command"]
        environment = scheduler_runtime_before["sacct_environment"]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                shell=False,
                env=dict(environment),
                timeout=SACCT_SUBPROCESS_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            result = None
            last_error = "sacct terminal query exceeded its bounded subprocess timeout"
        finally:
            scheduler_runtime_after = _capture_app_scheduler_query_runtime(
                provenance, expected_job_id=expected_job_id
            )
            _require(
                scheduler_runtime_before == scheduler_runtime_after,
                "external sacct runtime changed across the scheduler query",
            )
        if result is not None and result.returncode == 0 and result.stderr == "":
            try:
                _require(
                    isinstance(result.stdout, str)
                    and len(result.stdout.encode("utf-8")) <= 65536,
                    "sacct terminal query output exceeds its bound",
                )
                allocation = parse_sacct_terminal_record(
                    result.stdout, expected_job_id=expected_job_id
                )
                stdout = _publish_immutable_payload(
                    raw_path, result.stdout.encode("utf-8")
                )
                runtime_sha256 = _sha256(
                    _canonical_json_bytes(scheduler_runtime_before)
                )
                query_receipt_value = {
                    "schema_version": 1,
                    "profile": APP_SACCT_QUERY_RECEIPT_PROFILE,
                    "status": "live_governed_query_completed",
                    "producer": _scheduler_module_producer(),
                    "job_id": expected_job_id,
                    "transaction_id": expected_transaction_id,
                    "app_runtime_approval": provenance["artifact"],
                    "sacct_prelaunch_validation_receipt": scheduler_runtime_before[
                        "sacct_prelaunch_validation_receipt"
                    ],
                    "sacct_runtime_approval": scheduler_runtime_before[
                        "sacct_runtime_approval"
                    ],
                    "sacct_runtime_closure_sha256": scheduler_runtime_before[
                        "sacct_runtime_closure_sha256"
                    ],
                    "sacct_execution_surface": scheduler_runtime_before[
                        "sacct_execution_surface"
                    ],
                    "sacct_trace_evidence": scheduler_runtime_before[
                        "sacct_trace_evidence"
                    ],
                    "sacct_runtime_validation": {
                        "profile": "polaris_external_sacct_runtime_query_validation_v1",
                        "before_sha256": runtime_sha256,
                        "after_sha256": runtime_sha256,
                        "publication_recheck_sha256": runtime_sha256,
                        "identical": True,
                    },
                    "sacct_client": scheduler_runtime_before["sacct_client"],
                    "slurm_config": scheduler_runtime_before["slurm_config"],
                    "slurm_library": scheduler_runtime_before["slurm_library"],
                    "sacct_environment": environment,
                    "sacct_command": command,
                    "sacct_subprocess_timeout_seconds": SACCT_SUBPROCESS_TIMEOUT_SECONDS,
                    "stdout": _identity(stdout),
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "allocation_terminal": _allocation_projection(allocation),
                }
                query_receipt = publish_immutable_json(
                    query_receipt_path, query_receipt_value
                )
                scheduler_runtime_publication = _capture_app_scheduler_query_runtime(
                    provenance, expected_job_id=expected_job_id
                )
                _require(
                    scheduler_runtime_publication == scheduler_runtime_before,
                    "external sacct runtime changed before receipt publication",
                )
                validated_query_receipt = _validate_app_sacct_query_receipt(
                    query_receipt_path,
                    expected_job_id=expected_job_id,
                    expected_transaction_id=expected_transaction_id,
                    provenance=provenance,
                    scheduler_runtime=scheduler_runtime_publication,
                )
                _require(
                    _identity(validated_query_receipt) == _identity(query_receipt),
                    "AppLauncher governed query receipt changed after publication",
                )
                value = {
                    "schema_version": 3,
                    "profile": APP_TERMINAL_PROMOTION_PROFILE,
                    "status": "authoritative_non_scientific_completion",
                    "authoritative_completion": True,
                    "scientific_result": None,
                    "job_id": expected_job_id,
                    "transaction_id": expected_transaction_id,
                    "held_scheduler_record": _identity(held),
                    "running_scheduler_handoff": app["scheduler_handoff"],
                    "scheduler_step_terminal": app["scheduler_step_terminal"],
                    "preterminal_attestation": app["preterminal"],
                    "app_runtime_approval": provenance["artifact"],
                    "batch_script": provenance["batch_script"],
                    "submission_argv": provenance["submission_argv"],
                    "source_approval": app["source_approval"],
                    "pyxis_image": app["pyxis_image"],
                    "sacct_query_receipt": _identity(query_receipt),
                    "allocation_terminal": _allocation_projection(allocation),
                }
                artifact = publish_immutable_json(output_path, value)
                reread = validate_app_terminal_promotion(
                    output_path,
                    expected_job_id=expected_job_id,
                    expected_transaction_id=expected_transaction_id,
                    expected_app_runtime_approval_sha256=(
                        expected_app_runtime_approval_sha256
                    ),
                )
                _require(
                    _identity(reread) == _identity(artifact),
                    "AppLauncher promotion changed after publication",
                )
                return artifact
            except ValueError as error:
                if (
                    raw_path.exists()
                    or query_receipt_path.exists()
                    or output_path.exists()
                ):
                    raise
                last_error = str(error)
        elif result is not None:
            last_error = "sacct terminal query failed"
        if time.monotonic() >= deadline:
            raise TimeoutError(last_error)
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


def validate_app_terminal_promotion(
    path: Path,
    *,
    expected_job_id: int,
    expected_transaction_id: str,
    expected_app_runtime_approval_sha256: str,
) -> dict[str, Any]:
    artifact = validate_immutable_json(Path(path))
    value = artifact["value"]
    required = {
        "schema_version",
        "profile",
        "status",
        "authoritative_completion",
        "scientific_result",
        "job_id",
        "transaction_id",
        "held_scheduler_record",
        "running_scheduler_handoff",
        "scheduler_step_terminal",
        "preterminal_attestation",
        "app_runtime_approval",
        "batch_script",
        "submission_argv",
        "source_approval",
        "pyxis_image",
        "sacct_query_receipt",
        "allocation_terminal",
    }
    _require(
        isinstance(value, dict)
        and set(value) == required
        and value["schema_version"] == 3
        and value["profile"] == APP_TERMINAL_PROMOTION_PROFILE
        and value["status"] == "authoritative_non_scientific_completion"
        and value["authoritative_completion"] is True
        and value["scientific_result"] is None
        and value["job_id"] == expected_job_id
        and value["transaction_id"] == expected_transaction_id,
        "AppLauncher promotion schema mismatch",
    )
    held_path = Path(value["held_scheduler_record"]["path"])
    held = validate_persisted_scheduler_job(
        held_path,
        phase="held",
        expected_job_id=expected_job_id,
        expected_transaction_id=expected_transaction_id,
    )
    provenance = _validate_app_provenance(
        Path(value["app_runtime_approval"]["path"]),
        expected_sha256=expected_app_runtime_approval_sha256,
        expected_job_id=expected_job_id,
        expected_transaction_id=expected_transaction_id,
        held=held,
    )
    scheduler_runtime = _capture_app_scheduler_query_runtime(
        provenance, expected_job_id=expected_job_id
    )
    app = _validate_app_preterminal_tree(
        Path(value["preterminal_attestation"]["path"]),
        expected_job_id=expected_job_id,
        expected_transaction_id=expected_transaction_id,
        provenance=provenance,
    )
    query_receipt_path = Path(value["sacct_query_receipt"]["path"])
    _require(
        query_receipt_path.parent == Path(path).parent
        and query_receipt_path.name == APP_SACCT_QUERY_RECEIPT_FILENAME,
        "AppLauncher governed query receipt path escaped promotion provenance",
    )
    query_receipt = _validate_app_sacct_query_receipt(
        query_receipt_path,
        expected_job_id=expected_job_id,
        expected_transaction_id=expected_transaction_id,
        provenance=provenance,
        scheduler_runtime=scheduler_runtime,
    )
    _require(
        _identity(query_receipt) == value["sacct_query_receipt"],
        "AppLauncher promotion query receipt identity changed",
    )
    receipt_value = query_receipt["value"]
    expected = {
        "schema_version": 3,
        "profile": APP_TERMINAL_PROMOTION_PROFILE,
        "status": "authoritative_non_scientific_completion",
        "authoritative_completion": True,
        "scientific_result": None,
        "job_id": expected_job_id,
        "transaction_id": expected_transaction_id,
        "held_scheduler_record": _identity(held),
        "running_scheduler_handoff": app["scheduler_handoff"],
        "scheduler_step_terminal": app["scheduler_step_terminal"],
        "preterminal_attestation": app["preterminal"],
        "app_runtime_approval": provenance["artifact"],
        "batch_script": provenance["batch_script"],
        "submission_argv": provenance["submission_argv"],
        "source_approval": app["source_approval"],
        "pyxis_image": app["pyxis_image"],
        "sacct_query_receipt": _identity(query_receipt),
        "allocation_terminal": receipt_value["allocation_terminal"],
    }
    _require(
        value == expected,
        "AppLauncher promotion differs from its governed query receipt",
    )
    return artifact


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
    prelaunch = subparsers.add_parser("validate-sacct-runtime-approval")
    prelaunch.add_argument("--output", type=Path, required=True)
    prelaunch.add_argument("--approval", type=Path, required=True)
    prelaunch.add_argument("--expected-approval-sha256", required=True)
    prelaunch.add_argument("--source-approval", type=Path, required=True)
    verify_prelaunch = subparsers.add_parser("verify-sacct-prelaunch-validation")
    verify_prelaunch.add_argument("--receipt", type=Path, required=True)
    verify_prelaunch.add_argument("--expected-receipt-sha256", required=True)
    verify_prelaunch.add_argument("--approval", type=Path, required=True)
    verify_prelaunch.add_argument("--expected-approval-sha256", required=True)
    args = parser.parse_args()
    if args.command == "capture-job":
        result = capture_scheduler_job(
            args.output,
            phase=args.phase,
            expected_job_id=args.job_id,
            expected_transaction_id=args.transaction_id,
        )
    elif args.command == "attest-terminal":
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
    elif args.command == "validate-sacct-runtime-approval":
        result = publish_sacct_prelaunch_validation_receipt(
            args.output,
            approval_path=args.approval,
            expected_approval_sha256=args.expected_approval_sha256,
            source_approval_path=args.source_approval,
        )
    else:
        result = validate_sacct_prelaunch_validation_receipt(
            args.receipt,
            expected_sha256=args.expected_receipt_sha256,
            expected_approval_path=args.approval,
            expected_approval_sha256=args.expected_approval_sha256,
            live=True,
        )
    print(json.dumps(_identity(result), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()


__all__ = [
    "APP_TERMINAL_PROMOTION_FILENAME",
    "APP_TERMINAL_PROMOTION_PROFILE",
    "APP_SACCT_QUERY_RAW_FILENAME",
    "APP_SACCT_QUERY_RECEIPT_FILENAME",
    "SACCT_PRELAUNCH_RECEIPT_FILENAME",
    "SCHEDULER_JOB_PROFILE",
    "SCHEDULER_RUNNING_FILENAME",
    "SCHEDULER_TERMINAL_FILENAME",
    "SCHEDULER_TERMINAL_PROFILE",
    "attest_terminal",
    "build_terminal_attestation",
    "capture_scheduler_job",
    "parse_sacct_terminal_record",
    "parse_scontrol_job_record",
    "publish_sacct_prelaunch_validation_receipt",
    "validate_sacct_prelaunch_validation_receipt",
    "validate_persisted_scheduler_job",
    "validate_app_terminal_promotion",
]
