"""Policy-free public evaluator startup diagnostic.

This module is intentionally limited to the Python standard library.  Its
``exec`` entrypoint runs inside the evaluator's Pyxis ``srun`` step, captures
the step and GPU boundary, publishes one immutable pre-exec artifact, and then
replaces itself with the normal ``/.venv/bin/python scripts/eval.py`` process.
The PID is preserved across ``execve`` so the post-AppLauncher branch can prove
that it is still in the same process and Slurm step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


STARTUP_DIAGNOSTIC_MODE = "app_launcher_only"
PREEXEC_PROFILE = "polaris_public_eval_app_launcher_preexec_v1"
PRECLOSE_PROFILE = "polaris_public_eval_app_launcher_preclose_v1"
READY_PROFILE = "polaris_public_eval_app_launcher_ready_v1"
SCHEDULER_REQUEST_PROFILE = "polaris_public_eval_scheduler_request_v1"
SCHEDULER_HANDOFF_PROFILE = "polaris_public_eval_scheduler_handoff_v1"
SCHEDULER_TERMINAL_REQUEST_PROFILE = "polaris_public_eval_scheduler_terminal_request_v1"
SCHEDULER_TERMINAL_PROFILE = "polaris_public_eval_scheduler_terminal_v1"
LOG_IDENTITY_PROFILE = "polaris_public_eval_app_launcher_log_identity_v1"
EXPECTED_NVIDIA_GPU_NAME = "NVIDIA L40S"
EXPECTED_NVIDIA_DRIVER_VERSION = "580.105.08"
EXPECTED_SCONTROL_PATH = Path("/cm/local/apps/slurm/24.11/bin/scontrol")
EXPECTED_SACCT_PATH = Path("/cm/local/apps/slurm/24.11/bin/sacct")
EXPECTED_SCANCEL_PATH = Path("/cm/local/apps/slurm/24.11/bin/scancel")
EXPECTED_SRUN_PATH = Path("/cm/local/apps/slurm/24.11/bin/srun")
EXPECTED_SLURM_LIBRARY_PATH = Path(
    "/cm/local/apps/slurm/24.11/lib64/slurm/libslurmfull.so"
)
EXPECTED_SLURM_CONFIG_PATH = Path("/cm/shared/apps/slurm/etc/oci-ord-cs-004/slurm.conf")
NVIDIA_SMI_COMMAND = (
    "/usr/bin/nvidia-smi",
    "--query-gpu=uuid,name,driver_version,minor_number",
    "--format=csv,noheader,nounits",
)

GPU_UUID_PATTERN = re.compile(
    r"GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
)
DECIMAL_PATTERN = re.compile(r"0|[1-9][0-9]*")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SUBMISSION_TRANSACTION_PATTERN = re.compile(r"pi05-[0-9a-f]{40}")

FORBIDDEN_MODULE_PREFIXES = (
    "gym",
    "gymnasium",
    "isaaclab_tasks",
    "openpi",
    "openpi_client",
    "polaris.environments",
    "polaris.policy",
    "sentencepiece",
    "tokenizers",
    "transformers",
)

ZERO_WORK_COUNTERS = {
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

SOURCE_IDENTITY_ENVIRONMENT = (
    "BATCH_VERIFIED_POLARIS_SOURCE_TREE_SHA256",
    "POLARIS_IMPLEMENTATION_COMMIT",
    "SOURCE_APPROVAL_SHA256",
)
PREEXEC_GPU_ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "SLURM_JOB_GPUS",
    "SLURM_STEP_GPUS",
    "SLURM_GPUS_ON_NODE",
    "SLURM_GPUS_PER_TASK",
    "SLURM_TRES_PER_TASK",
)
SCHEDULER_HANDOFF_ENVIRONMENT = (
    "POLARIS_STARTUP_DIAGNOSTIC_SCHEDULER_HANDOFF_PATH",
    "POLARIS_STARTUP_DIAGNOSTIC_SCHEDULER_HANDOFF_SHA256",
)
EXECUTION_APPROVAL_SHA256_ENVIRONMENT = (
    "POLARIS_EXPECTED_SLURM_CONFIG_SHA256",
    "POLARIS_RUNTIME_CLOSURE_APPROVAL_SHA256",
    "POLARIS_EXPECTED_SCONTROL_SHA256",
    "POLARIS_EXPECTED_SLURM_LIBRARY_SHA256",
    "POLARIS_EXPECTED_SACCT_SHA256",
    "POLARIS_EXPECTED_SCANCEL_SHA256",
    "POLARIS_EXPECTED_SRUN_SHA256",
)
EXECUTION_APPROVAL_SIZE_ENVIRONMENT = (
    "POLARIS_EXPECTED_SCONTROL_SIZE",
    "POLARIS_EXPECTED_SLURM_LIBRARY_SIZE",
    "POLARIS_EXPECTED_SACCT_SIZE",
    "POLARIS_EXPECTED_SCANCEL_SIZE",
    "POLARIS_EXPECTED_SRUN_SIZE",
)
EXECUTION_APPROVAL_ENVIRONMENT = (
    *EXECUTION_APPROVAL_SHA256_ENVIRONMENT,
    *EXECUTION_APPROVAL_SIZE_ENVIRONMENT,
)
PYXIS_IMAGE_ENVIRONMENT = (
    "POLARIS_PYXIS_IMAGE_PATH",
    "POLARIS_EXPECTED_PYXIS_IMAGE_SHA256",
    "POLARIS_OBSERVED_PYXIS_IMAGE_SHA256",
    "POLARIS_OBSERVED_PYXIS_IMAGE_MODE",
    "POLARIS_OBSERVED_PYXIS_IMAGE_NLINK",
    "POLARIS_OBSERVED_PYXIS_IMAGE_SIZE",
)
PRETERMINAL_EVIDENCE_ENTRIES = frozenset(
    {
        "startup_preexec.json",
        "startup_preclose.json",
        "startup_preclose.ready.json",
        "scheduler_request.json",
        "scheduler_handoff.json",
        "scheduler_terminal_request.json",
        "scheduler_terminal.json",
        "app_launcher_only.log",
        "app_launcher_only.log.identity.json",
    }
)
PRETERMINAL_ATTESTATION_NAME = "preterminal_attestation.json"
PRETERMINAL_TASK_ENTRIES = PRETERMINAL_EVIDENCE_ENTRIES | {PRETERMINAL_ATTESTATION_NAME}
FAILURE_ATTESTATION_NAME = "failure_attestation.json"


class _CaughtSignal(InterruptedError):
    def __init__(self, signum: int):
        super().__init__(f"caught signal {signum}")
        self.signum = signum


def _raise_caught_signal(signum: int, _frame: Any) -> None:
    raise _CaughtSignal(signum)


def _install_caught_signal_handlers() -> dict[int, Any]:
    previous: dict[int, Any] = {}
    for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, _raise_caught_signal)
    return previous


def _restore_signal_handlers(previous: Mapping[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def canonical_json_bytes(value: Any) -> bytes:
    """Return the sole accepted JSON representation."""

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


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _same_stat(left: os.stat_result, right: os.stat_result) -> bool:
    return all(
        getattr(left, field) == getattr(right, field)
        for field in (
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
    )


def _canonical_absolute_path(path: Path, *, label: str) -> Path:
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise ValueError(f"{label} must be an absolute file path")
    if PurePosixPath(path.as_posix()).as_posix() != path.as_posix():
        raise ValueError(f"{label} is not lexically canonical")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError(f"{label} parent must be one real directory")
    if parent.resolve(strict=True) != parent:
        raise ValueError(f"{label} parent must use its canonical physical path")
    return path


def _read_descriptor(descriptor: int, size: int) -> bytes:
    payload = bytearray()
    offset = 0
    while offset < size:
        block = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        if not block:
            raise ValueError("short descriptor read")
        payload.extend(block)
        offset += len(block)
    if os.pread(descriptor, 1, size):
        raise ValueError("descriptor grew while reading")
    return bytes(payload)


def _artifact_identity(
    path: Path, metadata: os.stat_result, payload: bytes
) -> dict[str, Any]:
    return {
        "path": str(path),
        "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
        "nlink": metadata.st_nlink,
        "size": len(payload),
        "sha256": _sha256(payload),
    }


def _validate_parent_binding(parent_fd: int, parent: Path) -> None:
    opened = os.fstat(parent_fd)
    observed = os.stat(parent, follow_symlinks=False)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(observed.st_mode)
        or opened.st_dev != observed.st_dev
        or opened.st_ino != observed.st_ino
    ):
        raise RuntimeError(f"parent directory binding changed: {parent}")


def directory_identity(metadata: os.stat_result) -> str:
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("directory identity requires a directory")
    return ":".join(
        (
            str(metadata.st_dev),
            str(metadata.st_ino),
            str(metadata.st_uid),
            str(metadata.st_gid),
            f"{stat.S_IMODE(metadata.st_mode):04o}",
        )
    )


def capture_directory_identity(path: Path) -> str:
    path = _canonical_directory(path, label="identity directory")
    return directory_identity(os.stat(path, follow_symlinks=False))


def _validate_expected_directory_identity(
    descriptor: int, *, expected: str, label: str
) -> None:
    if re.fullmatch(r"[0-9]+:[0-9]+:[0-9]+:[0-9]+:[0-7]{4}", expected) is None:
        raise ValueError(f"{label} expected identity is malformed")
    observed = directory_identity(os.fstat(descriptor))
    if observed != expected:
        raise RuntimeError(f"{label} creator-observed identity mismatch")


def _stable_file_at(
    parent_fd: int,
    *,
    name: str,
    path: Path,
    expected_mode: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_fd,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"file is not one regular link: {path}")
        if expected_mode is not None and stat.S_IMODE(before.st_mode) != expected_mode:
            raise ValueError(f"file mode is not {expected_mode:04o}: {path}")
        payload = _read_descriptor(descriptor, before.st_size)
        after = os.fstat(descriptor)
        observed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not _same_stat(before, after) or not _same_stat(after, observed):
            raise ValueError(f"file changed while reading: {path}")
        return payload, _artifact_identity(path, after, payload)
    finally:
        os.close(descriptor)


def stable_file_identity(path: Path) -> dict[str, Any]:
    """Hash one stable regular file without following a final symlink."""

    path = _canonical_absolute_path(path, label="source file")
    parent_fd = os.open(
        path.parent,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        _validate_parent_binding(parent_fd, path.parent)
        _, identity = _stable_file_at(parent_fd, name=path.name, path=path)
        _validate_parent_binding(parent_fd, path.parent)
        return identity
    finally:
        os.close(parent_fd)


def _stable_read_immutable_json_at(
    parent_fd: int,
    *,
    path: Path,
    expected_sha256: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    payload, identity = _stable_file_at(
        parent_fd,
        name=path.name,
        path=path,
        expected_mode=0o444,
    )
    digest = identity["sha256"]
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("immutable input SHA-256 mismatch")
    value = json.loads(
        payload,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if canonical_json_bytes(value) != payload:
        raise ValueError("immutable input is not canonical JSON")
    return value, identity


def _publish_immutable_json_at(
    parent_fd: int,
    *,
    parent: Path,
    path: Path,
    value: Any,
) -> dict[str, Any]:
    payload = canonical_json_bytes(value)
    _validate_parent_binding(parent_fd, parent)
    temporary_name = f".{path.name}.partial-{os.getpid()}-{os.urandom(8).hex()}"
    descriptor = -1
    linked = False
    temporary_created = False
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o400,
            dir_fd=parent_fd,
        )
        temporary_created = True
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short immutable artifact write")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(
            temporary_name,
            path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        linked = True
        os.unlink(temporary_name, dir_fd=parent_fd)
        os.fsync(parent_fd)
        _, identity = _stable_read_immutable_json_at(
            parent_fd,
            path=path,
            expected_sha256=_sha256(payload),
        )
        _validate_parent_binding(parent_fd, parent)
        return identity
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        if linked:
            # A destination that was successfully linked is forensic evidence.
            # Never remove or replace it after publication.
            os.fsync(parent_fd)
        raise


def publish_immutable_json(
    path: Path,
    value: Any,
    *,
    expected_parent_identity: str | None = None,
) -> dict[str, Any]:
    """Atomically publish one canonical, non-replacing, mode-0444 JSON file."""

    path = _canonical_absolute_path(path, label="output")
    parent_fd = os.open(
        path.parent,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    active_error: BaseException | None = None
    try:
        if expected_parent_identity is not None:
            _validate_expected_directory_identity(
                parent_fd,
                expected=expected_parent_identity,
                label="immutable JSON parent",
            )
        return _publish_immutable_json_at(
            parent_fd,
            parent=path.parent,
            path=path,
            value=value,
        )
    except BaseException as error:
        active_error = error
        raise
    finally:
        try:
            _validate_parent_binding(parent_fd, path.parent)
        except BaseException as binding_error:
            if active_error is None:
                raise
            active_error.add_note(
                f"parent binding validation also failed: {binding_error}"
            )
        finally:
            os.close(parent_fd)


def stable_read_immutable_json(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_parent_identity: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Read and validate one immutable canonical JSON artifact."""

    path = _canonical_absolute_path(path, label="immutable input")
    parent_fd = os.open(
        path.parent,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    active_error: BaseException | None = None
    try:
        if expected_parent_identity is not None:
            _validate_expected_directory_identity(
                parent_fd,
                expected=expected_parent_identity,
                label="immutable JSON parent",
            )
        _validate_parent_binding(parent_fd, path.parent)
        value, identity = _stable_read_immutable_json_at(
            parent_fd,
            path=path,
            expected_sha256=expected_sha256,
        )
        _validate_parent_binding(parent_fd, path.parent)
        return value, identity
    except BaseException as error:
        active_error = error
        raise
    finally:
        try:
            _validate_parent_binding(parent_fd, path.parent)
        except BaseException as binding_error:
            if active_error is None:
                raise
            active_error.add_note(
                f"parent binding validation also failed: {binding_error}"
            )
        finally:
            os.close(parent_fd)


def capture_immutable_log(
    *,
    output_path: Path,
    identity_path: Path,
    input_stream: Any,
    mirror_stream: Any,
    expected_parent_identity: str | None = None,
) -> dict[str, Any]:
    """Mirror stdin while publishing a non-replacing immutable log and identity."""

    output_path = _canonical_absolute_path(output_path, label="log output")
    identity_path = _canonical_absolute_path(identity_path, label="log identity")
    if output_path.parent != identity_path.parent or output_path == identity_path:
        raise ValueError("log and identity must be distinct files in one stable parent")
    parent_fd = os.open(
        output_path.parent,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    temporary_name = f".{output_path.name}.partial-{os.getpid()}-{os.urandom(8).hex()}"
    descriptor = -1
    linked = False
    temporary_created = False
    active_error: BaseException | None = None
    try:
        if expected_parent_identity is not None:
            _validate_expected_directory_identity(
                parent_fd,
                expected=expected_parent_identity,
                label="immutable log parent",
            )
        _validate_parent_binding(parent_fd, output_path.parent)
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o400,
            dir_fd=parent_fd,
        )
        temporary_created = True
        while True:
            block = input_stream.read(1024 * 1024)
            if not block:
                break
            if not isinstance(block, bytes):
                raise TypeError("immutable log input must yield bytes")
            offset = 0
            while offset < len(block):
                written = os.write(descriptor, block[offset:])
                if written <= 0:
                    raise OSError("short immutable log write")
                offset += written
            mirror_stream.write(block)
            mirror_stream.flush()
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(
            temporary_name,
            output_path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        linked = True
        os.unlink(temporary_name, dir_fd=parent_fd)
        os.fsync(parent_fd)
        _, log_identity = _stable_file_at(
            parent_fd,
            name=output_path.name,
            path=output_path,
            expected_mode=0o444,
        )
        identity_value = {
            "schema_version": 1,
            "profile": LOG_IDENTITY_PROFILE,
            "status": "immutable_log_finalized",
            "log": log_identity,
        }
        identity_identity = _publish_immutable_json_at(
            parent_fd,
            parent=output_path.parent,
            path=identity_path,
            value=identity_value,
        )
        _, final_log_identity = _stable_file_at(
            parent_fd,
            name=output_path.name,
            path=output_path,
            expected_mode=0o444,
        )
        if final_log_identity != log_identity:
            raise RuntimeError("immutable log changed during identity publication")
        os.fsync(parent_fd)
        _validate_parent_binding(parent_fd, output_path.parent)
        return {"log": log_identity, "identity": identity_identity}
    except BaseException as error:
        active_error = error
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        if linked:
            # Keep a successfully linked forensic log; never remove/replace it.
            os.fsync(parent_fd)
        raise
    finally:
        try:
            _validate_parent_binding(parent_fd, output_path.parent)
        except BaseException as binding_error:
            if active_error is None:
                raise
            active_error.add_note(
                f"parent binding validation also failed: {binding_error}"
            )
        finally:
            os.close(parent_fd)


def validate_immutable_log_identity(
    identity_path: Path, *, expected_parent_identity: str | None = None
) -> dict[str, Any]:
    value, identity = stable_read_immutable_json(
        identity_path, expected_parent_identity=expected_parent_identity
    )
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "profile", "status", "log"}
        or value["schema_version"] != 1
        or value["profile"] != LOG_IDENTITY_PROFILE
        or value["status"] != "immutable_log_finalized"
        or not isinstance(value["log"], dict)
    ):
        raise ValueError("immutable log identity schema mismatch")
    log_path = Path(value["log"].get("path", ""))
    if log_path.parent != identity_path.parent:
        raise ValueError("immutable log escaped its identity parent")
    parent_fd = os.open(
        identity_path.parent,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        if expected_parent_identity is not None:
            _validate_expected_directory_identity(
                parent_fd,
                expected=expected_parent_identity,
                label="immutable log validation parent",
            )
        _, current = _stable_file_at(
            parent_fd,
            name=log_path.name,
            path=log_path,
            expected_mode=0o444,
        )
        _validate_parent_binding(parent_fd, identity_path.parent)
    finally:
        os.close(parent_fd)
    if current != value["log"] or current["mode"] != "0444" or current["nlink"] != 1:
        raise ValueError("immutable log identity no longer matches the log")
    return {"log": current, "identity": identity}


def cleanup_transient_evidence(
    *, task_dir: Path, expected_task_identity: str
) -> list[str]:
    """Remove only known unpublished partials after all writer groups are absent."""

    task_dir = _canonical_directory(task_dir, label="transient-cleanup task directory")
    task_fd = os.open(
        task_dir,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    removed: list[str] = []
    try:
        _validate_expected_directory_identity(
            task_fd,
            expected=expected_task_identity,
            label="transient-cleanup task directory",
        )
        _validate_parent_binding(task_fd, task_dir)
        allowed_destinations = PRETERMINAL_TASK_ENTRIES | {FAILURE_ATTESTATION_NAME}
        for name in sorted(os.listdir(task_fd)):
            matched_destination = next(
                (
                    destination
                    for destination in allowed_destinations
                    if re.fullmatch(
                        rf"\.{re.escape(destination)}\.partial-[0-9]+-[0-9a-f]{{16}}",
                        name,
                    )
                ),
                None,
            )
            if matched_destination is None:
                continue
            metadata = os.stat(name, dir_fd=task_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) not in {0o400, 0o444}
            ):
                raise ValueError(f"unsafe transient evidence entry: {name}")
            os.unlink(name, dir_fd=task_fd)
            removed.append(name)
        os.fsync(task_fd)
        _validate_parent_binding(task_fd, task_dir)
        return removed
    finally:
        os.close(task_fd)


def _canonical_directory(path: Path, *, label: str) -> Path:
    if (
        not path.is_absolute()
        or PurePosixPath(path.as_posix()).as_posix() != path.as_posix()
    ):
        raise ValueError(f"{label} must be one canonical absolute directory")
    if path.is_symlink() or not path.is_dir() or path.resolve(strict=True) != path:
        raise ValueError(f"{label} must be one canonical real directory")
    return path


def _validate_child_directory_binding(
    parent_fd: int, child_fd: int, *, name: str, path: Path
) -> None:
    opened = os.fstat(child_fd)
    observed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(observed.st_mode)
        or opened.st_dev != observed.st_dev
        or opened.st_ino != observed.st_ino
    ):
        raise RuntimeError(f"directory binding changed: {path}")


def _task_file_inventory(task_fd: int, task_dir: Path) -> dict[str, dict[str, Any]]:
    names = sorted(os.listdir(task_fd))
    inventory: dict[str, dict[str, Any]] = {}
    for name in names:
        if name in {"", ".", ".."} or "/" in name or name.startswith("."):
            raise ValueError(f"unsafe task evidence entry: {name!r}")
        _, identity = _stable_file_at(
            task_fd,
            name=name,
            path=task_dir / name,
            expected_mode=0o444,
        )
        inventory[name] = identity
    return inventory


def create_output_directories(
    *, run_dir: Path, task_dir: Path, expected_parent_identity: str
) -> dict[str, str]:
    if (
        not run_dir.is_absolute()
        or PurePosixPath(run_dir.as_posix()).as_posix() != run_dir.as_posix()
        or task_dir != run_dir / "app_launcher_only"
    ):
        raise ValueError("output directories do not use the closed canonical layout")
    parent = _canonical_directory(run_dir.parent, label="output root")
    parent_fd = os.open(
        parent,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    run_fd = -1
    task_fd = -1
    run_created = False
    task_created = False
    try:
        _validate_expected_directory_identity(
            parent_fd,
            expected=expected_parent_identity,
            label="approved output namespace parent",
        )
        _validate_parent_binding(parent_fd, parent)
        saved_umask = os.umask(0)
        try:
            os.mkdir(run_dir.name, 0o755, dir_fd=parent_fd)
            run_created = True
            run_fd = os.open(
                run_dir.name,
                os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
            _validate_child_directory_binding(
                parent_fd, run_fd, name=run_dir.name, path=run_dir
            )
            os.mkdir(task_dir.name, 0o755, dir_fd=run_fd)
            task_created = True
            task_fd = os.open(
                task_dir.name,
                os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=run_fd,
            )
        finally:
            os.umask(saved_umask)
        _validate_child_directory_binding(
            run_fd, task_fd, name=task_dir.name, path=task_dir
        )
        os.fchmod(run_fd, 0o755)
        os.fchmod(task_fd, 0o755)
        os.fsync(task_fd)
        os.fsync(run_fd)
        os.fsync(parent_fd)
        if (
            stat.S_IMODE(os.fstat(run_fd).st_mode) != 0o755
            or stat.S_IMODE(os.fstat(task_fd).st_mode) != 0o755
        ):
            raise RuntimeError("created output directory mode is not exactly 0755")
        result = {
            "namespace_parent": str(parent),
            "namespace_parent_identity": expected_parent_identity,
            "run_dir": str(run_dir),
            "run_identity": directory_identity(os.fstat(run_fd)),
            "task_dir": str(task_dir),
            "task_identity": directory_identity(os.fstat(task_fd)),
        }
        _validate_expected_directory_identity(
            parent_fd,
            expected=expected_parent_identity,
            label="approved output namespace parent",
        )
        _validate_parent_binding(parent_fd, parent)
        return result
    except BaseException:
        # Clean up only through the creator-held descriptors and only while
        # every name still resolves to the directory that those descriptors
        # identify.  A binding failure is forensic incomplete state; reopening
        # by pathname here could remove an attacker's replacement tree.
        bindings_safe = run_fd >= 0
        if bindings_safe:
            try:
                _validate_parent_binding(parent_fd, parent)
                _validate_child_directory_binding(
                    parent_fd, run_fd, name=run_dir.name, path=run_dir
                )
                if task_created:
                    if task_fd < 0:
                        raise RuntimeError("created task directory was never pinned")
                    _validate_child_directory_binding(
                        run_fd, task_fd, name=task_dir.name, path=task_dir
                    )
            except (OSError, RuntimeError):
                bindings_safe = False
        if bindings_safe:
            try:
                if task_created:
                    os.rmdir(task_dir.name, dir_fd=run_fd)
                if run_created:
                    os.rmdir(run_dir.name, dir_fd=parent_fd)
            except OSError:
                # Nonempty or otherwise non-removable directories remain
                # visibly unsealed instead of risking pathname-based cleanup.
                pass
        os.fsync(parent_fd)
        raise
    finally:
        if task_fd >= 0:
            os.close(task_fd)
        if run_fd >= 0:
            os.close(run_fd)
        os.close(parent_fd)


def publish_failure_attestation(
    *,
    task_dir: Path,
    primary_exit_code: int,
    srun_exit_code: int,
    log_exit_code: int,
    helper_exit_code: int,
    signal_name: str,
    expected_task_identity: str,
) -> dict[str, Any]:
    task_dir = _canonical_directory(task_dir, label="failure task directory")
    codes = {
        "primary_exit_code": primary_exit_code,
        "srun_exit_code": srun_exit_code,
        "log_exit_code": log_exit_code,
        "helper_exit_code": helper_exit_code,
    }
    if any(
        not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 255
        for value in codes.values()
    ):
        raise ValueError("failure exit codes must be bytes")
    if primary_exit_code == 0:
        raise ValueError("failure attestation requires a nonzero primary exit")
    if signal_name not in {"none", "INT", "TERM", "HUP"}:
        raise ValueError("failure signal is not in the closed set")
    task_fd = os.open(
        task_dir,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        _validate_expected_directory_identity(
            task_fd,
            expected=expected_task_identity,
            label="failure task directory",
        )
        _validate_parent_binding(task_fd, task_dir)
        inventory = _task_file_inventory(task_fd, task_dir)
        if FAILURE_ATTESTATION_NAME in inventory:
            raise FileExistsError("failure attestation already exists")
        if not set(inventory) <= PRETERMINAL_EVIDENCE_ENTRIES:
            raise ValueError("failure tree contains an unexpected evidence entry")
        _validate_terminal_closure_evidence(
            task_fd=task_fd,
            task_dir=task_dir,
            inventory=inventory,
        )
        value = {
            "schema_version": 1,
            "profile": "polaris_public_eval_app_launcher_failure_v2",
            "status": "non_scientific_diagnostic_failed",
            "signal": signal_name,
            "exit_codes": codes,
            "process_closure": (
                "scheduler_step_terminal_cgroup_unpopulated_and_"
                "tracked_groups_absent_before_attestation"
            ),
            "primary_exit_outcome": "nonzero",
            "cleanup_outcome": "completed",
            "seal_intent": "terminal_seal_required",
            "artifacts_before_attestation": inventory,
            "zero_work_counters": dict(ZERO_WORK_COUNTERS),
            "scientific_result": None,
        }
        identity = _publish_immutable_json_at(
            task_fd,
            parent=task_dir,
            path=task_dir / FAILURE_ATTESTATION_NAME,
            value=value,
        )
        _validate_parent_binding(task_fd, task_dir)
        return identity
    finally:
        os.close(task_fd)


def _validate_terminal_closure_evidence(
    *,
    task_fd: int,
    task_dir: Path,
    inventory: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    names = (
        "startup_preexec.json",
        "scheduler_request.json",
        "scheduler_handoff.json",
        "scheduler_terminal_request.json",
        "scheduler_terminal.json",
    )
    if not set(names) <= set(inventory):
        raise ValueError("failure lacks proven scheduler terminal closure")
    values: dict[str, Any] = {}
    identities: dict[str, dict[str, Any]] = {}
    for name in names:
        value, identity = _stable_read_immutable_json_at(task_fd, path=task_dir / name)
        if identity != inventory[name]:
            raise ValueError(f"terminal closure identity changed: {name}")
        values[name] = value
        identities[name] = identity
    request = _validate_scheduler_request(values["scheduler_request.json"])
    approvals = _validate_execution_approval_environment(os.environ)
    handoff = _validate_scheduler_handoff(
        values["scheduler_handoff.json"],
        request=request,
        request_identity=identities["scheduler_request.json"],
        **_scheduler_approval_kwargs(approvals),
    )
    expected_host_binding = scheduler_host_binding_value(
        environ=os.environ, request=request
    )
    if handoff["host_binding"] != expected_host_binding:
        raise ValueError(
            "terminal closure host binding differs from trusted helper env"
        )
    terminal_request = _validate_scheduler_terminal_request(
        values["scheduler_terminal_request.json"],
        request=request,
        request_identity=identities["scheduler_request.json"],
        handoff=handoff,
        handoff_identity=identities["scheduler_handoff.json"],
    )
    preexec = values["startup_preexec.json"]
    if not isinstance(preexec, dict):
        raise ValueError("terminal closure pre-exec evidence is malformed")
    expected_cgroup_path = (
        preexec.get("runtime", {}).get("cgroup", {}).get("job_step_path")
    )
    if not isinstance(expected_cgroup_path, str):
        raise ValueError("terminal closure pre-exec cgroup path is missing")
    return _validate_scheduler_terminal(
        values["scheduler_terminal.json"],
        request=request,
        request_identity=identities["scheduler_request.json"],
        handoff=handoff,
        handoff_identity=identities["scheduler_handoff.json"],
        terminal_request=terminal_request,
        terminal_request_identity=identities["scheduler_terminal_request.json"],
        expected_cgroup_path=expected_cgroup_path,
        approvals=approvals,
    )


def _validate_success_evidence_semantics(
    *,
    task_fd: int,
    task_dir: Path,
    inventory: Mapping[str, Mapping[str, Any]],
    expected_namespace_parent_identity: str,
    expected_run_identity: str,
    expected_task_identity: str,
) -> None:
    values: dict[str, Any] = {}
    identities: dict[str, dict[str, Any]] = {}
    for name in (
        "startup_preexec.json",
        "startup_preclose.json",
        "startup_preclose.ready.json",
        "scheduler_request.json",
        "scheduler_handoff.json",
        "scheduler_terminal_request.json",
        "scheduler_terminal.json",
    ):
        value, identity = _stable_read_immutable_json_at(task_fd, path=task_dir / name)
        if identity != inventory[name]:
            raise ValueError(f"success evidence identity changed: {name}")
        values[name] = value
        identities[name] = identity

    preexec = values["startup_preexec.json"]
    preclose = values["startup_preclose.json"]
    ready = values["startup_preclose.ready.json"]
    request = _validate_scheduler_request(values["scheduler_request.json"])
    if not isinstance(preclose, dict) or set(preclose) != {
        "schema_version",
        "profile",
        "status",
        "startup_diagnostic",
        "preexec",
        "runtime",
        "forbidden_module_prefixes",
        "forbidden_loaded_modules",
        "zero_work_counters",
        "bounded_diagnostic_counts",
    }:
        raise ValueError("pre-close evidence schema mismatch")
    if (
        isinstance(preclose["schema_version"], bool)
        or preclose["schema_version"] != 1
        or preclose["profile"] != PRECLOSE_PROFILE
        or preclose["status"] != "simulation_app_close_pending"
        or preclose["startup_diagnostic"] != STARTUP_DIAGNOSTIC_MODE
        or preclose["preexec"] != identities["startup_preexec.json"]
        or preclose["forbidden_module_prefixes"] != list(FORBIDDEN_MODULE_PREFIXES)
        or preclose["forbidden_loaded_modules"] != []
        or preclose["zero_work_counters"] != ZERO_WORK_COUNTERS
        or preclose["bounded_diagnostic_counts"]
        != {
            "nvidia_smi_invocations": 2,
            "scheduler_request_artifacts": 1,
            "scheduler_handoff_artifacts": 1,
            "job_scheduler_records": 1,
            "step_scheduler_records": 1,
            "preexec_artifacts": 1,
            "preclose_artifacts": 1,
            "ready_artifacts": 0,
            "simulation_app_close_calls": 0,
        }
    ):
        raise ValueError("pre-close evidence profile mismatch")
    if not isinstance(ready, dict) or set(ready) != {
        "schema_version",
        "profile",
        "status",
        "startup_diagnostic",
        "preexec",
        "preclose",
        "zero_work_counters",
        "bounded_diagnostic_counts",
    }:
        raise ValueError("ready evidence schema mismatch")
    if (
        isinstance(ready["schema_version"], bool)
        or ready["schema_version"] != 1
        or ready["profile"] != READY_PROFILE
        or ready["status"] != "ready_for_simulation_app_close"
        or ready["startup_diagnostic"] != STARTUP_DIAGNOSTIC_MODE
        or ready["preexec"] != identities["startup_preexec.json"]
        or ready["preclose"] != identities["startup_preclose.json"]
        or ready["zero_work_counters"] != ZERO_WORK_COUNTERS
        or ready["bounded_diagnostic_counts"]
        != {
            "nvidia_smi_invocations": 2,
            "scheduler_request_artifacts": 1,
            "scheduler_handoff_artifacts": 1,
            "job_scheduler_records": 1,
            "step_scheduler_records": 1,
            "preexec_artifacts": 1,
            "preclose_artifacts": 1,
            "ready_artifacts": 1,
            "simulation_app_close_calls": 0,
        }
    ):
        raise ValueError("ready evidence profile mismatch")

    if not isinstance(preexec, dict) or not isinstance(preclose["runtime"], dict):
        raise ValueError("startup runtime evidence is missing")
    _validate_context_continuity(preexec, preclose["runtime"])
    runtime = preexec["runtime"]
    if preclose["runtime"] != runtime:
        raise ValueError("pre-exec/pre-close runtime records are not identical")
    if not isinstance(runtime, dict) or set(runtime) != {
        "process",
        "slurm",
        "nvidia_smi",
        "cgroup",
        "device_nodes",
        "output_directories",
        "execution_approvals",
        "pyxis_image",
        "source",
    }:
        raise ValueError("pre-exec runtime schema mismatch")
    approvals = _validate_execution_approval_environment(os.environ)
    if runtime["execution_approvals"] != approvals:
        raise ValueError("startup execution approvals differ from trusted helper env")
    handoff = _validate_scheduler_handoff(
        values["scheduler_handoff.json"],
        request=request,
        request_identity=identities["scheduler_request.json"],
        **_scheduler_approval_kwargs(approvals),
    )
    scheduler_runtime = runtime.get("slurm", {}).get("scheduler_handoff")
    if scheduler_runtime != {
        "artifact": identities["scheduler_handoff.json"],
        "value": handoff,
    }:
        raise ValueError("runtime scheduler handoff cross-link mismatch")
    if handoff["request_value"] != request:
        raise ValueError("scheduler request/handoff value mismatch")
    terminal_request = _validate_scheduler_terminal_request(
        values["scheduler_terminal_request.json"],
        request=request,
        request_identity=identities["scheduler_request.json"],
        handoff=handoff,
        handoff_identity=identities["scheduler_handoff.json"],
    )
    trusted_host_binding = scheduler_host_binding_value(
        environ=os.environ, request=request
    )
    if handoff["host_binding"] != trusted_host_binding:
        raise ValueError("scheduler handoff differs from trusted helper env")

    output_directories = runtime["output_directories"]
    expected_output = {
        "namespace_parent": str(task_dir.parent.parent),
        "namespace_parent_identity": expected_namespace_parent_identity,
        "run_dir": str(task_dir.parent),
        "run_identity": expected_run_identity,
        "task_dir": str(task_dir),
        "task_identity": expected_task_identity,
    }
    if (
        output_directories != expected_output
        or preclose["runtime"].get("output_directories") != expected_output
    ):
        raise ValueError("startup output-directory evidence mismatch")
    process = runtime["process"]
    slurm = runtime["slurm"]
    nvidia = runtime["nvidia_smi"]
    if (
        not isinstance(process, dict)
        or set(process) != {"pid", "ppid", "executable", "cwd", "python_argv"}
        or not isinstance(process["pid"], int)
        or isinstance(process["pid"], bool)
        or process["pid"] <= 0
        or not isinstance(process["ppid"], int)
        or isinstance(process["ppid"], bool)
        or process["ppid"] <= 0
        or process["executable"] != "/.venv/bin/python"
        or process["cwd"] != "/polaris-source"
        or not isinstance(slurm, dict)
        or not isinstance(slurm.get("job_id"), int)
        or isinstance(slurm.get("job_id"), bool)
        or slurm["job_id"] <= 0
        or not isinstance(nvidia, dict)
        or GPU_UUID_PATTERN.fullmatch(nvidia.get("uuid", "")) is None
    ):
        raise ValueError("startup process/Slurm/GPU evidence mismatch")
    expected_slurm_environment = request["slurm_environment"]
    expected_gpu_environment = {
        "CUDA_VISIBLE_DEVICES": "0",
        "NVIDIA_VISIBLE_DEVICES": request["expected_gpu_uuid"],
        "SLURM_JOB_GPUS": expected_slurm_environment["SLURM_JOB_GPUS"],
        "SLURM_STEP_GPUS": expected_slurm_environment["SLURM_STEP_GPUS"],
        "SLURM_GPUS_ON_NODE": expected_slurm_environment["SLURM_GPUS_ON_NODE"],
        "SLURM_GPUS_PER_TASK": expected_slurm_environment["SLURM_GPUS_PER_TASK"],
        "SLURM_TRES_PER_TASK": expected_slurm_environment["SLURM_TRES_PER_TASK"],
    }
    if (
        set(slurm)
        != {
            "job_id",
            "step_id",
            "job_gpu_index",
            "step_gpu_index",
            "gpu_environment",
            "gpus_on_node",
            "gpus_per_task",
            "tres_per_task",
            "tres_per_task_items",
            "scheduler_handoff",
        }
        or slurm["job_id"] != request["job_id"]
        or slurm["step_id"] != request["step_id"]
        or slurm["job_gpu_index"] != int(expected_slurm_environment["SLURM_JOB_GPUS"])
        or slurm["step_gpu_index"] != int(expected_slurm_environment["SLURM_STEP_GPUS"])
        or slurm["gpu_environment"] != expected_gpu_environment
        or slurm["gpus_on_node"] != 1
        or slurm["gpus_per_task"] != 1
        or slurm["tres_per_task"] != expected_slurm_environment["SLURM_TRES_PER_TASK"]
        or slurm["tres_per_task_items"]
        != parse_tres_per_task(expected_slurm_environment["SLURM_TRES_PER_TASK"])
    ):
        raise ValueError("startup Slurm evidence differs from scheduler request")
    if (
        set(nvidia)
        != {
            "uuid",
            "name",
            "driver_version",
            "minor_number",
            "command",
            "row_count",
        }
        or nvidia["uuid"] != request["expected_gpu_uuid"]
        or nvidia["name"] != EXPECTED_NVIDIA_GPU_NAME
        or nvidia["driver_version"] != EXPECTED_NVIDIA_DRIVER_VERSION
        or nvidia["minor_number"] != slurm["job_gpu_index"]
        or nvidia["command"] != list(NVIDIA_SMI_COMMAND)
        or nvidia["row_count"] != 1
    ):
        raise ValueError("startup NVIDIA evidence differs from the closed contract")
    cgroup = runtime["cgroup"]
    if not isinstance(cgroup, dict) or set(cgroup) != {
        "raw_sha256",
        "records",
        "job_step_path",
    }:
        raise ValueError("startup cgroup evidence schema mismatch")
    records = cgroup["records"]
    if not isinstance(records, list) or not records:
        raise ValueError("startup cgroup evidence is empty")
    raw_cgroup_lines: list[str] = []
    for record in records:
        if (
            not isinstance(record, dict)
            or set(record) != {"hierarchy", "controllers", "path"}
            or not isinstance(record["hierarchy"], int)
            or isinstance(record["hierarchy"], bool)
            or record["hierarchy"] < 0
            or not isinstance(record["controllers"], list)
            or not all(isinstance(item, str) for item in record["controllers"])
            or not isinstance(record["path"], str)
        ):
            raise ValueError("startup cgroup record schema mismatch")
        raw_cgroup_lines.append(
            f"{record['hierarchy']}:{','.join(record['controllers'])}:{record['path']}\n"
        )
    rebuilt_cgroup = parse_cgroup_text(
        "".join(raw_cgroup_lines),
        job_id=str(request["job_id"]),
        step_id=str(request["step_id"]),
    )
    if rebuilt_cgroup != cgroup:
        raise ValueError("startup cgroup evidence is not canonical")
    terminal = _validate_scheduler_terminal(
        values["scheduler_terminal.json"],
        request=request,
        request_identity=identities["scheduler_request.json"],
        handoff=handoff,
        handoff_identity=identities["scheduler_handoff.json"],
        terminal_request=terminal_request,
        terminal_request_identity=identities["scheduler_terminal_request.json"],
        expected_cgroup_path=cgroup["job_step_path"],
        approvals=approvals,
    )
    if terminal_request["srun_exit_code"] != 0 or terminal["scancel_invoked"]:
        raise ValueError("success evidence does not have a clean scheduler exit")
    device_nodes = runtime["device_nodes"]
    if not isinstance(device_nodes, dict) or set(device_nodes) != {
        "all",
        "physical",
        "physical_count",
    }:
        raise ValueError("startup device-node evidence schema mismatch")
    rebuilt_nodes = validate_device_nodes(
        device_nodes["all"], expected_minor_number=nvidia["minor_number"]
    )
    if rebuilt_nodes != device_nodes:
        raise ValueError("startup device-node evidence is not canonical")
    target = validate_target_argv(
        preexec["target_argv"],
        preexec_path=task_dir / "startup_preexec.json",
        preclose_path=task_dir / "startup_preclose.json",
        expected_gpu_uuid=request["expected_gpu_uuid"],
        expected_port=20000 + request["job_id"] % 20000,
    )
    if process["python_argv"] != python_argv_after_exec(target):
        raise ValueError("startup process argv cross-link mismatch")

    source = runtime["source"]
    if not isinstance(source, dict) or set(source) != {
        "root",
        "eval_script",
        "diagnostic_module",
        "approval",
    }:
        raise ValueError("startup source evidence schema mismatch")
    trusted_source = _validate_source_environment(os.environ)
    if source["approval"] != trusted_source:
        raise ValueError("startup source approval differs from trusted helper env")
    if source["root"] != "/polaris-source":
        raise ValueError("startup source root differs from the mounted source root")
    host_source_root = Path(__file__).resolve(strict=True).parents[2]
    current_sources = {
        "eval_script": stable_file_identity(host_source_root / "scripts/eval.py"),
        "diagnostic_module": stable_file_identity(Path(__file__).resolve(strict=True)),
    }
    for name, expected_path in (
        ("eval_script", "/polaris-source/scripts/eval.py"),
        (
            "diagnostic_module",
            "/polaris-source/src/polaris/app_launcher_startup_diagnostic.py",
        ),
    ):
        recorded = source[name]
        current = current_sources[name]
        if (
            not isinstance(recorded, dict)
            or set(recorded) != {"path", "mode", "nlink", "size", "sha256"}
            or recorded["path"] != expected_path
            or any(
                recorded[field] != current[field]
                for field in ("mode", "nlink", "size", "sha256")
            )
        ):
            raise ValueError(f"startup source identity mismatch: {name}")

    trusted_pyxis = _validate_pyxis_image_environment(os.environ)
    if runtime["pyxis_image"] != trusted_pyxis:
        raise ValueError("startup Pyxis image evidence mismatch")
    if preclose["runtime"].get("execution_approvals") != approvals:
        raise ValueError("pre-close approvals differ from trusted helper env")
    if preclose["runtime"].get("pyxis_image") != trusted_pyxis:
        raise ValueError("pre-close Pyxis evidence differs from trusted helper env")
    if preclose["runtime"].get("source", {}).get("approval") != trusted_source:
        raise ValueError("pre-close source evidence differs from trusted helper env")


def seal_evidence_tree(
    *,
    task_dir: Path,
    run_dir: Path,
    outcome: str,
    expected_namespace_parent_identity: str,
    expected_task_identity: str,
    expected_run_identity: str,
    srun_exit_code: int | None = None,
    log_exit_code: int | None = None,
    helper_exit_code: int | None = None,
) -> dict[str, Any]:
    if outcome not in {"success", "failure"}:
        raise ValueError("evidence-tree outcome must be success or failure")
    run_dir = _canonical_directory(run_dir, label="run directory")
    task_dir = _canonical_directory(task_dir, label="task directory")
    if task_dir.parent != run_dir or task_dir.name != "app_launcher_only":
        raise ValueError("task directory is not the closed AppLauncher child")
    parent_fd = os.open(
        run_dir.parent,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    run_fd = -1
    task_fd = -1
    termination_mode: str | None = None
    log_sha256: str | None = None
    try:
        _validate_expected_directory_identity(
            parent_fd,
            expected=expected_namespace_parent_identity,
            label="approved output namespace parent",
        )
        _validate_parent_binding(parent_fd, run_dir.parent)
        run_fd = os.open(
            run_dir.name,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        _validate_child_directory_binding(
            parent_fd, run_fd, name=run_dir.name, path=run_dir
        )
        _validate_expected_directory_identity(
            run_fd,
            expected=expected_run_identity,
            label="run directory",
        )
        if set(os.listdir(run_fd)) != {"app_launcher_only"}:
            raise ValueError("run directory does not have exact one-child closure")
        task_fd = os.open(
            task_dir.name,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=run_fd,
        )
        _validate_child_directory_binding(
            run_fd, task_fd, name=task_dir.name, path=task_dir
        )
        _validate_expected_directory_identity(
            task_fd,
            expected=expected_task_identity,
            label="task directory",
        )
        inventory = _task_file_inventory(task_fd, task_dir)
        expected = set(PRETERMINAL_EVIDENCE_ENTRIES)
        if outcome == "failure":
            expected = set(inventory)
            if FAILURE_ATTESTATION_NAME not in expected:
                raise ValueError("failure tree is missing its terminal attestation")
            if not expected <= PRETERMINAL_EVIDENCE_ENTRIES | {
                FAILURE_ATTESTATION_NAME
            }:
                raise ValueError("failure tree contains an unexpected entry")
        if set(inventory) != expected:
            raise ValueError("task directory entry closure mismatch")
        if outcome == "failure":
            failure_value, failure_identity = _stable_read_immutable_json_at(
                task_fd,
                path=task_dir / FAILURE_ATTESTATION_NAME,
            )
            if failure_identity != inventory[FAILURE_ATTESTATION_NAME]:
                raise ValueError("failure attestation identity changed")
            expected_failure_keys = {
                "schema_version",
                "profile",
                "status",
                "signal",
                "exit_codes",
                "process_closure",
                "primary_exit_outcome",
                "cleanup_outcome",
                "seal_intent",
                "artifacts_before_attestation",
                "zero_work_counters",
                "scientific_result",
            }
            if (
                not isinstance(failure_value, dict)
                or set(failure_value) != expected_failure_keys
                or isinstance(failure_value["schema_version"], bool)
                or failure_value["schema_version"] != 1
                or failure_value["profile"]
                != "polaris_public_eval_app_launcher_failure_v2"
                or failure_value["status"] != "non_scientific_diagnostic_failed"
                or failure_value["signal"] not in {"none", "INT", "TERM", "HUP"}
                or failure_value["zero_work_counters"] != ZERO_WORK_COUNTERS
                or failure_value["scientific_result"] is not None
                or failure_value["process_closure"]
                != (
                    "scheduler_step_terminal_cgroup_unpopulated_and_"
                    "tracked_groups_absent_before_attestation"
                )
                or failure_value["primary_exit_outcome"] != "nonzero"
                or failure_value["cleanup_outcome"] != "completed"
                or failure_value["seal_intent"] != "terminal_seal_required"
                or not isinstance(failure_value["exit_codes"], dict)
                or set(failure_value["exit_codes"])
                != {
                    "primary_exit_code",
                    "srun_exit_code",
                    "log_exit_code",
                    "helper_exit_code",
                }
                or any(
                    not isinstance(code, int)
                    or isinstance(code, bool)
                    or not 0 <= code <= 255
                    for code in failure_value["exit_codes"].values()
                )
                or failure_value["exit_codes"]["primary_exit_code"] == 0
            ):
                raise ValueError("failure attestation schema mismatch")
            before_attestation = {
                name: identity
                for name, identity in inventory.items()
                if name != FAILURE_ATTESTATION_NAME
            }
            if failure_value["artifacts_before_attestation"] != before_attestation:
                raise ValueError("failure evidence changed after terminal attestation")
        else:
            pre_seal_exit_codes = {
                "srun_exit_code": srun_exit_code,
                "log_exit_code": log_exit_code,
                "helper_exit_code": helper_exit_code,
            }
            if any(
                not isinstance(code, int) or isinstance(code, bool) or code != 0
                for code in pre_seal_exit_codes.values()
            ):
                raise ValueError(
                    "success sealing requires worker-supplied zero pre-seal exit codes"
                )
            _validate_success_evidence_semantics(
                task_fd=task_fd,
                task_dir=task_dir,
                inventory=inventory,
                expected_namespace_parent_identity=(expected_namespace_parent_identity),
                expected_run_identity=expected_run_identity,
                expected_task_identity=expected_task_identity,
            )
            identity_value, _ = _stable_read_immutable_json_at(
                task_fd,
                path=task_dir / "app_launcher_only.log.identity.json",
            )
            log_payload, log_identity = _stable_file_at(
                task_fd,
                name="app_launcher_only.log",
                path=task_dir / "app_launcher_only.log",
                expected_mode=0o444,
            )
            if (
                not isinstance(identity_value, dict)
                or set(identity_value) != {"schema_version", "profile", "status", "log"}
                or identity_value["schema_version"] != 1
                or identity_value["profile"] != LOG_IDENTITY_PROFILE
                or identity_value["status"] != "immutable_log_finalized"
                or identity_value["log"] != log_identity
            ):
                raise ValueError("sealed task log identity mismatch")
            try:
                log_text = log_payload.decode("utf-8", errors="strict")
            except UnicodeDecodeError as error:
                raise ValueError("immutable evaluator log is not UTF-8") from error
            if not log_text.endswith("\n"):
                raise ValueError("immutable evaluator log is not newline terminated")
            log_lines = log_text.splitlines()
            if any(
                line.startswith("POLARIS_STARTUP_DIAGNOSTIC_CLOSE_ERROR=")
                for line in log_lines
            ):
                raise ValueError("evaluator log reports a SimulationApp close error")
            phase_lines = [
                line for line in log_lines if line.startswith("POLARIS_EVAL_PHASE=")
            ]
            required_phases = [
                "POLARIS_EVAL_PHASE=before_app_launcher",
                "POLARIS_EVAL_PHASE=after_app_launcher",
                "POLARIS_EVAL_PHASE=before_app_launcher_diagnostic_close",
            ]
            if phase_lines == required_phases:
                termination_mode = "process_exited_zero_before_postclose_marker"
            elif phase_lines == [
                *required_phases,
                "POLARIS_EVAL_PHASE=after_app_launcher_diagnostic_close",
            ]:
                termination_mode = "simulation_app_close_returned"
            else:
                raise ValueError(
                    "evaluator log phase sequence is not the closed schema"
                )
            log_sha256 = log_identity["sha256"]
            preterminal_value = {
                "schema_version": 1,
                "profile": "polaris_public_eval_app_launcher_preterminal_v1",
                "status": "awaiting_external_allocation_terminal_attestation",
                "authoritative_completion": False,
                "termination_mode": termination_mode,
                "log": log_identity,
                "pre_seal_worker_exit_codes": pre_seal_exit_codes,
                "pre_seal_exit_claim": (
                    "worker_supplied_zero_codes_validated_before_terminal_seal"
                ),
                "final_process_exit": (
                    "outside_attestation_scope_outer_shell_owns_final_exit"
                ),
                "required_external_terminal_contract": (
                    "allocation_COMPLETED_exit_0:0_Restarts_0"
                ),
                "artifacts_before_attestation": dict(inventory),
                "source_approval": _validate_source_environment(os.environ),
                "execution_approvals": _validate_execution_approval_environment(
                    os.environ
                ),
                "pyxis_image": _validate_pyxis_image_environment(os.environ),
                "zero_work_counters": dict(ZERO_WORK_COUNTERS),
                "scientific_result": None,
                "seal_intent": "terminal_mode0555_closure_required",
            }
            preterminal_identity = _publish_immutable_json_at(
                task_fd,
                parent=task_dir,
                path=task_dir / PRETERMINAL_ATTESTATION_NAME,
                value=preterminal_value,
            )
            inventory = _task_file_inventory(task_fd, task_dir)
            if set(inventory) != PRETERMINAL_TASK_ENTRIES:
                raise RuntimeError(
                    "preterminal attestation did not close the task tree"
                )
            reread_preterminal_value, reread_preterminal_identity = (
                _stable_read_immutable_json_at(
                    task_fd,
                    path=task_dir / PRETERMINAL_ATTESTATION_NAME,
                )
            )
            if (
                inventory[PRETERMINAL_ATTESTATION_NAME] != preterminal_identity
                or reread_preterminal_identity != preterminal_identity
                or reread_preterminal_value != preterminal_value
            ):
                raise RuntimeError(
                    "preterminal attestation changed before terminal directory seal"
                )
        os.fchmod(task_fd, 0o555)
        os.fsync(task_fd)
        os.fchmod(run_fd, 0o555)
        os.fsync(run_fd)
        os.fsync(parent_fd)
        _validate_child_directory_binding(
            run_fd, task_fd, name=task_dir.name, path=task_dir
        )
        _validate_child_directory_binding(
            parent_fd, run_fd, name=run_dir.name, path=run_dir
        )
        _validate_parent_binding(parent_fd, run_dir.parent)
        expected_task_sealed = expected_task_identity.rsplit(":", 1)[0] + ":0555"
        expected_run_sealed = expected_run_identity.rsplit(":", 1)[0] + ":0555"
        _validate_expected_directory_identity(
            task_fd,
            expected=expected_task_sealed,
            label="sealed task directory",
        )
        _validate_expected_directory_identity(
            run_fd,
            expected=expected_run_sealed,
            label="sealed run directory",
        )
        if stat.S_IMODE(os.fstat(task_fd).st_mode) != 0o555:
            raise RuntimeError("task directory was not sealed mode 0555")
        if stat.S_IMODE(os.fstat(run_fd).st_mode) != 0o555:
            raise RuntimeError("run directory was not sealed mode 0555")
        if set(os.listdir(run_fd)) != {"app_launcher_only"}:
            raise RuntimeError("sealed run directory closure changed")
        final_inventory = _task_file_inventory(task_fd, task_dir)
        if final_inventory != inventory:
            raise RuntimeError("sealed task directory closure changed")
        result = {
            "outcome": outcome,
            "closure_scope": "point_in_time_same_uid_mutable",
            "namespace_parent": str(run_dir.parent),
            "namespace_parent_identity": expected_namespace_parent_identity,
            "run_dir": str(run_dir),
            "run_mode": "0555",
            "task_dir": str(task_dir),
            "task_mode": "0555",
            "artifacts": final_inventory,
        }
        if outcome == "success":
            if termination_mode is None or log_sha256 is None:
                raise AssertionError("success log semantics were not retained")
            result["termination_mode"] = termination_mode
            result["log_sha256"] = log_sha256
        return result
    finally:
        if task_fd >= 0:
            os.close(task_fd)
        if run_fd >= 0:
            os.close(run_fd)
        os.close(parent_fd)


def parse_nvidia_smi_output(output: str) -> dict[str, Any]:
    rows = [line for line in output.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one NVIDIA GPU row, found {len(rows)}")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 4:
        raise ValueError("malformed nvidia-smi row")
    uuid, name, driver_version, minor_number = fields
    if GPU_UUID_PATTERN.fullmatch(uuid) is None:
        raise ValueError("malformed NVIDIA GPU UUID")
    if name != EXPECTED_NVIDIA_GPU_NAME:
        raise ValueError("NVIDIA GPU name mismatch")
    if driver_version != EXPECTED_NVIDIA_DRIVER_VERSION:
        raise ValueError("NVIDIA driver version mismatch")
    if DECIMAL_PATTERN.fullmatch(minor_number) is None:
        raise ValueError("malformed NVIDIA GPU minor number")
    return {
        "uuid": uuid,
        "name": name,
        "driver_version": driver_version,
        "minor_number": int(minor_number),
        "command": list(NVIDIA_SMI_COMMAND),
        "row_count": 1,
    }


def _run_nvidia_smi() -> str:
    completed = subprocess.run(
        NVIDIA_SMI_COMMAND,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.stderr:
        raise ValueError("nvidia-smi emitted unexpected stderr")
    return completed.stdout


def parse_cgroup_text(raw: str, *, job_id: str, step_id: str) -> dict[str, Any]:
    if not raw or not raw.endswith("\n"):
        raise ValueError("/proc/self/cgroup must be nonempty and newline terminated")
    records = []
    seen_lines = set()
    matching_paths = set()
    job_pattern = re.compile(rf"(?:^|/)job[_-]{re.escape(job_id)}(?:$|/)")
    step_pattern = re.compile(
        rf"(?:^|/)step[_-]{re.escape(step_id)}(?:\.scope)?(?:$|/)"
    )
    any_job_token = re.compile(r"(?:^|/)job[_-][0-9]+(?:$|/)")
    any_step_token = re.compile(
        r"(?:^|/)step[_-](?:[0-9]+|batch|extern)(?:\.scope)?(?:$|/)"
    )
    for line in raw.splitlines():
        if line in seen_lines:
            raise ValueError("duplicate cgroup record")
        seen_lines.add(line)
        fields = line.split(":", 2)
        if len(fields) != 3 or DECIMAL_PATTERN.fullmatch(fields[0]) is None:
            raise ValueError("malformed cgroup record")
        hierarchy, controllers, path = fields
        pure = PurePosixPath(path)
        if not path.startswith("/") or pure.as_posix() != path or ".." in pure.parts:
            raise ValueError("noncanonical cgroup path")
        job_match = job_pattern.search(path) is not None
        step_match = step_pattern.search(path) is not None
        if (any_job_token.search(path) is not None) != job_match:
            raise ValueError("cgroup job identity mismatch")
        if (any_step_token.search(path) is not None) != step_match:
            raise ValueError("cgroup step identity mismatch")
        if job_match != step_match:
            raise ValueError("cgroup job/step evidence is incomplete")
        if job_match:
            matching_paths.add(path)
        records.append(
            {
                "hierarchy": int(hierarchy),
                "controllers": controllers.split(",") if controllers else [],
                "path": path,
            }
        )
    if len(matching_paths) != 1:
        raise ValueError(
            f"expected exactly one cgroup job/step path, found {len(matching_paths)}"
        )
    return {
        "raw_sha256": _sha256(raw.encode("utf-8")),
        "records": records,
        "job_step_path": next(iter(matching_paths)),
    }


def capture_device_nodes() -> list[dict[str, Any]]:
    records = []
    for path in sorted(Path("/dev").glob("nvidia*"), key=lambda item: item.name):
        metadata = path.lstat()
        records.append(
            {
                "path": str(path),
                "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
                "file_type": "character" if stat.S_ISCHR(metadata.st_mode) else "other",
                "device_major": os.major(metadata.st_rdev)
                if stat.S_ISCHR(metadata.st_mode)
                else None,
                "device_minor": os.minor(metadata.st_rdev)
                if stat.S_ISCHR(metadata.st_mode)
                else None,
            }
        )
    return records


def validate_device_nodes(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_minor_number: int,
) -> dict[str, Any]:
    canonical = []
    physical = []
    seen_paths = set()
    for raw in records:
        if set(raw) != {
            "path",
            "mode",
            "file_type",
            "device_major",
            "device_minor",
        }:
            raise ValueError("device-node schema mismatch")
        path = raw["path"]
        if not isinstance(path, str) or path in seen_paths:
            raise ValueError("duplicate or malformed NVIDIA device path")
        seen_paths.add(path)
        match = re.fullmatch(r"/dev/nvidia([0-9]+)", path)
        record = dict(raw)
        canonical.append(record)
        if match is not None:
            if (
                record["file_type"] != "character"
                or record["device_major"] != 195
                or record["device_minor"] != int(match.group(1))
            ):
                raise ValueError("malformed physical NVIDIA device node")
            physical.append(record)
    if len(physical) != 1:
        raise ValueError(
            f"expected exactly one physical NVIDIA device node, found {len(physical)}"
        )
    if physical[0]["device_minor"] != expected_minor_number:
        raise ValueError("NVIDIA device node does not match nvidia-smi minor number")
    return {"all": canonical, "physical": physical, "physical_count": 1}


def _required_environment(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing required environment variable: {name}")
    return value


def _submission_transaction_id(environ: Mapping[str, str]) -> str:
    value = _required_environment(environ, "SUBMISSION_TRANSACTION_ID")
    if SUBMISSION_TRANSACTION_PATTERN.fullmatch(value) is None:
        raise ValueError("submission transaction ID is malformed")
    return value


def _validate_source_environment(environ: Mapping[str, str]) -> dict[str, str]:
    values = {
        name: _required_environment(environ, name)
        for name in SOURCE_IDENTITY_ENVIRONMENT
    }
    if (
        SHA256_PATTERN.fullmatch(values["BATCH_VERIFIED_POLARIS_SOURCE_TREE_SHA256"])
        is None
    ):
        raise ValueError("invalid source-tree SHA-256")
    if SHA256_PATTERN.fullmatch(values["SOURCE_APPROVAL_SHA256"]) is None:
        raise ValueError("invalid source-approval SHA-256")
    if re.fullmatch(r"[0-9a-f]{40}", values["POLARIS_IMPLEMENTATION_COMMIT"]) is None:
        raise ValueError("invalid implementation commit")
    return values


def _validate_execution_approval_environment(
    environ: Mapping[str, str],
) -> dict[str, str]:
    values = {
        name: _required_environment(environ, name)
        for name in EXECUTION_APPROVAL_ENVIRONMENT
    }
    for name in EXECUTION_APPROVAL_SHA256_ENVIRONMENT:
        value = values[name]
        if SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"invalid external execution approval digest: {name}")
    for name in EXECUTION_APPROVAL_SIZE_ENVIRONMENT:
        value = values[name]
        if DECIMAL_PATTERN.fullmatch(value) is None or value == "0":
            raise ValueError(f"invalid external execution approval size: {name}")
    return values


def _scheduler_approval_kwargs(approvals: Mapping[str, str]) -> dict[str, Any]:
    values = _validate_execution_approval_environment(approvals)
    return {
        "expected_slurm_config_sha256": values["POLARIS_EXPECTED_SLURM_CONFIG_SHA256"],
        "runtime_closure_approval_sha256": values[
            "POLARIS_RUNTIME_CLOSURE_APPROVAL_SHA256"
        ],
        "expected_scontrol_sha256": values["POLARIS_EXPECTED_SCONTROL_SHA256"],
        "expected_scontrol_size": int(values["POLARIS_EXPECTED_SCONTROL_SIZE"]),
        "expected_slurm_library_sha256": values[
            "POLARIS_EXPECTED_SLURM_LIBRARY_SHA256"
        ],
        "expected_slurm_library_size": int(
            values["POLARIS_EXPECTED_SLURM_LIBRARY_SIZE"]
        ),
        "expected_sacct_sha256": values["POLARIS_EXPECTED_SACCT_SHA256"],
        "expected_sacct_size": int(values["POLARIS_EXPECTED_SACCT_SIZE"]),
        "expected_scancel_sha256": values["POLARIS_EXPECTED_SCANCEL_SHA256"],
        "expected_scancel_size": int(values["POLARIS_EXPECTED_SCANCEL_SIZE"]),
        "expected_srun_sha256": values["POLARIS_EXPECTED_SRUN_SHA256"],
        "expected_srun_size": int(values["POLARIS_EXPECTED_SRUN_SIZE"]),
    }


def capture_pyxis_image_identity(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    if SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise ValueError("expected Pyxis image SHA-256 is malformed")
    identity = stable_file_identity(path)
    if identity["sha256"] != expected_sha256:
        raise ValueError("Pyxis image SHA-256 mismatch")
    return {"expected_sha256": expected_sha256, "observed": identity}


def _validate_pyxis_image_environment(environ: Mapping[str, str]) -> dict[str, Any]:
    values = {
        name: _required_environment(environ, name) for name in PYXIS_IMAGE_ENVIRONMENT
    }
    path = values["POLARIS_PYXIS_IMAGE_PATH"]
    pure_path = PurePosixPath(path)
    if (
        not path.startswith("/")
        or pure_path.as_posix() != path
        or ".." in pure_path.parts
    ):
        raise ValueError("Pyxis image path is not canonical absolute syntax")
    expected_sha256 = values["POLARIS_EXPECTED_PYXIS_IMAGE_SHA256"]
    observed_sha256 = values["POLARIS_OBSERVED_PYXIS_IMAGE_SHA256"]
    if (
        SHA256_PATTERN.fullmatch(expected_sha256) is None
        or observed_sha256 != expected_sha256
    ):
        raise ValueError("Pyxis image expected/observed SHA-256 mismatch")
    mode = values["POLARIS_OBSERVED_PYXIS_IMAGE_MODE"]
    nlink = values["POLARIS_OBSERVED_PYXIS_IMAGE_NLINK"]
    size = values["POLARIS_OBSERVED_PYXIS_IMAGE_SIZE"]
    if (
        re.fullmatch(r"[0-7]{4}", mode) is None
        or nlink != "1"
        or DECIMAL_PATTERN.fullmatch(size) is None
        or size == "0"
    ):
        raise ValueError("Pyxis image observed identity is malformed")
    return {
        "path": path,
        "expected_sha256": expected_sha256,
        "observed": {
            "path": path,
            "mode": mode,
            "nlink": 1,
            "size": int(size),
            "sha256": observed_sha256,
        },
    }


def _validate_pinned_file_identity(
    identity: Mapping[str, Any],
    *,
    path: Path,
    mode: str,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> dict[str, Any]:
    if (
        not isinstance(identity, dict)
        or set(identity) != {"path", "mode", "nlink", "size", "sha256"}
        or identity["path"] != str(path)
        or identity["mode"] != mode
        or identity["nlink"] != 1
        or not isinstance(identity["size"], int)
        or identity["size"] <= 0
        or (expected_size is not None and identity["size"] != expected_size)
        or not isinstance(identity["sha256"], str)
        or SHA256_PATTERN.fullmatch(identity["sha256"]) is None
        or (expected_sha256 is not None and identity["sha256"] != expected_sha256)
    ):
        raise ValueError(f"approved scheduler file identity mismatch: {path}")
    return dict(identity)


def validate_scheduler_client_identity(
    value: Any,
    *,
    expected_slurm_config_sha256: str,
    runtime_closure_approval_sha256: str,
    expected_scontrol_sha256: str,
    expected_scontrol_size: int,
    expected_slurm_library_sha256: str,
    expected_slurm_library_size: int,
    expected_sacct_sha256: str,
    expected_sacct_size: int,
    expected_scancel_sha256: str,
    expected_scancel_size: int,
    expected_srun_sha256: str,
    expected_srun_size: int,
) -> dict[str, Any]:
    if (
        SHA256_PATTERN.fullmatch(expected_slurm_config_sha256) is None
        or SHA256_PATTERN.fullmatch(runtime_closure_approval_sha256) is None
        or SHA256_PATTERN.fullmatch(expected_scontrol_sha256) is None
        or SHA256_PATTERN.fullmatch(expected_slurm_library_sha256) is None
        or SHA256_PATTERN.fullmatch(expected_sacct_sha256) is None
        or SHA256_PATTERN.fullmatch(expected_scancel_sha256) is None
        or SHA256_PATTERN.fullmatch(expected_srun_sha256) is None
        or not isinstance(expected_scontrol_size, int)
        or isinstance(expected_scontrol_size, bool)
        or expected_scontrol_size <= 0
        or not isinstance(expected_slurm_library_size, int)
        or isinstance(expected_slurm_library_size, bool)
        or expected_slurm_library_size <= 0
        or not isinstance(expected_sacct_size, int)
        or isinstance(expected_sacct_size, bool)
        or expected_sacct_size <= 0
        or not isinstance(expected_scancel_size, int)
        or isinstance(expected_scancel_size, bool)
        or expected_scancel_size <= 0
        or not isinstance(expected_srun_size, int)
        or isinstance(expected_srun_size, bool)
        or expected_srun_size <= 0
    ):
        raise ValueError("external scheduler approval digest is malformed")
    if not isinstance(value, dict) or set(value) != {
        "profile",
        "scontrol",
        "sacct",
        "scancel",
        "srun",
        "slurm_library",
        "slurm_config",
        "execution_environment",
        "runtime_closure_approval_sha256",
    }:
        raise ValueError("scheduler client identity schema mismatch")
    if value["profile"] != "polaris_approved_scontrol_24_11_v2":
        raise ValueError("scheduler client identity profile mismatch")
    if value["runtime_closure_approval_sha256"] != runtime_closure_approval_sha256:
        raise ValueError("scheduler runtime-closure approval mismatch")
    expected_environment = {
        "PATH": "/usr/bin:/bin",
        "SLURM_CONF": str(EXPECTED_SLURM_CONFIG_PATH),
        "LD_LIBRARY_PATH": (
            "/cm/local/apps/slurm/24.11/lib64:/cm/local/apps/slurm/24.11/lib64/slurm"
        ),
    }
    if value["execution_environment"] != expected_environment:
        raise ValueError("scheduler client execution environment mismatch")
    return {
        "profile": value["profile"],
        "runtime_closure_approval_sha256": runtime_closure_approval_sha256,
        "scontrol": _validate_pinned_file_identity(
            value["scontrol"],
            path=EXPECTED_SCONTROL_PATH,
            mode="0755",
            expected_sha256=expected_scontrol_sha256,
            expected_size=expected_scontrol_size,
        ),
        "sacct": _validate_pinned_file_identity(
            value["sacct"],
            path=EXPECTED_SACCT_PATH,
            mode="0755",
            expected_sha256=expected_sacct_sha256,
            expected_size=expected_sacct_size,
        ),
        "scancel": _validate_pinned_file_identity(
            value["scancel"],
            path=EXPECTED_SCANCEL_PATH,
            mode="0755",
            expected_sha256=expected_scancel_sha256,
            expected_size=expected_scancel_size,
        ),
        "srun": _validate_pinned_file_identity(
            value["srun"],
            path=EXPECTED_SRUN_PATH,
            mode="0755",
            expected_sha256=expected_srun_sha256,
            expected_size=expected_srun_size,
        ),
        "slurm_library": _validate_pinned_file_identity(
            value["slurm_library"],
            path=EXPECTED_SLURM_LIBRARY_PATH,
            mode="0644",
            expected_sha256=expected_slurm_library_sha256,
            expected_size=expected_slurm_library_size,
        ),
        "slurm_config": _validate_pinned_file_identity(
            value["slurm_config"],
            path=EXPECTED_SLURM_CONFIG_PATH,
            mode="0644",
            expected_sha256=expected_slurm_config_sha256,
        ),
        "execution_environment": expected_environment,
    }


def capture_scheduler_client_identity(
    *,
    expected_slurm_config_sha256: str,
    runtime_closure_approval_sha256: str,
    expected_scontrol_sha256: str,
    expected_scontrol_size: int,
    expected_slurm_library_sha256: str,
    expected_slurm_library_size: int,
    expected_sacct_sha256: str,
    expected_sacct_size: int,
    expected_scancel_sha256: str,
    expected_scancel_size: int,
    expected_srun_sha256: str,
    expected_srun_size: int,
) -> dict[str, Any]:
    return validate_scheduler_client_identity(
        {
            "profile": "polaris_approved_scontrol_24_11_v2",
            "runtime_closure_approval_sha256": runtime_closure_approval_sha256,
            "scontrol": stable_file_identity(EXPECTED_SCONTROL_PATH),
            "sacct": stable_file_identity(EXPECTED_SACCT_PATH),
            "scancel": stable_file_identity(EXPECTED_SCANCEL_PATH),
            "srun": stable_file_identity(EXPECTED_SRUN_PATH),
            "slurm_library": stable_file_identity(EXPECTED_SLURM_LIBRARY_PATH),
            "slurm_config": stable_file_identity(EXPECTED_SLURM_CONFIG_PATH),
            "execution_environment": {
                "PATH": "/usr/bin:/bin",
                "SLURM_CONF": str(EXPECTED_SLURM_CONFIG_PATH),
                "LD_LIBRARY_PATH": (
                    "/cm/local/apps/slurm/24.11/lib64:"
                    "/cm/local/apps/slurm/24.11/lib64/slurm"
                ),
            },
        },
        expected_slurm_config_sha256=expected_slurm_config_sha256,
        runtime_closure_approval_sha256=runtime_closure_approval_sha256,
        expected_scontrol_sha256=expected_scontrol_sha256,
        expected_scontrol_size=expected_scontrol_size,
        expected_slurm_library_sha256=expected_slurm_library_sha256,
        expected_slurm_library_size=expected_slurm_library_size,
        expected_sacct_sha256=expected_sacct_sha256,
        expected_sacct_size=expected_sacct_size,
        expected_scancel_sha256=expected_scancel_sha256,
        expected_scancel_size=expected_scancel_size,
        expected_srun_sha256=expected_srun_sha256,
        expected_srun_size=expected_srun_size,
    )


def _validate_slurm_gpu_environment(environ: Mapping[str, str]) -> dict[str, str]:
    values = {
        name: _required_environment(environ, name)
        for name in (
            "SLURM_JOB_ID",
            "SLURM_STEP_ID",
            "SLURM_JOB_GPUS",
            "SLURM_STEP_GPUS",
            "SLURM_GPUS_ON_NODE",
            "SLURM_GPUS_PER_TASK",
            "SLURM_TRES_PER_TASK",
            "SLURM_CPUS_PER_TASK",
            "SLURM_NTASKS",
            "SLURM_JOB_NUM_NODES",
            "SLURM_MEM_PER_NODE",
            "SLURM_JOB_ACCOUNT",
            "SLURM_JOB_PARTITION",
            "SLURM_JOB_QOS",
            "SLURM_JOB_USER",
        )
    }
    if (
        DECIMAL_PATTERN.fullmatch(values["SLURM_JOB_ID"]) is None
        or values["SLURM_JOB_ID"] == "0"
    ):
        raise ValueError("SLURM_JOB_ID must be one positive decimal job ID")
    if DECIMAL_PATTERN.fullmatch(values["SLURM_STEP_ID"]) is None:
        raise ValueError("SLURM_STEP_ID must be one numeric srun step ID")
    job_gpus = values["SLURM_JOB_GPUS"]
    step_gpus = values["SLURM_STEP_GPUS"]
    if (
        DECIMAL_PATTERN.fullmatch(job_gpus) is None
        or DECIMAL_PATTERN.fullmatch(step_gpus) is None
        or job_gpus != step_gpus
    ):
        raise ValueError("Slurm job/step GPU indices must be one matching index")
    if values["SLURM_GPUS_ON_NODE"] != "1":
        raise ValueError("SLURM_GPUS_ON_NODE must prove exactly one GPU")
    if values["SLURM_GPUS_PER_TASK"] != "1":
        raise ValueError("SLURM_GPUS_PER_TASK must prove exactly one GPU")
    expected_values = {
        "SLURM_CPUS_PER_TASK": "16",
        "SLURM_NTASKS": "1",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_MEM_PER_NODE": "131072",
        "SLURM_JOB_ACCOUNT": "nvr_lpr_rvp",
        "SLURM_JOB_PARTITION": "batch",
        "SLURM_JOB_QOS": "normal",
        "SLURM_JOB_USER": "lzha",
    }
    for name, expected in expected_values.items():
        if values[name] != expected:
            raise ValueError(f"{name} must be exactly {expected}")
    parse_tres_per_task(values["SLURM_TRES_PER_TASK"])
    return values


def parse_tres_per_task(value: str) -> dict[str, str]:
    """Parse the two observed closed Slurm per-task encodings."""

    if value not in {"cpu=16", "cpu=16,gres/gpu:1"}:
        raise ValueError(
            "SLURM_TRES_PER_TASK must be cpu=16 with at most one GPU clause"
        )
    parsed = {"cpu": "16"}
    if value.endswith(",gres/gpu:1"):
        parsed["gres/gpu"] = "1"
    return parsed


def _parse_tres_items(value: str, *, label: str) -> dict[str, str]:
    """Parse one scheduler TRES assignment list without normalization."""

    if (
        not isinstance(value, str)
        or not value
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{label} must be one nonempty scheduler TRES value")
    parsed: dict[str, str] = {}
    for item in value.split(","):
        key, separator, item_value = item.partition("=")
        if (
            not separator
            or re.fullmatch(r"[A-Za-z0-9_./:-]+", key) is None
            or re.fullmatch(r"[A-Za-z0-9_./:+-]+", item_value) is None
            or key in parsed
        ):
            raise ValueError(f"malformed or duplicate {label} item")
        parsed[key] = item_value
    if any(key.startswith("gres/gpu:") for key in parsed):
        raise ValueError(f"{label} must not contain an additional typed GPU TRES")
    return parsed


def parse_allocated_tres(value: str, *, label: str) -> dict[str, str]:
    parsed = _parse_tres_items(value, label=label)
    expected = {
        "billing": "1",
        "cpu": "16",
        "gres/gpu": "1",
        "mem": "128G",
        "node": "1",
    }
    if parsed != expected:
        raise ValueError(f"{label} differs from the closed one-GPU job TRES shape")
    return parsed


def parse_step_allocated_tres(value: str, *, label: str) -> dict[str, str]:
    parsed = _parse_tres_items(value, label=label)
    allowed_keysets = {
        frozenset({"cpu", "gres/gpu", "node"}),
        frozenset({"cpu", "gres/gpu", "mem", "node"}),
        frozenset({"billing", "cpu", "gres/gpu", "mem", "node"}),
    }
    if frozenset(parsed) not in allowed_keysets:
        raise ValueError(f"{label} has an unapproved exact keyset")
    if parsed["gres/gpu"] != "1":
        raise ValueError(f"{label} must allocate exactly one generic GPU")
    if parsed["cpu"] != "16":
        raise ValueError(f"{label} must allocate exactly the requested 16 CPUs")
    if parsed["node"] != "1":
        raise ValueError(f"{label} must allocate exactly one node")
    if "billing" in parsed and parsed["billing"] != "1":
        raise ValueError(f"{label} billing must be exactly one")
    if "mem" in parsed and parsed["mem"] != "128G":
        raise ValueError(f"{label} memory must be exactly 128G when present")
    return parsed


def _record_field(raw: str, name: str, *, label: str) -> str:
    if (
        not isinstance(raw, str)
        or not raw
        or "\n" in raw
        or "\r" in raw
        or "\x00" in raw
    ):
        raise ValueError(f"{label} must be exactly one nonempty scontrol record")
    matches = re.findall(rf"(?:^| ){re.escape(name)}=([^ ]+)", raw)
    if len(matches) != 1:
        raise ValueError(f"{label} must contain exactly one {name}")
    return matches[0]


def parse_job_scheduler_record(
    raw: str,
    *,
    expected_job_id: int,
    expected_transaction_id: str,
) -> dict[str, Any]:
    if SUBMISSION_TRANSACTION_PATTERN.fullmatch(expected_transaction_id) is None:
        raise ValueError("expected submission transaction ID is malformed")
    job_id = _record_field(raw, "JobId", label="job scheduler record")
    if job_id != str(expected_job_id):
        raise ValueError("job scheduler record identity mismatch")
    requested = _record_field(raw, "ReqTRES", label="job scheduler record")
    allocated = _record_field(raw, "AllocTRES", label="job scheduler record")
    fields = {
        name: _record_field(raw, name, label="job scheduler record")
        for name in (
            "UserId",
            "Account",
            "QOS",
            "JobState",
            "Partition",
            "NodeList",
            "BatchHost",
            "NumNodes",
            "NumCPUs",
            "NumTasks",
            "CPUs/Task",
            "OverSubscribe",
            "TresPerNode",
            "TresPerTask",
            "Requeue",
            "Restarts",
            "Comment",
        )
    }
    expected_fields = {
        "UserId": "lzha(158351)",
        "Account": "nvr_lpr_rvp",
        "QOS": "normal",
        "JobState": "RUNNING",
        "Partition": "batch",
        "NumNodes": "1",
        "NumCPUs": "16",
        "NumTasks": "1",
        "CPUs/Task": "16",
        "OverSubscribe": "OK",
        "TresPerNode": "gres/gpu:1",
        "TresPerTask": "cpu=16",
        "Requeue": "0",
        "Restarts": "0",
        "Comment": expected_transaction_id,
    }
    for name, expected in expected_fields.items():
        if fields[name] != expected:
            raise ValueError(f"job scheduler field {name} mismatch")
    if (
        re.fullmatch(r"pool0-[0-9]+", fields["NodeList"]) is None
        or fields["BatchHost"] != fields["NodeList"]
    ):
        raise ValueError("job scheduler node identity mismatch")
    return {
        "command": [
            str(EXPECTED_SCONTROL_PATH),
            "show",
            "job",
            "--oneliner",
            job_id,
        ],
        "raw": raw,
        "raw_sha256": _sha256(raw.encode("utf-8")),
        "job_id": expected_job_id,
        "fields": fields,
        "requested_tres": requested,
        "requested_tres_items": parse_allocated_tres(
            requested, label="job requested TRES"
        ),
        "allocated_tres": allocated,
        "allocated_tres_items": parse_allocated_tres(
            allocated, label="job allocated TRES"
        ),
    }


def parse_step_scheduler_record(
    raw: str,
    *,
    expected_job_id: int,
    expected_step_id: int,
    requested_tres_per_task: str,
    expected_node: str,
) -> dict[str, Any]:
    expected = f"{expected_job_id}.{expected_step_id}"
    step_id = _record_field(raw, "StepId", label="step scheduler record")
    if step_id != expected:
        raise ValueError("step scheduler record identity mismatch")
    allocated = _record_field(raw, "TRES", label="step scheduler record")
    fields = {
        name: _record_field(raw, name, label="step scheduler record")
        for name in (
            "UserId",
            "State",
            "Partition",
            "Nodes",
            "NodeList",
            "CPUs",
            "Tasks",
            "Name",
        )
    }
    expected_fields = {
        "UserId": "lzha(158351)",
        "State": "RUNNING",
        "Partition": "batch",
        "Nodes": "1",
        "NodeList": expected_node,
        "Tasks": "1",
        "Name": "python",
    }
    for name, expected_value in expected_fields.items():
        if fields[name] != expected_value:
            raise ValueError(f"step scheduler field {name} mismatch")
    if fields["CPUs"] != "16":
        raise ValueError("step scheduler CPUs must equal the requested 16")
    return {
        "command": [
            str(EXPECTED_SCONTROL_PATH),
            "show",
            "step",
            "--oneliner",
            expected,
        ],
        "raw": raw,
        "raw_sha256": _sha256(raw.encode("utf-8")),
        "job_id": expected_job_id,
        "step_id": expected_step_id,
        "fields": fields,
        "requested_tres": requested_tres_per_task,
        "requested_tres_items": parse_tres_per_task(requested_tres_per_task),
        "requested_tres_source": "SLURM_TRES_PER_TASK",
        "allocated_tres": allocated,
        "allocated_tres_items": parse_step_allocated_tres(
            allocated, label="step allocated TRES"
        ),
    }


def scheduler_request_value(
    *, environ: Mapping[str, str], expected_gpu_uuid: str
) -> dict[str, Any]:
    slurm = _validate_slurm_gpu_environment(environ)
    if GPU_UUID_PATTERN.fullmatch(expected_gpu_uuid) is None:
        raise ValueError("invalid expected batch GPU UUID")
    return {
        "schema_version": 1,
        "profile": SCHEDULER_REQUEST_PROFILE,
        "status": "awaiting_host_scheduler_records",
        "job_id": int(slurm["SLURM_JOB_ID"]),
        "step_id": int(slurm["SLURM_STEP_ID"]),
        "expected_gpu_uuid": expected_gpu_uuid,
        "submission_transaction_id": _submission_transaction_id(environ),
        "slurm_environment": slurm,
    }


def _validate_scheduler_request(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "profile",
        "status",
        "job_id",
        "step_id",
        "expected_gpu_uuid",
        "submission_transaction_id",
        "slurm_environment",
    }:
        raise ValueError("scheduler request schema mismatch")
    if (
        value["schema_version"] != 1
        or value["profile"] != SCHEDULER_REQUEST_PROFILE
        or value["status"] != "awaiting_host_scheduler_records"
        or not isinstance(value["job_id"], int)
        or isinstance(value["job_id"], bool)
        or value["job_id"] <= 0
        or not isinstance(value["step_id"], int)
        or isinstance(value["step_id"], bool)
        or value["step_id"] < 0
        or GPU_UUID_PATTERN.fullmatch(value["expected_gpu_uuid"]) is None
        or not isinstance(value["submission_transaction_id"], str)
        or SUBMISSION_TRANSACTION_PATTERN.fullmatch(value["submission_transaction_id"])
        is None
        or not isinstance(value["slurm_environment"], dict)
    ):
        raise ValueError("scheduler request profile mismatch")
    slurm = _validate_slurm_gpu_environment(value["slurm_environment"])
    if (
        value["slurm_environment"] != slurm
        or int(slurm["SLURM_JOB_ID"]) != value["job_id"]
        or int(slurm["SLURM_STEP_ID"]) != value["step_id"]
    ):
        raise ValueError("scheduler request environment identity mismatch")
    return value


def publish_scheduler_request(
    path: Path,
    *,
    environ: Mapping[str, str],
    expected_gpu_uuid: str,
    expected_parent_identity: str | None = None,
) -> dict[str, Any]:
    return publish_immutable_json(
        path,
        scheduler_request_value(environ=environ, expected_gpu_uuid=expected_gpu_uuid),
        expected_parent_identity=expected_parent_identity,
    )


def wait_for_scheduler_request(
    path: Path,
    *,
    timeout_seconds: float = 60.0,
    expected_parent_identity: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            value, identity = stable_read_immutable_json(
                path, expected_parent_identity=expected_parent_identity
            )
            return _validate_scheduler_request(value), identity
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for scheduler request") from None
            time.sleep(0.05)


def scheduler_host_binding_value(
    *, environ: Mapping[str, str], request: Mapping[str, Any]
) -> dict[str, str]:
    embedded = _validate_scheduler_request(request)
    names = (
        "SLURM_JOB_ID",
        "SLURM_JOB_GPUS",
        "SLURM_GPUS_ON_NODE",
        "SLURM_JOB_ACCOUNT",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_QOS",
        "SLURM_JOB_USER",
        "NVIDIA_VISIBLE_DEVICES",
        "SUBMISSION_TRANSACTION_ID",
    )
    values = {name: _required_environment(environ, name) for name in names}
    expected = embedded["slurm_environment"]
    for name in names[:-2]:
        if values[name] != expected[name]:
            raise ValueError(f"host/container scheduler binding mismatch: {name}")
    if values["NVIDIA_VISIBLE_DEVICES"] != embedded["expected_gpu_uuid"]:
        raise ValueError("host/container scheduler GPU UUID mismatch")
    if values["SUBMISSION_TRANSACTION_ID"] != embedded["submission_transaction_id"]:
        raise ValueError("host/container submission transaction mismatch")
    return values


def _validate_scheduler_handoff(
    value: Any,
    *,
    expected_slurm_config_sha256: str,
    runtime_closure_approval_sha256: str,
    expected_scontrol_sha256: str,
    expected_scontrol_size: int,
    expected_slurm_library_sha256: str,
    expected_slurm_library_size: int,
    expected_sacct_sha256: str,
    expected_sacct_size: int,
    expected_scancel_sha256: str,
    expected_scancel_size: int,
    expected_srun_sha256: str,
    expected_srun_size: int,
    request: Mapping[str, Any] | None = None,
    request_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "profile",
        "status",
        "request",
        "request_value",
        "scheduler_client",
        "host_binding",
        "job_record",
        "step_record",
    }:
        raise ValueError("scheduler handoff schema mismatch")
    if (
        value["schema_version"] != 1
        or value["profile"] != SCHEDULER_HANDOFF_PROFILE
        or value["status"] != "host_scheduler_records_sealed"
        or not isinstance(value["request"], dict)
    ):
        raise ValueError("scheduler handoff profile mismatch")
    request_artifact = value["request"]
    if (
        set(request_artifact) != {"path", "mode", "nlink", "size", "sha256"}
        or request_artifact["mode"] != "0444"
        or request_artifact["nlink"] != 1
        or not isinstance(request_artifact["size"], int)
        or request_artifact["size"] <= 0
        or SHA256_PATTERN.fullmatch(request_artifact["sha256"]) is None
    ):
        raise ValueError("scheduler handoff request identity mismatch")
    embedded_request = _validate_scheduler_request(value["request_value"])
    scheduler_client = validate_scheduler_client_identity(
        value["scheduler_client"],
        expected_slurm_config_sha256=expected_slurm_config_sha256,
        runtime_closure_approval_sha256=runtime_closure_approval_sha256,
        expected_scontrol_sha256=expected_scontrol_sha256,
        expected_scontrol_size=expected_scontrol_size,
        expected_slurm_library_sha256=expected_slurm_library_sha256,
        expected_slurm_library_size=expected_slurm_library_size,
        expected_sacct_sha256=expected_sacct_sha256,
        expected_sacct_size=expected_sacct_size,
        expected_scancel_sha256=expected_scancel_sha256,
        expected_scancel_size=expected_scancel_size,
        expected_srun_sha256=expected_srun_sha256,
        expected_srun_size=expected_srun_size,
    )
    if value["scheduler_client"] != scheduler_client:
        raise ValueError("scheduler handoff client identity is not canonical")
    if not isinstance(value["host_binding"], dict):
        raise ValueError("scheduler handoff host binding is missing")
    host_binding = scheduler_host_binding_value(
        environ=value["host_binding"], request=embedded_request
    )
    if value["host_binding"] != host_binding:
        raise ValueError("scheduler handoff host binding is not canonical")
    if request is not None and embedded_request != request:
        raise ValueError("scheduler handoff request value mismatch")
    if request_identity is not None and value["request"] != request_identity:
        raise ValueError("scheduler handoff request identity mismatch")
    job_id = embedded_request["job_id"]
    step_id = embedded_request["step_id"]
    job_record = value["job_record"]
    step_record = value["step_record"]
    if not isinstance(job_record, dict) or not isinstance(step_record, dict):
        raise ValueError("scheduler handoff records must be objects")
    job_raw = job_record.get("raw")
    step_raw = step_record.get("raw")
    if not isinstance(job_raw, str) or not isinstance(step_raw, str):
        raise ValueError("scheduler handoff records are missing raw values")
    expected_job = parse_job_scheduler_record(
        job_raw,
        expected_job_id=job_id,
        expected_transaction_id=embedded_request["submission_transaction_id"],
    )
    expected_step = parse_step_scheduler_record(
        step_raw,
        expected_job_id=job_id,
        expected_step_id=step_id,
        requested_tres_per_task=embedded_request["slurm_environment"][
            "SLURM_TRES_PER_TASK"
        ],
        expected_node=expected_job["fields"]["NodeList"],
    )
    if job_record != expected_job or step_record != expected_step:
        raise ValueError("scheduler handoff parsed record mismatch")
    return value


def seal_scheduler_handoff(
    *,
    request_path: Path,
    output_path: Path,
    job_record: str,
    step_record: str,
    scheduler_client: Mapping[str, Any] | None = None,
    host_environ: Mapping[str, str] | None = None,
    expected_parent_identity: str | None = None,
    expected_slurm_config_sha256: str,
    runtime_closure_approval_sha256: str,
    expected_scontrol_sha256: str,
    expected_scontrol_size: int,
    expected_slurm_library_sha256: str,
    expected_slurm_library_size: int,
    expected_sacct_sha256: str,
    expected_sacct_size: int,
    expected_scancel_sha256: str,
    expected_scancel_size: int,
    expected_srun_sha256: str,
    expected_srun_size: int,
) -> dict[str, Any]:
    request, request_identity = stable_read_immutable_json(
        request_path, expected_parent_identity=expected_parent_identity
    )
    request = _validate_scheduler_request(request)
    parsed_job = parse_job_scheduler_record(
        job_record,
        expected_job_id=request["job_id"],
        expected_transaction_id=request["submission_transaction_id"],
    )
    value = {
        "schema_version": 1,
        "profile": SCHEDULER_HANDOFF_PROFILE,
        "status": "host_scheduler_records_sealed",
        "request": request_identity,
        "request_value": request,
        "scheduler_client": validate_scheduler_client_identity(
            capture_scheduler_client_identity(
                expected_slurm_config_sha256=expected_slurm_config_sha256,
                runtime_closure_approval_sha256=(runtime_closure_approval_sha256),
                expected_scontrol_sha256=expected_scontrol_sha256,
                expected_scontrol_size=expected_scontrol_size,
                expected_slurm_library_sha256=expected_slurm_library_sha256,
                expected_slurm_library_size=expected_slurm_library_size,
                expected_sacct_sha256=expected_sacct_sha256,
                expected_sacct_size=expected_sacct_size,
                expected_scancel_sha256=expected_scancel_sha256,
                expected_scancel_size=expected_scancel_size,
                expected_srun_sha256=expected_srun_sha256,
                expected_srun_size=expected_srun_size,
            )
            if scheduler_client is None
            else scheduler_client,
            expected_slurm_config_sha256=expected_slurm_config_sha256,
            runtime_closure_approval_sha256=runtime_closure_approval_sha256,
            expected_scontrol_sha256=expected_scontrol_sha256,
            expected_scontrol_size=expected_scontrol_size,
            expected_slurm_library_sha256=expected_slurm_library_sha256,
            expected_slurm_library_size=expected_slurm_library_size,
            expected_sacct_sha256=expected_sacct_sha256,
            expected_sacct_size=expected_sacct_size,
            expected_scancel_sha256=expected_scancel_sha256,
            expected_scancel_size=expected_scancel_size,
            expected_srun_sha256=expected_srun_sha256,
            expected_srun_size=expected_srun_size,
        ),
        "host_binding": scheduler_host_binding_value(
            environ=os.environ if host_environ is None else host_environ,
            request=request,
        ),
        "job_record": parsed_job,
        "step_record": parse_step_scheduler_record(
            step_record,
            expected_job_id=request["job_id"],
            expected_step_id=request["step_id"],
            requested_tres_per_task=request["slurm_environment"]["SLURM_TRES_PER_TASK"],
            expected_node=parsed_job["fields"]["NodeList"],
        ),
    }
    _validate_scheduler_handoff(
        value,
        expected_slurm_config_sha256=expected_slurm_config_sha256,
        runtime_closure_approval_sha256=runtime_closure_approval_sha256,
        expected_scontrol_sha256=expected_scontrol_sha256,
        expected_scontrol_size=expected_scontrol_size,
        expected_slurm_library_sha256=expected_slurm_library_sha256,
        expected_slurm_library_size=expected_slurm_library_size,
        expected_sacct_sha256=expected_sacct_sha256,
        expected_sacct_size=expected_sacct_size,
        expected_scancel_sha256=expected_scancel_sha256,
        expected_scancel_size=expected_scancel_size,
        expected_srun_sha256=expected_srun_sha256,
        expected_srun_size=expected_srun_size,
        request=request,
        request_identity=request_identity,
    )
    return publish_immutable_json(
        output_path,
        value,
        expected_parent_identity=expected_parent_identity,
    )


def wait_for_scheduler_handoff(
    path: Path,
    *,
    request: Mapping[str, Any],
    request_identity: Mapping[str, Any],
    timeout_seconds: float = 60.0,
    expected_parent_identity: str | None = None,
    expected_slurm_config_sha256: str,
    runtime_closure_approval_sha256: str,
    expected_scontrol_sha256: str,
    expected_scontrol_size: int,
    expected_slurm_library_sha256: str,
    expected_slurm_library_size: int,
    expected_sacct_sha256: str,
    expected_sacct_size: int,
    expected_scancel_sha256: str,
    expected_scancel_size: int,
    expected_srun_sha256: str,
    expected_srun_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            value, identity = stable_read_immutable_json(
                path, expected_parent_identity=expected_parent_identity
            )
            return (
                _validate_scheduler_handoff(
                    value,
                    expected_slurm_config_sha256=expected_slurm_config_sha256,
                    runtime_closure_approval_sha256=(runtime_closure_approval_sha256),
                    expected_scontrol_sha256=expected_scontrol_sha256,
                    expected_scontrol_size=expected_scontrol_size,
                    expected_slurm_library_sha256=expected_slurm_library_sha256,
                    expected_slurm_library_size=expected_slurm_library_size,
                    expected_sacct_sha256=expected_sacct_sha256,
                    expected_sacct_size=expected_sacct_size,
                    expected_scancel_sha256=expected_scancel_sha256,
                    expected_scancel_size=expected_scancel_size,
                    expected_srun_sha256=expected_srun_sha256,
                    expected_srun_size=expected_srun_size,
                    request=request,
                    request_identity=request_identity,
                ),
                identity,
            )
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for scheduler handoff") from None
            time.sleep(0.05)


def publish_scheduler_terminal_request(
    *,
    request_path: Path,
    handoff_path: Path,
    output_path: Path,
    srun_exit_code: int,
    expected_parent_identity: str,
) -> dict[str, Any]:
    if (
        not isinstance(srun_exit_code, int)
        or isinstance(srun_exit_code, bool)
        or not 0 <= srun_exit_code <= 255
    ):
        raise ValueError("terminal request srun exit code must be one byte")
    request, request_identity = stable_read_immutable_json(
        request_path, expected_parent_identity=expected_parent_identity
    )
    request = _validate_scheduler_request(request)
    handoff, handoff_identity = stable_read_immutable_json(
        handoff_path, expected_parent_identity=expected_parent_identity
    )
    approvals = _validate_execution_approval_environment(os.environ)
    handoff = _validate_scheduler_handoff(
        handoff,
        request=request,
        request_identity=request_identity,
        **_scheduler_approval_kwargs(approvals),
    )
    value = {
        "schema_version": 1,
        "profile": SCHEDULER_TERMINAL_REQUEST_PROFILE,
        "status": "srun_reaped_writer_closed_local_group_absent",
        "job_id": request["job_id"],
        "step_id": request["step_id"],
        "srun_exit_code": srun_exit_code,
        "request": request_identity,
        "handoff": handoff_identity,
        "handoff_value_sha256": _sha256(canonical_json_bytes(handoff)),
    }
    return publish_immutable_json(
        output_path, value, expected_parent_identity=expected_parent_identity
    )


def _validate_scheduler_terminal_request(
    value: Any,
    *,
    request: Mapping[str, Any],
    request_identity: Mapping[str, Any],
    handoff: Mapping[str, Any],
    handoff_identity: Mapping[str, Any],
) -> dict[str, Any]:
    request = _validate_scheduler_request(request)
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "schema_version",
            "profile",
            "status",
            "job_id",
            "step_id",
            "srun_exit_code",
            "request",
            "handoff",
            "handoff_value_sha256",
        }
        or isinstance(value["schema_version"], bool)
        or value["schema_version"] != 1
        or value["profile"] != SCHEDULER_TERMINAL_REQUEST_PROFILE
        or value["status"] != "srun_reaped_writer_closed_local_group_absent"
        or value["job_id"] != request["job_id"]
        or value["step_id"] != request["step_id"]
        or not isinstance(value["srun_exit_code"], int)
        or isinstance(value["srun_exit_code"], bool)
        or not 0 <= value["srun_exit_code"] <= 255
        or value["request"] != request_identity
        or value["handoff"] != handoff_identity
        or value["handoff_value_sha256"] != _sha256(canonical_json_bytes(handoff))
    ):
        raise ValueError("scheduler terminal request schema mismatch")
    return value


def parse_sacct_terminal_record(
    raw: str,
    *,
    expected_job_id: int,
    expected_step_id: int,
    expected_node: str,
) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw.endswith("\n"):
        raise ValueError("sacct terminal record must be newline terminated")
    rows = [line for line in raw.splitlines() if line]
    if len(rows) != 1:
        raise ValueError("sacct must return exactly one terminal step row")
    fields = rows[0].split("|")
    if len(fields) != 8 or fields[-1] != "":
        raise ValueError("sacct terminal row is not exact parsable2 output")
    job_step, state, exit_code, start, end, elapsed_raw, node, _ = fields
    expected_job_step = f"{expected_job_id}.{expected_step_id}"
    terminal_states = {
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
    if (
        job_step != expected_job_step
        or state not in terminal_states
        or re.fullmatch(r"[0-9]+:[0-9]+", exit_code) is None
        or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}", start)
        is None
        or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}", end)
        is None
        or end < start
        or DECIMAL_PATTERN.fullmatch(elapsed_raw) is None
        or node != expected_node
    ):
        raise ValueError("sacct terminal row differs from the closed step contract")
    return {
        "command": [
            str(EXPECTED_SACCT_PATH),
            "--noheader",
            "--parsable2",
            f"--jobs={expected_job_step}",
            "--format=JobIDRaw,State,ExitCode,Start,End,ElapsedRaw,NodeList",
        ],
        "raw": raw,
        "raw_sha256": _sha256(raw.encode("utf-8")),
        "job_step": job_step,
        "state": state,
        "exit_code": exit_code,
        "start": start,
        "end": end,
        "elapsed_raw": int(elapsed_raw),
        "node": node,
    }


def _parse_cgroup_events_payload(payload: bytes, path: Path) -> dict[str, Any]:
    if not payload or len(payload) >= 4096 or not payload.endswith(b"\n"):
        raise ValueError("cgroup.events is empty, oversized, or unterminated")
    try:
        text = payload.decode("ascii", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("cgroup.events is not ASCII") from error
    values: dict[str, int] = {}
    for line in text.splitlines():
        key, separator, raw_value = line.partition(" ")
        if (
            separator != " "
            or re.fullmatch(r"[a-z_]+", key) is None
            or DECIMAL_PATTERN.fullmatch(raw_value) is None
            or key in values
        ):
            raise ValueError("cgroup.events contains a malformed record")
        values[key] = int(raw_value)
    if "populated" not in values:
        raise ValueError("cgroup.events omits populated")
    return {
        "path": str(path),
        "raw": text,
        "raw_sha256": _sha256(payload),
        "values": values,
    }


def _read_cgroup_events(descriptor: int, path: Path) -> dict[str, Any]:
    return _parse_cgroup_events_payload(os.pread(descriptor, 4096, 0), path)


def _validate_cgroup_events_evidence(
    value: Any, *, expected_path: str, expected_populated: int
) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != {"path", "raw", "raw_sha256", "values"}
        or value["path"] != expected_path
        or not isinstance(value["raw"], str)
    ):
        raise ValueError("scheduler cgroup evidence schema mismatch")
    try:
        canonical = _parse_cgroup_events_payload(
            value["raw"].encode("ascii", errors="strict"), Path(expected_path)
        )
    except UnicodeEncodeError as error:
        raise ValueError("scheduler cgroup evidence is not ASCII") from error
    if canonical != value or canonical["values"].get("populated") != expected_populated:
        raise ValueError("scheduler cgroup evidence is not canonical")
    return canonical


def _validate_scheduler_terminal(
    value: Any,
    *,
    request: Mapping[str, Any],
    request_identity: Mapping[str, Any],
    handoff: Mapping[str, Any],
    handoff_identity: Mapping[str, Any],
    terminal_request: Mapping[str, Any],
    terminal_request_identity: Mapping[str, Any],
    expected_cgroup_path: str,
    approvals: Mapping[str, str],
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "profile",
        "status",
        "request",
        "handoff",
        "terminal_request",
        "scheduler_client",
        "scontrol_absence",
        "sacct_terminal",
        "cgroup_events_live",
        "cgroup_events",
        "scancel_invoked",
    }:
        raise ValueError("scheduler terminal evidence schema mismatch")
    if (
        isinstance(value["schema_version"], bool)
        or value["schema_version"] != 1
        or value["profile"] != SCHEDULER_TERMINAL_PROFILE
        or value["status"] != "scheduler_step_terminal_and_cgroup_unpopulated"
        or value["request"] != request_identity
        or value["handoff"] != handoff_identity
        or value["terminal_request"] != terminal_request_identity
        or not isinstance(value["scancel_invoked"], bool)
    ):
        raise ValueError("scheduler terminal evidence profile mismatch")
    scheduler_client = validate_scheduler_client_identity(
        value["scheduler_client"], **_scheduler_approval_kwargs(approvals)
    )
    if scheduler_client != value["scheduler_client"]:
        raise ValueError("scheduler terminal client identity is not canonical")
    expected_job_step = f"{request['job_id']}.{request['step_id']}"
    absence = value["scontrol_absence"]
    if (
        not isinstance(absence, dict)
        or set(absence) != {"command", "returncode", "stdout", "stderr"}
        or absence["command"]
        != [
            str(EXPECTED_SCONTROL_PATH),
            "show",
            "step",
            "--oneliner",
            expected_job_step,
        ]
        or absence["returncode"] != 1
        or absence["stdout"] != ""
        or absence["stderr"]
        != (
            "scontrol: error: scontrol_print_step: "
            f"slurm_get_job_steps({expected_job_step}) failed: "
            "Invalid job id specified\n"
        )
    ):
        raise ValueError("scheduler terminal scontrol absence proof mismatch")
    terminal = value["sacct_terminal"]
    if not isinstance(terminal, dict) or terminal != parse_sacct_terminal_record(
        terminal.get("raw", ""),
        expected_job_id=request["job_id"],
        expected_step_id=request["step_id"],
        expected_node=handoff["job_record"]["fields"]["NodeList"],
    ):
        raise ValueError("scheduler terminal sacct proof mismatch")
    if terminal_request["srun_exit_code"] == 0 and (
        terminal["state"] != "COMPLETED" or terminal["exit_code"] != "0:0"
    ):
        raise ValueError("successful srun lacks COMPLETED|0:0 scheduler proof")
    cgroup = value["cgroup_events"]
    expected_cgroup_parts = PurePosixPath(expected_cgroup_path).parts
    expected_job_component = f"job_{request['job_id']}"
    expected_step_component = f"step_{request['step_id']}"
    if (
        not expected_cgroup_path.startswith("/")
        or PurePosixPath(expected_cgroup_path).as_posix() != expected_cgroup_path
        or ".." in expected_cgroup_parts
        or expected_cgroup_parts.count(expected_job_component) != 1
        or expected_cgroup_parts.count(expected_step_component) != 1
        or expected_cgroup_parts.index(expected_job_component)
        >= expected_cgroup_parts.index(expected_step_component)
    ):
        raise ValueError("scheduler terminal cgroup path is not request-bound")
    expected_step_index = expected_cgroup_parts.index(expected_step_component)
    expected_events_path = str(
        Path("/sys/fs/cgroup").joinpath(
            *expected_cgroup_parts[1 : expected_step_index + 1], "cgroup.events"
        )
    )
    _validate_cgroup_events_evidence(
        value["cgroup_events_live"],
        expected_path=expected_events_path,
        expected_populated=1,
    )
    _validate_cgroup_events_evidence(
        cgroup,
        expected_path=expected_events_path,
        expected_populated=0,
    )
    return value


def broker_scheduler_handoff(
    *,
    request_path: Path,
    output_path: Path,
    terminal_request_path: Path,
    terminal_output_path: Path,
    expected_parent_identity: str,
    timeout_seconds: float = 30.0,
    terminal_timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Capture live scheduler state, then prove exact terminal step closure."""

    for label, value, maximum in (
        ("handoff", timeout_seconds, 120.0),
        ("terminal", terminal_timeout_seconds, 600.0),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value != value
            or value <= 0
            or value > maximum
        ):
            raise ValueError(f"scheduler {label} timeout is outside the closed range")

    approvals = _validate_execution_approval_environment(os.environ)
    approval_kwargs = _scheduler_approval_kwargs(approvals)
    scheduler_client_before = capture_scheduler_client_identity(
        **approval_kwargs,
    )
    request, request_identity = wait_for_scheduler_request(
        request_path,
        timeout_seconds=timeout_seconds,
        expected_parent_identity=expected_parent_identity,
    )
    job_id = request["job_id"]
    step_id = request["step_id"]
    scheduler_environment = scheduler_client_before["execution_environment"]

    def run_scontrol(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(EXPECTED_SCONTROL_PATH), *arguments],
            env=scheduler_environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

    job_result = run_scontrol(["show", "job", "--oneliner", str(job_id)])
    job_record = job_result.stdout.rstrip("\n")
    if (
        job_result.returncode != 0
        or job_result.stderr
        or not job_record
        or "\n" in job_record
    ):
        raise RuntimeError("pinned scontrol did not return one exact job record")

    step_record = ""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() <= deadline:
        step_result = run_scontrol(
            ["show", "step", "--oneliner", f"{job_id}.{step_id}"]
        )
        candidate = step_result.stdout.rstrip("\n")
        if (
            step_result.returncode == 0
            and not step_result.stderr
            and candidate
            and "\n" not in candidate
        ):
            step_record = candidate
            break
        time.sleep(0.05)
    if not step_record:
        raise TimeoutError("timed out reading the exact srun step record")

    scheduler_client_after = capture_scheduler_client_identity(
        **approval_kwargs,
    )
    if scheduler_client_after != scheduler_client_before:
        raise RuntimeError("pinned scheduler client changed across record capture")
    handoff_identity = seal_scheduler_handoff(
        request_path=request_path,
        output_path=output_path,
        job_record=job_record,
        step_record=step_record,
        scheduler_client=scheduler_client_before,
        expected_parent_identity=expected_parent_identity,
        **approval_kwargs,
    )
    handoff, observed_handoff_identity = stable_read_immutable_json(
        output_path,
        expected_sha256=handoff_identity["sha256"],
        expected_parent_identity=expected_parent_identity,
    )
    if observed_handoff_identity != handoff_identity:
        raise RuntimeError("scheduler handoff identity changed after publication")
    handoff = _validate_scheduler_handoff(
        handoff,
        request=request,
        request_identity=request_identity,
        **approval_kwargs,
    )

    preexec_path = request_path.with_name("startup_preexec.json")
    preexec_deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            preexec, _ = stable_read_immutable_json(
                preexec_path, expected_parent_identity=expected_parent_identity
            )
            break
        except FileNotFoundError:
            if terminal_request_path.exists() or time.monotonic() >= preexec_deadline:
                raise TimeoutError(
                    "scheduler broker did not pin the live step cgroup before terminal request"
                ) from None
            time.sleep(0.02)
    cgroup_path_text = (
        preexec.get("runtime", {}).get("cgroup", {}).get("job_step_path")
        if isinstance(preexec, dict)
        else None
    )
    expected_job_component = f"job_{job_id}"
    expected_step_component = f"step_{step_id}"
    if not isinstance(cgroup_path_text, str):
        raise ValueError("pre-exec evidence omits the step cgroup path")
    cgroup_pure = PurePosixPath(cgroup_path_text)
    if (
        not cgroup_path_text.startswith("/")
        or cgroup_pure.as_posix() != cgroup_path_text
        or ".." in cgroup_pure.parts
        or expected_job_component not in cgroup_pure.parts
        or expected_step_component not in cgroup_pure.parts
    ):
        raise ValueError("pre-exec step cgroup path is not request-bound")
    step_component_index = cgroup_pure.parts.index(expected_step_component)
    step_cgroup_parts = cgroup_pure.parts[1 : step_component_index + 1]
    cgroup_events_path = Path("/sys/fs/cgroup").joinpath(
        *step_cgroup_parts, "cgroup.events"
    )
    cgroup_events_fd = os.open(
        cgroup_events_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    )
    live_cgroup_events = _read_cgroup_events(cgroup_events_fd, cgroup_events_path)
    if live_cgroup_events["values"]["populated"] != 1:
        os.close(cgroup_events_fd)
        raise RuntimeError("step cgroup was not populated when the broker pinned it")
    try:
        request_deadline = time.monotonic() + terminal_timeout_seconds
        while True:
            try:
                terminal_request, terminal_request_identity = (
                    stable_read_immutable_json(
                        terminal_request_path,
                        expected_parent_identity=expected_parent_identity,
                    )
                )
                break
            except FileNotFoundError:
                if time.monotonic() >= request_deadline:
                    raise TimeoutError(
                        "timed out waiting for scheduler terminal request"
                    ) from None
                time.sleep(0.02)
    except BaseException:
        os.close(cgroup_events_fd)
        raise
    terminal_request = _validate_scheduler_terminal_request(
        terminal_request,
        request=request,
        request_identity=request_identity,
        handoff=handoff,
        handoff_identity=handoff_identity,
    )

    expected_job_step = f"{job_id}.{step_id}"

    def run_sacct() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(EXPECTED_SACCT_PATH),
                "--noheader",
                "--parsable2",
                f"--jobs={expected_job_step}",
                "--format=JobIDRaw,State,ExitCode,Start,End,ElapsedRaw,NodeList",
            ],
            env=scheduler_environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def run_scancel() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(EXPECTED_SCANCEL_PATH), "--quiet", expected_job_step],
            env=scheduler_environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

    expected_absence_stderr = (
        "scontrol: error: scontrol_print_step: "
        f"slurm_get_job_steps({expected_job_step}) failed: "
        "Invalid job id specified\n"
    )
    scancel_invoked = False
    terminal_record: dict[str, Any] | None = None
    absence_record: dict[str, Any] | None = None
    cgroup_events: dict[str, Any] | None = None
    terminal_deadline = time.monotonic() + terminal_timeout_seconds
    final_deadline = terminal_deadline
    try:
        while time.monotonic() <= final_deadline:
            step_result = run_scontrol(
                ["show", "step", "--oneliner", expected_job_step]
            )
            step_absent = (
                step_result.returncode == 1
                and step_result.stdout == ""
                and step_result.stderr == expected_absence_stderr
            )
            sacct_result = run_sacct()
            parsed_terminal: dict[str, Any] | None = None
            if sacct_result.returncode == 0 and sacct_result.stderr == "":
                try:
                    parsed_terminal = parse_sacct_terminal_record(
                        sacct_result.stdout,
                        expected_job_id=job_id,
                        expected_step_id=step_id,
                        expected_node=handoff["job_record"]["fields"]["NodeList"],
                    )
                except ValueError:
                    parsed_terminal = None
            observed_cgroup = _read_cgroup_events(cgroup_events_fd, cgroup_events_path)
            if (
                step_absent
                and parsed_terminal is not None
                and observed_cgroup["values"]["populated"] == 0
            ):
                terminal_record = parsed_terminal
                absence_record = {
                    "command": [
                        str(EXPECTED_SCONTROL_PATH),
                        "show",
                        "step",
                        "--oneliner",
                        expected_job_step,
                    ],
                    "returncode": 1,
                    "stdout": "",
                    "stderr": step_result.stderr,
                }
                cgroup_events = observed_cgroup
                break
            if time.monotonic() >= terminal_deadline and not scancel_invoked:
                cancellation = run_scancel()
                if (
                    cancellation.returncode != 0
                    or cancellation.stdout
                    or cancellation.stderr
                ):
                    raise RuntimeError(
                        "pinned scancel failed to clean the terminal step"
                    )
                scancel_invoked = True
                final_deadline = time.monotonic() + 150.0
            time.sleep(0.05)
    finally:
        os.close(cgroup_events_fd)
    if terminal_record is None or absence_record is None or cgroup_events is None:
        raise TimeoutError("scheduler step did not reach proven terminal closure")
    if terminal_request["srun_exit_code"] == 0 and (
        terminal_record["state"] != "COMPLETED" or terminal_record["exit_code"] != "0:0"
    ):
        raise RuntimeError("zero-exit srun did not produce COMPLETED|0:0")
    scheduler_client_terminal = capture_scheduler_client_identity(**approval_kwargs)
    if scheduler_client_terminal != scheduler_client_before:
        raise RuntimeError("pinned scheduler client changed before terminal proof")
    terminal_value = {
        "schema_version": 1,
        "profile": SCHEDULER_TERMINAL_PROFILE,
        "status": "scheduler_step_terminal_and_cgroup_unpopulated",
        "request": request_identity,
        "handoff": handoff_identity,
        "terminal_request": terminal_request_identity,
        "scheduler_client": scheduler_client_terminal,
        "scontrol_absence": absence_record,
        "sacct_terminal": terminal_record,
        "cgroup_events_live": live_cgroup_events,
        "cgroup_events": cgroup_events,
        "scancel_invoked": scancel_invoked,
    }
    return publish_immutable_json(
        terminal_output_path,
        terminal_value,
        expected_parent_identity=expected_parent_identity,
    )


def validate_target_argv(
    argv: Sequence[str],
    *,
    preexec_path: Path,
    preclose_path: Path,
    expected_gpu_uuid: str,
    expected_port: int,
) -> list[str]:
    target = list(argv)
    if not 1 <= expected_port <= 65535:
        raise ValueError("invalid expected public evaluator port")
    preexec_path = _canonical_absolute_path(preexec_path, label="pre-exec output")
    preclose_path = _canonical_absolute_path(preclose_path, label="pre-close output")
    if (
        preexec_path.parent != preclose_path.parent
        or preexec_path.name != "startup_preexec.json"
        or preclose_path.name != "startup_preclose.json"
    ):
        raise ValueError("diagnostic target output paths do not use the closed layout")
    task_dir = preexec_path.parent
    expected = [
        "/.venv/bin/python",
        "scripts/eval.py",
        "--environment",
        "DROID-FoodBussing",
        "--control-mode",
        "joint-position",
        "--policy.client",
        "DroidJointPos",
        "--policy.host",
        "127.0.0.1",
        "--policy.port",
        str(expected_port),
        "--policy.open-loop-horizon",
        "8",
        "--policy.frame-description",
        "robot base frame",
        "--policy.action-frame",
        "robot_base",
        "--policy.dataset-name",
        "droid",
        "--policy.no-rotate-wrist-180",
        "--policy.no-render-every-step",
        "--policy.state-type",
        "joint_position",
        "--policy.expected-action-horizon",
        "15",
        "--policy.expected-action-dim",
        "8",
        "--policy.trace-path",
        str(task_dir / "policy_traces.forbidden"),
        "--run-folder",
        str(task_dir),
        "--rollouts",
        "1",
        "--environment-seed",
        "0",
        "--runtime-contract-path",
        str(task_dir / "runtime_contract.forbidden"),
        "--headless",
        "--startup-diagnostic",
        STARTUP_DIAGNOSTIC_MODE,
        "--startup-diagnostic-preexec-path",
        str(preexec_path),
        "--startup-diagnostic-preclose-path",
        str(preclose_path),
        "--startup-diagnostic-expected-gpu-uuid",
        expected_gpu_uuid,
    ]
    if target != expected:
        mismatch = next(
            (
                index
                for index, (actual, wanted) in enumerate(zip(target, expected))
                if actual != wanted
            ),
            min(len(target), len(expected)),
        )
        raise ValueError(
            "diagnostic target argv differs from the closed public schema "
            f"at token {mismatch}"
        )
    return target


def python_argv_after_exec(target_argv: Sequence[str]) -> list[str]:
    """Return Python's real ``sys.argv`` for the approved exec target."""

    target = list(target_argv)
    if len(target) < 2 or target[0] != "/.venv/bin/python":
        raise ValueError("invalid Python exec target")
    return target[1:]


def _scheduler_handoff_runtime_value(
    *,
    environ: Mapping[str, str],
    expected_gpu_uuid: str,
    supplied: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    approvals = _validate_execution_approval_environment(environ)
    approval_kwargs = _scheduler_approval_kwargs(approvals)
    request = scheduler_request_value(
        environ=environ, expected_gpu_uuid=expected_gpu_uuid
    )
    loaded_from_environment = supplied is None
    task_dir_identity: str | None = None
    if supplied is None:
        task_dir_identity = _required_environment(
            environ, "POLARIS_STARTUP_DIAGNOSTIC_TASK_DIR_IDENTITY"
        )
        path = Path(
            _required_environment(
                environ,
                "POLARIS_STARTUP_DIAGNOSTIC_SCHEDULER_HANDOFF_PATH",
            )
        )
        expected_sha256 = _required_environment(
            environ,
            "POLARIS_STARTUP_DIAGNOSTIC_SCHEDULER_HANDOFF_SHA256",
        )
        if SHA256_PATTERN.fullmatch(expected_sha256) is None:
            raise ValueError("invalid scheduler handoff SHA-256")
        value, identity = stable_read_immutable_json(
            path,
            expected_sha256=expected_sha256,
            expected_parent_identity=task_dir_identity,
        )
        supplied = {"artifact": identity, "value": value}
    if not isinstance(supplied, dict) or set(supplied) != {"artifact", "value"}:
        raise ValueError("scheduler handoff runtime schema mismatch")
    artifact = supplied["artifact"]
    if (
        not isinstance(artifact, dict)
        or set(artifact) != {"path", "mode", "nlink", "size", "sha256"}
        or artifact["mode"] != "0444"
        or artifact["nlink"] != 1
        or SHA256_PATTERN.fullmatch(artifact["sha256"]) is None
    ):
        raise ValueError("scheduler handoff artifact identity mismatch")
    value = _validate_scheduler_handoff(
        supplied["value"],
        **approval_kwargs,
    )
    if value["request_value"] != request:
        raise ValueError("scheduler handoff differs from the live Slurm environment")
    if loaded_from_environment:
        request_value, request_identity = stable_read_immutable_json(
            Path(value["request"]["path"]),
            expected_sha256=value["request"]["sha256"],
            expected_parent_identity=task_dir_identity,
        )
        if request_identity != value["request"] or request_value != request:
            raise ValueError("scheduler handoff request artifact continuity mismatch")
    return {"artifact": dict(artifact), "value": value}


def capture_output_directory_context(
    environ: Mapping[str, str],
) -> dict[str, str]:
    run_dir = _canonical_directory(
        Path(_required_environment(environ, "POLARIS_STARTUP_DIAGNOSTIC_RUN_DIR")),
        label="diagnostic run directory",
    )
    task_dir = _canonical_directory(
        Path(_required_environment(environ, "POLARIS_STARTUP_DIAGNOSTIC_TASK_DIR")),
        label="diagnostic task directory",
    )
    if task_dir.parent != run_dir or task_dir.name != "app_launcher_only":
        raise ValueError("diagnostic output directory layout mismatch")
    namespace_parent = _canonical_directory(
        run_dir.parent, label="diagnostic output namespace parent"
    )
    expected = {
        "namespace_parent_identity": _required_environment(
            environ, "POLARIS_OUTPUT_NAMESPACE_PARENT_IDENTITY"
        ),
        "run_identity": _required_environment(
            environ, "POLARIS_STARTUP_DIAGNOSTIC_RUN_DIR_IDENTITY"
        ),
        "task_identity": _required_environment(
            environ, "POLARIS_STARTUP_DIAGNOSTIC_TASK_DIR_IDENTITY"
        ),
    }
    parent_fd = os.open(
        namespace_parent,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    run_fd = -1
    task_fd = -1
    try:
        _validate_expected_directory_identity(
            parent_fd,
            expected=expected["namespace_parent_identity"],
            label="diagnostic output namespace parent",
        )
        run_fd = os.open(
            run_dir.name,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        _validate_child_directory_binding(
            parent_fd, run_fd, name=run_dir.name, path=run_dir
        )
        _validate_expected_directory_identity(
            run_fd, expected=expected["run_identity"], label="diagnostic run directory"
        )
        task_fd = os.open(
            task_dir.name,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=run_fd,
        )
        _validate_expected_directory_identity(
            task_fd,
            expected=expected["task_identity"],
            label="diagnostic task directory",
        )
        _validate_child_directory_binding(
            run_fd, task_fd, name=task_dir.name, path=task_dir
        )
        _validate_parent_binding(parent_fd, namespace_parent)
    finally:
        if task_fd >= 0:
            os.close(task_fd)
        if run_fd >= 0:
            os.close(run_fd)
        os.close(parent_fd)
    return {
        "namespace_parent": str(namespace_parent),
        "namespace_parent_identity": expected["namespace_parent_identity"],
        "run_dir": str(run_dir),
        "run_identity": expected["run_identity"],
        "task_dir": str(task_dir),
        "task_identity": expected["task_identity"],
    }


def capture_runtime_context(
    *,
    python_argv: Sequence[str],
    source_root: Path,
    expected_gpu_uuid: str,
    environ: Mapping[str, str] | None = None,
    nvidia_smi_output: str | None = None,
    cgroup_text: str | None = None,
    device_nodes: Sequence[Mapping[str, Any]] | None = None,
    pid: int | None = None,
    ppid: int | None = None,
    executable: str | None = None,
    cwd: Path | None = None,
    scheduler_handoff: Mapping[str, Any] | None = None,
    output_directories: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    environment = os.environ if environ is None else environ
    if GPU_UUID_PATTERN.fullmatch(expected_gpu_uuid) is None:
        raise ValueError("invalid expected batch GPU UUID")
    slurm_environment = _validate_slurm_gpu_environment(environment)
    job_id = slurm_environment["SLURM_JOB_ID"]
    step_id = slurm_environment["SLURM_STEP_ID"]
    job_gpus = slurm_environment["SLURM_JOB_GPUS"]
    step_gpus = slurm_environment["SLURM_STEP_GPUS"]
    visible_gpu = _required_environment(environment, "NVIDIA_VISIBLE_DEVICES")
    if visible_gpu != expected_gpu_uuid:
        raise ValueError("NVIDIA_VISIBLE_DEVICES differs from the batch GPU UUID")
    nvidia = parse_nvidia_smi_output(
        _run_nvidia_smi() if nvidia_smi_output is None else nvidia_smi_output
    )
    if nvidia["uuid"] != expected_gpu_uuid:
        raise ValueError("nvidia-smi GPU UUID differs from the batch GPU UUID")
    if int(job_gpus) != nvidia["minor_number"]:
        raise ValueError("Slurm GPU index differs from the NVIDIA device minor number")
    if cgroup_text is None:
        cgroup_text = Path("/proc/self/cgroup").read_text(encoding="utf-8")
    cgroup = parse_cgroup_text(cgroup_text, job_id=job_id, step_id=step_id)
    nodes = validate_device_nodes(
        capture_device_nodes() if device_nodes is None else device_nodes,
        expected_minor_number=nvidia["minor_number"],
    )
    source_root = source_root.resolve(strict=True)
    if not source_root.is_dir() or source_root.is_symlink():
        raise ValueError("source root must be one canonical real directory")
    script_path = source_root / "scripts/eval.py"
    module_path = Path(__file__).resolve(strict=True)
    expected_module_path = (
        source_root / "src/polaris/app_launcher_startup_diagnostic.py"
    )
    if module_path != expected_module_path:
        raise ValueError("startup diagnostic module escaped the approved source root")
    source = {
        "root": str(source_root),
        "eval_script": stable_file_identity(script_path),
        "diagnostic_module": stable_file_identity(module_path),
        "approval": _validate_source_environment(environment),
    }
    current_cwd = Path.cwd() if cwd is None else cwd
    current_executable = sys.executable if executable is None else executable
    if current_cwd.resolve(strict=True) != source_root:
        raise ValueError("startup diagnostic process cwd differs from the source root")
    if current_executable != "/.venv/bin/python":
        raise ValueError("startup diagnostic interpreter is not /.venv/bin/python")
    selected_gpu_environment = {
        name: environment.get(name) for name in PREEXEC_GPU_ENVIRONMENT
    }
    scheduler = _scheduler_handoff_runtime_value(
        environ=environment,
        expected_gpu_uuid=expected_gpu_uuid,
        supplied=scheduler_handoff,
    )
    execution_approvals = _validate_execution_approval_environment(environment)
    pyxis_image = _validate_pyxis_image_environment(environment)
    directory_context = (
        capture_output_directory_context(environment)
        if output_directories is None
        else dict(output_directories)
    )
    if set(directory_context) != {
        "namespace_parent",
        "namespace_parent_identity",
        "run_dir",
        "run_identity",
        "task_dir",
        "task_identity",
    }:
        raise ValueError("output directory runtime schema mismatch")
    return {
        "process": {
            "pid": os.getpid() if pid is None else pid,
            "ppid": os.getppid() if ppid is None else ppid,
            "executable": current_executable,
            "cwd": str(current_cwd),
            "python_argv": list(python_argv),
        },
        "slurm": {
            "job_id": int(job_id),
            "step_id": int(step_id),
            "job_gpu_index": int(job_gpus),
            "step_gpu_index": int(step_gpus),
            "gpu_environment": selected_gpu_environment,
            "gpus_on_node": int(slurm_environment["SLURM_GPUS_ON_NODE"]),
            "gpus_per_task": int(slurm_environment["SLURM_GPUS_PER_TASK"]),
            "tres_per_task": slurm_environment["SLURM_TRES_PER_TASK"],
            "tres_per_task_items": parse_tres_per_task(
                slurm_environment["SLURM_TRES_PER_TASK"]
            ),
            "scheduler_handoff": scheduler,
        },
        "nvidia_smi": nvidia,
        "cgroup": cgroup,
        "device_nodes": nodes,
        "output_directories": directory_context,
        "execution_approvals": execution_approvals,
        "pyxis_image": pyxis_image,
        "source": source,
    }


def build_closed_eval_environment(
    *,
    inherited: Mapping[str, str],
    source_root: Path,
    data_root: Path,
    cache_root: Path,
    preexec_sha256: str,
    scheduler_handoff_path: Path,
    scheduler_handoff_sha256: str,
    run_dir: Path,
    task_dir: Path,
    namespace_parent_identity: str,
    run_dir_identity: str,
    task_dir_identity: str,
) -> dict[str, str]:
    if SHA256_PATTERN.fullmatch(preexec_sha256) is None:
        raise ValueError("invalid pre-exec artifact SHA-256")
    if SHA256_PATTERN.fullmatch(scheduler_handoff_sha256) is None:
        raise ValueError("invalid scheduler handoff SHA-256")
    scheduler_handoff_path = _canonical_absolute_path(
        scheduler_handoff_path, label="scheduler handoff"
    )
    source_identity = _validate_source_environment(inherited)
    execution_approvals = _validate_execution_approval_environment(inherited)
    _validate_pyxis_image_environment(inherited)
    slurm = _validate_slurm_gpu_environment(inherited)
    submission_transaction_id = _submission_transaction_id(inherited)
    if _required_environment(inherited, "POLARIS_EVAL_MODE") != STARTUP_DIAGNOSTIC_MODE:
        raise ValueError("closed evaluator mode is not app_launcher_only")
    job_id = slurm["SLURM_JOB_ID"]
    step_id = slurm["SLURM_STEP_ID"]
    visible_gpu = _required_environment(inherited, "NVIDIA_VISIBLE_DEVICES")
    driver_capabilities = _required_environment(inherited, "NVIDIA_DRIVER_CAPABILITIES")
    if DECIMAL_PATTERN.fullmatch(job_id) is None or job_id == "0":
        raise ValueError("invalid inherited Slurm job ID")
    if DECIMAL_PATTERN.fullmatch(step_id) is None:
        raise ValueError("invalid inherited Slurm step ID")
    if GPU_UUID_PATTERN.fullmatch(visible_gpu) is None:
        raise ValueError("invalid inherited NVIDIA_VISIBLE_DEVICES")
    job_gpus = slurm["SLURM_JOB_GPUS"]
    step_gpus = slurm["SLURM_STEP_GPUS"]
    return {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "NVIDIA_VISIBLE_DEVICES": visible_gpu,
        "NVIDIA_DRIVER_CAPABILITIES": driver_capabilities,
        "VK_DRIVER_FILES": "/etc/vulkan/icd.d/nvidia_icd.json",
        "ACCEPT_EULA": "Y",
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "PRIVACY_CONSENT": "Y",
        "OMNI_KIT_ALLOW_ROOT": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": (
            f"{source_root}/src:"
            f"{source_root}/third_party/openpi/packages/openpi-client/src"
        ),
        "POLARIS_DATA_PATH": str(data_root),
        "XDG_CACHE_HOME": str(cache_root),
        "HF_HOME": str(cache_root / "huggingface"),
        "HOME": str(cache_root / "home"),
        "POLARIS_EVAL_MODE": STARTUP_DIAGNOSTIC_MODE,
        "SUBMISSION_TRANSACTION_ID": submission_transaction_id,
        "SLURM_JOB_ID": job_id,
        "SLURM_STEP_ID": step_id,
        "SLURM_JOB_GPUS": job_gpus,
        "SLURM_STEP_GPUS": step_gpus,
        "SLURM_GPUS_ON_NODE": slurm["SLURM_GPUS_ON_NODE"],
        "SLURM_GPUS_PER_TASK": slurm["SLURM_GPUS_PER_TASK"],
        "SLURM_TRES_PER_TASK": slurm["SLURM_TRES_PER_TASK"],
        "SLURM_CPUS_PER_TASK": slurm["SLURM_CPUS_PER_TASK"],
        "SLURM_NTASKS": slurm["SLURM_NTASKS"],
        "SLURM_JOB_NUM_NODES": slurm["SLURM_JOB_NUM_NODES"],
        "SLURM_MEM_PER_NODE": slurm["SLURM_MEM_PER_NODE"],
        "SLURM_JOB_ACCOUNT": slurm["SLURM_JOB_ACCOUNT"],
        "SLURM_JOB_PARTITION": slurm["SLURM_JOB_PARTITION"],
        "SLURM_JOB_QOS": slurm["SLURM_JOB_QOS"],
        "SLURM_JOB_USER": slurm["SLURM_JOB_USER"],
        "POLARIS_STARTUP_DIAGNOSTIC_RUN_DIR": str(run_dir),
        "POLARIS_STARTUP_DIAGNOSTIC_TASK_DIR": str(task_dir),
        "POLARIS_OUTPUT_NAMESPACE_PARENT_IDENTITY": namespace_parent_identity,
        "POLARIS_STARTUP_DIAGNOSTIC_RUN_DIR_IDENTITY": run_dir_identity,
        "POLARIS_STARTUP_DIAGNOSTIC_TASK_DIR_IDENTITY": task_dir_identity,
        "POLARIS_STARTUP_DIAGNOSTIC_PREEXEC_SHA256": preexec_sha256,
        "POLARIS_STARTUP_DIAGNOSTIC_SCHEDULER_HANDOFF_PATH": str(
            scheduler_handoff_path
        ),
        "POLARIS_STARTUP_DIAGNOSTIC_SCHEDULER_HANDOFF_SHA256": (
            scheduler_handoff_sha256
        ),
        **{
            name: _required_environment(inherited, name)
            for name in PYXIS_IMAGE_ENVIRONMENT
        },
        **execution_approvals,
        **source_identity,
    }


def _forbidden_loaded_modules() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in FORBIDDEN_MODULE_PREFIXES
        )
    )


def _validate_context_continuity(
    preexec: Mapping[str, Any],
    current: Mapping[str, Any],
) -> None:
    expected_fields = {
        "schema_version",
        "profile",
        "status",
        "startup_diagnostic",
        "runtime",
        "launcher_argv",
        "target_argv",
        "zero_work_counters",
        "bounded_diagnostic_counts",
    }
    if set(preexec) != expected_fields:
        raise ValueError("pre-exec artifact schema mismatch")
    if (
        preexec.get("schema_version") != 1
        or preexec.get("profile") != PREEXEC_PROFILE
        or preexec.get("status") != "captured_before_public_eval_exec"
        or preexec.get("startup_diagnostic") != STARTUP_DIAGNOSTIC_MODE
        or preexec.get("zero_work_counters") != ZERO_WORK_COUNTERS
        or preexec.get("bounded_diagnostic_counts")
        != {
            "nvidia_smi_invocations": 1,
            "scheduler_request_artifacts": 1,
            "scheduler_handoff_artifacts": 1,
            "job_scheduler_records": 1,
            "step_scheduler_records": 1,
            "preexec_artifacts": 1,
            "preclose_artifacts": 0,
            "ready_artifacts": 0,
            "simulation_app_close_calls": 0,
        }
    ):
        raise ValueError("pre-exec artifact profile mismatch")
    launcher_argv = preexec.get("launcher_argv")
    target_argv = preexec.get("target_argv")
    if (
        not isinstance(launcher_argv, list)
        or not launcher_argv
        or not all(isinstance(value, str) for value in launcher_argv)
        or not isinstance(target_argv, list)
        or not all(isinstance(value, str) for value in target_argv)
    ):
        raise ValueError("pre-exec argv schema mismatch")
    before = preexec.get("runtime")
    if not isinstance(before, dict):
        raise ValueError("pre-exec runtime record is missing")
    if before.get("process", {}).get("python_argv") != python_argv_after_exec(
        target_argv
    ):
        raise ValueError("pre-exec target/Python argv binding mismatch")
    for path in (
        ("process", "pid"),
        ("process", "ppid"),
        ("process", "executable"),
        ("process", "cwd"),
        ("process", "python_argv"),
        ("slurm", "job_id"),
        ("slurm", "step_id"),
        ("slurm", "job_gpu_index"),
        ("slurm", "step_gpu_index"),
        ("slurm", "gpus_on_node"),
        ("slurm", "gpus_per_task"),
        ("slurm", "tres_per_task"),
        ("slurm", "tres_per_task_items"),
        ("slurm", "scheduler_handoff"),
        ("nvidia_smi", "uuid"),
        ("nvidia_smi", "name"),
        ("nvidia_smi", "driver_version"),
        ("nvidia_smi", "minor_number"),
        ("cgroup", "raw_sha256"),
        ("cgroup", "job_step_path"),
        ("output_directories",),
        ("execution_approvals",),
        ("pyxis_image",),
        ("source", "root"),
        ("source", "eval_script", "sha256"),
        ("source", "diagnostic_module", "sha256"),
        ("source", "approval"),
        ("device_nodes", "physical"),
    ):
        left: Any = before
        right: Any = current
        for component in path:
            if not isinstance(left, dict) or not isinstance(right, dict):
                raise ValueError(f"runtime continuity schema mismatch at {path}")
            left = left.get(component)
            right = right.get(component)
        if left != right:
            raise ValueError(f"runtime continuity mismatch at {'.'.join(path)}")


def ready_path_for(preclose_path: Path) -> Path:
    return preclose_path.with_name(f"{preclose_path.stem}.ready{preclose_path.suffix}")


def _run_app_launcher_only_diagnostic(
    *,
    simulation_app: Any,
    preexec_path: Path,
    preclose_path: Path,
    expected_gpu_uuid: str,
) -> dict[str, Any]:
    """Finalize the policy-free branch and close the SimulationApp exactly once."""

    expected_preexec_sha256 = _required_environment(
        os.environ, "POLARIS_STARTUP_DIAGNOSTIC_PREEXEC_SHA256"
    )
    if SHA256_PATTERN.fullmatch(expected_preexec_sha256) is None:
        raise ValueError("invalid pre-exec digest handoff")
    task_dir_identity = _required_environment(
        os.environ, "POLARIS_STARTUP_DIAGNOSTIC_TASK_DIR_IDENTITY"
    )
    preexec, preexec_identity = stable_read_immutable_json(
        preexec_path,
        expected_sha256=expected_preexec_sha256,
        expected_parent_identity=task_dir_identity,
    )
    source_root = Path(preexec["runtime"]["source"]["root"])
    current = capture_runtime_context(
        python_argv=sys.argv,
        source_root=source_root,
        expected_gpu_uuid=expected_gpu_uuid,
    )
    _validate_context_continuity(preexec, current)
    forbidden_modules = _forbidden_loaded_modules()
    if forbidden_modules:
        raise RuntimeError(
            "policy-free AppLauncher boundary imported forbidden modules: "
            + ",".join(forbidden_modules)
        )
    preclose_value = {
        "schema_version": 1,
        "profile": PRECLOSE_PROFILE,
        "status": "simulation_app_close_pending",
        "startup_diagnostic": STARTUP_DIAGNOSTIC_MODE,
        "preexec": preexec_identity,
        "runtime": current,
        "forbidden_module_prefixes": list(FORBIDDEN_MODULE_PREFIXES),
        "forbidden_loaded_modules": [],
        "zero_work_counters": dict(ZERO_WORK_COUNTERS),
        "bounded_diagnostic_counts": {
            "nvidia_smi_invocations": 2,
            "scheduler_request_artifacts": 1,
            "scheduler_handoff_artifacts": 1,
            "job_scheduler_records": 1,
            "step_scheduler_records": 1,
            "preexec_artifacts": 1,
            "preclose_artifacts": 1,
            "ready_artifacts": 0,
            "simulation_app_close_calls": 0,
        },
    }
    preclose_identity = publish_immutable_json(
        preclose_path,
        preclose_value,
        expected_parent_identity=task_dir_identity,
    )
    ready_path = ready_path_for(preclose_path)
    ready_value = {
        "schema_version": 1,
        "profile": READY_PROFILE,
        "status": "ready_for_simulation_app_close",
        "startup_diagnostic": STARTUP_DIAGNOSTIC_MODE,
        "preexec": preexec_identity,
        "preclose": preclose_identity,
        "zero_work_counters": dict(ZERO_WORK_COUNTERS),
        "bounded_diagnostic_counts": {
            "nvidia_smi_invocations": 2,
            "scheduler_request_artifacts": 1,
            "scheduler_handoff_artifacts": 1,
            "job_scheduler_records": 1,
            "step_scheduler_records": 1,
            "preexec_artifacts": 1,
            "preclose_artifacts": 1,
            "ready_artifacts": 1,
            "simulation_app_close_calls": 0,
        },
    }
    ready_identity = publish_immutable_json(
        ready_path,
        ready_value,
        expected_parent_identity=task_dir_identity,
    )
    try:
        simulation_app.close()
    except _CaughtSignal:
        raise
    except BaseException as error:
        print(
            "POLARIS_STARTUP_DIAGNOSTIC_CLOSE_ERROR="
            f"{type(error).__module__}.{type(error).__qualname__}",
            file=sys.stderr,
            flush=True,
        )
        raise RuntimeError(
            "AppLauncher diagnostic SimulationApp.close() failed"
        ) from error
    return {
        "preexec": preexec_identity,
        "preclose": preclose_identity,
        "ready": ready_identity,
    }


def run_app_launcher_only_diagnostic(
    *,
    simulation_app: Any,
    preexec_path: Path,
    preclose_path: Path,
    expected_gpu_uuid: str,
) -> dict[str, Any]:
    """Run the imported boundary with scoped caught-signal close-once cleanup."""

    close_attempted = False

    class CloseOnce:
        def close(self) -> None:
            nonlocal close_attempted
            if close_attempted:
                raise RuntimeError("SimulationApp.close() was requested more than once")
            close_attempted = True
            simulation_app.close()

    previous_handlers = _install_caught_signal_handlers()
    try:
        return _run_app_launcher_only_diagnostic(
            simulation_app=CloseOnce(),
            preexec_path=preexec_path,
            preclose_path=preclose_path,
            expected_gpu_uuid=expected_gpu_uuid,
        )
    except _CaughtSignal:
        # A Kit close may terminate the process with exit code zero.  On the
        # caught-signal path, preserve the signal-derived failure for the
        # outer worker/Slurm cleanup instead of invoking hard-exit-capable
        # application teardown in this process.
        raise
    finally:
        _restore_signal_handlers(previous_handlers)


def prepare_public_eval_exec(
    *,
    target_argv: Sequence[str],
    preexec_path: Path,
    preclose_path: Path,
    expected_gpu_uuid: str,
    source_root: Path,
    data_root: Path,
    cache_root: Path,
    scheduler_request_path: Path,
    scheduler_handoff_path: Path,
    run_dir: Path,
    task_dir: Path,
    namespace_parent_identity: str,
    run_dir_identity: str,
    task_dir_identity: str,
) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    slurm = _validate_slurm_gpu_environment(os.environ)
    approvals = _validate_execution_approval_environment(os.environ)
    expected_port = 20000 + int(slurm["SLURM_JOB_ID"]) % 20000
    target = validate_target_argv(
        target_argv,
        preexec_path=preexec_path,
        preclose_path=preclose_path,
        expected_gpu_uuid=expected_gpu_uuid,
        expected_port=expected_port,
    )
    request_identity = publish_scheduler_request(
        scheduler_request_path,
        environ=os.environ,
        expected_gpu_uuid=expected_gpu_uuid,
        expected_parent_identity=task_dir_identity,
    )
    request, _ = stable_read_immutable_json(
        scheduler_request_path,
        expected_sha256=request_identity["sha256"],
        expected_parent_identity=task_dir_identity,
    )
    request = _validate_scheduler_request(request)
    handoff_value, handoff_identity = wait_for_scheduler_handoff(
        scheduler_handoff_path,
        request=request,
        request_identity=request_identity,
        expected_parent_identity=task_dir_identity,
        **_scheduler_approval_kwargs(approvals),
    )
    scheduler_runtime = {
        "artifact": handoff_identity,
        "value": handoff_value,
    }
    runtime = capture_runtime_context(
        python_argv=python_argv_after_exec(target),
        source_root=source_root,
        expected_gpu_uuid=expected_gpu_uuid,
        scheduler_handoff=scheduler_runtime,
        output_directories={
            "namespace_parent": str(run_dir.parent),
            "namespace_parent_identity": namespace_parent_identity,
            "run_dir": str(run_dir),
            "run_identity": run_dir_identity,
            "task_dir": str(task_dir),
            "task_identity": task_dir_identity,
        },
    )
    preexec_value = {
        "schema_version": 1,
        "profile": PREEXEC_PROFILE,
        "status": "captured_before_public_eval_exec",
        "startup_diagnostic": STARTUP_DIAGNOSTIC_MODE,
        "runtime": runtime,
        "launcher_argv": list(sys.argv),
        "target_argv": target,
        "zero_work_counters": dict(ZERO_WORK_COUNTERS),
        "bounded_diagnostic_counts": {
            "nvidia_smi_invocations": 1,
            "scheduler_request_artifacts": 1,
            "scheduler_handoff_artifacts": 1,
            "job_scheduler_records": 1,
            "step_scheduler_records": 1,
            "preexec_artifacts": 1,
            "preclose_artifacts": 0,
            "ready_artifacts": 0,
            "simulation_app_close_calls": 0,
        },
    }
    preexec_identity = publish_immutable_json(
        preexec_path,
        preexec_value,
        expected_parent_identity=task_dir_identity,
    )
    closed_environment = build_closed_eval_environment(
        inherited=os.environ,
        source_root=source_root,
        data_root=data_root,
        cache_root=cache_root,
        preexec_sha256=preexec_identity["sha256"],
        scheduler_handoff_path=scheduler_handoff_path,
        scheduler_handoff_sha256=handoff_identity["sha256"],
        run_dir=run_dir,
        task_dir=task_dir,
        namespace_parent_identity=namespace_parent_identity,
        run_dir_identity=run_dir_identity,
        task_dir_identity=task_dir_identity,
    )
    return target, closed_environment, preexec_identity


def _parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture a policy-free public evaluator pre-exec boundary"
    )
    parser.add_argument("--preexec-output", type=Path, required=True)
    parser.add_argument("--preclose-output", type=Path, required=True)
    parser.add_argument("--expected-batch-gpu-uuid", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--scheduler-request", type=Path, required=True)
    parser.add_argument("--scheduler-handoff", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--namespace-parent-identity", required=True)
    parser.add_argument("--run-dir-identity", required=True)
    parser.add_argument("--task-dir-identity", required=True)
    parser.add_argument("target", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.target[:1] == ["--"]:
        args.target = args.target[1:]
    return args


def _main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv[:1] == ["wait-scheduler-request"]:
        parser = argparse.ArgumentParser()
        parser.add_argument("command")
        parser.add_argument("--request", type=Path, required=True)
        parser.add_argument("--timeout-seconds", type=float, default=60.0)
        parser.add_argument("--task-dir-identity", required=True)
        args = parser.parse_args(raw_argv)
        request, _ = wait_for_scheduler_request(
            args.request,
            timeout_seconds=args.timeout_seconds,
            expected_parent_identity=args.task_dir_identity,
        )
        print(f"{request['job_id']}\t{request['step_id']}", flush=True)
        return 0
    if raw_argv[:1] == ["seal-scheduler-handoff"]:
        parser = argparse.ArgumentParser()
        parser.add_argument("command")
        parser.add_argument("--request", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--job-record", required=True)
        parser.add_argument("--step-record", required=True)
        parser.add_argument("--scheduler-client-json", required=True)
        parser.add_argument("--task-dir-identity", required=True)
        args = parser.parse_args(raw_argv)
        approvals = _validate_execution_approval_environment(os.environ)
        scheduler_client = json.loads(args.scheduler_client_json)
        if (
            canonical_json_bytes(scheduler_client).decode("ascii").rstrip("\n")
            != args.scheduler_client_json
        ):
            raise ValueError("scheduler client CLI JSON is not canonical")
        seal_scheduler_handoff(
            request_path=args.request,
            output_path=args.output,
            job_record=args.job_record,
            step_record=args.step_record,
            scheduler_client=scheduler_client,
            expected_parent_identity=args.task_dir_identity,
            **_scheduler_approval_kwargs(approvals),
        )
        return 0
    if raw_argv[:1] == ["scheduler-client-identity"]:
        parser = argparse.ArgumentParser()
        parser.add_argument("command")
        parser.parse_args(raw_argv)
        approvals = _validate_execution_approval_environment(os.environ)
        sys.stdout.buffer.write(
            canonical_json_bytes(
                capture_scheduler_client_identity(
                    **_scheduler_approval_kwargs(approvals),
                )
            )
        )
        return 0
    if raw_argv[:1] == ["broker-scheduler-handoff"]:
        parser = argparse.ArgumentParser()
        parser.add_argument("command")
        parser.add_argument("--request", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--terminal-request", type=Path, required=True)
        parser.add_argument("--terminal-output", type=Path, required=True)
        parser.add_argument("--task-dir-identity", required=True)
        parser.add_argument("--timeout-seconds", type=float, default=30.0)
        parser.add_argument("--terminal-timeout-seconds", type=float, default=120.0)
        args = parser.parse_args(raw_argv)
        broker_scheduler_handoff(
            request_path=args.request,
            output_path=args.output,
            terminal_request_path=args.terminal_request,
            terminal_output_path=args.terminal_output,
            expected_parent_identity=args.task_dir_identity,
            timeout_seconds=args.timeout_seconds,
            terminal_timeout_seconds=args.terminal_timeout_seconds,
        )
        return 0
    if raw_argv[:1] == ["publish-scheduler-terminal-request"]:
        parser = argparse.ArgumentParser()
        parser.add_argument("command")
        parser.add_argument("--request", type=Path, required=True)
        parser.add_argument("--handoff", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--srun-exit-code", type=int, required=True)
        parser.add_argument("--task-dir-identity", required=True)
        args = parser.parse_args(raw_argv)
        publish_scheduler_terminal_request(
            request_path=args.request,
            handoff_path=args.handoff,
            output_path=args.output,
            srun_exit_code=args.srun_exit_code,
            expected_parent_identity=args.task_dir_identity,
        )
        return 0
    if raw_argv[:1] == ["create-output-directories"]:
        parser = argparse.ArgumentParser()
        parser.add_argument("command")
        parser.add_argument("--run-dir", type=Path, required=True)
        parser.add_argument("--task-dir", type=Path, required=True)
        parser.add_argument("--namespace-parent-identity", required=True)
        args = parser.parse_args(raw_argv)
        created = create_output_directories(
            run_dir=args.run_dir,
            task_dir=args.task_dir,
            expected_parent_identity=args.namespace_parent_identity,
        )
        print(
            f"{created['namespace_parent_identity']}\t"
            f"{created['run_identity']}\t{created['task_identity']}",
            flush=True,
        )
        return 0
    if raw_argv[:1] == ["capture-pyxis-image-identity"]:
        parser = argparse.ArgumentParser()
        parser.add_argument("command")
        parser.add_argument("--path", type=Path, required=True)
        parser.add_argument("--expected-sha256", required=True)
        args = parser.parse_args(raw_argv)
        result = capture_pyxis_image_identity(
            args.path, expected_sha256=args.expected_sha256
        )
        observed = result["observed"]
        print(
            f"{observed['mode']}\t{observed['nlink']}\t"
            f"{observed['size']}\t{observed['sha256']}",
            flush=True,
        )
        return 0
    if raw_argv[:1] == ["immutable-log-tee"]:
        parser = argparse.ArgumentParser()
        parser.add_argument("command")
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--identity-output", type=Path, required=True)
        parser.add_argument("--task-dir-identity", required=True)
        args = parser.parse_args(raw_argv)
        capture_immutable_log(
            output_path=args.output,
            identity_path=args.identity_output,
            input_stream=sys.stdin.buffer,
            mirror_stream=sys.stdout.buffer,
            expected_parent_identity=args.task_dir_identity,
        )
        return 0
    if raw_argv[:1] == ["validate-immutable-log"]:
        parser = argparse.ArgumentParser()
        parser.add_argument("command")
        parser.add_argument("--identity", type=Path, required=True)
        parser.add_argument("--task-dir-identity", required=True)
        args = parser.parse_args(raw_argv)
        result = validate_immutable_log_identity(
            args.identity, expected_parent_identity=args.task_dir_identity
        )
        print(result["log"]["sha256"], flush=True)
        return 0
    if raw_argv[:1] == ["publish-failure-attestation"]:
        parser = argparse.ArgumentParser()
        parser.add_argument("command")
        parser.add_argument("--task-dir", type=Path, required=True)
        parser.add_argument("--task-dir-identity", required=True)
        parser.add_argument("--primary-exit-code", type=int, required=True)
        parser.add_argument("--srun-exit-code", type=int, required=True)
        parser.add_argument("--log-exit-code", type=int, required=True)
        parser.add_argument("--helper-exit-code", type=int, required=True)
        parser.add_argument("--signal", required=True)
        args = parser.parse_args(raw_argv)
        publish_failure_attestation(
            task_dir=args.task_dir,
            primary_exit_code=args.primary_exit_code,
            srun_exit_code=args.srun_exit_code,
            log_exit_code=args.log_exit_code,
            helper_exit_code=args.helper_exit_code,
            signal_name=args.signal,
            expected_task_identity=args.task_dir_identity,
        )
        return 0
    if raw_argv[:1] == ["cleanup-transients"]:
        parser = argparse.ArgumentParser()
        parser.add_argument("command")
        parser.add_argument("--task-dir", type=Path, required=True)
        parser.add_argument("--task-dir-identity", required=True)
        args = parser.parse_args(raw_argv)
        removed = cleanup_transient_evidence(
            task_dir=args.task_dir,
            expected_task_identity=args.task_dir_identity,
        )
        print(len(removed), flush=True)
        return 0
    if raw_argv[:1] == ["seal-evidence-tree"]:
        parser = argparse.ArgumentParser()
        parser.add_argument("command")
        parser.add_argument("--task-dir", type=Path, required=True)
        parser.add_argument("--run-dir", type=Path, required=True)
        parser.add_argument("--task-dir-identity", required=True)
        parser.add_argument("--run-dir-identity", required=True)
        parser.add_argument("--namespace-parent-identity", required=True)
        parser.add_argument("--outcome", choices=("success", "failure"), required=True)
        parser.add_argument("--srun-exit-code", type=int)
        parser.add_argument("--log-exit-code", type=int)
        parser.add_argument("--helper-exit-code", type=int)
        args = parser.parse_args(raw_argv)
        result = seal_evidence_tree(
            task_dir=args.task_dir,
            run_dir=args.run_dir,
            outcome=args.outcome,
            expected_namespace_parent_identity=args.namespace_parent_identity,
            expected_task_identity=args.task_dir_identity,
            expected_run_identity=args.run_dir_identity,
            srun_exit_code=args.srun_exit_code,
            log_exit_code=args.log_exit_code,
            helper_exit_code=args.helper_exit_code,
        )
        if args.outcome == "success":
            print(
                f"{result['termination_mode']}\t{result['log_sha256']}",
                flush=True,
            )
        else:
            sys.stdout.buffer.write(canonical_json_bytes(result))
        return 0
    args = _parse_cli(raw_argv)
    target, environment, _ = prepare_public_eval_exec(
        target_argv=args.target,
        preexec_path=args.preexec_output,
        preclose_path=args.preclose_output,
        expected_gpu_uuid=args.expected_batch_gpu_uuid,
        source_root=args.source_root,
        data_root=args.data_root,
        cache_root=args.cache_root,
        scheduler_request_path=args.scheduler_request,
        scheduler_handoff_path=args.scheduler_handoff,
        run_dir=args.run_dir,
        task_dir=args.task_dir,
        namespace_parent_identity=args.namespace_parent_identity,
        run_dir_identity=args.run_dir_identity,
        task_dir_identity=args.task_dir_identity,
    )
    os.execve(target[0], target, environment)
    raise AssertionError("os.execve unexpectedly returned")


def main(argv: Sequence[str] | None = None) -> int:
    previous_handlers = _install_caught_signal_handlers()
    try:
        return _main(argv)
    except _CaughtSignal as error:
        return 128 + error.signum
    finally:
        _restore_signal_handlers(previous_handlers)


if __name__ == "__main__":
    raise SystemExit(main())
