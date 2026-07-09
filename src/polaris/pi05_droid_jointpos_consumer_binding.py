"""Run-local consumer binding for the official pi0.5 PolaRiS evaluation.

The released checkpoint is intentionally not copied.  Instead, this module
opens every manifest object without following symlinks, hashes the opened
descriptors, records path/inode identity, and keeps those descriptors alive
while the policy is restored and served.  The same closed inventory is
captured before model load, after model load, and after rollout.

This closes ordinary path replacement and persistent mutation races.  It does
not claim protection from a malicious process with the same uid, which can
modify owned inodes or the consumer process itself.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import shutil
import stat
from typing import Any, Mapping, Sequence


CONSUMER_BINDING_SCHEMA_VERSION = 1
CONSUMER_BINDING_PROFILE = "openpi_pi05_droid_jointpos_consumer_binding_v1"
CONSUMER_BINDING_STAGES = ("pre_load", "post_load", "postrun")
CHECKPOINT_MANIFEST_SHA256 = (
    "7abd0c2294d442d429a77655783232206b2b30d95c508d435503135a5523a11c"
)
CHECKPOINT_MANIFEST_PREFIX = "checkpoints/polaris/pi05_droid_jointpos_polaris/"
CHECKPOINT_OBJECT_COUNT = 27
CHECKPOINT_TOTAL_BYTES = 12_434_530_837
TOKENIZER_URI = "gs://big_vision/paligemma_tokenizer.model"
TOKENIZER_GENERATION = "1711547605575873"
TOKENIZER_SIZE = 4_264_023
TOKENIZER_MD5_BASE64 = "FCCtyYVnIKVZ6KhyhLGV4g=="
TOKENIZER_SHA256 = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
POLARIS_COMMIT = "c5b52a9cebb2c797a84e3df374b6002005d20a4f"
POLARIS_TREE = "7fd5e1b0af26577fd323fb1d7f3595b91282e73f"
OPENPI_COMMIT = "bd70b8f4011e85b3f3b0f039f12113f78718e7bf"
SOURCE_APPROVAL_PROFILE = "openpi_pi05_droid_jointpos_source_approval_v1"
SOURCE_APPROVAL_TRUSTED_HASHER_SHA256 = (
    "7117f1455eb2cd9dc1a96d6e5e91adad59c411152e95b970771cda4f562c90f2"
)

_IDENTITY_FIELDS = (
    "device",
    "inode",
    "mode",
    "uid",
    "gid",
    "link_count",
    "size",
    "mtime_ns",
    "ctime_ns",
)
_SOURCE_REQUIRED_PATHS = (
    "scripts/eval.py",
    "scripts/polaris/eval_pi05_droid_jointpos_polaris.sh",
    "scripts/polaris/serve_pi05_droid_jointpos_attested.py",
    "src/polaris/pi05_droid_jointpos_consumer_binding.py",
    "src/polaris/pi05_droid_jointpos_serving_contract.py",
    "src/polaris/policy/droid_jointpos_client.py",
    "third_party/openpi/packages/openpi-client/src/openpi_client/websocket_client_policy.py",
)


class ConsumerBindingError(ValueError):
    """A consumer input failed closed identity validation."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _reject_duplicate_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _stat_identity(value: os.stat_result) -> dict[str, Any]:
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "mode": f"{stat.S_IMODE(value.st_mode):04o}",
        "uid": value.st_uid,
        "gid": value.st_gid,
        "link_count": value.st_nlink,
        "size": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return all(
        getattr(left, f"st_{field}") == getattr(right, f"st_{field}")
        for field in (
            "dev",
            "ino",
            "mode",
            "uid",
            "gid",
            "nlink",
            "size",
            "mtime_ns",
            "ctime_ns",
        )
    )


def _descriptor_hashes(descriptor: int) -> tuple[int, str, str]:
    size = os.fstat(descriptor).st_size
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    offset = 0
    while offset < size:
        block = os.pread(descriptor, min(16 * 1024 * 1024, size - offset), offset)
        if not block:
            raise ConsumerBindingError("descriptor became short while hashing")
        md5.update(block)
        sha256.update(block)
        offset += len(block)
    if os.pread(descriptor, 1, size):
        raise ConsumerBindingError("descriptor grew while hashing")
    return size, base64.b64encode(md5.digest()).decode("ascii"), sha256.hexdigest()


def _validate_relative_path(raw: str) -> tuple[str, ...]:
    path = PurePosixPath(raw)
    if (
        not raw
        or raw == "."
        or path.is_absolute()
        or path.as_posix() != raw
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ConsumerBindingError(f"unsafe relative path: {raw!r}")
    return path.parts


def _open_directory_path(root_fd: int, parts: Sequence[str]) -> int:
    current = os.dup(root_fd)
    try:
        for part in parts:
            next_descriptor = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=current,
            )
            os.close(current)
            current = next_descriptor
        return current
    except BaseException:
        os.close(current)
        raise


def _open_regular_beneath(root_fd: int, relative: str) -> int:
    parts = _validate_relative_path(relative)
    parent_fd = _open_directory_path(root_fd, parts[:-1])
    try:
        descriptor = os.open(
            parts[-1],
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
    finally:
        os.close(parent_fd)
    identity = os.fstat(descriptor)
    if not stat.S_ISREG(identity.st_mode):
        os.close(descriptor)
        raise ConsumerBindingError(f"consumer object is not regular: {relative}")
    return descriptor


def _open_root(path: Path) -> tuple[Path, int, os.stat_result]:
    requested = Path(path)
    if requested.is_symlink() or not requested.is_absolute():
        raise ConsumerBindingError(f"consumer root must be absolute and non-symlinked: {requested}")
    filesystem_root = os.open(
        "/", os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        try:
            descriptor = _open_directory_path(filesystem_root, requested.parts[1:])
        except OSError as error:
            raise ConsumerBindingError(
                f"consumer root has an unsafe or missing path component: {requested}"
            ) from error
    finally:
        os.close(filesystem_root)
    identity = os.fstat(descriptor)
    path_identity = os.stat(requested, follow_symlinks=False)
    if not stat.S_ISDIR(identity.st_mode) or not _same_identity(identity, path_identity):
        os.close(descriptor)
        raise ConsumerBindingError(f"consumer root changed while opening: {requested}")
    return requested, descriptor, identity


def _manifest_entries(manifest: Path) -> list[tuple[str, int, str]]:
    requested = Path(manifest)
    if requested.is_symlink() or not requested.is_file():
        raise ConsumerBindingError("checkpoint manifest must be one regular file")
    payload = requested.read_bytes()
    if hashlib.sha256(payload).hexdigest() != CHECKPOINT_MANIFEST_SHA256:
        raise ConsumerBindingError("checkpoint manifest SHA-256 mismatch")
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ConsumerBindingError("checkpoint manifest is not ASCII") from error
    entries: list[tuple[str, int, str]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        fields = line.split("\t")
        if len(fields) != 3 or not fields[0].startswith(CHECKPOINT_MANIFEST_PREFIX):
            raise ConsumerBindingError(f"invalid checkpoint manifest line {line_number}")
        relative = fields[0][len(CHECKPOINT_MANIFEST_PREFIX) :]
        _validate_relative_path(relative)
        try:
            size = int(fields[1])
            decoded = base64.b64decode(fields[2], validate=True)
        except (TypeError, ValueError) as error:
            raise ConsumerBindingError(f"invalid checkpoint manifest line {line_number}") from error
        if relative in seen or size < 0 or len(decoded) != 16:
            raise ConsumerBindingError(f"invalid checkpoint manifest line {line_number}")
        seen.add(relative)
        entries.append((relative, size, fields[2]))
    if len(entries) != CHECKPOINT_OBJECT_COUNT or sum(item[1] for item in entries) != CHECKPOINT_TOTAL_BYTES:
        raise ConsumerBindingError("checkpoint manifest inventory mismatch")
    return entries


def _walk_closed_tree(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = {"."}
    for current, names, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        current_stat = os.stat(current_path, follow_symlinks=False)
        if not stat.S_ISDIR(current_stat.st_mode):
            raise ConsumerBindingError(f"tree contains a non-directory: {current_path}")
        relative_current = current_path.relative_to(root).as_posix() or "."
        directories.add(relative_current)
        for name in names:
            child = current_path / name
            child_stat = os.stat(child, follow_symlinks=False)
            if not stat.S_ISDIR(child_stat.st_mode):
                raise ConsumerBindingError(f"tree contains a symlink or non-directory: {child}")
            directories.add(child.relative_to(root).as_posix())
        for name in filenames:
            child = current_path / name
            child_stat = os.stat(child, follow_symlinks=False)
            if not stat.S_ISREG(child_stat.st_mode):
                raise ConsumerBindingError(f"tree contains a symlink or non-file: {child}")
            files.add(child.relative_to(root).as_posix())
    return files, directories


class CheckpointBinding:
    """Open descriptor set for one exact checkpoint inventory."""

    def __init__(self, checkpoint: Path, manifest: Path) -> None:
        self.root_fd = -1
        self.directory_fds: dict[str, int] = {}
        self.file_fds: dict[str, int] = {}
        try:
            self.root, self.root_fd, self.root_identity = _open_root(Path(checkpoint))
            self.manifest = Path(manifest).resolve(strict=True)
            self.entries = _manifest_entries(self.manifest)
            expected_files = {entry[0] for entry in self.entries}
            actual_files, actual_directories = _walk_closed_tree(self.root)
            expected_directories = {"."}
            for relative in expected_files:
                parent = PurePosixPath(relative).parent
                while str(parent) not in {"", "."}:
                    expected_directories.add(parent.as_posix())
                    parent = parent.parent
            if (
                actual_files != expected_files
                or actual_directories != expected_directories
            ):
                raise ConsumerBindingError(
                    "checkpoint closure mismatch: "
                    f"missing_files={sorted(expected_files - actual_files)!r}, "
                    f"extra_files={sorted(actual_files - expected_files)!r}, "
                    f"missing_dirs={sorted(expected_directories - actual_directories)!r}, "
                    f"extra_dirs={sorted(actual_directories - expected_directories)!r}"
                )
            self.expected_files = expected_files
            self.expected_directories = expected_directories
            for relative in sorted(self.expected_directories - {"."}):
                self.directory_fds[relative] = _open_directory_path(
                    self.root_fd, _validate_relative_path(relative)
                )
            for relative, _size, _md5 in self.entries:
                self.file_fds[relative] = _open_regular_beneath(self.root_fd, relative)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        for descriptor in [*self.file_fds.values(), *self.directory_fds.values()]:
            with contextlib.suppress(OSError):
                os.close(descriptor)
        self.file_fds = {}
        self.directory_fds = {}
        if getattr(self, "root_fd", -1) >= 0:
            with contextlib.suppress(OSError):
                os.close(self.root_fd)
            self.root_fd = -1

    def __enter__(self) -> "CheckpointBinding":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def snapshot(self) -> dict[str, Any]:
        current_root = os.stat(self.root, follow_symlinks=False)
        opened_root = os.fstat(self.root_fd)
        if not _same_identity(self.root_identity, opened_root) or not _same_identity(opened_root, current_root):
            raise ConsumerBindingError("checkpoint root inode changed")
        actual_files, actual_directories = _walk_closed_tree(self.root)
        if (
            actual_files != self.expected_files
            or actual_directories != self.expected_directories
        ):
            raise ConsumerBindingError("checkpoint closure changed")
        directories = [{"relative_path": ".", **_stat_identity(opened_root)}]
        for relative, descriptor in sorted(self.directory_fds.items()):
            opened = os.fstat(descriptor)
            path_identity = os.stat(
                relative, dir_fd=self.root_fd, follow_symlinks=False
            )
            if not _same_identity(opened, path_identity):
                raise ConsumerBindingError(f"checkpoint directory inode changed: {relative}")
            directories.append({"relative_path": relative, **_stat_identity(opened)})
        objects = []
        entries_by_path = {relative: (size, md5) for relative, size, md5 in self.entries}
        for relative, descriptor in self.file_fds.items():
            before = os.fstat(descriptor)
            size, md5, sha256 = _descriptor_hashes(descriptor)
            after = os.fstat(descriptor)
            path_identity = os.stat(
                relative, dir_fd=self.root_fd, follow_symlinks=False
            )
            expected_size, expected_md5 = entries_by_path[relative]
            if (
                not _same_identity(before, after)
                or not _same_identity(after, path_identity)
                or size != expected_size
                or md5 != expected_md5
            ):
                raise ConsumerBindingError(f"checkpoint object changed: {relative}")
            objects.append(
                {
                    "relative_path": relative,
                    **_stat_identity(after),
                    "md5_base64": md5,
                    "sha256": sha256,
                }
            )
        objects.sort(key=lambda item: item["relative_path"])
        directories.sort(key=lambda item: item["relative_path"])
        content_objects = [
            {
                "relative_path": item["relative_path"],
                "size": item["size"],
                "md5_base64": item["md5_base64"],
                "sha256": item["sha256"],
            }
            for item in objects
        ]
        return {
            "root": str(self.root),
            "manifest_path": str(self.manifest),
            "manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
            "object_count": len(objects),
            "total_bytes": sum(item["size"] for item in objects),
            "objects_sha256": _sha256(content_objects),
            "inode_closure_sha256": _sha256(
                {"directories": directories, "objects": objects}
            ),
            "directories": directories,
            "objects": objects,
        }


class TokenizerBinding:
    def __init__(self, tokenizer: Path) -> None:
        self.parent_fd = -1
        self.fd = -1
        requested = Path(tokenizer)
        if requested.is_symlink() or not requested.is_absolute():
            raise ConsumerBindingError("tokenizer path must be absolute and non-symlinked")
        try:
            parent, self.parent_fd, self.parent_identity = _open_root(requested.parent)
            self.path = parent / requested.name
            self.name = requested.name
            self.fd = _open_regular_beneath(self.parent_fd, self.name)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        for name in ("fd", "parent_fd"):
            descriptor = getattr(self, name, -1)
            if descriptor >= 0:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
                setattr(self, name, -1)

    def __enter__(self) -> "TokenizerBinding":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def snapshot(self) -> dict[str, Any]:
        parent_opened = os.fstat(self.parent_fd)
        parent_path = os.stat(self.path.parent, follow_symlinks=False)
        before = os.fstat(self.fd)
        size, md5, sha256 = _descriptor_hashes(self.fd)
        after = os.fstat(self.fd)
        path_identity = os.stat(
            self.name, dir_fd=self.parent_fd, follow_symlinks=False
        )
        if (
            not _same_identity(self.parent_identity, parent_opened)
            or not _same_identity(parent_opened, parent_path)
            or not _same_identity(before, after)
            or not _same_identity(after, path_identity)
            or size != TOKENIZER_SIZE
            or md5 != TOKENIZER_MD5_BASE64
            or sha256 != TOKENIZER_SHA256
        ):
            raise ConsumerBindingError("tokenizer identity changed")
        return {
            "uri": TOKENIZER_URI,
            "generation": TOKENIZER_GENERATION,
            "path": str(self.path),
            **_stat_identity(after),
            "md5_base64": md5,
            "sha256": sha256,
        }


class SourceBinding:
    """Descriptor-backed closure for a content-addressed source snapshot."""

    def __init__(self, source: Path, expected_tree_sha256: str) -> None:
        if len(expected_tree_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_tree_sha256
        ):
            raise ConsumerBindingError("source tree identity must be one lowercase SHA-256")
        self.root_fd = -1
        self.directory_fds: dict[str, int] = {}
        self.file_fds: dict[str, int] = {}
        try:
            self.expected_tree_sha256 = expected_tree_sha256
            self.root, self.root_fd, self.root_identity = _open_root(Path(source))
            actual_files, actual_directories = _walk_closed_tree(self.root)
            missing = set(_SOURCE_REQUIRED_PATHS) - actual_files
            if missing:
                raise ConsumerBindingError(
                    f"source snapshot lacks runtime paths: {sorted(missing)!r}"
                )
            self.expected_files = actual_files
            self.expected_directories = actual_directories
            for relative in sorted(self.expected_directories - {"."}):
                self.directory_fds[relative] = _open_directory_path(
                    self.root_fd, _validate_relative_path(relative)
                )
            for relative in sorted(self.expected_files):
                self.file_fds[relative] = _open_regular_beneath(self.root_fd, relative)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        for descriptor in [*self.file_fds.values(), *self.directory_fds.values()]:
            with contextlib.suppress(OSError):
                os.close(descriptor)
        self.file_fds = {}
        self.directory_fds = {}
        if getattr(self, "root_fd", -1) >= 0:
            with contextlib.suppress(OSError):
                os.close(self.root_fd)
            self.root_fd = -1

    def __enter__(self) -> "SourceBinding":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def snapshot(self) -> dict[str, Any]:
        current_root = os.stat(self.root, follow_symlinks=False)
        opened_root = os.fstat(self.root_fd)
        if not _same_identity(self.root_identity, opened_root) or not _same_identity(opened_root, current_root):
            raise ConsumerBindingError("source root inode changed")
        actual_files, actual_directories = _walk_closed_tree(self.root)
        if (
            actual_files != self.expected_files
            or actual_directories != self.expected_directories
        ):
            raise ConsumerBindingError("source closure changed")
        directories = [{"relative_path": ".", **_stat_identity(opened_root)}]
        for relative, descriptor in sorted(self.directory_fds.items()):
            opened = os.fstat(descriptor)
            path_identity = os.stat(
                relative, dir_fd=self.root_fd, follow_symlinks=False
            )
            if not _same_identity(opened, path_identity):
                raise ConsumerBindingError(f"source directory inode changed: {relative}")
            directories.append({"relative_path": relative, **_stat_identity(opened)})
        files = []
        for relative, descriptor in sorted(self.file_fds.items()):
            before = os.fstat(descriptor)
            size, _md5, sha256 = _descriptor_hashes(descriptor)
            after = os.fstat(descriptor)
            path_identity = os.stat(
                relative, dir_fd=self.root_fd, follow_symlinks=False
            )
            if not _same_identity(before, after) or not _same_identity(after, path_identity):
                raise ConsumerBindingError(f"source file changed: {relative}")
            files.append(
                {
                    "relative_path": relative,
                    **_stat_identity(after),
                    "sha256": sha256,
                }
            )
        directories.sort(key=lambda item: item["relative_path"])
        files.sort(key=lambda item: item["relative_path"])
        content = {
            "directories": [
                {
                    "relative_path": item["relative_path"],
                    "mode": item["mode"],
                }
                for item in directories
            ],
            "files": [
                {
                    "relative_path": item["relative_path"],
                    "mode": item["mode"],
                    "size": item["size"],
                    "sha256": item["sha256"],
                }
                for item in files
            ],
        }
        tree_sha256 = _sha256(content)
        if tree_sha256 != self.expected_tree_sha256:
            raise ConsumerBindingError(
                f"source tree SHA-256 mismatch: expected {self.expected_tree_sha256}, got {tree_sha256}"
            )
        return {
            "root": str(self.root),
            "polaris_base_commit": POLARIS_COMMIT,
            "polaris_base_tree": POLARIS_TREE,
            "openpi_commit": OPENPI_COMMIT,
            "tree_sha256": tree_sha256,
            "file_count": len(files),
            "directory_count": len(directories),
            "total_bytes": sum(item["size"] for item in files),
            "inode_closure_sha256": _sha256(
                {"directories": directories, "files": files}
            ),
            "directories": directories,
            "files": files,
        }


class ConsumerBinding:
    """Keep checkpoint, tokenizer, and source descriptors alive together."""

    def __init__(
        self,
        *,
        checkpoint: Path,
        manifest: Path,
        tokenizer: Path,
        source: Path,
        expected_source_tree_sha256: str,
    ) -> None:
        self.checkpoint = CheckpointBinding(checkpoint, manifest)
        try:
            self.tokenizer = TokenizerBinding(tokenizer)
            self.source = SourceBinding(source, expected_source_tree_sha256)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        for item in ("source", "tokenizer", "checkpoint"):
            value = getattr(self, item, None)
            if value is not None:
                value.close()

    def __enter__(self) -> "ConsumerBinding":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def snapshot(self, stage: str) -> dict[str, Any]:
        if stage not in CONSUMER_BINDING_STAGES:
            raise ConsumerBindingError(f"unsupported consumer-binding stage: {stage}")
        identity = {
            "checkpoint": self.checkpoint.snapshot(),
            "tokenizer": self.tokenizer.snapshot(),
            "source": self.source.snapshot(),
        }
        return {
            "schema_version": CONSUMER_BINDING_SCHEMA_VERSION,
            "profile": CONSUMER_BINDING_PROFILE,
            "status": "pass",
            "stage": stage,
            "identity": identity,
            "binding_sha256": _sha256(identity),
            "residual_trust": "same_uid_transient_mutation_and_process_tampering_out_of_scope",
        }


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_absolute_report_path(value: Any, label: str) -> None:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ConsumerBindingError(f"invalid {label} path")


def _validate_inventory_record(
    value: Any,
    *,
    extra_fields: set[str],
    allow_root: bool,
) -> dict[str, Any]:
    required = {"relative_path", *_IDENTITY_FIELDS, *extra_fields}
    if not isinstance(value, dict) or set(value) != required:
        raise ConsumerBindingError("consumer-binding inventory record schema mismatch")
    relative = value["relative_path"]
    if relative != "." or not allow_root:
        if not isinstance(relative, str):
            raise ConsumerBindingError("consumer-binding inventory path is invalid")
        _validate_relative_path(relative)
    integer_fields = set(_IDENTITY_FIELDS) - {"mode"}
    if any(
        type(value[field]) is not int or value[field] < 0 for field in integer_fields
    ) or value["link_count"] < 1:
        raise ConsumerBindingError("consumer-binding inventory identity is invalid")
    mode = value["mode"]
    if (
        not isinstance(mode, str)
        or len(mode) != 4
        or any(character not in "01234567" for character in mode)
    ):
        raise ConsumerBindingError("consumer-binding inventory mode is invalid")
    return value


def _validate_checkpoint_report(value: Any) -> None:
    required = {
        "root",
        "manifest_path",
        "manifest_sha256",
        "object_count",
        "total_bytes",
        "objects_sha256",
        "inode_closure_sha256",
        "directories",
        "objects",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ConsumerBindingError("checkpoint consumer-binding schema mismatch")
    _validate_absolute_report_path(value["root"], "checkpoint root")
    _validate_absolute_report_path(value["manifest_path"], "checkpoint manifest")
    directories = value["directories"]
    objects = value["objects"]
    if not isinstance(directories, list) or not isinstance(objects, list):
        raise ConsumerBindingError("checkpoint consumer-binding inventory is invalid")
    for item in directories:
        _validate_inventory_record(item, extra_fields=set(), allow_root=True)
    for item in objects:
        _validate_inventory_record(
            item,
            extra_fields={"md5_base64", "sha256"},
            allow_root=False,
        )
        try:
            decoded_md5 = base64.b64decode(item["md5_base64"], validate=True)
        except (TypeError, ValueError) as error:
            raise ConsumerBindingError("checkpoint object MD5 is invalid") from error
        if len(decoded_md5) != 16 or not _valid_sha256(item["sha256"]):
            raise ConsumerBindingError("checkpoint object digest is invalid")
    directory_paths = [item["relative_path"] for item in directories]
    object_paths = [item["relative_path"] for item in objects]
    manifest_entries = _manifest_entries(Path(value["manifest_path"]))
    expected_objects = {
        relative: (size, md5) for relative, size, md5 in manifest_entries
    }
    expected_directories = {"."}
    for relative in expected_objects:
        parent = PurePosixPath(relative).parent
        while str(parent) not in {"", "."}:
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    if (
        directory_paths != sorted(set(directory_paths))
        or not directory_paths
        or directory_paths[0] != "."
        or set(directory_paths) != expected_directories
        or object_paths != sorted(set(object_paths))
        or set(object_paths) != set(expected_objects)
        or any(
            (item["size"], item["md5_base64"])
            != expected_objects[item["relative_path"]]
            for item in objects
        )
        or value["manifest_sha256"] != CHECKPOINT_MANIFEST_SHA256
        or value["object_count"] != CHECKPOINT_OBJECT_COUNT
        or len(objects) != CHECKPOINT_OBJECT_COUNT
        or value["total_bytes"] != CHECKPOINT_TOTAL_BYTES
        or sum(item["size"] for item in objects) != CHECKPOINT_TOTAL_BYTES
        or not _valid_sha256(value["objects_sha256"])
        or not _valid_sha256(value["inode_closure_sha256"])
    ):
        raise ConsumerBindingError("checkpoint consumer-binding identity mismatch")
    content_objects = [
        {
            "relative_path": item["relative_path"],
            "size": item["size"],
            "md5_base64": item["md5_base64"],
            "sha256": item["sha256"],
        }
        for item in objects
    ]
    if (
        value["objects_sha256"] != _sha256(content_objects)
        or value["inode_closure_sha256"]
        != _sha256({"directories": directories, "objects": objects})
    ):
        raise ConsumerBindingError("checkpoint consumer-binding closure mismatch")


def _validate_tokenizer_report(value: Any) -> None:
    required = {
        "uri",
        "generation",
        "path",
        *_IDENTITY_FIELDS,
        "md5_base64",
        "sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ConsumerBindingError("tokenizer consumer-binding schema mismatch")
    _validate_absolute_report_path(value["path"], "tokenizer")
    identity = {"relative_path": Path(value["path"]).name, **value}
    identity.pop("uri")
    identity.pop("generation")
    identity.pop("path")
    _validate_inventory_record(
        identity,
        extra_fields={"md5_base64", "sha256"},
        allow_root=False,
    )
    if (
        value["uri"] != TOKENIZER_URI
        or value["generation"] != TOKENIZER_GENERATION
        or value["size"] != TOKENIZER_SIZE
        or value["md5_base64"] != TOKENIZER_MD5_BASE64
        or value["sha256"] != TOKENIZER_SHA256
    ):
        raise ConsumerBindingError("tokenizer consumer-binding identity mismatch")


def _validate_source_report(value: Any) -> None:
    required = {
        "root",
        "polaris_base_commit",
        "polaris_base_tree",
        "openpi_commit",
        "tree_sha256",
        "file_count",
        "directory_count",
        "total_bytes",
        "inode_closure_sha256",
        "directories",
        "files",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ConsumerBindingError("source consumer-binding schema mismatch")
    _validate_absolute_report_path(value["root"], "source root")
    directories = value["directories"]
    files = value["files"]
    if not isinstance(directories, list) or not isinstance(files, list):
        raise ConsumerBindingError("source consumer-binding inventory is invalid")
    for item in directories:
        _validate_inventory_record(item, extra_fields=set(), allow_root=True)
    for item in files:
        _validate_inventory_record(
            item, extra_fields={"sha256"}, allow_root=False
        )
        if not _valid_sha256(item["sha256"]):
            raise ConsumerBindingError("source file digest is invalid")
    directory_paths = [item["relative_path"] for item in directories]
    file_paths = [item["relative_path"] for item in files]
    if (
        directory_paths != sorted(set(directory_paths))
        or not directory_paths
        or directory_paths[0] != "."
        or file_paths != sorted(set(file_paths))
        or not set(_SOURCE_REQUIRED_PATHS).issubset(file_paths)
        or value["polaris_base_commit"] != POLARIS_COMMIT
        or value["polaris_base_tree"] != POLARIS_TREE
        or value["openpi_commit"] != OPENPI_COMMIT
        or not _valid_sha256(value["tree_sha256"])
        or not _valid_sha256(value["inode_closure_sha256"])
        or value["file_count"] != len(files)
        or value["directory_count"] != len(directories)
        or value["total_bytes"] != sum(item["size"] for item in files)
    ):
        raise ConsumerBindingError("source consumer-binding identity mismatch")
    content = {
        "directories": [
            {"relative_path": item["relative_path"], "mode": item["mode"]}
            for item in directories
        ],
        "files": [
            {
                "relative_path": item["relative_path"],
                "mode": item["mode"],
                "size": item["size"],
                "sha256": item["sha256"],
            }
            for item in files
        ],
    }
    if (
        value["tree_sha256"] != _sha256(content)
        or value["inode_closure_sha256"]
        != _sha256({"directories": directories, "files": files})
    ):
        raise ConsumerBindingError("source consumer-binding closure mismatch")


def validate_consumer_binding_report(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "profile",
        "status",
        "stage",
        "identity",
        "binding_sha256",
        "residual_trust",
    }:
        raise ConsumerBindingError("consumer-binding report schema mismatch")
    if (
        value["schema_version"] != CONSUMER_BINDING_SCHEMA_VERSION
        or value["profile"] != CONSUMER_BINDING_PROFILE
        or value["status"] != "pass"
        or value["stage"] not in CONSUMER_BINDING_STAGES
        or value["residual_trust"]
        != "same_uid_transient_mutation_and_process_tampering_out_of_scope"
        or not isinstance(value["identity"], dict)
        or set(value["identity"]) != {"checkpoint", "tokenizer", "source"}
        or value["binding_sha256"] != _sha256(value["identity"])
    ):
        raise ConsumerBindingError("consumer-binding report identity mismatch")
    checkpoint = value["identity"]["checkpoint"]
    tokenizer = value["identity"]["tokenizer"]
    source = value["identity"]["source"]
    _validate_checkpoint_report(checkpoint)
    _validate_tokenizer_report(tokenizer)
    _validate_source_report(source)
    return json.loads(json.dumps(value, allow_nan=False))


def compare_consumer_binding_reports(*values: Mapping[str, Any]) -> dict[str, Any]:
    if not values:
        raise ConsumerBindingError("no consumer-binding reports were supplied")
    reports = [validate_consumer_binding_report(value) for value in values]
    stages = [report["stage"] for report in reports]
    if stages != list(CONSUMER_BINDING_STAGES[: len(reports)]):
        raise ConsumerBindingError("consumer-binding lifecycle stages are incomplete or out of order")
    if any(report["binding_sha256"] != reports[0]["binding_sha256"] for report in reports[1:]):
        raise ConsumerBindingError("consumer binding changed between lifecycle stages")
    return {
        "profile": CONSUMER_BINDING_PROFILE,
        "binding_sha256": reports[0]["binding_sha256"],
        "stages": stages,
        "checkpoint_objects_sha256": reports[0]["identity"]["checkpoint"]["objects_sha256"],
        "source_tree_sha256": reports[0]["identity"]["source"]["tree_sha256"],
        "tokenizer_sha256": reports[0]["identity"]["tokenizer"]["sha256"],
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_consumer_binding(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    validated = validate_consumer_binding_report(value)
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"consumer-binding artifact exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(validated) + b"\n"
    temporary = destination.with_name(
        f".{destination.name}.partial-{os.getpid()}-{secrets.token_hex(8)}"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    linked = False
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, destination, follow_symlinks=False)
        linked = True
        temporary.unlink()
        _fsync_directory(destination.parent)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        if linked:
            with contextlib.suppress(FileNotFoundError):
                destination.unlink()
        raise
    return validate_persisted_consumer_binding(destination)


def validate_persisted_consumer_binding(path: Path) -> dict[str, Any]:
    requested = Path(path)
    if requested.is_symlink():
        raise ConsumerBindingError("consumer-binding artifact must not be a symlink")
    before = os.stat(requested, follow_symlinks=False)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o444
        or before.st_nlink != 1
    ):
        raise ConsumerBindingError("consumer-binding artifact must be one mode-0444 link")
    payload = requested.read_bytes()
    after = os.stat(requested, follow_symlinks=False)
    if not _same_identity(before, after):
        raise ConsumerBindingError("consumer-binding artifact changed while reading")
    try:
        value = json.loads(payload, parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ConsumerBindingError("consumer-binding artifact is not strict JSON") from error
    validated = validate_consumer_binding_report(value)
    if payload != canonical_json_bytes(validated) + b"\n":
        raise ConsumerBindingError("consumer-binding artifact is not canonical JSON")
    return {
        "path": str(requested.resolve(strict=True)),
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "mode": "0444",
        "nlink": 1,
        "value": validated,
    }


def validate_persisted_source_approval(path: Path) -> dict[str, Any]:
    """Strictly validate the immutable external source-approval record."""

    requested = Path(path)
    if requested.is_symlink():
        raise ConsumerBindingError("source approval must not be a symlink")
    before = os.stat(requested, follow_symlinks=False)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o444
        or before.st_nlink != 1
    ):
        raise ConsumerBindingError("source approval must be one mode-0444 link")
    payload = requested.read_bytes()
    after = os.stat(requested, follow_symlinks=False)
    if not _same_identity(before, after):
        raise ConsumerBindingError("source approval changed while reading")
    try:
        value = json.loads(
            payload,
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
            object_pairs_hook=_reject_duplicate_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ConsumerBindingError("source approval is not strict JSON") from error
    required = {
        "schema_version",
        "profile",
        "snapshot_path",
        "source_tree_sha256",
        "implementation_commit",
        "polaris_base_commit",
        "polaris_base_tree",
        "openpi_commit",
        "trusted_hasher_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ConsumerBindingError("source approval schema mismatch")
    snapshot_path = value["snapshot_path"]
    if not isinstance(snapshot_path, str) or not Path(snapshot_path).is_absolute():
        raise ConsumerBindingError("source approval snapshot path is invalid")
    snapshot = Path(snapshot_path)
    try:
        canonical_snapshot = snapshot.resolve(strict=True)
    except OSError as error:
        raise ConsumerBindingError("source approval snapshot is unavailable") from error
    if canonical_snapshot != snapshot:
        raise ConsumerBindingError("source approval snapshot path is not canonical")
    _root, root_fd, _identity = _open_root(snapshot)
    os.close(root_fd)
    implementation_commit = value["implementation_commit"]
    if (
        value["schema_version"] != 1
        or value["profile"] != SOURCE_APPROVAL_PROFILE
        or not _valid_sha256(value["source_tree_sha256"])
        or not isinstance(implementation_commit, str)
        or len(implementation_commit) != 40
        or any(
            character not in "0123456789abcdef"
            for character in implementation_commit
        )
        or value["polaris_base_commit"] != POLARIS_COMMIT
        or value["polaris_base_tree"] != POLARIS_TREE
        or value["openpi_commit"] != OPENPI_COMMIT
        or value["trusted_hasher_sha256"]
        != SOURCE_APPROVAL_TRUSTED_HASHER_SHA256
    ):
        raise ConsumerBindingError("source approval identity mismatch")
    if payload != canonical_json_bytes(value) + b"\n":
        raise ConsumerBindingError("source approval is not canonical JSON")
    return {
        "path": str(requested.resolve(strict=True)),
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "mode": "0444",
        "nlink": 1,
        "value": json.loads(json.dumps(value, allow_nan=False)),
    }


def prepare_run_tokenizer(source: Path, destination: Path) -> dict[str, Any]:
    """Atomically copy the small tokenizer into a unique run-local path."""

    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"run-local tokenizer exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TokenizerBinding(Path(source)) as binding:
        source_report = binding.snapshot()
        temporary = destination.with_name(
            f".{destination.name}.partial-{os.getpid()}-{secrets.token_hex(8)}"
        )
        source_fd = os.dup(binding.fd)
        try:
            with os.fdopen(source_fd, "rb") as input_file, temporary.open("xb") as output:
                shutil.copyfileobj(input_file, output, length=16 * 1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            temporary.chmod(0o444)
            os.link(temporary, destination, follow_symlinks=False)
            temporary.unlink()
            _fsync_directory(destination.parent)
        finally:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()
    with TokenizerBinding(destination) as copied:
        destination_report = copied.snapshot()
    if any(source_report[key] != destination_report[key] for key in ("size", "md5_base64", "sha256")):
        raise ConsumerBindingError("run-local tokenizer copy differs from source")
    return destination_report


def open_consumer_binding(
    *,
    checkpoint: Path,
    manifest: Path,
    tokenizer: Path,
    source: Path,
    expected_source_tree_sha256: str,
) -> ConsumerBinding:
    return ConsumerBinding(
        checkpoint=checkpoint,
        manifest=manifest,
        tokenizer=tokenizer,
        source=source,
        expected_source_tree_sha256=expected_source_tree_sha256,
    )


def source_tree_sha256(source: Path) -> str:
    """Compute the content/mode digest used to approve a source snapshot."""

    root, root_descriptor, root_identity = _open_root(Path(source))
    try:
        files, directories = _walk_closed_tree(root)
        directory_records = []
        for relative in sorted(directories):
            if relative == ".":
                value = os.fstat(root_descriptor)
            else:
                descriptor = _open_directory_path(
                    root_descriptor, _validate_relative_path(relative)
                )
                try:
                    value = os.fstat(descriptor)
                    path_value = os.stat(
                        relative, dir_fd=root_descriptor, follow_symlinks=False
                    )
                    if not _same_identity(value, path_value):
                        raise ConsumerBindingError(
                            f"source directory changed while hashing: {relative}"
                        )
                finally:
                    os.close(descriptor)
            directory_records.append(
                {
                    "relative_path": relative,
                    "mode": f"{stat.S_IMODE(value.st_mode):04o}",
                }
            )
        file_records = []
        for relative in sorted(files):
            descriptor = _open_regular_beneath(root_descriptor, relative)
            try:
                before = os.fstat(descriptor)
                size, _md5, sha256 = _descriptor_hashes(descriptor)
                after = os.fstat(descriptor)
                path_value = os.stat(
                    relative, dir_fd=root_descriptor, follow_symlinks=False
                )
                if not _same_identity(before, after) or not _same_identity(
                    after, path_value
                ):
                    raise ConsumerBindingError(
                        f"source file changed while hashing: {relative}"
                    )
                mode = stat.S_IMODE(after.st_mode)
            finally:
                os.close(descriptor)
            file_records.append(
                {
                    "relative_path": relative,
                    "mode": f"{mode:04o}",
                    "size": size,
                    "sha256": sha256,
                }
            )
        final_root = os.fstat(root_descriptor)
        path_root = os.stat(root, follow_symlinks=False)
        final_files, final_directories = _walk_closed_tree(root)
        if (
            not _same_identity(root_identity, final_root)
            or not _same_identity(final_root, path_root)
            or final_files != files
            or final_directories != directories
        ):
            raise ConsumerBindingError("source closure changed while hashing")
    finally:
        os.close(root_descriptor)
    return _sha256({"directories": directory_records, "files": file_records})


def _load(path: Path) -> dict[str, Any]:
    return validate_persisted_consumer_binding(path)["value"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    digest = subparsers.add_parser("source-digest")
    digest.add_argument("source", type=Path)

    copy_tokenizer = subparsers.add_parser("prepare-tokenizer")
    copy_tokenizer.add_argument("source", type=Path)
    copy_tokenizer.add_argument("destination", type=Path)

    capture = subparsers.add_parser("capture")
    capture.add_argument("--stage", choices=CONSUMER_BINDING_STAGES, required=True)
    capture.add_argument("--checkpoint", type=Path, required=True)
    capture.add_argument("--manifest", type=Path, required=True)
    capture.add_argument("--tokenizer", type=Path, required=True)
    capture.add_argument("--source", type=Path, required=True)
    capture.add_argument("--expected-source-tree-sha256", required=True)
    capture.add_argument("--output", type=Path, required=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("artifacts", nargs="+", type=Path)

    args = parser.parse_args()
    if args.command == "source-digest":
        print(source_tree_sha256(args.source))
    elif args.command == "prepare-tokenizer":
        print(json.dumps(prepare_run_tokenizer(args.source, args.destination), sort_keys=True))
    elif args.command == "capture":
        with open_consumer_binding(
            checkpoint=args.checkpoint,
            manifest=args.manifest,
            tokenizer=args.tokenizer,
            source=args.source,
            expected_source_tree_sha256=args.expected_source_tree_sha256,
        ) as binding:
            artifact = publish_consumer_binding(args.output, binding.snapshot(args.stage))
        print(json.dumps(artifact, sort_keys=True))
    elif args.command == "compare":
        print(
            json.dumps(
                compare_consumer_binding_reports(*[_load(path) for path in args.artifacts]),
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()


__all__ = [
    "CHECKPOINT_MANIFEST_SHA256",
    "CHECKPOINT_OBJECT_COUNT",
    "CHECKPOINT_TOTAL_BYTES",
    "CONSUMER_BINDING_PROFILE",
    "CONSUMER_BINDING_SCHEMA_VERSION",
    "CONSUMER_BINDING_STAGES",
    "ConsumerBindingError",
    "SOURCE_APPROVAL_PROFILE",
    "SOURCE_APPROVAL_TRUSTED_HASHER_SHA256",
    "TOKENIZER_GENERATION",
    "TOKENIZER_MD5_BASE64",
    "TOKENIZER_SHA256",
    "TOKENIZER_SIZE",
    "TOKENIZER_URI",
    "compare_consumer_binding_reports",
    "open_consumer_binding",
    "prepare_run_tokenizer",
    "publish_consumer_binding",
    "source_tree_sha256",
    "validate_consumer_binding_report",
    "validate_persisted_consumer_binding",
    "validate_persisted_source_approval",
]
