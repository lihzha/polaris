"""Lightweight immutable-artifact primitives for joint-position evaluation.

This module deliberately has no dependency on the native joint-velocity
evaluation contract so the official joint-position evaluator can be ported to
the historical control branch without importing unrelated controller code.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    """Return strict canonical ASCII JSON with one trailing newline."""

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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_immutable_json(path: Path, value: Any) -> dict[str, Any]:
    """Create one canonical mode-0444 JSON artifact without overwriting."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_directory(path.parent)
    return validate_immutable_json(path)


def validate_immutable_json(path: Path) -> dict[str, Any]:
    """Read and bind one canonical mode-0444, single-link JSON artifact."""

    path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"Immutable JSON is not readable: {path}") from error
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
            raise ValueError(f"Immutable JSON must be one regular link: {path}")
        if stat.S_IMODE(file_stat.st_mode) != 0o444:
            raise ValueError(f"Immutable JSON must have mode 0444: {path}")
        chunks = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.stat(path, follow_symlinks=False)

    def identity(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mode,
            value.st_nlink,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if identity(file_stat) != identity(after) or identity(file_stat) != identity(
        current
    ):
        raise ValueError(f"Immutable JSON changed while being read: {path}")
    try:
        value = json.loads(
            payload,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON constant is forbidden: {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Immutable artifact is not strict JSON: {path}") from error
    if payload != canonical_json_bytes(value):
        raise ValueError(f"Immutable artifact is not canonical JSON: {path}")
    return {
        "path": str(path.resolve()),
        "size": len(payload),
        "sha256": sha256_bytes(payload),
        "mode": "0444",
        "nlink": 1,
        "value": value,
    }


def validate_immutable_file(path: Path) -> dict[str, Any]:
    """Bind one nonempty mode-0444 regular file without following symlinks."""

    path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"Immutable file is not readable: {path}") from error
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o444
        ):
            raise ValueError(f"Immutable file identity mismatch: {path}")
        while block := os.read(descriptor, 16 * 1024 * 1024):
            digest.update(block)
            size += len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.stat(path, follow_symlinks=False)
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_size",
        "st_mode",
        "st_nlink",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(
        getattr(before, field) != getattr(after, field)
        or getattr(before, field) != getattr(current, field)
        for field in identity_fields
    ):
        raise ValueError(f"Immutable file changed while being read: {path}")
    if size <= 0 or size != before.st_size:
        raise ValueError(f"Immutable file is empty or truncated: {path}")
    return {
        "path": str(path.resolve()),
        "size": size,
        "sha256": digest.hexdigest(),
        "mode": "0444",
        "nlink": 1,
    }
