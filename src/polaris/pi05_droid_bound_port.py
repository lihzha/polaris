"""Atomic OS-assigned WebSocket port publication for the pi0.5 canary.

The pinned OpenPI server doesn't expose a callback after its listening socket
is bound.  This module mirrors only its short ``run`` method, while retaining
the upstream handler and health endpoint, so the actual server process can
publish one immutable readiness record after it owns an ephemeral port.
"""

# ruff: noqa: SLF001 -- pinned OpenPI exposes no public bound-socket hook.

from __future__ import annotations

import asyncio
from collections.abc import Callable
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
from typing import Any


BOUND_PORT_SCHEMA_VERSION = 1
BOUND_PORT_PROFILE = "pi05_droid_websocket_os_assigned_atomic_bind_v1"
BOUND_PORT_HOST = "0.0.0.0"
BOUND_PORT_REQUESTED = 0
_TOKEN_PATTERN = re.compile(r"[0-9a-f]{64}")
_RECORD_KEYS = {
    "schema_version",
    "profile",
    "artifact_path",
    "host",
    "socket_family",
    "requested_port",
    "actual_port",
    "pid",
    "launch_token",
}
_IDENTITY_FIELDS = (
    "st_dev",
    "st_ino",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
    "st_mode",
    "st_nlink",
)


def _strict_int(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    return value


def _validate_port(value: object, *, name: str, allow_zero: bool) -> int:
    port = _strict_int(value, name=name)
    minimum = 0 if allow_zero else 1
    if not minimum <= port <= 65_535:
        raise ValueError(f"{name} must be in {minimum}..65535")
    return port


def _validate_token(value: object) -> str:
    if type(value) is not str or _TOKEN_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "launch_token must be exactly 64 lowercase hexadecimal characters"
        )
    return value


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return tuple(getattr(value, field) for field in _IDENTITY_FIELDS)


def _directory_identity(value: os.stat_result) -> tuple[int, ...]:
    # Directory entry creation legitimately changes size/timestamps.  The
    # stable handle must retain the same filesystem object and permissions.
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
    )


def _stable_identity(value: os.stat_result) -> str:
    parts = (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        stat.S_IMODE(value.st_mode),
        value.st_nlink,
    )
    return ":".join(str(part) for part in parts)


def _canonical_artifact_path(path: str | Path) -> Path:
    artifact = Path(path)
    if not artifact.is_absolute() or artifact.name in {"", ".", ".."}:
        raise ValueError("bound-port artifact path must be an absolute file path")
    parent = artifact.parent
    resolved_parent = parent.resolve(strict=True)
    if resolved_parent != parent:
        raise ValueError(
            "bound-port artifact parent must be canonical and contain no alias"
        )
    parent_stat = parent.lstat()
    if not stat.S_ISDIR(parent_stat.st_mode):
        raise ValueError("bound-port artifact parent must be a directory")
    return artifact


def _open_parent(artifact: Path) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(artifact.parent, flags)
    opened = os.fstat(descriptor)
    lexical = artifact.parent.lstat()
    if _directory_identity(opened) != _directory_identity(lexical):
        os.close(descriptor)
        raise ValueError("bound-port artifact parent changed while it was opened")
    return descriptor, opened


def _strict_json_object(payload: bytes) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("bound-port artifact must be UTF-8 JSON") from error
    if not text.endswith("\n") or text.count("\n") != 1:
        raise ValueError(
            "bound-port artifact must be one newline-terminated JSON record"
        )

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant is forbidden: {constant}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError("bound-port artifact must be strict JSON") from error
    if not isinstance(value, dict):
        raise ValueError("bound-port artifact must contain a JSON object")
    return value


def _validate_record(
    record: dict[str, Any],
    *,
    expected_artifact_path: Path,
    expected_pid: int | None = None,
    expected_launch_token: str | None = None,
    expected_requested_port: int | None = None,
) -> int:
    if set(record) != _RECORD_KEYS:
        raise ValueError(
            "bound-port artifact keys must match the closed schema; "
            f"expected={sorted(_RECORD_KEYS)} actual={sorted(record)}"
        )
    if (
        type(record["schema_version"]) is not int
        or record["schema_version"] != BOUND_PORT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported bound-port artifact schema_version")
    if record["profile"] != BOUND_PORT_PROFILE:
        raise ValueError("unsupported bound-port artifact profile")
    if record["artifact_path"] != str(expected_artifact_path):
        raise ValueError("bound-port artifact path binding mismatch")
    if record["host"] != BOUND_PORT_HOST:
        raise ValueError(f"bound-port artifact host must be {BOUND_PORT_HOST}")
    if record["socket_family"] != "AF_INET":
        raise ValueError("bound-port artifact socket_family must be AF_INET")

    requested_port = _validate_port(
        record["requested_port"], name="requested_port", allow_zero=True
    )
    actual_port = _validate_port(
        record["actual_port"], name="actual_port", allow_zero=False
    )
    pid = _strict_int(record["pid"], name="pid")
    if pid <= 0:
        raise ValueError("pid must be positive")
    launch_token = _validate_token(record["launch_token"])
    if requested_port not in (0, actual_port):
        raise ValueError("an explicit requested_port must equal actual_port")

    if expected_pid is not None:
        expected_pid = _strict_int(expected_pid, name="expected_pid")
        if expected_pid <= 0 or pid != expected_pid:
            raise ValueError(
                f"bound-port artifact pid mismatch: expected={expected_pid} actual={pid}"
            )
    if expected_launch_token is not None and launch_token != _validate_token(
        expected_launch_token
    ):
        raise ValueError("bound-port artifact launch_token mismatch")
    if expected_requested_port is not None:
        expected_requested_port = _validate_port(
            expected_requested_port,
            name="expected_requested_port",
            allow_zero=True,
        )
        if requested_port != expected_requested_port:
            raise ValueError(
                "bound-port artifact requested_port mismatch: "
                f"expected={expected_requested_port} actual={requested_port}"
            )
    return actual_port


def _require_live_process(pid: int) -> None:
    try:
        os.kill(pid, 0)
    except OSError as error:
        raise ValueError(f"bound-port server pid is not live: {pid}") from error


def publish_bound_port_record(path: str | Path, record: dict[str, Any]) -> None:
    """Publish one read-only listener-owned readiness record without overwrite."""

    artifact = _canonical_artifact_path(path)
    _validate_record(record, expected_artifact_path=artifact)
    payload = (
        json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    parent_fd, parent_before = _open_parent(artifact)
    descriptor: int | None = None
    try:
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(artifact.name, flags, 0o600, dir_fd=parent_fd)
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        readback = b""
        while len(readback) < len(payload):
            chunk = os.read(descriptor, len(payload) - len(readback))
            if not chunk:
                break
            readback += chunk
        published_stat = os.fstat(descriptor)
        if readback != payload:
            raise RuntimeError("bound-port artifact failed descriptor readback")
        if (
            not stat.S_ISREG(published_stat.st_mode)
            or stat.S_IMODE(published_stat.st_mode) != 0o444
            or published_stat.st_nlink != 1
        ):
            raise RuntimeError("bound-port artifact publication identity mismatch")
        entry_stat = os.stat(artifact.name, dir_fd=parent_fd, follow_symlinks=False)
        if _identity(entry_stat) != _identity(published_stat):
            raise RuntimeError("bound-port artifact changed during publication")
        os.fsync(parent_fd)
        parent_after = os.fstat(parent_fd)
        parent_lexical = artifact.parent.lstat()
        if _directory_identity(parent_before) != _directory_identity(
            parent_after
        ) or _directory_identity(parent_after) != _directory_identity(parent_lexical):
            raise RuntimeError("bound-port artifact parent changed during publication")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def load_bound_port_record(
    path: str | Path,
    *,
    expected_pid: int,
    expected_launch_token: str,
    expected_requested_port: int,
    require_live_pid: bool,
) -> tuple[dict[str, Any], int, str, str]:
    """Read one immutable record from stable descriptors and validate identity."""

    artifact = _canonical_artifact_path(path)
    expected_pid = _strict_int(expected_pid, name="expected_pid")
    if expected_pid <= 0:
        raise ValueError("expected_pid must be positive")
    _validate_token(expected_launch_token)
    _validate_port(
        expected_requested_port,
        name="expected_requested_port",
        allow_zero=True,
    )
    if type(require_live_pid) is not bool:
        raise TypeError("require_live_pid must be an exact boolean")
    if require_live_pid:
        _require_live_process(expected_pid)

    parent_fd, parent_before = _open_parent(artifact)
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(artifact.name, flags, dir_fd=parent_fd)
        except OSError as error:
            raise ValueError(
                "bound-port artifact is not a regular direct entry"
            ) from error
        opened = os.fstat(descriptor)
        entry_before = os.stat(artifact.name, dir_fd=parent_fd, follow_symlinks=False)
        if _identity(opened) != _identity(entry_before):
            raise ValueError("bound-port artifact changed while it was opened")
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o444
            or opened.st_nlink != 1
            or not 1 <= opened.st_size <= 4_096
        ):
            raise ValueError("bound-port artifact immutable identity mismatch")

        chunks: list[bytes] = []
        remaining = 4_097
        while remaining > 0:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        if remaining == 0:
            raise ValueError("bound-port artifact exceeds the accepted size")
        after = os.fstat(descriptor)
        entry_after = os.stat(artifact.name, dir_fd=parent_fd, follow_symlinks=False)
        parent_after = os.fstat(parent_fd)
        parent_lexical = artifact.parent.lstat()
        if (
            _identity(opened) != _identity(after)
            or _identity(after) != _identity(entry_after)
            or _directory_identity(parent_before) != _directory_identity(parent_after)
            or _directory_identity(parent_after) != _directory_identity(parent_lexical)
        ):
            raise ValueError("bound-port artifact changed while it was read")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)

    closed_parent = artifact.parent.lstat()
    try:
        closed_entry = artifact.lstat()
    except OSError as error:
        raise ValueError("bound-port artifact disappeared after it was read") from error
    if _directory_identity(closed_parent) != _directory_identity(
        parent_after
    ) or _identity(closed_entry) != _identity(after):
        raise ValueError("bound-port artifact changed after it was read")
    if require_live_pid:
        _require_live_process(expected_pid)
    payload = b"".join(chunks)
    record = _strict_json_object(payload)
    actual_port = _validate_record(
        record,
        expected_artifact_path=artifact,
        expected_pid=expected_pid,
        expected_launch_token=expected_launch_token,
        expected_requested_port=expected_requested_port,
    )
    canonical = (
        json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    if payload != canonical:
        raise ValueError("bound-port artifact is not canonical JSON")
    return (
        record,
        actual_port,
        hashlib.sha256(payload).hexdigest(),
        _stable_identity(after),
    )


class BoundPortWebsocketPolicyServer:
    """Run the pinned upstream handler and publish its OS-assigned port."""

    def __init__(
        self,
        *,
        websocket_policy_server: Any,
        policy: Any,
        host: str,
        port: int,
        metadata: dict[str, Any] | None,
        bound_port_output: str | Path,
        launch_token: str,
        publish_listener_artifacts: Callable[[int], None],
    ) -> None:
        if host != BOUND_PORT_HOST:
            raise ValueError(
                "pi0.5 bound-port serving is restricted to the IPv4 all-interface host"
            )
        if _validate_port(port, name="port", allow_zero=True) != BOUND_PORT_REQUESTED:
            raise ValueError("pi0.5 canary must request the OS-assigned port 0")
        _validate_token(launch_token)
        if not callable(publish_listener_artifacts):
            raise TypeError("publish_listener_artifacts must be callable")
        output = _canonical_artifact_path(bound_port_output)
        if os.path.lexists(output):
            raise FileExistsError(f"bound-port artifact already exists: {output}")
        self._websocket_policy_server = websocket_policy_server
        self._upstream = websocket_policy_server.WebsocketPolicyServer(
            policy=policy,
            host=host,
            port=port,
            metadata=metadata,
        )
        self._bound_port_output = output
        self._launch_token = launch_token
        self._publish_listener_artifacts = publish_listener_artifacts

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        module = self._websocket_policy_server
        async with module._server.serve(
            self._upstream._handler,
            self._upstream._host,
            self._upstream._port,
            compression=None,
            max_size=None,
            process_request=module._health_check,
        ) as server:
            sockets = tuple(server.sockets or ())
            if len(sockets) != 1:
                raise RuntimeError(
                    f"expected exactly one listening socket, got {len(sockets)}"
                )
            listening_socket = sockets[0]
            if listening_socket.family != socket.AF_INET:
                raise RuntimeError(
                    f"expected AF_INET listening socket, got {listening_socket.family!r}"
                )
            address = listening_socket.getsockname()
            if (
                not isinstance(address, tuple)
                or len(address) < 2
                or address[0] != BOUND_PORT_HOST
            ):
                raise RuntimeError(f"unexpected listening address: {address!r}")
            actual_port = _validate_port(
                address[1], name="actual_port", allow_zero=False
            )

            # The socket is already listening here.  Authoritative model and
            # serving artifacts are published while it remains held, followed
            # by the bound-port record as the final readiness transition.
            self._publish_listener_artifacts(actual_port)
            publish_bound_port_record(
                self._bound_port_output,
                {
                    "schema_version": BOUND_PORT_SCHEMA_VERSION,
                    "profile": BOUND_PORT_PROFILE,
                    "artifact_path": str(self._bound_port_output),
                    "host": BOUND_PORT_HOST,
                    "socket_family": "AF_INET",
                    "requested_port": BOUND_PORT_REQUESTED,
                    "actual_port": actual_port,
                    "pid": os.getpid(),
                    "launch_token": self._launch_token,
                },
            )
            await server.serve_forever()
