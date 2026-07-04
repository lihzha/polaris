import asyncio
from contextlib import suppress
import json
import os
from pathlib import Path
import socket
import stat
from types import SimpleNamespace

import pytest

from polaris import pi05_droid_bound_port as bound_port


TOKEN = "a" * 64


def _record(path: Path, **updates):
    value = {
        "schema_version": 1,
        "profile": bound_port.BOUND_PORT_PROFILE,
        "artifact_path": str(path),
        "host": "0.0.0.0",
        "socket_family": "AF_INET",
        "requested_port": 0,
        "actual_port": 43123,
        "pid": os.getpid(),
        "launch_token": TOKEN,
    }
    value.update(updates)
    return value


def _raw_record(path: Path, value: dict, *, mode: int = 0o444) -> None:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("ascii")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, payload)
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)


def test_bound_port_record_is_exclusive_immutable_and_live_pid_bound(tmp_path):
    path = tmp_path / "policy_bound_port.json"
    record = _record(path)
    bound_port.publish_bound_port_record(path, record)

    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert path.stat().st_nlink == 1
    observed, actual_port, digest, identity = bound_port.load_bound_port_record(
        path,
        expected_pid=os.getpid(),
        expected_launch_token=TOKEN,
        expected_requested_port=0,
        require_live_pid=True,
    )
    assert observed == record
    assert actual_port == 43123
    assert len(digest) == 64
    assert len(identity.split(":")) == 7

    with pytest.raises(FileExistsError):
        bound_port.publish_bound_port_record(path, record)
    with pytest.raises(ValueError, match="pid mismatch"):
        bound_port.load_bound_port_record(
            path,
            expected_pid=os.getpid() + 1,
            expected_launch_token=TOKEN,
            expected_requested_port=0,
            require_live_pid=False,
        )
    with pytest.raises(ValueError, match="launch_token mismatch"):
        bound_port.load_bound_port_record(
            path,
            expected_pid=os.getpid(),
            expected_launch_token="b" * 64,
            expected_requested_port=0,
            require_live_pid=False,
        )


def test_bound_port_record_rejects_nonlive_expected_server(tmp_path, monkeypatch):
    path = tmp_path / "policy_bound_port.json"
    _raw_record(path, _record(path))

    def missing_process(_pid, _signal):
        raise ProcessLookupError

    monkeypatch.setattr(bound_port.os, "kill", missing_process)
    with pytest.raises(ValueError, match="not live"):
        bound_port.load_bound_port_record(
            path,
            expected_pid=os.getpid(),
            expected_launch_token=TOKEN,
            expected_requested_port=0,
            require_live_pid=True,
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"schema_version": 1.0}, "schema_version"),
        ({"requested_port": False}, "exact integer"),
        ({"actual_port": 0}, "actual_port"),
        ({"pid": float(os.getpid())}, "exact integer"),
        ({"launch_token": "A" * 64}, "launch_token"),
        ({"artifact_path": "/tmp/not-the-artifact"}, "path binding"),
        ({"extra": "field"}, "closed schema"),
    ],
)
def test_bound_port_record_rejects_schema_and_type_drift(tmp_path, updates, message):
    path = tmp_path / "policy_bound_port.json"
    _raw_record(path, _record(path, **updates))
    with pytest.raises((TypeError, ValueError), match=message):
        bound_port.load_bound_port_record(
            path,
            expected_pid=os.getpid(),
            expected_launch_token=TOKEN,
            expected_requested_port=0,
            require_live_pid=False,
        )


def test_bound_port_record_rejects_symlink_hardlink_and_mode_drift(tmp_path):
    target = tmp_path / "target.json"
    _raw_record(target, _record(target))
    symlink = tmp_path / "policy_bound_port.json"
    symlink.symlink_to(target)
    with pytest.raises(ValueError):
        bound_port.load_bound_port_record(
            symlink,
            expected_pid=os.getpid(),
            expected_launch_token=TOKEN,
            expected_requested_port=0,
            require_live_pid=False,
        )

    hardlink = tmp_path / "hardlink.json"
    os.link(target, hardlink)
    with pytest.raises(ValueError, match="immutable identity"):
        bound_port.load_bound_port_record(
            target,
            expected_pid=os.getpid(),
            expected_launch_token=TOKEN,
            expected_requested_port=0,
            require_live_pid=False,
        )

    target.unlink()
    hardlink.unlink()
    mode_path = tmp_path / "mode.json"
    _raw_record(mode_path, _record(mode_path), mode=0o644)
    with pytest.raises(ValueError, match="immutable identity"):
        bound_port.load_bound_port_record(
            mode_path,
            expected_pid=os.getpid(),
            expected_launch_token=TOKEN,
            expected_requested_port=0,
            require_live_pid=False,
        )


def test_bound_port_record_rejects_duplicate_and_noncanonical_json(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}\n')
    duplicate.chmod(0o444)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        bound_port.load_bound_port_record(
            duplicate,
            expected_pid=os.getpid(),
            expected_launch_token=TOKEN,
            expected_requested_port=0,
            require_live_pid=False,
        )

    noncanonical = tmp_path / "noncanonical.json"
    value = _record(noncanonical)
    noncanonical.write_text(json.dumps(value, sort_keys=False, indent=2) + "\n")
    noncanonical.chmod(0o444)
    with pytest.raises(ValueError, match="one newline-terminated"):
        bound_port.load_bound_port_record(
            noncanonical,
            expected_pid=os.getpid(),
            expected_launch_token=TOKEN,
            expected_requested_port=0,
            require_live_pid=False,
        )


def test_bound_port_record_rejects_same_size_entry_replacement_during_read(
    tmp_path, monkeypatch
):
    path = tmp_path / "policy_bound_port.json"
    _raw_record(path, _record(path))
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(path.read_bytes())
    replacement.chmod(0o444)
    original_read = bound_port.os.read
    replaced = False

    def adversarial_read(descriptor, count):
        nonlocal replaced
        if not replaced:
            replaced = True
            os.replace(replacement, path)
        return original_read(descriptor, count)

    monkeypatch.setattr(bound_port.os, "read", adversarial_read)
    with pytest.raises(ValueError, match="changed while it was read"):
        bound_port.load_bound_port_record(
            path,
            expected_pid=os.getpid(),
            expected_launch_token=TOKEN,
            expected_requested_port=0,
            require_live_pid=False,
        )


def test_bound_port_record_rejects_parent_replacement_during_read(
    tmp_path, monkeypatch
):
    parent = tmp_path / "attempt"
    parent.mkdir()
    path = parent / "policy_bound_port.json"
    _raw_record(path, _record(path))
    payload = path.read_bytes()
    moved = tmp_path / "moved-attempt"
    original_read = bound_port.os.read
    replaced = False

    def adversarial_read(descriptor, count):
        nonlocal replaced
        if not replaced:
            replaced = True
            parent.rename(moved)
            parent.mkdir()
            replacement = parent / path.name
            replacement.write_bytes(payload)
            replacement.chmod(0o444)
        return original_read(descriptor, count)

    monkeypatch.setattr(bound_port.os, "read", adversarial_read)
    with pytest.raises(ValueError, match="changed while it was read"):
        bound_port.load_bound_port_record(
            path,
            expected_pid=os.getpid(),
            expected_launch_token=TOKEN,
            expected_requested_port=0,
            require_live_pid=False,
        )


def test_server_bind_failure_publishes_no_readiness_or_listener_artifacts(tmp_path):
    class Upstream:
        def __init__(self, *, policy, host, port, metadata):
            self._handler = policy
            self._host = host
            self._port = port

    class FailedContext:
        async def __aenter__(self):
            raise OSError("synthetic bind failure")

        async def __aexit__(self, *_):
            return False

    module = SimpleNamespace(
        WebsocketPolicyServer=Upstream,
        _server=SimpleNamespace(serve=lambda *_args, **_kwargs: FailedContext()),
        _health_check=object(),
    )
    artifact = tmp_path / "policy_bound_port.json"
    model_artifact = tmp_path / "model.json"
    server = bound_port.BoundPortWebsocketPolicyServer(
        websocket_policy_server=module,
        policy=object(),
        host="0.0.0.0",
        port=0,
        metadata={},
        bound_port_output=artifact,
        launch_token=TOKEN,
        publish_listener_artifacts=lambda _port: model_artifact.write_text("ready"),
    )
    with pytest.raises(OSError, match="synthetic bind failure"):
        asyncio.run(server.run())
    assert not artifact.exists()
    assert not model_artifact.exists()


def test_real_websocket_listener_owns_os_assigned_port_before_publication(tmp_path):
    websockets_server = pytest.importorskip("websockets.asyncio.server")
    websockets_client = pytest.importorskip("websockets.asyncio.client")

    class Upstream:
        def __init__(self, *, policy, host, port, metadata):
            self._handler = policy
            self._host = host
            self._port = port

    async def handler(connection):
        await connection.send(b"official-metadata")
        await connection.wait_closed()

    module = SimpleNamespace(
        WebsocketPolicyServer=Upstream,
        _server=websockets_server,
        _health_check=lambda *_: None,
    )
    artifact = tmp_path / "policy_bound_port.json"
    listener_artifact = tmp_path / "listener-artifact.txt"

    def publish_listener_artifacts(actual_port):
        assert actual_port != 0
        assert not artifact.exists()
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            assert probe.connect_ex(("127.0.0.1", actual_port)) == 0
        finally:
            probe.close()
        listener_artifact.write_text(str(actual_port))

    server = bound_port.BoundPortWebsocketPolicyServer(
        websocket_policy_server=module,
        policy=handler,
        host="0.0.0.0",
        port=0,
        metadata={},
        bound_port_output=artifact,
        launch_token=TOKEN,
        publish_listener_artifacts=publish_listener_artifacts,
    )

    async def scenario():
        task = asyncio.create_task(server.run())
        try:
            for _ in range(200):
                if artifact.exists():
                    break
                await asyncio.sleep(0.01)
            assert artifact.exists()
            _, actual_port, _, _ = bound_port.load_bound_port_record(
                artifact,
                expected_pid=os.getpid(),
                expected_launch_token=TOKEN,
                expected_requested_port=0,
                require_live_pid=True,
            )
            assert listener_artifact.read_text() == str(actual_port)
            async with websockets_client.connect(
                f"ws://127.0.0.1:{actual_port}", compression=None, proxy=None
            ) as connection:
                assert await connection.recv() == b"official-metadata"
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    asyncio.run(scenario())


def test_production_server_rejects_every_fixed_port(tmp_path):
    module = SimpleNamespace(WebsocketPolicyServer=object)
    for port in (1, 8000, 65535):
        with pytest.raises(ValueError, match="OS-assigned port 0"):
            bound_port.BoundPortWebsocketPolicyServer(
                websocket_policy_server=module,
                policy=object(),
                host="0.0.0.0",
                port=port,
                metadata={},
                bound_port_output=tmp_path / f"port-{port}.json",
                launch_token=TOKEN,
                publish_listener_artifacts=lambda _port: None,
            )
