#!/usr/bin/env python3
"""Perform one real WebSocket handshake for the position-adapter server."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys

from polaris.pi05_droid_native_eval_contract import publish_immutable_json
from polaris.pi05_droid_position_contract import (
    validate_persisted_position_serving_contract,
    validate_pi05_droid_position_server_metadata,
    verify_openpi_git_checkout,
)


HANDSHAKE_PROFILE = "pi05_droid_position_websocket_handshake_v1"


def _controlled_client_path(openpi_dir: Path) -> Path:
    if any(
        name == "openpi_client" or name.startswith("openpi_client.")
        for name in sys.modules
    ):
        raise RuntimeError("OpenPI client imported before path binding")
    checkout = verify_openpi_git_checkout(openpi_dir)
    root = Path(checkout["root"])
    raw = root / "packages/openpi-client/src"
    if raw.is_symlink() or not raw.is_dir():
        raise ValueError("controlled OpenPI client source is missing")
    controlled = raw.resolve()
    sys.path.insert(0, str(controlled))
    importlib.invalidate_caches()
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--expected-server-pid", required=True, type=int)
    parser.add_argument("--openpi-dir", required=True, type=Path)
    parser.add_argument("--serving-contract", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=3.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.host != "127.0.0.1" or not 1 <= args.port <= 65_535:
        raise ValueError("position handshake endpoint mismatch")
    if args.expected_server_pid <= 0 or not 0 < args.timeout_seconds <= 30:
        raise ValueError("position handshake process/timeout mismatch")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("position handshake output already exists")
    root = _controlled_client_path(args.openpi_dir)
    from openpi_client import msgpack_numpy
    import websockets.sync.client

    os.kill(args.expected_server_pid, 0)
    try:
        with websockets.sync.client.connect(
            f"ws://{args.host}:{args.port}",
            compression=None,
            max_size=None,
            proxy=None,
            open_timeout=args.timeout_seconds,
            close_timeout=args.timeout_seconds,
        ) as connection:
            metadata = msgpack_numpy.unpackb(
                connection.recv(timeout=args.timeout_seconds)
            )
    except (ConnectionRefusedError, OSError, TimeoutError) as error:
        print(f"position WebSocket handshake not ready: {error}", file=sys.stderr)
        return 3
    os.kill(args.expected_server_pid, 0)
    contract = validate_pi05_droid_position_server_metadata(metadata)
    persisted = validate_persisted_position_serving_contract(
        args.serving_contract, metadata
    )
    publish_immutable_json(
        args.output,
        {
            "schema_version": 1,
            "profile": HANDSHAKE_PROFILE,
            "host": args.host,
            "actual_port": args.port,
            "server_pid": args.expected_server_pid,
            "openpi_dir": str(root),
            "serving_contract": persisted,
            "contract_sha256": contract["contract_sha256"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
