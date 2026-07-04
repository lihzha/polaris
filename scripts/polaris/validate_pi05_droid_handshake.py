#!/usr/bin/env python3
"""Perform and persist one real pi0.5 WebSocket metadata handshake."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys

from polaris.pi05_droid_jointvelocity_contract import (
    validate_persisted_serving_contract,
    validate_pi05_droid_server_metadata,
    verify_openpi_git_checkout,
)
from polaris.pi05_droid_native_eval_contract import publish_immutable_json


HANDSHAKE_PROFILE = "pi05_droid_native_websocket_handshake_v1"


def _install_controlled_openpi_client_path(openpi_dir: Path) -> Path:
    if any(
        name == "openpi_client" or name.startswith("openpi_client.")
        for name in sys.modules
    ):
        raise RuntimeError("OpenPI client was imported before --openpi-dir was bound")
    checkout = verify_openpi_git_checkout(openpi_dir)
    root = Path(checkout["root"])
    requested = root / "packages/openpi-client/src"
    if requested.is_symlink() or not requested.is_dir():
        raise ValueError(f"Missing regular OpenPI client source root: {requested}")
    controlled = requested.resolve()
    controlled_string = str(controlled)
    sys.path[:] = [entry for entry in sys.path if entry != controlled_string]
    sys.path.insert(0, controlled_string)
    importlib.invalidate_caches()
    return root


def _require_live_pid(pid: int) -> None:
    if type(pid) is not int or pid <= 0:
        raise ValueError("expected server pid must be one positive exact integer")
    os.kill(pid, 0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--expected-server-pid", required=True, type=int)
    parser.add_argument("--openpi-dir", required=True, type=Path)
    parser.add_argument("--serving-contract", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=3.0)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.host != "127.0.0.1":
        raise ValueError("pi0.5 canary handshake host must be 127.0.0.1")
    if not 1 <= args.port <= 65_535:
        raise ValueError("pi0.5 canary handshake port must be in 1..65535")
    if not 0 < args.timeout_seconds <= 30:
        raise ValueError("handshake timeout must be in (0, 30] seconds")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing existing handshake output: {args.output}")

    openpi_root = _install_controlled_openpi_client_path(args.openpi_dir)
    from openpi_client import msgpack_numpy
    import websockets.sync.client

    _require_live_pid(args.expected_server_pid)
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
        print(f"pi0.5 WebSocket handshake not ready: {error}", file=sys.stderr)
        return 3

    _require_live_pid(args.expected_server_pid)
    contract = validate_pi05_droid_server_metadata(metadata)
    persisted = validate_persisted_serving_contract(args.serving_contract, metadata)
    publish_immutable_json(
        args.output,
        {
            "schema_version": 1,
            "profile": HANDSHAKE_PROFILE,
            "host": args.host,
            "actual_port": args.port,
            "server_pid": args.expected_server_pid,
            "openpi_dir": str(openpi_root),
            "serving_contract": persisted,
            "contract_sha256": contract["contract_sha256"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
