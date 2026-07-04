#!/usr/bin/env python3
"""Validate one listener-owned pi0.5 bound-port artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from polaris.pi05_droid_bound_port import load_bound_port_record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--expected-pid", required=True, type=int)
    parser.add_argument("--expected-launch-token", required=True)
    parser.add_argument("--expected-requested-port", required=True, type=int)
    parser.add_argument("--require-live-pid", action="store_true")
    parser.add_argument("--output-format", choices=("port", "tsv"), default="port")
    return parser


def main() -> None:
    args = _parser().parse_args()
    _, actual_port, content_sha256, stable_identity = load_bound_port_record(
        args.artifact,
        expected_pid=args.expected_pid,
        expected_launch_token=args.expected_launch_token,
        expected_requested_port=args.expected_requested_port,
        require_live_pid=args.require_live_pid,
    )
    if args.output_format == "tsv":
        print(f"{actual_port}\t{content_sha256}\t{stable_identity}")
    else:
        print(actual_port)


if __name__ == "__main__":
    main()
