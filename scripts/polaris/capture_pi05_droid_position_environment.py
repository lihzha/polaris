#!/usr/bin/env python3
"""Capture the exact OpenPI/JAX environment for the position canary."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

if __package__:
    from .capture_pi05_droid_native_environment import (
        capture_environment,
        validate_environment,
        validate_runtime_packages,
    )
else:
    from capture_pi05_droid_native_environment import (
        capture_environment,
        validate_environment,
        validate_runtime_packages,
    )
from polaris.pi05_droid_native_eval_contract import (
    PI05_DROID_NATIVE_CANARY_PROFILE,
    publish_immutable_json,
)
from polaris.pi05_droid_position_adapter import PI05_DROID_POSITION_ADAPTER_PROFILE


PROFILE = "openpi_pi05_droid_position_inference_environment_v1"
PROTOCOL = "polaris-native-droid-freshq-delta0p2-position-h8-canary1-v1"


def capture_position_environment(openpi_dir: Path) -> dict[str, Any]:
    value = copy.deepcopy(capture_environment(openpi_dir))
    if value.pop("profile") != PI05_DROID_NATIVE_CANARY_PROFILE:
        raise ValueError("unexpected inherited inference-environment profile")
    value["profile"] = PROFILE
    value["controller_profile"] = PI05_DROID_POSITION_ADAPTER_PROFILE
    value["protocol"] = PROTOCOL
    return validate_position_environment(
        value,
        openpi_dir,
        Path(openpi_dir).resolve() / ".venv/bin/python",
    )


def validate_position_environment(
    value: Any,
    expected_openpi_dir: Path,
    expected_python: Path,
) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or value.get("profile") != PROFILE
        or value.get("controller_profile") != PI05_DROID_POSITION_ADAPTER_PROFILE
        or value.get("protocol") != PROTOCOL
    ):
        raise ValueError("position inference-environment identity mismatch")
    inherited = copy.deepcopy(value)
    inherited.pop("controller_profile")
    inherited.pop("protocol")
    inherited["profile"] = PI05_DROID_NATIVE_CANARY_PROFILE
    validate_environment(inherited, expected_openpi_dir, expected_python)
    return copy.deepcopy(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-dir", type=Path, required=True)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output", type=Path)
    output.add_argument("--runtime-package-preflight", action="store_true")
    args = parser.parse_args()
    if args.runtime_package_preflight:
        print(validate_runtime_packages(args.openpi_dir))
    else:
        publish_immutable_json(
            args.output, capture_position_environment(args.openpi_dir)
        )


if __name__ == "__main__":
    main()
