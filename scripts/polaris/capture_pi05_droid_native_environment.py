#!/usr/bin/env python3
"""Capture the exact host inference environment used by native pi0.5-DROID."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import sys
import tomllib
from typing import Any

from polaris.pi05_droid_jointvelocity_contract import (
    PI05_DROID_OPENPI_INFERENCE_COMPATIBILITY_COMMIT,
    verify_openpi_git_checkout,
)
from polaris.pi05_droid_native_eval_contract import (
    PI05_DROID_NATIVE_CANARY_PROFILE,
    canonical_json_bytes,
    file_sha256,
    publish_immutable_json,
    sha256_bytes,
)


RELEVANT_PACKAGES = (
    "etils",
    "flax",
    "fsspec",
    "jax",
    "jaxlib",
    "msgpack",
    "numpy",
    "orbax-checkpoint",
    "pillow",
    "websockets",
)


def _canonical_name(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(".", "-")


def _git_bytes(repository: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
    ).stdout


def _installed_packages() -> list[dict[str, str]]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            continue
        name = _canonical_name(raw_name)
        version = distribution.version
        previous = packages.setdefault(name, version)
        if previous != version:
            raise ValueError(
                f"Multiple installed versions for {name}: {previous}, {version}"
            )
    return [{"name": name, "version": packages[name]} for name in sorted(packages)]


def _locked_versions(lock_path: Path) -> dict[str, list[str]]:
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    packages: dict[str, set[str]] = {}
    for package in lock.get("package", []):
        name = _canonical_name(package["name"])
        packages.setdefault(name, set()).add(str(package["version"]))
    return {name: sorted(versions) for name, versions in sorted(packages.items())}


def capture_environment(openpi_dir: Path) -> dict[str, Any]:
    checkout = verify_openpi_git_checkout(openpi_dir)
    root = Path(checkout["root"])
    lock_path = root / "uv.lock"
    if not lock_path.is_file() or lock_path.is_symlink():
        raise ValueError("OpenPI uv.lock must be one regular file")
    if _git_bytes(root, "show", "HEAD:uv.lock") != lock_path.read_bytes():
        raise ValueError("OpenPI uv.lock differs from the committed bytes")

    import jax
    import jaxlib
    import numpy

    if jax.config.x64_enabled:
        raise ValueError("Native pi0.5 inference requires jax_enable_x64=False")
    devices = jax.devices()
    if jax.default_backend() != "gpu" or len(devices) != 1:
        raise ValueError("Native canary requires exactly one visible JAX GPU")
    device_kind = str(getattr(devices[0], "device_kind", ""))
    if "L40S" not in device_kind:
        raise ValueError(f"Native canary requires NVIDIA L40S, got {device_kind!r}")

    installed = _installed_packages()
    installed_by_name = {item["name"]: item["version"] for item in installed}
    locked = _locked_versions(lock_path)
    relevant: dict[str, dict[str, Any]] = {}
    for name in RELEVANT_PACKAGES:
        if name not in installed_by_name or name not in locked:
            raise ValueError(f"Required locked inference package is missing: {name}")
        if installed_by_name[name] not in locked[name]:
            raise ValueError(
                f"Installed {name}={installed_by_name[name]} is absent from uv.lock {locked[name]}"
            )
        relevant[name] = {
            "installed_version": installed_by_name[name],
            "locked_versions": locked[name],
        }

    executable = root / ".venv/bin/python"
    if (
        not executable.exists()
        or Path(sys.executable).resolve() != executable.resolve()
    ):
        raise ValueError("Capture must run with the exact checkout-local OpenPI venv")
    value: dict[str, Any] = {
        "schema_version": 1,
        "profile": PI05_DROID_NATIVE_CANARY_PROFILE,
        "status": "pass",
        "python": {
            "executable": str(executable),
            "resolved_executable": str(executable.resolve()),
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "openpi": {
            "root": str(root),
            "git_head": checkout["git_head"],
            "git_tracked_and_untracked_clean": checkout[
                "git_tracked_and_untracked_clean"
            ],
            "uv_lock_sha256": file_sha256(lock_path),
        },
        "jax": {
            "version": jax.__version__,
            "jaxlib_version": jaxlib.__version__,
            "numpy_version": numpy.__version__,
            "enable_x64": False,
            "default_backend": "gpu",
            "devices": [
                {
                    "id": int(devices[0].id),
                    "platform": str(devices[0].platform),
                    "device_kind": device_kind,
                }
            ],
        },
        "relevant_packages": relevant,
        "installed_packages": installed,
        "installed_packages_sha256": sha256_bytes(canonical_json_bytes(installed)),
    }
    return validate_environment(value, root, executable)


def validate_environment(
    value: Any, expected_openpi_dir: Path, expected_python: Path
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "profile",
        "status",
        "python",
        "openpi",
        "jax",
        "relevant_packages",
        "installed_packages",
        "installed_packages_sha256",
    }:
        raise ValueError("Inference-environment schema mismatch")
    if (
        value["schema_version"] != 1
        or value["profile"] != PI05_DROID_NATIVE_CANARY_PROFILE
        or value["status"] != "pass"
    ):
        raise ValueError("Inference-environment identity mismatch")
    expected_openpi_dir = Path(expected_openpi_dir).resolve()
    expected_python = Path(expected_python)
    installed_now = _installed_packages()
    installed_by_name = {item["name"]: item["version"] for item in installed_now}
    locked_now = _locked_versions(expected_openpi_dir / "uv.lock")
    python = value["python"]
    if not isinstance(python, dict) or set(python) != {
        "executable",
        "resolved_executable",
        "version",
        "implementation",
    }:
        raise ValueError("Inference Python schema mismatch")
    if (
        Path(python["executable"]) != expected_python
        or Path(python["resolved_executable"]) != expected_python.resolve()
        or Path(sys.executable).resolve() != expected_python.resolve()
        or python["version"] != platform.python_version()
        or python["implementation"] != "CPython"
    ):
        raise ValueError("Inference Python identity mismatch")
    openpi = value["openpi"]
    if not isinstance(openpi, dict) or set(openpi) != {
        "root",
        "git_head",
        "git_tracked_and_untracked_clean",
        "uv_lock_sha256",
    }:
        raise ValueError("Inference OpenPI schema mismatch")
    if (
        Path(openpi["root"]) != expected_openpi_dir
        or openpi["git_head"] != PI05_DROID_OPENPI_INFERENCE_COMPATIBILITY_COMMIT
        or openpi["git_tracked_and_untracked_clean"] is not True
        or openpi["uv_lock_sha256"] != file_sha256(expected_openpi_dir / "uv.lock")
    ):
        raise ValueError("Inference OpenPI identity mismatch")
    jax = value["jax"]
    if (
        not isinstance(jax, dict)
        or set(jax)
        != {
            "version",
            "jaxlib_version",
            "numpy_version",
            "enable_x64",
            "default_backend",
            "devices",
        }
        or jax["enable_x64"] is not False
        or jax["default_backend"] != "gpu"
        or not isinstance(jax["devices"], list)
        or len(jax["devices"]) != 1
        or set(jax["devices"][0]) != {"id", "platform", "device_kind"}
        or jax["devices"][0]["platform"] != "gpu"
        or "L40S" not in jax["devices"][0]["device_kind"]
        or type(jax["devices"][0]["id"]) is not int
        or jax["version"] != installed_by_name.get("jax")
        or jax["jaxlib_version"] != installed_by_name.get("jaxlib")
        or jax["numpy_version"] != installed_by_name.get("numpy")
    ):
        raise ValueError("Inference JAX runtime mismatch")
    installed = value["installed_packages"]
    if (
        not isinstance(installed, list)
        or installed != installed_now
        or installed != sorted(installed, key=lambda item: item.get("name", ""))
        or any(
            not isinstance(item, dict)
            or set(item) != {"name", "version"}
            or not isinstance(item["name"], str)
            or not isinstance(item["version"], str)
            for item in installed
        )
        or value["installed_packages_sha256"]
        != sha256_bytes(canonical_json_bytes(installed))
    ):
        raise ValueError("Installed-package inventory mismatch")
    relevant = value["relevant_packages"]
    if not isinstance(relevant, dict) or set(relevant) != set(RELEVANT_PACKAGES):
        raise ValueError("Relevant-package inventory mismatch")
    for name in RELEVANT_PACKAGES:
        record = relevant[name]
        if (
            not isinstance(record, dict)
            or set(record) != {"installed_version", "locked_versions"}
            or record["installed_version"] != installed_by_name.get(name)
            or record["locked_versions"] != locked_now.get(name)
        ):
            raise ValueError(f"Relevant-package provenance mismatch: {name}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    publish_immutable_json(args.output, capture_environment(args.openpi_dir))


if __name__ == "__main__":
    main()
