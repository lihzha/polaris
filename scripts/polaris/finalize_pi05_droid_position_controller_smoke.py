#!/usr/bin/env python3
"""Host-finalize one completed DROID position-controller smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
from typing import Any

from polaris.pi05_droid_native_eval_contract import publish_immutable_json
from polaris.pi05_droid_position_adapter import canonical_json_bytes
from polaris.pi05_droid_position_smoke import validate_position_smoke


ATTESTATION_PROFILE = "openpi_pi05_droid_position_controller_smoke_attestation_v1"
IMAGE_SHA256 = "ad566a3a0bbb300cafb4a63e0f4c0056f501e4490a136881b0b1ae2d556b324a"
HUB_REVISION = "8c7e4103e266ef83d8b1ad2e9a63116edd5f155b"
ASSET_SHA256 = {
    "food_bussing/initial_conditions.json": (
        "40091faee14f692350220871d30705294f21f17ae3d2974cd3c09a34d560f5de"
    ),
    "food_bussing/scene.usda": (
        "82cd641e422935b394ce7ea7b6be55214c9952a2544000222921e544c409b489"
    ),
    "nvidia_droid/noninstanceable.usd": (
        "d8379925b103963dbf3e7c85bcc4ae101b81b7c1d7dabe7d2e964f41d069ec44"
    ),
}
SOURCE_PATHS = (
    "scripts/eval.py",
    "scripts/smoke_joint_velocity_controller.py",
    "scripts/smoke_pi05_droid_position_controller.py",
    "scripts/polaris/finalize_pi05_droid_position_controller_smoke.py",
    "scripts/polaris/l40s_pi05_droid_position_controller_smoke.sbatch",
    "scripts/polaris/serve_pi05_droid_position.py",
    "scripts/polaris/validate_pi05_droid_position_handshake.py",
    "src/polaris/environments/droid_cfg.py",
    "src/polaris/environments/manager_based_rl_splat_environment.py",
    "src/polaris/environments/pi05_droid_position_cfg.py",
    "src/polaris/environments/pi05_droid_position_robot_cfg.py",
    "src/polaris/native_gripper_runtime.py",
    "src/polaris/pi05_droid_native_eval_contract.py",
    "src/polaris/pi05_droid_native_lifecycle.py",
    "src/polaris/pi05_droid_position_adapter.py",
    "src/polaris/pi05_droid_position_contract.py",
    "src/polaris/pi05_droid_position_runtime.py",
    "src/polaris/pi05_droid_position_smoke.py",
    "src/polaris/policy/__init__.py",
    "src/polaris/policy/droid_delta_position_client.py",
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git(repo: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("cannot inspect position-smoke source checkout") from error


def _regular_identity(
    path: Path,
    field: str,
    *,
    mode: int | None = None,
    one_link: bool = False,
) -> tuple[dict[str, Any], bytes]:
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"{field} must not be a symlink")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise ValueError(f"{field} must be one readable regular file") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or (mode is not None and stat.S_IMODE(before.st_mode) != mode)
            or (one_link and before.st_nlink != 1)
        ):
            raise ValueError(f"{field} file identity mismatch")
        chunks = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.stat(path, follow_symlinks=False)
    keys = ("st_dev", "st_ino", "st_size", "st_mode", "st_nlink", "st_mtime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in keys) or any(
        getattr(before, key) != getattr(current, key) for key in keys
    ):
        raise ValueError(f"{field} changed while it was read")
    return (
        {
            "path": str(path.resolve()),
            "size": len(payload),
            "sha256": _sha256(payload),
            "mode": f"{stat.S_IMODE(before.st_mode):04o}",
            "nlink": before.st_nlink,
        },
        payload,
    )


def _canonical_json_artifact(
    path: Path, field: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity, payload = _regular_identity(path, field, mode=0o444, one_link=True)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} is not JSON") from error
    if not isinstance(value, dict) or payload != canonical_json_bytes(value) + b"\n":
        raise ValueError(f"{field} is not canonical JSON")
    return value, identity


def _source_provenance(repo_argument: Path, expected_commit: str) -> dict[str, Any]:
    if repo_argument.is_symlink():
        raise ValueError("PolaRiS repository must not be a symlink")
    repo = repo_argument.resolve()
    if not (repo / ".git").is_dir() or (repo / ".git").is_symlink():
        raise ValueError("position smoke requires a standalone Git checkout")
    if (
        Path(_git(repo, "rev-parse", "--show-toplevel")).resolve() != repo
        or Path(_git(repo, "rev-parse", "--absolute-git-dir")).resolve()
        != repo / ".git"
        or Path(
            _git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir")
        ).resolve()
        != repo / ".git"
        or _git(repo, "rev-parse", "--abbrev-ref", "HEAD") != "HEAD"
    ):
        raise ValueError(
            "position smoke source is not one detached standalone checkout"
        )
    head = _git(repo, "rev-parse", "HEAD")
    if (
        head != expected_commit
        or len(expected_commit) != 40
        or any(character not in "0123456789abcdef" for character in expected_commit)
        or _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    ):
        raise ValueError("position smoke source commit or cleanliness mismatch")
    files = {}
    for relative in SOURCE_PATHS:
        identity, payload = _regular_identity(repo / relative, f"source {relative}")
        committed = subprocess.run(
            ["git", "-C", str(repo), "show", f"HEAD:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        if payload != committed:
            raise ValueError(f"position smoke source differs from commit: {relative}")
        files[relative] = {
            "relative_path": relative,
            "size": identity["size"],
            "sha256": identity["sha256"],
            "git_blob_sha1": _git(repo, "rev-parse", f"HEAD:{relative}"),
        }
    return {"root": str(repo), "commit": head, "detached_clean": True, "files": files}


def _asset_provenance(data_dir: Path) -> dict[str, Any]:
    result = {}
    for relative, expected_sha in ASSET_SHA256.items():
        identity, _ = _regular_identity(data_dir / relative, f"asset {relative}")
        if identity["sha256"] != expected_sha:
            raise ValueError(f"asset SHA-256 mismatch: {relative}")
        metadata_path = (
            data_dir / ".cache/huggingface/download" / f"{relative}.metadata"
        )
        metadata_identity, metadata = _regular_identity(
            metadata_path, f"asset metadata {relative}"
        )
        try:
            revision = metadata.decode("utf-8").splitlines()[0]
        except (UnicodeDecodeError, IndexError) as error:
            raise ValueError(f"asset metadata unreadable: {relative}") from error
        if revision != HUB_REVISION:
            raise ValueError(f"asset Hub revision mismatch: {relative}")
        result[relative] = {
            "asset": identity,
            "metadata": metadata_identity,
            "hub_revision": revision,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--polaris-repo", type=Path, required=True)
    parser.add_argument("--expected-polaris-commit", required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--srun-status", type=Path, required=True)
    parser.add_argument("--gpu-inventory", type=Path, required=True)
    parser.add_argument("--saved-job-script", type=Path, required=True)
    parser.add_argument("--container-image", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = _source_provenance(args.polaris_repo, args.expected_polaris_commit)
    smoke_value, smoke_identity = _canonical_json_artifact(args.smoke, "position smoke")
    smoke = validate_position_smoke(smoke_value, require_parent_completion=True)
    if smoke_value != smoke:
        raise ValueError("position smoke canonical validation drift")

    raw_path = args.smoke.with_name(args.smoke.name + ".child-close.json")
    ready_path = raw_path.with_name(raw_path.name + ".ready.json")
    raw, raw_identity = _canonical_json_artifact(
        raw_path, "position smoke child capture"
    )
    validate_position_smoke(raw, require_parent_completion=False)
    ready, ready_identity = _canonical_json_artifact(ready_path, "position smoke ready")
    expected_ready = {
        "schema_version": 1,
        "status": "success",
        "stage": "kit_child_after_env_close_before_simulation_app_close",
        "raw_result": {
            "path": str(raw_path),
            "size_bytes": raw_identity["size"],
            "sha256": raw_identity["sha256"],
            "mode": "0444",
        },
    }
    if (
        ready != expected_ready
        or smoke["completion"]["raw_sha256"] != raw_identity["sha256"]
        or smoke["completion"]["ready_sha256"] != ready_identity["sha256"]
    ):
        raise ValueError("position smoke child-close evidence mismatch")

    status, status_identity = _canonical_json_artifact(args.srun_status, "srun status")
    if status != {"job_id": args.job_id, "srun_exit_code": 0}:
        raise ValueError("position smoke srun status mismatch")
    gpu, gpu_identity = _canonical_json_artifact(args.gpu_inventory, "GPU inventory")
    if (
        set(gpu) != {"schema_version", "job_id", "gpus"}
        or gpu["schema_version"] != 1
        or gpu["job_id"] != args.job_id
        or not isinstance(gpu["gpus"], list)
        or len(gpu["gpus"]) != 1
        or set(gpu["gpus"][0]) != {"uuid", "name", "driver_version"}
        or gpu["gpus"][0]["name"] != "NVIDIA L40S"
        or not str(gpu["gpus"][0]["uuid"]).startswith("GPU-")
    ):
        raise ValueError("position smoke requires exactly one allocated L40S")

    saved_identity, _ = _regular_identity(
        args.saved_job_script, "saved Slurm script", mode=0o444, one_link=True
    )
    expected_script = source["files"][
        "scripts/polaris/l40s_pi05_droid_position_controller_smoke.sbatch"
    ]
    if (
        saved_identity["size"] != expected_script["size"]
        or saved_identity["sha256"] != expected_script["sha256"]
    ):
        raise ValueError("saved Slurm script differs from committed source")
    image_identity, _ = _regular_identity(args.container_image, "container image")
    if image_identity["sha256"] != IMAGE_SHA256:
        raise ValueError("position smoke container image SHA-256 mismatch")

    runtime = smoke["runtime_contract"]
    payload = {
        "schema_version": 1,
        "profile": ATTESTATION_PROFILE,
        "status": "pass",
        "scope": "position_controller_only_no_model_or_checkpoint",
        "promotion": "forbidden_without_separate_checkpoint_canary",
        "slurm": {
            "job_id": args.job_id,
            "srun_exit_code": 0,
            "status_artifact": status_identity,
            "gpu_inventory": {"artifact": gpu_identity, "gpus": gpu["gpus"]},
            "saved_job_script": saved_identity,
        },
        "source": source,
        "container_image": image_identity,
        "assets": _asset_provenance(args.data_dir),
        "runtime_identity": {
            "container_pinned_by_sha256": True,
            "isaaclab_version": runtime["isaaclab_version"],
            "isaaclab_source_sha256": runtime["isaaclab_source_sha256"],
            "polaris_runtime_source_sha256": runtime["polaris_runtime_source_sha256"],
            "action_term_class": runtime["action_term_class"],
            "action_cfg_class": runtime["action_cfg_class"],
            "host_finalizer_python": {
                "executable": str(Path(sys.executable).resolve()),
                "implementation": platform.python_implementation(),
                "version": platform.python_version(),
            },
        },
        "smoke": smoke_identity,
        "child_close_capture": raw_identity,
        "child_ready_marker": ready_identity,
    }
    publish_immutable_json(args.output, payload)


if __name__ == "__main__":
    main()
