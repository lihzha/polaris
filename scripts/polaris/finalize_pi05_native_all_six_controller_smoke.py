#!/usr/bin/env python3
"""Finalize one pinned L40S native all-six controller-only smoke."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import stat
import subprocess
from typing import Any

from polaris.native_all_six_smoke import (
    validate_immutable_native_all_six_smoke,
)
from polaris.pi05_droid_native_eval_contract import publish_immutable_json


PROFILE = "pi05_droid_native_all_six_l40s_controller_smoke_v1"
BASE_COMMIT = "3e9df7f605baa75848a0ad8edd2783d629d105c5"
OPENPI_COMMIT = "bd70b8f4011e85b3f3b0f039f12113f78718e7bf"
CONTAINER_SHA256 = "ad566a3a0bbb300cafb4a63e0f4c0056f501e4490a136881b0b1ae2d556b324a"
HUB_REVISION = "8c7e4103e266ef83d8b1ad2e9a63116edd5f155b"
ASSETS = {
    "food_bussing/initial_conditions.json": {
        "sha256": "40091faee14f692350220871d30705294f21f17ae3d2974cd3c09a34d560f5de",
        "metadata_sha256": "852dd0345afb7e4d0c7526b5c327086b5132c40624ed97ff6942962126e90534",
    },
    "food_bussing/scene.usda": {
        "sha256": "82cd641e422935b394ce7ea7b6be55214c9952a2544000222921e544c409b489",
        "metadata_sha256": "accd9b67e90e510eb4ed44a789b9169df058e71ce557164f960de2d62a840e63",
    },
    "nvidia_droid/noninstanceable.usd": {
        "sha256": "d8379925b103963dbf3e7c85bcc4ae101b81b7c1d7dabe7d2e964f41d069ec44",
        "metadata_sha256": "208e0f85fc16fa32ffeca972aea0fd1b33b0c6c2a582e89ff3877823291a7754",
    },
}
SOURCE_PATHS = (
    "scripts/eval.py",
    "scripts/smoke_pi05_native_all_six_controller.py",
    "scripts/polaris/finalize_pi05_native_all_six_controller_smoke.py",
    "scripts/polaris/l40s_pi05_native_all_six_controller_smoke.sbatch",
    "scripts/polaris/submit_pi05_native_all_six_controller_smoke.sh",
    "src/polaris/environments/droid_cfg.py",
    "src/polaris/environments/manager_based_rl_splat_environment.py",
    "src/polaris/environments/robot_cfg.py",
    "src/polaris/joint_velocity_runtime.py",
    "src/polaris/native_all_six_smoke.py",
    "src/polaris/native_gripper_runtime.py",
    "src/polaris/pi05_droid_jointvelocity_contract.py",
    "src/polaris/pi05_droid_native_lifecycle.py",
    "src/polaris/policy/droid_jointvelocity_client.py",
)
UNCHANGED_MODEL_IO_PATHS = (
    "scripts/polaris/serve_pi05_droid_native_jointvelocity.py",
    "scripts/polaris/pi05_droid_native_gcs_manifest.tsv",
)
POLICY_SEMANTIC_PATH = "src/polaris/policy/droid_jointvelocity_client.py"
POLICY_SEMANTIC_FUNCTIONS = {
    "_image_contract",
    "process_native_jointvelocity_action",
}
POLICY_SEMANTIC_METHODS = {
    "_validate_args",
    "_validate_client_runtime_origin",
    "rerender",
    "visualize",
    "infer",
    "_resize_images",
    "_extract_observation",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(
    path: Path,
    field: str,
    *,
    mode: int | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    _require(not path.is_symlink(), f"{field} must not be a symlink")
    file_stat = path.stat()
    _require(
        stat.S_ISREG(file_stat.st_mode) and file_stat.st_nlink == 1,
        f"{field} must be one regular link",
    )
    if mode is not None:
        _require(stat.S_IMODE(file_stat.st_mode) == mode, f"{field} mode drift")
    digest = _sha256(path)
    if expected_sha256 is not None:
        _require(digest == expected_sha256, f"{field} SHA-256 drift")
    return {
        "path": str(path.resolve()),
        "size": file_stat.st_size,
        "sha256": digest,
        "mode": f"{stat.S_IMODE(file_stat.st_mode):04o}",
        "nlink": file_stat.st_nlink,
    }


def _git(repository: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
    ).stdout


def _policy_semantic_symbols(source: bytes) -> dict[str, str]:
    tree = ast.parse(source)
    symbols = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in POLICY_SEMANTIC_FUNCTIONS:
            symbols[node.name] = ast.dump(node, include_attributes=False)
        if isinstance(node, ast.ClassDef) and node.name == "DroidJointVelocityClient":
            for child in node.body:
                if (
                    isinstance(child, ast.FunctionDef)
                    and child.name in POLICY_SEMANTIC_METHODS
                ):
                    symbols[f"{node.name}.{child.name}"] = ast.dump(
                        child, include_attributes=False
                    )
    expected = POLICY_SEMANTIC_FUNCTIONS | {
        f"DroidJointVelocityClient.{name}" for name in POLICY_SEMANTIC_METHODS
    }
    _require(set(symbols) == expected, "official policy semantic symbol set drift")
    return symbols


def _source_provenance(repository: Path, expected_commit: str) -> dict[str, Any]:
    repository = repository.resolve()
    git_dir = repository / ".git"
    _require(
        git_dir.is_dir() and not git_dir.is_symlink(), "source must have in-root .git"
    )
    _require(
        Path(_git(repository, "rev-parse", "--show-toplevel").decode().strip())
        == repository,
        "source top-level drift",
    )
    _require(
        Path(_git(repository, "rev-parse", "--absolute-git-dir").decode().strip())
        == git_dir,
        "source Git directory drift",
    )
    _require(
        _git(repository, "rev-parse", "--abbrev-ref", "HEAD").decode().strip()
        == "HEAD",
        "source must use detached HEAD",
    )
    head = _git(repository, "rev-parse", "HEAD").decode().strip()
    _require(head == expected_commit, "source commit drift")
    _require(
        not _git(repository, "status", "--porcelain=v1", "--untracked-files=all"),
        "source checkout is dirty",
    )
    files = {}
    for relative in SOURCE_PATHS:
        path = repository / relative
        working = path.read_bytes()
        committed = _git(repository, "show", f"HEAD:{relative}")
        _require(working == committed, f"source differs from commit: {relative}")
        files[relative] = {
            "size": len(working),
            "sha256": hashlib.sha256(working).hexdigest(),
        }
    unchanged = {}
    for relative in UNCHANGED_MODEL_IO_PATHS:
        working = _git(repository, "show", f"HEAD:{relative}")
        base = _git(repository, "show", f"{BASE_COMMIT}:{relative}")
        _require(working == base, f"official model-I/O path changed: {relative}")
        unchanged[relative] = {
            "size": len(working),
            "sha256": hashlib.sha256(working).hexdigest(),
            "base_commit": BASE_COMMIT,
        }
    current_policy_semantics = _policy_semantic_symbols(
        _git(repository, "show", f"HEAD:{POLICY_SEMANTIC_PATH}")
    )
    base_policy_semantics = _policy_semantic_symbols(
        _git(repository, "show", f"{BASE_COMMIT}:{POLICY_SEMANTIC_PATH}")
    )
    _require(
        current_policy_semantics == base_policy_semantics,
        "official policy input/output semantics changed from integrated base",
    )
    openpi = repository / "third_party/openpi"
    _require((openpi / ".git").is_file(), "OpenPI submodule is not initialized")
    _require(
        _git(openpi, "rev-parse", "HEAD").decode().strip() == OPENPI_COMMIT,
        "OpenPI commit drift",
    )
    _require(
        not _git(openpi, "status", "--porcelain=v1", "--untracked-files=all"),
        "OpenPI checkout is dirty",
    )
    return {
        "repository": str(repository),
        "commit": head,
        "detached_and_clean": True,
        "openpi_commit": OPENPI_COMMIT,
        "files": files,
        "official_model_io_unchanged_from_base": unchanged,
    }


def _read_json(path: Path, field: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), f"{field} must be a JSON object")
    return value


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    source = _source_provenance(args.polaris_repo, args.expected_polaris_commit)
    smoke = validate_immutable_native_all_six_smoke(args.smoke)
    saved_wrapper = _regular_file(args.saved_wrapper, "saved wrapper", mode=0o444)
    expected_wrapper = source["files"][
        "scripts/polaris/l40s_pi05_native_all_six_controller_smoke.sbatch"
    ]
    _require(
        saved_wrapper["size"] == expected_wrapper["size"]
        and saved_wrapper["sha256"] == expected_wrapper["sha256"],
        "saved wrapper differs from committed source",
    )
    srun_identity = _regular_file(args.srun_status, "srun status", mode=0o444)
    _require(
        _read_json(args.srun_status, "srun status")
        == {"job_id": args.job_id, "srun_exit_code": 0},
        "srun status drift",
    )
    gpu_identity = _regular_file(args.gpu_inventory, "GPU inventory", mode=0o444)
    gpu = _read_json(args.gpu_inventory, "GPU inventory")
    _require(
        gpu.get("job_id") == args.job_id
        and isinstance(gpu.get("gpus"), list)
        and len(gpu["gpus"]) == 1
        and gpu["gpus"][0].get("name") == "NVIDIA L40S"
        and str(gpu["gpus"][0].get("uuid", "")).startswith("GPU-"),
        "GPU inventory drift",
    )
    container = _regular_file(
        args.container_image,
        "container image",
        expected_sha256=CONTAINER_SHA256,
    )
    assets = {}
    for relative, expected in ASSETS.items():
        asset = _regular_file(
            args.data_dir / relative,
            f"asset {relative}",
            expected_sha256=expected["sha256"],
        )
        metadata_path = (
            args.data_dir / ".cache/huggingface/download" / f"{relative}.metadata"
        )
        metadata = _regular_file(
            metadata_path,
            f"asset metadata {relative}",
            expected_sha256=expected["metadata_sha256"],
        )
        try:
            first_line = metadata_path.read_text().splitlines()[0]
        except (OSError, UnicodeDecodeError, IndexError) as error:
            raise ValueError(f"asset metadata unreadable: {relative}") from error
        _require(first_line == HUB_REVISION, f"asset Hub revision drift: {relative}")
        assets[relative] = {
            "asset": asset,
            "metadata": metadata,
            "hub_revision": HUB_REVISION,
        }
    return {
        "schema_version": 1,
        "profile": PROFILE,
        "status": "pass",
        "job_id": args.job_id,
        "scope": "controller_only_no_model_no_checkpoint",
        "task": "DROID-FoodBussing",
        "official_policy_io_changed": False,
        "checkpoint_loaded": False,
        "model_server_started": False,
        "source": source,
        "smoke": smoke,
        "saved_wrapper": saved_wrapper,
        "srun_status": srun_identity,
        "gpu_inventory": gpu_identity,
        "container": container,
        "polaris_hub_revision": HUB_REVISION,
        "assets": assets,
        "promotion": "forbidden_without_separate_official_checkpoint_canary",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--polaris-repo", type=Path, required=True)
    parser.add_argument("--expected-polaris-commit", required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--saved-wrapper", type=Path, required=True)
    parser.add_argument("--srun-status", type=Path, required=True)
    parser.add_argument("--gpu-inventory", type=Path, required=True)
    parser.add_argument("--container-image", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _require(args.job_id > 0, "job ID must be positive")
    _require(
        len(args.expected_polaris_commit) == 40
        and all(
            character in "0123456789abcdef"
            for character in args.expected_polaris_commit
        ),
        "malformed expected PolaRiS commit",
    )
    artifact = publish_immutable_json(args.output, finalize(args))
    print(f"completion_path={artifact['path']}")
    print(f"completion_sha256={artifact['sha256']}")


if __name__ == "__main__":
    main()
