#!/usr/bin/env python3
"""Finalize one official pi0.5-DROID native joint-velocity PolaRiS canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
from typing import Any

if __package__:
    from .capture_pi05_droid_native_environment import validate_environment
    from .validate_pi05_droid_jointvelocity_trace import audit_trace
else:
    from capture_pi05_droid_native_environment import validate_environment
    from validate_pi05_droid_jointvelocity_trace import audit_trace

from polaris.joint_velocity_runtime import validate_joint_velocity_runtime_report
from polaris.native_all_six_smoke import validate_immutable_native_all_six_smoke
from polaris.pi05_droid_jointvelocity_contract import (
    PI05_DROID_CHECKPOINT_BYTES,
    PI05_DROID_CHECKPOINT_MANIFEST_SHA256,
    PI05_DROID_CHECKPOINT_OBJECT_COUNT,
    PI05_DROID_CHECKPOINT_URI,
    PI05_DROID_CONTRACT_FILENAME,
    PI05_DROID_NORM_STATS_SHA256,
    PI05_DROID_OPENPI_INFERENCE_COMPATIBILITY_COMMIT,
    validate_openpi_runtime_attestation,
    validate_persisted_serving_contract,
)
from polaris.pi05_droid_native_eval_contract import (
    PI05_DROID_ALL_SIX_CONTROLLER_COMPLETION_PATH,
    PI05_DROID_ALL_SIX_CONTROLLER_COMPLETION_SHA256,
    PI05_DROID_ALL_SIX_CONTROLLER_COMPLETION_SIZE,
    PI05_DROID_ALL_SIX_CONTROLLER_CRITICAL_PATHS,
    PI05_DROID_ALL_SIX_CONTROLLER_JOB_ID,
    PI05_DROID_ALL_SIX_CONTROLLER_PROFILE,
    PI05_DROID_ALL_SIX_CONTROLLER_SOURCE_COMMIT,
    PI05_DROID_ALL_SIX_RUNTIME_SHA256,
    PI05_DROID_ALL_SIX_UNCHANGED_POLICY_IO_PATHS,
    PI05_DROID_BASE_CONTROLLER_COMPLETION_PATH,
    PI05_DROID_BASE_CONTROLLER_COMPLETION_SHA256,
    PI05_DROID_BASE_CONTROLLER_COMPLETION_SIZE,
    PI05_DROID_BASE_CONTROLLER_RUNTIME_SHA256,
    PI05_DROID_BASE_CONTROLLER_SOURCE_COMMIT,
    PI05_DROID_CANARY_ASSETS,
    PI05_DROID_CONTROLLER_CRITICAL_PATHS,
    PI05_DROID_CONTROLLER_JOB_ID,
    PI05_DROID_GRIPPER_DRIVE_PROFILE,
    PI05_DROID_HUB_REVISION,
    PI05_DROID_NATIVE_CANARY_PROFILE,
    PI05_DROID_NATIVE_EPISODE_STEPS,
    PI05_DROID_NATIVE_MODEL_EVAL_CONTRACT,
    PI05_DROID_NATIVE_NORM_REFERENCE_PROBES,
    PI05_DROID_NATIVE_POLICY_HZ,
    PI05_DROID_NATIVE_TASK,
    PI05_DROID_NATIVE_TRANSFORM_RUNTIME_CONTRACT,
    PI05_DROID_NATIVE_VIDEO_HEIGHT,
    PI05_DROID_NATIVE_VIDEO_WIDTH,
    PI05_DROID_PYXIS_SHA256,
    canonical_json_bytes,
    file_sha256,
    fsync_directory,
    publish_immutable_json,
    validate_bound_artifact,
    validate_bound_json_artifact,
    validate_environment_runtime_contract,
    validate_episode_sidecar_value,
    validate_immutable_json,
    validate_native_model_eval_contract,
    validate_native_terminal_outcome,
)


CONTROLLER_PROFILE = "pi05_droid_native_jointvelocity_l40s_controller_smoke_v1"
SOURCE_PATHS = (
    "scripts/eval.py",
    "scripts/polaris/capture_pi05_droid_native_environment.py",
    "scripts/polaris/eval_pi05_droid_native_jointvelocity.sh",
    "scripts/polaris/finalize_pi05_droid_native_jointvelocity_eval.py",
    "scripts/polaris/l40s_pi05_droid_native_jointvelocity_canary.sbatch",
    "scripts/polaris/serve_pi05_droid_native_jointvelocity.py",
    "scripts/polaris/submit_pi05_droid_native_jointvelocity_canary.sh",
    "scripts/polaris/validate_pi05_droid_jointvelocity_trace.py",
    "scripts/polaris/verify_pi05_droid_native_checkpoint.py",
    "src/polaris/config.py",
    "src/polaris/joint_velocity_runtime.py",
    "src/polaris/pi05_droid_jointvelocity_contract.py",
    "src/polaris/pi05_droid_native_eval_contract.py",
    "src/polaris/pi05_droid_native_lifecycle.py",
    "src/polaris/policy/droid_jointvelocity_client.py",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_string(value: Any, field: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{field} must be one lowercase SHA-256",
    )
    return value


def _git(repository: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"Git query failed: {' '.join(arguments)}") from error


def _regular_file(path: Path, field: str, *, mode: int | None = None) -> os.stat_result:
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"{field} must not be a symlink")
    file_stat = path.stat()
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
        raise ValueError(f"{field} must be one regular link")
    if mode is not None and stat.S_IMODE(file_stat.st_mode) != mode:
        raise ValueError(f"{field} must have mode {mode:04o}")
    return file_stat


def _seal_file(path: Path, field: str) -> dict[str, Any]:
    """Hash, chmod, fsync, reread, and bind one evaluator-produced file."""

    path = Path(path)
    before = _regular_file(path, field)
    if before.st_size <= 0:
        raise ValueError(f"{field} is empty")
    first_digest = file_sha256(path)
    path.chmod(0o444)
    with path.open("rb") as source:
        os.fsync(source.fileno())
    fsync_directory(path.parent)
    after = _regular_file(path, field, mode=0o444)
    second_digest = file_sha256(path)
    if before.st_dev != after.st_dev or before.st_ino != after.st_ino:
        raise ValueError(f"{field} identity changed while sealing")
    if first_digest != second_digest or before.st_size != after.st_size:
        raise ValueError(f"{field} bytes changed while sealing")
    return {
        "path": str(path.resolve()),
        "size": after.st_size,
        "sha256": second_digest,
        "mode": "0444",
        "nlink": 1,
    }


def _validate_sealed_sidecar_artifact(
    sidecar_artifact: Any,
    sealed_artifact: dict[str, Any],
    *,
    expected_path: Path,
    field: str,
) -> dict[str, Any]:
    """Bind a sidecar's lexical alias to one exact sealed host artifact."""

    bound = validate_bound_artifact(
        sidecar_artifact,
        expected_path=expected_path,
        field=field,
    )
    if any(
        bound[key] != sealed_artifact[key]
        for key in ("size", "sha256", "mode", "nlink")
    ):
        raise ValueError(f"{field} differs from sealed host artifact")
    return bound


def _source_provenance(repository: Path, expected_commit: str) -> dict[str, Any]:
    repository = Path(repository)
    if repository.is_symlink():
        raise ValueError("PolaRiS repository must not be a symlink")
    repository = repository.resolve()
    git_metadata = repository / ".git"
    if git_metadata.is_symlink() or not git_metadata.is_dir():
        raise ValueError("PolaRiS must be a standalone clone with an in-root .git")
    expected_git_dir = git_metadata.resolve()
    top_level = Path(
        _git(repository, "rev-parse", "--show-toplevel").decode().strip()
    ).resolve()
    git_dir = Path(
        _git(repository, "rev-parse", "--absolute-git-dir").decode().strip()
    ).resolve()
    common_dir = Path(
        _git(repository, "rev-parse", "--path-format=absolute", "--git-common-dir")
        .decode()
        .strip()
    ).resolve()
    if (
        top_level != repository
        or git_dir != expected_git_dir
        or common_dir != expected_git_dir
    ):
        raise ValueError("PolaRiS standalone Git layout mismatch")
    if _git(repository, "rev-parse", "--abbrev-ref", "HEAD").decode().strip() != "HEAD":
        raise ValueError("PolaRiS checkout must use detached HEAD")
    head = _git(repository, "rev-parse", "HEAD").decode().strip()
    if head != expected_commit:
        raise ValueError("PolaRiS commit mismatch")
    if _git(repository, "status", "--porcelain=v1", "--untracked-files=all").strip():
        raise ValueError("PolaRiS checkout is not completely clean")
    files = {}
    for relative_path in SOURCE_PATHS:
        path = repository / relative_path
        working = path.read_bytes()
        committed = _git(repository, "show", f"HEAD:{relative_path}")
        if working != committed:
            raise ValueError(f"Source differs from commit: {relative_path}")
        files[relative_path] = {
            "size": len(working),
            "sha256": hashlib.sha256(working).hexdigest(),
        }
    return {
        "repository": str(repository),
        "git_directory": str(git_dir),
        "git_common_directory": str(common_dir),
        "detached_head": True,
        "commit": head,
        "tracked_and_untracked_clean": True,
        "files": files,
    }


def _openpi_provenance(openpi_dir: Path) -> dict[str, Any]:
    openpi_dir = Path(openpi_dir)
    if openpi_dir.is_symlink():
        raise ValueError("OpenPI checkout must not be a symlink")
    root = openpi_dir.resolve()
    if (
        Path(_git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve()
        != root
    ):
        raise ValueError("OpenPI path is not the exact Git root")
    head = _git(root, "rev-parse", "HEAD").decode().strip()
    if head != PI05_DROID_OPENPI_INFERENCE_COMPATIBILITY_COMMIT:
        raise ValueError("OpenPI commit mismatch")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all").strip():
        raise ValueError("OpenPI checkout is not clean")
    return {
        "root": str(root),
        "commit": head,
        "status_clean": True,
        "uv_lock_sha256": file_sha256(root / "uv.lock"),
    }


def validate_base_controller_completion(
    completion_path: Path,
    expected_sha256: str,
    repository: Path,
) -> dict[str, Any]:
    """Bind the exact pre-cap arm/controller gate from job 1098174.

    Its controller bytes are intentionally not compared with the integrated
    checkout: the current accepted all-six job is the descendant gate that
    reattests the complete coupled controller.  Both immutable completions
    remain mandatory.
    """

    expected_sha256 = _sha256_string(expected_sha256, "controller completion SHA-256")
    if expected_sha256 != PI05_DROID_BASE_CONTROLLER_COMPLETION_SHA256:
        raise ValueError("Unexpected job1098174 completion SHA-256")
    if str(Path(completion_path)) != PI05_DROID_BASE_CONTROLLER_COMPLETION_PATH:
        raise ValueError("Unexpected job1098174 completion path")
    artifact = validate_immutable_json(completion_path)
    if (
        artifact["sha256"] != expected_sha256
        or artifact["size"] != PI05_DROID_BASE_CONTROLLER_COMPLETION_SIZE
    ):
        raise ValueError("Controller completion SHA-256 mismatch")
    value = artifact["value"]
    if not isinstance(value, dict):
        raise ValueError("Controller completion must be an object")
    if (
        value.get("schema_version") != 1
        or value.get("profile") != CONTROLLER_PROFILE
        or value.get("status") != "pass"
        or value.get("scope") != "controller_only_no_model_or_checkpoint"
        or value.get("slurm", {}).get("job_id") != PI05_DROID_CONTROLLER_JOB_ID
        or value.get("slurm", {}).get("srun_exit_code") != 0
    ):
        raise ValueError("Controller completion identity mismatch")
    attested_files = value.get("source", {}).get("files")
    source = value.get("source")
    if (
        not isinstance(source, dict)
        or source.get("commit") != PI05_DROID_BASE_CONTROLLER_SOURCE_COMMIT
        or source.get("detached_head") is not True
        or source.get("tracked_and_untracked_clean") is not True
        or source.get("head_reference") != "HEAD"
        or source.get("standalone_git_directory") is not True
        or not isinstance(attested_files, dict)
    ):
        raise ValueError("Controller completion lacks source hashes")
    # Requiring the files to be present in this exact immutable completion
    # detects a wrong controller gate without conflating it with the later
    # descendant source attested below.
    attested = {}
    for relative_path in PI05_DROID_CONTROLLER_CRITICAL_PATHS:
        record = attested_files.get(relative_path)
        if not isinstance(record, dict) or set(record) != {"size", "sha256"}:
            raise ValueError(f"Controller completion lacks {relative_path}")
        _sha256_string(record["sha256"], f"job1098174 {relative_path}")
        if type(record["size"]) is not int or record["size"] <= 0:
            raise ValueError(f"Controller completion size is invalid: {relative_path}")
        attested[relative_path] = record
    runtime = value.get("runtime_contract")
    if (
        not isinstance(runtime, dict)
        or runtime.get("runtime_sha256") != PI05_DROID_BASE_CONTROLLER_RUNTIME_SHA256
        or runtime.get("status") != "pass"
        or runtime.get("profile") != "openpi_pi05_droid_native_jointvelocity_v1"
        or runtime.get("policy_frequency_hz") != 15
        or runtime.get("physics_frequency_hz") != 120
        or runtime.get("decimation") != 8
    ):
        raise ValueError("job1098174 runtime identity mismatch")
    if (
        value.get("runtime", {}).get("container_image", {}).get("sha256")
        != PI05_DROID_PYXIS_SHA256
    ):
        raise ValueError("Controller completion used another container image")
    return {
        **{key: artifact[key] for key in ("path", "size", "sha256", "mode", "nlink")},
        "job_id": PI05_DROID_CONTROLLER_JOB_ID,
        "profile": CONTROLLER_PROFILE,
        "runtime_sha256": runtime["runtime_sha256"],
        "critical_source_files": attested,
        "descendant_source_authority": (
            f"required_job{PI05_DROID_ALL_SIX_CONTROLLER_JOB_ID}_all_six_gate"
        ),
    }


def _validate_bound_json_record(
    record: Any, *, field: str, expected_value: dict[str, Any] | None
) -> dict[str, Any]:
    _require(isinstance(record, dict), f"{field} record must be an object")
    artifact = validate_immutable_json(Path(record.get("path", "")))
    identity_keys = ("path", "size", "sha256", "mode", "nlink")
    _require(
        set(record) == set(identity_keys)
        and all(record[key] == artifact[key] for key in identity_keys)
        and (expected_value is None or artifact["value"] == expected_value),
        f"{field} identity mismatch",
    )
    return artifact


def validate_all_six_controller_completion(
    completion_path: Path,
    expected_sha256: str,
    expected_profile: str,
    repository: Path,
) -> dict[str, Any]:
    """Reverify the accepted all-six controller job and its full lifecycle."""

    expected_sha256 = _sha256_string(expected_sha256, "all-six completion SHA-256")
    if expected_sha256 != PI05_DROID_ALL_SIX_CONTROLLER_COMPLETION_SHA256:
        raise ValueError("Unexpected all-six completion SHA-256")
    if expected_profile != PI05_DROID_ALL_SIX_CONTROLLER_PROFILE:
        raise ValueError("Unexpected all-six completion profile")
    if str(Path(completion_path)) != PI05_DROID_ALL_SIX_CONTROLLER_COMPLETION_PATH:
        raise ValueError("Unexpected all-six completion path")
    artifact = validate_immutable_json(completion_path)
    if (
        artifact["sha256"] != expected_sha256
        or artifact["size"] != PI05_DROID_ALL_SIX_CONTROLLER_COMPLETION_SIZE
    ):
        raise ValueError("All-six completion identity mismatch")
    value = artifact["value"]
    required = {
        "schema_version",
        "profile",
        "status",
        "job_id",
        "scope",
        "task",
        "official_policy_io_changed",
        "checkpoint_loaded",
        "model_server_started",
        "source",
        "smoke",
        "saved_wrapper",
        "srun_status",
        "gpu_inventory",
        "container",
        "polaris_hub_revision",
        "assets",
        "promotion",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value["schema_version"] != 1
        or value["profile"] != PI05_DROID_ALL_SIX_CONTROLLER_PROFILE
        or value["status"] != "pass"
        or value["job_id"] != PI05_DROID_ALL_SIX_CONTROLLER_JOB_ID
        or value["scope"] != "controller_only_no_model_no_checkpoint"
        or value["task"] != PI05_DROID_NATIVE_TASK
        or value["official_policy_io_changed"] is not False
        or value["checkpoint_loaded"] is not False
        or value["model_server_started"] is not False
        or value["promotion"] != "forbidden_without_separate_official_checkpoint_canary"
        or value["polaris_hub_revision"] != PI05_DROID_HUB_REVISION
    ):
        raise ValueError("All-six completion schema or identity mismatch")

    source = value["source"]
    if (
        not isinstance(source, dict)
        or set(source)
        != {
            "repository",
            "commit",
            "detached_and_clean",
            "openpi_commit",
            "files",
            "official_model_io_unchanged_from_base",
        }
        or not isinstance(source["repository"], str)
        or not source["repository"]
        or source["commit"] != PI05_DROID_ALL_SIX_CONTROLLER_SOURCE_COMMIT
        or source["detached_and_clean"] is not True
        or source["openpi_commit"] != PI05_DROID_OPENPI_INFERENCE_COMPATIBILITY_COMMIT
        or not isinstance(source["files"], dict)
        or set(source["files"]) != set(PI05_DROID_ALL_SIX_CONTROLLER_CRITICAL_PATHS)
    ):
        raise ValueError("All-six source provenance mismatch")

    repository = Path(repository).resolve()
    source_files = {}
    for relative_path in PI05_DROID_ALL_SIX_CONTROLLER_CRITICAL_PATHS:
        record = source["files"][relative_path]
        path = repository / relative_path
        if (
            not isinstance(record, dict)
            or set(record) != {"size", "sha256"}
            or type(record["size"]) is not int
            or record["size"] <= 0
            or path.stat().st_size != record["size"]
            or file_sha256(path) != record["sha256"]
        ):
            raise ValueError(
                "Integrated source differs from "
                f"job{PI05_DROID_ALL_SIX_CONTROLLER_JOB_ID}: {relative_path}"
            )
        _sha256_string(
            record["sha256"],
            f"job{PI05_DROID_ALL_SIX_CONTROLLER_JOB_ID} {relative_path}",
        )
        source_files[relative_path] = record

    model_io = source["official_model_io_unchanged_from_base"]
    accepted_model_io_paths = set(PI05_DROID_ALL_SIX_UNCHANGED_POLICY_IO_PATHS)
    if not isinstance(model_io, dict) or set(model_io) != accepted_model_io_paths:
        raise ValueError("All-six official model-I/O attestation mismatch")
    for relative_path, record in model_io.items():
        if (
            not isinstance(record, dict)
            or set(record) != {"base_commit", "size", "sha256"}
            or record["base_commit"] != "3e9df7f605baa75848a0ad8edd2783d629d105c5"
            or type(record["size"]) is not int
            or record["size"] <= 0
        ):
            raise ValueError(
                f"All-six official model-I/O record mismatch: {relative_path}"
            )
        _sha256_string(
            record["sha256"],
            f"job{PI05_DROID_ALL_SIX_CONTROLLER_JOB_ID} model-I/O {relative_path}",
        )
    policy_io_files = {}
    for relative_path in PI05_DROID_ALL_SIX_UNCHANGED_POLICY_IO_PATHS:
        record = model_io[relative_path]
        path = repository / relative_path
        if (
            path.stat().st_size != record["size"]
            or file_sha256(path) != record["sha256"]
        ):
            raise ValueError(
                "Official policy I/O differs from "
                f"job{PI05_DROID_ALL_SIX_CONTROLLER_JOB_ID}: {relative_path}"
            )
        policy_io_files[relative_path] = record

    smoke_record = value["smoke"]
    if not isinstance(smoke_record, dict) or not isinstance(
        smoke_record.get("path"), str
    ):
        raise ValueError("All-six completion lacks smoke artifact")
    # The completion stores the resolved fs11 artifact identity, while the
    # child lifecycle was intentionally published through the stable fsw
    # namespace.  Reopen through the completion's lexical fsw sibling so the
    # embedded raw/ready paths remain exact, then compare the resolved identity.
    expected_smoke_path = Path(completion_path).with_name(
        f"native-all-six-smoke-{PI05_DROID_ALL_SIX_CONTROLLER_JOB_ID}.json"
    )
    if Path(smoke_record["path"]).resolve() != expected_smoke_path.resolve():
        raise ValueError("All-six smoke path binding mismatch")
    smoke = validate_immutable_native_all_six_smoke(expected_smoke_path)
    if smoke != smoke_record:
        raise ValueError("All-six smoke identity mismatch")
    if smoke["runtime_sha256"] != PI05_DROID_ALL_SIX_RUNTIME_SHA256:
        raise ValueError("All-six runtime identity mismatch")

    srun_status_artifact = _validate_bound_json_record(
        value["srun_status"],
        field="all-six srun status",
        expected_value={
            "job_id": PI05_DROID_ALL_SIX_CONTROLLER_JOB_ID,
            "srun_exit_code": 0,
        },
    )
    gpu_inventory_artifact = _validate_bound_json_record(
        value["gpu_inventory"],
        field="all-six GPU inventory",
        expected_value=None,
    )
    gpu_value = gpu_inventory_artifact["value"]
    if (
        not isinstance(gpu_value, dict)
        or set(gpu_value) != {"schema_version", "job_id", "gpus"}
        or gpu_value["schema_version"] != 1
        or gpu_value["job_id"] != PI05_DROID_ALL_SIX_CONTROLLER_JOB_ID
        or not isinstance(gpu_value["gpus"], list)
        or len(gpu_value["gpus"]) != 1
        or not isinstance(gpu_value["gpus"][0], dict)
        or set(gpu_value["gpus"][0]) != {"uuid", "name", "driver_version"}
        or gpu_value["gpus"][0]["name"] != "NVIDIA L40S"
        or not str(gpu_value["gpus"][0]["uuid"]).startswith("GPU-")
        or not isinstance(gpu_value["gpus"][0]["driver_version"], str)
        or not gpu_value["gpus"][0]["driver_version"]
    ):
        raise ValueError("All-six GPU inventory mismatch")
    container = value["container"]
    if (
        not isinstance(container, dict)
        or container.get("sha256") != PI05_DROID_PYXIS_SHA256
        or container.get("size") != 7_183_130_624
        or container.get("mode") != "0644"
        or container.get("nlink") != 1
    ):
        raise ValueError("All-six container identity mismatch")
    assets = value["assets"]
    if not isinstance(assets, dict) or set(assets) != set(PI05_DROID_CANARY_ASSETS):
        raise ValueError("All-six asset set mismatch")
    for relative_path, expected in PI05_DROID_CANARY_ASSETS.items():
        record = assets[relative_path]
        if (
            not isinstance(record, dict)
            or set(record) != {"asset", "metadata", "hub_revision"}
            or record["hub_revision"] != PI05_DROID_HUB_REVISION
            or record["asset"].get("sha256") != expected["sha256"]
            or record["metadata"].get("sha256") != expected["metadata_sha256"]
        ):
            raise ValueError(f"All-six asset identity mismatch: {relative_path}")

    return {
        **{key: artifact[key] for key in ("path", "size", "sha256", "mode", "nlink")},
        "job_id": PI05_DROID_ALL_SIX_CONTROLLER_JOB_ID,
        "controller_profile": PI05_DROID_ALL_SIX_CONTROLLER_PROFILE,
        "drive_profile": PI05_DROID_GRIPPER_DRIVE_PROFILE,
        "runtime_sha256": PI05_DROID_ALL_SIX_RUNTIME_SHA256,
        "source_commit": PI05_DROID_ALL_SIX_CONTROLLER_SOURCE_COMMIT,
        "critical_source_files": source_files,
        "unchanged_policy_io_files": policy_io_files,
        "smoke": smoke,
        "srun_status": {
            key: srun_status_artifact[key]
            for key in ("path", "size", "sha256", "mode", "nlink")
        },
        "gpu_inventory": {
            key: gpu_inventory_artifact[key]
            for key in ("path", "size", "sha256", "mode", "nlink")
        },
        "promotion": "prerequisite_only_for_one_checkpoint_canary",
    }


def _asset_provenance(data_dir: Path) -> dict[str, Any]:
    data_dir = Path(data_dir)
    if data_dir.is_symlink() or not data_dir.is_dir():
        raise ValueError("PolaRiS-Hub root must be one regular directory")
    root = data_dir.resolve()
    assets = {}
    for relative_path, expected in PI05_DROID_CANARY_ASSETS.items():
        path = root / relative_path
        file_stat = _regular_file(path, f"asset {relative_path}")
        digest = file_sha256(path)
        metadata = root / ".cache/huggingface/download" / f"{relative_path}.metadata"
        metadata_stat = _regular_file(metadata, f"metadata {relative_path}")
        metadata_payload = metadata.read_bytes()
        if (
            digest != expected["sha256"]
            or hashlib.sha256(metadata_payload).hexdigest()
            != expected["metadata_sha256"]
            or metadata_payload.decode("utf-8").splitlines()[0]
            != PI05_DROID_HUB_REVISION
        ):
            raise ValueError(f"PolaRiS-Hub provenance mismatch: {relative_path}")
        assets[relative_path] = {
            "path": str(path),
            "size": file_stat.st_size,
            "sha256": digest,
            "metadata_path": str(metadata),
            "metadata_size": metadata_stat.st_size,
            "metadata_sha256": expected["metadata_sha256"],
            "hub_revision": PI05_DROID_HUB_REVISION,
        }
    return {
        "root": str(root),
        "hub_revision": PI05_DROID_HUB_REVISION,
        "assets": assets,
    }


def _validate_checkpoint_artifact(path: Path) -> dict[str, Any]:
    artifact = validate_immutable_json(path)
    value = artifact["value"]
    required = {
        "schema_version",
        "status",
        "checkpoint_uri",
        "manifest_path",
        "sha256",
        "object_count",
        "total_bytes",
        "checkpoint_dir",
        "norm_stats_sha256",
        "full_md5",
        "norm_reference",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Checkpoint-verification schema mismatch")
    if (
        value["schema_version"] != 1
        or value["status"] != "pass"
        or value["checkpoint_uri"] != PI05_DROID_CHECKPOINT_URI
        or value["sha256"] != PI05_DROID_CHECKPOINT_MANIFEST_SHA256
        or value["object_count"] != PI05_DROID_CHECKPOINT_OBJECT_COUNT
        or value["total_bytes"] != PI05_DROID_CHECKPOINT_BYTES
        or value["norm_stats_sha256"] != PI05_DROID_NORM_STATS_SHA256
        or value["full_md5"] is not True
        or value["norm_reference"]
        != {
            "sha256": PI05_DROID_NORM_STATS_SHA256,
            "path_within_checkpoint": "assets/droid/norm_stats.json",
            "scope": "checkpoint_global_droid",
            "asset_id": "droid",
            "category_override": "forbidden",
            "probes": PI05_DROID_NATIVE_NORM_REFERENCE_PROBES,
            "action_semantics": "joint_velocity_no_delta_or_absolute_transform",
            "state_semantics": ("panda_joint_position_plus_closed_positive_gripper"),
        }
    ):
        raise ValueError("Checkpoint-verification identity mismatch")
    return {
        **{key: artifact[key] for key in ("path", "size", "sha256", "mode", "nlink")},
        "checkpoint": value,
    }


def _validate_model_runtime_artifact(
    path: Path, checkpoint: dict[str, Any]
) -> dict[str, Any]:
    artifact = validate_immutable_json(path)
    value = artifact["value"]
    required = {
        "schema_version",
        "profile",
        "status",
        "checkpoint",
        "train_config",
        "transform_runtime",
        "policy",
        "official_model_eval_contract",
        "openpi_runtime_attestation",
    }
    expected_train_config = {
        "name": "pi05_droid",
        "model_type": "pi05",
        "pi05": True,
        "dtype": "bfloat16",
        "action_horizon": 15,
        "action_dim": 32,
        "asset_id": "droid",
        "policy_metadata": None,
    }
    expected_checkpoint = {
        key: checkpoint["checkpoint"][key]
        for key in (
            "sha256",
            "object_count",
            "total_bytes",
            "checkpoint_dir",
            "norm_stats_sha256",
            "full_md5",
        )
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value["schema_version"] != 1
        or value["profile"] != PI05_DROID_NATIVE_CANARY_PROFILE
        or value["status"] != "pass"
        or value["checkpoint"] != expected_checkpoint
        or value["train_config"] != expected_train_config
        or value["transform_runtime"] != PI05_DROID_NATIVE_TRANSFORM_RUNTIME_CONTRACT
        or value["policy"]
        != {"metadata": {}, "sample_kwargs": {}, "rng_key_data": [0, 0]}
        or value["official_model_eval_contract"]
        != PI05_DROID_NATIVE_MODEL_EVAL_CONTRACT
    ):
        raise ValueError("Native model-runtime artifact mismatch")
    validate_openpi_runtime_attestation(value["openpi_runtime_attestation"])
    return {
        **{key: artifact[key] for key in ("path", "size", "sha256", "mode", "nlink")},
        "train_config": expected_train_config,
        "transform_runtime": PI05_DROID_NATIVE_TRANSFORM_RUNTIME_CONTRACT,
        "policy": value["policy"],
    }


def _validate_runtime_artifact(path: Path) -> dict[str, Any]:
    artifact = validate_immutable_json(path)
    value = artifact["value"]
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "profile",
        "environment",
        "rollouts",
        "episode_steps",
        "environment_runtime_contract",
        "runtime_contract",
    }:
        raise ValueError("Runtime-artifact schema mismatch")
    if (
        value["schema_version"] != 1
        or value["profile"] != PI05_DROID_NATIVE_CANARY_PROFILE
        or value["environment"] != PI05_DROID_NATIVE_TASK
        or value["rollouts"] != 1
        or value["episode_steps"] != PI05_DROID_NATIVE_EPISODE_STEPS
    ):
        raise ValueError("Runtime-artifact identity mismatch")
    runtime = validate_joint_velocity_runtime_report(value["runtime_contract"])
    environment_runtime = validate_environment_runtime_contract(
        value["environment_runtime_contract"]
    )
    return {
        **{key: artifact[key] for key in ("path", "size", "sha256", "mode", "nlink")},
        "runtime_sha256": runtime["runtime_sha256"],
        "runtime_contract": runtime,
        "environment_runtime_contract": environment_runtime,
    }


def _validate_close_ready(
    path: Path, runtime: dict[str, Any], run_dir: Path
) -> dict[str, Any]:
    artifact = validate_immutable_json(path)
    value = artifact["value"]
    required = {
        "schema_version",
        "profile",
        "status",
        "environment",
        "rollouts",
        "episode_steps",
        "env_close",
        "environment_runtime_contract_sha256",
        "terminal_outcome",
        "episode_sidecar",
        "runtime_artifact",
        "metrics_path",
        "trace_path",
        "video_path",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Evaluator close-ready schema mismatch")
    task_dir = run_dir / PI05_DROID_NATIVE_TASK
    expected_paths = {
        "metrics_path": task_dir / "eval_results.csv",
        "trace_path": task_dir / "policy_traces.jsonl",
        "video_path": task_dir / "episode_0.mp4",
    }
    terminal_outcome = validate_native_terminal_outcome(
        value.get("terminal_outcome"), runtime["environment_runtime_contract"]
    )
    # The evaluator publishes descriptors from the container-visible fsw
    # namespace, while the host finalizer may reopen the same files through
    # fs11. Revalidate and consume each recorded JSON through the same stable
    # descriptor-bound read instead of reopening its lexical path.
    bound_runtime_artifact = validate_bound_json_artifact(
        value["runtime_artifact"],
        expected_path=Path(runtime["path"]),
        field="close-ready runtime",
    )
    if any(
        bound_runtime_artifact[key] != runtime[key]
        for key in ("size", "sha256", "mode", "nlink")
    ):
        raise ValueError("Close-ready runtime changed across validation")
    sidecar_path = task_dir / "native_runtime" / "episode_000000.json"
    bound_sidecar_artifact = validate_bound_json_artifact(
        value["episode_sidecar"],
        expected_path=sidecar_path,
        field="close-ready episode sidecar",
    )
    sidecar = {
        "path": str(sidecar_path),
        **{
            key: bound_sidecar_artifact[key]
            for key in ("size", "sha256", "mode", "nlink")
        },
        "value": validate_episode_sidecar_value(
            bound_sidecar_artifact["value"],
            runtime["environment_runtime_contract"],
        ),
    }
    if any(
        sidecar[key] != value["episode_sidecar"][key]
        for key in ("size", "sha256", "mode", "nlink")
    ):
        raise ValueError("Close-ready episode sidecar identity drift")
    incident = sidecar["value"]["artifacts"]["incident"]
    if terminal_outcome.get("terminal_form") == (
        "native_all_joint_velocity_limit_failure"
    ):
        if (
            incident is None
            or Path(incident["path"]).resolve()
            != (task_dir / "native_failures" / "episode_000000.json").resolve()
        ):
            raise ValueError("Native terminal incident path mismatch")
    elif incident is not None:
        raise ValueError("Completed native terminal contains an incident")
    if (
        value["schema_version"] != 2
        or value["profile"] != PI05_DROID_NATIVE_CANARY_PROFILE
        or value["status"] != "simulation_app_close_pending"
        or value["environment"] != PI05_DROID_NATIVE_TASK
        or value["rollouts"] != 1
        or value["episode_steps"] != PI05_DROID_NATIVE_EPISODE_STEPS
        or value["env_close"] != "complete"
        or value["environment_runtime_contract_sha256"]
        != runtime["environment_runtime_contract"]["sha256"]
        or sidecar["value"]["terminal_outcome"] != terminal_outcome
        or any(
            Path(value[key]).resolve() != expected.resolve()
            for key, expected in expected_paths.items()
        )
    ):
        raise ValueError("Evaluator close-ready identity mismatch")
    return {
        **{key: artifact[key] for key in ("path", "size", "sha256", "mode", "nlink")},
        "environment_runtime_contract_sha256": value[
            "environment_runtime_contract_sha256"
        ],
        "terminal_outcome": terminal_outcome,
        "episode_sidecar": sidecar,
    }


def _fraction(value: str) -> float:
    numerator, denominator = value.split("/", maxsplit=1)
    return float(numerator) / float(denominator)


def _top_level_mp4_boxes(path: Path) -> list[str]:
    boxes = []
    with Path(path).open("rb") as source:
        while True:
            header = source.read(8)
            if not header:
                break
            if len(header) != 8:
                raise ValueError("Truncated MP4 box header")
            size = int.from_bytes(header[:4], "big")
            box_type = header[4:].decode("ascii", errors="strict")
            header_size = 8
            if size == 1:
                large = source.read(8)
                if len(large) != 8:
                    raise ValueError("Truncated large MP4 box")
                size = int.from_bytes(large, "big")
                header_size = 16
            elif size == 0:
                size = path.stat().st_size - source.tell() + header_size
            if size < header_size:
                raise ValueError("Invalid MP4 box size")
            boxes.append(box_type)
            source.seek(size - header_size, os.SEEK_CUR)
    return boxes


def probe_video(
    path: Path, *, require_faststart: bool, expected_frame_count: int
) -> dict[str, Any]:
    path = Path(path)
    _regular_file(path, "video")
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,pix_fmt,field_order,width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        probe = json.loads(
            subprocess.run(command, check=True, capture_output=True, text=True).stdout
        )
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise ValueError(f"ffprobe failed for {path}") from error
    streams = probe.get("streams")
    if not isinstance(streams, list) or len(streams) != 1:
        raise ValueError("Video must contain exactly one video stream")
    stream = streams[0]
    frame_count = int(stream.get("nb_read_frames") or stream.get("nb_frames") or 0)
    duration = float(stream["duration"])
    if (
        stream.get("codec_name") != "h264"
        or stream.get("pix_fmt") != "yuv420p"
        or stream.get("field_order") != "progressive"
        or int(stream.get("width", 0)) != PI05_DROID_NATIVE_VIDEO_WIDTH
        or int(stream.get("height", 0)) != PI05_DROID_NATIVE_VIDEO_HEIGHT
        or not math.isclose(
            _fraction(stream["avg_frame_rate"]),
            PI05_DROID_NATIVE_POLICY_HZ,
            abs_tol=1e-9,
        )
        or not math.isclose(
            _fraction(stream["r_frame_rate"]), PI05_DROID_NATIVE_POLICY_HZ, abs_tol=1e-9
        )
        or type(expected_frame_count) is not int
        or not 1 <= expected_frame_count <= PI05_DROID_NATIVE_EPISODE_STEPS
        or frame_count != expected_frame_count
        or not math.isclose(
            duration,
            expected_frame_count / PI05_DROID_NATIVE_POLICY_HZ,
            abs_tol=1e-6,
        )
    ):
        raise ValueError(f"Video contract mismatch: {path}: {stream}")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-threads",
                "1",
                "-i",
                str(path),
                "-f",
                "null",
                "-",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"Full video decode failed: {path}") from error
    boxes = _top_level_mp4_boxes(path)
    faststart = (
        "moov" in boxes
        and "mdat" in boxes
        and boxes.index("moov") < boxes.index("mdat")
    )
    if require_faststart and not faststart:
        raise ValueError("Summary MP4 does not use fast-start box ordering")
    return {
        "codec": "h264",
        "pixel_format": "yuv420p",
        "field_order": "progressive",
        "width": PI05_DROID_NATIVE_VIDEO_WIDTH,
        "height": PI05_DROID_NATIVE_VIDEO_HEIGHT,
        "fps": PI05_DROID_NATIVE_POLICY_HZ,
        "frame_count": frame_count,
        "duration_seconds": duration,
        "full_decode": "pass",
        "top_level_boxes": boxes,
        "faststart": faststart,
    }


def create_summary_video(
    source: Path, output: Path, *, source_frame_count: int
) -> dict[str, Any]:
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing existing summary video: {output}")
    temporary = output.with_name(f".{output.stem}.partial-{os.getpid()}.mp4")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"Refusing existing summary temporary: {temporary}")
    try:
        summary_frame_count = max(source_frame_count, 45)
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-threads",
                "1",
                "-i",
                str(source),
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-vf",
                "tpad=stop_mode=clone:stop_duration=3",
                "-frames:v",
                str(summary_frame_count),
                "-r",
                str(PI05_DROID_NATIVE_POLICY_HZ),
                str(temporary),
            ],
            check=True,
        )
        probe = probe_video(
            temporary,
            require_faststart=True,
            expected_frame_count=summary_frame_count,
        )
        temporary.chmod(0o444)
        with temporary.open("rb") as source_file:
            os.fsync(source_file.fileno())
        os.link(temporary, output)
        temporary.unlink()
        fsync_directory(output.parent)
        _regular_file(output, "summary video", mode=0o444)
        return probe
    except Exception:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
        raise


def _gpu_inventory(path: Path) -> dict[str, Any]:
    artifact = validate_immutable_json(path)
    value = artifact["value"]
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "job_id", "gpus"}
        or value["schema_version"] != 1
        or not isinstance(value["gpus"], list)
        or len(value["gpus"]) != 1
        or set(value["gpus"][0]) != {"uuid", "name", "driver_version"}
        or value["gpus"][0]["name"] != "NVIDIA L40S"
        or not value["gpus"][0]["uuid"].startswith("GPU-")
    ):
        raise ValueError("GPU inventory mismatch")
    return {
        **{key: artifact[key] for key in ("path", "size", "sha256", "mode", "nlink")},
        "gpu": value["gpus"][0],
    }


def _validate_run_record(
    path: Path,
    *,
    args: argparse.Namespace,
    run_dir: Path,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    artifact = validate_immutable_json(path)
    value = artifact["value"]
    value_keys = {
        "RUN_DIR",
        "CHECKPOINT_PATH",
        "POLARIS_DIR",
        "OPENPI_DIR",
        "EXPECTED_POLARIS_COMMIT",
        "CHECKPOINT_URI",
        "CHECKPOINT_MANIFEST",
        "POLARIS_PYXIS_IMAGE",
        "POLARIS_DATA_DIR",
        "CONTROLLER_COMPLETION",
        "EXPECTED_CONTROLLER_COMPLETION_SHA256",
        "ALL_SIX_CONTROLLER_COMPLETION",
        "EXPECTED_ALL_SIX_COMPLETION_SHA256",
        "EXPECTED_ALL_SIX_PROFILE",
        "PORT",
        "MODEL_RUNTIME_CONTRACT",
    }
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "schema_version",
            "profile",
            "job_id",
            "fresh_attempt_no_resume",
            "task",
            "rollouts",
            "values",
        }
        or value["schema_version"] != 1
        or value["profile"] != PI05_DROID_NATIVE_CANARY_PROFILE
        or value["job_id"] != args.job_id
        or value["fresh_attempt_no_resume"] is not True
        or value["task"] != PI05_DROID_NATIVE_TASK
        or value["rollouts"] != 1
        or not isinstance(value["values"], dict)
        or set(value["values"]) != value_keys
        or any(
            not isinstance(item, str) or not item for item in value["values"].values()
        )
    ):
        raise ValueError("Run-record schema or identity mismatch")
    values = value["values"]
    expected_paths = {
        "RUN_DIR": run_dir,
        "CHECKPOINT_PATH": Path(checkpoint["checkpoint"]["checkpoint_dir"]),
        "POLARIS_DIR": Path(args.polaris_repo),
        "OPENPI_DIR": Path(args.openpi_dir),
        "CHECKPOINT_MANIFEST": Path(args.polaris_repo)
        / "scripts/polaris/pi05_droid_native_gcs_manifest.tsv",
        "POLARIS_PYXIS_IMAGE": Path(args.container_image),
        "POLARIS_DATA_DIR": Path(args.data_dir),
        "CONTROLLER_COMPLETION": Path(args.controller_completion),
        "ALL_SIX_CONTROLLER_COMPLETION": Path(args.all_six_controller_completion),
        "MODEL_RUNTIME_CONTRACT": run_dir / "pi05_droid_native_model_runtime.json",
    }
    if any(
        Path(values[key]).resolve() != expected.resolve()
        for key, expected in expected_paths.items()
    ):
        raise ValueError("Run-record path binding mismatch")
    expected_scalars = {
        "EXPECTED_POLARIS_COMMIT": args.expected_polaris_commit,
        "CHECKPOINT_URI": PI05_DROID_CHECKPOINT_URI,
        "EXPECTED_CONTROLLER_COMPLETION_SHA256": (
            args.expected_controller_completion_sha256
        ),
        "EXPECTED_ALL_SIX_COMPLETION_SHA256": (args.expected_all_six_completion_sha256),
        "EXPECTED_ALL_SIX_PROFILE": args.expected_all_six_profile,
    }
    if any(values[key] != expected for key, expected in expected_scalars.items()):
        raise ValueError("Run-record scalar binding mismatch")
    try:
        port = int(values["PORT"])
    except ValueError as error:
        raise ValueError("Run-record port is invalid") from error
    if str(port) != values["PORT"] or not 1 <= port <= 65535:
        raise ValueError("Run-record port is invalid")
    return {
        **{key: artifact[key] for key in ("path", "size", "sha256", "mode", "nlink")},
        "port": port,
    }


def _validate_submission_record(
    path: Path,
    *,
    args: argparse.Namespace,
    run_dir: Path,
    source: dict[str, Any],
) -> dict[str, Any]:
    artifact = validate_immutable_json(path)
    value = artifact["value"]
    required = {
        "schema_version",
        "profile",
        "job_id",
        "run_dir",
        "polaris_dir",
        "polaris_commit",
        "sbatch_script",
        "sbatch_script_sha256",
        "container_image",
        "polaris_data_dir",
        "fresh_attempt_no_resume",
        "task",
        "rollouts",
    }
    sbatch_relative = (
        "scripts/polaris/l40s_pi05_droid_native_jointvelocity_canary.sbatch"
    )
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value["schema_version"] != 1
        or value["profile"] != PI05_DROID_NATIVE_CANARY_PROFILE
        or value["job_id"] != args.job_id
        or Path(value["run_dir"]).resolve() != run_dir
        or Path(value["polaris_dir"]).resolve() != Path(args.polaris_repo).resolve()
        or value["polaris_commit"] != args.expected_polaris_commit
        or Path(value["sbatch_script"]).resolve()
        != (Path(args.polaris_repo) / sbatch_relative).resolve()
        or value["sbatch_script_sha256"] != source["files"][sbatch_relative]["sha256"]
        or Path(value["container_image"]).resolve()
        != Path(args.container_image).resolve()
        or Path(value["polaris_data_dir"]).resolve() != Path(args.data_dir).resolve()
        or value["fresh_attempt_no_resume"] is not True
        or value["task"] != PI05_DROID_NATIVE_TASK
        or value["rollouts"] != 1
    ):
        raise ValueError("Submission-record schema or identity mismatch")
    return {key: artifact[key] for key in ("path", "size", "sha256", "mode", "nlink")}


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    requested_run_dir = Path(args.run_dir)
    if requested_run_dir.is_symlink():
        raise ValueError("Run directory is invalid")
    run_dir = requested_run_dir.resolve()
    if not run_dir.is_dir():
        raise ValueError("Run directory is invalid")
    task_dir = run_dir / PI05_DROID_NATIVE_TASK
    source = _source_provenance(args.polaris_repo, args.expected_polaris_commit)
    openpi = _openpi_provenance(args.openpi_dir)
    base_controller = validate_base_controller_completion(
        args.controller_completion,
        args.expected_controller_completion_sha256,
        args.polaris_repo,
    )
    all_six_controller = validate_all_six_controller_completion(
        args.all_six_controller_completion,
        args.expected_all_six_completion_sha256,
        args.expected_all_six_profile,
        args.polaris_repo,
    )
    checkpoint = _validate_checkpoint_artifact(run_dir / "checkpoint_verification.json")
    model_runtime = _validate_model_runtime_artifact(
        run_dir / "pi05_droid_native_model_runtime.json", checkpoint
    )
    run_record = _validate_run_record(
        run_dir / "run_record.json",
        args=args,
        run_dir=run_dir,
        checkpoint=checkpoint,
    )
    submission_record = _validate_submission_record(
        run_dir / f"submission-{args.job_id}.json",
        args=args,
        run_dir=run_dir,
        source=source,
    )
    serving_contract_path = run_dir / PI05_DROID_CONTRACT_FILENAME
    serving_contract = validate_persisted_serving_contract(serving_contract_path)
    runtime = _validate_runtime_artifact(task_dir / "joint_velocity_runtime.json")
    if runtime["runtime_sha256"] != all_six_controller["runtime_sha256"]:
        raise ValueError(
            "Canary runtime differs from the "
            f"job{PI05_DROID_ALL_SIX_CONTROLLER_JOB_ID} all-six-attested runtime"
        )
    close_ready = _validate_close_ready(
        task_dir / "evaluator_close_ready.json", runtime, run_dir
    )
    environment_artifact = validate_immutable_json(
        run_dir / "inference_environment.json"
    )
    validate_environment(
        environment_artifact["value"],
        Path(args.openpi_dir),
        Path(args.openpi_dir) / ".venv/bin/python",
    )
    environment = {
        key: environment_artifact[key]
        for key in ("path", "size", "sha256", "mode", "nlink")
    }

    status_artifact = validate_immutable_json(
        run_dir / f"srun-{args.job_id}.status.json"
    )
    if status_artifact["value"] != {"job_id": args.job_id, "srun_exit_code": 0}:
        raise ValueError("Evaluator srun status mismatch")
    gpu = _gpu_inventory(run_dir / f"gpu-{args.job_id}.json")
    gpu_value = validate_immutable_json(run_dir / f"gpu-{args.job_id}.json")["value"]
    if gpu_value["job_id"] != args.job_id:
        raise ValueError("GPU inventory job ID mismatch")
    image = Path(args.container_image)
    image_stat = _regular_file(image, "container image")
    image_digest = file_sha256(image)
    if image_digest != PI05_DROID_PYXIS_SHA256:
        raise ValueError("Container image SHA-256 mismatch")
    assets = _asset_provenance(args.data_dir)

    saved_script = run_dir / f"job-{args.job_id}.sbatch"
    saved_script_identity = _seal_file(saved_script, "saved Slurm script")
    expected_script = source["files"][
        "scripts/polaris/l40s_pi05_droid_native_jointvelocity_canary.sbatch"
    ]
    if (
        saved_script_identity["size"] != expected_script["size"]
        or saved_script_identity["sha256"] != expected_script["sha256"]
    ):
        raise ValueError("Saved Slurm script differs from committed source")
    commands_identity = _seal_file(run_dir / "commands.sh", "exact launch commands")
    server_log_identity = _seal_file(run_dir / "policy_server.log", "policy server log")
    eval_log_identity = _seal_file(task_dir / "eval.log", "evaluator log")

    trace_path = task_dir / "policy_traces.jsonl"
    metrics_path = task_dir / "eval_results.csv"
    video_path = task_dir / "episode_0.mp4"
    trace_summary = audit_trace(trace_path, metrics_path)
    if (
        trace_summary["environment_runtime_contract"]
        != runtime["environment_runtime_contract"]
    ):
        raise ValueError("Trace environment runtime differs from runtime artifact")
    if trace_summary["terminal_outcome"] != close_ready["terminal_outcome"]:
        raise ValueError("Trace terminal outcome differs from evaluator close evidence")
    trace_summary_path = task_dir / "trace_validation.json"
    trace_summary_artifact = publish_immutable_json(trace_summary_path, trace_summary)
    raw_video_probe = probe_video(
        video_path,
        require_faststart=False,
        expected_frame_count=trace_summary["metrics"]["episode_length"],
    )
    trace_identity = _seal_file(trace_path, "policy trace")
    metrics_identity = _seal_file(metrics_path, "metrics CSV")
    video_identity = _seal_file(video_path, "rollout video")
    if trace_identity["sha256"] != trace_summary["trace_sha256"]:
        raise ValueError("Sealed trace digest differs from trace audit")
    sidecar_artifacts = close_ready["episode_sidecar"]["value"]["artifacts"]
    _validate_sealed_sidecar_artifact(
        sidecar_artifacts["trace"],
        trace_identity,
        expected_path=trace_path,
        field="episode sidecar trace",
    )
    _validate_sealed_sidecar_artifact(
        sidecar_artifacts["video"],
        video_identity,
        expected_path=video_path,
        field="episode sidecar video",
    )

    summary_path = run_dir / "pi05_droid_native_jointvelocity_canary.mp4"
    summary_probe = create_summary_video(
        video_path,
        summary_path,
        source_frame_count=trace_summary["metrics"]["episode_length"],
    )
    summary_identity = _seal_file(summary_path, "summary video")
    summary_manifest = {
        "schema_version": 1,
        "profile": PI05_DROID_NATIVE_CANARY_PROFILE,
        "task": PI05_DROID_NATIVE_TASK,
        "selection": "the_only_canary_episode",
        "source_video": video_identity,
        "summary_video": summary_identity,
        "video_probe": summary_probe,
        "raw_success": trace_summary["metrics"]["success"],
        "progress": trace_summary["metrics"]["progress"],
    }
    summary_manifest_artifact = publish_immutable_json(
        summary_path.with_suffix(".json"), summary_manifest
    )

    completion = {
        "schema_version": 1,
        "profile": PI05_DROID_NATIVE_CANARY_PROFILE,
        "status": "pass",
        "scientific_outcome": (
            "numerical_failure"
            if trace_summary["metrics"]["numerical_failure"]
            else "completed_rollout"
        ),
        "scope": "one_rollout_wiring_canary_not_standard_success_rate",
        "job_id": args.job_id,
        "task": PI05_DROID_NATIVE_TASK,
        "rollouts": 1,
        "episode_steps": PI05_DROID_NATIVE_EPISODE_STEPS,
        "result": trace_summary["metrics"],
        "source": source,
        "openpi": openpi,
        "controllers": {
            "native_joint_velocity": base_controller,
            "native_all_six_coupled": all_six_controller,
        },
        "checkpoint": checkpoint,
        "model_runtime": model_runtime,
        "official_model_eval_contract": validate_native_model_eval_contract(
            PI05_DROID_NATIVE_MODEL_EVAL_CONTRACT
        ),
        "serving_contract": serving_contract,
        "runtime": runtime,
        "evaluator_close_ready": close_ready,
        "sensor_liveness": trace_summary["sensor_liveness"],
        "inference_environment": environment,
        "slurm": {
            "submission_record": submission_record,
            "run_record": run_record,
            "srun_status": {
                key: status_artifact[key]
                for key in ("path", "size", "sha256", "mode", "nlink")
            },
            "gpu_inventory": gpu,
            "saved_job_script": saved_script_identity,
            "exact_launch_commands": commands_identity,
        },
        "container_image": {
            "path": str(image.resolve()),
            "size": image_stat.st_size,
            "sha256": image_digest,
        },
        "polaris_hub": assets,
        "artifacts": {
            "policy_server_log": server_log_identity,
            "evaluator_log": eval_log_identity,
            "metrics": metrics_identity,
            "trace": trace_identity,
            "trace_validation": {
                key: trace_summary_artifact[key]
                for key in ("path", "size", "sha256", "mode", "nlink")
            },
            "episode_sidecar": {
                key: close_ready["episode_sidecar"][key]
                for key in ("path", "size", "sha256", "mode", "nlink")
            },
            "rollout_video": {**video_identity, "probe": raw_video_probe},
            "summary_video": {**summary_identity, "probe": summary_probe},
            "summary_manifest": {
                key: summary_manifest_artifact[key]
                for key in ("path", "size", "sha256", "mode", "nlink")
            },
        },
    }
    completion_path = run_dir / f"canary-completion-{args.job_id}.json"
    completion_artifact = publish_immutable_json(completion_path, completion)
    success = {
        "schema_version": 1,
        "profile": PI05_DROID_NATIVE_CANARY_PROFILE,
        "status": "success",
        "completion": {
            key: completion_artifact[key]
            for key in ("path", "size", "sha256", "mode", "nlink")
        },
    }
    publish_immutable_json(run_dir / "eval_success.txt", success)
    return completion


def _add_controller_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--polaris-repo", type=Path, required=True)
    parser.add_argument("--controller-completion", type=Path, required=True)
    parser.add_argument("--expected-controller-completion-sha256", required=True)
    parser.add_argument("--all-six-controller-completion", type=Path, required=True)
    parser.add_argument("--expected-all-six-completion-sha256", required=True)
    parser.add_argument("--expected-all-six-profile", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    _add_controller_arguments(preflight)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--job-id", type=int, required=True)
    finalize_parser.add_argument("--run-dir", type=Path, required=True)
    _add_controller_arguments(finalize_parser)
    finalize_parser.add_argument("--expected-polaris-commit", required=True)
    finalize_parser.add_argument("--openpi-dir", type=Path, required=True)
    finalize_parser.add_argument("--container-image", type=Path, required=True)
    finalize_parser.add_argument("--data-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        base = validate_base_controller_completion(
            args.controller_completion,
            args.expected_controller_completion_sha256,
            args.polaris_repo,
        )
        all_six = validate_all_six_controller_completion(
            args.all_six_controller_completion,
            args.expected_all_six_completion_sha256,
            args.expected_all_six_profile,
            args.polaris_repo,
        )
        print(
            canonical_json_bytes({"base": base, "all_six": all_six}).decode("ascii"),
            end="",
        )
        return
    completion = finalize(args)
    print(canonical_json_bytes(completion).decode("ascii"), end="")


if __name__ == "__main__":
    main()
