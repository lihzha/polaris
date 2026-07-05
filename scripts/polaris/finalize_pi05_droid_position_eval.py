#!/usr/bin/env python3
"""Preflight and host-finalize one corrected official pi0.5-DROID canary."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any

if __package__:
    from .capture_pi05_droid_position_environment import (
        validate_position_environment,
    )
    from .validate_pi05_droid_position_trace import audit_position_episode
    from .verify_pi05_droid_position_checkpoint import (
        INTEGRITY_MODE as CHECKPOINT_INTEGRITY_MODE,
        PROFILE as CHECKPOINT_ATTESTATION_PROFILE,
        SNAPSHOT_PROFILE,
    )
else:
    from capture_pi05_droid_position_environment import (
        validate_position_environment,
    )
    from validate_pi05_droid_position_trace import audit_position_episode
    from verify_pi05_droid_position_checkpoint import (
        INTEGRITY_MODE as CHECKPOINT_INTEGRITY_MODE,
        PROFILE as CHECKPOINT_ATTESTATION_PROFILE,
        SNAPSHOT_PROFILE,
    )

from polaris.pi05_droid_bound_port import load_bound_port_record
from polaris.pi05_droid_jointvelocity_contract import (
    PI05_DROID_OPENPI_INFERENCE_COMPATIBILITY_COMMIT,
    validate_openpi_runtime_attestation,
)
from polaris.pi05_droid_native_eval_contract import (
    PI05_DROID_CANARY_ASSETS,
    PI05_DROID_HUB_REVISION,
    PI05_DROID_NATIVE_EPISODE_STEPS,
    PI05_DROID_NATIVE_POLICY_HZ,
    PI05_DROID_NATIVE_TASK,
    PI05_DROID_NATIVE_VIDEO_HEIGHT,
    PI05_DROID_NATIVE_VIDEO_WIDTH,
    PI05_DROID_PYXIS_SHA256,
    canonical_json_bytes,
    file_sha256,
    fsync_directory,
    publish_immutable_json,
    validate_immutable_json,
)
from polaris.pi05_droid_position_adapter import (
    PI05_DROID_POSITION_ADAPTER_PROFILE,
    expected_position_limit_contract,
)
from polaris.pi05_droid_position_contract import (
    PI05_DROID_CHECKPOINT_BYTES,
    PI05_DROID_CHECKPOINT_MANIFEST_SHA256,
    PI05_DROID_CHECKPOINT_OBJECT_COUNT,
    PI05_DROID_CHECKPOINT_URI,
    PI05_DROID_CONTRACT_FILENAME,
    PI05_DROID_NORM_STATS_SHA256,
    PI05_DROID_POSITION_MODEL_EVAL_CONTRACT,
    PI05_DROID_POSITION_TRANSFORM_RUNTIME_CONTRACT,
    validate_persisted_position_serving_contract,
    validate_pi05_droid_position_server_metadata,
    verify_official_droid_git_checkout,
    verify_openpi_git_checkout,
    verify_profile_manifest,
)
from polaris.pi05_droid_position_smoke import validate_position_smoke


CANARY_PROFILE = "openpi_pi05_droid_fresh_jointdelta_position_polaris_canary_v1"
PROTOCOL = "polaris-native-droid-freshq-delta0p2-position-h8-canary1-v1"
MODEL_RUNTIME_PROFILE = "openpi_pi05_droid_position_model_runtime_v1"
HANDSHAKE_PROFILE = "pi05_droid_position_websocket_handshake_v1"
CONTROLLER_ATTESTATION_PROFILE = (
    "openpi_pi05_droid_position_controller_smoke_attestation_v1"
)
HOST_MEDIA_MANIFEST = "scripts/polaris/pi05_droid_native_host_media_tools.json"
HOST_MEDIA_PROFILE = "pi05_droid_native_host_media_tools_v1"
SUMMARY_PROFILE = "openpi_pi05_droid_position_canary_summary_video_v1"
REGISTRY_CANDIDATE_PROFILE = "ego_lap_polaris_external_eval_candidate_v1"
EVALUATION_ID = "polaris-food-bussing-flow-freshq-delta0p2-position-canary1-v1"
TASK_INSTRUCTION = "Put all the foods in the bowl"
FAILURE_PROFILE = "openpi_pi05_droid_position_canary_failure_v1"
EXPECTED_OPENPI_UV_LOCK_SHA256 = (
    "5e3a9a0a12d9a6048afea5591f4520c98585499cbd4a8343dcabfe2aaed94e3d"
)

SOURCE_PATHS = (
    "scripts/eval.py",
    "scripts/smoke_joint_velocity_controller.py",
    "scripts/smoke_pi05_droid_position_controller.py",
    "scripts/polaris/capture_pi05_droid_native_environment.py",
    "scripts/polaris/capture_pi05_droid_position_environment.py",
    "scripts/polaris/eval_pi05_droid_position.sh",
    "scripts/polaris/finalize_pi05_droid_position_controller_smoke.py",
    "scripts/polaris/finalize_pi05_droid_position_eval.py",
    "scripts/polaris/l40s_pi05_droid_position_canary.sbatch",
    "scripts/polaris/l40s_pi05_droid_position_controller_smoke.sbatch",
    "scripts/polaris/pi05_droid_native_gcs_manifest.tsv",
    HOST_MEDIA_MANIFEST,
    "scripts/polaris/pi05_droid_native_runtime_overlay_requirements.txt",
    "scripts/polaris/serve_pi05_droid_position.py",
    "scripts/polaris/serve_pi05_droid_native_jointvelocity.py",
    "scripts/polaris/submit_pi05_droid_position_canary.sh",
    "scripts/polaris/validate_pi05_droid_bound_port.py",
    "scripts/polaris/validate_pi05_droid_position_handshake.py",
    "scripts/polaris/validate_pi05_droid_position_trace.py",
    "scripts/polaris/verify_pi05_droid_position_checkpoint.py",
    "src/polaris/config.py",
    "src/polaris/environments/pi05_droid_position_cfg.py",
    "src/polaris/environments/pi05_droid_position_robot_cfg.py",
    "src/polaris/environments/droid_cfg.py",
    "src/polaris/environments/manager_based_rl_splat_environment.py",
    "src/polaris/environments/robot_cfg.py",
    "src/polaris/joint_velocity_runtime.py",
    "src/polaris/native_gripper_runtime.py",
    "src/polaris/pi05_droid_bound_port.py",
    "src/polaris/pi05_droid_jointvelocity_contract.py",
    "src/polaris/pi05_droid_native_eval_contract.py",
    "src/polaris/pi05_droid_native_lifecycle.py",
    "src/polaris/pi05_droid_position_adapter.py",
    "src/polaris/pi05_droid_position_contract.py",
    "src/polaris/pi05_droid_position_runtime.py",
    "src/polaris/pi05_droid_position_smoke.py",
    "src/polaris/policy/__init__.py",
    "src/polaris/policy/abstract_client.py",
    "src/polaris/policy/droid_delta_position_client.py",
    "src/polaris/policy/droid_jointpos_client.py",
)

CONTROLLER_GOVERNED_PATHS = (
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
    "src/polaris/pi05_droid_position_adapter.py",
    "src/polaris/pi05_droid_position_contract.py",
    "src/polaris/pi05_droid_position_runtime.py",
    "src/polaris/pi05_droid_position_smoke.py",
    "src/polaris/policy/__init__.py",
    "src/polaris/policy/droid_delta_position_client.py",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_BOUND_IDENTITY = re.compile(r"[0-9]+(?::[0-9]+){6}")


def _sha256(value: str, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be one lowercase SHA-256")
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


def _regular_file(
    path: Path,
    *,
    field: str,
    expected_mode: int | None = None,
) -> os.stat_result:
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"{field} must not be a symlink")
    result = path.stat()
    if not stat.S_ISREG(result.st_mode) or result.st_nlink != 1:
        raise ValueError(f"{field} must be one regular link")
    if expected_mode is not None and stat.S_IMODE(result.st_mode) != expected_mode:
        raise ValueError(f"{field} must have mode {expected_mode:04o}")
    return result


def _identity(
    path: Path,
    *,
    field: str,
    expected_mode: int | None = None,
) -> dict[str, Any]:
    result = _regular_file(path, field=field, expected_mode=expected_mode)
    return {
        "path": str(Path(path).resolve()),
        "size": result.st_size,
        "sha256": file_sha256(Path(path)),
        "mode": format(stat.S_IMODE(result.st_mode), "04o"),
        "nlink": result.st_nlink,
    }


def _seal_file(path: Path, *, field: str) -> dict[str, Any]:
    before = _regular_file(path, field=field)
    if before.st_size <= 0:
        raise ValueError(f"{field} is empty")
    digest = file_sha256(path)
    Path(path).chmod(0o444)
    with Path(path).open("rb") as source:
        os.fsync(source.fileno())
    fsync_directory(Path(path).parent)
    after = _regular_file(path, field=field, expected_mode=0o444)
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or file_sha256(path) != digest
    ):
        raise ValueError(f"{field} changed while sealing")
    return _identity(path, field=field, expected_mode=0o444)


def _artifact_identity(artifact: dict[str, Any]) -> dict[str, Any]:
    return {key: artifact[key] for key in ("path", "size", "sha256", "mode", "nlink")}


def _source_provenance(repository: Path, expected_commit: str) -> dict[str, Any]:
    requested = Path(repository)
    if requested.is_symlink():
        raise ValueError("PolaRiS repository must not be a symlink")
    root = requested.resolve()
    git_dir = root / ".git"
    if git_dir.is_symlink() or not git_dir.is_dir():
        raise ValueError("PolaRiS launch source must be a standalone clone")
    if (
        Path(_git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve()
        != root
        or Path(
            _git(root, "rev-parse", "--absolute-git-dir").decode().strip()
        ).resolve()
        != git_dir.resolve()
        or Path(
            _git(root, "rev-parse", "--path-format=absolute", "--git-common-dir")
            .decode()
            .strip()
        ).resolve()
        != git_dir.resolve()
    ):
        raise ValueError("PolaRiS standalone Git layout mismatch")
    if _git(root, "rev-parse", "--abbrev-ref", "HEAD").decode().strip() != "HEAD":
        raise ValueError("PolaRiS launch source must use detached HEAD")
    head = _git(root, "rev-parse", "HEAD").decode().strip()
    if head != expected_commit or not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ValueError("PolaRiS launch commit mismatch")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all").strip():
        raise ValueError("PolaRiS launch source must be completely clean")
    files: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_PATHS:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing committed canary source: {relative}")
        payload = path.read_bytes()
        if payload != _git(root, "show", f"HEAD:{relative}"):
            raise ValueError(f"working source differs from commit: {relative}")
        files[relative] = {
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "git_blob_sha1": _git(root, "rev-parse", f"HEAD:{relative}")
            .decode()
            .strip(),
        }
    return {
        "root": str(root),
        "commit": head,
        "detached_head": True,
        "tracked_and_untracked_clean": True,
        "files": files,
    }


def _openpi_provenance(openpi_dir: Path) -> dict[str, Any]:
    report = verify_openpi_git_checkout(openpi_dir)
    root = Path(report["root"])
    if report["git_head"] != PI05_DROID_OPENPI_INFERENCE_COMPATIBILITY_COMMIT:
        raise ValueError("OpenPI revision mismatch")
    if not report["git_tracked_and_untracked_clean"]:
        raise ValueError("OpenPI checkout is dirty")
    uv_lock = root / "uv.lock"
    digest = file_sha256(uv_lock)
    if digest != EXPECTED_OPENPI_UV_LOCK_SHA256:
        raise ValueError("OpenPI uv.lock SHA-256 mismatch")
    python = root / ".venv/bin/python"
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("OpenPI exact venv Python is unavailable")
    git_dir = Path(
        _git(root, "rev-parse", "--absolute-git-dir").decode().strip()
    ).resolve()
    git_common_dir = Path(
        _git(root, "rev-parse", "--path-format=absolute", "--git-common-dir")
        .decode()
        .strip()
    ).resolve()
    if not git_dir.is_dir() or not git_common_dir.is_dir():
        raise ValueError("OpenPI Git metadata directories are unavailable")
    git_metadata = root / ".git"
    if git_metadata.is_symlink() or not (
        git_metadata.is_file() or git_metadata.is_dir()
    ):
        raise ValueError("OpenPI checkout-local Git metadata link is invalid")
    metadata_identity = (
        _identity(git_metadata, field="OpenPI .git file")
        if git_metadata.is_file()
        else {
            "path": str(git_metadata.resolve()),
            "kind": "directory",
            "mode": format(stat.S_IMODE(git_metadata.stat().st_mode), "04o"),
        }
    )
    return {
        "root": str(root),
        "commit": report["git_head"],
        "tracked_and_untracked_clean": True,
        "git_dir": str(git_dir),
        "git_common_dir": str(git_common_dir),
        "checkout_git_metadata": metadata_identity,
        "uv_lock_sha256": digest,
        "venv_python": str(python),
    }


def validate_position_controller_attestation(
    path: Path,
    expected_sha256: str,
    *,
    canary_source: dict[str, Any],
    expected_image: dict[str, Any],
    expected_assets: dict[str, Any],
) -> dict[str, Any]:
    expected_sha256 = _sha256(
        expected_sha256, field="position controller attestation SHA-256"
    )
    artifact = validate_immutable_json(path)
    if artifact["sha256"] != expected_sha256:
        raise ValueError("position controller attestation SHA-256 mismatch")
    value = artifact["value"]
    required = {
        "schema_version",
        "profile",
        "status",
        "scope",
        "promotion",
        "slurm",
        "source",
        "container_image",
        "assets",
        "runtime_identity",
        "smoke",
        "child_close_capture",
        "child_ready_marker",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value["schema_version"] != 1
        or value["profile"] != CONTROLLER_ATTESTATION_PROFILE
        or value["status"] != "pass"
        or value["scope"] != "position_controller_only_no_model_or_checkpoint"
        or value["promotion"] != "forbidden_without_separate_checkpoint_canary"
    ):
        raise ValueError("position controller attestation identity mismatch")

    slurm = value["slurm"]
    if (
        not isinstance(slurm, dict)
        or set(slurm)
        != {
            "job_id",
            "srun_exit_code",
            "status_artifact",
            "gpu_inventory",
            "saved_job_script",
        }
        or type(slurm["job_id"]) is not int
        or slurm["job_id"] <= 0
        or slurm["srun_exit_code"] != 0
    ):
        raise ValueError("position controller Slurm identity mismatch")

    source = value["source"]
    if (
        not isinstance(source, dict)
        or set(source) != {"root", "commit", "detached_clean", "files"}
        or source["detached_clean"] is not True
        or not re.fullmatch(r"[0-9a-f]{40}", str(source["commit"]))
        or not isinstance(source["files"], dict)
        or set(source["files"]) != set(CONTROLLER_GOVERNED_PATHS)
    ):
        raise ValueError("position controller source provenance mismatch")
    repository = Path(canary_source["root"])
    ancestor = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "merge-base",
            "--is-ancestor",
            source["commit"],
            canary_source["commit"],
        ],
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise ValueError("position controller smoke commit is not a canary ancestor")
    for relative in CONTROLLER_GOVERNED_PATHS:
        smoke_file = source["files"].get(relative)
        canary_file = canary_source["files"].get(relative)
        expected_canary = (
            None
            if canary_file is None
            else {
                "relative_path": relative,
                "size": canary_file["size"],
                "sha256": canary_file["sha256"],
                "git_blob_sha1": canary_file["git_blob_sha1"],
            }
        )
        if smoke_file != expected_canary:
            raise ValueError(
                f"position controller governed source diverged: {relative}"
            )

    if value["container_image"] != expected_image:
        raise ValueError("position controller image differs from canary image")
    expected_smoke_assets = {
        relative: {
            "asset": {
                key: record[key] for key in ("path", "size", "sha256", "mode", "nlink")
            },
            "metadata": record["metadata"],
            "hub_revision": record["hub_revision"],
        }
        for relative, record in expected_assets["assets"].items()
    }
    if value["assets"] != expected_smoke_assets:
        raise ValueError("position controller assets differ from canary assets")

    def require_artifact_identity(identity: Any, field: str) -> dict[str, Any]:
        if not isinstance(identity, dict) or set(identity) != {
            "path",
            "size",
            "sha256",
            "mode",
            "nlink",
        }:
            raise ValueError(f"{field} artifact identity schema mismatch")
        actual = _identity(
            Path(identity["path"]),
            field=field,
            expected_mode=int(identity["mode"], 8),
        )
        if actual != identity:
            raise ValueError(f"{field} artifact changed")
        return actual

    status_identity = require_artifact_identity(
        slurm["status_artifact"], "position controller srun status"
    )
    status_value = validate_immutable_json(Path(status_identity["path"]))["value"]
    if status_value != {"job_id": slurm["job_id"], "srun_exit_code": 0}:
        raise ValueError("position controller srun status content mismatch")
    gpu = slurm["gpu_inventory"]
    if (
        not isinstance(gpu, dict)
        or set(gpu) != {"artifact", "gpus"}
        or not isinstance(gpu["gpus"], list)
        or len(gpu["gpus"]) != 1
        or gpu["gpus"][0].get("name") != "NVIDIA L40S"
    ):
        raise ValueError("position controller GPU inventory mismatch")
    require_artifact_identity(gpu["artifact"], "position controller GPU inventory")
    gpu_value = validate_immutable_json(Path(gpu["artifact"]["path"]))["value"]
    if gpu_value != {
        "schema_version": 1,
        "job_id": slurm["job_id"],
        "gpus": gpu["gpus"],
    }:
        raise ValueError("position controller GPU artifact content mismatch")
    saved_spool = require_artifact_identity(
        slurm["saved_job_script"], "position controller saved spool script"
    )
    expected_smoke_sbatch = source["files"][
        "scripts/polaris/l40s_pi05_droid_position_controller_smoke.sbatch"
    ]
    if (
        saved_spool["size"] != expected_smoke_sbatch["size"]
        or saved_spool["sha256"] != expected_smoke_sbatch["sha256"]
    ):
        raise ValueError("position controller saved spool/source mismatch")

    smoke_identity = require_artifact_identity(
        value["smoke"], "position controller smoke"
    )
    smoke_artifact = validate_immutable_json(Path(smoke_identity["path"]))
    smoke = validate_position_smoke(
        smoke_artifact["value"], require_parent_completion=True
    )
    child_close = require_artifact_identity(
        value["child_close_capture"], "position controller child-close capture"
    )
    child_ready = require_artifact_identity(
        value["child_ready_marker"], "position controller child-ready marker"
    )
    child_close_value = validate_immutable_json(Path(child_close["path"]))["value"]
    validate_position_smoke(child_close_value, require_parent_completion=False)
    if (
        smoke["completion"]["raw_sha256"] != child_close["sha256"]
        or smoke["completion"]["ready_sha256"] != child_ready["sha256"]
    ):
        raise ValueError("position controller child lifecycle binding mismatch")
    runtime_identity = value["runtime_identity"]
    runtime = smoke["runtime_contract"]
    if (
        not isinstance(runtime_identity, dict)
        or set(runtime_identity)
        != {
            "container_pinned_by_sha256",
            "isaaclab_version",
            "isaaclab_source_sha256",
            "polaris_runtime_source_sha256",
            "action_term_class",
            "action_cfg_class",
            "host_finalizer_python",
        }
        or runtime_identity["container_pinned_by_sha256"] is not True
        or runtime_identity["isaaclab_version"] != runtime["isaaclab_version"]
        or runtime_identity["isaaclab_source_sha256"]
        != runtime["isaaclab_source_sha256"]
        or runtime_identity["polaris_runtime_source_sha256"]
        != runtime["polaris_runtime_source_sha256"]
        or runtime_identity["action_term_class"] != runtime["action_term_class"]
        or runtime_identity["action_cfg_class"] != runtime["action_cfg_class"]
    ):
        raise ValueError("position controller runtime identity mismatch")
    return {
        **_artifact_identity(artifact),
        "job_id": slurm["job_id"],
        "source_commit": source["commit"],
        "source_is_ancestor": True,
        "governed_source_identity_match": True,
        "authorized_canary_protocol": PROTOCOL,
        "value": value,
        "validated_smoke": smoke,
    }


def validate_host_media_tools(
    repository: Path,
    *,
    expected_manifest_sha256: str,
    ffprobe_path: Path,
    expected_ffprobe_sha256: str,
    ffmpeg_path: Path,
    expected_ffmpeg_sha256: str,
) -> dict[str, Any]:
    manifest_path = Path(repository) / HOST_MEDIA_MANIFEST
    manifest_stat = _regular_file(manifest_path, field="host media manifest")
    payload = manifest_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != _sha256(
        expected_manifest_sha256, field="host media manifest SHA-256"
    ):
        raise ValueError("host media manifest SHA-256 mismatch")
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("host media manifest is not strict JSON") from error
    if (
        not isinstance(manifest, dict)
        or set(manifest)
        != {
            "schema_version",
            "profile",
            "architecture",
            "package_root",
            "source_archive",
            "tools",
        }
        or manifest["schema_version"] != 1
        or manifest["profile"] != HOST_MEDIA_PROFILE
        or manifest["architecture"] != "amd64"
        or set(manifest["tools"]) != {"ffprobe", "ffmpeg"}
    ):
        raise ValueError("host media manifest schema mismatch")
    package_root = Path(manifest["package_root"])
    if (
        package_root.is_symlink()
        or not package_root.is_dir()
        or stat.S_IMODE(package_root.stat().st_mode) != 0o555
    ):
        raise ValueError("host media package root must be a mode-0555 directory")
    provided = {
        "ffprobe": (Path(ffprobe_path), expected_ffprobe_sha256),
        "ffmpeg": (Path(ffmpeg_path), expected_ffmpeg_sha256),
    }
    tools: dict[str, Any] = {}
    for name, (path, expected_digest) in provided.items():
        record = manifest["tools"][name]
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "size", "sha256", "version_line"}
            or str(path) != record["path"]
            or path.parent != package_root
            or _sha256(expected_digest, field=f"host {name} SHA-256")
            != record["sha256"]
        ):
            raise ValueError(f"host {name} manifest binding mismatch")
        file_stat = _regular_file(path, field=f"host {name}", expected_mode=0o555)
        digest = file_sha256(path)
        if file_stat.st_size != record["size"] or digest != record["sha256"]:
            raise ValueError(f"host {name} content identity mismatch")
        try:
            version_line = subprocess.run(
                [str(path), "-version"],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
                env={"LANG": "C", "LC_ALL": "C"},
            ).stdout.splitlines()[0]
        except (
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as error:
            raise ValueError(f"host {name} executable failed") from error
        if version_line != record["version_line"]:
            raise ValueError(f"host {name} version mismatch")
        tools[name] = {
            **_identity(path, field=f"host {name}", expected_mode=0o555),
            "version_line": version_line,
        }
    return {
        "schema_version": 1,
        "profile": HOST_MEDIA_PROFILE,
        "manifest": {
            "path": str(manifest_path.resolve()),
            "size": manifest_stat.st_size,
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        "package_root": str(package_root.resolve()),
        "source_archive": manifest["source_archive"],
        "tools": tools,
    }


def _image_provenance(path: Path) -> dict[str, Any]:
    identity = _identity(path, field="PolaRiS Pyxis image")
    if identity["sha256"] != PI05_DROID_PYXIS_SHA256:
        raise ValueError("PolaRiS Pyxis image SHA-256 mismatch")
    return identity


def _asset_provenance(data_dir: Path) -> dict[str, Any]:
    requested = Path(data_dir)
    if requested.is_symlink() or not requested.is_dir():
        raise ValueError("PolaRiS-Hub root must be a regular directory")
    root = requested.resolve()
    assets: dict[str, Any] = {}
    for relative, expected in PI05_DROID_CANARY_ASSETS.items():
        path = root / relative
        identity = _identity(path, field=f"PolaRiS-Hub asset {relative}")
        metadata = root / ".cache/huggingface/download" / f"{relative}.metadata"
        metadata_identity = _identity(
            metadata, field=f"PolaRiS-Hub metadata {relative}"
        )
        lines = metadata.read_text(encoding="utf-8").splitlines()
        if (
            identity["sha256"] != expected["sha256"]
            or metadata_identity["sha256"] != expected["metadata_sha256"]
            or not lines
            or lines[0] != PI05_DROID_HUB_REVISION
        ):
            raise ValueError(f"PolaRiS-Hub provenance mismatch: {relative}")
        assets[relative] = {
            **identity,
            "metadata": metadata_identity,
            "hub_revision": PI05_DROID_HUB_REVISION,
        }
    return {
        "root": str(root),
        "hub_revision": PI05_DROID_HUB_REVISION,
        "assets": assets,
    }


def _validate_snapshot_creation(path: Path, run_dir: Path) -> dict[str, Any]:
    artifact = validate_immutable_json(path)
    value = artifact["value"]
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "schema_version",
            "profile",
            "status",
            "copy_semantics",
            "checkpoint_uri",
            "source_dir",
            "source_root_stat",
            "snapshot_dir",
            "snapshot_root_stat",
            "manifest_sha256",
            "object_count",
            "total_bytes",
            "objects",
        }
        or value["schema_version"] != 1
        or value["profile"] != SNAPSHOT_PROFILE
        or value["status"] != "pass"
        or value["copy_semantics"]
        != "independent_byte_copy_no_hardlinks_no_reflinks_v1"
        or value["checkpoint_uri"] != PI05_DROID_CHECKPOINT_URI
        or Path(value["snapshot_dir"]).resolve() != run_dir / "checkpoint_snapshot"
        or value["manifest_sha256"] != PI05_DROID_CHECKPOINT_MANIFEST_SHA256
        or value["object_count"] != PI05_DROID_CHECKPOINT_OBJECT_COUNT
        or value["total_bytes"] != PI05_DROID_CHECKPOINT_BYTES
        or not isinstance(value["objects"], list)
        or len(value["objects"]) != PI05_DROID_CHECKPOINT_OBJECT_COUNT
    ):
        raise ValueError("checkpoint snapshot-creation identity mismatch")
    return {**_artifact_identity(artifact), "snapshot": value}


def _validate_checkpoint_attestation(
    path: Path,
    *,
    expected_phase: str,
    run_dir: Path,
    expected_manifest_path: Path,
) -> dict[str, Any]:
    artifact = validate_immutable_json(path)
    value = artifact["value"]
    required = {
        "schema_version",
        "profile",
        "status",
        "verification_phase",
        "integrity_mode",
        "checkpoint_uri",
        "checkpoint_dir",
        "manifest",
        "root_stat",
        "objects",
        "normalization",
        "full_md5",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value["schema_version"] != 1
        or value["profile"] != CHECKPOINT_ATTESTATION_PROFILE
        or value["status"] != "pass"
        or value["verification_phase"] != expected_phase
        or value["integrity_mode"] != CHECKPOINT_INTEGRITY_MODE
        or value["checkpoint_uri"] != PI05_DROID_CHECKPOINT_URI
        or Path(value["checkpoint_dir"]).resolve() != run_dir / "checkpoint_snapshot"
        or Path(value["manifest"].get("path", "")).resolve()
        != Path(expected_manifest_path).resolve()
        or value["manifest"].get("sha256") != PI05_DROID_CHECKPOINT_MANIFEST_SHA256
        or value["manifest"].get("object_count") != PI05_DROID_CHECKPOINT_OBJECT_COUNT
        or value["manifest"].get("total_bytes") != PI05_DROID_CHECKPOINT_BYTES
        or value["normalization"].get("sha256") != PI05_DROID_NORM_STATS_SHA256
        or value["normalization"].get("scope") != "checkpoint_global_droid"
        or value["normalization"].get("category_override") != "forbidden"
        or value["normalization"].get("rejected_category_substitutions")
        != ["single_arm", "single-arm", "single arm"]
        or value["normalization"].get("model_action_semantics")
        != "normalized_droid_joint_velocity_command"
        or value["normalization"].get("simulator_adapter")
        != "fresh_measured_q_plus_0p2_times_clipped_command_to_absolute_position"
        or value["full_md5"] is not True
        or not isinstance(value["objects"], list)
        or len(value["objects"]) != PI05_DROID_CHECKPOINT_OBJECT_COUNT
    ):
        raise ValueError("checkpoint full-MD5 attestation mismatch")
    for item in value["objects"]:
        object_stat = item.get("stat") if isinstance(item, dict) else None
        if (
            not isinstance(object_stat, dict)
            or object_stat.get("mode") != "0444"
            or object_stat.get("nlink") != 1
        ):
            raise ValueError("checkpoint object is not one immutable snapshot file")
    if value["root_stat"].get("mode") != "0555":
        raise ValueError("checkpoint snapshot root is not mode 0555")
    return {**_artifact_identity(artifact), "checkpoint": value}


def _require_checkpoint_unchanged(
    before: dict[str, Any], after: dict[str, Any]
) -> None:
    before_value = copy.deepcopy(before["checkpoint"])
    after_value = copy.deepcopy(after["checkpoint"])
    before_value["verification_phase"] = "phase"
    after_value["verification_phase"] = "phase"
    if canonical_json_bytes(before_value) != canonical_json_bytes(after_value):
        raise ValueError(
            "checkpoint snapshot changed across server/evaluator execution"
        )


def _validate_model_runtime(
    path: Path,
    *,
    checkpoint: dict[str, Any],
    droid: dict[str, Any],
) -> dict[str, Any]:
    artifact = validate_immutable_json(path)
    value = artifact["value"]
    required = {
        "schema_version",
        "profile",
        "position_execution_profile",
        "checkpoint",
        "train_config",
        "transform_runtime",
        "policy",
        "official_model_eval_contract",
        "transform_contract_reference",
        "official_droid_source",
        "openpi_runtime_attestation",
    }
    expected_checkpoint = {
        "sha256": PI05_DROID_CHECKPOINT_MANIFEST_SHA256,
        "object_count": PI05_DROID_CHECKPOINT_OBJECT_COUNT,
        "total_bytes": PI05_DROID_CHECKPOINT_BYTES,
        "checkpoint_dir": checkpoint["checkpoint"]["checkpoint_dir"],
        "norm_stats_sha256": PI05_DROID_NORM_STATS_SHA256,
        "full_md5": True,
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
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value["schema_version"] != 1
        or value["profile"] != MODEL_RUNTIME_PROFILE
        or value["position_execution_profile"] != PI05_DROID_POSITION_ADAPTER_PROFILE
        or value["checkpoint"] != expected_checkpoint
        or value["train_config"] != expected_train_config
        or value["transform_runtime"] != PI05_DROID_POSITION_TRANSFORM_RUNTIME_CONTRACT
        or value["transform_contract_reference"]
        != PI05_DROID_POSITION_TRANSFORM_RUNTIME_CONTRACT
        or value["policy"]
        != {"metadata": {}, "sample_kwargs": {}, "rng_key_data": [0, 0]}
        or value["official_model_eval_contract"]
        != PI05_DROID_POSITION_MODEL_EVAL_CONTRACT
        or value["official_droid_source"] != droid
    ):
        raise ValueError("position model-runtime artifact mismatch")
    model_contract = value["official_model_eval_contract"]
    if (
        model_contract["normalization"]["scope"] != "checkpoint_global_droid"
        or model_contract["normalization"]["category_override"] != "forbidden"
        or model_contract["policy_input"]["image_order"]
        != ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb_masked"]
        or model_contract["policy_input"]["resize"]
        != "openpi_image_tools_resize_with_pad_224_v1"
        or model_contract["policy_input"]["wrist_rotation_degrees"] != 0
        or model_contract["inference"]
        != {
            "objective": "flow",
            "sampler": "flow_euler_t1_to_t0_num_steps10_rng_key0_v1",
            "response_shape": [15, 8],
            "execute_first": 8,
        }
        or model_contract["policy_output"]["simulator_command"]
        != "absolute_joint_position_target"
        or model_contract["policy_output"]["target_hold_physics_substeps"] != 8
    ):
        raise ValueError("position model train/eval contract mismatch")
    openpi_attestation = validate_openpi_runtime_attestation(
        value["openpi_runtime_attestation"]
    )
    return {
        **_artifact_identity(artifact),
        "value": value,
        "openpi_runtime_attestation": openpi_attestation,
    }


def _validate_inference_environment(path: Path, openpi_dir: Path) -> dict[str, Any]:
    artifact = validate_immutable_json(path)
    value = validate_position_environment(
        artifact["value"],
        openpi_dir,
        Path(openpi_dir) / ".venv/bin/python",
    )
    return {**_artifact_identity(artifact), "value": value}


def _validate_gpu_inventory(path: Path, *, job_id: int) -> dict[str, Any]:
    artifact = validate_immutable_json(path)
    value = artifact["value"]
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "job_id", "gpus"}
        or value["schema_version"] != 1
        or value["job_id"] != job_id
        or not isinstance(value["gpus"], list)
        or len(value["gpus"]) != 1
        or not isinstance(value["gpus"][0], dict)
        or set(value["gpus"][0]) != {"uuid", "name", "driver_version"}
        or value["gpus"][0]["name"] != "NVIDIA L40S"
        or not str(value["gpus"][0]["uuid"]).startswith("GPU-")
        or not value["gpus"][0]["driver_version"]
    ):
        raise ValueError("one-L40S GPU inventory mismatch")
    return {**_artifact_identity(artifact), "gpu": value["gpus"][0]}


def _validate_preflight_record(
    path: Path,
    *,
    source: dict[str, Any],
    openpi: dict[str, Any],
    droid: dict[str, Any],
    controller: dict[str, Any],
    host_media: dict[str, Any],
    image: dict[str, Any],
    assets: dict[str, Any],
) -> dict[str, Any]:
    artifact = validate_immutable_json(path)
    expected = {
        "schema_version": 1,
        "profile": CANARY_PROFILE,
        "protocol": PROTOCOL,
        "evaluation_id": EVALUATION_ID,
        "status": "pass",
        "source": source,
        "openpi": openpi,
        "official_droid": droid,
        "position_controller": controller,
        "host_media_tools": host_media,
        "container_image": image,
        "polaris_hub": assets,
        "checkpoint_manifest": verify_profile_manifest(
            Path(source["root"]) / "scripts/polaris/pi05_droid_native_gcs_manifest.tsv"
        ),
    }
    if canonical_json_bytes(artifact["value"]) != canonical_json_bytes(expected):
        raise ValueError("preflight record differs from final host validation")
    return _artifact_identity(artifact)


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
        "protocol",
        "job_id",
        "run_dir",
        "polaris_dir",
        "polaris_commit",
        "openpi_dir",
        "droid_dir",
        "sbatch_script",
        "sbatch_script_sha256",
        "container_image",
        "polaris_data_dir",
        "position_controller_attestation",
        "position_controller_attestation_sha256",
        "host_media",
        "checkpoint_snapshot",
        "fresh_attempt_no_resume",
        "task",
        "rollouts",
    }
    relative_sbatch = "scripts/polaris/l40s_pi05_droid_position_canary.sbatch"
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value["schema_version"] != 1
        or value["profile"] != CANARY_PROFILE
        or value["protocol"] != PROTOCOL
        or value["job_id"] != args.job_id
        or Path(value["run_dir"]).resolve() != run_dir
        or Path(value["polaris_dir"]).resolve() != Path(args.polaris_repo).resolve()
        or value["polaris_commit"] != args.expected_polaris_commit
        or Path(value["openpi_dir"]).resolve() != Path(args.openpi_dir).resolve()
        or Path(value["droid_dir"]).resolve() != Path(args.droid_dir).resolve()
        or Path(value["sbatch_script"]).resolve()
        != (Path(args.polaris_repo) / relative_sbatch).resolve()
        or value["sbatch_script_sha256"] != source["files"][relative_sbatch]["sha256"]
        or Path(value["container_image"]).resolve()
        != Path(args.container_image).resolve()
        or Path(value["polaris_data_dir"]).resolve() != Path(args.data_dir).resolve()
        or Path(value["position_controller_attestation"]).resolve()
        != Path(args.position_controller_attestation).resolve()
        or value["position_controller_attestation_sha256"]
        != args.expected_position_controller_attestation_sha256
        or Path(value["checkpoint_snapshot"]).resolve()
        != run_dir / "checkpoint_snapshot"
        or value["fresh_attempt_no_resume"] is not True
        or value["task"] != PI05_DROID_NATIVE_TASK
        or value["rollouts"] != 1
    ):
        raise ValueError("submission record identity mismatch")
    host_media = value["host_media"]
    if (
        not isinstance(host_media, dict)
        or set(host_media)
        != {
            "manifest_sha256",
            "ffprobe_path",
            "ffprobe_sha256",
            "ffmpeg_path",
            "ffmpeg_sha256",
        }
        or host_media["manifest_sha256"]
        != args.expected_host_media_tools_manifest_sha256
        or host_media["ffprobe_path"] != str(args.host_ffprobe_path)
        or host_media["ffprobe_sha256"] != args.expected_host_ffprobe_sha256
        or host_media["ffmpeg_path"] != str(args.host_ffmpeg_path)
        or host_media["ffmpeg_sha256"] != args.expected_host_ffmpeg_sha256
    ):
        raise ValueError("submission host-media binding mismatch")
    return _artifact_identity(artifact)


def resolved_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "profile": CANARY_PROFILE,
        "protocol": PROTOCOL,
        "evaluation_id": EVALUATION_ID,
        "scope": "one_rollout_wiring_canary_not_standard_success_rate",
        "visual_adjudication": "pending_workstation_review",
        "benchmark": "PolaRiS DROID task suite",
        "task": PI05_DROID_NATIVE_TASK,
        "instruction": TASK_INSTRUCTION,
        "rollouts": 1,
        "episode": {
            "policy_steps": 450,
            "policy_frequency_hz": 15,
            "physics_frequency_hz": 120,
            "physics_substeps_per_policy_step": 8,
            "expected_queries": 57,
            "expected_target_apply_calls": 3600,
            "initial_condition_index": 0,
            "auto_reset": False,
        },
        "checkpoint": {
            "uri": PI05_DROID_CHECKPOINT_URI,
            "content_manifest_sha256": PI05_DROID_CHECKPOINT_MANIFEST_SHA256,
            "object_count": PI05_DROID_CHECKPOINT_OBJECT_COUNT,
            "total_bytes": PI05_DROID_CHECKPOINT_BYTES,
            "integrity": "run_specific_0555_tree_0444_nlink1_files_full_md5_pre_post",
        },
        "openpi": {
            "commit": PI05_DROID_OPENPI_INFERENCE_COMPATIBILITY_COMMIT,
            "config": "pi05_droid",
            "model": "pi05",
            "dtype": "bfloat16",
            "jax_enable_x64": False,
        },
        "inference": {
            "mode": "flow",
            "sampler": "flow_euler_t1_to_t0_num_steps10_rng_key0_v1",
            "flow_steps": 10,
            "rng_seed": 0,
            "response_shape": [15, 8],
            "execute_first": 8,
        },
        "observations": {
            "state": "panda_joint1_through_7_radians_plus_closed_positive_gripper",
            "state_width": 8,
            "native_images": {
                "external": [720, 1280, 3],
                "wrist": [720, 1280, 3],
                "dtype": "uint8_rgb",
            },
            "model_image_order": [
                "base_0_rgb",
                "left_wrist_0_rgb",
                "right_wrist_0_rgb_masked",
            ],
            "model_image_shape": [224, 224, 3],
            "resize": "openpi_image_tools_resize_with_pad_224_v1",
            "wrist_rotation_degrees": 0,
            "third_image": "zero_masked_blank",
        },
        "normalization": {
            "scope": "checkpoint_global_droid",
            "asset_id": "droid",
            "sha256": PI05_DROID_NORM_STATS_SHA256,
            "category_override": "forbidden",
            "single_arm": "forbidden_checkpoint_has_global_stats",
        },
        "execution": {
            "client": "DroidDeltaJointPosition",
            "control_mode": "joint-position",
            "policy_profile": PI05_DROID_POSITION_ADAPTER_PROFILE,
            "frame_description": "robot base frame",
            "action_frame": "robot_base",
            "dataset_name": "droid",
            "arm_formula": "q_target_t=fresh_measured_q_t+0.2*clip(command_t,-1,1)",
            "fresh_measurement_each_executed_step": True,
            "simulator_command": "absolute_joint_position_target",
            "same_target_apply_calls": 8,
            "gripper": "absolute_closed_positive_binarize_gt_0p5",
            "position_limits": expected_position_limit_contract(),
        },
        "transport": {
            "host": "127.0.0.1",
            "requested_port": 0,
            "port_selection": "os_assigned_atomic_bind",
            "readiness": "real_websocket_metadata_handshake",
        },
        "lifecycle": {
            "order": [
                "env_close",
                "publish_mode0444_close_ready",
                "SimulationApp_close",
                "srun_exit_zero",
                "host_finalizer",
            ],
            "numerical_failure_is_distinct": True,
        },
    }


def _validate_resolved_contract(path: Path) -> dict[str, Any]:
    artifact = validate_immutable_json(path)
    if canonical_json_bytes(artifact["value"]) != canonical_json_bytes(
        resolved_contract()
    ):
        raise ValueError("resolved canary contract mismatch")
    return _artifact_identity(artifact)


def _validate_run_record(
    path: Path,
    *,
    args: argparse.Namespace,
    run_dir: Path,
    resolved: dict[str, Any],
) -> dict[str, Any]:
    artifact = validate_immutable_json(path)
    value = artifact["value"]
    required = {
        "schema_version",
        "profile",
        "protocol",
        "job_id",
        "task",
        "rollouts",
        "fresh_attempt_no_resume",
        "paths",
        "listener",
        "controller_authorization",
        "resolved_contract_artifact",
    }
    expected_paths = {
        "run_dir": run_dir,
        "polaris_dir": Path(args.polaris_repo),
        "openpi_dir": Path(args.openpi_dir),
        "openpi_git_dir": Path(
            _git(Path(args.openpi_dir), "rev-parse", "--absolute-git-dir")
            .decode()
            .strip()
        ),
        "openpi_git_common_dir": Path(
            _git(
                Path(args.openpi_dir),
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            )
            .decode()
            .strip()
        ),
        "droid_dir": Path(args.droid_dir),
        "checkpoint_snapshot": run_dir / "checkpoint_snapshot",
        "checkpoint_manifest": Path(args.polaris_repo)
        / "scripts/polaris/pi05_droid_native_gcs_manifest.tsv",
        "container_image": Path(args.container_image),
        "polaris_data_dir": Path(args.data_dir),
        "serving_contract": run_dir / PI05_DROID_CONTRACT_FILENAME,
        "model_runtime_contract": run_dir / "pi05_droid_position_model_runtime.json",
        "bound_port": run_dir / "policy_bound_port.json",
        "handshake": run_dir / "policy_handshake.json",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value["schema_version"] != 1
        or value["profile"] != CANARY_PROFILE
        or value["protocol"] != PROTOCOL
        or value["job_id"] != args.job_id
        or value["task"] != PI05_DROID_NATIVE_TASK
        or value["rollouts"] != 1
        or value["fresh_attempt_no_resume"] is not True
        or not isinstance(value["paths"], dict)
        or set(value["paths"]) != set(expected_paths)
        or any(
            Path(value["paths"][name]).resolve() != expected.resolve()
            for name, expected in expected_paths.items()
        )
        or value["controller_authorization"]
        != {
            "profile": CONTROLLER_ATTESTATION_PROFILE,
            "path": str(Path(args.position_controller_attestation).resolve()),
            "sha256": args.expected_position_controller_attestation_sha256,
            "old_jointvelocity_controller_gates": "forbidden",
        }
        or value["resolved_contract_artifact"] != resolved
    ):
        raise ValueError("run record identity mismatch")
    listener = value["listener"]
    if (
        not isinstance(listener, dict)
        or set(listener)
        != {
            "requested_port",
            "actual_port",
            "server_pid",
            "launch_token",
            "bound_port_sha256",
            "bound_port_stable_identity",
        }
        or listener["requested_port"] != 0
        or type(listener["actual_port"]) is not int
        or not 1 <= listener["actual_port"] <= 65535
        or type(listener["server_pid"]) is not int
        or listener["server_pid"] <= 0
        or _SHA256.fullmatch(listener["launch_token"]) is None
        or _SHA256.fullmatch(listener["bound_port_sha256"]) is None
        or _BOUND_IDENTITY.fullmatch(listener["bound_port_stable_identity"]) is None
    ):
        raise ValueError("run listener identity mismatch")
    return {**_artifact_identity(artifact), "listener": listener}


def _validate_bound_port(path: Path, listener: dict[str, Any]) -> dict[str, Any]:
    value, port, digest, stable_identity = load_bound_port_record(
        path,
        expected_pid=listener["server_pid"],
        expected_launch_token=listener["launch_token"],
        expected_requested_port=0,
        require_live_pid=False,
    )
    if (
        port != listener["actual_port"]
        or digest != listener["bound_port_sha256"]
        or stable_identity != listener["bound_port_stable_identity"]
    ):
        raise ValueError("bound-port artifact changed after handshake/evaluation")
    identity = [int(part) for part in stable_identity.split(":")]
    if len(identity) != 7 or identity[-2:] != [0o444, 1]:
        raise ValueError("bound-port artifact immutable identity mismatch")
    return {
        "path": str(Path(path).resolve()),
        "size": identity[2],
        "sha256": digest,
        "mode": "0444",
        "nlink": 1,
        "stable_identity": stable_identity,
        "value": value,
    }


def _validate_handshake(
    path: Path,
    *,
    listener: dict[str, Any],
    openpi_dir: Path,
    serving_contract: dict[str, Any],
) -> dict[str, Any]:
    artifact = validate_immutable_json(path)
    value = artifact["value"]
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "schema_version",
            "profile",
            "host",
            "actual_port",
            "server_pid",
            "openpi_dir",
            "serving_contract",
            "contract_sha256",
        }
        or value["schema_version"] != 1
        or value["profile"] != HANDSHAKE_PROFILE
        or value["host"] != "127.0.0.1"
        or value["actual_port"] != listener["actual_port"]
        or value["server_pid"] != listener["server_pid"]
        or Path(value["openpi_dir"]).resolve() != Path(openpi_dir).resolve()
        or value["serving_contract"] != serving_contract
        or value["contract_sha256"] != serving_contract["contract_sha256"]
    ):
        raise ValueError("position WebSocket handshake artifact mismatch")
    return _artifact_identity(artifact)


def _validate_srun_status(path: Path, *, job_id: int) -> dict[str, Any]:
    artifact = validate_immutable_json(path)
    expected = {
        "schema_version": 1,
        "profile": CANARY_PROFILE,
        "job_id": job_id,
        "srun_exit_code": 0,
        "tee_exit_code": 0,
        "bound_port_unchanged_after_eval": True,
        "checkpoint_post_attestation_present": True,
    }
    if artifact["value"] != expected:
        raise ValueError("evaluator child srun status mismatch")
    return _artifact_identity(artifact)


def _strict_json_lines(path: Path) -> list[dict[str, Any]]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"trace contains forbidden JSON constant {value}")

    records: list[dict[str, Any]] = []
    for index, line in enumerate(
        path.read_text(encoding="ascii").splitlines(), start=1
    ):
        if not line:
            raise ValueError(f"trace line {index} is empty")
        try:
            value = json.loads(line, parse_constant=reject_constant)
        except json.JSONDecodeError as error:
            raise ValueError(f"trace line {index} is invalid") from error
        if not isinstance(value, dict):
            raise ValueError(f"trace line {index} is not an object")
        records.append(value)
    return records


def _validate_query_image_contract(
    trace_path: Path, *, expected_queries: int
) -> dict[str, Any]:
    records = _strict_json_lines(trace_path)
    queries = [
        record
        for record in records
        if record.get("record_type") == "openpi_droid_position_query"
    ]
    if len(queries) != expected_queries:
        raise ValueError("position query count changed during image audit")
    expected_image_keys = {
        "native_external",
        "native_wrist",
        "external",
        "wrist",
        "blank_masked_right_wrist",
        "resize",
        "wrist_rotation_degrees",
    }
    for query_index, query in enumerate(queries):
        if (
            query.get("query_index") != query_index
            or query.get("prompt") != TASK_INSTRUCTION
            or query.get("model_image_order")
            != ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb_masked"]
            or query.get("normalization_scope") != "checkpoint_global_droid"
            or query.get("normalization_asset_id") != "droid"
            or query.get("sampler") != "flow_euler_t1_to_t0_num_steps10_rng_key0_v1"
            or query.get("response_action_shape") != [15, 8]
            or query.get("execution_horizon") != 8
        ):
            raise ValueError("position query train/eval contract mismatch")
        images = query.get("images")
        if (
            not isinstance(images, dict)
            or set(images) != expected_image_keys
            or images["resize"] != "openpi_image_tools_resize_with_pad_224_v1"
            or images["wrist_rotation_degrees"] != 0
        ):
            raise ValueError("position query image-routing schema mismatch")
        for name in ("native_external", "native_wrist"):
            identity = images[name]
            if (
                not isinstance(identity, dict)
                or set(identity) != {"shape", "dtype", "sha256"}
                or identity["shape"] != [720, 1280, 3]
                or identity["dtype"] != "uint8"
                or _SHA256.fullmatch(str(identity["sha256"])) is None
            ):
                raise ValueError(f"position query native image mismatch: {name}")
        for name in ("external", "wrist", "blank_masked_right_wrist"):
            identity = images[name]
            if (
                not isinstance(identity, dict)
                or set(identity) != {"shape", "dtype", "sha256"}
                or identity["shape"] != [224, 224, 3]
                or identity["dtype"] != "uint8"
                or _SHA256.fullmatch(str(identity["sha256"])) is None
            ):
                raise ValueError(f"position query resized image mismatch: {name}")
    return {
        "query_count": len(queries),
        "native_shape": [720, 1280, 3],
        "model_shape": [224, 224, 3],
        "model_order": [
            "base_0_rgb",
            "left_wrist_0_rgb",
            "right_wrist_0_rgb_masked",
        ],
        "resize": "openpi_image_tools_resize_with_pad_224_v1",
        "wrist_rotation_degrees": 0,
        "normalization_scope": "checkpoint_global_droid",
        "instruction": TASK_INSTRUCTION,
    }


def _fraction(value: str) -> float:
    numerator, denominator = value.split("/", maxsplit=1)
    return float(numerator) / float(denominator)


def _top_level_mp4_boxes(path: Path) -> list[str]:
    boxes: list[str] = []
    with path.open("rb") as source:
        while True:
            header = source.read(8)
            if not header:
                break
            if len(header) != 8:
                raise ValueError("truncated MP4 box")
            size = int.from_bytes(header[:4], "big")
            box_type = header[4:].decode("ascii", errors="strict")
            header_size = 8
            if size == 1:
                large = source.read(8)
                if len(large) != 8:
                    raise ValueError("truncated extended MP4 box")
                size = int.from_bytes(large, "big")
                header_size = 16
            elif size == 0:
                size = path.stat().st_size - source.tell() + header_size
            if size < header_size:
                raise ValueError("invalid MP4 box size")
            boxes.append(box_type)
            source.seek(size - header_size, os.SEEK_CUR)
    return boxes


def probe_video(
    path: Path,
    *,
    expected_frame_count: int,
    require_faststart: bool,
    ffprobe: Path,
    ffmpeg: Path,
) -> dict[str, Any]:
    _regular_file(path, field="video", expected_mode=0o444)
    try:
        payload = subprocess.run(
            [
                str(ffprobe),
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                (
                    "stream=codec_name,pix_fmt,field_order,width,height,"
                    "avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,duration"
                ),
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        value = json.loads(payload)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise ValueError(f"ffprobe failed: {path}") from error
    streams = value.get("streams")
    if not isinstance(streams, list) or len(streams) != 1:
        raise ValueError("video must contain exactly one video stream")
    stream = streams[0]
    frames = int(stream.get("nb_read_frames") or stream.get("nb_frames") or 0)
    duration = float(stream.get("duration", "nan"))
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
            _fraction(stream["r_frame_rate"]),
            PI05_DROID_NATIVE_POLICY_HZ,
            abs_tol=1e-9,
        )
        or not 1 <= expected_frame_count <= PI05_DROID_NATIVE_EPISODE_STEPS
        or frames != expected_frame_count
        or not math.isclose(
            duration,
            expected_frame_count / PI05_DROID_NATIVE_POLICY_HZ,
            abs_tol=1e-6,
        )
    ):
        raise ValueError(f"video stream contract mismatch: {stream}")
    try:
        subprocess.run(
            [
                str(ffmpeg),
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
        raise ValueError(f"full video decode failed: {path}") from error
    boxes = _top_level_mp4_boxes(path)
    faststart = (
        "moov" in boxes
        and "mdat" in boxes
        and boxes.index("moov") < boxes.index("mdat")
    )
    if require_faststart and not faststart:
        raise ValueError("summary video is not fast-start")
    return {
        "codec": "h264",
        "pixel_format": "yuv420p",
        "field_order": "progressive",
        "width": PI05_DROID_NATIVE_VIDEO_WIDTH,
        "height": PI05_DROID_NATIVE_VIDEO_HEIGHT,
        "fps": PI05_DROID_NATIVE_POLICY_HZ,
        "frame_count": frames,
        "duration_seconds": duration,
        "full_decode": "pass",
        "top_level_boxes": boxes,
        "faststart": faststart,
    }


def create_summary_video(
    source: Path,
    output: Path,
    *,
    source_frame_count: int,
    ffprobe: Path,
    ffmpeg: Path,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing existing summary video: {output}")
    temporary = output.with_name(f".{output.stem}.partial-{os.getpid()}.mp4")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"refusing existing summary temporary: {temporary}")
    frame_count = max(45, source_frame_count)
    try:
        subprocess.run(
            [
                str(ffmpeg),
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
                str(frame_count),
                "-r",
                str(PI05_DROID_NATIVE_POLICY_HZ),
                str(temporary),
            ],
            check=True,
        )
        temporary.chmod(0o444)
        with temporary.open("rb") as output_file:
            os.fsync(output_file.fileno())
        os.link(temporary, output)
        temporary.unlink()
        fsync_directory(output.parent)
        return probe_video(
            output,
            expected_frame_count=frame_count,
            require_faststart=True,
            ffprobe=ffprobe,
            ffmpeg=ffmpeg,
        )
    except BaseException:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
        raise


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    source = _source_provenance(args.polaris_repo, args.expected_polaris_commit)
    openpi = _openpi_provenance(args.openpi_dir)
    droid = verify_official_droid_git_checkout(args.droid_dir)
    host_media = validate_host_media_tools(
        args.polaris_repo,
        expected_manifest_sha256=args.expected_host_media_tools_manifest_sha256,
        ffprobe_path=args.host_ffprobe_path,
        expected_ffprobe_sha256=args.expected_host_ffprobe_sha256,
        ffmpeg_path=args.host_ffmpeg_path,
        expected_ffmpeg_sha256=args.expected_host_ffmpeg_sha256,
    )
    image = _image_provenance(args.container_image)
    assets = _asset_provenance(args.data_dir)
    controller = validate_position_controller_attestation(
        args.position_controller_attestation,
        args.expected_position_controller_attestation_sha256,
        canary_source=source,
        expected_image=image,
        expected_assets=assets,
    )
    manifest = verify_profile_manifest(
        Path(args.polaris_repo) / "scripts/polaris/pi05_droid_native_gcs_manifest.tsv"
    )
    value = {
        "schema_version": 1,
        "profile": CANARY_PROFILE,
        "protocol": PROTOCOL,
        "evaluation_id": EVALUATION_ID,
        "status": "pass",
        "source": source,
        "openpi": openpi,
        "official_droid": droid,
        "position_controller": controller,
        "host_media_tools": host_media,
        "container_image": image,
        "polaris_hub": assets,
        "checkpoint_manifest": manifest,
    }
    if args.output is not None:
        publish_immutable_json(args.output, value)
    return value


def publish_failure_artifact(
    *,
    output: Path,
    run_dir: Path,
    job_id: int,
    failure_stage: str,
    exit_code: int,
) -> dict[str, Any]:
    allowed_stages = {
        "checkpoint_pre_attestation",
        "inference_environment",
        "server_bind_and_readiness",
        "evaluator_execution",
        "checkpoint_post_attestation",
        "host_finalization",
        "unknown",
    }
    if failure_stage not in allowed_stages or job_id <= 0 or exit_code == 0:
        raise ValueError("failure artifact identity mismatch")
    root = Path(run_dir).resolve()
    if not root.is_dir() or Path(output).resolve().parent != root:
        raise ValueError("failure artifact must be inside the run directory")
    evidence: dict[str, Any] = {}
    candidates = {
        "preflight": root / "preflight.json",
        "snapshot_creation": root / "checkpoint_snapshot_creation.json",
        "checkpoint_pre": root / "checkpoint_pre_attestation.json",
        "checkpoint_post": root / "checkpoint_post_attestation.json",
        "inference_environment": root / "inference_environment.json",
        "bound_port": root / "policy_bound_port.json",
        "handshake": root / "policy_handshake.json",
        "run_record": root / "run_record.json",
        "resolved_contract": root / "resolved_contract.json",
        "srun_status": root / f"srun-{job_id}.status.json",
        "close_ready": root / PI05_DROID_NATIVE_TASK / "evaluator_close_ready.json",
    }
    for name, path in candidates.items():
        if path.exists() and not path.is_symlink() and path.is_file():
            evidence[name] = _identity(path, field=f"failure evidence {name}")
    value = {
        "schema_version": 1,
        "profile": FAILURE_PROFILE,
        "protocol": PROTOCOL,
        "status": "failed_not_ready_for_promotion",
        "job_id": job_id,
        "run_dir": str(root),
        "exit_code": exit_code,
        "failure_stage": failure_stage,
        "old_jointvelocity_controller_gates_used": False,
        "evidence": evidence,
    }
    publish_immutable_json(output, value)
    return value


def _compare_snapshot_creation(creation: dict[str, Any], pre: dict[str, Any]) -> None:
    created = {item["relative_path"]: item for item in creation["snapshot"]["objects"]}
    attested = {item["relative_path"]: item for item in pre["checkpoint"]["objects"]}
    if set(created) != set(attested):
        raise ValueError("snapshot creation/attestation object-set mismatch")
    if creation["snapshot"]["snapshot_root_stat"] != pre["checkpoint"]["root_stat"]:
        raise ValueError("snapshot creation root identity drift")
    for relative in created:
        if (
            created[relative]["size"] != attested[relative]["size"]
            or created[relative]["md5_base64"] != attested[relative]["md5_base64"]
            or created[relative]["snapshot_stat"] != attested[relative]["stat"]
        ):
            raise ValueError(f"snapshot creation identity drift: {relative}")


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    requested_run_dir = Path(args.run_dir)
    if requested_run_dir.is_symlink():
        raise ValueError("run directory must not be a symlink")
    run_dir = requested_run_dir.resolve()
    if not run_dir.is_dir():
        raise ValueError("run directory is missing")
    task_dir = run_dir / PI05_DROID_NATIVE_TASK

    source = _source_provenance(args.polaris_repo, args.expected_polaris_commit)
    openpi = _openpi_provenance(args.openpi_dir)
    droid = verify_official_droid_git_checkout(args.droid_dir)
    host_media_pre = validate_host_media_tools(
        args.polaris_repo,
        expected_manifest_sha256=args.expected_host_media_tools_manifest_sha256,
        ffprobe_path=args.host_ffprobe_path,
        expected_ffprobe_sha256=args.expected_host_ffprobe_sha256,
        ffmpeg_path=args.host_ffmpeg_path,
        expected_ffmpeg_sha256=args.expected_host_ffmpeg_sha256,
    )
    image = _image_provenance(args.container_image)
    assets = _asset_provenance(args.data_dir)
    controller = validate_position_controller_attestation(
        args.position_controller_attestation,
        args.expected_position_controller_attestation_sha256,
        canary_source=source,
        expected_image=image,
        expected_assets=assets,
    )
    preflight_record = _validate_preflight_record(
        run_dir / "preflight.json",
        source=source,
        openpi=openpi,
        droid=droid,
        controller=controller,
        host_media=host_media_pre,
        image=image,
        assets=assets,
    )
    snapshot_creation = _validate_snapshot_creation(
        run_dir / "checkpoint_snapshot_creation.json", run_dir
    )
    checkpoint_pre = _validate_checkpoint_attestation(
        run_dir / "checkpoint_pre_attestation.json",
        expected_phase="pre_server",
        run_dir=run_dir,
        expected_manifest_path=Path(args.polaris_repo)
        / "scripts/polaris/pi05_droid_native_gcs_manifest.tsv",
    )
    checkpoint_post = _validate_checkpoint_attestation(
        run_dir / "checkpoint_post_attestation.json",
        expected_phase="post_server",
        run_dir=run_dir,
        expected_manifest_path=Path(args.polaris_repo)
        / "scripts/polaris/pi05_droid_native_gcs_manifest.tsv",
    )
    _compare_snapshot_creation(snapshot_creation, checkpoint_pre)
    _require_checkpoint_unchanged(checkpoint_pre, checkpoint_post)

    resolved = _validate_resolved_contract(run_dir / "resolved_contract.json")
    run_record = _validate_run_record(
        run_dir / "run_record.json",
        args=args,
        run_dir=run_dir,
        resolved=resolved,
    )
    submission = _validate_submission_record(
        run_dir / f"submission-{args.job_id}.json",
        args=args,
        run_dir=run_dir,
        source=source,
    )
    model_runtime = _validate_model_runtime(
        run_dir / "pi05_droid_position_model_runtime.json",
        checkpoint=checkpoint_pre,
        droid=droid,
    )
    inference_environment = _validate_inference_environment(
        run_dir / "inference_environment.json", Path(args.openpi_dir)
    )

    serving_path = run_dir / PI05_DROID_CONTRACT_FILENAME
    serving_artifact = validate_persisted_position_serving_contract(serving_path)
    serving_metadata = json.loads(serving_path.read_bytes())
    serving_contract = validate_pi05_droid_position_server_metadata(serving_metadata)
    if (
        serving_artifact["contract_sha256"] != serving_contract["contract_sha256"]
        or model_runtime["value"]["official_model_eval_contract"]
        != PI05_DROID_POSITION_MODEL_EVAL_CONTRACT
    ):
        raise ValueError("serving/model position contract mismatch")
    bound_port = _validate_bound_port(
        run_dir / "policy_bound_port.json", run_record["listener"]
    )
    handshake = _validate_handshake(
        run_dir / "policy_handshake.json",
        listener=run_record["listener"],
        openpi_dir=Path(args.openpi_dir),
        serving_contract=serving_artifact,
    )
    srun_status = _validate_srun_status(
        run_dir / f"srun-{args.job_id}.status.json", job_id=args.job_id
    )
    gpu = _validate_gpu_inventory(
        run_dir / f"gpu-{args.job_id}.json", job_id=args.job_id
    )

    trace_path = task_dir / "policy_traces.jsonl"
    metrics_path = task_dir / "eval_results.csv"
    runtime_path = task_dir / "position_runtime.json"
    close_ready_path = task_dir / "evaluator_close_ready.json"
    sidecar_path = task_dir / "native_runtime/episode_000000.json"
    raw_video_path = task_dir / "episode_0.mp4"
    episode_audit = audit_position_episode(
        trace_path=trace_path,
        metrics_path=metrics_path,
        runtime_path=runtime_path,
        close_ready_path=close_ready_path,
        sidecar_path=sidecar_path,
        video_path=raw_video_path,
    )
    if (
        episode_audit["serving_contract_sha256"] != serving_contract["contract_sha256"]
        or episode_audit["serving_contract_artifact_sha256"]
        != serving_artifact["sha256"]
    ):
        raise ValueError("trace serving-contract binding mismatch")
    trace_audit_artifact = publish_immutable_json(
        task_dir / "position_trace_validation.json", episode_audit
    )
    query_images = _validate_query_image_contract(
        trace_path,
        expected_queries=episode_audit["trace_summary"]["queries"],
    )

    metrics = episode_audit["metrics"]
    raw_video_probe = probe_video(
        raw_video_path,
        expected_frame_count=metrics["episode_length"],
        require_faststart=False,
        ffprobe=Path(host_media_pre["tools"]["ffprobe"]["path"]),
        ffmpeg=Path(host_media_pre["tools"]["ffmpeg"]["path"]),
    )
    summary_path = run_dir / "pi05_droid_position_canary.mp4"
    summary_probe = create_summary_video(
        raw_video_path,
        summary_path,
        source_frame_count=metrics["episode_length"],
        ffprobe=Path(host_media_pre["tools"]["ffprobe"]["path"]),
        ffmpeg=Path(host_media_pre["tools"]["ffmpeg"]["path"]),
    )
    summary_identity = _identity(
        summary_path, field="position canary summary", expected_mode=0o444
    )
    summary_sidecar_value = {
        "schema_version": 1,
        "profile": SUMMARY_PROFILE,
        "protocol": PROTOCOL,
        "task": PI05_DROID_NATIVE_TASK,
        "selection": "the_only_canary_episode",
        "source_video": episode_audit["artifacts"]["video"],
        "summary_video": summary_identity,
        "raw_video_probe": raw_video_probe,
        "summary_video_probe": summary_probe,
        "scientific_outcome": episode_audit["scientific_outcome"],
        "raw_success": metrics["success"],
        "progress": metrics["progress"],
    }
    summary_sidecar = publish_immutable_json(
        summary_path.with_suffix(".json"), summary_sidecar_value
    )

    saved_spool = _seal_file(
        run_dir / f"job-{args.job_id}.sbatch", field="saved Slurm spool script"
    )
    expected_sbatch = source["files"][
        "scripts/polaris/l40s_pi05_droid_position_canary.sbatch"
    ]
    if (
        saved_spool["size"] != expected_sbatch["size"]
        or saved_spool["sha256"] != expected_sbatch["sha256"]
    ):
        raise ValueError("saved Slurm spool script differs from committed blob")
    commands = _seal_file(run_dir / "commands.sh", field="exact launch commands")
    server_log = _seal_file(run_dir / "policy_server.log", field="policy server log")
    evaluator_log = _seal_file(task_dir / "eval.log", field="evaluator log")
    slurm_job = _seal_file(
        run_dir / f"slurm-job-{args.job_id}.txt", field="Slurm job record"
    )
    slurm_text = Path(slurm_job["path"]).read_text(encoding="utf-8")
    for token in (
        f"JobId={args.job_id}",
        "JobName=pi05_pos_canary",
        "Partition=batch",
        "Account=nvr_lpr_rvp",
    ):
        if token not in slurm_text:
            raise ValueError(f"Slurm job record is missing {token}")
    if "ArrayJobId=" in slurm_text or "ArrayTaskId=" in slurm_text:
        raise ValueError("position canary must be an ordinary non-array job")

    host_media_post = validate_host_media_tools(
        args.polaris_repo,
        expected_manifest_sha256=args.expected_host_media_tools_manifest_sha256,
        ffprobe_path=args.host_ffprobe_path,
        expected_ffprobe_sha256=args.expected_host_ffprobe_sha256,
        ffmpeg_path=args.host_ffmpeg_path,
        expected_ffmpeg_sha256=args.expected_host_ffmpeg_sha256,
    )
    if canonical_json_bytes(host_media_post) != canonical_json_bytes(host_media_pre):
        raise ValueError("host media tools changed during finalization")
    if canonical_json_bytes(
        _image_provenance(args.container_image)
    ) != canonical_json_bytes(image):
        raise ValueError("container image changed during evaluation")
    if canonical_json_bytes(_asset_provenance(args.data_dir)) != canonical_json_bytes(
        assets
    ):
        raise ValueError("PolaRiS-Hub assets changed during evaluation")
    controller_post = validate_position_controller_attestation(
        args.position_controller_attestation,
        args.expected_position_controller_attestation_sha256,
        canary_source=source,
        expected_image=image,
        expected_assets=assets,
    )
    if canonical_json_bytes(controller_post) != canonical_json_bytes(controller):
        raise ValueError("position controller authorization changed during evaluation")

    completion = {
        "schema_version": 1,
        "profile": CANARY_PROFILE,
        "protocol": PROTOCOL,
        "evaluation_id": EVALUATION_ID,
        "status": "pass",
        "scientific_outcome": episode_audit["scientific_outcome"],
        "scope": "one_rollout_wiring_canary_not_standard_success_rate",
        "visual_adjudication": "pending_workstation_review",
        "job_id": args.job_id,
        "task": PI05_DROID_NATIVE_TASK,
        "rollouts": 1,
        "episode_steps": metrics["episode_length"],
        "result": metrics,
        "source": source,
        "openpi": openpi,
        "official_droid": droid,
        "position_controller_authorization": controller,
        "checkpoint": {
            "snapshot_creation": snapshot_creation,
            "pre": checkpoint_pre,
            "post": checkpoint_post,
            "identity_match_excluding_only_phase": True,
        },
        "resolved_contract": {
            **resolved,
            "value": resolved_contract(),
        },
        "model_runtime": model_runtime,
        "serving_contract": {**serving_artifact, "value": serving_contract},
        "serving_transport": {
            "bound_port": bound_port,
            "websocket_handshake": handshake,
        },
        "query_image_contract": query_images,
        "episode_transaction": episode_audit,
        "inference_environment": inference_environment,
        "host_finalization": {
            "profile": "pi05_droid_position_separate_post_srun_finalizer_v1",
            "host_media_tools_pre": host_media_pre,
            "host_media_tools_post": host_media_post,
            "full_video_decode": True,
        },
        "slurm": {
            "submission": submission,
            "preflight": preflight_record,
            "run_record": run_record,
            "srun_status": srun_status,
            "gpu_inventory": gpu,
            "saved_spool_script": saved_spool,
            "job_record": slurm_job,
            "exact_commands": commands,
            "ordinary_non_array_job": True,
        },
        "container_image": image,
        "polaris_hub": assets,
        "artifacts": {
            "policy_server_log": server_log,
            "evaluator_log": evaluator_log,
            "metrics": episode_audit["artifacts"]["metrics"],
            "trace": episode_audit["artifacts"]["trace"],
            "trace_validation": _artifact_identity(trace_audit_artifact),
            "runtime": episode_audit["artifacts"]["runtime"],
            "evaluator_close_ready": episode_audit["artifacts"]["close_ready"],
            "episode_sidecar": episode_audit["artifacts"]["sidecar"],
            "rollout_video": {
                **episode_audit["artifacts"]["video"],
                "probe": raw_video_probe,
            },
            "summary_video": {**summary_identity, "probe": summary_probe},
            "summary_sidecar": _artifact_identity(summary_sidecar),
        },
    }
    completion_artifact = publish_immutable_json(
        run_dir / f"canary-completion-{args.job_id}.json", completion
    )
    candidate = {
        "schema_version": 1,
        "profile": REGISTRY_CANDIDATE_PROFILE,
        "canary": True,
        "authoritative_standard_metric": False,
        "visual_adjudication": "required_before_registry_publication",
        "checkpoint_uri": PI05_DROID_CHECKPOINT_URI,
        "checkpoint_step": None,
        "evaluation_id": EVALUATION_ID,
        "benchmark": "PolaRiS DROID task suite",
        "task": PI05_DROID_NATIVE_TASK,
        "policy_mode": "flow",
        "model_profile": "pi05_droid",
        "protocol_variant": PROTOCOL,
        "job_id": args.job_id,
        "status": "complete",
        "scientific_outcome": episode_audit["scientific_outcome"],
        "metrics": {
            "episodes": 1,
            "successes": int(metrics["success"]),
            "progress": metrics["progress"],
            "numerical_failures": int(metrics["numerical_failure"]),
            "queries": episode_audit["execution_cadence"]["queries"],
            "executions": episode_audit["execution_cadence"]["executions"],
            "apply_calls": episode_audit["execution_cadence"]["apply_calls"],
        },
        "completion": _artifact_identity(completion_artifact),
        "summary_video": summary_identity,
        "summary_sidecar": _artifact_identity(summary_sidecar),
    }
    candidate_artifact = publish_immutable_json(
        run_dir / "registry_candidate.json", candidate
    )
    success = {
        "schema_version": 1,
        "profile": CANARY_PROFILE,
        "protocol": PROTOCOL,
        "status": "technical_contract_complete",
        "completion": _artifact_identity(completion_artifact),
        "registry_candidate": _artifact_identity(candidate_artifact),
    }
    publish_immutable_json(run_dir / "eval_success.txt", success)
    return completion


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--polaris-repo", type=Path, required=True)
    parser.add_argument("--expected-polaris-commit", required=True)
    parser.add_argument("--openpi-dir", type=Path, required=True)
    parser.add_argument("--droid-dir", type=Path, required=True)
    parser.add_argument("--position-controller-attestation", type=Path, required=True)
    parser.add_argument(
        "--expected-position-controller-attestation-sha256", required=True
    )
    parser.add_argument("--container-image", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--expected-host-media-tools-manifest-sha256", required=True)
    parser.add_argument("--host-ffprobe-path", type=Path, required=True)
    parser.add_argument("--expected-host-ffprobe-sha256", required=True)
    parser.add_argument("--host-ffmpeg-path", type=Path, required=True)
    parser.add_argument("--expected-host-ffmpeg-sha256", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser("preflight")
    _add_runtime_arguments(preflight_parser)
    preflight_parser.add_argument("--output", type=Path)

    resolved_parser = subparsers.add_parser("write-resolved-contract")
    resolved_parser.add_argument("--output", type=Path, required=True)

    failure_parser = subparsers.add_parser("publish-failure")
    failure_parser.add_argument("--output", type=Path, required=True)
    failure_parser.add_argument("--run-dir", type=Path, required=True)
    failure_parser.add_argument("--job-id", type=int, required=True)
    failure_parser.add_argument("--failure-stage", required=True)
    failure_parser.add_argument("--exit-code", type=int, required=True)

    finalize_parser = subparsers.add_parser("finalize")
    _add_runtime_arguments(finalize_parser)
    finalize_parser.add_argument("--job-id", type=int, required=True)
    finalize_parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        value = preflight(args)
    elif args.command == "write-resolved-contract":
        value = resolved_contract()
        publish_immutable_json(args.output, value)
    elif args.command == "publish-failure":
        value = publish_failure_artifact(
            output=args.output,
            run_dir=args.run_dir,
            job_id=args.job_id,
            failure_stage=args.failure_stage,
            exit_code=args.exit_code,
        )
    else:
        value = finalize(args)
    print(canonical_json_bytes(value).decode("ascii"), end="")


if __name__ == "__main__":
    main()
