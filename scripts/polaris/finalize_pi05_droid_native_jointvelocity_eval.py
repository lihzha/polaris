#!/usr/bin/env python3
"""Finalize one official pi0.5-DROID native joint-velocity PolaRiS canary."""

from __future__ import annotations

import argparse
import copy
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
    PI05_DROID_BASE_CONTROLLER_COMPLETION_PATH,
    PI05_DROID_BASE_CONTROLLER_COMPLETION_SHA256,
    PI05_DROID_BASE_CONTROLLER_COMPLETION_SIZE,
    PI05_DROID_BASE_CONTROLLER_RUNTIME_SHA256,
    PI05_DROID_BASE_CONTROLLER_SOURCE_COMMIT,
    PI05_DROID_CANARY_ASSETS,
    PI05_DROID_CONTROLLER_CRITICAL_PATHS,
    PI05_DROID_CONTROLLER_JOB_ID,
    PI05_DROID_GRIPPER_CONTROLLER_COMPLETION_PATH,
    PI05_DROID_GRIPPER_CONTROLLER_COMPLETION_SHA256,
    PI05_DROID_GRIPPER_CONTROLLER_COMPLETION_SIZE,
    PI05_DROID_GRIPPER_CONTROLLER_JOB_ID,
    PI05_DROID_GRIPPER_CONTROLLER_PROFILE,
    PI05_DROID_GRIPPER_CONTROLLER_SOURCE_COMMIT,
    PI05_DROID_GRIPPER_DRIVE_PROFILE,
    PI05_DROID_GRIPPER_RUNTIME_SHA256,
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
    validate_environment_runtime_contract,
    validate_immutable_json,
    validate_native_model_eval_contract,
    validate_terminal_rollout_evidence,
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
    checkout: job 1098204 is the descendant gate that reattests those bytes
    after the gripper-cap fix.  Both immutable completions remain mandatory.
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
        "descendant_source_authority": "required_job1098204_gate",
    }


def _runtime_sha256(report: dict[str, Any]) -> str:
    payload = copy.deepcopy(report)
    payload.pop("runtime_sha256", None)
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _validate_scalar_tensor(
    value: Any, *, field: str, device: str, expected: float
) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != {"shape", "dtype", "device", "values"}
        or value["shape"] != [1, 1]
        or value["dtype"] != "torch.float32"
        or value["device"] != device
        or value["values"] != [[expected]]
    ):
        raise ValueError(f"Gripper-cap {field} tensor mismatch")


def _validate_gripper_runtime(report: Any) -> dict[str, Any]:
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != 1
        or report.get("profile") != "openpi_pi05_droid_native_jointvelocity_v1"
        or report.get("status") != "pass"
        or report.get("policy_frequency_hz") != 15
        or report.get("physics_frequency_hz") != 120
        or report.get("decimation") != 8
        or report.get("runtime_sha256") != PI05_DROID_GRIPPER_RUNTIME_SHA256
        or _runtime_sha256(report) != PI05_DROID_GRIPPER_RUNTIME_SHA256
    ):
        raise ValueError("Gripper-cap runtime identity mismatch")
    gripper = report.get("gripper")
    if not isinstance(gripper, dict):
        raise ValueError("Gripper-cap runtime lacks gripper evidence")
    drive = gripper.get("drive")
    if not isinstance(drive, dict) or set(drive) != {
        "profile",
        "configured",
        "actuator",
        "direct_physx",
    }:
        raise ValueError("Gripper-cap drive schema mismatch")
    expected_configured = {
        "joint_names_expr": ["finger_joint"],
        "stiffness": None,
        "damping": None,
        "effort_limit": 200.0,
        "effort_limit_sim": 200.0,
        "velocity_limit": 5.0,
        "velocity_limit_sim": 5.0,
    }
    if (
        drive["profile"] != PI05_DROID_GRIPPER_DRIVE_PROFILE
        or drive["configured"] != expected_configured
    ):
        raise ValueError("Gripper-cap configured drive mismatch")
    actuator_expected = {
        "stiffness": 5729.578125,
        "damping": 0.011459155939519405,
        "effort_limit": 200.0,
        "effort_limit_sim": 200.0,
        "velocity_limit": 5.0,
        "velocity_limit_sim": 5.0,
    }
    direct_expected = {
        "stiffness": 5729.578125,
        "damping": 0.011459155939519405,
        "effort_limit": 200.0,
        "velocity_limit": 5.0,
    }
    if not isinstance(drive["actuator"], dict) or set(drive["actuator"]) != set(
        actuator_expected
    ):
        raise ValueError("Gripper-cap actuator drive schema mismatch")
    if not isinstance(drive["direct_physx"], dict) or set(drive["direct_physx"]) != set(
        direct_expected
    ):
        raise ValueError("Gripper-cap direct PhysX schema mismatch")
    for field, expected in actuator_expected.items():
        _validate_scalar_tensor(
            drive["actuator"][field],
            field=f"CUDA actuator {field}",
            device="cuda:0",
            expected=expected,
        )
    for field, expected in direct_expected.items():
        _validate_scalar_tensor(
            drive["direct_physx"][field],
            field=f"CPU direct PhysX {field}",
            device="cpu",
            expected=expected,
        )
    return report


def _validate_gripper_smoke(
    smoke_record: Any, runtime: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(smoke_record, dict):
        raise ValueError("Gripper-cap completion lacks smoke artifact")
    smoke_path = Path(smoke_record.get("path", ""))
    smoke = validate_immutable_json(smoke_path)
    if (
        smoke_record.get("status") != "pass"
        or smoke_record.get("mode") != "0444"
        or smoke_record.get("nlink") != 1
        or smoke_record.get("sha256") != smoke["sha256"]
        or smoke_record.get("size") != smoke["size"]
        or smoke_record.get("runtime_sha256") != PI05_DROID_GRIPPER_RUNTIME_SHA256
        or Path(smoke_record.get("path", "")).resolve() != Path(smoke["path"])
    ):
        raise ValueError("Gripper-cap smoke identity mismatch")
    value = smoke["value"]
    required = {
        "schema_version",
        "smoke_profile",
        "controller_profile",
        "environment",
        "command_magnitude",
        "settle_steps",
        "expected_gripper_drive_profile",
        "gripper_precondition_steps",
        "runtime_contract",
        "cases",
        "case_count",
        "reset_probe",
        "lifecycle",
        "status",
        "completion",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value["schema_version"] != 1
        or value["smoke_profile"]
        != "pi05_droid_native_jointvelocity_controller_smoke_v2"
        or value["controller_profile"] != "openpi_pi05_droid_native_jointvelocity_v1"
        or value["environment"] != PI05_DROID_NATIVE_TASK
        or value["command_magnitude"] != 0.25
        or value["settle_steps"] != 5
        or value["expected_gripper_drive_profile"] != PI05_DROID_GRIPPER_DRIVE_PROFILE
        or value["gripper_precondition_steps"] != 5
        or value["runtime_contract"] != runtime
        or value["case_count"] != 20
        or value["status"] != "pass"
        or value["lifecycle"]
        != {
            "env_close": "complete",
            "simulation_app_close": "invoked_then_child_exited_zero",
            "capture_stage": "stdlib_parent_after_kit_child_exit",
        }
    ):
        raise ValueError("Gripper-cap smoke schema or identity mismatch")
    cases = value["cases"]
    expected_labels = {
        "hold",
        *(
            f"panda_joint{joint}_{sign}"
            for joint in range(1, 8)
            for sign in ("positive", "negative")
        ),
        "gripper_open",
        "gripper_closed",
        "gripper_boundary_0p5",
        "positive_action_limit",
        "negative_action_limit",
    }
    if (
        not isinstance(cases, list)
        or len(cases) != 20
        or {case.get("label") for case in cases if isinstance(case, dict)}
        != expected_labels
    ):
        raise ValueError("Gripper-cap smoke case set mismatch")
    gripper_cases = {
        case["label"]: case
        for case in cases
        if isinstance(case, dict) and case.get("kind") == "gripper"
    }
    expected_gripper_cases = {
        "gripper_open": {
            "command": 0.0,
            "slew": -4.978439211845398,
            "target": 0.0,
            "processed_target": 0.0,
            "sign": -1,
        },
        "gripper_closed": {
            "command": 1.0,
            "slew": 4.971333146095276,
            "target": 0.7853981633974483,
            "processed_target": 0.7853981852531433,
            "sign": 1,
        },
        "gripper_boundary_0p5": {
            "command": 0.5,
            "slew": -4.978439211845398,
            "target": 0.0,
            "processed_target": 0.0,
            "sign": -1,
        },
    }
    if set(gripper_cases) != set(expected_gripper_cases):
        raise ValueError("Gripper-cap measured case set mismatch")
    measured_slew = {}
    for label, expected in expected_gripper_cases.items():
        case = gripper_cases[label]
        action = case.get("action")
        q_before = case.get("joint_velocity_before")
        q_after = case.get("joint_velocity_after")
        slew = case.get("finger_average_slew_rad_s")
        if (
            not isinstance(action, list)
            or action != [0.0] * 7 + [expected["command"]]
            or case.get("processed_joint_velocity") != [0.0] * 7
            or case.get("articulation_joint_velocity_target") != [0.0] * 7
            or case.get("expected_finger_target") != expected["target"]
            or case.get("processed_finger_position_target")
            != expected["processed_target"]
            or case.get("expected_motion_sign") != expected["sign"]
            or type(slew) is not float
            or slew != expected["slew"]
            or abs(slew) > 5.0
            or type(case.get("finger_velocity_after")) is not float
            or abs(case["finger_velocity_after"]) > 5.0
            or not isinstance(q_before, list)
            or not isinstance(q_after, list)
            or len(q_before) != 7
            or len(q_after) != 7
            or any(type(item) not in (int, float) for item in q_before + q_after)
            or max(abs(float(item)) for item in q_before + q_after) > 0.001
            or expected["sign"]
            * (case.get("finger_position_after") - case.get("finger_position_before"))
            <= 0.0
            or case.get("terminated") is not False
            or case.get("truncated") is not False
        ):
            raise ValueError(f"Gripper-cap measured slew mismatch: {label}")
        measured_slew[label] = abs(slew)

    default_q = [
        0.0,
        -0.6283185482025146,
        0.0,
        -2.5132741928100586,
        0.0,
        1.884955644607544,
        0.0,
    ]
    if value["reset_probe"] != {
        "default_joint_position": default_q,
        "joint_position": default_q,
        "joint_velocity": [0.0] * 7,
        "joint_velocity_target": [0.0] * 7,
        "default_finger_position": 0.0,
        "finger_position": 0.0,
        "finger_velocity": 0.0,
        "finger_position_target": 0.0,
    }:
        raise ValueError("Gripper-cap exact reset mismatch")

    completion = value["completion"]
    if (
        not isinstance(completion, dict)
        or set(completion)
        != {
            "child_exit_code",
            "publication_stage",
            "child_capture_sha256",
            "child_capture_size",
            "child_capture_mode",
            "child_capture_path",
            "child_ready_marker_sha256",
            "child_ready_marker_size",
            "child_ready_marker_mode",
            "child_ready_marker_path",
        }
        or completion["child_exit_code"] != 0
        or completion["publication_stage"] != "stdlib_parent_after_child_exit"
        or completion["child_capture_mode"] != "0444"
        or completion["child_ready_marker_mode"] != "0444"
    ):
        raise ValueError("Gripper-cap parent lifecycle mismatch")
    child = validate_immutable_json(Path(completion["child_capture_path"]))
    ready = validate_immutable_json(Path(completion["child_ready_marker_path"]))
    if (
        child["sha256"] != completion["child_capture_sha256"]
        or child["size"] != completion["child_capture_size"]
        or ready["sha256"] != completion["child_ready_marker_sha256"]
        or ready["size"] != completion["child_ready_marker_size"]
    ):
        raise ValueError("Gripper-cap child lifecycle identity mismatch")
    expected_child = copy.deepcopy(value)
    expected_child.pop("completion")
    expected_child["status"] = "close_validated_pending_parent"
    expected_child["lifecycle"] = {
        "env_close": "complete",
        "simulation_app_close": "pending_child_exit",
        "capture_stage": ("kit_child_after_env_close_before_simulation_app_close"),
    }
    expected_ready = {
        "schema_version": 1,
        "status": "success",
        "stage": "kit_child_after_env_close_before_simulation_app_close",
        "raw_result": {
            "path": completion["child_capture_path"],
            "size_bytes": child["size"],
            "sha256": child["sha256"],
            "mode": "0444",
        },
    }
    if child["value"] != expected_child or ready["value"] != expected_ready:
        raise ValueError("Gripper-cap child-close/ready lifecycle mismatch")
    return {
        "artifact": {
            key: smoke[key] for key in ("path", "size", "sha256", "mode", "nlink")
        },
        "child_close": {
            key: child[key] for key in ("path", "size", "sha256", "mode", "nlink")
        },
        "child_ready": {
            key: ready[key] for key in ("path", "size", "sha256", "mode", "nlink")
        },
        "measured_average_slew_magnitude_rad_s": measured_slew,
        "measured_arm_velocity_abs_max_limit_rad_s": 0.001,
        "exact_reset": True,
    }


def validate_gripper_cap_controller_completion(
    completion_path: Path,
    expected_sha256: str,
    expected_profile: str,
    repository: Path,
) -> dict[str, Any]:
    """Reverify the exact terminal job1098204 cap completion and smoke."""

    expected_sha256 = _sha256_string(expected_sha256, "gripper-cap completion SHA-256")
    if expected_sha256 != PI05_DROID_GRIPPER_CONTROLLER_COMPLETION_SHA256:
        raise ValueError("Unexpected gripper-cap completion SHA-256")
    if expected_profile != PI05_DROID_GRIPPER_DRIVE_PROFILE:
        raise ValueError("Unexpected gripper-cap drive profile")
    if str(Path(completion_path)) != PI05_DROID_GRIPPER_CONTROLLER_COMPLETION_PATH:
        raise ValueError("Unexpected gripper-cap completion path")
    artifact = validate_immutable_json(completion_path)
    if (
        artifact["sha256"] != expected_sha256
        or artifact["size"] != PI05_DROID_GRIPPER_CONTROLLER_COMPLETION_SIZE
    ):
        raise ValueError("Gripper-cap completion identity mismatch")
    value = artifact["value"]
    required = {
        "schema_version",
        "profile",
        "status",
        "scope",
        "promotion",
        "candidate_intent",
        "source",
        "runtime",
        "runtime_contract",
        "slurm",
        "smoke_artifact",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value["schema_version"] != 1
        or value["profile"] != PI05_DROID_GRIPPER_CONTROLLER_PROFILE
        or value["status"] != "pass"
        or value["scope"] != "controller_only_no_model_or_checkpoint"
        or value["promotion"] != "forbidden_without_separate_checkpoint_canary"
        or value["candidate_intent"]
        != {"expected_gripper_drive_profile": PI05_DROID_GRIPPER_DRIVE_PROFILE}
        or value["slurm"].get("job_id") != PI05_DROID_GRIPPER_CONTROLLER_JOB_ID
        or value["slurm"].get("srun_exit_code") != 0
        or value["runtime"].get("container_image", {}).get("sha256")
        != PI05_DROID_PYXIS_SHA256
    ):
        raise ValueError("Gripper-cap completion schema or identity mismatch")
    source = value["source"]
    if (
        not isinstance(source, dict)
        or source.get("commit") != PI05_DROID_GRIPPER_CONTROLLER_SOURCE_COMMIT
        or source.get("detached_head") is not True
        or source.get("tracked_and_untracked_clean") is not True
        or source.get("head_reference") != "HEAD"
        or source.get("standalone_git_directory") is not True
        or not isinstance(source.get("files"), dict)
    ):
        raise ValueError("Gripper-cap source provenance mismatch")
    repository = Path(repository).resolve()
    source_files = {}
    for relative_path in PI05_DROID_CONTROLLER_CRITICAL_PATHS:
        record = source["files"].get(relative_path)
        path = repository / relative_path
        if (
            not isinstance(record, dict)
            or set(record) != {"size", "sha256"}
            or path.stat().st_size != record["size"]
            or file_sha256(path) != record["sha256"]
        ):
            raise ValueError(
                f"Integrated source differs from job1098204: {relative_path}"
            )
        source_files[relative_path] = record
    runtime = _validate_gripper_runtime(value["runtime_contract"])
    smoke = _validate_gripper_smoke(value["smoke_artifact"], runtime)
    return {
        **{key: artifact[key] for key in ("path", "size", "sha256", "mode", "nlink")},
        "job_id": PI05_DROID_GRIPPER_CONTROLLER_JOB_ID,
        "controller_profile": PI05_DROID_GRIPPER_CONTROLLER_PROFILE,
        "drive_profile": PI05_DROID_GRIPPER_DRIVE_PROFILE,
        "runtime_sha256": PI05_DROID_GRIPPER_RUNTIME_SHA256,
        "source_commit": PI05_DROID_GRIPPER_CONTROLLER_SOURCE_COMMIT,
        "critical_source_files": source_files,
        "smoke": smoke,
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
        "terminal_rollout",
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
    terminal_rollout = validate_terminal_rollout_evidence(
        value.get("terminal_rollout"), runtime["environment_runtime_contract"]
    )
    if (
        value["schema_version"] != 1
        or value["profile"] != PI05_DROID_NATIVE_CANARY_PROFILE
        or value["status"] != "simulation_app_close_pending"
        or value["environment"] != PI05_DROID_NATIVE_TASK
        or value["rollouts"] != 1
        or value["episode_steps"] != PI05_DROID_NATIVE_EPISODE_STEPS
        or value["env_close"] != "complete"
        or value["environment_runtime_contract_sha256"]
        != runtime["environment_runtime_contract"]["sha256"]
        or value["runtime_artifact"]
        != {key: runtime[key] for key in ("path", "size", "sha256", "mode", "nlink")}
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
        "terminal_rollout": terminal_rollout,
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


def probe_video(path: Path, *, require_faststart: bool) -> dict[str, Any]:
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
        or frame_count != PI05_DROID_NATIVE_EPISODE_STEPS
        or not math.isclose(
            duration,
            PI05_DROID_NATIVE_EPISODE_STEPS / PI05_DROID_NATIVE_POLICY_HZ,
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


def create_summary_video(source: Path, output: Path) -> dict[str, Any]:
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing existing summary video: {output}")
    temporary = output.with_name(f".{output.stem}.partial-{os.getpid()}.mp4")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"Refusing existing summary temporary: {temporary}")
    try:
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
                "-r",
                str(PI05_DROID_NATIVE_POLICY_HZ),
                str(temporary),
            ],
            check=True,
        )
        probe = probe_video(temporary, require_faststart=True)
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
        "GRIPPER_CAP_CONTROLLER_COMPLETION",
        "EXPECTED_GRIPPER_CAP_COMPLETION_SHA256",
        "EXPECTED_GRIPPER_CAP_PROFILE",
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
        "GRIPPER_CAP_CONTROLLER_COMPLETION": Path(args.gripper_cap_completion),
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
        "EXPECTED_GRIPPER_CAP_COMPLETION_SHA256": (
            args.expected_gripper_cap_completion_sha256
        ),
        "EXPECTED_GRIPPER_CAP_PROFILE": args.expected_gripper_cap_profile,
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
    gripper_controller = validate_gripper_cap_controller_completion(
        args.gripper_cap_completion,
        args.expected_gripper_cap_completion_sha256,
        args.expected_gripper_cap_profile,
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
    if runtime["runtime_sha256"] != gripper_controller["runtime_sha256"]:
        raise ValueError(
            "Canary runtime differs from the job1098204 controller-attested runtime"
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
    if trace_summary["terminal_rollout"] != close_ready["terminal_rollout"]:
        raise ValueError("Trace terminal rollout differs from evaluator close evidence")
    trace_summary_path = task_dir / "trace_validation.json"
    trace_summary_artifact = publish_immutable_json(trace_summary_path, trace_summary)
    raw_video_probe = probe_video(video_path, require_faststart=False)
    trace_identity = _seal_file(trace_path, "policy trace")
    metrics_identity = _seal_file(metrics_path, "metrics CSV")
    video_identity = _seal_file(video_path, "rollout video")
    if trace_identity["sha256"] != trace_summary["trace_sha256"]:
        raise ValueError("Sealed trace digest differs from trace audit")

    summary_path = run_dir / "pi05_droid_native_jointvelocity_canary.mp4"
    summary_probe = create_summary_video(video_path, summary_path)
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
            "native_gripper_cap": gripper_controller,
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
    parser.add_argument("--gripper-cap-completion", type=Path, required=True)
    parser.add_argument("--expected-gripper-cap-completion-sha256", required=True)
    parser.add_argument("--expected-gripper-cap-profile", required=True)


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
        gripper = validate_gripper_cap_controller_completion(
            args.gripper_cap_completion,
            args.expected_gripper_cap_completion_sha256,
            args.expected_gripper_cap_profile,
            args.polaris_repo,
        )
        print(
            canonical_json_bytes({"base": base, "gripper_cap": gripper}).decode(
                "ascii"
            ),
            end="",
        )
        return
    completion = finalize(args)
    print(canonical_json_bytes(completion).decode("ascii"), end="")


if __name__ == "__main__":
    main()
