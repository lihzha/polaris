#!/usr/bin/env python3
"""Finalize a pinned L40S native joint-velocity controller-only smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


PROFILE = "pi05_droid_native_jointvelocity_l40s_controller_smoke_v1"
SMOKE_PROFILE = "pi05_droid_native_jointvelocity_controller_smoke_v1"
CONTROLLER_PROFILE = "openpi_pi05_droid_native_jointvelocity_v1"
ISAACLAB_SOURCES = {
    "actions_cfg.py": "94722a5c0d6da3639b5507130d1ec2e7f62d7490e4625d4bf30ada8691ef63d4",
    "joint_actions.py": "1b3dcb55d969d886cee500e660f12331e49f8638fec98cde30d2b818a1ca1692",
    "actuator_cfg.py": "3963167d6678c6f052d87202678822d80a98b0b0f3492b859774768b1bd80520",
    "actuator_pd.py": "1d2a9b80812714f5aade3ed7bbb7c74a403ab718868aa73d187eb77173695beb",
    "articulation.py": "9cc03b85642c36c801ff9683e94b8ccc3fbef1178761338974b433dacc78ef75",
}
POLARIS_RUNTIME_SOURCES = {
    "droid_cfg.py": "111c34d8d707f6edf31e9166c9aafd999ff3e7ea72344fc31fe5c9b8d6e175ee"
}
JOINT_NAMES = [f"panda_joint{index}" for index in range(1, 8)]
VELOCITY_LIMITS = [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]
FLOAT32_VELOCITY_LIMITS = [
    2.174999952316284,
    2.174999952316284,
    2.174999952316284,
    2.174999952316284,
    2.609999895095825,
    2.609999895095825,
    2.609999895095825,
]
EFFORT_LIMITS = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]
FLOAT32_PI_OVER_4 = 0.7853981852531433
IMAGE_SHA256 = "ad566a3a0bbb300cafb4a63e0f4c0056f501e4490a136881b0b1ae2d556b324a"
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
    "scripts/smoke_joint_velocity_controller.py",
    "scripts/polaris/finalize_pi05_droid_jointvelocity_controller_smoke.py",
    "scripts/polaris/l40s_pi05_droid_jointvelocity_controller_smoke.sbatch",
    "scripts/polaris/submit_pi05_droid_jointvelocity_controller_smoke.sh",
    "src/polaris/environments/droid_cfg.py",
    "src/polaris/environments/robot_cfg.py",
    "src/polaris/joint_velocity_runtime.py",
    "src/polaris/joint_velocity_smoke.py",
    "src/polaris/pi05_droid_jointvelocity_contract.py",
)


def _canonical_json(value: Any, *, newline: bool) -> bytes:
    rendered = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return rendered + (b"\n" if newline else b"")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _read_canonical_json(path: Path, field: str, *, mode: int = 0o444):
    file_stat = _regular_file(path, field, mode=mode)
    payload = Path(path).read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} is not strict JSON") from error
    if payload != _canonical_json(value, newline=True):
        raise ValueError(f"{field} is not canonical JSON")
    return value, payload, file_stat


def _git(repository: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            f"Git provenance query failed: {' '.join(arguments)}"
        ) from error


def _finite_vector(value: Any, length: int, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field} width mismatch")
    if any(type(item) not in (int, float) or not math.isfinite(item) for item in value):
        raise ValueError(f"{field} must contain finite numbers")
    return [float(item) for item in value]


def _flatten_shape(value: Any, shape: tuple[int, ...], field: str) -> list[float]:
    if not shape:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{field} contains a non-finite number")
        return [float(value)]
    if not isinstance(value, list) or len(value) != shape[0]:
        raise ValueError(f"{field} shape mismatch")
    flattened: list[float] = []
    for child in value:
        flattened.extend(_flatten_shape(child, shape[1:], field))
    return flattened


def _validate_array(
    report: Any, shape: tuple[int, ...], expected: list[float], field: str
) -> None:
    if not isinstance(report, dict) or set(report) != {
        "shape",
        "dtype",
        "device",
        "values",
    }:
        raise ValueError(f"{field} array schema mismatch")
    if report["shape"] != list(shape):
        raise ValueError(f"{field} shape mismatch")
    if report["dtype"] != "torch.float32" or report["device"] != "cuda:0":
        raise ValueError(f"{field} must be a cuda:0 torch.float32 tensor")
    values = _flatten_shape(report["values"], shape, field)
    if values != expected:
        raise ValueError(f"{field} values mismatch")


def _validate_runtime(report: Any) -> dict[str, Any]:
    required = {
        "schema_version",
        "profile",
        "status",
        "isaaclab_version",
        "isaaclab_source_sha256",
        "polaris_runtime_source_sha256",
        "policy_frequency_hz",
        "physics_frequency_hz",
        "decimation",
        "joint_names",
        "action_term_class",
        "action_cfg_class",
        "scale",
        "offset",
        "use_default_offset",
        "clip",
        "position_integration",
        "velocity_drive",
        "gripper",
        "runtime_sha256",
    }
    if not isinstance(report, dict) or set(report) != required:
        raise ValueError("Runtime contract schema mismatch")
    body = dict(report)
    claimed = body.pop("runtime_sha256")
    if not isinstance(claimed, str) or claimed != _sha256_bytes(
        _canonical_json(body, newline=False)
    ):
        raise ValueError("Runtime contract SHA-256 mismatch")
    expected = {
        "schema_version": 1,
        "profile": CONTROLLER_PROFILE,
        "status": "pass",
        "isaaclab_version": "2.3.0",
        "policy_frequency_hz": 15,
        "physics_frequency_hz": 120,
        "decimation": 8,
        "joint_names": JOINT_NAMES,
        "action_term_class": (
            "isaaclab.envs.mdp.actions.joint_actions.JointVelocityAction"
        ),
        "action_cfg_class": (
            "isaaclab.envs.mdp.actions.actions_cfg.JointVelocityActionCfg"
        ),
        "scale": 1.0,
        "offset": 0.0,
        "use_default_offset": False,
        "position_integration": "absent_by_exact_action_class",
    }
    for key, value in expected.items():
        if report[key] != value or type(report[key]) is not type(value):
            raise ValueError(f"Runtime contract {key} mismatch")
    if report["isaaclab_source_sha256"] != ISAACLAB_SOURCES:
        raise ValueError("Runtime Isaac Lab source mismatch")
    if report["polaris_runtime_source_sha256"] != POLARIS_RUNTIME_SOURCES:
        raise ValueError("Runtime PolaRiS source mismatch")
    _validate_array(
        report["clip"],
        (1, 7, 2),
        [-1.0, 1.0] * 7,
        "action clip",
    )
    drive = report["velocity_drive"]
    if not isinstance(drive, dict) or set(drive) != {
        "position_stiffness",
        "velocity_damping",
        "buffered",
        "direct_physx",
    }:
        raise ValueError("Runtime drive schema mismatch")
    if (
        type(drive["position_stiffness"]) is not float
        or drive["position_stiffness"] != 0.0
    ):
        raise ValueError("Runtime stiffness mismatch")
    if (
        type(drive["velocity_damping"]) is not float
        or drive["velocity_damping"] != 80.0
    ):
        raise ValueError("Runtime damping mismatch")
    arrays = {
        "stiffness": ((1, 7), [0.0] * 7),
        "damping": ((1, 7), [80.0] * 7),
        "effort_limit": ((1, 7), EFFORT_LIMITS),
        "velocity_limit": ((1, 7), FLOAT32_VELOCITY_LIMITS),
    }
    for surface in ("buffered", "direct_physx"):
        if not isinstance(drive[surface], dict) or set(drive[surface]) != set(arrays):
            raise ValueError(f"Runtime {surface} schema mismatch")
        for name, (shape, values) in arrays.items():
            _validate_array(drive[surface][name], shape, values, f"{surface} {name}")
    gripper = report["gripper"]
    if not isinstance(gripper, dict) or set(gripper) != {
        "action_class",
        "joint_name",
        "threshold",
        "open_command",
        "closed_command",
    }:
        raise ValueError("Runtime gripper schema mismatch")
    if gripper["action_class"] != (
        "polaris.environments.droid_cfg.BinaryJointPositionZeroToOneAction"
    ):
        raise ValueError("Runtime gripper class mismatch")
    if gripper["joint_name"] != "finger_joint":
        raise ValueError("Runtime gripper joint mismatch")
    if gripper["threshold"] != "closed_if_gt_0p5_else_open":
        raise ValueError("Runtime gripper threshold mismatch")
    _validate_array(gripper["open_command"], (1, 1), [0.0], "gripper open")
    _validate_array(
        gripper["closed_command"],
        (1, 1),
        [FLOAT32_PI_OVER_4],
        "gripper closed",
    )
    return report


def _smoke_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = [
        {"label": "hold", "action": [0.0] * 8, "kind": "hold"}
    ]
    for index, name in enumerate(JOINT_NAMES):
        for sign_name, sign in (("positive", 1), ("negative", -1)):
            action = [0.0] * 8
            action[index] = float(sign) * 0.25
            cases.append(
                {
                    "label": f"{name}_{sign_name}",
                    "action": action,
                    "kind": "signed_joint",
                    "joint_index": index,
                    "sign": sign,
                }
            )
    cases.extend(
        [
            {
                "label": "gripper_open",
                "action": [0.0] * 8,
                "kind": "gripper",
                "expected_finger_target": 0.0,
            },
            {
                "label": "gripper_closed",
                "action": [0.0] * 7 + [1.0],
                "kind": "gripper",
                "expected_finger_target": math.pi / 4.0,
            },
            {
                "label": "gripper_boundary_0p5",
                "action": [0.0] * 7 + [0.5],
                "kind": "gripper",
                "expected_finger_target": 0.0,
                "threshold_boundary": 0.5,
            },
            {
                "label": "positive_action_limit",
                "action": [1.0] * 7 + [0.0],
                "kind": "limit",
            },
            {
                "label": "negative_action_limit",
                "action": [-1.0] * 7 + [0.0],
                "kind": "limit",
            },
        ]
    )
    return cases


def _validate_smoke(payload: Any) -> dict[str, Any]:
    required = {
        "schema_version",
        "smoke_profile",
        "controller_profile",
        "environment",
        "command_magnitude",
        "settle_steps",
        "runtime_contract",
        "cases",
        "reset_probe",
        "lifecycle",
        "completion",
        "status",
        "case_count",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("Smoke result schema mismatch")
    scalars = {
        "schema_version": 1,
        "smoke_profile": SMOKE_PROFILE,
        "controller_profile": CONTROLLER_PROFILE,
        "environment": "DROID-FoodBussing",
        "command_magnitude": 0.25,
        "settle_steps": 5,
        "status": "pass",
        "case_count": 20,
    }
    for key, value in scalars.items():
        if payload[key] != value or type(payload[key]) is not type(value):
            raise ValueError(f"Smoke result {key} mismatch")
    if payload["lifecycle"] != {
        "env_close": "complete",
        "simulation_app_close": "complete",
        "capture_stage": "stdlib_parent_after_kit_child_exit",
    }:
        raise ValueError("Smoke lifecycle is not close-complete")
    completion = payload["completion"]
    if not isinstance(completion, dict) or set(completion) != {
        "child_exit_code",
        "publication_stage",
        "child_capture_sha256",
        "child_capture_size",
        "child_capture_mode",
        "child_capture_path",
    }:
        raise ValueError("Smoke parent completion schema mismatch")
    if (
        completion["child_exit_code"] != 0
        or type(completion["child_exit_code"]) is not int
        or completion["publication_stage"] != "stdlib_parent_after_child_exit"
        or not isinstance(completion["child_capture_sha256"], str)
        or len(completion["child_capture_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in completion["child_capture_sha256"]
        )
        or type(completion["child_capture_size"]) is not int
        or completion["child_capture_size"] <= 0
        or completion["child_capture_mode"] != "0400"
        or not isinstance(completion["child_capture_path"], str)
        or not Path(completion["child_capture_path"]).is_absolute()
    ):
        raise ValueError("Smoke parent completion mismatch")
    _validate_runtime(payload["runtime_contract"])
    expected_cases = _smoke_cases()
    cases = payload["cases"]
    if not isinstance(cases, list) or len(cases) != len(expected_cases):
        raise ValueError("Smoke case set mismatch")
    result_keys = {
        "joint_position_before",
        "joint_velocity_before",
        "joint_position_after",
        "joint_velocity_after",
        "processed_joint_velocity",
        "articulation_joint_velocity_target",
        "soft_joint_position_limits",
        "finger_position_target",
        "processed_finger_position_target",
        "terminated",
        "truncated",
    }
    for index, (case, expected) in enumerate(zip(cases, expected_cases, strict=True)):
        if not isinstance(case, dict) or set(case) != set(expected) | result_keys:
            raise ValueError(f"Smoke case {index} schema mismatch")
        for key, value in expected.items():
            if case[key] != value or (
                key != "action" and type(case[key]) is not type(value)
            ):
                raise ValueError(f"Smoke case {index} {key} mismatch")
        if case["terminated"] is not False or case["truncated"] is not False:
            raise ValueError(f"Smoke case {index} ended")
        action = _finite_vector(case["action"], 8, f"case {index} action")
        before = _finite_vector(case["joint_position_before"], 7, "q before")
        after = _finite_vector(case["joint_position_after"], 7, "q after")
        _finite_vector(case["joint_velocity_before"], 7, "dq before")
        measured = _finite_vector(case["joint_velocity_after"], 7, "dq after")
        processed = _finite_vector(case["processed_joint_velocity"], 7, "processed")
        target = _finite_vector(case["articulation_joint_velocity_target"], 7, "target")
        if processed != action[:7] or target != action[:7]:
            raise ValueError(f"Smoke case {index} command path mismatch")
        if any(
            abs(value) > limit + 1e-4
            for value, limit in zip(measured, VELOCITY_LIMITS, strict=True)
        ):
            raise ValueError(f"Smoke case {index} velocity limit exceeded")
        limits = case["soft_joint_position_limits"]
        if not isinstance(limits, list) or len(limits) != 7:
            raise ValueError(f"Smoke case {index} soft-limit shape mismatch")
        for value, pair in zip(after, limits, strict=True):
            pair = _finite_vector(pair, 2, "soft limit")
            if pair[0] > pair[1] or value < pair[0] - 1e-5 or value > pair[1] + 1e-5:
                raise ValueError(f"Smoke case {index} position limit exceeded")
        if expected["kind"] == "signed_joint":
            joint = expected["joint_index"]
            sign = expected["sign"]
            if sign * (after[joint] - before[joint]) <= 1e-7:
                raise ValueError(f"Smoke case {index} direction mismatch")
            if sign * measured[joint] <= 1e-6:
                raise ValueError(f"Smoke case {index} velocity sign mismatch")
        expected_finger = FLOAT32_PI_OVER_4 if action[7] > 0.5 else 0.0
        for field in ("processed_finger_position_target", "finger_position_target"):
            value = case[field]
            if type(value) not in (int, float) or not math.isclose(
                value, expected_finger, rel_tol=0.0, abs_tol=1e-6
            ):
                raise ValueError(f"Smoke case {index} {field} mismatch")
    reset = payload["reset_probe"]
    if not isinstance(reset, dict) or set(reset) != {
        "default_joint_position",
        "joint_position",
        "joint_velocity",
        "joint_velocity_target",
    }:
        raise ValueError("Smoke reset schema mismatch")
    default = _finite_vector(reset["default_joint_position"], 7, "default q")
    reset_q = _finite_vector(reset["joint_position"], 7, "reset q")
    reset_dq = _finite_vector(reset["joint_velocity"], 7, "reset dq")
    reset_target = _finite_vector(reset["joint_velocity_target"], 7, "reset target")
    if any(
        abs(actual - expected) > 2e-3
        for actual, expected in zip(reset_q, default, strict=True)
    ):
        raise ValueError("Smoke reset position mismatch")
    if max(map(abs, reset_dq)) > 2e-2 or reset_target != [0.0] * 7:
        raise ValueError("Smoke reset velocity mismatch")
    return payload


def _source_provenance(repository: Path, expected_commit: str) -> dict[str, Any]:
    repository = Path(repository)
    if repository.is_symlink():
        raise ValueError("PolaRiS repository path must not be a symlink")
    repository = repository.resolve()
    top_level = Path(
        _git(repository, "rev-parse", "--show-toplevel").decode().strip()
    ).resolve()
    if top_level != repository:
        raise ValueError("PolaRiS repository path is not the exact Git root")
    head = _git(repository, "rev-parse", "HEAD").decode().strip()
    if head != expected_commit:
        raise ValueError("PolaRiS commit mismatch")
    if _git(repository, "status", "--porcelain=v1", "--untracked-files=all").strip():
        raise ValueError("PolaRiS repository is not completely clean")
    files = {}
    for relative_path in SOURCE_PATHS:
        path = repository / relative_path
        _regular_file(path, f"source {relative_path}")
        working = path.read_bytes()
        committed = _git(repository, "show", f"HEAD:{relative_path}")
        if working != committed:
            raise ValueError(f"Source differs from commit: {relative_path}")
        files[relative_path] = {
            "size": len(working),
            "sha256": _sha256_bytes(working),
        }
    return {
        "repository": str(repository),
        "commit": head,
        "tracked_and_untracked_clean": True,
        "files": files,
    }


def _asset_provenance(data_dir: Path) -> dict[str, Any]:
    data_dir = Path(data_dir)
    if data_dir.is_symlink() or not data_dir.is_dir():
        raise ValueError("PolaRiS-Hub data root must be a regular directory")
    data_dir = data_dir.resolve()
    assets = {}
    for relative_path, expected in ASSETS.items():
        path = data_dir / relative_path
        file_stat = _regular_file(path, f"asset {relative_path}")
        digest = _file_sha256(path)
        if digest != expected["sha256"]:
            raise ValueError(f"Asset digest mismatch: {relative_path}")
        metadata_path = (
            data_dir / ".cache/huggingface/download" / (relative_path + ".metadata")
        )
        metadata_stat = _regular_file(metadata_path, f"metadata {relative_path}")
        metadata_payload = metadata_path.read_bytes()
        if _sha256_bytes(metadata_payload) != expected["metadata_sha256"]:
            raise ValueError(f"Asset metadata digest mismatch: {relative_path}")
        first_line = metadata_payload.decode("utf-8").splitlines()[0]
        if first_line != HUB_REVISION:
            raise ValueError(f"Asset Hub revision mismatch: {relative_path}")
        assets[relative_path] = {
            "path": str(path),
            "size": file_stat.st_size,
            "sha256": digest,
            "metadata_path": str(metadata_path),
            "metadata_size": metadata_stat.st_size,
            "metadata_sha256": expected["metadata_sha256"],
            "hub_revision": HUB_REVISION,
        }
    return {"root": str(data_dir), "hub_revision": HUB_REVISION, "assets": assets}


def _build_attestation(args: argparse.Namespace) -> dict[str, Any]:
    if not (
        len(args.expected_polaris_commit) == 40
        and all(
            character in "0123456789abcdef"
            for character in args.expected_polaris_commit
        )
    ):
        raise ValueError("Expected PolaRiS commit is malformed")
    env_job_id = os.environ.get("SLURM_JOB_ID")
    if env_job_id != str(args.job_id) or args.job_id <= 0:
        raise ValueError("SLURM_JOB_ID mismatch")
    if args.smoke_artifact.name != f"joint-velocity-smoke-{args.job_id}.json":
        raise ValueError("Smoke artifact filename does not bind job ID")
    if args.completion.name != f"controller-smoke-{args.job_id}.completion.json":
        raise ValueError("Completion filename does not bind job ID")
    smoke, smoke_bytes, smoke_stat = _read_canonical_json(
        args.smoke_artifact, "smoke artifact"
    )
    _validate_smoke(smoke)
    expected_child_path = args.smoke_artifact.with_name(
        args.smoke_artifact.name + ".child-close.json"
    ).resolve()
    completion_evidence = smoke["completion"]
    if Path(completion_evidence["child_capture_path"]).resolve() != expected_child_path:
        raise ValueError("Smoke child-capture path mismatch")
    child, child_bytes, child_stat = _read_canonical_json(
        expected_child_path, "child close capture", mode=0o400
    )
    expected_child = dict(smoke)
    expected_child.pop("completion")
    expected_child["status"] = "close_validated_pending_parent"
    expected_child["lifecycle"] = {
        "env_close": "complete",
        "simulation_app_close": "pending_child_exit",
        "capture_stage": "kit_child_after_env_close_before_simulation_app_close",
    }
    if child != expected_child:
        raise ValueError("Child close capture differs from promoted smoke evidence")
    if (
        completion_evidence["child_capture_sha256"] != _sha256_bytes(child_bytes)
        or completion_evidence["child_capture_size"] != len(child_bytes)
        or completion_evidence["child_capture_mode"] != "0400"
    ):
        raise ValueError("Smoke completion does not bind exact child capture")

    status, status_bytes, _ = _read_canonical_json(args.srun_status, "srun status")
    if (
        status != {"job_id": args.job_id, "srun_exit_code": 0}
        or type(status.get("job_id")) is not int
    ):
        raise ValueError("srun status mismatch")

    gpu_stat = _regular_file(args.gpu_inventory, "GPU inventory", mode=0o444)
    gpu_bytes = args.gpu_inventory.read_bytes()
    lines = gpu_bytes.decode("utf-8").splitlines()
    if len(lines) != 1:
        raise ValueError("Exactly one allocated GPU is required")
    fields = [field.strip() for field in lines[0].split(",")]
    if (
        len(fields) != 3
        or fields[1] != "NVIDIA L40S"
        or not fields[0].startswith("GPU-")
    ):
        raise ValueError("Allocated GPU is not exactly one NVIDIA L40S")

    source = _source_provenance(args.polaris_repo, args.expected_polaris_commit)
    job_script = source["files"][
        "scripts/polaris/l40s_pi05_droid_jointvelocity_controller_smoke.sbatch"
    ]
    saved_stat = _regular_file(args.saved_job_script, "saved Slurm script", mode=0o444)
    saved_sha = _file_sha256(args.saved_job_script)
    if saved_sha != job_script["sha256"]:
        raise ValueError("Saved Slurm script differs from committed launch script")

    image_stat = _regular_file(args.container_image, "container image")
    image_sha = _file_sha256(args.container_image)
    if image_sha != IMAGE_SHA256:
        raise ValueError("Container image SHA-256 mismatch")
    data = _asset_provenance(args.data_dir)

    return {
        "schema_version": 1,
        "profile": PROFILE,
        "status": "pass",
        "scope": "controller_only_no_model_or_checkpoint",
        "promotion": "forbidden_without_separate_checkpoint_canary",
        "slurm": {
            "job_id": args.job_id,
            "srun_exit_code": 0,
            "status_artifact": {
                "path": str(args.srun_status.resolve()),
                "size": len(status_bytes),
                "sha256": _sha256_bytes(status_bytes),
                "mode": "0444",
            },
            "gpu_inventory": {
                "path": str(args.gpu_inventory.resolve()),
                "size": gpu_stat.st_size,
                "sha256": _sha256_bytes(gpu_bytes),
                "mode": "0444",
                "uuid": fields[0],
                "name": fields[1],
                "driver_version": fields[2],
            },
            "saved_job_script": {
                "path": str(args.saved_job_script.resolve()),
                "size": saved_stat.st_size,
                "sha256": saved_sha,
                "mode": "0444",
            },
        },
        "source": source,
        "runtime": {
            "container_image": {
                "path": str(args.container_image.resolve()),
                "size": image_stat.st_size,
                "sha256": image_sha,
            },
            "polaris_hub": data,
        },
        "smoke_artifact": {
            "path": str(args.smoke_artifact.resolve()),
            "size": smoke_stat.st_size,
            "sha256": _sha256_bytes(smoke_bytes),
            "mode": "0444",
            "nlink": 1,
            "status": "pass",
            "case_count": 20,
            "runtime_sha256": smoke["runtime_contract"]["runtime_sha256"],
            "child_close_capture": {
                "path": str(expected_child_path),
                "size": child_stat.st_size,
                "sha256": _sha256_bytes(child_bytes),
                "mode": "0400",
                "status": "close_validated_pending_parent",
            },
        },
        "runtime_contract": smoke["runtime_contract"],
    }


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(value, newline=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("finalize", "verify"))
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--smoke-artifact", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--srun-status", type=Path, required=True)
    parser.add_argument("--gpu-inventory", type=Path, required=True)
    parser.add_argument("--saved-job-script", type=Path, required=True)
    parser.add_argument("--polaris-repo", type=Path, required=True)
    parser.add_argument("--expected-polaris-commit", required=True)
    parser.add_argument("--container-image", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    expected = _build_attestation(args)
    if args.mode == "finalize":
        if args.completion.exists() or args.completion.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {args.completion}")
        _publish(args.completion, expected)
    actual, actual_bytes, _ = _read_canonical_json(args.completion, "completion")
    if actual != expected:
        raise ValueError("Completion attestation does not match live provenance")
    print(
        json.dumps(
            {
                "completion": str(args.completion.resolve()),
                "sha256": _sha256_bytes(actual_bytes),
                "status": "pass",
                "scope": "controller_only_no_model_or_checkpoint",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"FINALIZATION_ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
