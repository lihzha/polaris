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


PROFILE = "pi05_droid_native_jointvelocity_l40s_controller_smoke_v2"
SMOKE_PROFILE = "pi05_droid_native_jointvelocity_controller_smoke_v2"
CONTROLLER_PROFILE = "openpi_pi05_droid_native_jointvelocity_v1"
GRIPPER_DRIVE_PROFILE = (
    "implicit_gripper_physx_velocity_limit5_followers5_every_reset_"
    "cuda_actuator_cpu_static_physx_v1"
)
GRIPPER_VELOCITY_LIMIT = 5.0
GRIPPER_EFFORT_LIMIT = 200.0
GRIPPER_STIFFNESS = 5729.578125
GRIPPER_DAMPING = 0.011459155939519405
GRIPPER_PRECONDITION_STEPS = 5
GRIPPER_PRECONDITION_POSITION_TOLERANCE = 0.02
GRIPPER_MEASURED_VELOCITY_TOLERANCE = 0.001
ISAACLAB_SOURCES = {
    "actions_cfg.py": "94722a5c0d6da3639b5507130d1ec2e7f62d7490e4625d4bf30ada8691ef63d4",
    "joint_actions.py": "1b3dcb55d969d886cee500e660f12331e49f8638fec98cde30d2b818a1ca1692",
    "binary_joint_actions.py": "84bf343dc4a609d2327f1ee8b965439f49f3167f9a45f652e9aa6b652c9c0630",
    "actuator_cfg.py": "3963167d6678c6f052d87202678822d80a98b0b0f3492b859774768b1bd80520",
    "actuator_pd.py": "1d2a9b80812714f5aade3ed7bbb7c74a403ab718868aa73d187eb77173695beb",
    "articulation.py": "9cc03b85642c36c801ff9683e94b8ccc3fbef1178761338974b433dacc78ef75",
    "action_manager.py": "a95d7c45048b08d8c7b526ba5ab88d40087b4bcbacb248d6e9b039c5f3b9afb9",
    "event_manager.py": "21864ba786aa30b842023809adacf1548cc75a4e46443a1fc7266d908ddc9090",
    "manager_based_env.py": "a7d694e20e190678410330e24c365381a80d4f538debcced696d2c8e05cb940e",
    "manager_based_rl_env.py": "8ec2759541c8320ed725411d49564f5617b3ab6b9d340c860bededee13d557d4",
}
POLARIS_RUNTIME_SOURCES = {
    "droid_cfg.py": "19ceaceeb06c06e1f708af2380d47dded2a5f7030c436e266c4efca255509287",
    "robot_cfg.py": "26874c2a7807cc69028e538d6428853b75969edc20c05637385303e838144aa7",
    "native_gripper_runtime.py": (
        "8b5cd2e7c0f912418a878544fce621c00a4126551a21ab72d486686b67d70ff1"
    ),
    "manager_based_rl_splat_environment.py": (
        "9381b2704e86aae6447eb2cc229471612104c5eec2acc9d913252a09159da426"
    ),
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
    "scripts/eval.py",
    "scripts/smoke_joint_velocity_controller.py",
    "scripts/polaris/finalize_pi05_droid_jointvelocity_controller_smoke.py",
    "scripts/polaris/l40s_pi05_droid_jointvelocity_controller_smoke.sbatch",
    "scripts/polaris/submit_pi05_droid_jointvelocity_controller_smoke.sh",
    "src/polaris/environments/droid_cfg.py",
    "src/polaris/environments/manager_based_rl_splat_environment.py",
    "src/polaris/environments/robot_cfg.py",
    "src/polaris/config.py",
    "src/polaris/joint_velocity_runtime.py",
    "src/polaris/joint_velocity_smoke.py",
    "src/polaris/native_gripper_runtime.py",
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


def _declared_absolute_path(path: str | Path, field: str) -> Path:
    """Preserve one normalized absolute spelling without resolving aliases."""

    rendered = os.fspath(path)
    if (
        not isinstance(rendered, str)
        or not rendered.startswith("/")
        or rendered.startswith("//")
        or "\0" in rendered
        or rendered != os.path.normpath(rendered)
    ):
        raise ValueError(f"{field} must use one normalized absolute path spelling")
    return Path(rendered)


def _file_identity(file_stat: os.stat_result) -> tuple[int, ...]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_size,
        file_stat.st_mode,
        file_stat.st_nlink,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _bind_artifact_path(
    recorded_path: str,
    declared_path: Path,
    file_stat: os.stat_result,
    field: str,
) -> dict[str, Any]:
    """Bind producer and host spellings only when they name the same exact inode."""

    if not isinstance(recorded_path, str):
        raise ValueError(f"{field} recorded path must be a string")
    recorded = _declared_absolute_path(recorded_path, f"{field} recorded path")
    declared = _declared_absolute_path(declared_path, f"{field} declared path")
    if recorded.name != declared.name:
        raise ValueError(f"{field} final path component mismatch")
    try:
        recorded_before = os.stat(recorded, follow_symlinks=False)
        declared_before = os.stat(declared, follow_symlinks=False)
        if stat.S_ISLNK(recorded_before.st_mode) or stat.S_ISLNK(
            declared_before.st_mode
        ):
            raise ValueError(f"{field} final path component must not be a symlink")
        recorded_resolved = recorded.resolve(strict=True)
        declared_resolved = declared.resolve(strict=True)
        recorded_resolved_stat = os.stat(recorded_resolved, follow_symlinks=False)
        declared_resolved_stat = os.stat(declared_resolved, follow_symlinks=False)
        recorded_after = os.stat(recorded, follow_symlinks=False)
        declared_after = os.stat(declared, follow_symlinks=False)
    except ValueError:
        raise
    except OSError as error:
        raise ValueError(f"{field} path spelling is not readable") from error
    expected_identity = _file_identity(file_stat)
    if (
        any(
            not stat.S_ISREG(candidate.st_mode)
            or _file_identity(candidate) != expected_identity
            for candidate in (
                recorded_before,
                declared_before,
                recorded_resolved_stat,
                declared_resolved_stat,
                recorded_after,
                declared_after,
            )
        )
        or recorded_resolved != declared_resolved
    ):
        raise ValueError(f"{field} path spellings do not identify the same exact file")
    return {
        "path": str(recorded),
        "host_declared_path": str(declared),
        "resolved_path": str(declared_resolved),
        "path_alias_equivalent": (
            recorded != declared
            or recorded != recorded_resolved
            or declared != declared_resolved
        ),
        "producer_host_spelling_match": recorded == declared,
        "device": format(file_stat.st_dev, "x"),
        "inode": file_stat.st_ino,
    }


def _read_canonical_json(path: Path, field: str, *, mode: int = 0o444):
    path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{field} must be one readable regular file") from error
    try:
        file_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_nlink != 1
            or stat.S_IMODE(file_stat.st_mode) != mode
        ):
            raise ValueError(f"{field} must be one mode-{mode:04o} regular link")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.stat(path, follow_symlinks=False)
    identity = (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_size,
        file_stat.st_mode,
        file_stat.st_nlink,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )
    if (
        identity
        != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mode,
            after.st_nlink,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        or identity
        != (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mode,
            current.st_nlink,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        or not stat.S_ISREG(current.st_mode)
        or current.st_nlink != 1
        or stat.S_IMODE(current.st_mode) != mode
    ):
        raise ValueError(f"{field} changed while it was being read")
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
    report: Any,
    shape: tuple[int, ...],
    expected: list[float],
    field: str,
    *,
    device: str,
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
    if report["dtype"] != "torch.float32" or report["device"] != device:
        raise ValueError(f"{field} must be a {device} torch.float32 tensor")
    values = _flatten_shape(report["values"], shape, field)
    if values != expected:
        raise ValueError(f"{field} values mismatch")


def _validate_runtime(report: Any) -> dict[str, Any]:
    from polaris.joint_velocity_runtime import validate_joint_velocity_runtime_report

    return validate_joint_velocity_runtime_report(report)

    # Kept below as a historical independent description of the retired
    # driver-only schema.  The active all-six schema is centralized above so
    # old controller completions cannot accidentally authorize the new path.
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
        "action_buffers",
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
        device="cuda:0",
    )
    action_buffers = report["action_buffers"]
    if not isinstance(action_buffers, dict) or set(action_buffers) != {
        "raw_action",
        "processed_action",
    }:
        raise ValueError("Runtime arm action-buffer schema mismatch")
    for name in ("raw_action", "processed_action"):
        _validate_array(
            action_buffers[name],
            (1, 7),
            [0.0] * 7,
            f"arm {name}",
            device="cuda:0",
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
    for surface, device in (("buffered", "cuda:0"), ("direct_physx", "cpu")):
        if not isinstance(drive[surface], dict) or set(drive[surface]) != set(arrays):
            raise ValueError(f"Runtime {surface} schema mismatch")
        for name, (shape, values) in arrays.items():
            _validate_array(
                drive[surface][name],
                shape,
                values,
                f"{surface} {name}",
                device=device,
            )
    gripper = report["gripper"]
    if not isinstance(gripper, dict) or set(gripper) != {
        "action_class",
        "joint_name",
        "threshold",
        "open_command",
        "closed_command",
        "raw_action",
        "processed_action",
        "drive",
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
    _validate_array(
        gripper["open_command"],
        (1,),
        [0.0],
        "gripper open",
        device="cuda:0",
    )
    _validate_array(
        gripper["closed_command"],
        (1,),
        [FLOAT32_PI_OVER_4],
        "gripper closed",
        device="cuda:0",
    )
    for name in ("raw_action", "processed_action"):
        _validate_array(
            gripper[name],
            (1, 1),
            [0.0],
            f"gripper {name}",
            device="cuda:0",
        )
    gripper_drive = gripper["drive"]
    if not isinstance(gripper_drive, dict) or set(gripper_drive) != {
        "profile",
        "configured",
        "actuator",
        "direct_physx",
    }:
        raise ValueError("Runtime gripper drive schema mismatch")
    if gripper_drive["profile"] != GRIPPER_DRIVE_PROFILE:
        raise ValueError("Runtime gripper drive profile mismatch")
    expected_configured = {
        "joint_names_expr": ["finger_joint"],
        "stiffness": None,
        "damping": None,
        "effort_limit": GRIPPER_EFFORT_LIMIT,
        "effort_limit_sim": GRIPPER_EFFORT_LIMIT,
        "velocity_limit": GRIPPER_VELOCITY_LIMIT,
        "velocity_limit_sim": GRIPPER_VELOCITY_LIMIT,
    }
    configured = gripper_drive["configured"]
    if configured != expected_configured or any(
        type(configured[key]) is not type(value)
        for key, value in expected_configured.items()
    ):
        raise ValueError("Runtime configured gripper drive mismatch")
    live_arrays = {
        "stiffness": ((1, 1), [GRIPPER_STIFFNESS]),
        "damping": ((1, 1), [GRIPPER_DAMPING]),
        "effort_limit": ((1, 1), [GRIPPER_EFFORT_LIMIT]),
        "velocity_limit": ((1, 1), [GRIPPER_VELOCITY_LIMIT]),
    }
    actuator_arrays = {
        **live_arrays,
        "effort_limit_sim": ((1, 1), [GRIPPER_EFFORT_LIMIT]),
        "velocity_limit_sim": ((1, 1), [GRIPPER_VELOCITY_LIMIT]),
    }
    for surface, arrays, device in (
        ("actuator", actuator_arrays, "cuda:0"),
        ("direct_physx", live_arrays, "cpu"),
    ):
        surface_report = gripper_drive[surface]
        if not isinstance(surface_report, dict) or set(surface_report) != set(arrays):
            raise ValueError(f"Runtime gripper {surface} schema mismatch")
        for name, (shape, values) in arrays.items():
            _validate_array(
                surface_report[name],
                shape,
                values,
                f"gripper {surface} {name}",
                device=device,
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
                "precondition_finger_target": math.pi / 4.0,
                "expected_finger_target": 0.0,
                "expected_motion_sign": -1,
            },
            {
                "label": "gripper_closed",
                "action": [0.0] * 7 + [1.0],
                "kind": "gripper",
                "precondition_finger_target": 0.0,
                "expected_finger_target": math.pi / 4.0,
                "expected_motion_sign": 1,
            },
            {
                "label": "gripper_boundary_0p5",
                "action": [0.0] * 7 + [0.5],
                "kind": "gripper",
                "precondition_finger_target": math.pi / 4.0,
                "expected_finger_target": 0.0,
                "expected_motion_sign": -1,
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
        "expected_gripper_drive_profile",
        "gripper_precondition_steps",
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
        "expected_gripper_drive_profile": GRIPPER_DRIVE_PROFILE,
        "gripper_precondition_steps": GRIPPER_PRECONDITION_STEPS,
        "status": "pass",
        "case_count": 20,
    }
    for key, value in scalars.items():
        if payload[key] != value or type(payload[key]) is not type(value):
            raise ValueError(f"Smoke result {key} mismatch")
    if payload["lifecycle"] != {
        "env_close": "complete",
        "simulation_app_close": "invoked_then_child_exited_zero",
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
        "child_ready_marker_sha256",
        "child_ready_marker_size",
        "child_ready_marker_mode",
        "child_ready_marker_path",
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
        or completion["child_capture_mode"] != "0444"
        or not isinstance(completion["child_capture_path"], str)
        or not Path(completion["child_capture_path"]).is_absolute()
        or not isinstance(completion["child_ready_marker_sha256"], str)
        or len(completion["child_ready_marker_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in completion["child_ready_marker_sha256"]
        )
        or type(completion["child_ready_marker_size"]) is not int
        or completion["child_ready_marker_size"] <= 0
        or completion["child_ready_marker_mode"] != "0444"
        or not isinstance(completion["child_ready_marker_path"], str)
        or not Path(completion["child_ready_marker_path"]).is_absolute()
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
        "finger_position_before",
        "finger_velocity_before",
        "finger_position_after",
        "finger_velocity_after",
        "finger_average_slew_rad_s",
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
        finger_position_before = _finite_vector(
            [case["finger_position_before"]], 1, "finger position before"
        )[0]
        finger_velocity_before = _finite_vector(
            [case["finger_velocity_before"]], 1, "finger velocity before"
        )[0]
        finger_position_after = _finite_vector(
            [case["finger_position_after"]], 1, "finger position after"
        )[0]
        finger_velocity_after = _finite_vector(
            [case["finger_velocity_after"]], 1, "finger velocity after"
        )[0]
        finger_average_slew = _finite_vector(
            [case["finger_average_slew_rad_s"]], 1, "finger average slew"
        )[0]
        expected_average_slew = (finger_position_after - finger_position_before) * 15.0
        if not math.isclose(
            finger_average_slew,
            expected_average_slew,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(f"Smoke case {index} finger slew mismatch")
        if any(
            abs(value) > GRIPPER_VELOCITY_LIMIT + GRIPPER_MEASURED_VELOCITY_TOLERANCE
            for value in (
                finger_velocity_before,
                finger_velocity_after,
                finger_average_slew,
            )
        ):
            raise ValueError(f"Smoke case {index} gripper velocity limit exceeded")
        if expected["kind"] == "gripper":
            if not math.isclose(
                finger_position_before,
                expected["precondition_finger_target"],
                rel_tol=0.0,
                abs_tol=GRIPPER_PRECONDITION_POSITION_TOLERANCE,
            ):
                raise ValueError(f"Smoke case {index} gripper precondition mismatch")
            motion_sign = expected["expected_motion_sign"]
            if motion_sign * (finger_position_after - finger_position_before) <= 1e-4:
                raise ValueError(f"Smoke case {index} gripper direction mismatch")
            if motion_sign * finger_velocity_after <= 1e-4:
                raise ValueError(
                    f"Smoke case {index} gripper velocity direction mismatch"
                )
    reset = payload["reset_probe"]
    if not isinstance(reset, dict) or set(reset) != {
        "default_joint_position",
        "joint_position",
        "joint_velocity",
        "joint_velocity_target",
        "default_finger_position",
        "finger_position",
        "finger_velocity",
        "finger_position_target",
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
    default_finger = _finite_vector(
        [reset["default_finger_position"]], 1, "default finger"
    )[0]
    reset_finger = _finite_vector([reset["finger_position"]], 1, "reset finger")[0]
    reset_finger_velocity = _finite_vector(
        [reset["finger_velocity"]], 1, "reset finger velocity"
    )[0]
    reset_finger_target = _finite_vector(
        [reset["finger_position_target"]], 1, "reset finger target"
    )[0]
    if (
        abs(reset_finger - default_finger) > GRIPPER_PRECONDITION_POSITION_TOLERANCE
        or abs(reset_finger_velocity) > GRIPPER_MEASURED_VELOCITY_TOLERANCE
        or not math.isclose(reset_finger_target, 0.0, rel_tol=0.0, abs_tol=1e-6)
    ):
        raise ValueError("Smoke gripper reset mismatch")
    return payload


def _source_provenance(repository: Path, expected_commit: str) -> dict[str, Any]:
    declared_repository = _declared_absolute_path(repository, "PolaRiS repository")
    if declared_repository.is_symlink():
        raise ValueError("PolaRiS repository path must not be a symlink")
    repository = declared_repository.resolve()
    git_metadata = repository / ".git"
    if git_metadata.is_symlink() or not git_metadata.is_dir():
        raise ValueError(
            "PolaRiS repository must be a standalone clone with an in-root .git directory"
        )
    expected_git_directory = git_metadata.resolve()
    if expected_git_directory != git_metadata:
        raise ValueError(
            "PolaRiS .git directory must resolve inside the repository root"
        )
    top_level = Path(
        _git(repository, "rev-parse", "--show-toplevel").decode().strip()
    ).resolve()
    if top_level != repository:
        raise ValueError("PolaRiS repository path is not the exact Git root")
    git_directory = Path(
        _git(repository, "rev-parse", "--absolute-git-dir").decode().strip()
    ).resolve()
    common_directory = Path(
        _git(
            repository,
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        )
        .decode()
        .strip()
    ).resolve()
    if (
        git_directory != expected_git_directory
        or common_directory != expected_git_directory
    ):
        raise ValueError(
            "PolaRiS Git directory and common directory must both be the in-root .git directory"
        )
    head_reference = (
        _git(repository, "rev-parse", "--abbrev-ref", "HEAD").decode().strip()
    )
    if head_reference != "HEAD":
        raise ValueError("PolaRiS repository must be checked out at a detached HEAD")
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
        "repository": str(declared_repository),
        "resolved_repository": str(repository),
        "git_directory": str(git_directory),
        "git_common_directory": str(common_directory),
        "standalone_git_directory": True,
        "head_reference": "HEAD",
        "detached_head": True,
        "commit": head,
        "tracked_and_untracked_clean": True,
        "files": files,
    }


def _asset_provenance(data_dir: Path) -> dict[str, Any]:
    declared_data_dir = _declared_absolute_path(data_dir, "PolaRiS-Hub data root")
    if declared_data_dir.is_symlink() or not declared_data_dir.is_dir():
        raise ValueError("PolaRiS-Hub data root must be a regular directory")
    data_dir = declared_data_dir.resolve()
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
            "path": str(declared_data_dir / relative_path),
            "resolved_path": str(path),
            "size": file_stat.st_size,
            "sha256": digest,
            "metadata_path": str(
                declared_data_dir
                / ".cache/huggingface/download"
                / (relative_path + ".metadata")
            ),
            "metadata_resolved_path": str(metadata_path),
            "metadata_size": metadata_stat.st_size,
            "metadata_sha256": expected["metadata_sha256"],
            "hub_revision": HUB_REVISION,
        }
    return {
        "root": str(declared_data_dir),
        "resolved_root": str(data_dir),
        "hub_revision": HUB_REVISION,
        "assets": assets,
    }


def _build_attestation(args: argparse.Namespace) -> dict[str, Any]:
    if args.expected_gripper_drive_profile != GRIPPER_DRIVE_PROFILE:
        raise ValueError("Expected gripper drive profile mismatch")
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
    smoke_path = _declared_absolute_path(args.smoke_artifact, "smoke artifact")
    completion_path = _declared_absolute_path(args.completion, "completion")
    srun_status_path = _declared_absolute_path(args.srun_status, "srun status")
    gpu_inventory_path = _declared_absolute_path(args.gpu_inventory, "GPU inventory")
    saved_job_path = _declared_absolute_path(
        args.saved_job_script, "saved Slurm script"
    )
    container_image_path = _declared_absolute_path(
        args.container_image, "container image"
    )
    if smoke_path.name != f"joint-velocity-smoke-{args.job_id}.json":
        raise ValueError("Smoke artifact filename does not bind job ID")
    if completion_path.name != f"controller-smoke-{args.job_id}.completion.json":
        raise ValueError("Completion filename does not bind job ID")
    smoke, smoke_bytes, smoke_stat = _read_canonical_json(smoke_path, "smoke artifact")
    _validate_smoke(smoke)
    if smoke["expected_gripper_drive_profile"] != args.expected_gripper_drive_profile:
        raise ValueError("Smoke does not bind expected gripper drive profile")
    expected_child_path = smoke_path.with_name(smoke_path.name + ".child-close.json")
    expected_ready_path = expected_child_path.with_name(
        expected_child_path.name + ".ready.json"
    )
    expected_failure_path = expected_child_path.with_name(
        expected_child_path.name + ".failure.json"
    )
    if expected_failure_path.exists() or expected_failure_path.is_symlink():
        raise ValueError("Smoke child failure status exists beside a pass artifact")
    completion_evidence = smoke["completion"]
    _, child_bytes, child_stat = _read_canonical_json(
        expected_child_path, "child close capture", mode=0o444
    )
    child_path_binding = _bind_artifact_path(
        completion_evidence["child_capture_path"],
        expected_child_path,
        child_stat,
        "smoke child capture",
    )
    _, ready_bytes, ready_stat = _read_canonical_json(
        expected_ready_path, "child ready marker", mode=0o444
    )
    ready_path_binding = _bind_artifact_path(
        completion_evidence["child_ready_marker_path"],
        expected_ready_path,
        ready_stat,
        "smoke child ready marker",
    )
    expected_ready = {
        "schema_version": 1,
        "status": "success",
        "stage": "kit_child_after_env_close_before_simulation_app_close",
        "raw_result": {
            "path": child_path_binding["path"],
            "size_bytes": len(child_bytes),
            "sha256": _sha256_bytes(child_bytes),
            "mode": "0444",
        },
    }
    if ready_bytes != _canonical_json(expected_ready, newline=True):
        raise ValueError("Child ready marker does not bind exact child capture")
    expected_child = dict(smoke)
    expected_child.pop("completion")
    expected_child["status"] = "close_validated_pending_parent"
    expected_child["lifecycle"] = {
        "env_close": "complete",
        "simulation_app_close": "pending_child_exit",
        "capture_stage": "kit_child_after_env_close_before_simulation_app_close",
    }
    if child_bytes != _canonical_json(expected_child, newline=True):
        raise ValueError("Child close capture differs from promoted smoke evidence")
    if (
        completion_evidence["child_capture_sha256"] != _sha256_bytes(child_bytes)
        or completion_evidence["child_capture_size"] != len(child_bytes)
        or completion_evidence["child_capture_mode"] != "0444"
    ):
        raise ValueError("Smoke completion does not bind exact child capture")
    if (
        completion_evidence["child_ready_marker_sha256"] != _sha256_bytes(ready_bytes)
        or completion_evidence["child_ready_marker_size"] != len(ready_bytes)
        or completion_evidence["child_ready_marker_mode"] != "0444"
    ):
        raise ValueError("Smoke completion does not bind exact child ready marker")

    status, status_bytes, status_stat = _read_canonical_json(
        srun_status_path, "srun status"
    )
    if (
        status != {"job_id": args.job_id, "srun_exit_code": 0}
        or type(status.get("job_id")) is not int
    ):
        raise ValueError("srun status mismatch")

    gpu_stat = _regular_file(gpu_inventory_path, "GPU inventory", mode=0o444)
    gpu_bytes = gpu_inventory_path.read_bytes()
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
    saved_stat = _regular_file(saved_job_path, "saved Slurm script", mode=0o444)
    saved_sha = _file_sha256(saved_job_path)
    if saved_sha != job_script["sha256"]:
        raise ValueError("Saved Slurm script differs from committed launch script")

    image_stat = _regular_file(container_image_path, "container image")
    image_sha = _file_sha256(container_image_path)
    if image_sha != IMAGE_SHA256:
        raise ValueError("Container image SHA-256 mismatch")
    data = _asset_provenance(args.data_dir)

    return {
        "schema_version": 1,
        "profile": PROFILE,
        "status": "pass",
        "scope": "controller_only_no_model_or_checkpoint",
        "promotion": "forbidden_without_separate_checkpoint_canary",
        "candidate_intent": {
            "expected_gripper_drive_profile": args.expected_gripper_drive_profile,
        },
        "slurm": {
            "job_id": args.job_id,
            "srun_exit_code": 0,
            "status_artifact": {
                **_bind_artifact_path(
                    str(srun_status_path),
                    srun_status_path,
                    status_stat,
                    "srun status",
                ),
                "size": len(status_bytes),
                "sha256": _sha256_bytes(status_bytes),
                "mode": "0444",
            },
            "gpu_inventory": {
                **_bind_artifact_path(
                    str(gpu_inventory_path),
                    gpu_inventory_path,
                    gpu_stat,
                    "GPU inventory",
                ),
                "size": gpu_stat.st_size,
                "sha256": _sha256_bytes(gpu_bytes),
                "mode": "0444",
                "uuid": fields[0],
                "name": fields[1],
                "driver_version": fields[2],
            },
            "saved_job_script": {
                **_bind_artifact_path(
                    str(saved_job_path),
                    saved_job_path,
                    saved_stat,
                    "saved Slurm script",
                ),
                "size": saved_stat.st_size,
                "sha256": saved_sha,
                "mode": "0444",
            },
        },
        "source": source,
        "runtime": {
            "container_image": {
                **_bind_artifact_path(
                    str(container_image_path),
                    container_image_path,
                    image_stat,
                    "container image",
                ),
                "size": image_stat.st_size,
                "sha256": image_sha,
            },
            "polaris_hub": data,
        },
        "smoke_artifact": {
            **_bind_artifact_path(
                str(smoke_path), smoke_path, smoke_stat, "smoke artifact"
            ),
            "size": smoke_stat.st_size,
            "sha256": _sha256_bytes(smoke_bytes),
            "mode": "0444",
            "nlink": 1,
            "status": "pass",
            "case_count": 20,
            "runtime_sha256": smoke["runtime_contract"]["runtime_sha256"],
            "child_close_capture": {
                **child_path_binding,
                "size": child_stat.st_size,
                "sha256": _sha256_bytes(child_bytes),
                "mode": "0444",
                "status": "close_validated_pending_parent",
            },
            "child_ready_marker": {
                **ready_path_binding,
                "size": ready_stat.st_size,
                "sha256": _sha256_bytes(ready_bytes),
                "mode": "0444",
                "status": "success",
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
    parser.add_argument("--smoke-artifact", required=True)
    parser.add_argument("--completion", required=True)
    parser.add_argument("--srun-status", required=True)
    parser.add_argument("--gpu-inventory", required=True)
    parser.add_argument("--saved-job-script", required=True)
    parser.add_argument("--polaris-repo", required=True)
    parser.add_argument("--expected-polaris-commit", required=True)
    parser.add_argument("--expected-gripper-drive-profile", required=True)
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args(argv)

    expected = _build_attestation(args)
    completion_path = _declared_absolute_path(args.completion, "completion")
    if args.mode == "finalize":
        if completion_path.exists() or completion_path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {completion_path}")
        _publish(completion_path, expected)
    actual, actual_bytes, _ = _read_canonical_json(completion_path, "completion")
    if actual != expected:
        raise ValueError("Completion attestation does not match live provenance")
    print(
        json.dumps(
            {
                "completion": str(completion_path),
                "completion_resolved": str(completion_path.resolve(strict=True)),
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
