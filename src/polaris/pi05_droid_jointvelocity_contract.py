r"""Fail-closed contract for the official native-velocity :math:`\pi_{0.5}` DROID policy.

This module is deliberately lightweight.  It can be imported by the PolaRiS
client, launch preflights, and host tests without importing Isaac Lab or JAX.
The server metadata is an exact attestation: the client rejects missing,
additional, or changed fields instead of guessing a checkpoint from its
response shape.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


PI05_DROID_JOINTVELOCITY_PROFILE = "openpi_pi05_droid_native_jointvelocity_v1"
PI05_DROID_CONTRACT_METADATA_KEY = "polaris_pi05_droid_contract"
PI05_DROID_CHECKPOINT_URI = "gs://openpi-assets/checkpoints/pi05_droid"
PI05_DROID_CHECKPOINT_MANIFEST_SHA256 = (
    "6f9ccfa5695c669962ad10dbe0dcb7d44bf903918e5fffe33e5d1ff531287922"
)
PI05_DROID_CHECKPOINT_OBJECT_COUNT = 20
PI05_DROID_CHECKPOINT_BYTES = 12_429_488_598
PI05_DROID_NORM_STATS_SHA256 = (
    "403b3a22f897e9ae5dd617966a3c8f7d1835ac79dfd5a8993179514be26a3b8b"
)
PI05_DROID_OPENPI_COMMIT = "bd70b8f4011e85b3f3b0f039f12113f78718e7bf"
PI05_DROID_ISAACLAB_VERSION = "2.3.0"
PI05_DROID_ISAACLAB_REVISION = "3c6e67bb5c7ada942a6d1884ab69338f57596f77"
PI05_DROID_ISAACLAB_SOURCE_SHA256 = {
    "actions_cfg.py": "94722a5c0d6da3639b5507130d1ec2e7f62d7490e4625d4bf30ada8691ef63d4",
    "joint_actions.py": "1b3dcb55d969d886cee500e660f12331e49f8638fec98cde30d2b818a1ca1692",
    "actuator_cfg.py": "3963167d6678c6f052d87202678822d80a98b0b0f3492b859774768b1bd80520",
    "actuator_pd.py": "1d2a9b80812714f5aade3ed7bbb7c74a403ab718868aa73d187eb77173695beb",
    "articulation.py": "9cc03b85642c36c801ff9683e94b8ccc3fbef1178761338974b433dacc78ef75",
}

PANDA_ARM_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
PANDA_ARM_VELOCITY_LIMITS = (2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61)
PANDA_ARM_EFFORT_LIMITS = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)
PANDA_ARM_VELOCITY_DRIVE_STIFFNESS = 0.0
PANDA_ARM_VELOCITY_DRIVE_DAMPING = 80.0

_OPENPI_SOURCE_SHA256 = {
    "docs/norm_stats.md": (
        "62e4a39ea7fcdae53b4466942db348c3e38fc016b7f7a5273cafed649ebe614b"
    ),
    "examples/droid/main.py": (
        "4bdbdbdde068ebb30ba90d62bf8c51df156b4baae5c5ac5870c08f7b03d4afc2"
    ),
    "scripts/serve_policy.py": (
        "eccc0448b4873fd30a1fff3355c5de7a7c227138e6db04b7dcaa5bb38a6a5809"
    ),
    "packages/openpi-client/src/openpi_client/image_tools.py": (
        "d48b4bd7f44e79fe6db8a8e07c9161144fa250be686e1245014a8b47e6171977"
    ),
    "packages/openpi-client/src/openpi_client/websocket_client_policy.py": (
        "36557cb0b91ccf31cd4fb4b508306850d76ed0feb4028dac5182d0f5a5d88005"
    ),
    "src/openpi/models/pi0.py": (
        "24b32d7c6ed5e409459afa8ca5af96a4dae09c2dcdf48e828907e411f4f12342"
    ),
    "src/openpi/models/pi0_config.py": (
        "e15d71586f1a561b35c5651227607613fb16d2df411f3708b1c9d36b02544285"
    ),
    "src/openpi/policies/droid_policy.py": (
        "7d0893555964d485e231f70084125084a99495705ddd74b29461229d524fff26"
    ),
    "src/openpi/policies/policy.py": (
        "a2ac5236e23b11d0f5a840d4a8fef148d89dba921d5452ec5dfcd5a6ffa46f20"
    ),
    "src/openpi/policies/policy_config.py": (
        "aaf42ab04a33b6c91d2447926211646c33ebd8ebfa427f026d7b6c4f7c45ec52"
    ),
    "src/openpi/training/config.py": (
        "c34231f888dba5dc981d808765b3d98775ab845b04313bf53fcecd567c520b0d"
    ),
    "src/openpi/shared/normalize.py": (
        "6c2cea4946fb07e51801530400d2b2fd94730e195cd290d6b6960114eca9739d"
    ),
    "src/openpi/transforms.py": (
        "a1b94e9e72849a18834778f229c6bb389a495eb7fbe0aa800edea728b9424ff4"
    ),
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def contract_sha256(contract: dict[str, Any]) -> str:
    """Return the identity of a contract document without its identity field."""

    payload = copy.deepcopy(contract)
    payload.pop("contract_sha256", None)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def expected_pi05_droid_jointvelocity_contract() -> dict[str, Any]:
    """Build the one accepted native joint-velocity serving contract."""

    contract: dict[str, Any] = {
        "schema_version": 1,
        "profile": PI05_DROID_JOINTVELOCITY_PROFILE,
        "checkpoint": {
            "uri": PI05_DROID_CHECKPOINT_URI,
            "content_manifest_profile": "gcs_path_size_md5_v1",
            "content_manifest_sha256": PI05_DROID_CHECKPOINT_MANIFEST_SHA256,
            "object_count": PI05_DROID_CHECKPOINT_OBJECT_COUNT,
            "total_bytes": PI05_DROID_CHECKPOINT_BYTES,
        },
        "normalization": {
            "asset_id": "droid",
            "scope": "checkpoint_global_droid",
            "path": "assets/droid/norm_stats.json",
            "sha256": PI05_DROID_NORM_STATS_SHA256,
            "formula": ("openpi_quantile_q01_q99_epsilon1e-6_no_clip_float64_stats_v1"),
            "statistics_dtype": "numpy_float64_from_checkpoint_json",
            "normalization_arithmetic_dtype": "numpy_float64",
            "state_stats_width": 32,
            "action_stats_width": 32,
            "served_action_dimensions": list(range(8)),
        },
        "openpi": {
            "commit": PI05_DROID_OPENPI_COMMIT,
            "policy_config": "pi05_droid",
            "model_type": "pi05",
            "model_compute_dtype": "bfloat16",
            "model_action_horizon": 15,
            "model_action_dim": 32,
            "sampler": "flow_euler_t1_to_t0_num_steps10_rng_key0_v1",
            "jax_enable_x64": False,
            "droid_output_projection": "leading_dimensions_0_through_7_v1",
            "source_sha256": dict(_OPENPI_SOURCE_SHA256),
        },
        "policy_input": {
            "state": "7_panda_joint_positions_radians_plus_closed_positive_gripper",
            "state_width": 8,
            "request_state_dtype": "float32",
            "images": [
                {
                    "model_slot": "base_0_rgb",
                    "source": "external",
                    "shape": [224, 224, 3],
                    "dtype": "uint8",
                    "masked": False,
                },
                {
                    "model_slot": "left_wrist_0_rgb",
                    "source": "wrist",
                    "shape": [224, 224, 3],
                    "dtype": "uint8",
                    "masked": False,
                },
                {
                    "model_slot": "right_wrist_0_rgb",
                    "source": "zero_blank",
                    "shape": [224, 224, 3],
                    "dtype": "uint8",
                    "masked": True,
                },
            ],
            "resize": "openpi_image_tools_resize_with_pad_224_v1",
            "wrist_rotation_degrees": 0,
        },
        "policy_output": {
            "response_shape": [15, 8],
            "response_dtype": "float64_after_checkpoint_unnormalize",
            "execute_first": 8,
            "arm": "panda_joint1_through_7_velocity_radians_per_second",
            "gripper": "absolute_closed_positive_binarize_gt_0p5_else_open",
            "processing_order": [
                "binarize_gripper_gt_0p5",
                "clip_all_8_dimensions_minus1_plus1",
            ],
        },
        "control": {
            "mode": "joint_velocity",
            "isaaclab_version": PI05_DROID_ISAACLAB_VERSION,
            "isaaclab_revision": PI05_DROID_ISAACLAB_REVISION,
            "isaaclab_source_sha256": dict(PI05_DROID_ISAACLAB_SOURCE_SHA256),
            "policy_frequency_hz": 15,
            "physics_frequency_hz": 120,
            "decimation": 8,
            "joint_names": list(PANDA_ARM_JOINT_NAMES),
            "action_cfg": "isaaclab_JointVelocityActionCfg",
            "scale": 1.0,
            "offset": 0.0,
            "use_default_offset": False,
            "clip": [-1.0, 1.0],
            "position_integration": "forbidden",
            "velocity_drive": {
                "actuator": "implicit_physx_pd",
                "position_stiffness": PANDA_ARM_VELOCITY_DRIVE_STIFFNESS,
                "velocity_damping": PANDA_ARM_VELOCITY_DRIVE_DAMPING,
                "effort_limit_sim": list(PANDA_ARM_EFFORT_LIMITS),
                "velocity_limit_sim": list(PANDA_ARM_VELOCITY_LIMITS),
            },
        },
    }
    contract["contract_sha256"] = contract_sha256(contract)
    return contract


def expected_pi05_droid_server_metadata() -> dict[str, Any]:
    """Return exact metadata for a wrapped official OpenPI server."""

    return {
        PI05_DROID_CONTRACT_METADATA_KEY: expected_pi05_droid_jointvelocity_contract()
    }


def validate_pi05_droid_server_metadata(metadata: Any) -> dict[str, Any]:
    """Validate metadata by exact canonical equality and return a private copy."""

    expected = expected_pi05_droid_server_metadata()
    if not isinstance(metadata, dict):
        raise ValueError("pi0.5-DROID server metadata must be an object")
    if metadata != expected:
        expected_keys = sorted(expected)
        actual_keys = (
            sorted(metadata) if all(isinstance(key, str) for key in metadata) else []
        )
        raise ValueError(
            "pi0.5-DROID native joint-velocity contract mismatch: "
            f"expected top-level keys {expected_keys}, got {actual_keys}"
        )
    contract = metadata[PI05_DROID_CONTRACT_METADATA_KEY]
    if contract.get("contract_sha256") != contract_sha256(contract):
        raise ValueError("pi0.5-DROID contract SHA-256 is invalid")
    return copy.deepcopy(contract)


def verify_profile_source_files(openpi_dir: Path) -> dict[str, str]:
    """Verify the OpenPI files that define native DROID inference semantics."""

    openpi_dir = openpi_dir.resolve()
    actual: dict[str, str] = {}
    for relative_path, expected_digest in _OPENPI_SOURCE_SHA256.items():
        path = openpi_dir / relative_path
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing regular OpenPI contract source: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_digest:
            raise ValueError(
                f"OpenPI contract source mismatch for {relative_path}: {digest}"
            )
        actual[relative_path] = digest
    return actual


def verify_profile_manifest(manifest_path: Path) -> dict[str, Any]:
    """Verify the pinned base-checkpoint manifest without downloading weights."""

    manifest_path = Path(manifest_path)
    if manifest_path.is_symlink():
        raise ValueError(f"pi0.5-DROID manifest must not be a symlink: {manifest_path}")
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise ValueError(f"Missing regular pi0.5-DROID manifest: {manifest_path}")
    payload = manifest_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != PI05_DROID_CHECKPOINT_MANIFEST_SHA256:
        raise ValueError(f"pi0.5-DROID manifest SHA-256 mismatch: {digest}")
    lines = payload.decode("ascii").splitlines()
    if len(lines) != PI05_DROID_CHECKPOINT_OBJECT_COUNT:
        raise ValueError(f"pi0.5-DROID manifest object count mismatch: {len(lines)}")
    total_bytes = 0
    for line_number, line in enumerate(lines, start=1):
        fields = line.split("\t")
        if len(fields) != 3 or not fields[0].startswith("checkpoints/pi05_droid/"):
            raise ValueError(f"Invalid pi0.5-DROID manifest line {line_number}")
        try:
            total_bytes += int(fields[1])
        except ValueError as error:
            raise ValueError(
                f"Invalid pi0.5-DROID manifest size on line {line_number}"
            ) from error
    if total_bytes != PI05_DROID_CHECKPOINT_BYTES:
        raise ValueError(f"pi0.5-DROID manifest byte count mismatch: {total_bytes}")
    return {
        "sha256": digest,
        "object_count": len(lines),
        "total_bytes": total_bytes,
    }


def verify_pi05_droid_checkpoint(
    checkpoint_dir: Path, manifest_path: Path, *, full_md5: bool = True
) -> dict[str, Any]:
    """Verify the complete immutable checkpoint object set used by the server."""

    manifest_report = verify_profile_manifest(manifest_path)
    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir.is_symlink():
        raise ValueError(
            f"pi0.5-DROID checkpoint must not be a symlink: {checkpoint_dir}"
        )
    checkpoint_dir = checkpoint_dir.resolve()
    if not checkpoint_dir.is_dir():
        raise ValueError(
            f"pi0.5-DROID checkpoint must be a regular directory: {checkpoint_dir}"
        )
    entries: list[tuple[str, int, str]] = []
    prefix = "checkpoints/pi05_droid/"
    for line in manifest_path.read_text(encoding="ascii").splitlines():
        object_path, size, md5_base64 = line.split("\t")
        entries.append((object_path[len(prefix) :], int(size), md5_base64))

    expected_paths = {entry[0] for entry in entries}
    actual_paths: set[str] = set()
    for path in checkpoint_dir.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Checkpoint must not contain symlinks: {path}")
        if path.is_file():
            actual_paths.add(path.relative_to(checkpoint_dir).as_posix())
    if actual_paths != expected_paths:
        raise ValueError(
            "pi0.5-DROID checkpoint file set mismatch: "
            f"missing={sorted(expected_paths - actual_paths)}, "
            f"extra={sorted(actual_paths - expected_paths)}"
        )

    for relative_path, expected_size, expected_md5 in entries:
        path = checkpoint_dir / relative_path
        if path.stat().st_size != expected_size:
            raise ValueError(f"Checkpoint size mismatch for {relative_path}")
        if full_md5:
            digest = hashlib.md5(usedforsecurity=False)
            with path.open("rb") as source:
                for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
                    digest.update(block)
            actual_md5 = base64.b64encode(digest.digest()).decode("ascii")
            if actual_md5 != expected_md5:
                raise ValueError(f"Checkpoint MD5 mismatch for {relative_path}")

    norm_path = checkpoint_dir / "assets/droid/norm_stats.json"
    norm_sha256 = hashlib.sha256(norm_path.read_bytes()).hexdigest()
    if norm_sha256 != PI05_DROID_NORM_STATS_SHA256:
        raise ValueError(f"pi0.5-DROID norm-stats SHA-256 mismatch: {norm_sha256}")
    return {
        **manifest_report,
        "checkpoint_dir": str(checkpoint_dir),
        "norm_stats_sha256": norm_sha256,
        "full_md5": full_md5,
    }
