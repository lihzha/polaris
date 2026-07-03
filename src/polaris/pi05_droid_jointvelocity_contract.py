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
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


PI05_DROID_JOINTVELOCITY_PROFILE = "openpi_pi05_droid_native_jointvelocity_v1"
PI05_DROID_CONTRACT_METADATA_KEY = "ego_lap_serving_contract"
PI05_DROID_CONTRACT_FILENAME = "ego_lap_serving_contract.json"
PI05_DROID_CHECKPOINT_URI = "gs://openpi-assets/checkpoints/pi05_droid"
PI05_DROID_CHECKPOINT_MANIFEST_SHA256 = (
    "6f9ccfa5695c669962ad10dbe0dcb7d44bf903918e5fffe33e5d1ff531287922"
)
PI05_DROID_CHECKPOINT_OBJECT_COUNT = 20
PI05_DROID_CHECKPOINT_BYTES = 12_429_488_598
PI05_DROID_NORM_STATS_SHA256 = (
    "403b3a22f897e9ae5dd617966a3c8f7d1835ac79dfd5a8993179514be26a3b8b"
)
PI05_DROID_OPENPI_INFERENCE_COMPATIBILITY_COMMIT = (
    "bd70b8f4011e85b3f3b0f039f12113f78718e7bf"
)
PI05_DROID_ISAACLAB_VERSION = "2.3.0"
PI05_DROID_ISAACLAB_REVISION = "3c6e67bb5c7ada942a6d1884ab69338f57596f77"
PI05_DROID_ISAACLAB_SOURCE_SHA256 = {
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
PI05_DROID_POLARIS_RUNTIME_SOURCE_SHA256 = {
    "droid_cfg.py": "19ceaceeb06c06e1f708af2380d47dded2a5f7030c436e266c4efca255509287",
    "robot_cfg.py": "d514b32e07b54f98deb6d9dbc7a5201fff5337cdc4600d9351ee2a95e5c4c4c5",
    "native_gripper_runtime.py": (
        "8b5cd2e7c0f912418a878544fce621c00a4126551a21ab72d486686b67d70ff1"
    ),
    "manager_based_rl_splat_environment.py": (
        "9381b2704e86aae6447eb2cc229471612104c5eec2acc9d913252a09159da426"
    ),
}

PANDA_ARM_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
PANDA_ARM_VELOCITY_LIMITS = (2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61)
PANDA_ARM_EFFORT_LIMITS = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)
PANDA_ARM_VELOCITY_DRIVE_STIFFNESS = 0.0
PANDA_ARM_VELOCITY_DRIVE_DAMPING = 80.0
NATIVE_GRIPPER_DRIVE_PROFILE = (
    "implicit_gripper_physx_velocity_limit5_followers5_every_reset_"
    "cuda_actuator_cpu_static_physx_v1"
)
NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S = 5.0
NATIVE_GRIPPER_EFFORT_LIMIT = 200.0
NATIVE_GRIPPER_STIFFNESS = 5729.578125
NATIVE_GRIPPER_DAMPING = 0.011459155939519405
NATIVE_GRIPPER_PRECONDITION_STEPS = 5
NATIVE_GRIPPER_PRECONDITION_POSITION_TOLERANCE = 0.02
NATIVE_GRIPPER_MEASURED_VELOCITY_TOLERANCE = 0.001

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
    "packages/openpi-client/src/openpi_client/base_policy.py": (
        "88e98e0f06293f3c4767db675061664c4bba12f87242ba1e64377d1b8de8069e"
    ),
    "packages/openpi-client/src/openpi_client/msgpack_numpy.py": (
        "c04568948fcee52b691e3be4b6cffb759f7e79ad67530fcd5d23095a0d13c057"
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
    "src/openpi/models/model.py": (
        "0d74bc1d8f4623ac3d8e543710b8bf75f9b95e588a3ebf6127c659e001385cff"
    ),
    "src/openpi/models/tokenizer.py": (
        "965be8b3c393a6811875bbc32da9e01a5d01cc2f87802de801cf7293e049748c"
    ),
    "src/openpi/models/gemma.py": (
        "7e42ada4ae7e9995f0ef3e33a0c224758323c2015e2d9d72ec15726a8a001d2a"
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
    "src/openpi/serving/websocket_policy_server.py": (
        "1370d345e6c3c5b8f15573050e485e60a5b423d1df33e24b237805e6b442b026"
    ),
    "src/openpi/training/config.py": (
        "c34231f888dba5dc981d808765b3d98775ab845b04313bf53fcecd567c520b0d"
    ),
    "src/openpi/shared/normalize.py": (
        "6c2cea4946fb07e51801530400d2b2fd94730e195cd290d6b6960114eca9739d"
    ),
    "src/openpi/shared/download.py": (
        "1fee4957a275ab632fd4dd98af240516a2a708da2db38a838132f05c04da425d"
    ),
    "src/openpi/shared/nnx_utils.py": (
        "4da7111ba3011c4d1c381c9ed17257771df6a50ed89182647c3fb6f79208005e"
    ),
    "src/openpi/training/checkpoints.py": (
        "3cc32682ec3609e9075dabd991488edb9d7f6c045443f58963d733f99c468dbd"
    ),
    "src/openpi/transforms.py": (
        "a1b94e9e72849a18834778f229c6bb389a495eb7fbe0aa800edea728b9424ff4"
    ),
}
_REFERENCE_OPENPI_NAMESPACE_PACKAGES = [
    {"module": "openpi.models.utils", "relative_paths": ["src/openpi/models/utils"]},
    {
        "module": "openpi.models_pytorch",
        "relative_paths": ["src/openpi/models_pytorch"],
    },
    {"module": "openpi.policies", "relative_paths": ["src/openpi/policies"]},
    {"module": "openpi.serving", "relative_paths": ["src/openpi/serving"]},
    {"module": "openpi.training", "relative_paths": ["src/openpi/training"]},
    {
        "module": "openpi.training.misc",
        "relative_paths": ["src/openpi/training/misc"],
    },
]


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


def _module_name_for_source(relative_path: str) -> str | None:
    if relative_path.startswith("src/openpi/") and relative_path.endswith(".py"):
        suffix = relative_path.removeprefix("src/").removesuffix(".py")
        return suffix.replace("/__init__", "").replace("/", ".")
    client_prefix = "packages/openpi-client/src/"
    if relative_path.startswith(client_prefix) and relative_path.endswith(".py"):
        suffix = relative_path.removeprefix(client_prefix).removesuffix(".py")
        return suffix.replace("/__init__", "").replace("/", ".")
    return None


def verify_openpi_git_checkout(openpi_dir: Path) -> dict[str, Any]:
    """Require one exact, fully clean OpenPI inference checkout."""

    requested = Path(openpi_dir)
    if requested.is_symlink():
        raise ValueError("OpenPI checkout path must not be a symlink")
    root = requested.resolve()
    git_metadata = root / ".git"
    if (
        not root.is_dir()
        or git_metadata.is_symlink()
        or not (git_metadata.is_file() or git_metadata.is_dir())
    ):
        raise ValueError(f"OpenPI path is not a Git checkout: {root}")

    def git(*arguments: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(root), *arguments],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError(f"Cannot inspect OpenPI Git checkout: {root}") from error

    top_level = Path(git("rev-parse", "--show-toplevel")).resolve()
    if top_level != root:
        raise ValueError(
            f"--openpi-dir must name the checkout root exactly: {root} != {top_level}"
        )
    head = git("rev-parse", "HEAD")
    if head != PI05_DROID_OPENPI_INFERENCE_COMPATIBILITY_COMMIT:
        raise ValueError(
            "OpenPI inference compatibility HEAD mismatch: "
            f"expected {PI05_DROID_OPENPI_INFERENCE_COMPATIBILITY_COMMIT}, got {head}"
        )
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ValueError("OpenPI inference checkout must be completely clean")
    return {
        "root": str(root),
        "git_head": head,
        "git_tracked_and_untracked_clean": True,
    }


def make_openpi_runtime_attestation(
    module_records: list[dict[str, str]],
    namespace_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a closed attestation for the imported inference-compatible runtime."""

    if not isinstance(module_records, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("module"), str)
        for item in module_records
    ):
        raise ValueError("OpenPI module records must be a list of named objects")
    if namespace_records is None:
        namespace_records = _REFERENCE_OPENPI_NAMESPACE_PACKAGES
    if not isinstance(namespace_records, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("module"), str)
        for item in namespace_records
    ):
        raise ValueError("OpenPI namespace records must be a list of named objects")
    normalized = sorted(copy.deepcopy(module_records), key=lambda item: item["module"])
    normalized_namespaces = sorted(
        copy.deepcopy(namespace_records), key=lambda item: item["module"]
    )
    attestation: dict[str, Any] = {
        "schema_version": 1,
        "compatibility_role": "inference_only_not_training_provenance",
        "git_head": PI05_DROID_OPENPI_INFERENCE_COMPATIBILITY_COMMIT,
        "git_tracked_and_untracked_clean": True,
        "critical_source_sha256": dict(_OPENPI_SOURCE_SHA256),
        "imported_modules": normalized,
        "imported_modules_sha256": hashlib.sha256(
            _canonical_json(normalized)
        ).hexdigest(),
        "namespace_packages": normalized_namespaces,
        "namespace_packages_sha256": hashlib.sha256(
            _canonical_json(normalized_namespaces)
        ).hexdigest(),
    }
    validate_openpi_runtime_attestation(attestation)
    return attestation


def reference_openpi_runtime_attestation() -> dict[str, Any]:
    """Return a deterministic complete-source attestation for host-side tests."""

    records = []
    for relative_path, digest in _OPENPI_SOURCE_SHA256.items():
        module_name = _module_name_for_source(relative_path)
        if module_name is not None:
            records.append(
                {
                    "module": module_name,
                    "relative_path": relative_path,
                    "sha256": digest,
                }
            )
    return make_openpi_runtime_attestation(
        records, copy.deepcopy(_REFERENCE_OPENPI_NAMESPACE_PACKAGES)
    )


def validate_openpi_runtime_attestation(attestation: Any) -> dict[str, Any]:
    """Validate Git identity, source identity, and every imported module origin."""

    if not isinstance(attestation, dict):
        raise ValueError("OpenPI runtime attestation must be an object")
    required_keys = {
        "schema_version",
        "compatibility_role",
        "git_head",
        "git_tracked_and_untracked_clean",
        "critical_source_sha256",
        "imported_modules",
        "imported_modules_sha256",
        "namespace_packages",
        "namespace_packages_sha256",
    }
    if set(attestation) != required_keys:
        raise ValueError("OpenPI runtime attestation schema mismatch")
    if attestation["schema_version"] != 1:
        raise ValueError("OpenPI runtime attestation version mismatch")
    if attestation["compatibility_role"] != "inference_only_not_training_provenance":
        raise ValueError("OpenPI compatibility role must not claim training provenance")
    if attestation["git_head"] != PI05_DROID_OPENPI_INFERENCE_COMPATIBILITY_COMMIT:
        raise ValueError("OpenPI inference-compatibility HEAD mismatch")
    if attestation["git_tracked_and_untracked_clean"] is not True:
        raise ValueError("OpenPI inference checkout must be completely clean")
    if attestation["critical_source_sha256"] != _OPENPI_SOURCE_SHA256:
        raise ValueError("OpenPI critical source manifest mismatch")
    records = attestation["imported_modules"]
    if not isinstance(records, list) or not records:
        raise ValueError("OpenPI imported-module attestation is empty")
    if any(not isinstance(item, dict) for item in records) or records != sorted(
        records, key=lambda item: item.get("module", "")
    ):
        raise ValueError("OpenPI imported-module attestation must be sorted")
    seen_modules: set[str] = set()
    seen_paths: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {
            "module",
            "relative_path",
            "sha256",
        }:
            raise ValueError(f"OpenPI imported-module record {index} schema mismatch")
        module = record["module"]
        relative_path = record["relative_path"]
        digest = record["sha256"]
        if (
            not isinstance(module, str)
            or not (
                module == "openpi"
                or module.startswith("openpi.")
                or module == "openpi_client"
                or module.startswith("openpi_client.")
            )
            or module in seen_modules
        ):
            raise ValueError(f"Invalid OpenPI imported module at record {index}")
        if (
            not isinstance(relative_path, str)
            or relative_path.startswith(("/", "../"))
            or "/../" in relative_path
            or not relative_path.endswith(".py")
            or relative_path in seen_paths
            or not relative_path.startswith(
                ("src/openpi/", "packages/openpi-client/src/openpi_client/")
            )
        ):
            raise ValueError(f"Invalid OpenPI module origin at record {index}")
        if _module_name_for_source(relative_path) != module:
            raise ValueError(f"OpenPI module/origin mapping mismatch at record {index}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"Invalid OpenPI module digest at record {index}")
        expected_digest = _OPENPI_SOURCE_SHA256.get(relative_path)
        if expected_digest is not None and digest != expected_digest:
            raise ValueError(
                f"OpenPI critical imported source digest mismatch at record {index}"
            )
        seen_modules.add(module)
        seen_paths.add(relative_path)
    required_imported_paths = {
        "src/openpi/models/model.py",
        "src/openpi/models/tokenizer.py",
        "src/openpi/models/pi0.py",
        "src/openpi/policies/policy.py",
        "src/openpi/policies/policy_config.py",
        "src/openpi/serving/websocket_policy_server.py",
        "src/openpi/transforms.py",
    }
    if not required_imported_paths.issubset(seen_paths):
        raise ValueError(
            "OpenPI imported runtime is missing execution-critical modules: "
            f"{sorted(required_imported_paths - seen_paths)}"
        )
    expected_manifest = hashlib.sha256(_canonical_json(records)).hexdigest()
    if attestation["imported_modules_sha256"] != expected_manifest:
        raise ValueError("OpenPI imported-module manifest SHA-256 mismatch")
    namespaces = attestation["namespace_packages"]
    if (
        not isinstance(namespaces, list)
        or any(not isinstance(item, dict) for item in namespaces)
        or namespaces != sorted(namespaces, key=lambda item: item.get("module", ""))
    ):
        raise ValueError("OpenPI namespace-package attestation must be a sorted list")
    seen_namespace_modules: set[str] = set()
    for index, record in enumerate(namespaces):
        if not isinstance(record, dict) or set(record) != {
            "module",
            "relative_paths",
        }:
            raise ValueError(f"OpenPI namespace record {index} schema mismatch")
        module = record["module"]
        relative_paths = record["relative_paths"]
        if (
            not isinstance(module, str)
            or not (module.startswith("openpi.") or module.startswith("openpi_client."))
            or module in seen_namespace_modules
            or not isinstance(relative_paths, list)
            or not relative_paths
            or relative_paths != sorted(set(relative_paths))
        ):
            raise ValueError(f"Invalid OpenPI namespace record {index}")
        for relative_path in relative_paths:
            if (
                not isinstance(relative_path, str)
                or relative_path.startswith(("/", "../"))
                or "/../" in relative_path
                or relative_path.endswith(".py")
                or not relative_path.startswith(
                    ("src/openpi/", "packages/openpi-client/src/openpi_client/")
                )
            ):
                raise ValueError(f"Invalid OpenPI namespace origin at record {index}")
            expected_module = (
                relative_path.removeprefix("src/")
                .removeprefix("packages/openpi-client/src/")
                .replace("/", ".")
            )
            if expected_module != module:
                raise ValueError(
                    f"OpenPI namespace module/origin mismatch at record {index}"
                )
        seen_namespace_modules.add(module)
    required_namespaces = {
        "openpi.models.utils",
        "openpi.models_pytorch",
        "openpi.policies",
        "openpi.serving",
        "openpi.training",
        "openpi.training.misc",
    }
    if not required_namespaces.issubset(seen_namespace_modules):
        raise ValueError(
            "OpenPI runtime is missing namespace packages: "
            f"{sorted(required_namespaces - seen_namespace_modules)}"
        )
    expected_namespace_manifest = hashlib.sha256(
        _canonical_json(namespaces)
    ).hexdigest()
    if attestation["namespace_packages_sha256"] != expected_namespace_manifest:
        raise ValueError("OpenPI namespace-package manifest SHA-256 mismatch")
    return copy.deepcopy(attestation)


def expected_pi05_droid_jointvelocity_contract(
    openpi_runtime_attestation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the one accepted native joint-velocity serving contract."""

    if openpi_runtime_attestation is None:
        openpi_runtime_attestation = reference_openpi_runtime_attestation()
    openpi_runtime_attestation = validate_openpi_runtime_attestation(
        openpi_runtime_attestation
    )

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
            "inference_compatibility_commit": (
                PI05_DROID_OPENPI_INFERENCE_COMPATIBILITY_COMMIT
            ),
            "training_revision_provenance": "unavailable_in_released_checkpoint",
            "policy_config": "pi05_droid",
            "model_type": "pi05",
            "model_compute_dtype": "bfloat16",
            "model_action_horizon": 15,
            "model_action_dim": 32,
            "sampler": "flow_euler_t1_to_t0_num_steps10_rng_key0_v1",
            "jax_enable_x64": False,
            "droid_output_projection": "leading_dimensions_0_through_7_v1",
            "source_sha256": dict(_OPENPI_SOURCE_SHA256),
            "runtime_attestation": openpi_runtime_attestation,
        },
        "artifact": {
            "filename": PI05_DROID_CONTRACT_FILENAME,
            "encoding": "canonical_json_sort_keys_compact_ascii_newline_v1",
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
            "polaris_runtime_source_sha256": dict(
                PI05_DROID_POLARIS_RUNTIME_SOURCE_SHA256
            ),
            "policy_frequency_hz": 15,
            "physics_frequency_hz": 120,
            "decimation": 8,
            "joint_names": list(PANDA_ARM_JOINT_NAMES),
            "action_cfg": "polaris_AuditedDroidJointVelocityActionCfg",
            "action_cfg_base": "isaaclab_JointVelocityActionCfg",
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
            "gripper_drive": {
                "profile": NATIVE_GRIPPER_DRIVE_PROFILE,
                "joint_name": "finger_joint",
                "configured": {
                    "stiffness": None,
                    "damping": None,
                    "effort_limit": NATIVE_GRIPPER_EFFORT_LIMIT,
                    "effort_limit_sim": NATIVE_GRIPPER_EFFORT_LIMIT,
                    "velocity_limit": NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S,
                    "velocity_limit_sim": NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S,
                },
                "live": {
                    "actuator_cuda": {
                        "device": "cuda:0",
                        "dtype": "torch.float32",
                        "shape": [1, 1],
                        "stiffness": NATIVE_GRIPPER_STIFFNESS,
                        "damping": NATIVE_GRIPPER_DAMPING,
                        "effort_limit": NATIVE_GRIPPER_EFFORT_LIMIT,
                        "effort_limit_sim": NATIVE_GRIPPER_EFFORT_LIMIT,
                        "velocity_limit": NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S,
                        "velocity_limit_sim": NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S,
                    },
                    "direct_physx_cpu": {
                        "device": "cpu",
                        "dtype": "torch.float32",
                        "shape": [1, 1],
                        "stiffness": NATIVE_GRIPPER_STIFFNESS,
                        "damping": NATIVE_GRIPPER_DAMPING,
                        "effort_limit": NATIVE_GRIPPER_EFFORT_LIMIT,
                        "velocity_limit": NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S,
                    },
                },
                "precondition_steps": NATIVE_GRIPPER_PRECONDITION_STEPS,
                "precondition_position_tolerance": (
                    NATIVE_GRIPPER_PRECONDITION_POSITION_TOLERANCE
                ),
                "measured_velocity_tolerance": (
                    NATIVE_GRIPPER_MEASURED_VELOCITY_TOLERANCE
                ),
            },
        },
    }
    contract["contract_sha256"] = contract_sha256(contract)
    return contract


def expected_pi05_droid_server_metadata(
    openpi_runtime_attestation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return exact metadata for a wrapped official OpenPI server."""

    return {
        PI05_DROID_CONTRACT_METADATA_KEY: expected_pi05_droid_jointvelocity_contract(
            openpi_runtime_attestation
        )
    }


def validate_pi05_droid_server_metadata(metadata: Any) -> dict[str, Any]:
    """Validate metadata by exact canonical equality and return a private copy."""

    if not isinstance(metadata, dict):
        raise ValueError("pi0.5-DROID server metadata must be an object")
    if set(metadata) != {PI05_DROID_CONTRACT_METADATA_KEY}:
        raise ValueError("pi0.5-DROID server handshake schema mismatch")
    contract = metadata[PI05_DROID_CONTRACT_METADATA_KEY]
    if not isinstance(contract, dict):
        raise ValueError("pi0.5-DROID serving contract must be an object")
    if contract.get("contract_sha256") != contract_sha256(contract):
        raise ValueError("pi0.5-DROID contract SHA-256 is invalid")
    runtime_attestation = validate_openpi_runtime_attestation(
        contract.get("openpi", {}).get("runtime_attestation")
    )
    expected = expected_pi05_droid_server_metadata(runtime_attestation)
    if metadata != expected:
        raise ValueError("pi0.5-DROID native joint-velocity contract mismatch")
    return copy.deepcopy(contract)


def serving_contract_bytes(metadata: dict[str, Any]) -> bytes:
    """Return the only accepted durable serialization of the full handshake."""

    validate_pi05_droid_server_metadata(metadata)
    return _canonical_json(metadata) + b"\n"


def serving_contract_artifact_sha256(metadata: dict[str, Any]) -> str:
    return hashlib.sha256(serving_contract_bytes(metadata)).hexdigest()


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_immutable_serving_contract(
    path: Path, metadata: dict[str, Any]
) -> dict[str, Any]:
    """Create, fsync, chmod, reread, and hash the complete handshake once."""

    path = Path(path)
    if path.name != PI05_DROID_CONTRACT_FILENAME:
        raise ValueError(
            f"Serving contract must be named {PI05_DROID_CONTRACT_FILENAME}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = serving_contract_bytes(metadata)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444)
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
    return validate_persisted_serving_contract(path, metadata)


def validate_persisted_serving_contract(
    path: Path, metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Validate immutable bytes and optionally cross-check the live handshake."""

    path = Path(path)
    if path.name != PI05_DROID_CONTRACT_FILENAME or path.is_symlink():
        raise ValueError("Persisted serving contract path is invalid")
    file_stat = path.stat()
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
        raise ValueError("Persisted serving contract must be one regular link")
    if stat.S_IMODE(file_stat.st_mode) != 0o444:
        raise ValueError("Persisted serving contract must have mode 0444")
    payload = path.read_bytes()
    try:
        persisted_metadata = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Persisted serving contract is not strict JSON") from error
    validate_pi05_droid_server_metadata(persisted_metadata)
    expected_payload = serving_contract_bytes(persisted_metadata)
    if payload != expected_payload:
        raise ValueError("Persisted serving contract is not canonical JSON")
    if metadata is not None:
        validate_pi05_droid_server_metadata(metadata)
        if payload != serving_contract_bytes(metadata):
            raise ValueError("Persisted serving contract differs from live handshake")
    return {
        "path": str(path.resolve()),
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "contract_sha256": persisted_metadata[PI05_DROID_CONTRACT_METADATA_KEY][
            "contract_sha256"
        ],
        "mode": "0444",
        "nlink": 1,
    }


def verify_profile_source_files(openpi_dir: Path) -> dict[str, str]:
    """Verify the OpenPI files that define native DROID inference semantics."""

    requested = Path(openpi_dir)
    if requested.is_symlink():
        raise ValueError("OpenPI checkout path must not be a symlink")
    openpi_dir = requested.resolve()
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


def attest_imported_openpi_modules(openpi_dir: Path) -> dict[str, Any]:
    """Bind every currently imported OpenPI/OpenPI-client module to one checkout."""

    requested = Path(openpi_dir)
    if requested.is_symlink():
        raise ValueError("OpenPI checkout path must not be a symlink")
    openpi_dir = requested.resolve()
    records: list[dict[str, str]] = []
    namespace_records: list[dict[str, Any]] = []
    for module_name, module in sorted(sys.modules.items()):
        if not (
            module_name == "openpi"
            or module_name.startswith("openpi.")
            or module_name == "openpi_client"
            or module_name.startswith("openpi_client.")
        ):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            search_locations = getattr(module, "__path__", None)
            if search_locations is None:
                raise ValueError(
                    f"Imported OpenPI module has no source origin: {module_name}"
                )
            relative_paths = []
            for location in search_locations:
                raw_location = Path(location)
                if raw_location.is_symlink():
                    raise ValueError(
                        f"OpenPI namespace origin is a symlink: {module_name}"
                    )
                resolved_location = raw_location.resolve()
                try:
                    relative_path = resolved_location.relative_to(openpi_dir).as_posix()
                except ValueError as error:
                    raise ValueError(
                        "Imported OpenPI namespace escaped --openpi-dir: "
                        f"{module_name}={resolved_location}"
                    ) from error
                if not resolved_location.is_dir():
                    raise ValueError(
                        f"OpenPI namespace origin is not a directory: {module_name}"
                    )
                relative_paths.append(relative_path)
            namespace_records.append(
                {
                    "module": module_name,
                    "relative_paths": sorted(relative_paths),
                }
            )
            continue
        if not isinstance(module_file, str):
            raise ValueError(f"Invalid OpenPI module origin: {module_name}")
        path = Path(module_file)
        if path.suffix in {".pyc", ".pyo"}:
            try:
                path = Path(importlib.util.source_from_cache(str(path)))
            except (ValueError, NotImplementedError) as error:
                raise ValueError(
                    f"Cannot resolve OpenPI source origin for {module_name}"
                ) from error
        if path.is_symlink():
            raise ValueError(
                f"Imported OpenPI module source is a symlink: {module_name}={path}"
            )
        resolved = path.resolve()
        try:
            relative_path = resolved.relative_to(openpi_dir).as_posix()
        except ValueError as error:
            raise ValueError(
                f"Imported OpenPI module escaped --openpi-dir: {module_name}={resolved}"
            ) from error
        if not resolved.is_file() or resolved.suffix != ".py":
            raise ValueError(
                f"Imported OpenPI module is not a regular Python source: {resolved}"
            )
        records.append(
            {
                "module": module_name,
                "relative_path": relative_path,
                "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
            }
        )
    return make_openpi_runtime_attestation(records, namespace_records)


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
