"""Fail-closed serving contract for the released PolaRiS pi0.5 checkpoint.

The module has no JAX, OpenPI, or simulator imports.  The attested server uses
it before importing OpenPI, while clients and launch workers use the same pure
validators for the WebSocket handshake and durable evidence.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib.metadata as importlib_metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import tomllib
from typing import Any

import numpy as np


PI05_DROID_JOINTPOS_PROFILE = "openpi_pi05_droid_jointpos_polaris_flow_v1"
PI05_DROID_JOINTPOS_METADATA_KEY = "ego_lap_pi05_droid_jointpos_contract"
PI05_DROID_JOINTPOS_SERVING_CONTRACT_FILENAME = (
    "pi05_droid_jointpos_serving_contract.json"
)
PI05_DROID_JOINTPOS_MODEL_RUNTIME_FILENAME = "pi05_droid_jointpos_model_runtime.json"
PI05_DROID_JOINTPOS_MODEL_RUNTIME_PROFILE = (
    "openpi_pi05_droid_jointpos_polaris_model_runtime_v1"
)
PI05_DROID_JOINTPOS_BIND_HOST = "127.0.0.1"
PI05_DROID_JOINTPOS_RNG_STREAM_FILENAME = "pi05_droid_jointpos_rng_stream.json"
PI05_DROID_JOINTPOS_RNG_STREAM_PROFILE = "openpi_pi05_droid_jointpos_jax_rng_stream_v1"
PI05_DROID_JOINTPOS_HOST_RUNTIME_PROFILE = "openpi_pi05_droid_jointpos_host_runtime_v1"
PI05_DROID_JOINTPOS_CHECKPOINT_URI = (
    "gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris"
)
PI05_DROID_JOINTPOS_CONFIG = "pi05_droid_jointpos_polaris"
PI05_DROID_JOINTPOS_OPENPI_COMMIT = "bd70b8f4011e85b3f3b0f039f12113f78718e7bf"
PI05_DROID_JOINTPOS_MANIFEST_SHA256 = (
    "7abd0c2294d442d429a77655783232206b2b30d95c508d435503135a5523a11c"
)
PI05_DROID_JOINTPOS_OBJECT_COUNT = 27
PI05_DROID_JOINTPOS_CHECKPOINT_BYTES = 12_434_530_837
PI05_DROID_JOINTPOS_NORM_SHA256 = (
    "57ce9956f9e07d65f8a8205aabec72d436a2c8927f53edb40c7a77b14a5a90c7"
)
PI05_DROID_JOINTPOS_NORM_VALUES_SHA256 = (
    "ebb4dcce9f706f40eb784257d01e27b8218cdc07cf985b732b426cb87b175bc7"
)
PI05_DROID_JOINTPOS_TOKENIZER_URI = "gs://big_vision/paligemma_tokenizer.model"
PI05_DROID_JOINTPOS_TOKENIZER_GENERATION = "1711547605575873"
PI05_DROID_JOINTPOS_TOKENIZER_SIZE = 4_264_023
PI05_DROID_JOINTPOS_TOKENIZER_MD5_BASE64 = "FCCtyYVnIKVZ6KhyhLGV4g=="
PI05_DROID_JOINTPOS_TOKENIZER_SHA256 = (
    "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
)
PI05_DROID_JOINTPOS_TOKENIZER_VOCAB_SIZE = 257_152
PI05_DROID_JOINTPOS_UV_LOCK_SHA256 = (
    "5e3a9a0a12d9a6048afea5591f4520c98585499cbd4a8343dcabfe2aaed94e3d"
)
PI05_DROID_JOINTPOS_PYPROJECT_SHA256 = (
    "b115474a844299a32543792c9676d658ad2e9e4158ffc3d12d9a37e302f6bfc0"
)
PI05_DROID_JOINTPOS_REQUIRED_PACKAGE_VERSIONS = {
    "flax": "0.10.2",
    "jax": "0.5.3",
    "jax-cuda12-pjrt": "0.5.3",
    "jax-cuda12-plugin": "0.5.3",
    "jaxlib": "0.5.3",
    "numpy": "1.26.4",
    "openpi": "0.1.0",
    "openpi-client": "0.1.0",
    "orbax-checkpoint": "0.11.13",
    "sentencepiece": "0.2.0",
    "tensorstore": "0.1.74",
    "websockets": "15.0.1",
}
PI05_DROID_JOINTPOS_RECORD_VERIFICATION_EXEMPTIONS = (
    "openpi",
    "openpi-client",
)
PI05_DROID_JOINTPOS_REQUIRED_RUNTIME_ENVIRONMENT = {
    "JAX_PLATFORMS": "cuda",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.35",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
}
PI05_DROID_JOINTPOS_OPTIONAL_RUNTIME_ENVIRONMENT = (
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_VISIBLE_DEVICES",
    "JAX_COMPILATION_CACHE_DIR",
    "JAX_DEFAULT_DTYPE_BITS",
    "JAX_DEFAULT_MATMUL_PRECISION",
    "JAX_DEFAULT_PRNG_IMPL",
    "JAX_DISABLE_JIT",
    "JAX_ENABLE_X64",
    "JAX_LEGACY_PRNG_KEY",
    "JAX_NUM_CPU_DEVICES",
    "JAX_PLATFORM_NAME",
    "JAX_RANDOM_SEED_OFFSET",
    "JAX_THREEFRY_PARTITIONABLE",
    "NVIDIA_TF32_OVERRIDE",
    "NVIDIA_VISIBLE_DEVICES",
    "XLA_FLAGS",
)
PI05_DROID_JOINTPOS_ALLOWED_OPTIONAL_RUNTIME_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES",
    "JAX_COMPILATION_CACHE_DIR",
    "NVIDIA_VISIBLE_DEVICES",
}
PI05_DROID_JOINTPOS_JAX_CONFIG = {
    "default_prng_impl": "threefry2x32",
    "legacy_prng_key": "allow",
    "threefry_partitionable": True,
    "random_seed_offset": 0,
    "default_matmul_precision": None,
    "disable_jit": False,
    "enable_x64": False,
}
PI05_DROID_JOINTPOS_MANIFEST_PREFIX = "checkpoints/polaris/pi05_droid_jointpos_polaris/"
PI05_DROID_JOINTPOS_DELTA_MASK = (True,) * 7 + (False,)
PI05_DROID_JOINTPOS_SERVER_MODEL_RESIZE = (
    "openpi.transforms.ResizeImages_openpi_client_PIL_bilinear_"
    "symmetric_zero_pad_224x224"
)
PI05_DROID_JOINTPOS_RESIZE_PROBE_INPUT_SHA256 = (
    "6a053848cfe83c0fc8c8a81dd84f6b946c63193cd25cdd5dad45f08be8019e89"
)
PI05_DROID_JOINTPOS_RESIZE_PROBE_OUTPUT_SHA256 = (
    "b34903a6acd5abf88f07297420cf3dfc82c6ecc98032d976713a18fa4a11e12f"
)

_REQUIRED_OPENPI_MODULES = {
    "openpi_client.image_tools",
    "openpi.models.model",
    "openpi.models.pi0",
    "openpi.models.pi0_config",
    "openpi.models.tokenizer",
    "openpi.policies.droid_policy",
    "openpi.policies.policy",
    "openpi.policies.policy_config",
    "openpi.serving.websocket_policy_server",
    "openpi.shared.image_tools",
    "openpi.shared.normalize",
    "openpi.training.checkpoints",
    "openpi.training.config",
    "openpi.training.misc.polaris_config",
    "openpi.transforms",
}


def canonical_json_bytes(value: Any) -> bytes:
    """Encode strict canonical JSON used by every contract identity."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _bytes_md5_base64(payload: bytes) -> str:
    digest = hashlib.md5(payload, usedforsecurity=False).digest()
    return base64.b64encode(digest).decode("ascii")


def _expected_sentencepiece_identity(
    *, wrapper_class: str, max_length: int, processor_attribute: str
) -> dict[str, Any]:
    return {
        "wrapper_class": wrapper_class,
        "wrapper_max_length": max_length,
        "fast_wrapper": wrapper_class == "openpi.models.tokenizer.FASTTokenizer",
        "processor_attribute": processor_attribute,
        "processor_class": "sentencepiece.SentencePieceProcessor",
        "serialized_model_proto_size": PI05_DROID_JOINTPOS_TOKENIZER_SIZE,
        "serialized_model_proto_md5_base64": (PI05_DROID_JOINTPOS_TOKENIZER_MD5_BASE64),
        "serialized_model_proto_sha256": PI05_DROID_JOINTPOS_TOKENIZER_SHA256,
        "vocab_size": PI05_DROID_JOINTPOS_TOKENIZER_VOCAB_SIZE,
        "bos_id": 2,
        "eos_id": 1,
        "pad_id": 0,
        "unk_id": 3,
    }


def attest_loaded_tokenizer_sentencepiece(tokenizer: object) -> dict[str, Any]:
    """Hash the serialized proto held by a live standard or FAST tokenizer."""

    wrapper_class = _class_path(tokenizer)
    processor_attributes = {
        "openpi.models.tokenizer.PaligemmaTokenizer": "_tokenizer",
        "openpi.models.tokenizer.FASTTokenizer": "_paligemma_tokenizer",
    }
    if wrapper_class not in processor_attributes:
        raise ValueError(f"Unsupported live OpenPI tokenizer: {wrapper_class}")
    processor_attribute = processor_attributes[wrapper_class]
    processor = getattr(tokenizer, processor_attribute, None)
    if _class_path(processor) != "sentencepiece.SentencePieceProcessor":
        raise ValueError("Loaded tokenizer SentencePiece processor class mismatch")
    try:
        proto = processor.serialized_model_proto()
        identity = {
            "wrapper_class": wrapper_class,
            "wrapper_max_length": tokenizer._max_len,
            "fast_wrapper": wrapper_class == "openpi.models.tokenizer.FASTTokenizer",
            "processor_attribute": processor_attribute,
            "processor_class": _class_path(processor),
            "serialized_model_proto_size": len(proto),
            "serialized_model_proto_md5_base64": _bytes_md5_base64(proto),
            "serialized_model_proto_sha256": hashlib.sha256(proto).hexdigest(),
            "vocab_size": processor.vocab_size(),
            "bos_id": processor.bos_id(),
            "eos_id": processor.eos_id(),
            "pad_id": processor.pad_id(),
            "unk_id": processor.unk_id(),
        }
    except (AttributeError, TypeError) as error:
        raise ValueError(
            "Loaded tokenizer SentencePiece processor is incomplete"
        ) from error
    expected = _expected_sentencepiece_identity(
        wrapper_class=wrapper_class,
        max_length=identity["wrapper_max_length"],
        processor_attribute=processor_attribute,
    )
    _require_exact(
        identity,
        expected,
        "Loaded tokenizer serialized SentencePiece proto mismatch",
    )
    return identity


def validate_paligemma_tokenizer_artifact(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "status",
        "uri",
        "remote",
        "local",
    }:
        raise ValueError("PaliGemma tokenizer artifact schema mismatch")
    remote = value["remote"]
    local = value["local"]
    if (
        value["schema_version"] != 1
        or value["status"] != "pass"
        or value["uri"] != PI05_DROID_JOINTPOS_TOKENIZER_URI
        or not isinstance(remote, dict)
        or remote
        != {
            "generation": PI05_DROID_JOINTPOS_TOKENIZER_GENERATION,
            "size": PI05_DROID_JOINTPOS_TOKENIZER_SIZE,
            "md5_base64": PI05_DROID_JOINTPOS_TOKENIZER_MD5_BASE64,
        }
        or not isinstance(local, dict)
        or set(local) != {"path", "size", "md5_base64", "sha256"}
        or not isinstance(local["path"], str)
        or not Path(local["path"]).is_absolute()
        or local["size"] != PI05_DROID_JOINTPOS_TOKENIZER_SIZE
        or local["md5_base64"] != PI05_DROID_JOINTPOS_TOKENIZER_MD5_BASE64
        or local["sha256"] != PI05_DROID_JOINTPOS_TOKENIZER_SHA256
    ):
        raise ValueError("PaliGemma tokenizer artifact identity mismatch")
    return copy.deepcopy(value)


def verify_paligemma_tokenizer_artifact(download_module: object) -> dict[str, Any]:
    """Verify the exact live GCS generation and cache bytes used by OpenPI."""

    try:
        local_path = Path(
            download_module.maybe_download(
                PI05_DROID_JOINTPOS_TOKENIZER_URI, gs={"token": "anon"}
            )
        )
        filesystem, _ = download_module.fsspec.core.url_to_fs(
            PI05_DROID_JOINTPOS_TOKENIZER_URI, token="anon"
        )
        remote_info = filesystem.info(PI05_DROID_JOINTPOS_TOKENIZER_URI)
    except (AttributeError, OSError) as error:
        raise ValueError("Cannot resolve the PaliGemma tokenizer GCS object") from error
    if local_path.is_symlink() or not local_path.is_file():
        raise ValueError("PaliGemma tokenizer cache entry must be one regular file")
    payload = local_path.read_bytes()
    report = {
        "schema_version": 1,
        "status": "pass",
        "uri": PI05_DROID_JOINTPOS_TOKENIZER_URI,
        "remote": {
            "generation": str(remote_info.get("generation")),
            "size": remote_info.get("size"),
            "md5_base64": remote_info.get("md5Hash"),
        },
        "local": {
            "path": str(local_path.resolve()),
            "size": len(payload),
            "md5_base64": _bytes_md5_base64(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
    }
    return validate_paligemma_tokenizer_artifact(report)


def _require_exact(value: Any, expected: Any, message: str) -> None:
    if canonical_json_bytes(value) != canonical_json_bytes(expected):
        raise ValueError(f"{message}: expected {expected!r}, got {value!r}")


def _class_path(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _expected_resize_runtime_probe() -> dict[str, Any]:
    return {
        "transform": "openpi.transforms.ResizeImages",
        "bound_module": "openpi_client.image_tools",
        "bound_function": "openpi_client.image_tools.resize_with_pad",
        "backend": "PIL.Image.resize",
        "method": "PIL.Image.Resampling.BILINEAR",
        "padding": "symmetric_zero",
        "input_shape": [5, 9, 3],
        "input_dtype": "uint8",
        "input_sha256": PI05_DROID_JOINTPOS_RESIZE_PROBE_INPUT_SHA256,
        "target_shape": [224, 224, 3],
        "output_dtype": "uint8",
        "output_sha256": PI05_DROID_JOINTPOS_RESIZE_PROBE_OUTPUT_SHA256,
    }


def _expected_observation_image_conversion() -> dict[str, Any]:
    return {
        "implementation": "openpi.models.model.Observation.from_dict",
        "input_dtype": "uint8",
        "output_dtype": "float32",
        "mapping": "value_div_255_times_2_minus_1",
    }


def _expected_model_preprocess_resize() -> dict[str, Any]:
    return {
        "implementation": "openpi.shared.image_tools.resize_with_pad",
        "backend": "jax.image.resize",
        "active": False,
        "inactivity_condition": "input_spatial_shape_equals_224x224",
    }


def attest_openpi_resize_transform(resize_transform: object) -> dict[str, Any]:
    """Bind ``ResizeImages`` to its live helper and a deterministic byte result."""

    if _class_path(resize_transform) != "openpi.transforms.ResizeImages":
        raise ValueError("OpenPI resize transform class mismatch")
    if (
        getattr(resize_transform, "height", None) != 224
        or getattr(resize_transform, "width", None) != 224
    ):
        raise ValueError("OpenPI resize transform target mismatch")

    call = getattr(type(resize_transform), "__call__", None)
    call_globals = getattr(call, "__globals__", None)
    bound_module = (
        call_globals.get("image_tools") if isinstance(call_globals, dict) else None
    )
    module_name = getattr(bound_module, "__name__", None)
    bound_function = getattr(bound_module, "resize_with_pad", None)
    function_name = (
        f"{getattr(bound_function, '__module__', '')}."
        f"{getattr(bound_function, '__qualname__', '')}"
    )
    if module_name != "openpi_client.image_tools" or function_name != (
        "openpi_client.image_tools.resize_with_pad"
    ):
        raise ValueError(
            "OpenPI ResizeImages helper binding mismatch: "
            f"module={module_name!r}, function={function_name!r}"
        )

    probe = np.arange(5 * 9 * 3, dtype=np.uint8).reshape(5, 9, 3)
    probe_input_sha256 = hashlib.sha256(probe.tobytes()).hexdigest()
    payload = {"image": {"probe_rgb": probe.copy()}}
    try:
        transformed = resize_transform(payload)
        output = np.ascontiguousarray(transformed["image"]["probe_rgb"])
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ValueError("OpenPI ResizeImages behavior probe failed") from error
    report = {
        "transform": _class_path(resize_transform),
        "bound_module": module_name,
        "bound_function": function_name,
        "backend": "PIL.Image.resize",
        "method": "PIL.Image.Resampling.BILINEAR",
        "padding": "symmetric_zero",
        "input_shape": list(probe.shape),
        "input_dtype": str(probe.dtype),
        "input_sha256": probe_input_sha256,
        "target_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "output_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
    }
    _require_exact(
        report,
        _expected_resize_runtime_probe(),
        "OpenPI ResizeImages deterministic behavior mismatch",
    )
    return report


def pi05_droid_jointpos_server_contract_sha256(contract: dict[str, Any]) -> str:
    """Return the identity of a contract without trusting its identity field."""

    if not isinstance(contract, dict):
        raise ValueError("pi0.5 joint-position serving contract must be an object")
    payload = copy.deepcopy(contract)
    payload.pop("contract_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _runtime_attestation_sha256(attestation: dict[str, Any]) -> str:
    payload = copy.deepcopy(attestation)
    payload.pop("attestation_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def validate_openpi_runtime_attestation(value: Any) -> dict[str, Any]:
    """Validate a path-independent record of every imported OpenPI module."""

    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "git_commit",
        "import_roots",
        "modules",
        "namespace_packages",
        "attestation_sha256",
    }:
        raise ValueError("OpenPI runtime attestation schema mismatch")
    if value["schema_version"] != 1:
        raise ValueError("OpenPI runtime attestation version mismatch")
    if value["git_commit"] != PI05_DROID_JOINTPOS_OPENPI_COMMIT:
        raise ValueError("OpenPI runtime attestation commit mismatch")
    if value["import_roots"] != ["packages/openpi-client/src", "src"]:
        raise ValueError("OpenPI runtime import roots mismatch")
    modules = value["modules"]
    if (
        not isinstance(modules, list)
        or not modules
        or modules != sorted(modules, key=lambda item: item.get("module", ""))
    ):
        raise ValueError("OpenPI module records must be one nonempty sorted list")
    seen_modules: set[str] = set()
    seen_paths: set[str] = set()
    for index, record in enumerate(modules):
        if not isinstance(record, dict) or set(record) != {
            "module",
            "relative_path",
            "sha256",
        }:
            raise ValueError(f"OpenPI module record {index} schema mismatch")
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
            raise ValueError(f"Invalid OpenPI module name at record {index}")
        if (
            not isinstance(relative_path, str)
            or relative_path.startswith(("/", "../"))
            or "/../" in relative_path
            or not relative_path.endswith(".py")
            or not relative_path.startswith(
                ("src/openpi/", "packages/openpi-client/src/openpi_client/")
            )
            or relative_path in seen_paths
        ):
            raise ValueError(f"Invalid OpenPI module origin at record {index}")
        expected_module = _module_name_from_relative_path(relative_path)
        if module != expected_module:
            raise ValueError(f"OpenPI module/origin mismatch at record {index}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"Invalid OpenPI source digest at record {index}")
        seen_modules.add(module)
        seen_paths.add(relative_path)
    if not _REQUIRED_OPENPI_MODULES.issubset(seen_modules):
        raise ValueError(
            "OpenPI runtime is missing execution-critical modules: "
            f"{sorted(_REQUIRED_OPENPI_MODULES - seen_modules)}"
        )
    namespaces = value["namespace_packages"]
    if not isinstance(namespaces, list) or namespaces != sorted(
        namespaces, key=lambda item: item.get("module", "")
    ):
        raise ValueError("OpenPI namespace records must be sorted")
    seen_namespaces: set[str] = set()
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
            or module in seen_namespaces
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
        seen_namespaces.add(module)
    if value["attestation_sha256"] != _runtime_attestation_sha256(value):
        raise ValueError("OpenPI runtime attestation SHA-256 mismatch")
    return copy.deepcopy(value)


def expected_pi05_droid_jointpos_server_contract(
    openpi_runtime_attestation: dict[str, Any],
) -> dict[str, Any]:
    """Build the one accepted, path-independent WebSocket handshake."""

    runtime = validate_openpi_runtime_attestation(openpi_runtime_attestation)
    contract: dict[str, Any] = {
        "schema_version": 1,
        "profile": PI05_DROID_JOINTPOS_PROFILE,
        "checkpoint": {
            "uri": PI05_DROID_JOINTPOS_CHECKPOINT_URI,
            "manifest_profile": "gcs_path_size_md5_v1",
            "manifest_sha256": PI05_DROID_JOINTPOS_MANIFEST_SHA256,
            "object_count": PI05_DROID_JOINTPOS_OBJECT_COUNT,
            "total_bytes": PI05_DROID_JOINTPOS_CHECKPOINT_BYTES,
            "full_md5_verified_before_serve": True,
        },
        "normalization": {
            "asset_id": "droid",
            "scope": "checkpoint_global_droid",
            "path": "assets/droid/norm_stats.json",
            "sha256": PI05_DROID_JOINTPOS_NORM_SHA256,
            "canonical_values_sha256": PI05_DROID_JOINTPOS_NORM_VALUES_SHA256,
            "use_quantile_norm": True,
            "formula": "q01_q99_epsilon1e-6_to_minus1_plus1_v1",
            "state_stats_width": 32,
            "action_stats_width": 32,
            "category_override": "forbidden",
            "rejected_category_substitutions": [
                "single_arm",
                "single-arm",
                "single arm",
            ],
        },
        "tokenizer": {
            "uri": PI05_DROID_JOINTPOS_TOKENIZER_URI,
            "generation": PI05_DROID_JOINTPOS_TOKENIZER_GENERATION,
            "size": PI05_DROID_JOINTPOS_TOKENIZER_SIZE,
            "md5_base64": PI05_DROID_JOINTPOS_TOKENIZER_MD5_BASE64,
            "sha256": PI05_DROID_JOINTPOS_TOKENIZER_SHA256,
            "active_wrapper": "openpi.models.tokenizer.PaligemmaTokenizer",
            "fast_wrapper_active": False,
            "standard_sentencepiece_attribute": "_tokenizer",
            "fast_sentencepiece_attribute": "_paligemma_tokenizer",
            "serialized_model_proto_sha256": (PI05_DROID_JOINTPOS_TOKENIZER_SHA256),
        },
        "openpi": {
            "git_commit": PI05_DROID_JOINTPOS_OPENPI_COMMIT,
            "config": PI05_DROID_JOINTPOS_CONFIG,
            "model_class": "openpi.models.pi0.Pi0",
            "model_type": "pi05",
            "objective": "flow_matching",
            "compute_dtype": "bfloat16",
            "action_horizon": 15,
            "action_dim": 32,
            "sampler": {
                "algorithm": "euler_t1_to_t0",
                "num_steps": 10,
                "source": "default_sample_actions_argument",
                "sample_kwargs": {},
                "initial_jax_key_data": [0, 0],
            },
            "runtime_attestation": runtime,
        },
        "transform_pipeline": {
            "input_order": [
                "openpi.transforms.InjectDefaultPrompt",
                "openpi.policies.droid_policy.DroidInputs",
                "openpi.transforms.DeltaActions",
                "openpi.transforms.Normalize",
                "openpi.transforms.InjectDefaultPrompt",
                "openpi.transforms.ResizeImages",
                "openpi.transforms.TokenizePrompt",
                "openpi.transforms.PadStatesAndActions",
            ],
            "output_order": [
                "openpi.transforms.Unnormalize",
                "openpi.transforms.AbsoluteActions",
                "openpi.policies.droid_policy.DroidOutputs",
            ],
            "delta_action_mask": list(PI05_DROID_JOINTPOS_DELTA_MASK),
            "absolute_action_mask": list(PI05_DROID_JOINTPOS_DELTA_MASK),
            "droid_input_model_type": "pi05",
            "policy_default_prompt": None,
            "model_default_prompt": None,
            "resize": [224, 224],
            "tokenizer": "openpi.models.tokenizer.PaligemmaTokenizer",
            "tokenizer_max_length": 200,
            "discrete_state_input": True,
            "pad_state_and_action_width": 32,
        },
        "policy_input": {
            "request_keys": [
                "observation/exterior_image_1_left",
                "observation/wrist_image_left",
                "observation/joint_position",
                "observation/gripper_position",
                "prompt",
            ],
            "state": "7_joint_positions_radians_then_closed_positive_gripper",
            "state_width_before_padding": 8,
            "request_image_shape": [720, 1280, 3],
            "request_image_dtype": "uint8",
            "client_model_spatial_transform": None,
            "images": [
                {
                    "request": "observation/exterior_image_1_left",
                    "model_slot": "base_0_rgb",
                    "request_shape": [720, 1280, 3],
                    "request_dtype": "uint8",
                    "mask": True,
                },
                {
                    "request": "observation/wrist_image_left",
                    "model_slot": "left_wrist_0_rgb",
                    "request_shape": [720, 1280, 3],
                    "request_dtype": "uint8",
                    "mask": True,
                },
                {
                    "request": None,
                    "model_slot": "right_wrist_0_rgb",
                    "source": "DroidInputs_zeros_like_native_base_before_resize",
                    "mask": False,
                },
            ],
            "server_resize": {
                "transform": "openpi.transforms.ResizeImages",
                "implementation": "openpi_client.image_tools.resize_with_pad",
                "backend": "PIL.Image.resize",
                "method": "PIL.Image.Resampling.BILINEAR",
                "padding": "symmetric_zero",
                "target_shape": [224, 224, 3],
                "output_dtype": "uint8",
                "application_count": 1,
                "runtime_probe": _expected_resize_runtime_probe(),
                "observation_conversion": _expected_observation_image_conversion(),
                "model_preprocess_resize": _expected_model_preprocess_resize(),
            },
            "model_image_shape": [224, 224, 3],
            "client_visualization_resize": (
                "openpi_client.image_tools.resize_with_pad_PIL_bilinear_non_model_only"
            ),
            "client_wrist_rotation_degrees": 0,
        },
        "policy_output": {
            "model_shape": [15, 32],
            "absolute_reconstruction": "current_state_added_to_first_7_only",
            "projection": "DroidOutputs_leading_dimensions_0_through_7",
            "response_shape": [15, 8],
            "execute_first": 8,
            "simulator_semantics": "absolute_joint_position_targets",
        },
        "serving": {
            "implementation": (
                "openpi.serving.websocket_policy_server.WebsocketPolicyServer"
            ),
            "lifecycle": "asyncio_task_of_official_WebsocketPolicyServer.run",
            "bind_host": PI05_DROID_JOINTPOS_BIND_HOST,
            "network_scope": "ipv4_loopback_only",
            "metadata_empty": False,
            "policy_wrapper": None,
            "rng_stream": {
                "initial_key_data": [0, 0],
                "recurrence": ("stored_key_next_equals_jax_random_split_stored_key_0"),
                "serialization": (
                    "official_server_synchronous_policy_infer_single_event_loop"
                ),
                "quiescence_barrier": (
                    "cancel_official_run_then_async_context_close_wait_closed"
                ),
                "request_observation": (
                    "stored_key_after_listener_close_and_all_handlers_returned"
                ),
                "finalization_signal": "SIGUSR1",
                "final_artifact": PI05_DROID_JOINTPOS_RNG_STREAM_FILENAME,
                "policy_infer_wrapper": None,
            },
        },
    }
    contract["contract_sha256"] = pi05_droid_jointpos_server_contract_sha256(contract)
    return contract


def expected_pi05_droid_jointpos_server_metadata(
    openpi_runtime_attestation: dict[str, Any],
) -> dict[str, Any]:
    return {
        PI05_DROID_JOINTPOS_METADATA_KEY: (
            expected_pi05_droid_jointpos_server_contract(openpi_runtime_attestation)
        )
    }


def validate_pi05_droid_jointpos_server_metadata(
    metadata: Any,
) -> dict[str, Any]:
    """Validate exact live metadata and return a private canonical contract."""

    if not isinstance(metadata, dict) or set(metadata) != {
        PI05_DROID_JOINTPOS_METADATA_KEY
    }:
        raise ValueError("pi0.5 joint-position server metadata schema mismatch")
    contract = metadata[PI05_DROID_JOINTPOS_METADATA_KEY]
    if not isinstance(contract, dict):
        raise ValueError("pi0.5 joint-position server contract must be an object")
    if contract.get("contract_sha256") != (
        pi05_droid_jointpos_server_contract_sha256(contract)
    ):
        raise ValueError("pi0.5 joint-position server contract SHA-256 mismatch")
    runtime = validate_openpi_runtime_attestation(
        contract.get("openpi", {}).get("runtime_attestation")
    )
    expected = expected_pi05_droid_jointpos_server_metadata(runtime)
    _require_exact(metadata, expected, "pi0.5 joint-position server contract mismatch")
    return copy.deepcopy(contract)


def verify_openpi_git_checkout(openpi_dir: Path) -> dict[str, Any]:
    """Require an exact, clean, non-symlinked OpenPI checkout."""

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
            raise ValueError(f"Cannot inspect OpenPI checkout: {root}") from error

    if Path(git("rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("--openpi-dir must name the exact checkout root")
    head = git("rev-parse", "HEAD")
    if head != PI05_DROID_JOINTPOS_OPENPI_COMMIT:
        raise ValueError(
            "OpenPI commit mismatch: "
            f"expected {PI05_DROID_JOINTPOS_OPENPI_COMMIT}, got {head}"
        )
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("OpenPI checkout must be completely clean")
    return {
        "root": str(root),
        "git_commit": head,
        "tracked_and_untracked_clean": True,
    }


def _regular_file_identity(
    path: Path, *, expected_sha256: str | None = None
) -> dict[str, Any]:
    requested = Path(path)
    if requested.is_symlink() or not requested.is_file():
        raise ValueError(f"Host runtime file must be one regular file: {requested}")
    payload = requested.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Host runtime file SHA-256 mismatch: {requested}")
    return {
        "path": str(requested.resolve()),
        "size": len(payload),
        "sha256": digest,
    }


def _normalized_distribution_name(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def _verify_distribution_record(
    distribution_name: str, environment_root: Path
) -> dict[str, Any]:
    """Verify every installed wheel file against its RECORD hash."""

    distribution = importlib_metadata.distribution(distribution_name)
    files = list(distribution.files or ())
    if not files:
        raise ValueError(f"Installed distribution has no RECORD: {distribution_name}")
    identities = []
    unhashed = []
    for package_path in files:
        relative_path = str(package_path)
        expected_hash = package_path.hash
        if expected_hash is None:
            unhashed.append(relative_path)
            continue
        if expected_hash.mode != "sha256":
            raise ValueError(
                f"Installed distribution uses a non-SHA256 RECORD: {distribution_name}"
            )
        raw_path = Path(distribution.locate_file(package_path))
        if raw_path.is_symlink() or not raw_path.is_file():
            raise ValueError(
                f"Installed distribution RECORD file is unsafe: {raw_path}"
            )
        resolved = raw_path.resolve()
        try:
            resolved.relative_to(environment_root)
        except ValueError as error:
            raise ValueError(
                f"Installed distribution escaped the OpenPI environment: {raw_path}"
            ) from error
        payload = raw_path.read_bytes()
        digest = hashlib.sha256(payload).digest()
        record_digest = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        if record_digest != expected_hash.value:
            raise ValueError(
                f"Installed distribution RECORD mismatch: {distribution_name} "
                f"{relative_path}"
            )
        identities.append(
            {
                "path": relative_path,
                "size": len(payload),
                "sha256": digest.hex(),
            }
        )
    if len(unhashed) != 1 or not unhashed[0].endswith(".dist-info/RECORD"):
        raise ValueError(
            f"Installed distribution has unexpected unhashed RECORD entries: "
            f"{distribution_name} {unhashed}"
        )
    identities.sort(key=lambda item: item["path"])
    record_path = Path(distribution.locate_file(unhashed[0]))
    try:
        record_path.resolve().relative_to(environment_root)
    except ValueError as error:
        raise ValueError(
            f"Installed distribution RECORD escaped the OpenPI environment: "
            f"{record_path}"
        ) from error
    return {
        "name": distribution_name,
        "version": distribution.version,
        "file_count": len(files),
        "hashed_file_count": len(identities),
        "record": _regular_file_identity(record_path),
        "verified_files_sha256": hashlib.sha256(
            canonical_json_bytes(identities)
        ).hexdigest(),
    }


def verify_openpi_package_environment(openpi_dir: Path) -> dict[str, Any]:
    """Verify every noneditable installed file against RECORD and uv.lock."""

    checkout = verify_openpi_git_checkout(openpi_dir)
    root = Path(checkout["root"])
    declared_executable = Path(sys.executable)
    if declared_executable != root / ".venv/bin/python":
        raise ValueError(
            "Package verification requires the checkout-local OpenPI interpreter"
        )
    environment_root = Path(sys.prefix).resolve()
    site_packages = environment_root / "lib/python3.11/site-packages"
    if not site_packages.is_dir():
        raise ValueError("OpenPI site-packages directory is missing")
    distributions = []
    for distribution in importlib_metadata.distributions():
        name = distribution.metadata.get("Name")
        if not isinstance(name, str) or not name:
            raise ValueError("Installed Python distribution has no canonical name")
        distributions.append(
            {
                "name": _normalized_distribution_name(name),
                "version": distribution.version,
            }
        )
    distributions.sort(key=lambda item: (item["name"], item["version"]))
    if len({item["name"] for item in distributions}) != len(distributions):
        raise ValueError("Installed Python distribution names are not unique")
    installed_versions = {item["name"]: item["version"] for item in distributions}
    required_versions = {
        name: installed_versions.get(name)
        for name in PI05_DROID_JOINTPOS_REQUIRED_PACKAGE_VERSIONS
    }
    if required_versions != PI05_DROID_JOINTPOS_REQUIRED_PACKAGE_VERSIONS:
        raise ValueError(
            "OpenPI installed package versions differ from the locked inference set"
        )
    try:
        lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
        locked_versions: dict[str, set[str]] = {}
        for package in lock["package"]:
            name = _normalized_distribution_name(package["name"])
            locked_versions.setdefault(name, set()).add(package["version"])
    except (
        OSError,
        UnicodeDecodeError,
        tomllib.TOMLDecodeError,
        KeyError,
        TypeError,
    ) as error:
        raise ValueError("Cannot parse the attested OpenPI uv.lock") from error
    unlocked_distributions = [
        item
        for item in distributions
        if item["name"] not in locked_versions
        or item["version"] not in locked_versions[item["name"]]
    ]
    if unlocked_distributions:
        raise ValueError(
            "Installed OpenPI distributions differ from uv.lock: "
            f"{unlocked_distributions}"
        )
    record_verified_distributions = [
        _verify_distribution_record(item["name"], environment_root)
        for item in distributions
        if item["name"] not in PI05_DROID_JOINTPOS_RECORD_VERIFICATION_EXEMPTIONS
    ]
    record_verification_exemptions = [
        {
            "name": name,
            "version": installed_versions[name],
            "reason": "source_attested_editable_openpi_checkout",
        }
        for name in PI05_DROID_JOINTPOS_RECORD_VERIFICATION_EXEMPTIONS
    ]
    return {
        "required_versions": required_versions,
        "all_installed_versions_allowed_by_uv_lock": True,
        "all_noneditable_record_hashes_verified": True,
        "record_verified_distributions": record_verified_distributions,
        "record_verification_exemptions": record_verification_exemptions,
        "installed_distributions": distributions,
        "installed_distributions_sha256": hashlib.sha256(
            canonical_json_bytes(distributions)
        ).hexdigest(),
    }


def capture_openpi_host_runtime(openpi_dir: Path, jax_module: object) -> dict[str, Any]:
    """Seal the interpreter, lockfiles, package inventory, and live JAX host."""

    checkout = verify_openpi_git_checkout(openpi_dir)
    root = Path(checkout["root"])
    declared_executable = Path(sys.executable)
    expected_executable = root / ".venv/bin/python"
    if declared_executable != expected_executable:
        raise ValueError(
            "Attested server must run with the checkout-local OpenPI interpreter"
        )
    executable_target = declared_executable.resolve()
    executable = _regular_file_identity(executable_target)
    package_environment = verify_openpi_package_environment(root)
    devices = [
        {
            "id": device.id,
            "platform": device.platform,
            "device_kind": device.device_kind,
            "process_index": device.process_index,
        }
        for device in jax_module.devices()
    ]
    if devices != [
        {
            "id": 0,
            "platform": "gpu",
            "device_kind": "NVIDIA L40S",
            "process_index": 0,
        }
    ]:
        raise ValueError(f"Attested JAX device must be one NVIDIA L40S: {devices}")
    required_environment = {
        name: os.environ.get(name)
        for name in PI05_DROID_JOINTPOS_REQUIRED_RUNTIME_ENVIRONMENT
    }
    if required_environment != PI05_DROID_JOINTPOS_REQUIRED_RUNTIME_ENVIRONMENT:
        raise ValueError(
            "OpenPI process environment differs from the attested launch contract"
        )
    optional_environment = {
        name: os.environ.get(name)
        for name in PI05_DROID_JOINTPOS_OPTIONAL_RUNTIME_ENVIRONMENT
    }
    forbidden_environment = {
        name: value
        for name, value in optional_environment.items()
        if name not in PI05_DROID_JOINTPOS_ALLOWED_OPTIONAL_RUNTIME_ENVIRONMENT
        and value is not None
    }
    if forbidden_environment:
        raise ValueError(
            "Semantics-changing JAX/CUDA environment variables must be unset: "
            f"{sorted(forbidden_environment)}"
        )
    jax_config = {
        "default_prng_impl": jax_module.config.jax_default_prng_impl,
        "legacy_prng_key": jax_module.config.jax_legacy_prng_key,
        "threefry_partitionable": jax_module.config.jax_threefry_partitionable,
        "random_seed_offset": jax_module.config.jax_random_seed_offset,
        "default_matmul_precision": jax_module.config.jax_default_matmul_precision,
        "disable_jit": jax_module.config.jax_disable_jit,
        "enable_x64": jax_module.config.jax_enable_x64,
    }
    if jax_config != PI05_DROID_JOINTPOS_JAX_CONFIG:
        raise ValueError(
            f"OpenPI JAX config differs from the official runtime: {jax_config}"
        )
    backend = jax_module.lib.xla_bridge.get_backend()
    report = {
        "schema_version": 1,
        "profile": PI05_DROID_JOINTPOS_HOST_RUNTIME_PROFILE,
        "python": {
            "declared_executable": str(declared_executable),
            "resolved_executable": executable,
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "cache_tag": sys.implementation.cache_tag,
            "prefix": str(Path(sys.prefix).resolve()),
            "base_prefix": str(Path(sys.base_prefix).resolve()),
        },
        "locked_source_environment": {
            "uv_lock": _regular_file_identity(
                root / "uv.lock",
                expected_sha256=PI05_DROID_JOINTPOS_UV_LOCK_SHA256,
            ),
            "pyproject": _regular_file_identity(
                root / "pyproject.toml",
                expected_sha256=PI05_DROID_JOINTPOS_PYPROJECT_SHA256,
            ),
        },
        "packages": package_environment,
        "process_environment": {
            "required": required_environment,
            "optional": optional_environment,
        },
        "jax": {
            "jax_version": importlib_metadata.version("jax"),
            "jaxlib_version": importlib_metadata.version("jaxlib"),
            "jax_cuda12_pjrt_version": importlib_metadata.version("jax-cuda12-pjrt"),
            "jax_cuda12_plugin_version": importlib_metadata.version(
                "jax-cuda12-plugin"
            ),
            "enable_x64": jax_module.config.x64_enabled,
            "config": jax_config,
            "default_backend": jax_module.default_backend(),
            "platform_version": backend.platform_version,
            "devices": devices,
        },
    }
    return validate_openpi_host_runtime(report)


def validate_openpi_host_runtime(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "profile",
        "python",
        "locked_source_environment",
        "packages",
        "process_environment",
        "jax",
    }:
        raise ValueError("OpenPI host-runtime schema mismatch")
    if (
        value["schema_version"] != 1
        or value["profile"] != PI05_DROID_JOINTPOS_HOST_RUNTIME_PROFILE
    ):
        raise ValueError("OpenPI host-runtime identity mismatch")
    python = value["python"]
    if not isinstance(python, dict) or set(python) != {
        "declared_executable",
        "resolved_executable",
        "implementation",
        "version",
        "cache_tag",
        "prefix",
        "base_prefix",
    }:
        raise ValueError("OpenPI Python runtime schema mismatch")
    if (
        not all(
            isinstance(python[name], str) and Path(python[name]).is_absolute()
            for name in ("declared_executable", "prefix", "base_prefix")
        )
        or python["implementation"] != "CPython"
        or not python["version"].startswith("3.11.")
        or python["cache_tag"] != "cpython-311"
    ):
        raise ValueError("OpenPI Python runtime mismatch")
    executable = python["resolved_executable"]
    if not isinstance(executable, dict) or set(executable) != {
        "path",
        "size",
        "sha256",
    }:
        raise ValueError("OpenPI Python executable identity schema mismatch")
    _validate_file_identity_record(executable, "OpenPI Python executable")
    locked = value["locked_source_environment"]
    if not isinstance(locked, dict) or set(locked) != {"uv_lock", "pyproject"}:
        raise ValueError("OpenPI locked source environment schema mismatch")
    for name, expected_digest in (
        ("uv_lock", PI05_DROID_JOINTPOS_UV_LOCK_SHA256),
        ("pyproject", PI05_DROID_JOINTPOS_PYPROJECT_SHA256),
    ):
        _validate_file_identity_record(locked[name], f"OpenPI {name}")
        if locked[name]["sha256"] != expected_digest:
            raise ValueError(f"OpenPI {name} identity mismatch")
    packages = value["packages"]
    if not isinstance(packages, dict) or set(packages) != {
        "required_versions",
        "all_installed_versions_allowed_by_uv_lock",
        "all_noneditable_record_hashes_verified",
        "record_verified_distributions",
        "record_verification_exemptions",
        "installed_distributions",
        "installed_distributions_sha256",
    }:
        raise ValueError("OpenPI package inventory schema mismatch")
    if (
        packages["required_versions"] != PI05_DROID_JOINTPOS_REQUIRED_PACKAGE_VERSIONS
        or packages["all_installed_versions_allowed_by_uv_lock"] is not True
        or packages["all_noneditable_record_hashes_verified"] is not True
    ):
        raise ValueError("OpenPI required package versions mismatch")
    distributions = packages["installed_distributions"]
    if (
        not isinstance(distributions, list)
        or not distributions
        or distributions
        != sorted(
            distributions,
            key=lambda item: (item.get("name", ""), item.get("version", "")),
        )
        or any(
            not isinstance(item, dict)
            or set(item) != {"name", "version"}
            or not all(isinstance(item[field], str) and item[field] for field in item)
            for item in distributions
        )
        or len({item["name"] for item in distributions}) != len(distributions)
    ):
        raise ValueError("OpenPI installed package inventory mismatch")
    installed = {item["name"]: item["version"] for item in distributions}
    record_verified = packages["record_verified_distributions"]
    expected_record_names = [
        item["name"]
        for item in packages["installed_distributions"]
        if item["name"] not in PI05_DROID_JOINTPOS_RECORD_VERIFICATION_EXEMPTIONS
    ]
    if (
        not isinstance(record_verified, list)
        or not all(isinstance(item, dict) for item in record_verified)
        or [item.get("name") for item in record_verified] != expected_record_names
    ):
        raise ValueError("OpenPI RECORD-verified package inventory mismatch")
    for item in record_verified:
        if (
            set(item)
            != {
                "name",
                "version",
                "file_count",
                "hashed_file_count",
                "record",
                "verified_files_sha256",
            }
            or item["version"] != installed.get(item["name"])
            or type(item["file_count"]) is not int
            or type(item["hashed_file_count"]) is not int
            or item["file_count"] <= 1
            or item["hashed_file_count"] != item["file_count"] - 1
        ):
            raise ValueError("OpenPI RECORD-verified package identity mismatch")
        _validate_file_identity_record(item["record"], f"{item['name']} RECORD")
        digest = item["verified_files_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("OpenPI verified package digest mismatch")
    exemptions = packages["record_verification_exemptions"]
    expected_exemptions = [
        {
            "name": name,
            "version": installed.get(name),
            "reason": "source_attested_editable_openpi_checkout",
        }
        for name in PI05_DROID_JOINTPOS_RECORD_VERIFICATION_EXEMPTIONS
    ]
    if exemptions != expected_exemptions:
        raise ValueError("OpenPI RECORD verification exemptions mismatch")
    if (
        packages["installed_distributions_sha256"]
        != hashlib.sha256(canonical_json_bytes(distributions)).hexdigest()
    ):
        raise ValueError("OpenPI installed package inventory SHA-256 mismatch")
    if any(
        installed.get(name) != version
        for name, version in PI05_DROID_JOINTPOS_REQUIRED_PACKAGE_VERSIONS.items()
    ):
        raise ValueError("OpenPI installed package lock mismatch")
    process_environment = value["process_environment"]
    if not isinstance(process_environment, dict) or set(process_environment) != {
        "required",
        "optional",
    }:
        raise ValueError("OpenPI process-environment schema mismatch")
    if (
        process_environment["required"]
        != PI05_DROID_JOINTPOS_REQUIRED_RUNTIME_ENVIRONMENT
        or not isinstance(process_environment["optional"], dict)
        or set(process_environment["optional"])
        != set(PI05_DROID_JOINTPOS_OPTIONAL_RUNTIME_ENVIRONMENT)
        or any(
            item is not None and not isinstance(item, str)
            for item in process_environment["optional"].values()
        )
        or any(
            value is not None
            for name, value in process_environment["optional"].items()
            if name not in PI05_DROID_JOINTPOS_ALLOWED_OPTIONAL_RUNTIME_ENVIRONMENT
        )
    ):
        raise ValueError("OpenPI process environment mismatch")
    jax = value["jax"]
    if not isinstance(jax, dict) or set(jax) != {
        "jax_version",
        "jaxlib_version",
        "jax_cuda12_pjrt_version",
        "jax_cuda12_plugin_version",
        "enable_x64",
        "config",
        "default_backend",
        "platform_version",
        "devices",
    }:
        raise ValueError("OpenPI JAX runtime schema mismatch")
    if (
        jax["jax_version"] != "0.5.3"
        or jax["jaxlib_version"] != "0.5.3"
        or jax["jax_cuda12_pjrt_version"] != "0.5.3"
        or jax["jax_cuda12_plugin_version"] != "0.5.3"
        or jax["enable_x64"] is not False
        or jax["config"] != PI05_DROID_JOINTPOS_JAX_CONFIG
        or jax["default_backend"] != "gpu"
        or not isinstance(jax["platform_version"], str)
        or not jax["platform_version"]
        or jax["devices"]
        != [
            {
                "id": 0,
                "platform": "gpu",
                "device_kind": "NVIDIA L40S",
                "process_index": 0,
            }
        ]
    ):
        raise ValueError("OpenPI JAX runtime mismatch")
    return copy.deepcopy(value)


def _validate_file_identity_record(value: Any, field: str) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != {"path", "size", "sha256"}
        or not isinstance(value["path"], str)
        or not Path(value["path"]).is_absolute()
        or type(value["size"]) is not int
        or value["size"] <= 0
        or not isinstance(value["sha256"], str)
        or len(value["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in value["sha256"])
    ):
        raise ValueError(f"{field} identity mismatch")


def validate_pi05_droid_jointpos_loopback_listener(
    server_pid: int, port: int
) -> dict[str, Any]:
    """Attest the live Linux listener and its ownership by the server process."""

    if type(server_pid) is not int or server_pid <= 0:
        raise ValueError("Policy-server PID must be one positive integer")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Policy-server port must be in [1, 65535]")
    process_root = Path(f"/proc/{server_pid}")
    fd_root = process_root / "fd"
    try:
        descriptors = list(fd_root.iterdir())
    except (FileNotFoundError, PermissionError, ProcessLookupError) as error:
        raise ValueError("Cannot inspect policy-server socket ownership") from error
    process_socket_inodes = set()
    for descriptor in descriptors:
        try:
            target = os.readlink(descriptor)
        except FileNotFoundError:
            continue
        except (PermissionError, ProcessLookupError) as error:
            raise ValueError("Cannot inspect policy-server socket ownership") from error
        if target.startswith("socket:[") and target.endswith("]"):
            process_socket_inodes.add(target.removeprefix("socket:[").removesuffix("]"))

    listeners: list[dict[str, str]] = []
    for family, table_name in (("ipv4", "tcp"), ("ipv6", "tcp6")):
        table_path = process_root / "net" / table_name
        try:
            lines = table_path.read_text(encoding="ascii").splitlines()[1:]
        except (OSError, UnicodeDecodeError) as error:
            raise ValueError(
                "Cannot inspect policy-server network namespace"
            ) from error
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":
                continue
            local_address, local_port = fields[1].split(":", 1)
            if int(local_port, 16) != port:
                continue
            listeners.append(
                {
                    "family": family,
                    "address_hex": local_address,
                    "inode": fields[9],
                }
            )
    if len(listeners) != 1:
        raise ValueError(
            f"Expected one policy listener on port {port}, got {listeners}"
        )
    listener = listeners[0]
    if (
        listener["family"] != "ipv4"
        or listener["address_hex"] != "0100007F"
        or listener["inode"] not in process_socket_inodes
    ):
        raise ValueError("Policy listener is not an owned IPv4 127.0.0.1-only socket")
    return {
        "schema_version": 1,
        "profile": "openpi_pi05_droid_jointpos_live_listener_v1",
        "status": "pass",
        "server_pid": server_pid,
        "bind_host": PI05_DROID_JOINTPOS_BIND_HOST,
        "port": port,
        "family": "ipv4",
        "network_scope": "ipv4_loopback_only",
        "socket_inode": listener["inode"],
    }


def _module_name_from_relative_path(relative_path: str) -> str:
    prefixes = ("src/", "packages/openpi-client/src/")
    for prefix in prefixes:
        if relative_path.startswith(prefix):
            module = relative_path.removeprefix(prefix).removesuffix(".py")
            return module.removesuffix("/__init__").replace("/", ".")
    raise ValueError(f"OpenPI source is outside controlled roots: {relative_path}")


def _source_path(module_file: str) -> Path:
    path = Path(module_file)
    if path.suffix in {".pyc", ".pyo"}:
        try:
            path = Path(importlib.util.source_from_cache(str(path)))
        except ValueError as error:
            raise ValueError(
                f"Cannot resolve OpenPI bytecode source: {path}"
            ) from error
    return path


def attest_imported_openpi_modules(openpi_dir: Path) -> dict[str, Any]:
    """Bind all imported OpenPI modules to source bytes in one exact checkout."""

    checkout = verify_openpi_git_checkout(openpi_dir)
    root = Path(checkout["root"])
    records: list[dict[str, str]] = []
    namespaces: list[dict[str, Any]] = []
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
                raise ValueError(f"OpenPI module has no source origin: {module_name}")
            relative_paths = []
            for location in search_locations:
                raw_path = Path(location)
                if raw_path.is_symlink():
                    raise ValueError(f"OpenPI namespace is a symlink: {module_name}")
                resolved = raw_path.resolve()
                try:
                    relative = resolved.relative_to(root).as_posix()
                except ValueError as error:
                    raise ValueError(
                        f"OpenPI namespace escaped checkout: {module_name}"
                    ) from error
                relative_paths.append(relative)
            namespaces.append(
                {
                    "module": module_name,
                    "relative_paths": sorted(set(relative_paths)),
                }
            )
            continue
        raw_path = _source_path(module_file)
        if raw_path.is_symlink():
            raise ValueError(f"OpenPI source is a symlink: {module_name}")
        source = raw_path.resolve()
        if not source.is_file():
            raise ValueError(f"OpenPI source is missing: {source}")
        try:
            relative_path = source.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(
                f"OpenPI module escaped checkout: {module_name}"
            ) from error
        if _module_name_from_relative_path(relative_path) != module_name:
            raise ValueError(f"OpenPI module origin mismatch: {module_name}")
        records.append(
            {
                "module": module_name,
                "relative_path": relative_path,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        )
    attestation: dict[str, Any] = {
        "schema_version": 1,
        "git_commit": PI05_DROID_JOINTPOS_OPENPI_COMMIT,
        "import_roots": ["packages/openpi-client/src", "src"],
        "modules": sorted(records, key=lambda item: item["module"]),
        "namespace_packages": sorted(namespaces, key=lambda item: item["module"]),
    }
    attestation["attestation_sha256"] = _runtime_attestation_sha256(attestation)
    return validate_openpi_runtime_attestation(attestation)


def _manifest_entries(manifest: Path) -> list[tuple[str, int, str]]:
    requested = Path(manifest)
    if requested.is_symlink() or not requested.is_file():
        raise ValueError("Checkpoint manifest must be one regular file")
    payload = requested.read_bytes()
    if hashlib.sha256(payload).hexdigest() != PI05_DROID_JOINTPOS_MANIFEST_SHA256:
        raise ValueError("Checkpoint manifest SHA-256 mismatch")
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("Checkpoint manifest must be ASCII") from error
    entries: list[tuple[str, int, str]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        fields = line.split("\t")
        if len(fields) != 3 or not fields[0].startswith(
            PI05_DROID_JOINTPOS_MANIFEST_PREFIX
        ):
            raise ValueError(f"Invalid checkpoint manifest line {line_number}")
        relative_path = fields[0][len(PI05_DROID_JOINTPOS_MANIFEST_PREFIX) :]
        relative = Path(relative_path)
        try:
            size = int(fields[1])
            base64.b64decode(fields[2], validate=True)
        except (ValueError, TypeError) as error:
            raise ValueError(
                f"Invalid checkpoint manifest line {line_number}"
            ) from error
        if (
            not relative_path
            or relative.is_absolute()
            or ".." in relative.parts
            or relative_path in seen
            or size < 0
        ):
            raise ValueError(f"Unsafe checkpoint manifest line {line_number}")
        seen.add(relative_path)
        entries.append((relative_path, size, fields[2]))
    if (
        len(entries) != PI05_DROID_JOINTPOS_OBJECT_COUNT
        or sum(size for _, size, _ in entries) != PI05_DROID_JOINTPOS_CHECKPOINT_BYTES
    ):
        raise ValueError("Checkpoint manifest inventory mismatch")
    return entries


def _md5_base64(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return base64.b64encode(digest.digest()).decode("ascii")


def _strict_norm_stats(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Checkpoint norm stats must be one regular file")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != PI05_DROID_JOINTPOS_NORM_SHA256:
        raise ValueError(f"Checkpoint norm-stats SHA-256 mismatch: {digest}")

    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite norm-stats constant is forbidden: {value}")

    try:
        value = json.loads(payload, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Checkpoint norm stats are not strict JSON") from error
    if not isinstance(value, dict) or set(value) != {"norm_stats"}:
        raise ValueError("Checkpoint norm-stats root schema mismatch")
    groups = value["norm_stats"]
    if not isinstance(groups, dict) or set(groups) != {"actions", "state"}:
        raise ValueError("Checkpoint norm-stats group schema mismatch")
    for group_name in ("actions", "state"):
        group = groups[group_name]
        if not isinstance(group, dict) or set(group) != {
            "mean",
            "std",
            "q01",
            "q99",
        }:
            raise ValueError(f"Checkpoint {group_name} stats schema mismatch")
        for statistic in ("mean", "std", "q01", "q99"):
            vector = group[statistic]
            if (
                not isinstance(vector, list)
                or len(vector) != 32
                or any(
                    type(item) not in (int, float)
                    or isinstance(item, bool)
                    or not math.isfinite(item)
                    for item in vector
                )
            ):
                raise ValueError(f"Checkpoint {group_name} {statistic} vector mismatch")
    canonical_values = {
        group_name: {
            statistic: [float(item) for item in groups[group_name][statistic]]
            for statistic in ("mean", "std", "q01", "q99")
        }
        for group_name in ("actions", "state")
    }
    values_sha256 = hashlib.sha256(canonical_json_bytes(canonical_values)).hexdigest()
    if values_sha256 != PI05_DROID_JOINTPOS_NORM_VALUES_SHA256:
        raise ValueError(
            "Checkpoint norm-stats canonical numeric values SHA-256 mismatch"
        )
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "values_sha256": values_sha256,
        "asset_id": "droid",
        "scope": "checkpoint_global_droid",
        "state_width": 32,
        "action_width": 32,
    }


def verify_pi05_droid_jointpos_checkpoint(
    checkpoint_dir: Path, manifest: Path
) -> dict[str, Any]:
    """Verify the full released checkpoint object set, sizes, and every MD5."""

    requested = Path(checkpoint_dir)
    if requested.is_symlink():
        raise ValueError("Checkpoint root must not be a symlink")
    root = requested.resolve()
    if not root.is_dir():
        raise ValueError("Checkpoint root is missing")
    entries = _manifest_entries(Path(manifest))
    expected_paths = {relative for relative, _, _ in entries}
    actual_paths: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Checkpoint contains a symlink: {path}")
        if path.is_file():
            actual_paths.add(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise ValueError(f"Checkpoint contains a non-file object: {path}")
    if actual_paths != expected_paths:
        raise ValueError(
            "Checkpoint file-set mismatch: "
            f"missing={sorted(expected_paths - actual_paths)}, "
            f"extra={sorted(actual_paths - expected_paths)}"
        )
    objects = []
    for relative_path, expected_size, expected_md5 in entries:
        path = root / relative_path
        file_stat = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size != expected_size:
            raise ValueError(f"Checkpoint object identity mismatch: {relative_path}")
        actual_md5 = _md5_base64(path)
        if actual_md5 != expected_md5:
            raise ValueError(f"Checkpoint object MD5 mismatch: {relative_path}")
        objects.append(
            {
                "relative_path": relative_path,
                "size": expected_size,
                "md5_base64": actual_md5,
            }
        )
    norm = _strict_norm_stats(root / "assets/droid/norm_stats.json")
    return {
        "schema_version": 1,
        "status": "pass",
        "checkpoint_uri": PI05_DROID_JOINTPOS_CHECKPOINT_URI,
        "checkpoint_dir": str(root),
        "manifest_path": str(Path(manifest).resolve()),
        "manifest_sha256": PI05_DROID_JOINTPOS_MANIFEST_SHA256,
        "object_count": len(entries),
        "total_bytes": sum(size for _, size, _ in entries),
        "full_md5": True,
        "objects_sha256": hashlib.sha256(canonical_json_bytes(objects)).hexdigest(),
        "normalization": norm,
    }


def _expected_train_config_report() -> dict[str, Any]:
    return {
        "name": PI05_DROID_JOINTPOS_CONFIG,
        "model_class": "openpi.models.pi0_config.Pi0Config",
        "model_type": "pi05",
        "pi05": True,
        "dtype": "bfloat16",
        "action_horizon": 15,
        "action_dim": 32,
        "max_token_len": 200,
        "discrete_state_input": True,
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "data_factory": "openpi.training.config.RLDSDroidDataConfig",
        "assets_dir": (
            "gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris/assets"
        ),
        "asset_id": "droid",
        "action_space": "JOINT_POSITION",
        "rlds_data_dir": "<path_to_droid_rlds_dataset>",
        "datasets": [
            {
                "name": "droid",
                "version": "1.0.1",
                "weight": 0.9,
                "filter_dict_path": (
                    "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json"
                ),
            },
            {
                "name": "polaris_droid_cotrain_dataset",
                "version": "1.0.0",
                "weight": 0.1,
                "filter_dict_path": (
                    "gs://openpi-assets/droid/"
                    "polaris_droid_cotrain_dataset_sample_ranges_v1_0_0.json"
                ),
            },
        ],
        "weight_loader": "openpi.training.weight_loaders.CheckpointWeightLoader",
        "weight_loader_path": PI05_DROID_JOINTPOS_CHECKPOINT_URI + "/params",
        "policy_metadata": None,
        "pytorch_weight_path": None,
    }


def validate_official_train_config(train_config: object) -> dict[str, Any]:
    """Bind the exact pi0.5/FLOW model and JOINT_POSITION training config."""

    try:
        datasets = [
            {
                "name": item.name,
                "version": item.version,
                "weight": item.weight,
                "filter_dict_path": item.filter_dict_path,
            }
            for item in train_config.data.datasets
        ]
        observed = {
            "name": train_config.name,
            "model_class": _class_path(train_config.model),
            "model_type": train_config.model.model_type.value,
            "pi05": train_config.model.pi05,
            "dtype": train_config.model.dtype,
            "action_horizon": train_config.model.action_horizon,
            "action_dim": train_config.model.action_dim,
            "max_token_len": train_config.model.max_token_len,
            "discrete_state_input": train_config.model.discrete_state_input,
            "paligemma_variant": train_config.model.paligemma_variant,
            "action_expert_variant": train_config.model.action_expert_variant,
            "data_factory": _class_path(train_config.data),
            "assets_dir": train_config.data.assets.assets_dir,
            "asset_id": train_config.data.assets.asset_id,
            "action_space": train_config.data.action_space.name,
            "rlds_data_dir": train_config.data.rlds_data_dir,
            "datasets": datasets,
            "weight_loader": _class_path(train_config.weight_loader),
            "weight_loader_path": train_config.weight_loader.params_path,
            "policy_metadata": train_config.policy_metadata,
            "pytorch_weight_path": train_config.pytorch_weight_path,
        }
    except AttributeError as error:
        raise ValueError(
            "Resolved OpenPI joint-position config is incomplete"
        ) from error
    expected = _expected_train_config_report()
    _require_exact(observed, expected, "Resolved OpenPI train config mismatch")
    return observed


def _transform_paths(values: object, field: str) -> list[str]:
    if type(values) not in (list, tuple):
        raise ValueError(f"OpenPI {field} transform sequence type mismatch")
    return [_class_path(value) for value in values]


def _validate_norm_objects(norm_stats: Any) -> dict[str, Any]:
    if not isinstance(norm_stats, dict) or set(norm_stats) != {"actions", "state"}:
        raise ValueError("OpenPI runtime norm-stats groups mismatch")
    report: dict[str, Any] = {}
    for group_name in ("actions", "state"):
        group = norm_stats[group_name]
        if _class_path(group) != "openpi.shared.normalize.NormStats":
            raise ValueError(f"OpenPI {group_name} norm-stats class mismatch")
        widths = {}
        for statistic in ("mean", "std", "q01", "q99"):
            vector = getattr(group, statistic, None)
            shape = getattr(vector, "shape", None)
            if shape != (32,):
                raise ValueError(f"OpenPI {group_name} {statistic} norm width mismatch")
            widths[statistic] = 32
        report[group_name] = widths
    return report


def _runtime_norm_values_sha256(norm_stats: Any) -> str:
    """Hash loaded NormStats values after canonical float conversion."""

    _validate_norm_objects(norm_stats)
    canonical_values: dict[str, dict[str, list[float]]] = {}
    for group_name in ("actions", "state"):
        group = norm_stats[group_name]
        canonical_values[group_name] = {}
        for statistic in ("mean", "std", "q01", "q99"):
            raw_values = getattr(group, statistic).tolist()
            values = [float(item) for item in raw_values]
            if len(values) != 32 or not all(math.isfinite(item) for item in values):
                raise ValueError(
                    f"OpenPI {group_name} {statistic} norm values mismatch"
                )
            canonical_values[group_name][statistic] = values
    return hashlib.sha256(canonical_json_bytes(canonical_values)).hexdigest()


def _expected_data_config_report() -> dict[str, Any]:
    widths = {name: 32 for name in ("mean", "std", "q01", "q99")}
    return {
        "asset_id": "droid",
        "use_quantile_norm": True,
        "action_space": "JOINT_POSITION",
        "repack_inputs": ["openpi.transforms.RepackTransform"],
        "repack_outputs": [],
        "data_inputs": [
            "openpi.policies.droid_policy.DroidInputs",
            "openpi.transforms.DeltaActions",
        ],
        "data_outputs": [
            "openpi.transforms.AbsoluteActions",
            "openpi.policies.droid_policy.DroidOutputs",
        ],
        "model_inputs": [
            "openpi.transforms.InjectDefaultPrompt",
            "openpi.transforms.ResizeImages",
            "openpi.transforms.TokenizePrompt",
            "openpi.transforms.PadStatesAndActions",
        ],
        "model_outputs": [],
        "sequence_types": {
            "repack_inputs": "list",
            "repack_outputs": "tuple",
            "data_inputs": "tuple",
            "data_outputs": "tuple",
            "model_inputs": "list",
            "model_outputs": "tuple",
        },
        "droid_input_model_type": "pi05",
        "delta_action_mask": list(PI05_DROID_JOINTPOS_DELTA_MASK),
        "absolute_action_mask": list(PI05_DROID_JOINTPOS_DELTA_MASK),
        "droid_outputs_fields": [],
        "model_default_prompt": None,
        "resize": [224, 224],
        "tokenizer": "openpi.models.tokenizer.PaligemmaTokenizer",
        "tokenizer_max_length": 200,
        "tokenizer_runtime": _expected_sentencepiece_identity(
            wrapper_class="openpi.models.tokenizer.PaligemmaTokenizer",
            max_length=200,
            processor_attribute="_tokenizer",
        ),
        "discrete_state_input": True,
        "model_action_dim": 32,
        "normalization": {
            "asset_id": "droid",
            "use_quantiles": True,
            "groups": {
                "actions": dict(widths),
                "state": dict(widths),
            },
        },
        "image_routing": {
            "base_0_rgb": "observation/exterior_image_1_left",
            "left_wrist_0_rgb": "observation/wrist_image_left",
            "right_wrist_0_rgb": "zeros_like_base_mask_false",
        },
        "image_preprocessing": {
            "request_shape": [720, 1280, 3],
            "request_dtype": "uint8",
            "client_model_spatial_transform": None,
            "resize_transform": "openpi.transforms.ResizeImages",
            "resize_implementation": "openpi_client.image_tools.resize_with_pad",
            "resize_backend": "PIL.Image.resize",
            "resize_method": "PIL.Image.Resampling.BILINEAR",
            "padding": "symmetric_zero",
            "model_shape": [224, 224, 3],
            "resize_output_dtype": "uint8",
            "resize_application_count": 1,
            "resize_runtime_probe": _expected_resize_runtime_probe(),
            "observation_conversion": _expected_observation_image_conversion(),
            "model_preprocess_resize": _expected_model_preprocess_resize(),
        },
        "output_projection": "DroidOutputs_leading8",
    }


def validate_official_data_config(data_config: object) -> dict[str, Any]:
    """Bind transform order, parameters, image routing, and output projection."""

    repack = getattr(data_config, "repack_transforms", None)
    data = getattr(data_config, "data_transforms", None)
    model = getattr(data_config, "model_transforms", None)
    if any(group is None for group in (repack, data, model)):
        raise ValueError("OpenPI transform groups are missing")
    observed = {
        "asset_id": getattr(data_config, "asset_id", None),
        "use_quantile_norm": getattr(data_config, "use_quantile_norm", None),
        "action_space": getattr(
            getattr(data_config, "action_space", None), "name", None
        ),
        "repack_inputs": _transform_paths(repack.inputs, "repack inputs"),
        "repack_outputs": _transform_paths(repack.outputs, "repack outputs"),
        "data_inputs": _transform_paths(data.inputs, "data inputs"),
        "data_outputs": _transform_paths(data.outputs, "data outputs"),
        "model_inputs": _transform_paths(model.inputs, "model inputs"),
        "model_outputs": _transform_paths(model.outputs, "model outputs"),
        "sequence_types": {
            "repack_inputs": type(repack.inputs).__name__,
            "repack_outputs": type(repack.outputs).__name__,
            "data_inputs": type(data.inputs).__name__,
            "data_outputs": type(data.outputs).__name__,
            "model_inputs": type(model.inputs).__name__,
            "model_outputs": type(model.outputs).__name__,
        },
    }
    expected = {
        "asset_id": "droid",
        "use_quantile_norm": True,
        "action_space": "JOINT_POSITION",
        "repack_inputs": ["openpi.transforms.RepackTransform"],
        "repack_outputs": [],
        "data_inputs": [
            "openpi.policies.droid_policy.DroidInputs",
            "openpi.transforms.DeltaActions",
        ],
        "data_outputs": [
            "openpi.transforms.AbsoluteActions",
            "openpi.policies.droid_policy.DroidOutputs",
        ],
        "model_inputs": [
            "openpi.transforms.InjectDefaultPrompt",
            "openpi.transforms.ResizeImages",
            "openpi.transforms.TokenizePrompt",
            "openpi.transforms.PadStatesAndActions",
        ],
        "model_outputs": [],
        "sequence_types": {
            "repack_inputs": "list",
            "repack_outputs": "tuple",
            "data_inputs": "tuple",
            "data_outputs": "tuple",
            "model_inputs": "list",
            "model_outputs": "tuple",
        },
    }
    _require_exact(observed, expected, "OpenPI data transform order mismatch")

    repack_transform = repack.inputs[0]
    expected_repack = {
        "observation/exterior_image_1_left": "observation/image",
        "observation/wrist_image_left": "observation/wrist_image",
        "observation/joint_position": "observation/joint_position",
        "observation/gripper_position": "observation/gripper_position",
        "actions": "actions",
        "prompt": "prompt",
    }
    _require_exact(
        getattr(repack_transform, "structure", None),
        expected_repack,
        "OpenPI DROID repack mapping mismatch",
    )
    droid_inputs, delta_actions = data.inputs
    absolute_actions, droid_outputs = data.outputs
    parameter_report = {
        "droid_input_model_type": getattr(
            getattr(droid_inputs, "model_type", None), "value", None
        ),
        "delta_action_mask": list(getattr(delta_actions, "mask", ())),
        "absolute_action_mask": list(getattr(absolute_actions, "mask", ())),
        "droid_outputs_fields": sorted(vars(droid_outputs)),
        "model_default_prompt": getattr(model.inputs[0], "prompt", "missing"),
        "resize": [
            getattr(model.inputs[1], "height", None),
            getattr(model.inputs[1], "width", None),
        ],
        "tokenizer": _class_path(getattr(model.inputs[2], "tokenizer", None)),
        "tokenizer_max_length": getattr(
            getattr(model.inputs[2], "tokenizer", None), "_max_len", None
        ),
        "tokenizer_runtime": attest_loaded_tokenizer_sentencepiece(
            getattr(model.inputs[2], "tokenizer", None)
        ),
        "discrete_state_input": getattr(model.inputs[2], "discrete_state_input", None),
        "model_action_dim": getattr(model.inputs[3], "model_action_dim", None),
    }
    expected_parameters = {
        "droid_input_model_type": "pi05",
        "delta_action_mask": list(PI05_DROID_JOINTPOS_DELTA_MASK),
        "absolute_action_mask": list(PI05_DROID_JOINTPOS_DELTA_MASK),
        "droid_outputs_fields": [],
        "model_default_prompt": None,
        "resize": [224, 224],
        "tokenizer": "openpi.models.tokenizer.PaligemmaTokenizer",
        "tokenizer_max_length": 200,
        "tokenizer_runtime": _expected_sentencepiece_identity(
            wrapper_class="openpi.models.tokenizer.PaligemmaTokenizer",
            max_length=200,
            processor_attribute="_tokenizer",
        ),
        "discrete_state_input": True,
        "model_action_dim": 32,
    }
    _require_exact(
        parameter_report,
        expected_parameters,
        "OpenPI data transform parameters mismatch",
    )
    norm_report = _validate_norm_objects(getattr(data_config, "norm_stats", None))
    resize_runtime_probe = attest_openpi_resize_transform(model.inputs[1])
    report = {
        **observed,
        **parameter_report,
        "normalization": {
            "asset_id": "droid",
            "use_quantiles": True,
            "groups": norm_report,
        },
        "image_routing": {
            "base_0_rgb": "observation/exterior_image_1_left",
            "left_wrist_0_rgb": "observation/wrist_image_left",
            "right_wrist_0_rgb": "zeros_like_base_mask_false",
        },
        "image_preprocessing": {
            "request_shape": [720, 1280, 3],
            "request_dtype": "uint8",
            "client_model_spatial_transform": None,
            "resize_transform": _class_path(model.inputs[1]),
            "resize_implementation": "openpi_client.image_tools.resize_with_pad",
            "resize_backend": "PIL.Image.resize",
            "resize_method": "PIL.Image.Resampling.BILINEAR",
            "padding": "symmetric_zero",
            "model_shape": [224, 224, 3],
            "resize_output_dtype": "uint8",
            "resize_application_count": 1,
            "resize_runtime_probe": resize_runtime_probe,
            "observation_conversion": _expected_observation_image_conversion(),
            "model_preprocess_resize": _expected_model_preprocess_resize(),
        },
        "output_projection": "DroidOutputs_leading8",
    }
    _require_exact(
        report,
        _expected_data_config_report(),
        "OpenPI complete data-config report mismatch",
    )
    return report


def _expected_policy_runtime_report() -> dict[str, Any]:
    return {
        "policy_class": "openpi.policies.policy.Policy",
        "model_class": "openpi.models.pi0.Pi0",
        "model_pi05": True,
        "model_action_horizon": 15,
        "model_action_dim": 32,
        "model_max_token_len": 200,
        "metadata": {},
        "sample_kwargs": {},
        "initial_jax_key_data": [0, 0],
        "jax_enable_x64": False,
        "jax_backend": "gpu",
        "is_pytorch_model": False,
        "input_transform_order": [
            "openpi.transforms.InjectDefaultPrompt",
            "openpi.policies.droid_policy.DroidInputs",
            "openpi.transforms.DeltaActions",
            "openpi.transforms.Normalize",
            "openpi.transforms.InjectDefaultPrompt",
            "openpi.transforms.ResizeImages",
            "openpi.transforms.TokenizePrompt",
            "openpi.transforms.PadStatesAndActions",
        ],
        "output_transform_order": [
            "openpi.transforms.Unnormalize",
            "openpi.transforms.AbsoluteActions",
            "openpi.policies.droid_policy.DroidOutputs",
        ],
        "policy_default_prompt": None,
        "model_default_prompt": None,
        "tokenizer_runtime": _expected_sentencepiece_identity(
            wrapper_class="openpi.models.tokenizer.PaligemmaTokenizer",
            max_length=200,
            processor_attribute="_tokenizer",
        ),
        "normalization": {
            "input": "Normalize_quantile_droid_width32",
            "output": "Unnormalize_quantile_droid_width32",
            "checkpoint_values_sha256": PI05_DROID_JOINTPOS_NORM_VALUES_SHA256,
            "input_values_sha256": PI05_DROID_JOINTPOS_NORM_VALUES_SHA256,
            "output_values_sha256": PI05_DROID_JOINTPOS_NORM_VALUES_SHA256,
        },
        "sampler": {
            "objective": "flow_matching",
            "algorithm": "euler_t1_to_t0",
            "num_steps": 10,
            "num_steps_source": "Pi0.sample_actions_default",
            "sample_kwargs": {},
        },
        "response": "AbsoluteActions_then_DroidOutputs_leading8",
    }


def validate_official_policy_runtime(
    *,
    policy: object,
    jax_module: object,
    expected_norm_values_sha256: str,
) -> dict[str, Any]:
    """Bind the loaded policy's untouched FLOW sampler and initial RNG state."""

    try:
        key_data = jax_module.random.key_data(policy._rng).tolist()
        input_transforms = policy._input_transform.transforms
        output_transforms = policy._output_transform.transforms
        model = policy._model
        sample_kwargs = policy._sample_kwargs
        metadata = policy.metadata
        is_pytorch = policy._is_pytorch_model
    except AttributeError as error:
        raise ValueError("Loaded OpenPI policy runtime is incomplete") from error
    observed = {
        "policy_class": _class_path(policy),
        "model_class": _class_path(model),
        "model_pi05": getattr(model, "pi05", None),
        "model_action_horizon": getattr(model, "action_horizon", None),
        "model_action_dim": getattr(model, "action_dim", None),
        "model_max_token_len": getattr(model, "max_token_len", None),
        "metadata": metadata,
        "sample_kwargs": sample_kwargs,
        "initial_jax_key_data": key_data,
        "jax_enable_x64": jax_module.config.x64_enabled,
        "jax_backend": jax_module.default_backend(),
        "is_pytorch_model": is_pytorch,
        "input_transform_order": _transform_paths(
            input_transforms, "policy runtime inputs"
        ),
        "output_transform_order": _transform_paths(
            output_transforms, "policy runtime outputs"
        ),
        "policy_default_prompt": getattr(input_transforms[0], "prompt", "missing"),
        "model_default_prompt": getattr(input_transforms[4], "prompt", "missing"),
        "tokenizer_runtime": attest_loaded_tokenizer_sentencepiece(
            getattr(input_transforms[6], "tokenizer", None)
        ),
    }
    expected_observed = {
        key: value
        for key, value in _expected_policy_runtime_report().items()
        if key not in {"normalization", "sampler", "response"}
    }
    _require_exact(observed, expected_observed, "Loaded OpenPI policy runtime mismatch")
    normalize = input_transforms[3]
    unnormalize = output_transforms[0]
    if (
        getattr(normalize, "use_quantiles", None) is not True
        or getattr(normalize, "strict", None) is not False
        or getattr(unnormalize, "use_quantiles", None) is not True
    ):
        raise ValueError("Loaded OpenPI policy quantile normalization mismatch")
    _validate_norm_objects(normalize.norm_stats)
    _validate_norm_objects(unnormalize.norm_stats)
    input_norm_values_sha256 = _runtime_norm_values_sha256(normalize.norm_stats)
    output_norm_values_sha256 = _runtime_norm_values_sha256(unnormalize.norm_stats)
    if (
        expected_norm_values_sha256 != PI05_DROID_JOINTPOS_NORM_VALUES_SHA256
        or input_norm_values_sha256 != expected_norm_values_sha256
        or output_norm_values_sha256 != expected_norm_values_sha256
    ):
        raise ValueError(
            "Loaded OpenPI normalization values differ from the exact checkpoint"
        )
    signature = inspect.signature(type(model).sample_actions)
    num_steps = signature.parameters.get("num_steps")
    noise = signature.parameters.get("noise")
    if (
        num_steps is None
        or num_steps.default != 10
        or noise is None
        or noise.default is not None
        or type(model).sample_actions.__module__ != "openpi.models.pi0"
        or type(model).sample_actions.__qualname__ != "Pi0.sample_actions"
    ):
        raise ValueError("Loaded OpenPI FLOW sampler defaults mismatch")
    report = {
        **observed,
        "normalization": {
            "input": "Normalize_quantile_droid_width32",
            "output": "Unnormalize_quantile_droid_width32",
            "checkpoint_values_sha256": expected_norm_values_sha256,
            "input_values_sha256": input_norm_values_sha256,
            "output_values_sha256": output_norm_values_sha256,
        },
        "sampler": {
            "objective": "flow_matching",
            "algorithm": "euler_t1_to_t0",
            "num_steps": 10,
            "num_steps_source": "Pi0.sample_actions_default",
            "sample_kwargs": {},
        },
        "response": "AbsoluteActions_then_DroidOutputs_leading8",
    }
    _require_exact(
        report,
        _expected_policy_runtime_report(),
        "OpenPI complete policy-runtime report mismatch",
    )
    return report


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_atomic_immutable_json(path: Path, value: Any) -> dict[str, Any]:
    """Publish complete bytes atomically without replacing an existing artifact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"Refusing existing contract artifact: {path}")
    payload = canonical_json_bytes(value) + b"\n"
    temporary = path.parent / f".{path.name}.partial-{os.getpid()}"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    linked = False
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, path, follow_symlinks=False)
        linked = True
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
        if linked and path.exists() and not path.is_symlink():
            path.unlink()
        raise
    return _validate_immutable_json_bytes(path, payload)


def _validate_immutable_json_bytes(
    path: Path, expected_payload: bytes | None = None
) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Contract artifact must not be a symlink")
    file_stat = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISREG(file_stat.st_mode)
        or stat.S_IMODE(file_stat.st_mode) != 0o444
        or file_stat.st_nlink != 1
    ):
        raise ValueError("Contract artifact must be one mode-0444 regular link")
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Contract artifact is not strict JSON") from error
    canonical = canonical_json_bytes(value) + b"\n"
    if payload != canonical:
        raise ValueError("Contract artifact is not canonical JSON")
    if expected_payload is not None and payload != expected_payload:
        raise ValueError("Contract artifact payload changed during publication")
    return {
        "path": str(path.resolve()),
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "mode": "0444",
        "nlink": 1,
        "value": value,
    }


def publish_pi05_droid_jointpos_serving_contract(
    path: Path, metadata: dict[str, Any]
) -> dict[str, Any]:
    if Path(path).name != PI05_DROID_JOINTPOS_SERVING_CONTRACT_FILENAME:
        raise ValueError("Unexpected pi0.5 joint-position serving artifact filename")
    validate_pi05_droid_jointpos_server_metadata(metadata)
    return _publish_atomic_immutable_json(Path(path), metadata)


def validate_persisted_pi05_droid_jointpos_serving_contract(
    path: Path, metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    if Path(path).name != PI05_DROID_JOINTPOS_SERVING_CONTRACT_FILENAME:
        raise ValueError("Unexpected pi0.5 joint-position serving artifact filename")
    artifact = _validate_immutable_json_bytes(Path(path))
    validate_pi05_droid_jointpos_server_metadata(artifact["value"])
    if metadata is not None:
        validate_pi05_droid_jointpos_server_metadata(metadata)
        _require_exact(
            artifact["value"], metadata, "Persisted contract differs from live metadata"
        )
    return artifact


def make_pi05_droid_jointpos_rng_stream_report(
    *,
    server_pid: int,
    initial_key_data: list[int],
    final_key_data: list[int],
    expected_final_key_data: list[int],
    expected_request_count: int,
    metadata_contract_sha256: str,
) -> dict[str, Any]:
    """Build final proof for the official Policy.infer RNG recurrence."""

    report = {
        "schema_version": 1,
        "profile": PI05_DROID_JOINTPOS_RNG_STREAM_PROFILE,
        "status": "pass",
        "bind_host": PI05_DROID_JOINTPOS_BIND_HOST,
        "server_pid": server_pid,
        "server_class": (
            "openpi.serving.websocket_policy_server.WebsocketPolicyServer"
        ),
        "policy_class": "openpi.policies.policy.Policy",
        "policy_wrapper": None,
        "rng_recurrence": "stored_key_next_equals_jax_random_split_stored_key_0",
        "request_serialization": (
            "official_server_synchronous_policy_infer_single_event_loop"
        ),
        "listener_quiescence": (
            "cancel_official_run_then_async_context_close_wait_closed_before_snapshot"
        ),
        "snapshot_after_all_connection_handlers_returned": True,
        "initial_key_data": initial_key_data,
        "final_key_data": final_key_data,
        "expected_final_key_data": expected_final_key_data,
        "expected_request_count": expected_request_count,
        "observed_request_count": expected_request_count,
        "request_count_inference": (
            "final_key_equals_exact_recurrence_after_expected_requests"
        ),
        "metadata_contract_sha256": metadata_contract_sha256,
    }
    return validate_pi05_droid_jointpos_rng_stream_report(report)


def validate_pi05_droid_jointpos_rng_stream_report(value: Any) -> dict[str, Any]:
    required = {
        "schema_version",
        "profile",
        "status",
        "bind_host",
        "server_pid",
        "server_class",
        "policy_class",
        "policy_wrapper",
        "rng_recurrence",
        "request_serialization",
        "listener_quiescence",
        "snapshot_after_all_connection_handlers_returned",
        "initial_key_data",
        "final_key_data",
        "expected_final_key_data",
        "expected_request_count",
        "observed_request_count",
        "request_count_inference",
        "metadata_contract_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("pi0.5 joint-position RNG-stream schema mismatch")
    if (
        value["schema_version"] != 1
        or value["profile"] != PI05_DROID_JOINTPOS_RNG_STREAM_PROFILE
        or value["status"] != "pass"
        or value["bind_host"] != PI05_DROID_JOINTPOS_BIND_HOST
        or type(value["server_pid"]) is not int
        or value["server_pid"] <= 0
        or value["server_class"]
        != "openpi.serving.websocket_policy_server.WebsocketPolicyServer"
        or value["policy_class"] != "openpi.policies.policy.Policy"
        or value["policy_wrapper"] is not None
        or value["rng_recurrence"]
        != "stored_key_next_equals_jax_random_split_stored_key_0"
        or value["request_serialization"]
        != "official_server_synchronous_policy_infer_single_event_loop"
        or value["listener_quiescence"]
        != "cancel_official_run_then_async_context_close_wait_closed_before_snapshot"
        or value["snapshot_after_all_connection_handlers_returned"] is not True
        or value["request_count_inference"]
        != "final_key_equals_exact_recurrence_after_expected_requests"
    ):
        raise ValueError("pi0.5 joint-position RNG-stream identity mismatch")
    for name in (
        "initial_key_data",
        "final_key_data",
        "expected_final_key_data",
    ):
        key_data = value[name]
        if (
            not isinstance(key_data, list)
            or len(key_data) != 2
            or any(
                type(item) is not int or not 0 <= item <= 2**32 - 1 for item in key_data
            )
        ):
            raise ValueError(f"pi0.5 joint-position {name} mismatch")
    if value["initial_key_data"] != [0, 0]:
        raise ValueError("pi0.5 joint-position initial RNG key mismatch")
    if value["final_key_data"] != value["expected_final_key_data"]:
        raise ValueError("pi0.5 joint-position final RNG key mismatch")
    if (
        type(value["expected_request_count"]) is not int
        or value["expected_request_count"] <= 0
        or value["observed_request_count"] != value["expected_request_count"]
    ):
        raise ValueError("pi0.5 joint-position request count mismatch")
    digest = value["metadata_contract_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("pi0.5 joint-position RNG metadata identity mismatch")
    return copy.deepcopy(value)


def publish_pi05_droid_jointpos_rng_stream(
    path: Path, value: dict[str, Any]
) -> dict[str, Any]:
    if Path(path).name != PI05_DROID_JOINTPOS_RNG_STREAM_FILENAME:
        raise ValueError("Unexpected pi0.5 joint-position RNG artifact filename")
    canonical = validate_pi05_droid_jointpos_rng_stream_report(value)
    return _publish_atomic_immutable_json(Path(path), canonical)


def validate_persisted_pi05_droid_jointpos_rng_stream(
    path: Path,
    *,
    expected_request_count: int | None = None,
    expected_contract_sha256: str | None = None,
) -> dict[str, Any]:
    if Path(path).name != PI05_DROID_JOINTPOS_RNG_STREAM_FILENAME:
        raise ValueError("Unexpected pi0.5 joint-position RNG artifact filename")
    artifact = _validate_immutable_json_bytes(Path(path))
    report = validate_pi05_droid_jointpos_rng_stream_report(artifact["value"])
    if (
        expected_request_count is not None
        and report["expected_request_count"] != expected_request_count
    ):
        raise ValueError("Persisted RNG request count differs from trace")
    if (
        expected_contract_sha256 is not None
        and report["metadata_contract_sha256"] != expected_contract_sha256
    ):
        raise ValueError("Persisted RNG stream differs from serving contract")
    artifact["value"] = report
    return artifact


def make_pi05_droid_jointpos_model_runtime(
    *,
    checkpoint: dict[str, Any],
    train_config: dict[str, Any],
    data_config: dict[str, Any],
    policy_runtime: dict[str, Any],
    openpi_checkout: dict[str, Any],
    openpi_runtime_attestation: dict[str, Any],
    host_runtime: dict[str, Any],
    tokenizer_artifact: dict[str, Any],
    expected_request_count: int,
    serving_metadata: dict[str, Any],
) -> dict[str, Any]:
    contract = validate_pi05_droid_jointpos_server_metadata(serving_metadata)
    runtime = validate_openpi_runtime_attestation(openpi_runtime_attestation)
    if contract["openpi"]["runtime_attestation"] != runtime:
        raise ValueError("Serving and model-runtime OpenPI attestations differ")
    return {
        "schema_version": 1,
        "profile": PI05_DROID_JOINTPOS_MODEL_RUNTIME_PROFILE,
        "status": "pass",
        "checkpoint": checkpoint,
        "train_config": train_config,
        "data_config": data_config,
        "policy_runtime": policy_runtime,
        "openpi_checkout": openpi_checkout,
        "openpi_runtime_attestation": runtime,
        "host_runtime": validate_openpi_host_runtime(host_runtime),
        "tokenizer_artifact": validate_paligemma_tokenizer_artifact(tokenizer_artifact),
        "server": {
            "class": ("openpi.serving.websocket_policy_server.WebsocketPolicyServer"),
            "lifecycle": "asyncio_task_of_official_WebsocketPolicyServer.run",
            "bind_host": PI05_DROID_JOINTPOS_BIND_HOST,
            "network_scope": "ipv4_loopback_only",
            "policy_wrapper": None,
            "rng_snapshot_barrier": (
                "official_run_cancel_close_wait_closed_before_policy_rng_read"
            ),
            "rng_stream_final_artifact": PI05_DROID_JOINTPOS_RNG_STREAM_FILENAME,
            "expected_request_count": expected_request_count,
            "metadata_contract_sha256": contract["contract_sha256"],
            "request_image_contract": {
                "shape": [720, 1280, 3],
                "dtype": "uint8",
                "client_model_spatial_transform": None,
                "server_resize_transform": "openpi.transforms.ResizeImages",
                "server_resize_implementation": (
                    "openpi_client.image_tools.resize_with_pad"
                ),
                "server_resize_backend": "PIL.Image.resize",
                "server_resize_method": "PIL.Image.Resampling.BILINEAR",
                "server_padding": "symmetric_zero",
                "model_shape": [224, 224, 3],
                "server_resize_output_dtype": "uint8",
                "resize_application_count": 1,
                "server_resize_runtime_probe": _expected_resize_runtime_probe(),
                "observation_conversion": _expected_observation_image_conversion(),
                "model_preprocess_resize": _expected_model_preprocess_resize(),
            },
        },
    }


def _validate_model_runtime_value(
    value: Any, serving_metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    required = {
        "schema_version",
        "profile",
        "status",
        "checkpoint",
        "train_config",
        "data_config",
        "policy_runtime",
        "openpi_checkout",
        "openpi_runtime_attestation",
        "host_runtime",
        "tokenizer_artifact",
        "server",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("pi0.5 joint-position model-runtime schema mismatch")
    if (
        value["schema_version"] != 1
        or value["profile"] != PI05_DROID_JOINTPOS_MODEL_RUNTIME_PROFILE
        or value["status"] != "pass"
    ):
        raise ValueError("pi0.5 joint-position model-runtime identity mismatch")
    checkpoint = value["checkpoint"]
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "schema_version",
        "status",
        "checkpoint_uri",
        "checkpoint_dir",
        "manifest_path",
        "manifest_sha256",
        "object_count",
        "total_bytes",
        "full_md5",
        "objects_sha256",
        "normalization",
    }:
        raise ValueError("pi0.5 joint-position checkpoint report schema mismatch")
    expected_checkpoint = {
        "schema_version": 1,
        "status": "pass",
        "checkpoint_uri": PI05_DROID_JOINTPOS_CHECKPOINT_URI,
        "manifest_sha256": PI05_DROID_JOINTPOS_MANIFEST_SHA256,
        "object_count": PI05_DROID_JOINTPOS_OBJECT_COUNT,
        "total_bytes": PI05_DROID_JOINTPOS_CHECKPOINT_BYTES,
        "full_md5": True,
    }
    for key, expected in expected_checkpoint.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"pi0.5 checkpoint report {key} mismatch")
    if not all(
        isinstance(checkpoint.get(key), str) and Path(checkpoint[key]).is_absolute()
        for key in ("checkpoint_dir", "manifest_path")
    ):
        raise ValueError("pi0.5 checkpoint report paths must be absolute")
    if (
        not isinstance(checkpoint.get("objects_sha256"), str)
        or len(checkpoint["objects_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in checkpoint["objects_sha256"]
        )
    ):
        raise ValueError("pi0.5 checkpoint object identity mismatch")
    normalization = checkpoint["normalization"]
    if (
        not isinstance(normalization, dict)
        or set(normalization)
        != {
            "path",
            "sha256",
            "values_sha256",
            "asset_id",
            "scope",
            "state_width",
            "action_width",
        }
        or {
            key: normalization.get(key)
            for key in (
                "sha256",
                "values_sha256",
                "asset_id",
                "scope",
                "state_width",
                "action_width",
            )
        }
        != {
            "sha256": PI05_DROID_JOINTPOS_NORM_SHA256,
            "values_sha256": PI05_DROID_JOINTPOS_NORM_VALUES_SHA256,
            "asset_id": "droid",
            "scope": "checkpoint_global_droid",
            "state_width": 32,
            "action_width": 32,
        }
    ):
        raise ValueError("pi0.5 checkpoint normalization report mismatch")
    if (
        not isinstance(normalization.get("path"), str)
        or not Path(normalization["path"]).is_absolute()
    ):
        raise ValueError("pi0.5 normalization path must be absolute")
    _require_exact(
        value["train_config"],
        _expected_train_config_report(),
        "Persisted OpenPI train-config report mismatch",
    )
    _require_exact(
        value["data_config"],
        _expected_data_config_report(),
        "Persisted OpenPI data-config report mismatch",
    )
    _require_exact(
        value["policy_runtime"],
        _expected_policy_runtime_report(),
        "Persisted OpenPI policy-runtime report mismatch",
    )
    checkout = value["openpi_checkout"]
    if not isinstance(checkout, dict) or set(checkout) != {
        "root",
        "git_commit",
        "tracked_and_untracked_clean",
    }:
        raise ValueError("OpenPI checkout report schema mismatch")
    if (
        not Path(checkout["root"]).is_absolute()
        or checkout["git_commit"] != PI05_DROID_JOINTPOS_OPENPI_COMMIT
        or checkout["tracked_and_untracked_clean"] is not True
    ):
        raise ValueError("OpenPI checkout report mismatch")
    runtime = validate_openpi_runtime_attestation(value["openpi_runtime_attestation"])
    host_runtime = validate_openpi_host_runtime(value["host_runtime"])
    validate_paligemma_tokenizer_artifact(value["tokenizer_artifact"])
    checkout_root = Path(checkout["root"])
    if (
        host_runtime["python"]["declared_executable"]
        != str(checkout_root / ".venv/bin/python")
        or host_runtime["python"]["prefix"] != str((checkout_root / ".venv").resolve())
        or host_runtime["locked_source_environment"]["uv_lock"]["path"]
        != str((checkout_root / "uv.lock").resolve())
        or host_runtime["locked_source_environment"]["pyproject"]["path"]
        != str((checkout_root / "pyproject.toml").resolve())
    ):
        raise ValueError("OpenPI host runtime is not rooted in the attested checkout")
    server = value["server"]
    if not isinstance(server, dict) or set(server) != {
        "class",
        "lifecycle",
        "bind_host",
        "network_scope",
        "policy_wrapper",
        "rng_snapshot_barrier",
        "rng_stream_final_artifact",
        "expected_request_count",
        "metadata_contract_sha256",
        "request_image_contract",
    }:
        raise ValueError("pi0.5 server runtime schema mismatch")
    if (
        server["class"]
        != "openpi.serving.websocket_policy_server.WebsocketPolicyServer"
        or server["lifecycle"] != "asyncio_task_of_official_WebsocketPolicyServer.run"
        or server["bind_host"] != PI05_DROID_JOINTPOS_BIND_HOST
        or server["network_scope"] != "ipv4_loopback_only"
        or server["policy_wrapper"] is not None
        or server["rng_snapshot_barrier"]
        != "official_run_cancel_close_wait_closed_before_policy_rng_read"
        or server["rng_stream_final_artifact"]
        != PI05_DROID_JOINTPOS_RNG_STREAM_FILENAME
        or type(server["expected_request_count"]) is not int
        or server["expected_request_count"] <= 0
        or server["request_image_contract"]
        != {
            "shape": [720, 1280, 3],
            "dtype": "uint8",
            "client_model_spatial_transform": None,
            "server_resize_transform": "openpi.transforms.ResizeImages",
            "server_resize_implementation": (
                "openpi_client.image_tools.resize_with_pad"
            ),
            "server_resize_backend": "PIL.Image.resize",
            "server_resize_method": "PIL.Image.Resampling.BILINEAR",
            "server_padding": "symmetric_zero",
            "model_shape": [224, 224, 3],
            "server_resize_output_dtype": "uint8",
            "resize_application_count": 1,
            "server_resize_runtime_probe": _expected_resize_runtime_probe(),
            "observation_conversion": _expected_observation_image_conversion(),
            "model_preprocess_resize": _expected_model_preprocess_resize(),
        }
    ):
        raise ValueError("pi0.5 server runtime mismatch")
    expected_contract = expected_pi05_droid_jointpos_server_contract(runtime)
    if server["metadata_contract_sha256"] != expected_contract["contract_sha256"]:
        raise ValueError("Model-runtime serving contract identity mismatch")
    if serving_metadata is not None:
        contract = validate_pi05_droid_jointpos_server_metadata(serving_metadata)
        if contract["openpi"]["runtime_attestation"] != runtime:
            raise ValueError("Live metadata and model-runtime attestation differ")
        if server["metadata_contract_sha256"] != contract["contract_sha256"]:
            raise ValueError("Live metadata and model-runtime contract differ")
    return copy.deepcopy(value)


def publish_pi05_droid_jointpos_model_runtime(
    path: Path, value: dict[str, Any], serving_metadata: dict[str, Any]
) -> dict[str, Any]:
    if Path(path).name != PI05_DROID_JOINTPOS_MODEL_RUNTIME_FILENAME:
        raise ValueError("Unexpected pi0.5 joint-position runtime artifact filename")
    _validate_model_runtime_value(value, serving_metadata)
    return _publish_atomic_immutable_json(Path(path), value)


def validate_persisted_pi05_droid_jointpos_model_runtime(
    path: Path, serving_metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    if Path(path).name != PI05_DROID_JOINTPOS_MODEL_RUNTIME_FILENAME:
        raise ValueError("Unexpected pi0.5 joint-position runtime artifact filename")
    artifact = _validate_immutable_json_bytes(Path(path))
    artifact["value"] = _validate_model_runtime_value(
        artifact["value"], serving_metadata
    )
    return artifact


__all__ = [
    "PI05_DROID_JOINTPOS_BIND_HOST",
    "PI05_DROID_JOINTPOS_ALLOWED_OPTIONAL_RUNTIME_ENVIRONMENT",
    "PI05_DROID_JOINTPOS_CHECKPOINT_URI",
    "PI05_DROID_JOINTPOS_CONFIG",
    "PI05_DROID_JOINTPOS_MANIFEST_SHA256",
    "PI05_DROID_JOINTPOS_METADATA_KEY",
    "PI05_DROID_JOINTPOS_MODEL_RUNTIME_FILENAME",
    "PI05_DROID_JOINTPOS_NORM_SHA256",
    "PI05_DROID_JOINTPOS_NORM_VALUES_SHA256",
    "PI05_DROID_JOINTPOS_OPENPI_COMMIT",
    "PI05_DROID_JOINTPOS_HOST_RUNTIME_PROFILE",
    "PI05_DROID_JOINTPOS_JAX_CONFIG",
    "PI05_DROID_JOINTPOS_OPTIONAL_RUNTIME_ENVIRONMENT",
    "PI05_DROID_JOINTPOS_PYPROJECT_SHA256",
    "PI05_DROID_JOINTPOS_RECORD_VERIFICATION_EXEMPTIONS",
    "PI05_DROID_JOINTPOS_REQUIRED_PACKAGE_VERSIONS",
    "PI05_DROID_JOINTPOS_REQUIRED_RUNTIME_ENVIRONMENT",
    "PI05_DROID_JOINTPOS_RESIZE_PROBE_INPUT_SHA256",
    "PI05_DROID_JOINTPOS_RESIZE_PROBE_OUTPUT_SHA256",
    "PI05_DROID_JOINTPOS_RNG_STREAM_FILENAME",
    "PI05_DROID_JOINTPOS_RNG_STREAM_PROFILE",
    "PI05_DROID_JOINTPOS_SERVER_MODEL_RESIZE",
    "PI05_DROID_JOINTPOS_SERVING_CONTRACT_FILENAME",
    "PI05_DROID_JOINTPOS_TOKENIZER_GENERATION",
    "PI05_DROID_JOINTPOS_TOKENIZER_MD5_BASE64",
    "PI05_DROID_JOINTPOS_TOKENIZER_SHA256",
    "PI05_DROID_JOINTPOS_TOKENIZER_SIZE",
    "PI05_DROID_JOINTPOS_TOKENIZER_URI",
    "PI05_DROID_JOINTPOS_TOKENIZER_VOCAB_SIZE",
    "PI05_DROID_JOINTPOS_UV_LOCK_SHA256",
    "attest_imported_openpi_modules",
    "attest_loaded_tokenizer_sentencepiece",
    "attest_openpi_resize_transform",
    "capture_openpi_host_runtime",
    "expected_pi05_droid_jointpos_server_metadata",
    "make_pi05_droid_jointpos_model_runtime",
    "make_pi05_droid_jointpos_rng_stream_report",
    "pi05_droid_jointpos_server_contract_sha256",
    "publish_pi05_droid_jointpos_model_runtime",
    "publish_pi05_droid_jointpos_rng_stream",
    "publish_pi05_droid_jointpos_serving_contract",
    "validate_official_data_config",
    "validate_official_policy_runtime",
    "validate_official_train_config",
    "validate_openpi_host_runtime",
    "validate_paligemma_tokenizer_artifact",
    "validate_persisted_pi05_droid_jointpos_model_runtime",
    "validate_persisted_pi05_droid_jointpos_rng_stream",
    "validate_persisted_pi05_droid_jointpos_serving_contract",
    "validate_pi05_droid_jointpos_server_metadata",
    "validate_pi05_droid_jointpos_rng_stream_report",
    "validate_pi05_droid_jointpos_loopback_listener",
    "verify_openpi_git_checkout",
    "verify_openpi_package_environment",
    "verify_paligemma_tokenizer_artifact",
    "verify_pi05_droid_jointpos_checkpoint",
]
