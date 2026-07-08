import asyncio
import copy
import importlib.util
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace

import numpy as np
import pytest

import polaris.pi05_droid_jointpos_serving_contract as contract


ROOT = Path(__file__).resolve().parents[1]
GPU_UUID = "GPU-01234567-89ab-cdef-0123-456789abcdef"
RNG_VERIFIER_PATH = ROOT / "scripts/polaris/verify_pi05_droid_jointpos_rng_stream.py"
RNG_VERIFIER_SPEC = importlib.util.spec_from_file_location(
    "verify_pi05_droid_jointpos_rng_stream", RNG_VERIFIER_PATH
)
assert RNG_VERIFIER_SPEC is not None and RNG_VERIFIER_SPEC.loader is not None
RNG_VERIFIER = importlib.util.module_from_spec(RNG_VERIFIER_SPEC)
RNG_VERIFIER_SPEC.loader.exec_module(RNG_VERIFIER)
SERVER_PATH = ROOT / "scripts/polaris/serve_pi05_droid_jointpos_attested.py"
SERVER_SPEC = importlib.util.spec_from_file_location(
    "serve_pi05_droid_jointpos_attested", SERVER_PATH
)
assert SERVER_SPEC is not None and SERVER_SPEC.loader is not None
SERVER_MODULE = importlib.util.module_from_spec(SERVER_SPEC)
SERVER_SPEC.loader.exec_module(SERVER_MODULE)


def _expected_resize_probe():
    return {
        "transform": "openpi.transforms.ResizeImages",
        "bound_module": "openpi_client.image_tools",
        "bound_function": "openpi_client.image_tools.resize_with_pad",
        "backend": "PIL.Image.resize",
        "method": "PIL.Image.Resampling.BILINEAR",
        "padding": "symmetric_zero",
        "input_shape": [5, 9, 3],
        "input_dtype": "uint8",
        "input_sha256": contract.PI05_DROID_JOINTPOS_RESIZE_PROBE_INPUT_SHA256,
        "target_shape": [224, 224, 3],
        "output_dtype": "uint8",
        "output_sha256": contract.PI05_DROID_JOINTPOS_RESIZE_PROBE_OUTPUT_SHA256,
    }


def _expected_observation_conversion():
    return {
        "implementation": "openpi.models.model.Observation.from_dict",
        "input_dtype": "uint8",
        "output_dtype": "float32",
        "mapping": "value_div_255_times_2_minus_1",
    }


def _expected_inactive_model_resize():
    return {
        "implementation": "openpi.shared.image_tools.resize_with_pad",
        "backend": "jax.image.resize",
        "active": False,
        "inactivity_condition": "input_spatial_shape_equals_224x224",
    }


def _runtime_attestation():
    modules = []
    for index, module in enumerate(sorted(contract._REQUIRED_OPENPI_MODULES)):
        source_root = (
            "packages/openpi-client/src/"
            if module.startswith("openpi_client.")
            else "src/"
        )
        relative = source_root + module.replace(".", "/") + ".py"
        modules.append(
            {
                "module": module,
                "relative_path": relative,
                "sha256": f"{index + 1:064x}",
            }
        )
    value = {
        "schema_version": 1,
        "git_commit": contract.PI05_DROID_JOINTPOS_OPENPI_COMMIT,
        "import_roots": ["packages/openpi-client/src", "src"],
        "modules": modules,
        "namespace_packages": [],
    }
    value["attestation_sha256"] = contract._runtime_attestation_sha256(value)
    return value


def _metadata():
    return contract.expected_pi05_droid_jointpos_server_metadata(_runtime_attestation())


def _checkpoint_report(tmp_path):
    return {
        "schema_version": 1,
        "status": "pass",
        "checkpoint_uri": contract.PI05_DROID_JOINTPOS_CHECKPOINT_URI,
        "checkpoint_dir": str((tmp_path / "checkpoint").resolve()),
        "manifest_path": str((tmp_path / "manifest.tsv").resolve()),
        "manifest_sha256": contract.PI05_DROID_JOINTPOS_MANIFEST_SHA256,
        "object_count": contract.PI05_DROID_JOINTPOS_OBJECT_COUNT,
        "total_bytes": contract.PI05_DROID_JOINTPOS_CHECKPOINT_BYTES,
        "full_md5": True,
        "objects_sha256": "a" * 64,
        "normalization": {
            "path": str(
                (tmp_path / "checkpoint/assets/droid/norm_stats.json").resolve()
            ),
            "sha256": contract.PI05_DROID_JOINTPOS_NORM_SHA256,
            "values_sha256": contract.PI05_DROID_JOINTPOS_NORM_VALUES_SHA256,
            "asset_id": "droid",
            "scope": "checkpoint_global_droid",
            "state_width": 32,
            "action_width": 32,
        },
    }


def _host_runtime(tmp_path):
    versions = dict(contract.PI05_DROID_JOINTPOS_REQUIRED_PACKAGE_VERSIONS)
    versions.update(
        {
            name: version
            for name, version, _url, _digest, _size in (
                contract.PI05_DROID_JOINTPOS_PINNED_WHEEL_ARTIFACTS
            )
        }
    )
    distributions = [
        {"name": name, "version": version} for name, version in sorted(versions.items())
    ]
    record_verified = []
    for index, name in enumerate(
        name
        for name in sorted(versions)
        if name not in contract.PI05_DROID_JOINTPOS_RECORD_VERIFICATION_EXEMPTIONS
    ):
        hex_count = sum(
            1
            for entry_name, entry_version, _path, _size, _digest in (
                contract.PI05_DROID_JOINTPOS_PINNED_HEX_RECORD_ENTRIES
            )
            if entry_name == name and entry_version == versions[name]
        )
        overlap_resolutions = [
            contract._expected_record_overlap_resolution(profile)
            for profile in contract.PI05_DROID_JOINTPOS_PINNED_RECORD_OVERLAP_RESOLUTIONS
            if profile[0] == name and profile[1] == versions[name]
        ]
        overlap_resolutions.sort(key=lambda item: item["path"])
        overlap_count = len(overlap_resolutions)
        hashed_file_count = max(1, hex_count + overlap_count + 1)
        record_verified.append(
            {
                "name": name,
                "version": versions[name],
                "file_count": hashed_file_count + 1,
                "hashed_file_count": hashed_file_count,
                "record_validation_counts": {
                    contract.PI05_DROID_JOINTPOS_RECORD_VALIDATION_MODES[0]: (
                        hashed_file_count - hex_count - overlap_count
                    ),
                    contract.PI05_DROID_JOINTPOS_RECORD_VALIDATION_MODES[1]: hex_count,
                    contract.PI05_DROID_JOINTPOS_RECORD_VALIDATION_MODES[
                        2
                    ]: overlap_count,
                },
                "record_overlap_resolutions": overlap_resolutions,
                "record": {
                    "path": str(
                        (tmp_path / f"site-packages/{name}.dist-info/RECORD").resolve()
                    ),
                    "size": 1,
                    "sha256": f"{index + 1:064x}",
                },
                "verified_files_sha256": f"{index + 11:064x}",
            }
        )
    record_exemptions = [
        {
            "name": name,
            "version": versions[name],
            "reason": "source_attested_editable_openpi_checkout",
        }
        for name in contract.PI05_DROID_JOINTPOS_RECORD_VERIFICATION_EXEMPTIONS
    ]
    return {
        "schema_version": 2,
        "profile": contract.PI05_DROID_JOINTPOS_HOST_RUNTIME_PROFILE,
        "python": {
            "declared_executable": str(
                (tmp_path / "openpi/.venv/bin/python").resolve()
            ),
            "resolved_executable": {
                "path": str((tmp_path / "python3.11").resolve()),
                "size": 1,
                "sha256": "1" * 64,
            },
            "implementation": "CPython",
            "version": "3.11.15",
            "cache_tag": "cpython-311",
            "prefix": str((tmp_path / "openpi/.venv").resolve()),
            "base_prefix": str(tmp_path.resolve()),
        },
        "locked_source_environment": {
            "uv_lock": {
                "path": str((tmp_path / "openpi/uv.lock").resolve()),
                "size": 1,
                "sha256": contract.PI05_DROID_JOINTPOS_UV_LOCK_SHA256,
            },
            "pyproject": {
                "path": str((tmp_path / "openpi/pyproject.toml").resolve()),
                "size": 1,
                "sha256": contract.PI05_DROID_JOINTPOS_PYPROJECT_SHA256,
            },
        },
        "packages": {
            "required_versions": dict(
                contract.PI05_DROID_JOINTPOS_REQUIRED_PACKAGE_VERSIONS
            ),
            "all_installed_versions_allowed_by_uv_lock": True,
            "all_noneditable_files_bound_to_locked_records_or_pinned_overlap": True,
            "pinned_wheel_artifacts": contract._expected_pinned_wheel_artifacts(),
            "record_verified_distributions": record_verified,
            "record_verification_exemptions": record_exemptions,
            "installed_distributions": distributions,
            "installed_distributions_sha256": contract.hashlib.sha256(
                contract.canonical_json_bytes(distributions)
            ).hexdigest(),
        },
        "process_environment": {
            "required": dict(contract.PI05_DROID_JOINTPOS_REQUIRED_RUNTIME_ENVIRONMENT),
            "optional": {
                name: GPU_UUID if name == "NVIDIA_VISIBLE_DEVICES" else None
                for name in contract.PI05_DROID_JOINTPOS_OPTIONAL_RUNTIME_ENVIRONMENT
            },
        },
        "jax": {
            "jax_version": "0.5.3",
            "jaxlib_version": "0.5.3",
            "jax_cuda12_pjrt_version": "0.5.3",
            "jax_cuda12_plugin_version": "0.5.3",
            "enable_x64": False,
            "config": dict(contract.PI05_DROID_JOINTPOS_JAX_CONFIG),
            "default_backend": "gpu",
            "platform_version": "CUDA 12 test runtime",
            "devices": [
                {
                    "id": 0,
                    "platform": "gpu",
                    "device_kind": contract.PI05_DROID_JOINTPOS_NVIDIA_GPU_NAME,
                    "process_index": 0,
                }
            ],
            "nvidia_smi": {
                "query": list(contract.PI05_DROID_JOINTPOS_NVIDIA_SMI_QUERY),
                "uuid": GPU_UUID,
                "name": contract.PI05_DROID_JOINTPOS_NVIDIA_GPU_NAME,
                "driver_version": (contract.PI05_DROID_JOINTPOS_NVIDIA_DRIVER_VERSION),
            },
        },
    }


def test_checkout_local_interpreter_accepts_resolved_lustre_alias(
    tmp_path, monkeypatch
):
    real_root = tmp_path / "openpi"
    expected_bin = real_root / ".venv/bin"
    expected_bin.mkdir(parents=True)
    python_target = tmp_path / "python3.11"
    python_target.write_bytes(b"python")
    (expected_bin / "python").symlink_to(python_target)
    alias_root = tmp_path / "alias-openpi"
    alias_root.symlink_to(real_root, target_is_directory=True)

    monkeypatch.setattr(contract.sys, "prefix", str(alias_root / ".venv"))
    monkeypatch.setattr(
        contract.sys, "executable", str(alias_root / ".venv/bin/python")
    )
    declared = contract._require_checkout_local_openpi_interpreter(real_root)
    assert declared == real_root / ".venv/bin/python"

    metadata = _metadata()
    runtime = _model_runtime(tmp_path, metadata)
    runtime["host_runtime"]["python"]["declared_executable"] = str(declared)
    output = tmp_path / contract.PI05_DROID_JOINTPOS_MODEL_RUNTIME_FILENAME
    published = contract.publish_pi05_droid_jointpos_model_runtime(
        output, runtime, metadata
    )
    assert published["value"]["host_runtime"]["python"]["declared_executable"] == str(
        real_root / ".venv/bin/python"
    )


def test_checkout_local_interpreter_rejects_other_venv_with_same_python(
    tmp_path, monkeypatch
):
    real_root = tmp_path / "real/openpi"
    expected_bin = real_root / ".venv/bin"
    expected_bin.mkdir(parents=True)
    python_target = tmp_path / "python3.11"
    python_target.write_bytes(b"python")
    (expected_bin / "python").symlink_to(python_target)
    other_bin = tmp_path / "other/.venv/bin"
    other_bin.mkdir(parents=True)
    (other_bin / "python").symlink_to(python_target)

    monkeypatch.setattr(contract.sys, "prefix", str(other_bin.parent))
    monkeypatch.setattr(contract.sys, "executable", str(other_bin / "python"))
    with pytest.raises(ValueError, match="checkout-local OpenPI interpreter"):
        contract._require_checkout_local_openpi_interpreter(real_root)


def _tokenizer_artifact(tmp_path):
    return {
        "schema_version": 1,
        "status": "pass",
        "uri": contract.PI05_DROID_JOINTPOS_TOKENIZER_URI,
        "remote": {
            "generation": contract.PI05_DROID_JOINTPOS_TOKENIZER_GENERATION,
            "size": contract.PI05_DROID_JOINTPOS_TOKENIZER_SIZE,
            "md5_base64": contract.PI05_DROID_JOINTPOS_TOKENIZER_MD5_BASE64,
        },
        "local": {
            "path": str((tmp_path / "paligemma_tokenizer.model").resolve()),
            "size": contract.PI05_DROID_JOINTPOS_TOKENIZER_SIZE,
            "md5_base64": contract.PI05_DROID_JOINTPOS_TOKENIZER_MD5_BASE64,
            "sha256": contract.PI05_DROID_JOINTPOS_TOKENIZER_SHA256,
        },
    }


def _model_runtime(tmp_path, metadata):
    runtime = _runtime_attestation()
    return contract.make_pi05_droid_jointpos_model_runtime(
        checkpoint=_checkpoint_report(tmp_path),
        train_config=contract._expected_train_config_report(),
        data_config=contract._expected_data_config_report(),
        policy_runtime=contract._expected_policy_runtime_report(),
        openpi_checkout={
            "root": str((tmp_path / "openpi").resolve()),
            "git_commit": contract.PI05_DROID_JOINTPOS_OPENPI_COMMIT,
            "tracked_and_untracked_clean": True,
        },
        openpi_runtime_attestation=runtime,
        host_runtime=_host_runtime(tmp_path),
        tokenizer_artifact=_tokenizer_artifact(tmp_path),
        expected_request_count=57,
        serving_metadata=metadata,
    )


def test_exact_handshake_binds_joint_position_flow_and_is_path_independent():
    metadata = _metadata()
    validated = contract.validate_pi05_droid_jointpos_server_metadata(metadata)
    assert validated["checkpoint"]["full_md5_verified_before_serve"] is True
    assert validated["normalization"] == {
        "asset_id": "droid",
        "scope": "checkpoint_global_droid",
        "path": "assets/droid/norm_stats.json",
        "sha256": contract.PI05_DROID_JOINTPOS_NORM_SHA256,
        "canonical_values_sha256": (contract.PI05_DROID_JOINTPOS_NORM_VALUES_SHA256),
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
    }
    assert validated["openpi"]["model_type"] == "pi05"
    assert validated["openpi"]["objective"] == "flow_matching"
    assert validated["openpi"]["compute_dtype"] == "bfloat16"
    assert validated["tokenizer"] == {
        "uri": contract.PI05_DROID_JOINTPOS_TOKENIZER_URI,
        "generation": contract.PI05_DROID_JOINTPOS_TOKENIZER_GENERATION,
        "size": contract.PI05_DROID_JOINTPOS_TOKENIZER_SIZE,
        "md5_base64": contract.PI05_DROID_JOINTPOS_TOKENIZER_MD5_BASE64,
        "sha256": contract.PI05_DROID_JOINTPOS_TOKENIZER_SHA256,
        "active_wrapper": "openpi.models.tokenizer.PaligemmaTokenizer",
        "fast_wrapper_active": False,
        "standard_sentencepiece_attribute": "_tokenizer",
        "fast_sentencepiece_attribute": "_paligemma_tokenizer",
        "serialized_model_proto_sha256": contract.PI05_DROID_JOINTPOS_TOKENIZER_SHA256,
    }
    assert validated["openpi"]["sampler"] == {
        "algorithm": "euler_t1_to_t0",
        "num_steps": 10,
        "source": "default_sample_actions_argument",
        "sample_kwargs": {},
        "initial_jax_key_data": [0, 0],
    }
    assert validated["transform_pipeline"]["input_order"][:5] == [
        "openpi.transforms.InjectDefaultPrompt",
        "openpi.policies.droid_policy.DroidInputs",
        "openpi.transforms.DeltaActions",
        "openpi.transforms.Normalize",
        "openpi.transforms.InjectDefaultPrompt",
    ]
    assert validated["transform_pipeline"]["output_order"] == [
        "openpi.transforms.Unnormalize",
        "openpi.transforms.AbsoluteActions",
        "openpi.policies.droid_policy.DroidOutputs",
    ]
    assert validated["policy_output"]["response_shape"] == [15, 8]
    assert validated["policy_output"]["execute_first"] == 8
    assert validated["policy_input"]["request_image_shape"] == [720, 1280, 3]
    assert validated["policy_input"]["request_image_dtype"] == "uint8"
    assert validated["policy_input"]["client_model_spatial_transform"] is None
    assert validated["policy_input"]["server_resize"] == {
        "transform": "openpi.transforms.ResizeImages",
        "implementation": "openpi_client.image_tools.resize_with_pad",
        "backend": "PIL.Image.resize",
        "method": "PIL.Image.Resampling.BILINEAR",
        "padding": "symmetric_zero",
        "target_shape": [224, 224, 3],
        "output_dtype": "uint8",
        "application_count": 1,
        "runtime_probe": _expected_resize_probe(),
        "observation_conversion": _expected_observation_conversion(),
        "model_preprocess_resize": _expected_inactive_model_resize(),
    }
    assert validated["policy_input"]["model_image_shape"] == [224, 224, 3]
    assert "non_model_only" in validated["policy_input"]["client_visualization_resize"]
    assert validated["serving"]["bind_host"] == "127.0.0.1"
    assert validated["serving"]["network_scope"] == "ipv4_loopback_only"
    assert validated["serving"]["rng_stream"]["policy_infer_wrapper"] is None
    assert validated["serving"]["rng_stream"]["serialization"] == (
        "official_server_synchronous_policy_infer_single_event_loop"
    )
    assert validated["serving"]["rng_stream"]["quiescence_barrier"] == (
        "cancel_official_run_then_async_context_close_wait_closed"
    )
    assert "/home/" not in contract.canonical_json_bytes(metadata).decode()
    assert validated["contract_sha256"] == (
        contract.pi05_droid_jointpos_server_contract_sha256(validated)
    )


def test_resize_transform_binds_openpi_client_pil_and_matches_pinned_bytes(
    monkeypatch,
):
    from openpi import transforms as openpi_transforms
    from openpi_client import image_tools as openpi_client_image_tools

    resize = openpi_transforms.ResizeImages(224, 224)
    assert contract.attest_openpi_resize_transform(resize) == _expected_resize_probe()

    wrong_namespace = SimpleNamespace(
        __name__="openpi.shared.image_tools",
        resize_with_pad=openpi_client_image_tools.resize_with_pad,
    )
    monkeypatch.setattr(openpi_transforms, "image_tools", wrong_namespace)
    with pytest.raises(ValueError, match="helper binding mismatch"):
        contract.attest_openpi_resize_transform(resize)

    def drifted_resize(_image, height, width):
        return np.zeros((height, width, 3), dtype=np.uint8)

    drifted_resize.__module__ = "openpi_client.image_tools"
    drifted_resize.__qualname__ = "resize_with_pad"
    behavior_drift = SimpleNamespace(
        __name__="openpi_client.image_tools",
        resize_with_pad=drifted_resize,
    )
    monkeypatch.setattr(openpi_transforms, "image_tools", behavior_drift)
    with pytest.raises(ValueError, match="deterministic behavior mismatch"):
        contract.attest_openpi_resize_transform(resize)


def test_loaded_norm_numeric_identity_changes_on_any_live_value_drift():
    norm_stats_type = type(
        "NormStats",
        (),
        {"__module__": "openpi.shared.normalize"},
    )

    def group(offset=0.0):
        value = norm_stats_type()
        value.mean = np.arange(32, dtype=np.float64) + offset
        value.std = np.arange(32, dtype=np.float64) + 1.0
        value.q01 = np.arange(32, dtype=np.float64) - 2.0
        value.q99 = np.arange(32, dtype=np.float64) + 2.0
        return value

    original = {"actions": group(), "state": group(0.5)}
    duplicate = {"actions": group(), "state": group(0.5)}
    drifted = {"actions": group(), "state": group(0.5)}
    drifted["actions"].q99[17] = np.nextafter(drifted["actions"].q99[17], np.inf)
    expected = contract._runtime_norm_values_sha256(original)
    assert contract._runtime_norm_values_sha256(duplicate) == expected
    assert contract._runtime_norm_values_sha256(drifted) != expected


def test_tokenizer_attestation_hashes_live_standard_and_fast_sentencepiece(
    monkeypatch,
):
    payload = b"exact sentencepiece model proto"
    monkeypatch.setattr(contract, "PI05_DROID_JOINTPOS_TOKENIZER_SIZE", len(payload))
    monkeypatch.setattr(
        contract,
        "PI05_DROID_JOINTPOS_TOKENIZER_MD5_BASE64",
        contract._bytes_md5_base64(payload),
    )
    monkeypatch.setattr(
        contract,
        "PI05_DROID_JOINTPOS_TOKENIZER_SHA256",
        contract.hashlib.sha256(payload).hexdigest(),
    )
    monkeypatch.setattr(contract, "PI05_DROID_JOINTPOS_TOKENIZER_VOCAB_SIZE", 17)

    processor_type = type(
        "SentencePieceProcessor",
        (),
        {
            "__module__": "sentencepiece",
            "serialized_model_proto": lambda self: payload,
            "vocab_size": lambda self: 17,
            "bos_id": lambda self: 2,
            "eos_id": lambda self: 1,
            "pad_id": lambda self: 0,
            "unk_id": lambda self: 3,
        },
    )
    standard_type = type(
        "PaligemmaTokenizer",
        (),
        {"__module__": "openpi.models.tokenizer"},
    )
    standard = standard_type()
    standard._max_len = 200
    standard._tokenizer = processor_type()
    standard_report = contract.attest_loaded_tokenizer_sentencepiece(standard)
    assert standard_report["processor_attribute"] == "_tokenizer"
    assert standard_report["fast_wrapper"] is False

    fast_type = type(
        "FASTTokenizer",
        (),
        {"__module__": "openpi.models.tokenizer"},
    )
    fast = fast_type()
    fast._max_len = 256
    fast._paligemma_tokenizer = processor_type()
    fast_report = contract.attest_loaded_tokenizer_sentencepiece(fast)
    assert fast_report["processor_attribute"] == "_paligemma_tokenizer"
    assert fast_report["fast_wrapper"] is True
    fast._paligemma_tokenizer.serialized_model_proto = lambda: payload + b"drift"
    with pytest.raises(ValueError, match="serialized SentencePiece proto"):
        contract.attest_loaded_tokenizer_sentencepiece(fast)


def test_tokenizer_artifact_pins_remote_generation_and_local_bytes(
    tmp_path, monkeypatch
):
    payload = b"exact GCS tokenizer object"
    tokenizer_path = tmp_path / "paligemma_tokenizer.model"
    tokenizer_path.write_bytes(payload)
    generation = "123456789"
    md5_base64 = contract._bytes_md5_base64(payload)
    sha256 = contract.hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(
        contract, "PI05_DROID_JOINTPOS_TOKENIZER_GENERATION", generation
    )
    monkeypatch.setattr(contract, "PI05_DROID_JOINTPOS_TOKENIZER_SIZE", len(payload))
    monkeypatch.setattr(
        contract, "PI05_DROID_JOINTPOS_TOKENIZER_MD5_BASE64", md5_base64
    )
    monkeypatch.setattr(contract, "PI05_DROID_JOINTPOS_TOKENIZER_SHA256", sha256)

    class Filesystem:
        def info(self, uri):
            assert uri == contract.PI05_DROID_JOINTPOS_TOKENIZER_URI
            return {
                "generation": generation,
                "size": len(payload),
                "md5Hash": md5_base64,
            }

    download = SimpleNamespace(
        maybe_download=lambda uri, **kwargs: tokenizer_path,
        fsspec=SimpleNamespace(
            core=SimpleNamespace(url_to_fs=lambda uri, **kwargs: (Filesystem(), uri))
        ),
    )
    report = contract.verify_paligemma_tokenizer_artifact(download)
    assert report["remote"]["generation"] == generation
    assert report["local"]["sha256"] == sha256
    report["remote"]["generation"] = "wrong"
    with pytest.raises(ValueError, match="artifact identity"):
        contract.validate_paligemma_tokenizer_artifact(report)


def test_handshake_rejects_metadata_transform_and_runtime_tampering():
    metadata = _metadata()
    for mutate in (
        lambda value: value.update({"extra": True}),
        lambda value: value[contract.PI05_DROID_JOINTPOS_METADATA_KEY][
            "transform_pipeline"
        ]["delta_action_mask"].__setitem__(0, False),
        lambda value: value[contract.PI05_DROID_JOINTPOS_METADATA_KEY]["openpi"][
            "runtime_attestation"
        ].update({"git_commit": "0" * 40}),
    ):
        tampered = copy.deepcopy(metadata)
        mutate(tampered)
        with pytest.raises(ValueError):
            contract.validate_pi05_droid_jointpos_server_metadata(tampered)


def test_contracts_publish_atomically_and_runtime_validation_is_not_opaque(tmp_path):
    metadata = _metadata()
    serving_path = tmp_path / contract.PI05_DROID_JOINTPOS_SERVING_CONTRACT_FILENAME
    serving = contract.publish_pi05_droid_jointpos_serving_contract(
        serving_path, metadata
    )
    assert serving["mode"] == "0444"
    assert serving["nlink"] == 1
    assert serving_path.read_bytes() == contract.canonical_json_bytes(metadata) + b"\n"
    with pytest.raises(FileExistsError):
        contract.publish_pi05_droid_jointpos_serving_contract(serving_path, metadata)

    runtime_value = _model_runtime(tmp_path, metadata)
    runtime_path = tmp_path / contract.PI05_DROID_JOINTPOS_MODEL_RUNTIME_FILENAME
    runtime = contract.publish_pi05_droid_jointpos_model_runtime(
        runtime_path, runtime_value, metadata
    )
    assert runtime["mode"] == "0444"
    validated = contract.validate_persisted_pi05_droid_jointpos_model_runtime(
        runtime_path, metadata
    )
    assert validated["sha256"] == runtime["sha256"]
    assert validated["value"]["data_config"]["image_preprocessing"] == {
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
        "resize_runtime_probe": _expected_resize_probe(),
        "observation_conversion": _expected_observation_conversion(),
        "model_preprocess_resize": _expected_inactive_model_resize(),
    }
    assert validated["value"]["server"]["request_image_contract"]["shape"] == [
        720,
        1280,
        3,
    ]
    assert (
        validated["value"]["server"]["request_image_contract"][
            "client_model_spatial_transform"
        ]
        is None
    )

    for field in ("train_config", "data_config", "policy_runtime"):
        tampered = copy.deepcopy(runtime_value)
        tampered[field][next(iter(tampered[field]))] = "tampered"
        other = tmp_path / field / contract.PI05_DROID_JOINTPOS_MODEL_RUNTIME_FILENAME
        with pytest.raises(ValueError):
            contract.publish_pi05_droid_jointpos_model_runtime(
                other, tampered, metadata
            )


def test_rng_stream_seals_exact_request_count_and_rejects_extra_request(tmp_path):
    metadata = _metadata()
    contract_hash = metadata[contract.PI05_DROID_JOINTPOS_METADATA_KEY][
        "contract_sha256"
    ]
    report = contract.make_pi05_droid_jointpos_rng_stream_report(
        server_pid=1234,
        initial_key_data=[0, 0],
        final_key_data=[11, 22],
        expected_final_key_data=[11, 22],
        expected_request_count=57,
        metadata_contract_sha256=contract_hash,
    )
    path = tmp_path / contract.PI05_DROID_JOINTPOS_RNG_STREAM_FILENAME
    artifact = contract.publish_pi05_droid_jointpos_rng_stream(path, report)
    validated = contract.validate_persisted_pi05_droid_jointpos_rng_stream(
        path,
        expected_request_count=57,
        expected_contract_sha256=contract_hash,
    )
    assert artifact["sha256"] == validated["sha256"]
    assert validated["value"]["snapshot_after_all_connection_handlers_returned"] is True
    tampered = copy.deepcopy(report)
    tampered["final_key_data"] = [11, 23]
    with pytest.raises(ValueError, match="final RNG key"):
        contract.validate_pi05_droid_jointpos_rng_stream_report(tampered)
    with pytest.raises(ValueError, match="differs from trace"):
        contract.validate_persisted_pi05_droid_jointpos_rng_stream(
            path, expected_request_count=58
        )


def test_worker_joins_trace_to_complete_official_rng_stream(tmp_path, monkeypatch):
    metadata = _metadata()
    contract_hash = metadata[contract.PI05_DROID_JOINTPOS_METADATA_KEY][
        "contract_sha256"
    ]
    serving_path = tmp_path / contract.PI05_DROID_JOINTPOS_SERVING_CONTRACT_FILENAME
    model_path = tmp_path / contract.PI05_DROID_JOINTPOS_MODEL_RUNTIME_FILENAME
    rng_path = tmp_path / contract.PI05_DROID_JOINTPOS_RNG_STREAM_FILENAME
    trace_path = tmp_path / "policy_trace_summary.json"
    contract.publish_pi05_droid_jointpos_serving_contract(serving_path, metadata)
    contract.publish_pi05_droid_jointpos_model_runtime(
        model_path, _model_runtime(tmp_path, metadata), metadata
    )
    contract.publish_pi05_droid_jointpos_rng_stream(
        rng_path,
        contract.make_pi05_droid_jointpos_rng_stream_report(
            server_pid=1234,
            initial_key_data=[0, 0],
            final_key_data=[11, 22],
            expected_final_key_data=[11, 22],
            expected_request_count=57,
            metadata_contract_sha256=contract_hash,
        ),
    )
    trace = {
        "schema_version": 4,
        "status": "pass",
        "reset_count": 1,
        "episode_lengths": [450],
        "episode_query_counts": [57],
        "cumulative_query_counts": [57],
        "query_records": 57,
        "global_query_indices_contiguous": True,
        "server_contract_sha256": contract_hash,
    }
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    monkeypatch.setattr(RNG_VERIFIER, "_expected_rng_key_data", lambda count: [11, 22])
    proof = RNG_VERIFIER.verify_rng_stream(
        rng_stream_path=rng_path,
        trace_summary_path=trace_path,
        serving_contract_path=serving_path,
        model_runtime_path=model_path,
        expected_rollouts=1,
        expected_server_pid=1234,
    )
    assert proof["request_count"] == 57
    assert proof["proof"] == (
        "trace_requests_equal_complete_official_policy_rng_stream"
    )
    trace["query_records"] = 56
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    with pytest.raises(ValueError, match="exact evaluator requests"):
        RNG_VERIFIER.verify_rng_stream(
            rng_stream_path=rng_path,
            trace_summary_path=trace_path,
            serving_contract_path=serving_path,
            model_runtime_path=model_path,
            expected_rollouts=1,
            expected_server_pid=1234,
        )


def test_host_runtime_seals_lockfiles_packages_python_and_jax(tmp_path):
    runtime = _host_runtime(tmp_path)
    assert contract.validate_openpi_host_runtime(runtime) == runtime
    assert runtime["schema_version"] == 2
    assert runtime["profile"].endswith("_v2")
    assert runtime["jax"]["nvidia_smi"] == {
        "query": list(contract.PI05_DROID_JOINTPOS_NVIDIA_SMI_QUERY),
        "uuid": GPU_UUID,
        "name": contract.PI05_DROID_JOINTPOS_NVIDIA_GPU_NAME,
        "driver_version": contract.PI05_DROID_JOINTPOS_NVIDIA_DRIVER_VERSION,
    }
    assert runtime["packages"]["required_versions"]["sentencepiece"] == "0.2.0"
    assert runtime["packages"]["required_versions"]["tensorstore"] == "0.1.74"
    assert len(runtime["packages"]["record_verification_exemptions"]) == 2
    for mutate in (
        lambda value: value["locked_source_environment"]["uv_lock"].update(
            {"sha256": "0" * 64}
        ),
        lambda value: value["packages"]["required_versions"].update({"jax": "0.5.4"}),
        lambda value: value["packages"].update(
            {"all_noneditable_files_bound_to_locked_records_or_pinned_overlap": False}
        ),
        lambda value: value["packages"]["pinned_wheel_artifacts"][0].update(
            {"sha256": "0" * 64}
        ),
        lambda value: value["jax"].update({"default_backend": "cpu"}),
        lambda value: value["jax"]["nvidia_smi"].update(
            {"driver_version": "580.95.05"}
        ),
        lambda value: value["jax"]["nvidia_smi"].update({"name": "NVIDIA A40"}),
        lambda value: value["jax"]["nvidia_smi"].update({"query": ["nvidia-smi"]}),
        lambda value: value["process_environment"]["optional"].update(
            {"NVIDIA_VISIBLE_DEVICES": "GPU-fedcba98-7654-3210-fedc-ba9876543210"}
        ),
        lambda value: value["process_environment"]["required"].update(
            {"JAX_PLATFORMS": "cpu"}
        ),
        lambda value: value["python"].update({"version": "3.12.0"}),
    ):
        tampered = copy.deepcopy(runtime)
        mutate(tampered)
        with pytest.raises(ValueError):
            contract.validate_openpi_host_runtime(tampered)
    tampered = copy.deepcopy(runtime)
    headless = next(
        item
        for item in tampered["packages"]["record_verified_distributions"]
        if item["name"] == "opencv-python-headless"
    )
    headless["record_overlap_resolutions"][0]["active_record"]["size"] += 1
    with pytest.raises(ValueError, match="RECORD validation inventory mismatch"):
        contract.validate_openpi_host_runtime(tampered)


def test_nvidia_smi_capture_uses_exact_query_and_binds_one_gpu(monkeypatch):
    observed = {}

    def fake_run(argv, **kwargs):
        observed["argv"] = argv
        observed["kwargs"] = kwargs
        return SimpleNamespace(
            stdout=(
                f" {GPU_UUID}, {contract.PI05_DROID_JOINTPOS_NVIDIA_GPU_NAME}, "
                f"{contract.PI05_DROID_JOINTPOS_NVIDIA_DRIVER_VERSION} \n"
            )
        )

    monkeypatch.setattr(contract.subprocess, "run", fake_run)
    identity = contract._capture_nvidia_smi_gpu_identity()
    assert observed == {
        "argv": list(contract.PI05_DROID_JOINTPOS_NVIDIA_SMI_QUERY),
        "kwargs": {
            "check": True,
            "capture_output": True,
            "text": True,
            "timeout": 30,
        },
    }
    assert identity == {
        "query": list(contract.PI05_DROID_JOINTPOS_NVIDIA_SMI_QUERY),
        "uuid": GPU_UUID,
        "name": contract.PI05_DROID_JOINTPOS_NVIDIA_GPU_NAME,
        "driver_version": contract.PI05_DROID_JOINTPOS_NVIDIA_DRIVER_VERSION,
    }


@pytest.mark.parametrize(
    ("stdout", "message"),
    [
        ("", "exactly one NVIDIA GPU"),
        (
            f"{GPU_UUID}, NVIDIA L40S, 580.105.08\n"
            "GPU-fedcba98-7654-3210-fedc-ba9876543210, NVIDIA L40S, 580.105.08\n",
            "exactly one NVIDIA GPU",
        ),
        (f"{GPU_UUID}, NVIDIA L40S\n", "Cannot parse"),
        ("GPU-not-a-uuid, NVIDIA L40S, 580.105.08\n", "identity mismatch"),
        (f"{GPU_UUID}, NVIDIA A40, 580.105.08\n", "identity mismatch"),
        (f"{GPU_UUID}, NVIDIA L40S, 580.95.05\n", "identity mismatch"),
    ],
)
def test_nvidia_smi_capture_rejects_count_schema_and_identity_drift(
    monkeypatch, stdout, message
):
    monkeypatch.setattr(
        contract.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=stdout),
    )
    with pytest.raises(ValueError, match=message):
        contract._capture_nvidia_smi_gpu_identity()


def test_model_runtime_v2_rejects_legacy_host_or_model_schema(tmp_path):
    metadata = _metadata()
    runtime = _model_runtime(tmp_path, metadata)
    assert runtime["schema_version"] == 2
    assert runtime["profile"] == contract.PI05_DROID_JOINTPOS_MODEL_RUNTIME_PROFILE

    legacy_model = copy.deepcopy(runtime)
    legacy_model["schema_version"] = 1
    with pytest.raises(ValueError, match="model-runtime identity mismatch"):
        contract._validate_model_runtime_value(legacy_model, metadata)

    legacy_host = copy.deepcopy(runtime)
    legacy_host["host_runtime"]["schema_version"] = 1
    with pytest.raises(ValueError, match="host-runtime identity mismatch"):
        contract._validate_model_runtime_value(legacy_host, metadata)


def test_server_loads_before_publication_and_uses_unwrapped_official_websocket():
    source = (ROOT / "scripts/polaris/serve_pi05_droid_jointpos_attested.py").read_text(
        encoding="utf-8"
    )
    checkpoint = source.index("verify_pi05_droid_jointpos_checkpoint(")
    policy = source.index("policy = policy_config.create_trained_policy(")
    runtime = source.index("validate_official_policy_runtime(")
    publication = source.index("publish_pi05_droid_jointpos_serving_contract(")
    server = source.index("server = websocket_policy_server.WebsocketPolicyServer(")
    lifecycle = source.index("_serve_then_publish_final_rng(", server)
    assert checkpoint < policy < runtime < publication < server < lifecycle
    assert "policy=policy" in source
    assert "metadata=metadata" in source
    assert "host=PI05_DROID_JOINTPOS_BIND_HOST" in source
    assert "loop.add_signal_handler(signal.SIGUSR1" in source
    assert "policy._rng" in source
    assert "policy=policy" in source
    assert "PolicyRecorder" not in source
    assert "threading" not in source
    assert "os._exit" not in source
    assert "await _run_official_server_until_quiesced" in source
    assert source.index("await _run_official_server_until_quiesced") < source.index(
        "publish_final_rng_snapshot()"
    )


def test_official_server_is_cancelled_and_drained_before_rng_snapshot():
    class FakeOfficialServer:
        def __init__(self):
            self.started = asyncio.Event()
            self.listener_open = False
            self.active_handlers = 0
            self.closed_and_drained = False

        async def run(self):
            self.listener_open = True
            self.active_handlers = 1
            self.started.set()
            try:
                await asyncio.Future()
            finally:
                self.listener_open = False
                await asyncio.sleep(0)
                self.active_handlers = 0
                self.closed_and_drained = True

    async def scenario():
        server = FakeOfficialServer()
        shutdown = asyncio.Event()
        lifecycle = asyncio.create_task(
            SERVER_MODULE._run_official_server_until_quiesced(server, shutdown)
        )
        await server.started.wait()
        shutdown.set()
        await lifecycle
        assert server.listener_open is False
        assert server.active_handlers == 0
        assert server.closed_and_drained is True

    asyncio.run(scenario())


def test_live_listener_attestation_requires_owned_ipv4_loopback_socket():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        report = contract.validate_pi05_droid_jointpos_loopback_listener(
            os.getpid(), port
        )
        assert report["bind_host"] == "127.0.0.1"
        assert report["network_scope"] == "ipv4_loopback_only"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("0.0.0.0", 0))
        listener.listen()
        with pytest.raises(ValueError, match="127.0.0.1-only"):
            contract.validate_pi05_droid_jointpos_loopback_listener(
                os.getpid(), listener.getsockname()[1]
            )


def test_worker_fail_closes_live_contract_hub_metadata_and_seed_range():
    source = (ROOT / "scripts/polaris/eval_pi05_droid_jointpos_polaris.sh").read_text(
        encoding="utf-8"
    )
    assert "serve_pi05_droid_jointpos_attested.py" in source
    assert "validate_live_server_attestation" in source
    assert source.count("validate_live_server_attestation)") == 2
    assert "pi05_droid_jointpos_serving_contract.json" in source
    assert "pi05_droid_jointpos_model_runtime.json" in source
    assert "pi05_droid_jointpos_rng_stream.json" in source
    assert "--rng-stream-output" in source
    assert "--expected-request-count" in source
    assert "validate_pi05_droid_jointpos_loopback_listener" in source
    assert 'kill -USR1 "${rng_server_pid}"' in source
    assert "verify_pi05_droid_jointpos_rng_stream.py" in source
    assert "SERVING_CONTRACT_SHA256" in source
    assert "MODEL_RUNTIME_CONTRACT_SHA256" in source
    assert "pi05_droid_jointpos_runtime.json" in source
    assert "--runtime-contract-path" in source
    assert "--policy.no-rotate-wrist-180" in source
    assert "validate_jointpos_runtime_artifact" in source
    assert "--expected-server-contract-sha256" in source
    assert "--runtime-contract" in source
    assert "runtime_contract_sha256" in source
    assert "ENVIRONMENT_SEED + ROLLOUTS - 1" in source
    assert "7bdb6f27d35b66fc848df6f94b8773bba30ea3a7f06f114100d14154a235a34b" in source
    assert 'EXPECTED_NVIDIA_DRIVER_VERSION="580.105.08"' in source
    assert 'EXPECTED_NVIDIA_GPU_NAME="NVIDIA L40S"' in source
    assert "--query-gpu=uuid,name,driver_version" in source
    assert (
        'actual_nvidia_driver_version}" == "${EXPECTED_NVIDIA_DRIVER_VERSION' in source
    )
    assert "NVIDIA_DRIVER_VERSION=%q" in source
    assert "852dd0345afb7e4d0c7526b5c327086b5132c40624ed97ff6942962126e90534" in source
    assert "accd9b67e90e510eb4ed44a789b9169df058e71ce557164f960de2d62a840e63" in source
    assert "208e0f85fc16fa32ffeca972aea0fd1b33b0c6c2a582e89ff3877823291a7754" in source
    for digest in (
        "19ac250464ccc7b723ab0a02f9c6345987b2ce74d4a8539db0df1145f8d3c306",
        "39ba7b391d90d1e8eb11759ad9240155f34c6d92589a20647687c56e00c199fc",
        "c9bfa54d0fd1de261ddcdc9c9ebdec6690ad20d33d3cfe4cef1e43e92f2affeb",
        "efea8aa97431bc8b8a61a52af2b756cce670b271f7fff53bdeb1121c6309d840",
        "aa493d82bc28d748fe5e1ad542fcff9531d1844022465c38e57681f3d74a2a2b",
        "22563e7f5a132094f1072edbf69089307c249dbc4cf915ceb3164ac58b8ae8b4",
        "8e7677d03cf57b1c163657fdcbdd674a436d5396e8b8ebcd2af7e03bff5be672",
        "08d9c737be826da7243a62e8e8249b9b879fb6b839792d0f0eba3da7c1a666f2",
        "bcaa11e0e598149c9da03cbcf3b8f30eb9d3cce975622528495f82305acc4a99",
        "c634e51498d67db91835bd9fb5d071640408737920bdec4bc1192beeb826d052",
    ):
        assert digest in source
    assert "nvidia_droid/noninstanceable.usd" in source
    assert "POLARIS_DATA_REVISION" in source


def test_setup_rebuilds_without_cache_and_cpu_verifies_packages_and_tokenizer():
    source = (ROOT / "scripts/polaris/setup_pi05_droid_jointpos_polaris.sh").read_text(
        encoding="utf-8"
    )
    assert 'rm -rf -- "${OPENPI_DIR}/.venv"' in source
    assert "uv sync" in source
    assert "--frozen --no-cache --reinstall --link-mode copy" in source
    assert "--frozen --no-cache --reinstall-package opencv-python --link-mode copy" in (
        source
    )
    assert (
        source.index("--frozen --no-cache --reinstall --link-mode copy")
        < source.index("--reinstall-package opencv-python")
        < source.index("uv pip install")
        < source.index("verify_openpi_package_environment")
    )
    assert "verify_openpi_package_environment" in source
    assert "verify_paligemma_tokenizer_artifact" in source
    assert "attest_loaded_tokenizer_sentencepiece" in source
    assert contract.PI05_DROID_JOINTPOS_TOKENIZER_GENERATION in source
    assert str(contract.PI05_DROID_JOINTPOS_TOKENIZER_SIZE) in source
    assert contract.PI05_DROID_JOINTPOS_TOKENIZER_MD5_BASE64 in source
    assert contract.PI05_DROID_JOINTPOS_TOKENIZER_SHA256 in source


def test_record_hash_classifier_accepts_only_pinned_augmax_hex_entries():
    name, version, relative_path, size, digest_hex = (
        contract.PI05_DROID_JOINTPOS_PINNED_HEX_RECORD_ENTRIES[0]
    )
    digest = bytes.fromhex(digest_hex)
    assert (
        contract._classify_distribution_record_hash(
            distribution_name=name,
            distribution_version=version,
            relative_path=relative_path,
            expected_size=size,
            expected_hash=digest_hex,
            actual_size=size,
            digest=digest,
        )
        == contract.PI05_DROID_JOINTPOS_RECORD_VALIDATION_MODES[1]
    )
    base64_digest = contract.base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    assert (
        contract._classify_distribution_record_hash(
            distribution_name="ordinary-wheel",
            distribution_version="1.0",
            relative_path="ordinary.py",
            expected_size=size,
            expected_hash=base64_digest,
            actual_size=size,
            digest=digest,
        )
        == contract.PI05_DROID_JOINTPOS_RECORD_VALIDATION_MODES[0]
    )
    with pytest.raises(ValueError, match="RECORD mismatch"):
        contract._classify_distribution_record_hash(
            distribution_name="ordinary-wheel",
            distribution_version="1.0",
            relative_path="ordinary.py",
            expected_size=size,
            expected_hash=digest_hex,
            actual_size=size,
            digest=digest,
        )


def test_record_hash_classifier_rejects_augmax_path_version_and_digest_drift():
    name, version, relative_path, size, digest_hex = (
        contract.PI05_DROID_JOINTPOS_PINNED_HEX_RECORD_ENTRIES[0]
    )
    digest = bytes.fromhex(digest_hex)
    cases = (
        (name, "0.4.2", relative_path, size, digest_hex, size, digest),
        (name, version, f"{relative_path}.changed", size, digest_hex, size, digest),
        (name, version, relative_path, size + 1, digest_hex, size, digest),
        (name, version, relative_path, size, "0" * 64, size, digest),
        (name, version, relative_path, size, digest_hex, size, bytes.fromhex("1" * 64)),
    )
    for (
        case_name,
        case_version,
        case_path,
        expected_size,
        expected_hash,
        actual_size,
        case_digest,
    ) in cases:
        with pytest.raises(ValueError, match="RECORD mismatch"):
            contract._classify_distribution_record_hash(
                distribution_name=case_name,
                distribution_version=case_version,
                relative_path=case_path,
                expected_size=expected_size,
                expected_hash=expected_hash,
                actual_size=actual_size,
                digest=case_digest,
            )


def test_pinned_augmax_hex_record_manifest_is_exact_and_complete():
    entries = contract.PI05_DROID_JOINTPOS_PINNED_HEX_RECORD_ENTRIES
    assert len(entries) == 11
    assert len(set(entries)) == len(entries)
    assert {(name, version) for name, version, _path, _size, _digest in entries} == {
        ("augmax", "0.4.1")
    }
    assert {path for _name, _version, path, _size, _digest in entries} == {
        "augmax-0.4.1.dist-info/METADATA",
        "augmax-0.4.1.dist-info/WHEEL",
        "augmax/__init__.py",
        "augmax/base.py",
        "augmax/colorspace.py",
        "augmax/functional/__init__.py",
        "augmax/functional/colorspace.py",
        "augmax/geometric.py",
        "augmax/imagelevel.py",
        "augmax/optimized.py",
        "augmax/utils.py",
    }
    assert all(
        size >= 0 and len(digest) == 64
        for _name, _version, _path, size, digest in entries
    )


def test_pinned_opencv_overlap_requires_exact_active_record(monkeypatch, tmp_path):
    profile = contract.PI05_DROID_JOINTPOS_PINNED_RECORD_OVERLAP_RESOLUTIONS[0]
    (
        losing_name,
        losing_version,
        relative_path,
        losing_size,
        losing_hash,
        active_name,
        active_version,
        active_size,
        active_hash,
    ) = profile
    active_file = tmp_path / "cv2.abi3.so"
    active_file.write_bytes(b"active OpenCV file identity is supplied by caller")

    class FakePackagePath:
        size = active_size
        hash = SimpleNamespace(mode="sha256", value=active_hash)

        def __str__(self):
            return relative_path

    class FakeDistribution:
        metadata = {"Name": active_name}
        version = active_version
        files = [FakePackagePath()]

        @staticmethod
        def locate_file(_package_path):
            return active_file

    monkeypatch.setattr(
        contract.importlib_metadata,
        "distribution",
        lambda name: FakeDistribution() if name == active_name else None,
    )
    digest = contract.base64.urlsafe_b64decode(active_hash + "=")
    resolution = contract._verify_pinned_record_overlap(
        distribution_name=losing_name,
        distribution_version=losing_version,
        relative_path=relative_path,
        expected_size=losing_size,
        expected_hash=losing_hash,
        raw_path=active_file,
        actual_size=active_size,
        digest=digest,
    )
    assert resolution == contract._expected_record_overlap_resolution(profile)
    assert resolution["active_record"]["distribution"] == "opencv-python"

    assert (
        contract._verify_pinned_record_overlap(
            distribution_name=losing_name,
            distribution_version=losing_version,
            relative_path=f"{relative_path}.changed",
            expected_size=losing_size,
            expected_hash=losing_hash,
            raw_path=active_file,
            actual_size=active_size,
            digest=digest,
        )
        is None
    )
    FakeDistribution.version = "4.11.0.87"
    with pytest.raises(ValueError, match="active distribution mismatch"):
        contract._verify_pinned_record_overlap(
            distribution_name=losing_name,
            distribution_version=losing_version,
            relative_path=relative_path,
            expected_size=losing_size,
            expected_hash=losing_hash,
            raw_path=active_file,
            actual_size=active_size,
            digest=digest,
        )


def test_pinned_wheel_artifacts_match_attested_openpi_lock():
    lock = contract.tomllib.loads(
        (ROOT / "third_party/openpi/uv.lock").read_text(encoding="utf-8")
    )
    assert contract._verify_pinned_wheel_artifacts(lock) == (
        contract._expected_pinned_wheel_artifacts()
    )
    assert {
        (artifact["name"], artifact["version"], artifact["sha256"], artifact["size"])
        for artifact in contract._expected_pinned_wheel_artifacts()
    } == {
        (
            "augmax",
            "0.4.1",
            "60f9711a4ffc08f27d1ff0783f7c51c01e6f78e20d4581d075ebf2d904ab2d14",
            17_299,
        ),
        (
            "opencv-python",
            "4.11.0.86",
            "6b02611523803495003bd87362db3e1d2a0454a6a63025dc6658a9830570aa0d",
            62_986_597,
        ),
        (
            "opencv-python-headless",
            "4.11.0.86",
            "0e0a27c19dd1f40ddff94976cfe43066fbbe9dfbb2ec1907d66c19caef42a57b",
            49_969_856,
        ),
    }


def test_every_pinned_wheel_distribution_is_required_at_setup():
    artifacts = contract._expected_pinned_wheel_artifacts()
    installed = {artifact["name"]: artifact["version"] for artifact in artifacts}
    assert (
        contract._require_pinned_wheel_distributions_installed(installed, artifacts)
        is None
    )
    for artifact in artifacts:
        missing = dict(installed)
        del missing[artifact["name"]]
        with pytest.raises(ValueError, match="missing or version-drifted"):
            contract._require_pinned_wheel_distributions_installed(missing, artifacts)
        drifted = dict(installed)
        drifted[artifact["name"]] = "0.0.0"
        with pytest.raises(ValueError, match="missing or version-drifted"):
            contract._require_pinned_wheel_distributions_installed(drifted, artifacts)


def test_checkpoint_constants_match_public_manifest_and_norm_identity():
    manifest = ROOT / "scripts/polaris/pi05_droid_jointpos_polaris_gcs_manifest.tsv"
    assert contract.hashlib.sha256(manifest.read_bytes()).hexdigest() == (
        contract.PI05_DROID_JOINTPOS_MANIFEST_SHA256
    )
    lines = manifest.read_text(encoding="ascii").splitlines()
    assert len(lines) == contract.PI05_DROID_JOINTPOS_OBJECT_COUNT
    assert sum(int(line.split("\t")[1]) for line in lines) == (
        contract.PI05_DROID_JOINTPOS_CHECKPOINT_BYTES
    )
    norm_line = next(
        line for line in lines if line.split("\t")[0].endswith("norm_stats.json")
    )
    assert norm_line.split("\t")[1:] == ["4540", "OFFWtClbLv18NC1QY7zjIQ=="]
