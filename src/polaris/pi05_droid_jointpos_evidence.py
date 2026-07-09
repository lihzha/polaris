"""Immutable close transaction for official pi0.5 joint-position evaluation.

The simulator writes its large trace, metrics, videos, and logs incrementally.  This
module is intentionally host-runnable and seals those completed outputs only after
the evaluator and policy server have exited.  One canonical immutable manifest then
binds every accepted output by absolute path, byte count, SHA-256, mode, and link
count.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import stat
from typing import Any

from polaris.pi05_droid_jointpos_runtime import (
    PI05_DROID_JOINTPOS_GRAPHICS_CV2_LOADER_SEARCH_SAFETY_PROFILE,
    PI05_DROID_JOINTPOS_GRAPHICS_LIBRARY_IDENTITIES,
    PI05_DROID_JOINTPOS_GRAPHICS_RUNTIME_PROFILE,
    PI05_DROID_JOINTPOS_GRAPHICS_RUNTIME_SHA256,
    expected_graphics_cv2_loader_identity,
    validate_jointpos_runtime_artifact,
)
from polaris.pi05_droid_jointpos_serving_contract import (
    PI05_DROID_JOINTPOS_CHECKPOINT_URI,
    PI05_DROID_JOINTPOS_NVIDIA_DRIVER_VERSION,
    PI05_DROID_JOINTPOS_NUMPYDANTIC_WARNING_FILTER,
    PI05_DROID_JOINTPOS_SERVER_MODEL_RESIZE,
    PI05_DROID_JOINTPOS_VULKAN_ICD_CONTAINER_PATH,
    PI05_DROID_JOINTPOS_VULKAN_ICD_SHA256,
    validate_persisted_pi05_droid_jointpos_model_runtime,
    validate_persisted_pi05_droid_jointpos_rng_stream,
    validate_persisted_pi05_droid_jointpos_serving_contract,
)
from polaris.pi05_droid_jointpos_video import (
    PI05_DROID_JOINTPOS_PYXIS_SHA256,
    validate_persisted_video_report,
)
from polaris.pi05_droid_jointpos_immutable import (
    file_sha256,
    fsync_directory,
    publish_immutable_json,
    validate_immutable_file,
    validate_immutable_json,
)


PI05_DROID_JOINTPOS_EVIDENCE_PROFILE = (
    "openpi_pi05_droid_jointpos_polaris_evidence_transaction_v8"
)
PI05_DROID_JOINTPOS_EVIDENCE_MANIFEST = "pi05_droid_jointpos_evidence_manifest.json"

_RUN_ARTIFACTS = {
    "asset_dependency_manifest": "polaris_asset_dependency_manifest.json",
    "checkpoint_verification": "checkpoint_verification.json",
    "commands": "commands.sh",
    "gpu_environment": "gpu_environment.csv",
    "model_runtime_contract": "pi05_droid_jointpos_model_runtime.json",
    "polaris_git_status": "polaris_git_status.txt",
    "polaris_submodules": "polaris_submodules.txt",
    "policy_server_log": "policy_server.log",
    "request_proof": "pi05_droid_jointpos_request_proof.json",
    "rng_stream": "pi05_droid_jointpos_rng_stream.json",
    "run_metadata": "run_metadata.env",
    "serving_contract": "pi05_droid_jointpos_serving_contract.json",
}
_TASK_ARTIFACTS = {
    "eval_log": "eval.log",
    "metrics_csv": "eval_results.csv",
    "policy_trace": "policy_traces.jsonl",
    "runtime_contract": "pi05_droid_jointpos_runtime.json",
    "trace_summary": "policy_trace_summary.json",
    "video_validation": "pi05_droid_jointpos_video_validation.json",
    "video_validation_log": "video_validation.log",
}
_IDENTITY_KEYS = ("path", "size", "sha256", "mode", "nlink")
_EXPECTED_PROMPTS = {
    "DROID-BlockStackKitchen": "Place and stack the blocks on top of the green tray",
    "DROID-FoodBussing": "Put all the foods in the bowl",
    "DROID-PanClean": "Use the yellow sponge to scrub the blue handle frying pan",
    "DROID-MoveLatteCup": "put the latte art cup on top of the cutting board",
    "DROID-OrganizeTools": "put the scissor into the large container",
    "DROID-TapeIntoContainer": "put the tape into the container",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _hex_sha256(value: Any, field: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{field} must be one lowercase SHA-256",
    )
    return value


def _strict_json(path: Path, field: str) -> Any:
    try:
        return json.loads(
            path.read_bytes(),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{field} contains non-finite JSON: {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} is not readable strict JSON") from error


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _validate_package_run_metadata(
    path: Path, model_host_runtime: Any
) -> dict[str, Any]:
    """Bind the outer evaluation package probes to the attested model process."""

    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError("run metadata is not readable UTF-8") from error
    _require(text.endswith("\n"), "run metadata must end in one newline")
    encoded: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, raw_value = line.partition("=")
        _require(
            separator == "="
            and key
            and key[0].isalpha()
            and key.replace("_", "A").isalnum()
            and key.upper() == key,
            "run metadata assignment schema mismatch",
        )
        _require(key not in encoded, f"duplicate run metadata assignment: {key}")
        encoded[key] = raw_value

    required = (
        "PYTHONWARNINGS",
        "PREFLIGHT_PACKAGE_ENVIRONMENT_SHA256",
        "POSTRUN_PACKAGE_ENVIRONMENT_SHA256",
    )
    decoded: dict[str, str] = {}
    for key in required:
        _require(key in encoded, f"run metadata is missing {key}")
        try:
            words = shlex.split(encoded[key], comments=False, posix=True)
        except ValueError as error:
            raise ValueError(
                f"run metadata {key} is not valid shell quoting"
            ) from error
        _require(len(words) == 1, f"run metadata {key} must encode one value")
        decoded[key] = words[0]

    _require(
        isinstance(model_host_runtime, dict)
        and isinstance(model_host_runtime.get("packages"), dict)
        and isinstance(model_host_runtime.get("package_import_stability"), dict),
        "model runtime lacks package-import stability",
    )
    package_sha256 = hashlib.sha256(
        _canonical_json_bytes(model_host_runtime["packages"])
    ).hexdigest()
    stability = model_host_runtime["package_import_stability"]
    _require(
        stability
        == {
            "preimport_sha256": package_sha256,
            "postimport_sha256": package_sha256,
            "unchanged": True,
        },
        "model package-import stability differs from package report",
    )
    _require(
        decoded["PREFLIGHT_PACKAGE_ENVIRONMENT_SHA256"] == package_sha256
        and decoded["POSTRUN_PACKAGE_ENVIRONMENT_SHA256"] == package_sha256,
        "outer evaluation package probes differ from model runtime",
    )
    _require(
        decoded["PYTHONWARNINGS"] == PI05_DROID_JOINTPOS_NUMPYDANTIC_WARNING_FILTER,
        "run metadata warning filter mismatch",
    )
    return {
        "openpi_package_environment_sha256": package_sha256,
        "openpi_package_preflight_postrun_unchanged": True,
        "numpydantic_warning_filter": decoded["PYTHONWARNINGS"],
    }


def _regular_single_link(path: Path, field: str) -> os.stat_result:
    if path.is_symlink():
        raise ValueError(f"{field} must not be a symlink")
    try:
        result = path.stat()
    except OSError as error:
        raise ValueError(f"{field} is unavailable") from error
    if not stat.S_ISREG(result.st_mode) or result.st_nlink != 1:
        raise ValueError(f"{field} must be one regular link")
    if result.st_size <= 0:
        raise ValueError(f"{field} must be nonempty")
    return result


def seal_file(path: Path, *, field: str) -> dict[str, Any]:
    """Hash, chmod, fsync, reread, and bind one completed output file."""

    path = Path(path)
    before = _regular_single_link(path, field)
    digest_before = file_sha256(path)
    path.chmod(0o444)
    with path.open("rb") as source:
        os.fsync(source.fileno())
    fsync_directory(path.parent)
    after = _regular_single_link(path, field)
    digest_after = file_sha256(path)
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or stat.S_IMODE(after.st_mode) != 0o444
        or digest_before != digest_after
    ):
        raise ValueError(f"{field} changed while being sealed")
    return {
        "path": str(path.resolve()),
        "size": after.st_size,
        "sha256": digest_after,
        "mode": "0444",
        "nlink": 1,
    }


def _identity(value: dict[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in _IDENTITY_KEYS}


def _validate_identity(
    value: Any, *, expected_path: Path, field: str
) -> dict[str, Any]:
    _require(
        isinstance(value, dict) and set(value) == set(_IDENTITY_KEYS),
        f"{field} identity schema mismatch",
    )
    expected_resolved = str(expected_path.resolve(strict=True))
    _require(value.get("path") == expected_resolved, f"{field} path mismatch")
    observed = validate_immutable_file(expected_path)
    _require(_identity(observed) == value, f"{field} identity mismatch")
    return dict(value)


def _expected_video_paths(task_dir: Path, expected_rollouts: int) -> list[Path]:
    _require(
        type(expected_rollouts) is int and expected_rollouts > 0,
        "expected rollouts must be one positive integer",
    )
    expected = [task_dir / f"episode_{index}.mp4" for index in range(expected_rollouts)]
    actual = set(task_dir.glob("episode_*.mp4"))
    _require(actual == set(expected), "rollout video filename/count mismatch")
    return expected


def _expected_terminal_image_paths(
    task_dir: Path, expected_rollouts: int
) -> list[Path]:
    expected = [
        task_dir / f"episode_{index}_terminal.png" for index in range(expected_rollouts)
    ]
    actual = set(task_dir.glob("episode_*_terminal.png"))
    _require(actual == set(expected), "terminal-image filename/count mismatch")
    return expected


def _artifact_paths(
    run_dir: Path, task_dir: Path, expected_rollouts: int
) -> tuple[dict[str, Path], list[Path], list[Path]]:
    paths = {
        **{name: run_dir / relative for name, relative in _RUN_ARTIFACTS.items()},
        **{name: task_dir / relative for name, relative in _TASK_ARTIFACTS.items()},
    }
    return (
        paths,
        _expected_video_paths(task_dir, expected_rollouts),
        _expected_terminal_image_paths(task_dir, expected_rollouts),
    )


def _validate_trace_summary(
    summary_path: Path,
    *,
    trace_identity: dict[str, Any],
    metrics_identity: dict[str, Any],
    expected_rollouts: int,
    independently_audited: dict[str, Any],
) -> dict[str, Any]:
    summary = _strict_json(summary_path, "trace summary")
    _require(isinstance(summary, dict), "trace summary must be one object")
    _require(summary.get("schema_version") == 4, "trace summary schema mismatch")
    _require(summary.get("status") == "pass", "trace summary did not pass")
    _require(
        summary.get("trace_sha256") == trace_identity["sha256"],
        "trace summary does not bind the sealed trace",
    )
    _require(
        summary.get("metrics_sha256") == metrics_identity["sha256"],
        "trace summary does not bind the sealed metrics CSV",
    )
    _require(
        summary.get("reset_count") == expected_rollouts,
        "trace summary rollout count mismatch",
    )
    _require(
        summary.get("episode_lengths") == [450] * expected_rollouts,
        "trace summary episode lengths mismatch",
    )
    _require(
        summary.get("episode_query_counts") == [57] * expected_rollouts
        and summary.get("cumulative_query_counts")
        == [57 * (index + 1) for index in range(expected_rollouts)]
        and summary.get("query_records") == 57 * expected_rollouts
        and summary.get("global_query_indices_contiguous") is True,
        "trace summary request stream mismatch",
    )
    _require(
        summary.get("native_image_shape") == [720, 1280, 3]
        and summary.get("request_image_shape") == [720, 1280, 3]
        and summary.get("request_image_dtype") == "uint8"
        and summary.get("client_model_spatial_transform") is None
        and summary.get("server_model_resize")
        == PI05_DROID_JOINTPOS_SERVER_MODEL_RESIZE
        and summary.get("model_image_shape") == [224, 224, 3]
        and summary.get("visualization_image_shape") == [224, 224, 3]
        and summary.get("visualization_is_model_input") is False,
        "trace summary image-preprocessing boundary mismatch",
    )
    terminal_hashes = summary.get("terminal_visualization_sha256")
    _require(
        summary.get("terminal_visualization_shape") == [224, 448, 3]
        and summary.get("terminal_visualization_dtype") == "uint8"
        and summary.get("terminal_visualization_source")
        == "post_action450_returned_expensive_splat_observation"
        and isinstance(terminal_hashes, list)
        and len(terminal_hashes) == expected_rollouts,
        "trace summary terminal-visualization boundary mismatch",
    )
    for index, digest in enumerate(terminal_hashes):
        _hex_sha256(digest, f"episode {index} terminal visualization")
    _hex_sha256(summary.get("server_contract_sha256"), "trace server contract")
    _hex_sha256(summary.get("runtime_contract_sha256"), "trace runtime contract")
    _require(
        _canonical_json_bytes(summary) == _canonical_json_bytes(independently_audited),
        "sealed trace/CSV re-audit differs from the persisted trace summary",
    )
    return summary


def _independently_audit_sealed_trace(
    paths: dict[str, Path],
    *,
    environment: str,
    expected_environment_seed: int,
    server_contract_sha256: str,
) -> dict[str, Any]:
    """Rerun the raw trace/CSV validator only after both inputs are sealed."""

    if environment not in _EXPECTED_PROMPTS:
        raise ValueError("Unsupported joint-position evidence environment")
    if (
        type(expected_environment_seed) is not int
        or not 0 <= expected_environment_seed <= 2**32 - 1
    ):
        raise ValueError("Evidence environment seed must be one uint32 integer")
    validator_script = (
        Path(__file__).resolve().parents[2] / "scripts/polaris/validate_pi05_trace.py"
    )
    specification = importlib.util.spec_from_file_location(
        "validate_pi05_jointpos_trace_for_sealed_evidence", validator_script
    )
    if specification is None or specification.loader is None:
        raise ValueError("Sealed policy-trace validator is unavailable")
    validator_module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(validator_module)
    audited = validator_module.audit_trace(
        paths["policy_trace"],
        expected_prompt=_EXPECTED_PROMPTS[environment],
        metrics_csv=paths["metrics_csv"],
        expected_environment_seed=expected_environment_seed,
        expected_server_contract_sha256=server_contract_sha256,
        runtime_contract_path=paths["runtime_contract"],
    )
    if not isinstance(audited, dict) or audited.get("status") != "pass":
        raise ValueError(
            "Sealed policy trace and metrics did not pass independent audit"
        )
    return audited


def _validate_specialized_contracts(
    paths: dict[str, Path],
    expected_rollouts: int,
    video_identities: list[dict[str, Any]],
    terminal_image_identities: list[dict[str, Any]],
    terminal_pixel_sha256: list[str],
) -> dict[str, Any]:
    serving = validate_persisted_pi05_droid_jointpos_serving_contract(
        paths["serving_contract"]
    )
    model = validate_persisted_pi05_droid_jointpos_model_runtime(
        paths["model_runtime_contract"], serving["value"]
    )
    runtime = validate_jointpos_runtime_artifact(paths["runtime_contract"])
    asset_script = (
        Path(__file__).resolve().parents[2]
        / "scripts/polaris/polaris_asset_dependency_manifest.py"
    )
    specification = importlib.util.spec_from_file_location(
        "polaris_asset_dependency_manifest", asset_script
    )
    if specification is None or specification.loader is None:
        raise ValueError("PolaRiS asset dependency validator is unavailable")
    asset_module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(asset_module)
    asset = asset_module.validate_asset_manifest_artifact(
        paths["asset_dependency_manifest"]
    )
    contract = serving["value"]["ego_lap_pi05_droid_jointpos_contract"]
    expected_request_count = 57 * expected_rollouts
    rng = validate_persisted_pi05_droid_jointpos_rng_stream(
        paths["rng_stream"],
        expected_request_count=expected_request_count,
        expected_contract_sha256=contract["contract_sha256"],
    )
    verifier_script = (
        Path(__file__).resolve().parents[2]
        / "scripts/polaris/verify_pi05_droid_jointpos_rng_stream.py"
    )
    verifier_specification = importlib.util.spec_from_file_location(
        "verify_pi05_droid_jointpos_rng_stream", verifier_script
    )
    if verifier_specification is None or verifier_specification.loader is None:
        raise ValueError("Policy RNG-stream verifier is unavailable")
    verifier_module = importlib.util.module_from_spec(verifier_specification)
    verifier_specification.loader.exec_module(verifier_module)
    independently_verified_proof = verifier_module.verify_rng_stream(
        rng_stream_path=paths["rng_stream"],
        trace_summary_path=paths["trace_summary"],
        serving_contract_path=paths["serving_contract"],
        model_runtime_path=paths["model_runtime_contract"],
        expected_rollouts=expected_rollouts,
        expected_server_pid=rng["value"]["server_pid"],
    )
    proof_artifact = validate_immutable_json(paths["request_proof"])
    _require(
        proof_artifact["value"] == independently_verified_proof,
        "sealed request proof differs from the independent JAX recurrence",
    )
    _require(
        model["value"]["server"]["metadata_contract_sha256"]
        == contract["contract_sha256"],
        "model-runtime and serving contracts differ",
    )
    _require(
        model["value"]["server"]["expected_request_count"] == expected_request_count,
        "model-runtime request count differs from the evaluation protocol",
    )
    video = validate_persisted_video_report(
        paths["video_validation"],
        expected_rollouts=expected_rollouts,
        expected_video_identities=video_identities,
        expected_terminal_image_identities=terminal_image_identities,
        expected_terminal_pixel_sha256=terminal_pixel_sha256,
    )
    media_tools = video["value"]["execution_environment"]["tools"]
    gpu_vulkan = _validate_gpu_vulkan_runtime_agreement(
        model["value"]["host_runtime"], runtime["execution_environment"]
    )
    package_runtime = _validate_package_run_metadata(
        paths["run_metadata"], model["value"]["host_runtime"]
    )
    return {
        "asset_manifest_sha256": asset["manifest_sha256"],
        "asset_tree_sha256": asset["tree_sha256"],
        "server_contract_sha256": contract["contract_sha256"],
        "runtime_contract_sha256": runtime["runtime_sha256"],
        "rng_stream_artifact_sha256": rng["sha256"],
        "request_proof_artifact_sha256": proof_artifact["sha256"],
        "video_validation_artifact_sha256": video["sha256"],
        "video_container_image_sha256": PI05_DROID_JOINTPOS_PYXIS_SHA256,
        "video_ffprobe_sha256": media_tools["ffprobe"]["sha256"],
        "video_ffmpeg_sha256": media_tools["ffmpeg"]["sha256"],
        **gpu_vulkan,
        **package_runtime,
    }


def _validate_gpu_vulkan_runtime_agreement(
    model_host_runtime: Any, simulator_execution_environment: Any
) -> dict[str, Any]:
    """Bind the model and simulator to one allocated GPU and Vulkan runtime."""

    _require(
        isinstance(model_host_runtime, dict)
        and isinstance(model_host_runtime.get("jax"), dict)
        and isinstance(model_host_runtime["jax"].get("nvidia_smi"), dict),
        "model runtime lacks NVIDIA identity",
    )
    _require(
        isinstance(simulator_execution_environment, dict)
        and isinstance(simulator_execution_environment.get("nvidia_smi"), dict)
        and isinstance(simulator_execution_environment.get("vulkan"), dict),
        "simulator runtime lacks GPU/Vulkan identity",
    )
    model_gpu = model_host_runtime["jax"]["nvidia_smi"]
    simulator_gpu = simulator_execution_environment["nvidia_smi"]
    identity_fields = ("uuid", "name", "driver_version")
    _require(
        all(
            model_gpu.get(field) == simulator_gpu.get(field)
            for field in identity_fields
        ),
        "model and simulator NVIDIA identities differ",
    )
    _require(
        model_gpu.get("name") == "NVIDIA L40S"
        and model_gpu.get("driver_version")
        == PI05_DROID_JOINTPOS_NVIDIA_DRIVER_VERSION,
        "evaluation did not use the canonical L40S driver runtime",
    )
    vulkan = simulator_execution_environment["vulkan"]
    _require(
        vulkan.get("vk_driver_files") == PI05_DROID_JOINTPOS_VULKAN_ICD_CONTAINER_PATH
        and isinstance(vulkan.get("icd"), dict)
        and vulkan["icd"].get("path") == PI05_DROID_JOINTPOS_VULKAN_ICD_CONTAINER_PATH
        and vulkan["icd"].get("sha256") == PI05_DROID_JOINTPOS_VULKAN_ICD_SHA256,
        "simulator Vulkan ICD identity differs from the canonical runtime",
    )
    graphics = simulator_execution_environment.get("graphics_runtime")
    expected_cv2_loader = expected_graphics_cv2_loader_identity()
    cv2_loader = graphics.get("cv2_loader") if isinstance(graphics, dict) else None
    cv2_module = cv2_loader.get("module") if isinstance(cv2_loader, dict) else None
    cv2_loader_search_safety = (
        cv2_loader.get("loader_search_safety") if isinstance(cv2_loader, dict) else None
    )
    cv2_module_static = (
        {
            name: item
            for name, item in cv2_module.items()
            if name not in {"native_maps_device", "native_maps_inode"}
        }
        if isinstance(cv2_module, dict)
        else None
    )
    _require(
        isinstance(graphics, dict)
        and graphics.get("profile") == PI05_DROID_JOINTPOS_GRAPHICS_RUNTIME_PROFILE
        and isinstance(cv2_loader, dict)
        and cv2_loader.get("profile") == expected_cv2_loader["profile"]
        and cv2_loader.get("files") == expected_cv2_loader["files"]
        and cv2_module_static == expected_cv2_loader["module"]
        and isinstance(cv2_module.get("native_maps_device"), str)
        and isinstance(cv2_module.get("native_maps_inode"), int)
        and isinstance(cv2_loader_search_safety, dict)
        and set(cv2_loader_search_safety)
        == {
            "profile",
            "working_directory",
            "working_directory_binding",
            "working_directory_read_only",
            "normalized_cv2_binary_path",
            "normalized_cv2_binary_path_exists",
            "working_directory_library_candidates",
        }
        and cv2_loader_search_safety.get("profile")
        == PI05_DROID_JOINTPOS_GRAPHICS_CV2_LOADER_SEARCH_SAFETY_PROFILE
        and cv2_loader_search_safety.get("working_directory_binding")
        == "equals_runtime_module_repository_root"
        and cv2_loader_search_safety.get("working_directory_read_only") is True
        and cv2_loader_search_safety.get("normalized_cv2_binary_path")
        == "/.venv/lib/python3.11/lib64"
        and cv2_loader_search_safety.get("normalized_cv2_binary_path_exists") is False
        and cv2_loader_search_safety.get("working_directory_library_candidates") == []
        and isinstance(graphics.get("libraries"), list)
        and len(graphics["libraries"])
        == len(PI05_DROID_JOINTPOS_GRAPHICS_LIBRARY_IDENTITIES)
        and graphics.get("graphics_runtime_sha256")
        == PI05_DROID_JOINTPOS_GRAPHICS_RUNTIME_SHA256,
        "simulator mapped graphics-runtime identity differs from the canonical runtime",
    )
    return {
        "nvidia_gpu_uuid": model_gpu["uuid"],
        "nvidia_gpu_name": model_gpu["name"],
        "nvidia_driver_version": model_gpu["driver_version"],
        "vulkan_icd_container_path": vulkan["icd"]["path"],
        "vulkan_icd_sha256": vulkan["icd"]["sha256"],
        "graphics_runtime_profile": graphics["profile"],
        "graphics_runtime_sha256": graphics["graphics_runtime_sha256"],
        "graphics_cv2_loader_profile": graphics["cv2_loader"]["profile"],
        "graphics_cv2_module_identity": graphics["cv2_loader"]["module"],
        "graphics_cv2_loader_search_safety": graphics["cv2_loader"][
            "loader_search_safety"
        ],
        "graphics_cv2_loader_files": graphics["cv2_loader"]["files"],
        "graphics_library_count": len(graphics["libraries"]),
    }


def _validate_closed_contracts(
    paths: dict[str, Path],
    *,
    video_identities: list[dict[str, Any]],
    terminal_image_identities: list[dict[str, Any]],
    trace_identity: dict[str, Any],
    metrics_identity: dict[str, Any],
    environment: str,
    expected_environment_seed: int,
    expected_rollouts: int,
) -> dict[str, Any]:
    summary_candidate = _strict_json(paths["trace_summary"], "trace summary")
    _require(
        isinstance(summary_candidate, dict)
        and isinstance(summary_candidate.get("terminal_visualization_sha256"), list),
        "trace summary lacks terminal-visualization digests",
    )
    contracts = _validate_specialized_contracts(
        paths,
        expected_rollouts,
        video_identities,
        terminal_image_identities,
        summary_candidate["terminal_visualization_sha256"],
    )
    independently_audited = _independently_audit_sealed_trace(
        paths,
        environment=environment,
        expected_environment_seed=expected_environment_seed,
        server_contract_sha256=contracts["server_contract_sha256"],
    )
    summary = _validate_trace_summary(
        paths["trace_summary"],
        trace_identity=trace_identity,
        metrics_identity=metrics_identity,
        expected_rollouts=expected_rollouts,
        independently_audited=independently_audited,
    )
    _require(
        summary["server_contract_sha256"] == contracts["server_contract_sha256"]
        and summary["runtime_contract_sha256"] == contracts["runtime_contract_sha256"],
        "trace and immutable contract identities differ",
    )
    contracts["sealed_trace_csv_reaudit_sha256"] = hashlib.sha256(
        _canonical_json_bytes(independently_audited)
    ).hexdigest()
    return contracts


def _manifest_value(
    *,
    run_dir: Path,
    task_dir: Path,
    environment: str,
    expected_environment_seed: int,
    expected_rollouts: int,
    polaris_commit: str,
    artifacts: dict[str, dict[str, Any]],
    videos: list[dict[str, Any]],
    terminal_images: list[dict[str, Any]],
    contracts: dict[str, Any],
) -> dict[str, Any]:
    _require(
        isinstance(environment, str) and environment.startswith("DROID-"),
        "environment identity is invalid",
    )
    _require(
        isinstance(polaris_commit, str)
        and len(polaris_commit) == 40
        and all(character in "0123456789abcdef" for character in polaris_commit),
        "PolaRiS commit must be one full lowercase Git SHA",
    )
    return {
        "schema_version": 1,
        "profile": PI05_DROID_JOINTPOS_EVIDENCE_PROFILE,
        "status": "pass",
        "checkpoint_uri": PI05_DROID_JOINTPOS_CHECKPOINT_URI,
        "polaris_commit": polaris_commit,
        "environment": environment,
        "environment_seed": expected_environment_seed,
        "expected_rollouts": expected_rollouts,
        "run_dir": str(run_dir.resolve()),
        "task_dir": str(task_dir.resolve()),
        "contracts": contracts,
        "artifacts": artifacts,
        "videos": videos,
        "terminal_images": terminal_images,
    }


def validate_evidence_manifest(
    manifest_path: Path,
    *,
    run_dir: Path,
    task_dir: Path,
    environment: str,
    expected_environment_seed: int,
    expected_rollouts: int,
    polaris_commit: str,
) -> dict[str, Any]:
    """Reopen an immutable manifest and every artifact at its exact expected path."""

    run_dir = Path(run_dir)
    task_dir = Path(task_dir)
    _require(
        not run_dir.is_symlink() and not task_dir.is_symlink(),
        "evidence directories must not be symlinks",
    )
    run_dir = run_dir.resolve(strict=True)
    task_dir = task_dir.resolve(strict=True)
    _require(task_dir.parent == run_dir, "task directory must be a direct run child")
    manifest_path = Path(manifest_path)
    _require(
        manifest_path == run_dir / PI05_DROID_JOINTPOS_EVIDENCE_MANIFEST,
        "unexpected evidence manifest path",
    )
    manifest_artifact = validate_immutable_json(manifest_path)
    value = manifest_artifact["value"]
    _require(
        isinstance(value, dict)
        and set(value)
        == {
            "schema_version",
            "profile",
            "status",
            "checkpoint_uri",
            "polaris_commit",
            "environment",
            "environment_seed",
            "expected_rollouts",
            "run_dir",
            "task_dir",
            "contracts",
            "artifacts",
            "videos",
            "terminal_images",
        },
        "evidence manifest schema mismatch",
    )
    _require(
        value["schema_version"] == 1
        and value["profile"] == PI05_DROID_JOINTPOS_EVIDENCE_PROFILE
        and value["status"] == "pass"
        and value["checkpoint_uri"] == PI05_DROID_JOINTPOS_CHECKPOINT_URI
        and value["polaris_commit"] == polaris_commit
        and value["environment"] == environment
        and value["environment_seed"] == expected_environment_seed
        and value["expected_rollouts"] == expected_rollouts
        and value["run_dir"] == str(run_dir)
        and value["task_dir"] == str(task_dir),
        "evidence manifest run identity mismatch",
    )
    paths, video_paths, terminal_image_paths = _artifact_paths(
        run_dir, task_dir, expected_rollouts
    )
    artifacts = value["artifacts"]
    _require(
        isinstance(artifacts, dict) and set(artifacts) == set(paths),
        "evidence artifact inventory mismatch",
    )
    for name, path in paths.items():
        _validate_identity(artifacts[name], expected_path=path, field=name)
    videos = value["videos"]
    _require(
        isinstance(videos, list) and len(videos) == expected_rollouts,
        "evidence video inventory mismatch",
    )
    for index, path in enumerate(video_paths):
        _validate_identity(videos[index], expected_path=path, field=f"video {index}")
    terminal_images = value["terminal_images"]
    _require(
        isinstance(terminal_images, list) and len(terminal_images) == expected_rollouts,
        "evidence terminal-image inventory mismatch",
    )
    for index, path in enumerate(terminal_image_paths):
        _validate_identity(
            terminal_images[index],
            expected_path=path,
            field=f"terminal image {index}",
        )
    contracts = _validate_closed_contracts(
        paths,
        video_identities=videos,
        terminal_image_identities=terminal_images,
        trace_identity=artifacts["policy_trace"],
        metrics_identity=artifacts["metrics_csv"],
        environment=environment,
        expected_environment_seed=expected_environment_seed,
        expected_rollouts=expected_rollouts,
    )
    _require(value["contracts"] == contracts, "evidence contract identities mismatch")
    return {
        "manifest": _identity(manifest_artifact),
        "value": value,
    }


def finalize_evidence(
    *,
    run_dir: Path,
    task_dir: Path,
    environment: str,
    expected_environment_seed: int,
    expected_rollouts: int,
    polaris_commit: str,
) -> dict[str, Any]:
    """Seal all completed outputs, publish the manifest, and validate it again."""

    run_dir = Path(run_dir)
    task_dir = Path(task_dir)
    _require(
        not run_dir.is_symlink() and not task_dir.is_symlink(),
        "evidence directories must not be symlinks",
    )
    run_dir = run_dir.resolve(strict=True)
    task_dir = task_dir.resolve(strict=True)
    _require(task_dir.parent == run_dir, "task directory must be a direct run child")
    for directory, field in ((run_dir, "run directory"), (task_dir, "task directory")):
        _require(
            not directory.is_symlink() and directory.is_dir(), f"{field} is invalid"
        )
    for marker in (
        run_dir / "SUCCESS",
        run_dir / "FAILED",
        task_dir / "SUCCESS",
        task_dir / "FAILED",
    ):
        _require(
            not marker.exists() and not marker.is_symlink(), "premature terminal marker"
        )
    manifest_path = run_dir / PI05_DROID_JOINTPOS_EVIDENCE_MANIFEST
    _require(
        not manifest_path.exists() and not manifest_path.is_symlink(),
        "evidence manifest already exists",
    )
    paths, video_paths, terminal_image_paths = _artifact_paths(
        run_dir, task_dir, expected_rollouts
    )
    artifacts = {
        name: seal_file(path, field=name) for name, path in sorted(paths.items())
    }
    videos = [
        seal_file(path, field=f"rollout video {index}")
        for index, path in enumerate(video_paths)
    ]
    terminal_images = [
        seal_file(path, field=f"terminal image {index}")
        for index, path in enumerate(terminal_image_paths)
    ]
    contracts = _validate_closed_contracts(
        paths,
        video_identities=videos,
        terminal_image_identities=terminal_images,
        trace_identity=artifacts["policy_trace"],
        metrics_identity=artifacts["metrics_csv"],
        environment=environment,
        expected_environment_seed=expected_environment_seed,
        expected_rollouts=expected_rollouts,
    )
    value = _manifest_value(
        run_dir=run_dir,
        task_dir=task_dir,
        environment=environment,
        expected_environment_seed=expected_environment_seed,
        expected_rollouts=expected_rollouts,
        polaris_commit=polaris_commit,
        artifacts=artifacts,
        videos=videos,
        terminal_images=terminal_images,
        contracts=contracts,
    )
    publish_immutable_json(manifest_path, value)
    return validate_evidence_manifest(
        manifest_path,
        run_dir=run_dir,
        task_dir=task_dir,
        environment=environment,
        expected_environment_seed=expected_environment_seed,
        expected_rollouts=expected_rollouts,
        polaris_commit=polaris_commit,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--expected-environment-seed", type=int, required=True)
    parser.add_argument("--expected-rollouts", type=int, required=True)
    parser.add_argument("--polaris-commit", required=True)
    args = parser.parse_args()
    result = finalize_evidence(
        run_dir=args.run_dir,
        task_dir=args.task_dir,
        environment=args.environment,
        expected_environment_seed=args.expected_environment_seed,
        expected_rollouts=args.expected_rollouts,
        polaris_commit=args.polaris_commit,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()


__all__ = [
    "PI05_DROID_JOINTPOS_EVIDENCE_MANIFEST",
    "PI05_DROID_JOINTPOS_EVIDENCE_PROFILE",
    "finalize_evidence",
    "seal_file",
    "validate_evidence_manifest",
]
