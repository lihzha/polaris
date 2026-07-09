"""Behavior-neutral image attestation for official pi0.5 PolaRiS evaluation."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata as importlib_metadata
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


IMAGE_PROFILE = "polaris_official_pi05_hybrid_image_path_v1"
IMAGE_SCHEMA_VERSION = 1
FINAL_IMAGE_SHAPE = (720, 1280, 3)
HALF_IMAGE_SHAPE = (360, 640, 3)
MASK_SHAPE = (720, 1280, 1)
WIRE_IMAGE_SHAPE = (224, 224, 3)
CAMERA_NAMES = ("external_cam", "wrist_cam")
CLIENT_RESIZE_PROFILE = (
    "openpi_client_resize_with_pad_pil_bilinear_720x1280_to_224x224_v1"
)
MANAGER_SOURCE_SHA256 = (
    "9381b2704e86aae6447eb2cc229471612104c5eec2acc9d913252a09159da426"
)
SPLAT_RENDERER_SOURCE_SHA256 = (
    "b9104be8620738b6fe5bea88950e0a3b721d8d23455402071745d798c662a8aa"
)
IMAGE_TOOLS_SOURCE_SHA256 = (
    "d48b4bd7f44e79fe6db8a8e07c9161144fa250be686e1245014a8b47e6171977"
)
PILLOW_VERSION = "11.3.0"
FILTER_PROBE = {
    "renderer_float_sha256": (
        "84b1702a09ef71e44c00704af15fd730cab06561c9fa19c09669807c3f012656"
    ),
    "pre_filter_uint8_sha256": (
        "4bb5ee7f3b8bdef6bd6719ae376badddb0a60f7e9d69efd5debab124df308cc3"
    ),
    "half_resolution_uint8_sha256": (
        "f89ee6cfec592254524d524473518c8ac4afff55a09b872e5c1ca68da83dc0f1"
    ),
    "post_filter_uint8_sha256": (
        "033b156a9af3732f4cd8d2d79ac6ccdb245077dbf9b56eec4c1090d665d115f2"
    ),
}
CLIENT_RESIZE_PROBE_SHA256 = (
    "4485703601c6d6fa2d256374d7b7e2fb9c60d585e8278aa4251983a96ec74cc5"
)

_CV2_CACHE: set[tuple[str, str]] = set()
_IMAGE_TOOLS_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _identity(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": _sha256(array.tobytes()),
    }


def _require_array(
    value: Any, *, shape: tuple[int, ...], dtype: Any, field: str
) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value))
    expected_dtype = np.dtype(dtype)
    if array.shape != shape or array.dtype != expected_dtype:
        raise ValueError(
            f"{field} must be {expected_dtype} {list(shape)}, got "
            f"{array.dtype} {list(array.shape)}"
        )
    return array


def _source_identity(path: Path, expected_sha256: str, field: str) -> dict[str, Any]:
    requested = Path(path)
    if requested.is_symlink() or not requested.is_file():
        raise ValueError(f"{field} source must be a regular file")
    path = requested.resolve()
    raw = path.read_bytes()
    digest = _sha256(raw)
    if digest != expected_sha256:
        raise ValueError(f"{field} source SHA-256 mismatch")
    return {"path": str(path), "size": len(raw), "sha256": digest}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _helper_source_identity() -> dict[str, Any]:
    requested = Path(__file__)
    if requested.is_symlink() or not requested.is_file():
        raise ValueError("image instrumentation helper must be a regular file")
    path = requested.resolve()
    raw = path.read_bytes()
    relative_path = path.relative_to(_repository_root())
    return {
        "path": relative_path.as_posix(),
        "size": len(raw),
        "sha256": _sha256(raw),
    }


def source_contract() -> dict[str, Any]:
    root = _repository_root()
    manager = _source_identity(
        root / "src/polaris/environments/manager_based_rl_splat_environment.py",
        MANAGER_SOURCE_SHA256,
        "manager",
    )
    manager["path"] = "src/polaris/environments/manager_based_rl_splat_environment.py"
    splat_renderer = _source_identity(
        root / "src/polaris/splat_renderer/splat_renderer.py",
        SPLAT_RENDERER_SOURCE_SHA256,
        "splat renderer",
    )
    splat_renderer["path"] = "src/polaris/splat_renderer/splat_renderer.py"
    return {
        "helper": _helper_source_identity(),
        "manager": manager,
        "splat_renderer": splat_renderer,
    }


def _probe_input() -> np.ndarray:
    y, x, channel = np.indices(FINAL_IMAGE_SHAPE, dtype=np.int32)
    values = ((x * 17 + y * 29 + channel * 53) % 263).astype(np.float32)
    return np.ascontiguousarray(values / np.float32(255.0) - np.float32(4.0 / 255.0))


def _renderer_to_uint8(renderer: Any) -> np.ndarray:
    renderer = _require_array(
        renderer, shape=FINAL_IMAGE_SHAPE, dtype=np.float32, field="renderer layer"
    )
    if not np.isfinite(renderer).all():
        raise ValueError("renderer layer contains non-finite values")
    clipped = np.clip(renderer, np.float32(0.0), np.float32(1.0))
    return np.ascontiguousarray((clipped * np.float32(255.0)).astype(np.uint8))


def _filter_uint8(pre: Any, cv2_module: Any) -> tuple[np.ndarray, np.ndarray]:
    pre = _require_array(
        pre, shape=FINAL_IMAGE_SHAPE, dtype=np.uint8, field="pre-filter layer"
    )
    if getattr(cv2_module, "INTER_LINEAR", None) != 1:
        raise ValueError("cv2.INTER_LINEAR constant mismatch")
    half = cv2_module.resize(pre, (640, 360), interpolation=cv2_module.INTER_LINEAR)
    half = _require_array(
        half, shape=HALF_IMAGE_SHAPE, dtype=np.uint8, field="half-filter layer"
    )
    post = cv2_module.resize(half, (1280, 720), interpolation=cv2_module.INTER_LINEAR)
    post = _require_array(
        post, shape=FINAL_IMAGE_SHAPE, dtype=np.uint8, field="post-filter layer"
    )
    return half, post


def _filter_probe(cv2_module: Any) -> dict[str, str]:
    renderer = _probe_input()
    pre = _renderer_to_uint8(renderer)
    half, post = _filter_uint8(pre, cv2_module)
    return {
        "renderer_float_sha256": _identity(renderer)["sha256"],
        "pre_filter_uint8_sha256": _identity(pre)["sha256"],
        "half_resolution_uint8_sha256": _identity(half)["sha256"],
        "post_filter_uint8_sha256": _identity(post)["sha256"],
    }


def _validate_cv2(cv2_module: Any) -> dict[str, Any]:
    version = str(getattr(cv2_module, "__version__", ""))
    path = str(Path(getattr(cv2_module, "__file__", "")).resolve())
    package_version = importlib_metadata.version("opencv-python-headless")
    if version != "4.11.0":
        raise ValueError("OpenCV version mismatch")
    if package_version != "4.11.0.86":
        raise ValueError("opencv-python-headless package version mismatch")
    key = (version, path)
    if key not in _CV2_CACHE:
        if _filter_probe(cv2_module) != FILTER_PROBE:
            raise ValueError("OpenCV image-filter probe mismatch")
        _CV2_CACHE.add(key)
    return {
        "provider": "opencv-python-headless",
        "package_version": package_version,
        "opencv_version": version,
        "module_path": path,
        "interpolation": "cv2.INTER_LINEAR",
        "interpolation_value": 1,
        "probe": dict(FILTER_PROBE),
    }


def filter_renderer_layer(
    renderer: Any, *, cv2_module: Any, camera_name: str
) -> tuple[np.ndarray, dict[str, Any]]:
    runtime = _validate_cv2(cv2_module)
    renderer = _require_array(
        renderer,
        shape=FINAL_IMAGE_SHAPE,
        dtype=np.float32,
        field=f"{camera_name} renderer layer",
    )
    pre = _renderer_to_uint8(renderer)
    half, post = _filter_uint8(pre, cv2_module)
    return post, {
        "source": "splat_renderer_output",
        "cv2_runtime": runtime,
        "renderer_float": _identity(renderer),
        "pre_filter_uint8": _identity(pre),
        "half_resolution_uint8": _identity(half),
        "post_filter_splat_uint8": _identity(post),
    }


def _image_tools_runtime(image_tools_module: Any) -> dict[str, Any]:
    source = _source_identity(
        Path(getattr(image_tools_module, "__file__", "")),
        IMAGE_TOOLS_SOURCE_SHA256,
        "OpenPI image tools",
    )
    pillow_version = importlib_metadata.version("Pillow")
    if pillow_version != PILLOW_VERSION:
        raise ValueError(f"OpenPI client Pillow version mismatch: {pillow_version!r}")
    key = (source["path"], source["sha256"])
    if key in _IMAGE_TOOLS_CACHE:
        return copy.deepcopy(_IMAGE_TOOLS_CACHE[key])
    probe = _renderer_to_uint8(_probe_input())
    resized = image_tools_module.resize_with_pad(probe, 224, 224)
    resized = _require_array(
        resized, shape=WIRE_IMAGE_SHAPE, dtype=np.uint8, field="client resize probe"
    )
    if _identity(resized)["sha256"] != CLIENT_RESIZE_PROBE_SHA256:
        raise ValueError("OpenPI client resize probe mismatch")
    if image_tools_module.resize_with_pad(resized, 224, 224) is not resized:
        raise ValueError("OpenPI server 224 resize is not the pinned identity shortcut")
    report = {
        "profile": CLIENT_RESIZE_PROFILE,
        "implementation": "openpi_client.image_tools.resize_with_pad",
        "backend": "PIL.Image.resize",
        "method": "PIL.Image.Resampling.BILINEAR",
        "padding": "symmetric_zero",
        "source": source,
        "probe_output_sha256": CLIENT_RESIZE_PROBE_SHA256,
        "server_224_to_224": "early_return_same_array_no_pixel_change",
        "pillow_version": pillow_version,
        "pillow_module": "PIL.Image",
    }
    _IMAGE_TOOLS_CACHE[key] = copy.deepcopy(report)
    return report


def resize_final_composite_for_wire(
    final: Any, *, image_tools_module: Any, camera_name: str
) -> tuple[np.ndarray, dict[str, Any]]:
    final = _require_array(
        final,
        shape=FINAL_IMAGE_SHAPE,
        dtype=np.uint8,
        field=f"{camera_name} final composite",
    )
    runtime = _image_tools_runtime(image_tools_module)
    wire = image_tools_module.resize_with_pad(final, 224, 224)
    wire = _require_array(
        wire,
        shape=WIRE_IMAGE_SHAPE,
        dtype=np.uint8,
        field=f"{camera_name} wire request",
    )
    return wire, {
        "profile": CLIENT_RESIZE_PROFILE,
        "runtime": runtime,
        "input_final_composite": _identity(final),
        "wire_request": _identity(wire),
    }


def static_image_contract() -> dict[str, Any]:
    value = {
        "schema_version": IMAGE_SCHEMA_VERSION,
        "profile": IMAGE_PROFILE,
        "camera_names": list(CAMERA_NAMES),
        "environment": {
            "renderer_shape": list(FINAL_IMAGE_SHAPE),
            "renderer_dtype": "float32",
            "conversion": "clip_0_1_multiply_float32_255_astype_uint8",
            "splat_filter": "cv2_INTER_LINEAR_720x1280_to_360x640_to_720x1280",
            "filter_scope": "splat_layer_only_before_sim_composite",
            "semantic_mask": "np_where_semantic_id_ge_2_int64",
            "composite": "np_where_mask_sim_rgb_else_filtered_splat",
            "missing_renderer_fallback": "np_zeros_like_sim_rgb_after_filter_stage",
            "final_shape": list(FINAL_IMAGE_SHAPE),
            "final_dtype": "uint8",
            "filter_probe": dict(FILTER_PROBE),
        },
        "client_wire": {
            "profile": CLIENT_RESIZE_PROFILE,
            "input_shape": list(FINAL_IMAGE_SHAPE),
            "output_shape": list(WIRE_IMAGE_SHAPE),
            "implementation": "openpi_client.image_tools.resize_with_pad",
            "source_sha256": IMAGE_TOOLS_SOURCE_SHA256,
            "probe_output_sha256": CLIENT_RESIZE_PROBE_SHA256,
            "server_resize": "invoked_then_early_return_identity_at_224",
        },
        "training_distribution": {
            "sim_dataset": "polaris_droid_cotrain_dataset/1.0.0",
            "features_revision": "e6348a1ec7de061b5bacf113f8efbf096a6ce43e",
            "features_sha256": (
                "7b36373767b08bfacb8f7a47e1f373e7a676474024e0823b304e8249e895d03b"
            ),
            "encoded_camera_storage": "uint8_JPEG_180x320x3",
            "real_droid_camera_storage": "uint8_JPEG_180x320x3",
            "train_path": "decode_JPEG_180x320_then_model_ResizeImages_to_224",
            "eval_path": (
                "raw_hybrid_720x1280_then_client_ResizeImages_to_224_then_"
                "wire224_server_identity"
            ),
            "official_domain_gap": (
                "published_protocol_preserved_no_eval_coercion_to_train_storage"
            ),
        },
        "sources": source_contract(),
        "instrumentation": {
            "profile": "jointpos_instance_bound_method_observer_v1",
            "scope": "audited_DroidJointPos_environment_instance_only",
            "wrapped_methods": [
                "splat_renderer.render",
                "render_splat",
                "get_robot_from_sim",
                "custom_render",
            ],
            "returned_pixels_mutated": False,
        },
    }
    value["contract_sha256"] = _sha256(_canonical_bytes(value))
    return value


def _root_env(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def install_jointpos_image_instrumentation(env: Any) -> dict[str, Any]:
    """Wrap one live environment without changing any returned image bytes."""

    root = _root_env(env)
    if hasattr(root, "_pi05_jointpos_image_instrumentation"):
        raise RuntimeError("joint-position image instrumentation installed twice")
    source_contract()
    cv2_module = sys.modules.get("cv2")
    if cv2_module is None:
        raise RuntimeError(
            "manager OpenCV module was not imported before instrumentation"
        )
    if len(root.splat_renderer.pcds) <= 0:
        raise ValueError("official joint-position evaluation requires nonempty splats")

    original_splat_render = root.splat_renderer.render
    original_render_splat = root.render_splat
    original_robot = root.get_robot_from_sim
    original_custom = root.custom_render
    state: dict[str, Any] = {
        "raw": None,
        "filter": None,
        "filter_arrays": None,
        "robot": None,
        "latest": None,
    }

    def wrapped_splat_render(extrinsics_dict):
        result = original_splat_render(extrinsics_dict)
        state["raw"] = {
            name: np.ascontiguousarray(value.detach().cpu().numpy()).copy()
            for name, value in result.items()
        }
        return result

    def wrapped_render_splat():
        result = original_render_splat()
        raw = state.get("raw")
        if not isinstance(raw, dict) or set(raw) != set(result):
            raise ValueError("renderer/filter camera sets differ")
        evidence = {}
        arrays = {}
        for name, actual in result.items():
            expected, stages = filter_renderer_layer(
                raw[name], cv2_module=cv2_module, camera_name=name
            )
            actual = _require_array(
                actual,
                shape=FINAL_IMAGE_SHAPE,
                dtype=np.uint8,
                field=f"{name} manager filtered splat",
            )
            if not np.array_equal(expected, actual):
                raise ValueError(f"{name} manager splat filter differs from contract")
            evidence[name] = stages
            arrays[name] = actual.copy()
        state["filter"] = evidence
        state["filter_arrays"] = arrays
        return result

    def wrapped_robot():
        result = original_robot()
        state["robot"] = {
            name: {
                "rgb": np.ascontiguousarray(value["rgb"]).copy(),
                "mask": np.ascontiguousarray(value["mask"]).copy(),
            }
            for name, value in result.items()
        }
        return result

    def wrapped_custom(expensive: bool, transform_static: bool = False):
        result = original_custom(expensive, transform_static=transform_static)
        if not expensive:
            state["latest"] = None
            return result
        filters = state.get("filter")
        robot = state.get("robot")
        if not isinstance(filters, dict) or not isinstance(robot, dict):
            raise ValueError("expensive image render lacks captured layer evidence")
        if set(result) != set(robot):
            raise ValueError("final/simulator camera sets differ")
        cameras = {}
        for name, final in result.items():
            sim = _require_array(
                robot[name]["rgb"],
                shape=FINAL_IMAGE_SHAPE,
                dtype=np.uint8,
                field=f"{name} simulator RGB",
            )
            mask = _require_array(
                robot[name]["mask"],
                shape=MASK_SHAPE,
                dtype=np.int64,
                field=f"{name} semantic mask",
            )
            if name in filters:
                result_background = filters[name]["post_filter_splat_uint8"]
                background_array = _require_array(
                    state["filter_arrays"][name],
                    shape=FINAL_IMAGE_SHAPE,
                    dtype=np.uint8,
                    field=f"{name} filtered background",
                )
                background_source = "filtered_splat"
                renderer_stages = filters[name]
                if _identity(background_array) != result_background:
                    raise ValueError(f"{name} filtered background identity drift")
            else:
                background_array = np.zeros_like(sim)
                background_source = "official_missing_renderer_zero_fallback"
                renderer_stages = None
            expected_final = np.where(mask, sim, background_array)
            unique_mask = set(np.unique(mask).tolist())
            true_pixels = int(np.count_nonzero(mask))
            total_pixels = int(mask.size)
            if not unique_mask <= {0, 1} or not 0 < true_pixels < total_pixels:
                raise ValueError(
                    f"{name} semantic mask must contain both background and sim pixels"
                )
            final_array = _require_array(
                final,
                shape=FINAL_IMAGE_SHAPE,
                dtype=np.uint8,
                field=f"{name} final composite",
            )
            if not np.array_equal(expected_final, final_array):
                raise ValueError(
                    f"{name} final manager composite differs from contract"
                )
            cameras[name] = {
                "schema_version": IMAGE_SCHEMA_VERSION,
                "profile": IMAGE_PROFILE,
                "contract_sha256": static_image_contract()["contract_sha256"],
                "camera_name": name,
                "background_source": background_source,
                "renderer_stages": renderer_stages,
                "composite_background_uint8": _identity(background_array),
                "composite_mask_int64": _identity(mask),
                "composite_mask_coverage": {
                    "true_pixel_count": true_pixels,
                    "total_pixel_count": total_pixels,
                    "true_fraction": true_pixels / total_pixels,
                },
                "sim_rgb_layer_uint8": _identity(sim),
                "final_composite_uint8": _identity(final_array),
            }
        state["latest"] = cameras
        return result

    root.splat_renderer.render = wrapped_splat_render
    root.render_splat = wrapped_render_splat
    root.get_robot_from_sim = wrapped_robot
    root.custom_render = wrapped_custom
    root._pi05_jointpos_image_instrumentation = state
    return {"profile": IMAGE_PROFILE, "contract": static_image_contract()}


def get_jointpos_image_evidence(env: Any, obs: Any) -> dict[str, Any]:
    root = _root_env(env)
    state = getattr(root, "_pi05_jointpos_image_instrumentation", None)
    if not isinstance(state, dict) or not isinstance(state.get("latest"), dict):
        raise ValueError("no current expensive joint-position image evidence")
    evidence = state["latest"]
    if set(evidence) != set(CAMERA_NAMES):
        raise ValueError("joint-position evidence camera set mismatch")
    if not isinstance(obs, dict) or not isinstance(obs.get("splat"), dict):
        raise ValueError("joint-position observation lacks splat cameras")
    for name in CAMERA_NAMES:
        validate_camera_evidence(
            evidence[name], camera_name=name, require_filtered_splat=True
        )
        final = _require_array(
            obs["splat"].get(name),
            shape=FINAL_IMAGE_SHAPE,
            dtype=np.uint8,
            field=f"{name} policy observation",
        )
        if _identity(final) != evidence[name]["final_composite_uint8"]:
            raise ValueError(f"{name} policy observation hash differs from composite")
    return copy.deepcopy(evidence)


def _validate_identity(
    value: Any, *, shape: tuple[int, ...], dtype: str, field: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"shape", "dtype", "sha256"}:
        raise ValueError(f"{field} identity schema mismatch")
    if value["shape"] != list(shape) or value["dtype"] != dtype:
        raise ValueError(f"{field} identity shape/dtype mismatch")
    digest = value["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{field} identity SHA-256 mismatch")
    return copy.deepcopy(value)


def validate_camera_evidence(
    value: Any, *, camera_name: str, require_filtered_splat: bool = True
) -> dict[str, Any]:
    required = {
        "schema_version",
        "profile",
        "contract_sha256",
        "camera_name",
        "background_source",
        "renderer_stages",
        "composite_background_uint8",
        "composite_mask_int64",
        "composite_mask_coverage",
        "sim_rgb_layer_uint8",
        "final_composite_uint8",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"{camera_name} camera evidence schema mismatch")
    if (
        value["schema_version"] != IMAGE_SCHEMA_VERSION
        or value["profile"] != IMAGE_PROFILE
        or value["contract_sha256"] != static_image_contract()["contract_sha256"]
        or value["camera_name"] != camera_name
    ):
        raise ValueError(f"{camera_name} camera evidence identity mismatch")
    source = value["background_source"]
    if require_filtered_splat and source != "filtered_splat":
        raise ValueError(f"{camera_name} unexpectedly used renderer fallback")
    if source == "filtered_splat":
        stages = value["renderer_stages"]
        if not isinstance(stages, dict) or set(stages) != {
            "source",
            "cv2_runtime",
            "renderer_float",
            "pre_filter_uint8",
            "half_resolution_uint8",
            "post_filter_splat_uint8",
        }:
            raise ValueError(f"{camera_name} renderer evidence schema mismatch")
        if stages["source"] != "splat_renderer_output":
            raise ValueError(f"{camera_name} renderer source mismatch")
        cv2_runtime = stages["cv2_runtime"]
        if (
            not isinstance(cv2_runtime, dict)
            or set(cv2_runtime)
            != {
                "provider",
                "package_version",
                "opencv_version",
                "module_path",
                "interpolation",
                "interpolation_value",
                "probe",
            }
            or cv2_runtime["provider"] != "opencv-python-headless"
            or cv2_runtime["package_version"] != "4.11.0.86"
            or cv2_runtime["opencv_version"] != "4.11.0"
            or not Path(cv2_runtime["module_path"]).is_absolute()
            or cv2_runtime["interpolation"] != "cv2.INTER_LINEAR"
            or cv2_runtime["interpolation_value"] != 1
            or cv2_runtime["probe"] != FILTER_PROBE
        ):
            raise ValueError(f"{camera_name} cv2 runtime mismatch")
        _validate_identity(
            stages["renderer_float"],
            shape=FINAL_IMAGE_SHAPE,
            dtype="float32",
            field=f"{camera_name} renderer",
        )
        _validate_identity(
            stages["pre_filter_uint8"],
            shape=FINAL_IMAGE_SHAPE,
            dtype="uint8",
            field=f"{camera_name} pre-filter",
        )
        _validate_identity(
            stages["half_resolution_uint8"],
            shape=HALF_IMAGE_SHAPE,
            dtype="uint8",
            field=f"{camera_name} half-filter",
        )
        post = _validate_identity(
            stages["post_filter_splat_uint8"],
            shape=FINAL_IMAGE_SHAPE,
            dtype="uint8",
            field=f"{camera_name} post-filter",
        )
        if value["composite_background_uint8"] != post:
            raise ValueError(f"{camera_name} filtered background identity mismatch")
    elif source == "official_missing_renderer_zero_fallback":
        if value["renderer_stages"] is not None:
            raise ValueError(f"{camera_name} fallback has renderer stages")
    else:
        raise ValueError(f"{camera_name} background source mismatch")
    _validate_identity(
        value["composite_background_uint8"],
        shape=FINAL_IMAGE_SHAPE,
        dtype="uint8",
        field=f"{camera_name} composite background",
    )
    _validate_identity(
        value["composite_mask_int64"],
        shape=MASK_SHAPE,
        dtype="int64",
        field=f"{camera_name} composite mask",
    )
    coverage = value["composite_mask_coverage"]
    if (
        not isinstance(coverage, dict)
        or set(coverage) != {"true_pixel_count", "total_pixel_count", "true_fraction"}
        or coverage["total_pixel_count"] != MASK_SHAPE[0] * MASK_SHAPE[1]
        or not 0 < coverage["true_pixel_count"] < coverage["total_pixel_count"]
        or coverage["true_fraction"]
        != coverage["true_pixel_count"] / coverage["total_pixel_count"]
    ):
        raise ValueError(f"{camera_name} mask coverage mismatch")
    _validate_identity(
        value["sim_rgb_layer_uint8"],
        shape=FINAL_IMAGE_SHAPE,
        dtype="uint8",
        field=f"{camera_name} sim layer",
    )
    _validate_identity(
        value["final_composite_uint8"],
        shape=FINAL_IMAGE_SHAPE,
        dtype="uint8",
        field=f"{camera_name} final composite",
    )
    return copy.deepcopy(value)


def validate_client_resize_evidence(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "profile",
        "runtime",
        "input_final_composite",
        "wire_request",
    }:
        raise ValueError("client resize evidence schema mismatch")
    if value["profile"] != CLIENT_RESIZE_PROFILE:
        raise ValueError("client resize evidence profile mismatch")
    _validate_identity(
        value["input_final_composite"],
        shape=FINAL_IMAGE_SHAPE,
        dtype="uint8",
        field="client resize input",
    )
    _validate_identity(
        value["wire_request"],
        shape=WIRE_IMAGE_SHAPE,
        dtype="uint8",
        field="client resize wire",
    )
    runtime = value["runtime"]
    if (
        not isinstance(runtime, dict)
        or set(runtime)
        != {
            "profile",
            "implementation",
            "backend",
            "method",
            "padding",
            "source",
            "probe_output_sha256",
            "server_224_to_224",
            "pillow_version",
            "pillow_module",
        }
        or runtime["profile"] != CLIENT_RESIZE_PROFILE
        or runtime["implementation"] != "openpi_client.image_tools.resize_with_pad"
        or runtime["backend"] != "PIL.Image.resize"
        or runtime["method"] != "PIL.Image.Resampling.BILINEAR"
        or runtime["padding"] != "symmetric_zero"
        or runtime["probe_output_sha256"] != CLIENT_RESIZE_PROBE_SHA256
        or runtime["server_224_to_224"] != "early_return_same_array_no_pixel_change"
        or runtime["pillow_version"] != PILLOW_VERSION
        or runtime["pillow_module"] != "PIL.Image"
    ):
        raise ValueError("client resize runtime mismatch")
    source = runtime["source"]
    if (
        not isinstance(source, dict)
        or set(source) != {"path", "size", "sha256"}
        or not Path(source["path"]).is_absolute()
        or type(source["size"]) is not int
        or source["size"] <= 0
        or source["sha256"] != IMAGE_TOOLS_SOURCE_SHA256
    ):
        raise ValueError("client resize source identity mismatch")
    return copy.deepcopy(value)


__all__ = [
    "CAMERA_NAMES",
    "CLIENT_RESIZE_PROFILE",
    "FINAL_IMAGE_SHAPE",
    "HALF_IMAGE_SHAPE",
    "IMAGE_PROFILE",
    "IMAGE_SCHEMA_VERSION",
    "MASK_SHAPE",
    "PILLOW_VERSION",
    "WIRE_IMAGE_SHAPE",
    "filter_renderer_layer",
    "get_jointpos_image_evidence",
    "install_jointpos_image_instrumentation",
    "resize_final_composite_for_wire",
    "source_contract",
    "static_image_contract",
    "validate_camera_evidence",
    "validate_client_resize_evidence",
]
