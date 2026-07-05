"""Real-runtime FoodBussing splat/image contract smoke without a policy server.

This script launches Isaac Sim and exercises the production
``ManagerBasedRLSplatEnv.custom_render`` path.  It deliberately makes no
checkpoint, policy-action, task-metric, or canary-authorization claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import traceback
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


SCHEMA_VERSION = 1
PROFILE = "polaris_foodbussing_splat_image_contract_smoke_v1"
SCOPE = "image_contract_only_no_checkpoint_policy_action_metric_or_canary"
ENVIRONMENT = "DROID-FoodBussing"
FOODBUSSING_INSTRUCTION = "Put all the foods in the bowl"
CAMERA_KEYS = ("external_cam", "wrist_cam")
NATIVE_SHAPE = (720, 1280, 3)
PREPROCESSED_SHAPE = (224, 224, 3)
RESIZED_CONTENT_SHAPE = (126, 224, 3)
PAD_TOP = 49
PAD_BOTTOM = 49
FOODBUSSING_SCENE_SHA256 = (
    "82cd641e422935b394ce7ea7b6be55214c9952a2544000222921e544c409b489"
)
FOODBUSSING_SCENE_SIZE_BYTES = 14914
FOODBUSSING_INITIAL_CONDITIONS_SHA256 = (
    "40091faee14f692350220871d30705294f21f17ae3d2974cd3c09a34d560f5de"
)
FOODBUSSING_INITIAL_CONDITIONS_SIZE_BYTES = 173951
POLARIS_HUB_REVISION = "8c7e4103e266ef83d8b1ad2e9a63116edd5f155b"
FOODBUSSING_METADATA_SHA256 = {
    "initial_conditions": "852dd0345afb7e4d0c7526b5c327086b5132c40624ed97ff6942962126e90534",
    "scene": "accd9b67e90e510eb4ed44a789b9169df058e71ce557164f960de2d62a840e63",
}
HEX64 = frozenset("0123456789abcdef")

TOP_LEVEL_FIELDS = {
    "schema_version",
    "profile",
    "scope",
    "stage",
    "status",
    "promotion_authorized",
    "host_finalization_required",
    "source",
    "launch_provenance",
    "result",
    "failure",
    "close_failures",
    "persistence_failures",
}
SOURCE_FIELDS = {
    "root",
    "commit",
    "tree",
    "tracked_clean",
    "expected_commit",
    "expected_tree",
}
LAUNCH_PROVENANCE_FIELDS = {
    "container_image",
    "saved_sbatch",
    "expected_scene_sha256",
}
EXPECTED_FILE_FIELDS = {"path", "expected_sha256", "exists", "size_bytes"}
RESULT_FIELDS = {
    "environment",
    "production_path",
    "contracts",
    "cameras",
    "artifacts",
}
ENVIRONMENT_FIELDS = {
    "id",
    "runtime_class",
    "scene_file",
    "initial_conditions_file",
    "initial_condition_index",
    "instruction",
    "hub_revision",
    "hub_metadata",
    "camera_sensor_keys",
    "renderer_camera_keys",
}
PRODUCTION_PATH_FIELDS = {
    "bound_render_splat_is_production_method",
    "events",
    "renderer_render_calls",
    "render_splat_calls",
    "get_robot_from_sim_calls",
}
CONTRACT_FIELDS = {
    "renderer_conversion",
    "robot_compositing",
    "ego_lap_preprocessing",
    "msgpack_roundtrip",
    "removed_resize_counterfactual",
}
CAMERA_RESULT_FIELDS = {
    "raw_renderer",
    "native_uint8",
    "robot_rgb",
    "robot_mask",
    "composited_uint8",
    "preprocessed_uint8",
    "conversion",
    "compositing",
    "preprocessing",
    "counterfactual",
}
ARRAY_SUMMARY_FIELDS = {"shape", "dtype", "min", "max"}
ARTIFACT_IDENTITY_FIELDS = {
    "path",
    "size_bytes",
    "sha256",
    "mode",
    "kind",
    "array",
}
READY_FIELDS = {"schema_version", "profile", "stage", "raw_result"}
RAW_IDENTITY_FIELDS = {"path", "size_bytes", "sha256", "mode"}


class ContractError(RuntimeError):
    """Raised when the image smoke observes contract drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _is_int(value: Any) -> bool:
    return type(value) is int


def _is_number(value: Any) -> bool:
    return (type(value) in (int, float)) and math.isfinite(float(value))


def _is_sha256(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and not (set(value) - HEX64)


def _is_git_oid(value: Any) -> bool:
    return type(value) is str and len(value) == 40 and not (set(value) - HEX64)


def _strict_json_value(value: Any) -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Non-finite float cannot be serialized")
        return value
    if isinstance(value, np.generic):
        return _strict_json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    raise TypeError(f"Unsupported strict JSON value: {type(value).__name__}")


def strict_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            _strict_json_value(payload),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def strict_json_loads(payload: bytes, *, field: str) -> Any:
    try:
        return json.loads(
            payload,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value!r}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ContractError(f"{field} is not strict JSON: {error}") from error


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_immutable_bytes(path: Path, payload: bytes) -> None:
    """Publish new bytes without replacing any prior artifact."""

    path = path.resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def publish_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    publish_immutable_bytes(path, strict_json_bytes(payload))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _mode(path: Path) -> str:
    return f"{stat.S_IMODE(path.stat().st_mode):04o}"


def file_identity(path: Path, *, kind: str, array: np.ndarray | None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
        "mode": _mode(resolved),
        "kind": kind,
        "array": None if array is None else array_summary(array),
    }


def raw_file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
        "mode": _mode(resolved),
    }


def array_summary(array: np.ndarray) -> dict[str, Any]:
    value = np.asarray(array)
    _require(value.size > 0, "artifact array must be nonempty")
    _require(np.isfinite(value).all(), "artifact array must be finite")
    minimum = value.min().item()
    maximum = value.max().item()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "min": minimum,
        "max": maximum,
    }


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, np.asarray(array), allow_pickle=False)
    return stream.getvalue()


def _png_bytes(array: np.ndarray) -> bytes:
    value = np.asarray(array)
    _require(value.dtype == np.uint8, "PNG array must be uint8")
    _require(
        value.ndim == 3 and value.shape[2] == 3,
        f"PNG array must be RGB; got {value.shape}",
    )
    stream = io.BytesIO()
    Image.fromarray(value, mode="RGB").save(stream, format="PNG", optimize=False)
    return stream.getvalue()


class ArtifactWriter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir.resolve(strict=False)
        self.identities: dict[str, dict[str, Any]] = {}

    def array(self, name: str, array: np.ndarray) -> None:
        _require(name not in self.identities, f"duplicate artifact key {name}")
        path = self.output_dir / f"{name}.npy"
        value = np.asarray(array)
        publish_immutable_bytes(path, _npy_bytes(value))
        self.identities[name] = file_identity(path, kind="npy", array=value)

    def png(self, name: str, array: np.ndarray) -> None:
        _require(name not in self.identities, f"duplicate artifact key {name}")
        path = self.output_dir / f"{name}.png"
        value = np.asarray(array)
        publish_immutable_bytes(path, _png_bytes(value))
        self.identities[name] = file_identity(path, kind="png", array=value)

    def bytes(self, name: str, suffix: str, payload: bytes, *, kind: str) -> None:
        _require(name not in self.identities, f"duplicate artifact key {name}")
        path = self.output_dir / f"{name}{suffix}"
        publish_immutable_bytes(path, payload)
        self.identities[name] = file_identity(path, kind=kind, array=None)


def _exception_evidence(error: BaseException) -> dict[str, str]:
    try:
        message = str(error)
    except BaseException:
        message = "<unprintable exception>"
    try:
        formatted = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
    except BaseException:
        formatted = "<traceback unavailable>"
    return {
        "type": f"{type(error).__module__}.{type(error).__qualname__}",
        "message": message,
        "traceback": formatted,
    }


def _git_output(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def capture_source_identity(
    root: Path, *, expected_commit: str, expected_tree: str
) -> dict[str, Any]:
    resolved = root.resolve(strict=True)
    commit = _git_output(resolved, "rev-parse", "HEAD")
    tree = _git_output(resolved, "rev-parse", "HEAD^{tree}")
    tracked_status = _git_output(
        resolved, "status", "--porcelain=v1", "--untracked-files=no"
    )
    _require(commit == expected_commit, "source commit drift")
    _require(tree == expected_tree, "source tree drift")
    _require(tracked_status == "", "source tracked state is dirty")
    return {
        "root": str(resolved),
        "commit": commit,
        "tree": tree,
        "tracked_clean": True,
        "expected_commit": expected_commit,
        "expected_tree": expected_tree,
    }


def expected_file_reference(path: Path, expected_sha256: str) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    _require(_is_sha256(expected_sha256), "expected file SHA-256 is malformed")
    return {
        "path": str(resolved),
        "expected_sha256": expected_sha256,
        "exists": True,
        "size_bytes": resolved.stat().st_size,
    }


def independent_renderer_conversion(raw: np.ndarray) -> np.ndarray:
    """Expected conversion, intentionally independent of the production helper."""

    value = np.asarray(raw)
    _require(value.dtype.kind == "f", "renderer output must have floating dtype")
    _require(value.shape == NATIVE_SHAPE, f"renderer shape drift: {value.shape}")
    _require(np.isfinite(value).all(), "renderer output contains non-finite values")
    return (np.clip(value, 0.0, 1.0) * 255).astype(np.uint8)


def old_half_down_up_counterfactual(image: np.ndarray) -> np.ndarray:
    """Reproduce the removed OpenCV operation for diagnostic comparison only."""

    import cv2  # Imported only for the explicitly labeled counterfactual.

    value = np.asarray(image)
    _require(value.shape == NATIVE_SHAPE, "counterfactual input shape drift")
    _require(value.dtype == np.uint8, "counterfactual input dtype drift")
    half = cv2.resize(value, (value.shape[1] // 2, value.shape[0] // 2))
    return cv2.resize(half, (half.shape[1] * 2, half.shape[0] * 2))


def _preview_panel(image: np.ndarray, label: str) -> np.ndarray:
    source = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    source.thumbnail((640, 340), resample=Image.Resampling.BILINEAR)
    panel = Image.new("RGB", (640, 360), color=(24, 24, 24))
    left = (640 - source.width) // 2
    top = 20 + (340 - source.height) // 2
    panel.paste(source, (left, top))
    ImageDraw.Draw(panel).text((8, 4), label, fill=(255, 255, 255))
    return np.asarray(panel, dtype=np.uint8)


def make_contact_sheet(images: Mapping[str, Mapping[str, np.ndarray]]) -> np.ndarray:
    rows = []
    for camera in CAMERA_KEYS:
        camera_images = images[camera]
        panels = [
            _preview_panel(camera_images["native"], f"{camera}: native"),
            _preview_panel(camera_images["old"], f"{camera}: removed half down/up"),
            _preview_panel(camera_images["diff_preview"], f"{camera}: abs diff x4"),
            _preview_panel(camera_images["composited"], f"{camera}: composited"),
            _preview_panel(camera_images["preprocessed"], f"{camera}: Ego-LAP 224"),
        ]
        rows.append(np.concatenate(panels, axis=1))
    return np.concatenate(rows, axis=0)


def expected_artifact_keys() -> set[str]:
    keys = {"contact_sheet", "ego_lap_request_msgpack"}
    for camera in CAMERA_KEYS:
        keys.update(
            {
                f"{camera}_renderer_raw_float",
                f"{camera}_native_uint8",
                f"{camera}_robot_rgb",
                f"{camera}_robot_mask",
                f"{camera}_composited_uint8",
                f"{camera}_preprocessed_uint8",
                f"{camera}_old_half_down_up_uint8",
                f"{camera}_old_abs_diff_uint8",
                f"{camera}_native_png",
                f"{camera}_composited_png",
                f"{camera}_preprocessed_png",
                f"{camera}_old_half_down_up_png",
                f"{camera}_old_abs_diff_x4_png",
            }
        )
    return keys


def _validate_array_summary(value: Any, *, field: str) -> None:
    _require(isinstance(value, dict), f"{field} must be an object")
    _require(set(value) == ARRAY_SUMMARY_FIELDS, f"{field} schema drift")
    _require(
        isinstance(value["shape"], list)
        and value["shape"]
        and all(_is_int(item) and item > 0 for item in value["shape"]),
        f"{field}.shape drift",
    )
    _require(type(value["dtype"]) is str and value["dtype"], f"{field}.dtype drift")
    _require(_is_number(value["min"]), f"{field}.min drift")
    _require(_is_number(value["max"]), f"{field}.max drift")
    _require(value["min"] <= value["max"], f"{field} range drift")


def _validate_artifact_identity(value: Any, *, field: str) -> None:
    _require(isinstance(value, dict), f"{field} must be an object")
    _require(set(value) == ARTIFACT_IDENTITY_FIELDS, f"{field} schema drift")
    _require(Path(value["path"]).is_absolute(), f"{field}.path must be absolute")
    _require(_is_int(value["size_bytes"]) and value["size_bytes"] > 0, f"{field}.size")
    _require(_is_sha256(value["sha256"]), f"{field}.sha256")
    _require(value["mode"] == "0444", f"{field}.mode")
    _require(value["kind"] in {"npy", "png", "msgpack"}, f"{field}.kind")
    if value["kind"] == "msgpack":
        _require(value["array"] is None, f"{field}.array must be null")
    else:
        _validate_array_summary(value["array"], field=f"{field}.array")


def validate_passed_raw_payload(payload: Any) -> None:
    """Strictly validate the successful raw result's closed semantic schema."""

    _require(isinstance(payload, dict), "raw payload must be an object")
    _require(set(payload) == TOP_LEVEL_FIELDS, "raw top-level schema drift")
    _require(payload["schema_version"] == SCHEMA_VERSION, "raw schema version drift")
    _require(payload["profile"] == PROFILE, "raw profile drift")
    _require(payload["scope"] == SCOPE, "raw scope drift")
    _require(payload["stage"] == "simulation_app_close_pending", "raw stage drift")
    _require(payload["status"] == "passed", "raw status drift")
    _require(payload["promotion_authorized"] is False, "raw promotion claim drift")
    _require(payload["host_finalization_required"] is True, "host-finalizer gate drift")
    _require(payload["failure"] is None, "successful raw contains failure")
    _require(payload["close_failures"] == [], "successful raw contains close failure")
    _require(
        payload["persistence_failures"] == [],
        "successful raw contains persistence failure",
    )

    source = payload["source"]
    _require(isinstance(source, dict) and set(source) == SOURCE_FIELDS, "source schema")
    _require(Path(source["root"]).is_absolute(), "source root")
    _require(_is_git_oid(source["commit"]), "source commit")
    _require(_is_git_oid(source["tree"]), "source tree")
    _require(source["tracked_clean"] is True, "source tracked state")
    _require(source["commit"] == source["expected_commit"], "source commit binding")
    _require(source["tree"] == source["expected_tree"], "source tree binding")

    launch = payload["launch_provenance"]
    _require(
        isinstance(launch, dict) and set(launch) == LAUNCH_PROVENANCE_FIELDS,
        "launch provenance schema",
    )
    for name in ("container_image", "saved_sbatch"):
        reference = launch[name]
        _require(
            isinstance(reference, dict) and set(reference) == EXPECTED_FILE_FIELDS,
            f"{name} schema",
        )
        _require(Path(reference["path"]).is_absolute(), f"{name} path")
        _require(_is_sha256(reference["expected_sha256"]), f"{name} sha256")
        _require(reference["exists"] is True, f"{name} existence")
        _require(
            _is_int(reference["size_bytes"]) and reference["size_bytes"] > 0,
            f"{name} size",
        )
    _require(
        launch["expected_scene_sha256"] == FOODBUSSING_SCENE_SHA256,
        "expected scene digest drift",
    )

    result = payload["result"]
    _require(isinstance(result, dict) and set(result) == RESULT_FIELDS, "result schema")
    environment = result["environment"]
    _require(
        isinstance(environment, dict) and set(environment) == ENVIRONMENT_FIELDS,
        "environment schema",
    )
    _require(environment["id"] == ENVIRONMENT, "environment id drift")
    _require(
        environment["runtime_class"]
        == "polaris.environments.manager_based_rl_splat_environment.ManagerBasedRLSplatEnv",
        "environment runtime class drift",
    )
    _require(environment["camera_sensor_keys"] == list(CAMERA_KEYS), "sensor keys")
    _require(environment["renderer_camera_keys"] == list(CAMERA_KEYS), "renderer keys")
    scene = environment["scene_file"]
    _require(
        isinstance(scene, dict) and set(scene) == RAW_IDENTITY_FIELDS,
        "scene identity schema",
    )
    _require(Path(scene["path"]).is_absolute(), "scene path drift")
    _require(scene["size_bytes"] == FOODBUSSING_SCENE_SIZE_BYTES, "scene size drift")
    _require(scene["sha256"] == FOODBUSSING_SCENE_SHA256, "scene digest drift")
    _require(scene["mode"] == "0640", "scene mode drift")
    initial_conditions = environment["initial_conditions_file"]
    _require(
        isinstance(initial_conditions, dict)
        and set(initial_conditions) == RAW_IDENTITY_FIELDS,
        "initial-conditions identity schema",
    )
    _require(Path(initial_conditions["path"]).is_absolute(), "initial-conditions path")
    _require(
        initial_conditions["size_bytes"] == FOODBUSSING_INITIAL_CONDITIONS_SIZE_BYTES,
        "initial-conditions size",
    )
    _require(
        initial_conditions["sha256"] == FOODBUSSING_INITIAL_CONDITIONS_SHA256,
        "initial-conditions digest drift",
    )
    _require(initial_conditions["mode"] == "0640", "initial-conditions mode drift")
    _require(
        environment["initial_condition_index"] == 0, "initial-condition index drift"
    )
    _require(
        environment["instruction"] == FOODBUSSING_INSTRUCTION,
        "initial-conditions instruction drift",
    )
    _require(environment["hub_revision"] == POLARIS_HUB_REVISION, "Hub revision drift")
    hub_metadata = environment["hub_metadata"]
    _require(
        isinstance(hub_metadata, dict)
        and set(hub_metadata) == {"initial_conditions", "scene"},
        "Hub metadata schema",
    )
    for name, expected_sha256 in FOODBUSSING_METADATA_SHA256.items():
        identity = hub_metadata[name]
        _require(
            isinstance(identity, dict) and set(identity) == RAW_IDENTITY_FIELDS,
            f"Hub metadata {name} identity schema",
        )
        _require(Path(identity["path"]).is_absolute(), f"Hub metadata {name} path")
        _require(identity["size_bytes"] == 101, f"Hub metadata {name} size")
        _require(identity["sha256"] == expected_sha256, f"Hub metadata {name} digest")
        _require(identity["mode"] == "0640", f"Hub metadata {name} mode")

    path_result = result["production_path"]
    _require(
        isinstance(path_result, dict) and set(path_result) == PRODUCTION_PATH_FIELDS,
        "production path schema",
    )
    _require(
        path_result["bound_render_splat_is_production_method"] is True, "bound path"
    )
    _require(
        path_result["events"]
        == [
            "ManagerBasedRLSplatEnv.render_splat.enter",
            "SplatRenderer.render",
            "ManagerBasedRLSplatEnv.render_splat.exit",
            "ManagerBasedRLSplatEnv.get_robot_from_sim",
        ],
        "production event order drift",
    )
    for count_field in (
        "renderer_render_calls",
        "render_splat_calls",
        "get_robot_from_sim_calls",
    ):
        _require(path_result[count_field] == 1, f"{count_field} drift")

    contracts = result["contracts"]
    _require(
        isinstance(contracts, dict) and set(contracts) == CONTRACT_FIELDS,
        "contract schema",
    )
    _require(
        contracts["renderer_conversion"]
        == {
            "formula": "(clip(raw_float_rgb,0,1)*255).astype(uint8)",
            "pixel_exact": True,
            "shape_preserved": True,
            "channel_order": "RGB",
            "bgr_conversion": False,
        },
        "renderer conversion contract drift",
    )
    _require(
        contracts["robot_compositing"]
        == {
            "formula": "np.where(robot_mask,sim_rgb,native_splat_rgb)",
            "pixel_exact": True,
            "native_shape": list(NATIVE_SHAPE),
        },
        "robot compositing contract drift",
    )
    preprocess = contracts["ego_lap_preprocessing"]
    _require(
        isinstance(preprocess, dict)
        and set(preprocess)
        == {
            "actual_client_class",
            "constructor_bypassed_no_network",
            "method",
            "call_events",
            "native_shape",
            "resized_content_shape",
            "preprocessed_shape",
            "padding_rows",
            "wrist_operation_order",
            "operation_order_probe",
            "pixel_exact_request_binding",
        },
        "preprocessing contract schema",
    )
    _require(
        preprocess["actual_client_class"]
        == "polaris.policy.lap_eef_pose_client.EgoLAPEefPoseClient",
        "actual client class drift",
    )
    _require(preprocess["constructor_bypassed_no_network"] is True, "network gate")
    _require(preprocess["method"] == "_build_request", "client method drift")
    _require(
        preprocess["call_events"]
        == [
            "resize:external:720x1280->224x224",
            "resize:wrist:720x1280->224x224",
            "rotate180:wrist:224x224->224x224",
        ],
        "client preprocessing call order drift",
    )
    _require(preprocess["native_shape"] == list(NATIVE_SHAPE), "native shape drift")
    _require(
        preprocess["resized_content_shape"] == list(RESIZED_CONTENT_SHAPE),
        "resized content shape drift",
    )
    _require(
        preprocess["preprocessed_shape"] == list(PREPROCESSED_SHAPE),
        "preprocessed shape drift",
    )
    _require(
        preprocess["padding_rows"] == {"top": PAD_TOP, "bottom": PAD_BOTTOM},
        "padding drift",
    )
    _require(
        preprocess["wrist_operation_order"] == "resize_pad_then_rotate_180",
        "wrist order drift",
    )
    order_probe = preprocess["operation_order_probe"]
    _require(
        isinstance(order_probe, dict)
        and set(order_probe)
        == {
            "profile",
            "input_shape",
            "target_shape",
            "resize_then_rotate_sha256",
            "rotate_then_resize_sha256",
            "differing_values",
            "production_matches_resize_then_rotate",
        },
        "operation-order probe schema",
    )
    _require(
        order_probe["profile"] == "odd_5x8_to_7x7_asymmetric_padding_v1",
        "operation-order probe profile",
    )
    _require(order_probe["input_shape"] == [5, 8, 3], "operation-order input")
    _require(order_probe["target_shape"] == [7, 7, 3], "operation-order target")
    _require(_is_sha256(order_probe["resize_then_rotate_sha256"]), "order digest")
    _require(_is_sha256(order_probe["rotate_then_resize_sha256"]), "order digest")
    _require(
        order_probe["resize_then_rotate_sha256"]
        != order_probe["rotate_then_resize_sha256"],
        "operation-order probe digests collide",
    )
    _require(order_probe["differing_values"] > 0, "operation-order probe commutes")
    _require(
        order_probe["production_matches_resize_then_rotate"] is True,
        "operation-order production mismatch",
    )
    _require(preprocess["pixel_exact_request_binding"] is True, "request binding")

    msgpack_contract = contracts["msgpack_roundtrip"]
    _require(
        isinstance(msgpack_contract, dict)
        and set(msgpack_contract)
        == {"implementation", "exact_arrays", "exact_image_bytes", "packed_sha256"},
        "msgpack contract schema",
    )
    _require(
        msgpack_contract["implementation"] == "openpi_client.msgpack_numpy",
        "msgpack implementation drift",
    )
    _require(msgpack_contract["exact_arrays"] is True, "msgpack array drift")
    _require(msgpack_contract["exact_image_bytes"] is True, "msgpack byte drift")
    _require(_is_sha256(msgpack_contract["packed_sha256"]), "msgpack digest drift")
    _require(
        contracts["removed_resize_counterfactual"]
        == {
            "profile": "removed_cv2_default_linear_half_down_up_v1",
            "live_path": False,
            "required_to_change_pixels": True,
        },
        "counterfactual contract drift",
    )

    cameras = result["cameras"]
    _require(
        isinstance(cameras, dict) and tuple(cameras) == CAMERA_KEYS, "camera schema"
    )
    for camera, camera_result in cameras.items():
        _require(
            isinstance(camera_result, dict)
            and set(camera_result) == CAMERA_RESULT_FIELDS,
            f"{camera} schema",
        )
        for array_field in (
            "raw_renderer",
            "native_uint8",
            "robot_rgb",
            "robot_mask",
            "composited_uint8",
            "preprocessed_uint8",
        ):
            _validate_array_summary(
                camera_result[array_field], field=f"{camera}.{array_field}"
            )
        _require(
            camera_result["raw_renderer"]["shape"] == list(NATIVE_SHAPE), "raw shape"
        )
        _require(
            camera_result["raw_renderer"]["dtype"].startswith("float"), "raw dtype"
        )
        for field in ("native_uint8", "robot_rgb", "composited_uint8"):
            _require(
                camera_result[field]["shape"] == list(NATIVE_SHAPE),
                f"{camera} {field} shape",
            )
            _require(
                camera_result[field]["dtype"] == "uint8", f"{camera} {field} dtype"
            )
        _require(
            camera_result["preprocessed_uint8"]["shape"] == list(PREPROCESSED_SHAPE),
            f"{camera} preprocessed shape",
        )
        _require(
            camera_result["preprocessed_uint8"]["dtype"] == "uint8",
            f"{camera} preprocessed dtype",
        )
        conversion = camera_result["conversion"]
        _require(
            isinstance(conversion, dict)
            and set(conversion)
            == {
                "pixel_exact",
                "same_shape",
                "finite_raw",
                "rgb_red_blue_differing_values",
            },
            f"{camera} conversion schema",
        )
        _require(conversion["pixel_exact"] is True, f"{camera} pixel conversion")
        _require(conversion["same_shape"] is True, f"{camera} shape preservation")
        _require(conversion["finite_raw"] is True, f"{camera} finite raw")
        _require(
            _is_int(conversion["rgb_red_blue_differing_values"])
            and conversion["rgb_red_blue_differing_values"] > 0,
            f"{camera} RGB order probe",
        )
        compositing = camera_result["compositing"]
        _require(
            isinstance(compositing, dict)
            and set(compositing)
            == {"pixel_exact", "mask_true_values", "mask_false_values"},
            f"{camera} compositing schema",
        )
        _require(compositing["pixel_exact"] is True, f"{camera} compositing exactness")
        _require(compositing["mask_true_values"] > 0, f"{camera} empty robot mask")
        _require(compositing["mask_false_values"] > 0, f"{camera} full robot mask")
        camera_preprocess = camera_result["preprocessing"]
        _require(
            isinstance(camera_preprocess, dict)
            and set(camera_preprocess)
            == {
                "request_key",
                "request_pixel_exact",
                "top_pad_zero",
                "bottom_pad_zero",
            },
            f"{camera} preprocessing schema",
        )
        _require(
            camera_preprocess["request_pixel_exact"] is True, f"{camera} request pixels"
        )
        _require(camera_preprocess["top_pad_zero"] is True, f"{camera} top pad")
        _require(camera_preprocess["bottom_pad_zero"] is True, f"{camera} bottom pad")
        counterfactual = camera_result["counterfactual"]
        _require(
            isinstance(counterfactual, dict)
            and set(counterfactual)
            == {"changed_values", "changed_pixels", "mean_abs_diff", "max_abs_diff"},
            f"{camera} counterfactual schema",
        )
        _require(counterfactual["changed_values"] > 0, f"{camera} unchanged old path")
        _require(counterfactual["changed_pixels"] > 0, f"{camera} unchanged pixels")
        _require(counterfactual["mean_abs_diff"] > 0, f"{camera} zero mean diff")
        _require(counterfactual["max_abs_diff"] > 0, f"{camera} zero max diff")

    artifacts = result["artifacts"]
    _require(isinstance(artifacts, dict), "artifacts must be an object")
    _require(set(artifacts) == expected_artifact_keys(), "artifact key schema drift")
    for name, identity in artifacts.items():
        _validate_artifact_identity(identity, field=f"artifacts.{name}")
    _require(
        artifacts["ego_lap_request_msgpack"]["sha256"]
        == msgpack_contract["packed_sha256"],
        "msgpack artifact digest binding drift",
    )


def validate_ready_marker(marker: Any, raw_path: Path, raw_bytes: bytes) -> None:
    _require(isinstance(marker, dict), "ready marker must be an object")
    _require(set(marker) == READY_FIELDS, "ready marker schema drift")
    _require(marker["schema_version"] == SCHEMA_VERSION, "ready schema version")
    _require(marker["profile"] == PROFILE, "ready profile")
    _require(marker["stage"] == "simulation_app_close_pending", "ready stage")
    _require(
        marker["raw_result"]
        == {
            "path": str(raw_path.resolve(strict=True)),
            "size_bytes": len(raw_bytes),
            "sha256": _sha256_bytes(raw_bytes),
            "mode": "0444",
        },
        "ready marker raw binding drift",
    )


def _camera_names(scene: Any, camera_type: type) -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name, sensor in scene.sensors.items()
            if isinstance(sensor, camera_type)
        )
    )


def _capture_production_render(runtime: Any, manager_class: type) -> dict[str, Any]:
    """Capture one real custom_render transaction through wrapped boundaries."""

    original_renderer_render = runtime.splat_renderer.render
    original_render_splat = runtime.render_splat
    original_get_robot = runtime.get_robot_from_sim
    events: list[str] = []
    raw_captures: list[dict[str, np.ndarray]] = []
    direct_captures: list[dict[str, np.ndarray]] = []
    robot_captures: list[dict[str, dict[str, np.ndarray]]] = []

    bound_is_production = (
        getattr(original_render_splat, "__self__", None) is runtime
        and getattr(original_render_splat, "__func__", None)
        is manager_class.render_splat
    )
    _require(bound_is_production, "render_splat is not the production bound method")

    def capture_renderer_render(extrinsics: Any) -> Any:
        events.append("SplatRenderer.render")
        rendered = original_renderer_render(extrinsics)
        _require(isinstance(rendered, dict), "SplatRenderer.render returned non-dict")
        raw_captures.append(
            {
                key: value.detach().cpu().numpy().copy()
                for key, value in rendered.items()
            }
        )
        return rendered

    def capture_render_splat() -> Any:
        events.append("ManagerBasedRLSplatEnv.render_splat.enter")
        rendered = original_render_splat()
        events.append("ManagerBasedRLSplatEnv.render_splat.exit")
        _require(isinstance(rendered, dict), "render_splat returned non-dict")
        direct_captures.append(
            {key: np.asarray(value).copy() for key, value in rendered.items()}
        )
        return rendered

    def capture_get_robot() -> Any:
        events.append("ManagerBasedRLSplatEnv.get_robot_from_sim")
        rendered = original_get_robot()
        _require(isinstance(rendered, dict), "get_robot_from_sim returned non-dict")
        robot_captures.append(
            {
                key: {
                    "rgb": np.asarray(value["rgb"]).copy(),
                    "mask": np.asarray(value["mask"]).copy(),
                }
                for key, value in rendered.items()
            }
        )
        return rendered

    runtime.splat_renderer.render = capture_renderer_render
    runtime.render_splat = capture_render_splat
    runtime.get_robot_from_sim = capture_get_robot
    try:
        composited = runtime.custom_render(True, transform_static=False)
        composited = {
            key: np.asarray(value).copy() for key, value in composited.items()
        }
    finally:
        del runtime.splat_renderer.render
        del runtime.render_splat
        del runtime.get_robot_from_sim

    _require(len(raw_captures) == 1, "expected one SplatRenderer.render call")
    _require(len(direct_captures) == 1, "expected one render_splat call")
    _require(len(robot_captures) == 1, "expected one get_robot_from_sim call")
    expected_events = [
        "ManagerBasedRLSplatEnv.render_splat.enter",
        "SplatRenderer.render",
        "ManagerBasedRLSplatEnv.render_splat.exit",
        "ManagerBasedRLSplatEnv.get_robot_from_sim",
    ]
    _require(events == expected_events, f"production call order drift: {events}")
    return {
        "raw": raw_captures[0],
        "direct": direct_captures[0],
        "robot": robot_captures[0],
        "composited": composited,
        "events": events,
        "bound_is_production": bound_is_production,
    }


def _run_client_preprocessing(
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Use the real client methods without constructing its network client."""

    from openpi_client import msgpack_numpy
    from polaris.config import LAP_EEF_FRAME, PolicyArgs
    from polaris.policy.ego_lap_contract import R6_COLUMNS_STATE_LAYOUT
    import polaris.policy.lap_eef_pose_client as lap_module

    client = object.__new__(lap_module.EgoLAPEefPoseClient)
    client.args = PolicyArgs(client="EgoLAPEefPose", eef_frame=LAP_EEF_FRAME)
    client.frame_description = "image contract smoke; no checkpoint or policy server"
    client.dataset_name = "droid"
    client.state_type = "eef_pose"
    client.state_layout = R6_COLUMNS_STATE_LAYOUT
    client.rotate_wrist_180 = True
    client._image_io_logged = True

    current = client._extract_observation(dict(observation))
    in_h, in_w = current["external_image"].shape[:2]
    ratio = np.maximum(
        np.float32(in_w) / np.float32(PREPROCESSED_SHAPE[1]),
        np.float32(in_h) / np.float32(PREPROCESSED_SHAPE[0]),
    )
    resized_h = int(np.floor(np.float32(in_h) / ratio))
    resized_w = int(np.floor(np.float32(in_w) / ratio))
    _require(
        (resized_h, resized_w, 3) == RESIZED_CONTENT_SHAPE,
        "Ego-LAP aspect resize geometry drift",
    )
    _require(
        (PREPROCESSED_SHAPE[0] - resized_h) // 2 == PAD_TOP
        and PREPROCESSED_SHAPE[0] - resized_h - PAD_TOP == PAD_BOTTOM,
        "Ego-LAP symmetric padding geometry drift",
    )
    call_events: list[str] = []
    original_resize = lap_module.resize_lap_image
    original_rotate = lap_module.rotate_image_180
    resize_calls = 0

    def tracked_resize(image: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        nonlocal resize_calls
        resize_calls += 1
        label = "external" if resize_calls == 1 else "wrist"
        value = np.asarray(image)
        result = original_resize(value, *args, **kwargs)
        call_events.append(
            f"resize:{label}:{value.shape[0]}x{value.shape[1]}"
            f"->{result.shape[0]}x{result.shape[1]}"
        )
        return result

    def tracked_rotate(image: np.ndarray) -> np.ndarray:
        value = np.asarray(image)
        result = original_rotate(value)
        call_events.append(
            f"rotate180:wrist:{value.shape[0]}x{value.shape[1]}"
            f"->{result.shape[0]}x{result.shape[1]}"
        )
        return result

    lap_module.resize_lap_image = tracked_resize
    lap_module.rotate_image_180 = tracked_rotate
    try:
        request, external, wrist = client._build_request(
            current, "Put all the foods in the bowl"
        )
    finally:
        lap_module.resize_lap_image = original_resize
        lap_module.rotate_image_180 = original_rotate

    _require(
        call_events
        == [
            "resize:external:720x1280->224x224",
            "resize:wrist:720x1280->224x224",
            "rotate180:wrist:224x224->224x224",
        ],
        f"Ego-LAP preprocessing call order drift: {call_events}",
    )
    expected_external = original_resize(current["external_image"])
    unrotated_wrist = original_resize(current["wrist_image"])
    expected_wrist = original_rotate(unrotated_wrist)
    np.testing.assert_array_equal(external, expected_external)
    np.testing.assert_array_equal(wrist, expected_wrist)
    np.testing.assert_array_equal(request["observation"]["base_0_rgb"], external)
    np.testing.assert_array_equal(request["observation"]["left_wrist_0_rgb"], wrist)

    for name, image in (("external", external), ("wrist", wrist)):
        _require(image.shape == PREPROCESSED_SHAPE, f"{name} preprocessing shape")
        _require(image.dtype == np.uint8, f"{name} preprocessing dtype")
        _require(not image[:PAD_TOP].any(), f"{name} top padding is nonzero")
        _require(not image[-PAD_BOTTOM:].any(), f"{name} bottom padding is nonzero")

    sentinel = (
        np.arange(5 * 8 * 3, dtype=np.uint16).reshape(5, 8, 3) % 251 + 1
    ).astype(np.uint8)
    production_order_probe = lap_module.preprocess_lap_wrist_image(
        sentinel,
        rotate_180=True,
        target_h=7,
        target_w=7,
    )
    resize_then_rotate = original_rotate(original_resize(sentinel, 7, 7))
    rotate_then_resize = original_resize(original_rotate(sentinel), 7, 7)
    np.testing.assert_array_equal(production_order_probe, resize_then_rotate)
    differing_order_values = int(
        np.count_nonzero(resize_then_rotate != rotate_then_resize)
    )
    _require(
        differing_order_values > 0,
        "odd asymmetric wrist operation-order probe unexpectedly commutes",
    )

    packed = msgpack_numpy.packb(request)
    unpacked = msgpack_numpy.unpackb(packed)
    for key, expected in (
        ("base_0_rgb", external),
        ("left_wrist_0_rgb", wrist),
    ):
        actual = unpacked["observation"][key]
        _require(actual.shape == expected.shape, f"msgpack {key} shape drift")
        _require(actual.dtype == expected.dtype, f"msgpack {key} dtype drift")
        np.testing.assert_array_equal(actual, expected)
        _require(actual.tobytes() == expected.tobytes(), f"msgpack {key} byte drift")

    return {
        "external": external.copy(),
        "wrist": wrist.copy(),
        "call_events": call_events,
        "request": request,
        "packed": packed,
        "packed_sha256": _sha256_bytes(packed),
        "operation_order_probe": {
            "profile": "odd_5x8_to_7x7_asymmetric_padding_v1",
            "input_shape": list(sentinel.shape),
            "target_shape": list(resize_then_rotate.shape),
            "resize_then_rotate_sha256": _sha256_bytes(resize_then_rotate.tobytes()),
            "rotate_then_resize_sha256": _sha256_bytes(rotate_then_resize.tobytes()),
            "differing_values": differing_order_values,
            "production_matches_resize_then_rotate": True,
        },
    }


def _scene_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    identity = raw_file_identity(resolved)
    _require(identity["sha256"] == FOODBUSSING_SCENE_SHA256, "FoodBussing scene drift")
    _require(identity["mode"] == "0640", "FoodBussing scene mode drift")
    return identity


def _hub_metadata_identity(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    payload = resolved.read_bytes()
    try:
        revision = payload.decode("utf-8").splitlines()[0]
    except (UnicodeDecodeError, IndexError) as error:
        raise ContractError(f"invalid Hub metadata file: {resolved}") from error
    _require(
        revision == POLARIS_HUB_REVISION, f"Hub metadata revision drift: {resolved}"
    )
    identity = raw_file_identity(resolved)
    _require(
        identity["sha256"] == expected_sha256, f"Hub metadata digest drift: {resolved}"
    )
    _require(identity["mode"] == "0640", f"Hub metadata mode drift: {resolved}")
    return identity


def run_smoke(args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    import gymnasium as gym
    from isaaclab.sensors.camera.camera import Camera
    from isaaclab_tasks.utils import parse_env_cfg
    import polaris.environments  # noqa: F401
    from polaris.environments.manager_based_rl_splat_environment import (
        ManagerBasedRLSplatEnv,
    )
    from polaris.utils import load_eval_initial_conditions

    state["stage"] = "capture_source_identity"
    source = capture_source_identity(
        args.source_root,
        expected_commit=args.expected_source_commit,
        expected_tree=args.expected_source_tree,
    )
    launch_provenance = {
        "container_image": expected_file_reference(
            args.container_image, args.container_image_sha256
        ),
        "saved_sbatch": expected_file_reference(
            args.saved_sbatch, args.saved_sbatch_sha256
        ),
        "expected_scene_sha256": args.expected_scene_sha256,
    }
    state["source"] = source
    state["launch_provenance"] = launch_provenance
    _require(
        args.expected_scene_sha256 == FOODBUSSING_SCENE_SHA256,
        "expected scene SHA-256 is not the pinned FoodBussing scene",
    )

    state["stage"] = "create_environment"
    env_cfg = parse_env_cfg(
        ENVIRONMENT,
        device="cuda",
        num_envs=1,
        use_fabric=True,
    )
    env = gym.make(ENVIRONMENT, cfg=env_cfg)
    state["env"] = env
    runtime = env.unwrapped
    _require(type(runtime) is ManagerBasedRLSplatEnv, "unexpected runtime class")

    state["stage"] = "reset_environment"
    initial_conditions_path = Path(runtime.usd_file).parent / "initial_conditions.json"
    initial_conditions_identity = raw_file_identity(initial_conditions_path)
    _require(
        initial_conditions_identity["sha256"] == FOODBUSSING_INITIAL_CONDITIONS_SHA256,
        "FoodBussing initial-conditions digest drift",
    )
    _require(
        initial_conditions_identity["mode"] == "0640",
        "FoodBussing initial-conditions mode drift",
    )
    instruction, initial_conditions = load_eval_initial_conditions(
        usd=runtime.usd_file,
        initial_conditions_file=str(initial_conditions_path),
        rollouts=1,
    )
    _require(
        isinstance(initial_conditions, list)
        and len(initial_conditions) == 1
        and isinstance(initial_conditions[0], dict),
        "FoodBussing initial-condition index 0 is unavailable",
    )
    _require(
        instruction == FOODBUSSING_INSTRUCTION,
        "FoodBussing instruction drift",
    )
    hub_metadata_root = (
        Path(runtime.usd_file).parent.parent
        / ".cache/huggingface/download/food_bussing"
    )
    hub_metadata = {
        "initial_conditions": _hub_metadata_identity(
            hub_metadata_root / "initial_conditions.json.metadata",
            expected_sha256=FOODBUSSING_METADATA_SHA256["initial_conditions"],
        ),
        "scene": _hub_metadata_identity(
            hub_metadata_root / "scene.usda.metadata",
            expected_sha256=FOODBUSSING_METADATA_SHA256["scene"],
        ),
    }
    reset_observation, _ = env.reset(
        object_positions=initial_conditions[0],
        expensive=False,
    )
    runtime.sim.render()
    runtime.scene.update(0)
    # Establish static-camera extrinsics before installing the single-call
    # capture wrappers.  The captured transaction then contains exactly one
    # SplatRenderer.render invocation owned by render_splat.
    runtime.transform_sim_to_splat(transform_static=True)

    sensor_keys = _camera_names(runtime.scene, Camera)
    renderer_keys = tuple(sorted(runtime.splat_renderer.cameras))
    _require(sensor_keys == CAMERA_KEYS, f"camera sensor key drift: {sensor_keys}")
    _require(
        renderer_keys == CAMERA_KEYS, f"renderer camera key drift: {renderer_keys}"
    )

    state["stage"] = "capture_production_custom_render"
    capture = _capture_production_render(runtime, ManagerBasedRLSplatEnv)
    for boundary in ("raw", "direct", "robot", "composited"):
        keys = tuple(sorted(capture[boundary]))
        _require(keys == CAMERA_KEYS, f"{boundary} camera key drift: {keys}")

    state["stage"] = "validate_production_pixels"
    per_camera: dict[str, dict[str, Any]] = {}
    exact_arrays: dict[str, dict[str, np.ndarray]] = {}
    client_observation = {
        "splat": capture["composited"],
        "policy": reset_observation["policy"],
    }
    client_result = _run_client_preprocessing(client_observation)
    writer = ArtifactWriter(args.output_dir)

    for camera in CAMERA_KEYS:
        raw = capture["raw"][camera]
        direct = capture["direct"][camera]
        robot_rgb = capture["robot"][camera]["rgb"]
        robot_mask = capture["robot"][camera]["mask"]
        composited = capture["composited"][camera]
        preprocessed = (
            client_result["external"]
            if camera == "external_cam"
            else client_result["wrist"]
        )

        expected_direct = independent_renderer_conversion(raw)
        _require(direct.shape == NATIVE_SHAPE, f"{camera} direct shape drift")
        _require(direct.dtype == np.uint8, f"{camera} direct dtype drift")
        np.testing.assert_array_equal(direct, expected_direct)
        red_blue_differences = int(np.count_nonzero(direct[..., 0] != direct[..., 2]))
        _require(red_blue_differences > 0, f"{camera} RGB order probe is degenerate")

        _require(robot_rgb.shape == NATIVE_SHAPE, f"{camera} robot RGB shape drift")
        _require(robot_rgb.dtype == np.uint8, f"{camera} robot RGB dtype drift")
        _require(
            robot_mask.shape in {(720, 1280, 1), NATIVE_SHAPE},
            f"{camera} robot mask shape drift: {robot_mask.shape}",
        )
        _require(np.isin(robot_mask, [0, 1]).all(), f"{camera} robot mask values drift")
        boolean_mask = robot_mask.astype(bool, copy=False)
        mask_true = int(np.count_nonzero(boolean_mask))
        mask_false = int(boolean_mask.size - mask_true)
        _require(mask_true > 0 and mask_false > 0, f"{camera} mask is degenerate")
        expected_composited = np.where(boolean_mask, robot_rgb, direct)
        _require(expected_composited.shape == NATIVE_SHAPE, "composite shape drift")
        _require(expected_composited.dtype == np.uint8, "composite dtype drift")
        np.testing.assert_array_equal(composited, expected_composited)

        old = old_half_down_up_counterfactual(direct)
        _require(old.shape == NATIVE_SHAPE, "counterfactual shape drift")
        diff = np.abs(direct.astype(np.int16) - old.astype(np.int16)).astype(np.uint8)
        changed_values = int(np.count_nonzero(diff))
        changed_pixels = int(np.count_nonzero(np.any(diff != 0, axis=2)))
        mean_abs_diff = float(diff.astype(np.float64).mean())
        max_abs_diff = int(diff.max())
        _require(
            changed_values > 0 and changed_pixels > 0, "old resize changed no pixels"
        )
        _require(mean_abs_diff > 0 and max_abs_diff > 0, "old resize diff is zero")
        diff_preview = np.minimum(diff.astype(np.uint16) * 4, 255).astype(np.uint8)

        request_key = "base_0_rgb" if camera == "external_cam" else "left_wrist_0_rgb"
        np.testing.assert_array_equal(
            client_result["request"]["observation"][request_key], preprocessed
        )

        exact_arrays[camera] = {
            "raw": raw,
            "native": direct,
            "robot_rgb": robot_rgb,
            "robot_mask": robot_mask,
            "composited": composited,
            "preprocessed": preprocessed,
            "old": old,
            "diff": diff,
            "diff_preview": diff_preview,
        }
        per_camera[camera] = {
            "raw_renderer": array_summary(raw),
            "native_uint8": array_summary(direct),
            "robot_rgb": array_summary(robot_rgb),
            "robot_mask": array_summary(robot_mask),
            "composited_uint8": array_summary(composited),
            "preprocessed_uint8": array_summary(preprocessed),
            "conversion": {
                "pixel_exact": True,
                "same_shape": raw.shape == direct.shape,
                "finite_raw": bool(np.isfinite(raw).all()),
                "rgb_red_blue_differing_values": red_blue_differences,
            },
            "compositing": {
                "pixel_exact": True,
                "mask_true_values": mask_true,
                "mask_false_values": mask_false,
            },
            "preprocessing": {
                "request_key": request_key,
                "request_pixel_exact": True,
                "top_pad_zero": bool(not preprocessed[:PAD_TOP].any()),
                "bottom_pad_zero": bool(not preprocessed[-PAD_BOTTOM:].any()),
            },
            "counterfactual": {
                "changed_values": changed_values,
                "changed_pixels": changed_pixels,
                "mean_abs_diff": mean_abs_diff,
                "max_abs_diff": max_abs_diff,
            },
        }

    state["stage"] = "publish_lossless_artifacts"
    for camera in CAMERA_KEYS:
        arrays = exact_arrays[camera]
        writer.array(f"{camera}_renderer_raw_float", arrays["raw"])
        writer.array(f"{camera}_native_uint8", arrays["native"])
        writer.array(f"{camera}_robot_rgb", arrays["robot_rgb"])
        writer.array(f"{camera}_robot_mask", arrays["robot_mask"])
        writer.array(f"{camera}_composited_uint8", arrays["composited"])
        writer.array(f"{camera}_preprocessed_uint8", arrays["preprocessed"])
        writer.array(f"{camera}_old_half_down_up_uint8", arrays["old"])
        writer.array(f"{camera}_old_abs_diff_uint8", arrays["diff"])
        writer.png(f"{camera}_native_png", arrays["native"])
        writer.png(f"{camera}_composited_png", arrays["composited"])
        writer.png(f"{camera}_preprocessed_png", arrays["preprocessed"])
        writer.png(f"{camera}_old_half_down_up_png", arrays["old"])
        writer.png(f"{camera}_old_abs_diff_x4_png", arrays["diff_preview"])
    contact_sheet = make_contact_sheet(exact_arrays)
    writer.png("contact_sheet", contact_sheet)
    writer.bytes(
        "ego_lap_request_msgpack",
        ".msgpack",
        client_result["packed"],
        kind="msgpack",
    )
    _require(set(writer.identities) == expected_artifact_keys(), "artifact set drift")

    result = {
        "environment": {
            "id": ENVIRONMENT,
            "runtime_class": (
                "polaris.environments.manager_based_rl_splat_environment."
                "ManagerBasedRLSplatEnv"
            ),
            "scene_file": _scene_identity(Path(runtime.usd_file)),
            "initial_conditions_file": initial_conditions_identity,
            "initial_condition_index": 0,
            "instruction": instruction,
            "hub_revision": POLARIS_HUB_REVISION,
            "hub_metadata": hub_metadata,
            "camera_sensor_keys": list(sensor_keys),
            "renderer_camera_keys": list(renderer_keys),
        },
        "production_path": {
            "bound_render_splat_is_production_method": capture["bound_is_production"],
            "events": capture["events"],
            "renderer_render_calls": 1,
            "render_splat_calls": 1,
            "get_robot_from_sim_calls": 1,
        },
        "contracts": {
            "renderer_conversion": {
                "formula": "(clip(raw_float_rgb,0,1)*255).astype(uint8)",
                "pixel_exact": True,
                "shape_preserved": True,
                "channel_order": "RGB",
                "bgr_conversion": False,
            },
            "robot_compositing": {
                "formula": "np.where(robot_mask,sim_rgb,native_splat_rgb)",
                "pixel_exact": True,
                "native_shape": list(NATIVE_SHAPE),
            },
            "ego_lap_preprocessing": {
                "actual_client_class": (
                    "polaris.policy.lap_eef_pose_client.EgoLAPEefPoseClient"
                ),
                "constructor_bypassed_no_network": True,
                "method": "_build_request",
                "call_events": client_result["call_events"],
                "native_shape": list(NATIVE_SHAPE),
                "resized_content_shape": list(RESIZED_CONTENT_SHAPE),
                "preprocessed_shape": list(PREPROCESSED_SHAPE),
                "padding_rows": {"top": PAD_TOP, "bottom": PAD_BOTTOM},
                "wrist_operation_order": "resize_pad_then_rotate_180",
                "operation_order_probe": client_result["operation_order_probe"],
                "pixel_exact_request_binding": True,
            },
            "msgpack_roundtrip": {
                "implementation": "openpi_client.msgpack_numpy",
                "exact_arrays": True,
                "exact_image_bytes": True,
                "packed_sha256": client_result["packed_sha256"],
            },
            "removed_resize_counterfactual": {
                "profile": "removed_cv2_default_linear_half_down_up_v1",
                "live_path": False,
                "required_to_change_pixels": True,
            },
        },
        "cameras": per_camera,
        "artifacts": writer.identities,
    }
    return result


def build_parser(
    add_app_launcher_args: Callable[[argparse.ArgumentParser], None] | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree", required=True)
    parser.add_argument("--container-image", type=Path, required=True)
    parser.add_argument("--container-image-sha256", required=True)
    parser.add_argument("--saved-sbatch", type=Path, required=True)
    parser.add_argument("--saved-sbatch-sha256", required=True)
    parser.add_argument("--expected-scene-sha256", default=FOODBUSSING_SCENE_SHA256)
    if add_app_launcher_args is not None:
        add_app_launcher_args(parser)
    return parser


def _result_payload(state: Mapping[str, Any]) -> dict[str, Any]:
    passed = (
        state["stage"] == "simulation_app_close_pending"
        and state["result"] is not None
        and state["failure"] is None
        and not state["close_failures"]
        and not state["persistence_failures"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "profile": PROFILE,
        "scope": SCOPE,
        "stage": state["stage"],
        "status": "passed" if passed else "failed",
        "promotion_authorized": False,
        "host_finalization_required": True,
        "source": state["source"],
        "launch_provenance": state["launch_provenance"],
        "result": state["result"],
        "failure": state["failure"],
        "close_failures": state["close_failures"],
        "persistence_failures": state["persistence_failures"],
    }


def _load_runtime() -> tuple[argparse.Namespace, Any]:
    from isaaclab.app import AppLauncher

    parser = build_parser(AppLauncher.add_app_launcher_args)
    args, _ = parser.parse_known_args()
    args.enable_cameras = True
    args.headless = True
    return args, AppLauncher


def main() -> None:
    args, app_launcher_type = _load_runtime()
    args.output_dir = args.output_dir.resolve(strict=False)
    args.output_json = args.output_json.resolve(strict=False)
    _require(
        args.output_json.parent == args.output_dir, "output JSON must be in output dir"
    )
    _require(not args.output_json.exists(), "output JSON already exists")
    _require(
        not args.output_json.with_name(args.output_json.name + ".ready.json").exists(),
        "ready marker already exists",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _require(not any(args.output_dir.iterdir()), "output directory must start empty")
    for digest_name in ("expected_source_commit", "expected_source_tree"):
        _require(_is_git_oid(getattr(args, digest_name)), f"{digest_name} malformed")
    for digest_name in (
        "container_image_sha256",
        "saved_sbatch_sha256",
        "expected_scene_sha256",
    ):
        _require(_is_sha256(getattr(args, digest_name)), f"{digest_name} malformed")

    state: dict[str, Any] = {
        "stage": "launch_simulation_app",
        "source": None,
        "launch_provenance": None,
        "result": None,
        "failure": None,
        "close_failures": [],
        "persistence_failures": [],
        "env": None,
    }
    exit_code = 1
    simulation_app = None
    try:
        app_launcher = app_launcher_type(args)
        simulation_app = app_launcher.app
        state["stage"] = "run_smoke"
        state["result"] = run_smoke(args, state)
        exit_code = 0
    except BaseException as error:
        state["failure"] = _exception_evidence(error)
        traceback.print_exception(
            type(error), error, error.__traceback__, file=sys.stderr
        )
        exit_code = 1

    env = state["env"]
    if state["failure"] is None:
        state["stage"] = "close_environment"
    if env is not None:
        try:
            env.close()
        except BaseException as error:
            evidence = _exception_evidence(error)
            evidence["component"] = "environment"
            state["close_failures"].append(evidence)
            traceback.print_exception(
                type(error), error, error.__traceback__, file=sys.stderr
            )
            exit_code = 1

    if (
        simulation_app is not None
        and exit_code == 0
        and state["failure"] is None
        and not state["close_failures"]
    ):
        state["stage"] = "simulation_app_close_pending"

    raw_published = False
    raw_payload = _result_payload(state)
    if raw_payload["status"] == "passed":
        validate_passed_raw_payload(raw_payload)
    try:
        publish_immutable_json(args.output_json, raw_payload)
        raw_published = True
    except BaseException as error:
        state["persistence_failures"].append(
            {**_exception_evidence(error), "phase": "publish_immutable_raw"}
        )
        traceback.print_exception(
            type(error), error, error.__traceback__, file=sys.stderr
        )
        exit_code = 1

    raw_ready = False
    if (
        raw_published
        and raw_payload["status"] == "passed"
        and simulation_app is not None
        and _mode(args.output_json) == "0444"
    ):
        try:
            raw_bytes = args.output_json.read_bytes()
            ready_path = args.output_json.with_name(
                args.output_json.name + ".ready.json"
            )
            ready_payload = {
                "schema_version": SCHEMA_VERSION,
                "profile": PROFILE,
                "stage": "simulation_app_close_pending",
                "raw_result": {
                    "path": str(args.output_json.resolve(strict=True)),
                    "size_bytes": len(raw_bytes),
                    "sha256": _sha256_bytes(raw_bytes),
                    "mode": "0444",
                },
            }
            validate_ready_marker(ready_payload, args.output_json, raw_bytes)
            print(f"POLARIS_IMAGE_SMOKE_RAW_PREPARED={args.output_json}", flush=True)
            print(
                f"POLARIS_IMAGE_SMOKE_RAW_SHA256={_sha256_bytes(raw_bytes)}",
                flush=True,
            )
            print(f"POLARIS_IMAGE_SMOKE_READY_MARKER={ready_path}", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            publish_immutable_json(ready_path, ready_payload)
            raw_ready = True
            simulation_app.close()
        except BaseException as error:
            raw_ready = False
            traceback.print_exception(
                type(error), error, error.__traceback__, file=sys.stderr
            )
            exit_code = 1
    else:
        exit_code = 1

    # The pinned SimulationApp.close() normally hard-exits.  A successful
    # return is still treated as failure; host promotion requires zero srun and
    # Slurm accounting after the immutable raw/ready pair exists.
    if simulation_app is not None and not raw_ready:
        print("POLARIS_IMAGE_SMOKE_CLOSE_SKIPPED=raw_not_ready", flush=True)
    if exit_code != 0 and simulation_app is not None:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(1)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
