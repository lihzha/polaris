from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


REPO_ROOT = Path(__file__).parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/smoke_splat_image_contract.py"
ENVIRONMENT_PATH = (
    REPO_ROOT / "src/polaris/environments/manager_based_rl_splat_environment.py"
)
CLIENT_PATH = REPO_ROOT / "src/polaris/policy/lap_eef_pose_client.py"


def _load_smoke() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "smoke_splat_image_contract", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = _load_smoke()


class _TensorLike:
    def __init__(self, value: np.ndarray):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def _native_raw(offset: float) -> np.ndarray:
    rows = np.linspace(-0.1, 1.1, 720, dtype=np.float32)[:, None]
    columns = np.linspace(0.0, 1.0, 1280, dtype=np.float32)[None, :]
    value = np.empty(smoke.NATIVE_SHAPE, dtype=np.float32)
    value[..., 0] = rows + np.float32(offset)
    value[..., 1] = columns
    value[..., 2] = np.float32(0.75) - rows * np.float32(0.25)
    return value


class _FakeRenderer:
    def __init__(self):
        self.raw = {
            "external_cam": _native_raw(0.0),
            "wrist_cam": _native_raw(0.05),
        }

    def render(self, _extrinsics):
        return {key: _TensorLike(value) for key, value in self.raw.items()}


class _FakeManager:
    def __init__(self):
        self.splat_renderer = _FakeRenderer()
        self.robot = {}
        for index, camera in enumerate(smoke.CAMERA_KEYS):
            rgb = np.full(smoke.NATIVE_SHAPE, 20 + index, dtype=np.uint8)
            mask = np.zeros((720, 1280, 1), dtype=np.uint8)
            mask[20:60, 30:90] = 1
            self.robot[camera] = {"rgb": rgb, "mask": mask}

    def render_splat(self):
        raw = self.splat_renderer.render({})
        return {
            key: (np.clip(value.numpy(), 0.0, 1.0) * 255).astype(np.uint8)
            for key, value in raw.items()
        }

    def get_robot_from_sim(self):
        return self.robot

    def custom_render(self, expensive: bool, transform_static: bool = False):
        assert expensive is True
        assert transform_static is False
        rendered = self.render_splat()
        robot = self.get_robot_from_sim()
        for camera in robot:
            rendered[camera] = np.where(
                robot[camera]["mask"], robot[camera]["rgb"], rendered[camera]
            )
        return rendered


def _summary(
    shape: list[int], dtype: str, minimum: int | float, maximum: int | float
) -> dict[str, object]:
    return {
        "shape": shape,
        "dtype": dtype,
        "min": minimum,
        "max": maximum,
    }


def _artifact_identity(tmp_path: Path, name: str) -> dict[str, object]:
    is_msgpack = name == "ego_lap_request_msgpack"
    if is_msgpack:
        kind = "msgpack"
        array = None
    else:
        kind = "png" if name.endswith("png") or name == "contact_sheet" else "npy"
        array = _summary([1, 1, 3], "uint8", 0, 255)
    suffix = ".msgpack" if is_msgpack else f".{kind}"
    return {
        "path": str((tmp_path / f"{name}{suffix}").resolve()),
        "size_bytes": 1,
        "sha256": "c" * 64,
        "mode": "0444",
        "kind": kind,
        "array": array,
    }


def _passing_payload(tmp_path: Path) -> dict[str, object]:
    cameras = {}
    for camera in smoke.CAMERA_KEYS:
        cameras[camera] = {
            "raw_renderer": _summary(list(smoke.NATIVE_SHAPE), "float32", 0.0, 1.0),
            "native_uint8": _summary(list(smoke.NATIVE_SHAPE), "uint8", 0, 255),
            "robot_rgb": _summary(list(smoke.NATIVE_SHAPE), "uint8", 0, 255),
            "robot_mask": _summary([720, 1280, 1], "uint8", 0, 1),
            "composited_uint8": _summary(list(smoke.NATIVE_SHAPE), "uint8", 0, 255),
            "preprocessed_uint8": _summary(
                list(smoke.PREPROCESSED_SHAPE), "uint8", 0, 255
            ),
            "conversion": {
                "pixel_exact": True,
                "same_shape": True,
                "finite_raw": True,
                "rgb_red_blue_differing_values": 100,
            },
            "compositing": {
                "pixel_exact": True,
                "mask_true_values": 100,
                "mask_false_values": 100,
            },
            "preprocessing": {
                "request_key": (
                    "base_0_rgb" if camera == "external_cam" else "left_wrist_0_rgb"
                ),
                "request_pixel_exact": True,
                "top_pad_zero": True,
                "bottom_pad_zero": True,
            },
            "counterfactual": {
                "changed_values": 100,
                "changed_pixels": 50,
                "mean_abs_diff": 0.5,
                "max_abs_diff": 12,
            },
        }
    artifacts = {
        name: _artifact_identity(tmp_path, name)
        for name in smoke.expected_artifact_keys()
    }
    return {
        "schema_version": smoke.SCHEMA_VERSION,
        "profile": smoke.PROFILE,
        "scope": smoke.SCOPE,
        "stage": "simulation_app_close_pending",
        "status": "passed",
        "promotion_authorized": False,
        "host_finalization_required": True,
        "source": {
            "root": str(tmp_path.resolve()),
            "commit": "a" * 40,
            "tree": "b" * 40,
            "tracked_clean": True,
            "expected_commit": "a" * 40,
            "expected_tree": "b" * 40,
        },
        "launch_provenance": {
            "container_image": {
                "path": str((tmp_path / "image.sqsh").resolve()),
                "expected_sha256": "d" * 64,
                "exists": True,
                "size_bytes": 1,
            },
            "saved_sbatch": {
                "path": str((tmp_path / "launch.sbatch").resolve()),
                "expected_sha256": "e" * 64,
                "exists": True,
                "size_bytes": 1,
            },
            "expected_scene_sha256": smoke.FOODBUSSING_SCENE_SHA256,
        },
        "result": {
            "environment": {
                "id": smoke.ENVIRONMENT,
                "runtime_class": (
                    "polaris.environments.manager_based_rl_splat_environment."
                    "ManagerBasedRLSplatEnv"
                ),
                "scene_file": {
                    "path": str((tmp_path / "scene.usda").resolve()),
                    "size_bytes": smoke.FOODBUSSING_SCENE_SIZE_BYTES,
                    "sha256": smoke.FOODBUSSING_SCENE_SHA256,
                    "mode": "0640",
                },
                "initial_conditions_file": {
                    "path": str((tmp_path / "initial_conditions.json").resolve()),
                    "size_bytes": smoke.FOODBUSSING_INITIAL_CONDITIONS_SIZE_BYTES,
                    "sha256": smoke.FOODBUSSING_INITIAL_CONDITIONS_SHA256,
                    "mode": "0640",
                },
                "initial_condition_index": 0,
                "instruction": "Put all the foods in the bowl",
                "hub_revision": smoke.POLARIS_HUB_REVISION,
                "hub_metadata": {
                    "initial_conditions": {
                        "path": str(
                            (tmp_path / "initial_conditions.json.metadata").resolve()
                        ),
                        "size_bytes": 101,
                        "sha256": smoke.FOODBUSSING_METADATA_SHA256[
                            "initial_conditions"
                        ],
                        "mode": "0640",
                    },
                    "scene": {
                        "path": str((tmp_path / "scene.usda.metadata").resolve()),
                        "size_bytes": 101,
                        "sha256": smoke.FOODBUSSING_METADATA_SHA256["scene"],
                        "mode": "0640",
                    },
                },
                "camera_sensor_keys": list(smoke.CAMERA_KEYS),
                "renderer_camera_keys": list(smoke.CAMERA_KEYS),
            },
            "production_path": {
                "bound_render_splat_is_production_method": True,
                "events": [
                    "ManagerBasedRLSplatEnv.render_splat.enter",
                    "SplatRenderer.render",
                    "ManagerBasedRLSplatEnv.render_splat.exit",
                    "ManagerBasedRLSplatEnv.get_robot_from_sim",
                ],
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
                    "native_shape": list(smoke.NATIVE_SHAPE),
                },
                "ego_lap_preprocessing": {
                    "actual_client_class": (
                        "polaris.policy.lap_eef_pose_client.EgoLAPEefPoseClient"
                    ),
                    "constructor_bypassed_no_network": True,
                    "method": "_build_request",
                    "call_events": [
                        "resize:external:720x1280->224x224",
                        "resize:wrist:720x1280->224x224",
                        "rotate180:wrist:224x224->224x224",
                    ],
                    "native_shape": list(smoke.NATIVE_SHAPE),
                    "resized_content_shape": list(smoke.RESIZED_CONTENT_SHAPE),
                    "preprocessed_shape": list(smoke.PREPROCESSED_SHAPE),
                    "padding_rows": {"top": 49, "bottom": 49},
                    "wrist_operation_order": "resize_pad_then_rotate_180",
                    "operation_order_probe": {
                        "profile": "odd_5x8_to_7x7_asymmetric_padding_v1",
                        "input_shape": [5, 8, 3],
                        "target_shape": [7, 7, 3],
                        "resize_then_rotate_sha256": "1" * 64,
                        "rotate_then_resize_sha256": "2" * 64,
                        "differing_values": 10,
                        "production_matches_resize_then_rotate": True,
                    },
                    "pixel_exact_request_binding": True,
                },
                "msgpack_roundtrip": {
                    "implementation": "openpi_client.msgpack_numpy",
                    "exact_arrays": True,
                    "exact_image_bytes": True,
                    "packed_sha256": "c" * 64,
                },
                "removed_resize_counterfactual": {
                    "profile": "removed_cv2_default_linear_half_down_up_v1",
                    "live_path": False,
                    "required_to_change_pixels": True,
                },
            },
            "cameras": cameras,
            "artifacts": artifacts,
        },
        "failure": None,
        "close_failures": [],
        "persistence_failures": [],
    }


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def test_capture_wraps_one_real_production_transaction() -> None:
    runtime = _FakeManager()

    capture = smoke._capture_production_render(runtime, _FakeManager)

    assert capture["bound_is_production"] is True
    assert capture["events"] == [
        "ManagerBasedRLSplatEnv.render_splat.enter",
        "SplatRenderer.render",
        "ManagerBasedRLSplatEnv.render_splat.exit",
        "ManagerBasedRLSplatEnv.get_robot_from_sim",
    ]
    assert tuple(sorted(capture["raw"])) == smoke.CAMERA_KEYS
    for camera in smoke.CAMERA_KEYS:
        expected_direct = smoke.independent_renderer_conversion(capture["raw"][camera])
        np.testing.assert_array_equal(capture["direct"][camera], expected_direct)
        expected_composited = np.where(
            runtime.robot[camera]["mask"],
            runtime.robot[camera]["rgb"],
            expected_direct,
        )
        np.testing.assert_array_equal(
            capture["composited"][camera], expected_composited
        )
    assert "render" not in runtime.splat_renderer.__dict__
    assert "render_splat" not in runtime.__dict__
    assert "get_robot_from_sim" not in runtime.__dict__


def test_independent_conversion_is_exact_rgb_clip_truncate_without_resampling() -> None:
    raw = _native_raw(0.0)

    converted = smoke.independent_renderer_conversion(raw)
    expected = (np.clip(raw, 0.0, 1.0) * 255).astype(np.uint8)

    assert converted.shape == raw.shape == smoke.NATIVE_SHAPE
    assert converted.dtype == np.uint8
    np.testing.assert_array_equal(converted, expected)
    assert np.count_nonzero(converted[..., 0] != converted[..., 2]) > 0


def test_removed_resize_counterfactual_changes_pixels_but_keeps_native_shape() -> None:
    native = smoke.independent_renderer_conversion(_native_raw(0.0))

    old = smoke.old_half_down_up_counterfactual(native)
    difference = np.abs(native.astype(np.int16) - old.astype(np.int16))

    assert old.shape == smoke.NATIVE_SHAPE
    assert old.dtype == np.uint8
    assert np.count_nonzero(difference) > 0
    assert difference.max() > 0


def test_artifact_writer_is_nonoverwriting_immutable_and_lossless(
    tmp_path: Path,
) -> None:
    writer = smoke.ArtifactWriter(tmp_path)
    array = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)

    writer.array("sample", array)
    writer.png("sample_png", array)

    np.testing.assert_array_equal(np.load(tmp_path / "sample.npy"), array)
    from PIL import Image

    np.testing.assert_array_equal(
        np.asarray(Image.open(tmp_path / "sample_png.png")), array
    )
    assert writer.identities["sample"]["mode"] == "0444"
    assert writer.identities["sample_png"]["mode"] == "0444"
    with pytest.raises((FileExistsError, smoke.ContractError)):
        writer.array("sample", array)


def test_strict_json_rejects_nonfinite_and_non_json_values() -> None:
    with pytest.raises(ValueError, match="Non-finite"):
        smoke.strict_json_bytes({"value": float("nan")})
    with pytest.raises(TypeError, match="Unsupported"):
        smoke.strict_json_bytes({"value": object()})
    with pytest.raises(smoke.ContractError, match="strict JSON"):
        smoke.strict_json_loads(b'{"value": NaN}', field="probe")


def test_passed_payload_closes_full_semantic_schema(tmp_path: Path) -> None:
    payload = _passing_payload(tmp_path)

    smoke.validate_passed_raw_payload(payload)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(promotion_authorized=True),
        lambda value: value["source"].update(tracked_clean=False),
        lambda value: value["result"]["environment"]["scene_file"].update(
            sha256="f" * 64
        ),
        lambda value: value["result"]["production_path"].update(
            renderer_render_calls=2
        ),
        lambda value: value["result"]["contracts"][
            "removed_resize_counterfactual"
        ].update(live_path=True),
        lambda value: value["result"]["contracts"]["ego_lap_preprocessing"].update(
            padding_rows={"top": 48, "bottom": 50}
        ),
        lambda value: value["result"]["cameras"]["external_cam"]["conversion"].update(
            pixel_exact=False
        ),
        lambda value: value["result"]["artifacts"].update(unexpected={}),
        lambda value: value["result"]["artifacts"]["ego_lap_request_msgpack"].update(
            sha256="0" * 64
        ),
    ],
)
def test_passed_payload_rejects_semantic_tamper(tmp_path: Path, mutation) -> None:
    payload = copy.deepcopy(_passing_payload(tmp_path))
    mutation(payload)

    with pytest.raises(smoke.ContractError):
        smoke.validate_passed_raw_payload(payload)


def test_ready_marker_binds_exact_immutable_raw(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.json"
    smoke.publish_immutable_bytes(raw_path, b'{"ok": true}\n')
    raw_bytes = raw_path.read_bytes()
    marker = {
        "schema_version": smoke.SCHEMA_VERSION,
        "profile": smoke.PROFILE,
        "stage": "simulation_app_close_pending",
        "raw_result": smoke.raw_file_identity(raw_path),
    }

    smoke.validate_ready_marker(marker, raw_path, raw_bytes)
    marker["raw_result"]["size_bytes"] += 1
    with pytest.raises(smoke.ContractError, match="raw binding"):
        smoke.validate_ready_marker(marker, raw_path, raw_bytes)


def test_production_environment_source_closes_actual_call_path() -> None:
    tree = ast.parse(ENVIRONMENT_PATH.read_text())
    custom_render = _function(tree, "custom_render")
    render_splat = _function(tree, "render_splat")
    custom_source = ast.unparse(custom_render)
    render_source = ast.unparse(render_splat)

    assert "self.render_splat()" in custom_source
    assert "self.get_robot_from_sim()" in custom_source
    assert "np.where(mask, sim_img, og_img)" in custom_source
    assert "self.splat_renderer.render(cam_extrinsics_dict)" in render_source
    assert "splat_rgb_float_to_uint8(v)" in render_source
    assert ".resize(" not in render_source
    assert "cvtColor" not in render_source


def test_actual_ego_lap_source_closes_resize_pad_rotate_request_path() -> None:
    tree = ast.parse(CLIENT_PATH.read_text())
    model_images = _function(tree, "_model_images")
    wrist_preprocess = _function(tree, "preprocess_lap_wrist_image")
    build_request = _function(tree, "_build_request")
    model_source = ast.unparse(model_images)
    wrist_source = ast.unparse(wrist_preprocess)
    request_source = ast.unparse(build_request)

    assert "resize_lap_image(current['external_image'])" in model_source
    assert "preprocess_lap_wrist_image" in model_source
    resize_index = wrist_source.index("resize_lap_image")
    rotate_index = wrist_source.index("rotate_image_180")
    assert resize_index < rotate_index
    assert "self._model_images(current)" in request_source
    assert "'base_0_rgb': exterior_image" in request_source
    assert "'left_wrist_0_rgb': wrist_image" in request_source


def test_smoke_source_uses_real_client_without_network_and_seals_before_close() -> None:
    source = SCRIPT_PATH.read_text()
    tree = ast.parse(source)
    client_path = _function(tree, "_run_client_preprocessing")
    client_source = ast.unparse(client_path)
    capture_path = _function(tree, "_capture_production_render")
    capture_source = ast.unparse(capture_path)
    run_source = ast.unparse(_function(tree, "run_smoke"))

    assert "object.__new__(lap_module.EgoLAPEefPoseClient)" in client_source
    assert "client._extract_observation" in client_source
    assert "client._build_request" in client_source
    assert "msgpack_numpy.packb" in client_source
    assert "msgpack_numpy.unpackb" in client_source
    assert "WebsocketClientPolicy" not in client_source
    assert "runtime.custom_render(True, transform_static=False)" in capture_source
    assert "load_eval_initial_conditions" in run_source
    assert "rollouts=1" in run_source
    assert "object_positions=initial_conditions[0]" in run_source
    ready_publish = "publish_immutable_json(ready_path, ready_payload)"
    simulation_close = "simulation_app.close()"
    assert source.index(ready_publish) < source.index(simulation_close)


def test_smoke_has_no_checkpoint_action_metric_or_authorization_surface() -> None:
    parser = smoke.build_parser()
    option_strings = {
        option for action in parser._actions for option in action.option_strings
    }
    assert not any(
        forbidden in option
        for option in option_strings
        for forbidden in ("checkpoint", "server", "action", "rollout", "metric")
    )
    assert "promotion_authorized" in smoke.TOP_LEVEL_FIELDS
    assert smoke.SCOPE.endswith("no_checkpoint_policy_action_metric_or_canary")
