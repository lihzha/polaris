import copy
import importlib.metadata
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import polaris.pi05_droid_jointpos_image_contract as image


_ORIGINAL_METADATA_VERSION = importlib.metadata.version


def _pin_pillow(monkeypatch, version=image.PILLOW_VERSION):
    monkeypatch.setattr(
        image.importlib_metadata,
        "version",
        lambda name: version if name == "Pillow" else _ORIGINAL_METADATA_VERSION(name),
    )
    monkeypatch.setattr(image, "_IMAGE_TOOLS_CACHE", {})


class _TensorLike:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def _renderer(seed):
    rng = np.random.default_rng(seed)
    return rng.random((720, 1280, 3), dtype=np.float32)


def test_filter_and_client_resize_probes_are_exact(monkeypatch):
    _pin_pillow(monkeypatch)
    raw = _renderer(0)
    post, stages = image.filter_renderer_layer(
        raw, cv2_module=cv2, camera_name="external_cam"
    )
    assert stages["renderer_float"]["dtype"] == "float32"
    assert stages["half_resolution_uint8"]["shape"] == [360, 640, 3]
    assert (
        stages["post_filter_splat_uint8"]["sha256"] == image._identity(post)["sha256"]
    )

    from openpi_client import image_tools

    wire, resize = image.resize_final_composite_for_wire(
        post, image_tools_module=image_tools, camera_name="external_cam"
    )
    assert wire.shape == (224, 224, 3)
    assert resize["input_final_composite"] == image._identity(post)
    assert resize["wire_request"] == image._identity(wire)
    assert image_tools.resize_with_pad(wire, 224, 224) is wire
    assert resize["runtime"] == {
        "profile": image.CLIENT_RESIZE_PROFILE,
        "implementation": "openpi_client.image_tools.resize_with_pad",
        "backend": "PIL.Image.resize",
        "method": "PIL.Image.Resampling.BILINEAR",
        "padding": "symmetric_zero",
        "source": resize["runtime"]["source"],
        "probe_output_sha256": image.CLIENT_RESIZE_PROBE_SHA256,
        "server_224_to_224": "early_return_same_array_no_pixel_change",
        "pillow_version": image.PILLOW_VERSION,
        "pillow_module": "PIL.Image",
    }
    assert resize["runtime"]["source"]["sha256"] == image.IMAGE_TOOLS_SOURCE_SHA256
    assert image.validate_client_resize_evidence(resize) == resize


def test_filter_probe_rejects_interpolation_drift():
    class NearestCV2:
        __version__ = cv2.__version__
        __file__ = "/tmp/nearest-cv2.py"
        INTER_LINEAR = cv2.INTER_LINEAR

        @staticmethod
        def resize(value, size, interpolation):
            return cv2.resize(value, size, interpolation=cv2.INTER_NEAREST)

    with pytest.raises(ValueError, match="probe"):
        image.filter_renderer_layer(
            _renderer(1), cv2_module=NearestCV2, camera_name="external_cam"
        )


class _SplatRenderer:
    def __init__(self, raw):
        self.raw = raw
        self.pcds = {"scene": object()}

    def render(self, _extrinsics):
        result = {name: _TensorLike(value.copy()) for name, value in self.raw.items()}
        self.last_render_result = result
        return result


class _Env:
    @property
    def unwrapped(self):
        return self

    def __init__(self):
        self.raw = {"external_cam": _renderer(2), "wrist_cam": _renderer(3)}
        self.splat_renderer = _SplatRenderer(self.raw)

    def render_splat(self):
        values = self.splat_renderer.render({})
        result = {}
        for name, value in values.items():
            pre = (np.clip(value.numpy(), 0, 1) * 255).astype(np.uint8)
            half = cv2.resize(pre, (640, 360))
            result[name] = cv2.resize(half, (1280, 720))
        self.last_render_splat_result = result
        return result

    def get_robot_from_sim(self):
        result = {}
        for index, name in enumerate(("external_cam", "wrist_cam")):
            mask = np.zeros((720, 1280, 1), dtype=np.int64)
            mask[10 + index : 30 + index, 20:40] = 1
            result[name] = {
                "mask": mask,
                "rgb": np.full((720, 1280, 3), 200 + index, dtype=np.uint8),
            }
        self.last_robot_result = result
        return result

    def custom_render(self, expensive, transform_static=False):
        assert transform_static in {False, True}
        if not expensive:
            result = {
                name: value["rgb"] for name, value in self.get_robot_from_sim().items()
            }
            self.last_custom_result = result
            return result
        rgb = self.render_splat()
        robot = self.get_robot_from_sim()
        for name, value in robot.items():
            background = rgb[name] if name in rgb else np.zeros_like(value["rgb"])
            rgb[name] = np.where(value["mask"], value["rgb"], background)
        self.last_custom_result = rgb
        return rgb


def test_instance_instrumentation_is_behavior_neutral_and_hash_binds_layers(
    monkeypatch,
):
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    baseline_env = _Env()
    baseline = baseline_env.custom_render(True)
    env = _Env()
    image.install_jointpos_image_instrumentation(env)
    observed = env.custom_render(True)
    assert observed is env.last_custom_result
    assert observed is env.last_render_splat_result
    for name in image.CAMERA_NAMES:
        np.testing.assert_array_equal(observed[name], baseline[name])
    evidence = image.get_jointpos_image_evidence(env, {"splat": observed})
    for name in image.CAMERA_NAMES:
        assert evidence[name]["background_source"] == "filtered_splat"
        assert evidence[name]["composite_mask_coverage"]["true_pixel_count"] == 400
        assert evidence[name]["final_composite_uint8"] == image._identity(
            observed[name]
        )

    changed = copy.deepcopy(observed)
    changed["external_cam"][0, 0, 0] ^= np.uint8(1)
    with pytest.raises(ValueError, match="hash differs"):
        image.get_jointpos_image_evidence(env, {"splat": changed})


def test_shared_manager_remains_exact_upstream_blob():
    contract = image.source_contract()
    assert contract["manager"]["sha256"] == image.MANAGER_SOURCE_SHA256
    assert contract["splat_renderer"]["sha256"] == image.SPLAT_RENDERER_SOURCE_SHA256
    assert (
        contract["helper"]["path"]
        == "src/polaris/pi05_droid_jointpos_image_contract.py"
    )
    assert contract["manager"]["path"].startswith("src/")
    assert contract["splat_renderer"]["path"].startswith("src/")


def test_instrumentation_rejects_no_pcd_and_double_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    no_pcd = _Env()
    no_pcd.splat_renderer.pcds = {}
    with pytest.raises(ValueError, match="nonempty splats"):
        image.install_jointpos_image_instrumentation(no_pcd)

    env = _Env()
    image.install_jointpos_image_instrumentation(env)
    with pytest.raises(RuntimeError, match="installed twice"):
        image.install_jointpos_image_instrumentation(env)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.zeros((720, 1280, 3), dtype=np.float64), "renderer layer"),
        (np.zeros((719, 1280, 3), dtype=np.float32), "renderer layer"),
    ],
)
def test_filter_rejects_wrong_renderer_dtype_or_shape(value, message):
    with pytest.raises(ValueError, match=message):
        image.filter_renderer_layer(value, cv2_module=cv2, camera_name="external_cam")


@pytest.mark.parametrize("dtype", [np.uint8, np.int32])
def test_instrumentation_rejects_non_int64_manager_mask(monkeypatch, dtype):
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    env = _Env()

    def wrong_robot():
        result = _Env.get_robot_from_sim(env)
        for value in result.values():
            value["mask"] = value["mask"].astype(dtype)
        return result

    env.get_robot_from_sim = wrong_robot
    image.install_jointpos_image_instrumentation(env)
    with pytest.raises(ValueError, match="semantic mask"):
        env.custom_render(True)


def test_camera_validator_rejects_mask_source_and_cv2_runtime_drift(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    env = _Env()
    image.install_jointpos_image_instrumentation(env)
    observed = env.custom_render(True)
    evidence = image.get_jointpos_image_evidence(env, {"splat": observed})[
        "external_cam"
    ]

    mutations = [
        ("background_source", "official_missing_renderer_zero_fallback"),
        ("camera_name", "wrist_cam"),
    ]
    for key, value in mutations:
        changed = copy.deepcopy(evidence)
        changed[key] = value
        with pytest.raises(ValueError):
            image.validate_camera_evidence(
                changed, camera_name="external_cam", require_filtered_splat=True
            )

    changed = copy.deepcopy(evidence)
    changed["composite_mask_coverage"] = {
        "true_pixel_count": 0,
        "total_pixel_count": 720 * 1280,
        "true_fraction": 0.0,
    }
    with pytest.raises(ValueError, match="mask coverage"):
        image.validate_camera_evidence(changed, camera_name="external_cam")

    for key, value in (
        ("package_version", "4.11.0.85"),
        ("interpolation", "cv2.INTER_NEAREST"),
        ("probe", {}),
    ):
        changed = copy.deepcopy(evidence)
        changed["renderer_stages"]["cv2_runtime"][key] = value
        with pytest.raises(ValueError, match="cv2 runtime"):
            image.validate_camera_evidence(changed, camera_name="external_cam")


def test_client_resize_rejects_pillow_source_runtime_and_probe_drift(
    monkeypatch, tmp_path
):
    from openpi_client import image_tools

    _pin_pillow(monkeypatch)
    _, evidence = image.resize_final_composite_for_wire(
        np.zeros(image.FINAL_IMAGE_SHAPE, dtype=np.uint8),
        image_tools_module=image_tools,
        camera_name="external_cam",
    )
    for key, value in (
        ("backend", "cv2.resize"),
        ("method", "nearest"),
        ("padding", "none"),
        ("pillow_version", "11.2.1"),
        ("pillow_module", "PIL.ImageOps"),
    ):
        changed = copy.deepcopy(evidence)
        changed["runtime"][key] = value
        with pytest.raises(ValueError, match="runtime"):
            image.validate_client_resize_evidence(changed)
    changed = copy.deepcopy(evidence)
    changed["runtime"]["source"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source identity"):
        image.validate_client_resize_evidence(changed)

    _pin_pillow(monkeypatch, version="11.2.1")
    with pytest.raises(ValueError, match="Pillow version mismatch"):
        image.resize_final_composite_for_wire(
            np.zeros(image.FINAL_IMAGE_SHAPE, dtype=np.uint8),
            image_tools_module=image_tools,
            camera_name="external_cam",
        )

    _pin_pillow(monkeypatch)

    def wrong_resize(value, height, width):
        if value.shape[-3:-1] == (height, width):
            return value
        return np.zeros((height, width, 3), dtype=np.uint8)

    wrong_image_tools = SimpleNamespace(
        __file__=image_tools.__file__, resize_with_pad=wrong_resize
    )
    with pytest.raises(ValueError, match="resize probe"):
        image.resize_final_composite_for_wire(
            np.zeros(image.FINAL_IMAGE_SHAPE, dtype=np.uint8),
            image_tools_module=wrong_image_tools,
            camera_name="external_cam",
        )

    target = tmp_path / "source.py"
    target.write_bytes(b"source")
    link = tmp_path / "source-link.py"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="regular file"):
        image._source_identity(link, image._sha256(target.read_bytes()), "test symlink")


def test_helper_source_identity_rejects_symlink(monkeypatch, tmp_path):
    target = tmp_path / "helper.py"
    target.write_bytes(b"helper")
    link = tmp_path / "helper-link.py"
    link.symlink_to(target)
    monkeypatch.setattr(image, "__file__", str(link))
    with pytest.raises(ValueError, match="regular file"):
        image._helper_source_identity()
