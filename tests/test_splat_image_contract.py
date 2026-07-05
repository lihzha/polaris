import ast
from pathlib import Path

import numpy as np

from polaris.splat_image_contract import splat_rgb_float_to_uint8


class _TensorLike:
    def __init__(self, array: np.ndarray):
        self._array = array
        self.calls: list[str] = []

    def detach(self):
        self.calls.append("detach")
        return self

    def cpu(self):
        self.calls.append("cpu")
        return self

    def numpy(self):
        self.calls.append("numpy")
        return self._array


def test_splat_rgb_conversion_preserves_odd_shape_and_exact_pixels() -> None:
    image = np.empty((5, 7, 3), dtype=np.float32)
    image[..., 0] = -0.25
    image[..., 1] = 0.5
    image[..., 2] = 1.25
    tensor = _TensorLike(image)

    actual = splat_rgb_float_to_uint8(tensor)
    expected = np.empty((5, 7, 3), dtype=np.uint8)
    expected[..., 0] = 0
    expected[..., 1] = 127
    expected[..., 2] = 255

    assert tensor.calls == ["detach", "cpu", "numpy"]
    assert actual.shape == image.shape
    assert actual.dtype == np.uint8
    np.testing.assert_array_equal(actual, expected)


def test_splat_rgb_conversion_keeps_clipping_and_truncation_semantics() -> None:
    image = np.array(
        [
            [[0.0, 0.1, 0.25], [0.5, 0.75, 0.999]],
            [[1.0, 1.01, -0.01], [0.2, 0.4, 0.8]],
        ],
        dtype=np.float32,
    )
    expected = np.array(
        [
            [[0, 25, 63], [127, 191, 254]],
            [[255, 255, 0], [51, 102, 204]],
        ],
        dtype=np.uint8,
    )

    actual = splat_rgb_float_to_uint8(_TensorLike(image))

    np.testing.assert_array_equal(actual, expected)


def test_environment_converts_every_rendered_camera_without_cv2_resize() -> None:
    environment_path = (
        Path(__file__).parents[1]
        / "src/polaris/environments/manager_based_rl_splat_environment.py"
    )
    tree = ast.parse(environment_path.read_text())

    imported_modules = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "cv2" not in imported_modules

    render_splat = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "render_splat"
    )
    calls = [node for node in ast.walk(render_splat) if isinstance(node, ast.Call)]
    assert not any(
        isinstance(call.func, ast.Attribute) and call.func.attr == "resize"
        for call in calls
    )

    conversion_loops = [
        node
        for node in ast.walk(render_splat)
        if isinstance(node, ast.For) and ast.unparse(node.iter) == "rgb.items()"
    ]
    assert len(conversion_loops) == 1
    helper_calls = [
        node
        for node in ast.walk(conversion_loops[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "splat_rgb_float_to_uint8"
    ]
    assert len(helper_calls) == 1
    assert len(helper_calls[0].args) == 1
    assert isinstance(helper_calls[0].args[0], ast.Name)
    assert helper_calls[0].args[0].id == "v"
