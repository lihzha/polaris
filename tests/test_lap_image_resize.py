import unittest
from pathlib import Path

import numpy as np

from polaris.policy.lap_eef_pose_client import preprocess_lap_wrist_image
from polaris.policy.lap_eef_pose_client import resize_lap_image
from polaris.policy.lap_eef_pose_client import rotate_image_180

try:
    import tensorflow as tf
except ImportError:
    # TensorFlow is an optional test oracle, never a runtime dependency.
    tf = None


def _tensorflow_training_resize(
    image: np.ndarray, target_h: int = 224, target_w: int = 224
) -> np.ndarray:
    """Exact copy of Ego-LAP training's uint8 ``_tf_resize_with_pad`` path."""

    if tf is None:
        raise RuntimeError("TensorFlow test oracle is unavailable")
    in_h = tf.shape(image)[0]
    in_w = tf.shape(image)[1]
    h_f = tf.cast(in_h, tf.float32)
    w_f = tf.cast(in_w, tf.float32)
    ratio = tf.maximum(
        w_f / tf.cast(target_w, tf.float32),
        h_f / tf.cast(target_h, tf.float32),
    )
    resized_h = tf.cast(tf.math.floor(h_f / ratio), tf.int32)
    resized_w = tf.cast(tf.math.floor(w_f / ratio), tf.int32)
    resized_f32 = tf.image.resize(
        tf.cast(image, tf.float32),
        [resized_h, resized_w],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    resized = tf.cast(tf.clip_by_value(tf.round(resized_f32), 0.0, 255.0), tf.uint8)
    pad_h_total = target_h - resized_h
    pad_w_total = target_w - resized_w
    return tf.pad(
        resized,
        [
            [pad_h_total // 2, pad_h_total - pad_h_total // 2],
            [pad_w_total // 2, pad_w_total - pad_w_total // 2],
            [0, 0],
        ],
        constant_values=0,
    ).numpy()


def _tensorflow_training_wrist(
    image: np.ndarray, target_h: int = 224, target_w: int = 224
) -> np.ndarray:
    """Match ``make_decode_images_fn``: resize/pad first, then rotate 180."""

    resized = _tensorflow_training_resize(image, target_h, target_w)
    return tf.reverse(resized, axis=[0, 1]).numpy()


class LAPImageResizeTest(unittest.TestCase):
    def test_static_tensorflow_golden_and_symmetric_padding(self):
        image = np.array(
            [
                [[0, 10, 20], [30, 40, 50], [60, 70, 80]],
                [[90, 100, 110], [120, 130, 140], [150, 160, 170]],
            ],
            dtype=np.uint8,
        )
        expected = np.array(
            [
                [[0, 0, 0]] * 5,
                [[0, 10, 20], [12, 22, 32], [30, 40, 50], [48, 58, 68], [60, 70, 80]],
                [
                    [45, 55, 65],
                    [57, 67, 77],
                    [75, 85, 95],
                    [93, 103, 113],
                    [105, 115, 125],
                ],
                [
                    [90, 100, 110],
                    [102, 112, 122],
                    [120, 130, 140],
                    [138, 148, 158],
                    [150, 160, 170],
                ],
                [[0, 0, 0]] * 5,
            ],
            dtype=np.uint8,
        )

        actual = resize_lap_image(image, 5, 5)

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.uint8)
        self.assertTrue(actual.flags.c_contiguous)

    def test_invalid_image_contract_fails_loudly(self):
        with self.assertRaisesRegex(TypeError, "dtype uint8"):
            resize_lap_image(np.zeros((3, 4, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "nonzero"):
            resize_lap_image(np.zeros((0, 4, 3), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "positive"):
            resize_lap_image(np.zeros((3, 4, 3), dtype=np.uint8), 0, 224)

    @unittest.skipIf(tf is None, "TensorFlow is only needed as the golden test oracle")
    def test_synthetic_and_natural_images_match_tensorflow_training(self):
        rng = np.random.default_rng(42)
        images = {
            "polaris_720p": rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8),
            "droid_320x180": rng.integers(0, 256, size=(180, 320, 3), dtype=np.uint8),
            "odd_landscape": rng.integers(0, 256, size=(17, 29, 3), dtype=np.uint8),
            "odd_portrait": rng.integers(0, 256, size=(29, 17, 3), dtype=np.uint8),
        }
        natural_path = Path(__file__).parents[1] / "docs/images/Teaser Figure.png"
        images["natural"] = tf.io.decode_png(
            tf.io.read_file(str(natural_path)), channels=3
        ).numpy()

        for name, image in images.items():
            with self.subTest(name=name, shape=image.shape):
                expected = _tensorflow_training_resize(image)
                actual = resize_lap_image(image)
                np.testing.assert_array_equal(actual, expected)

    @unittest.skipIf(tf is None, "TensorFlow is only needed as the golden test oracle")
    def test_composed_wrist_pipeline_matches_training_order(self):
        rng = np.random.default_rng(42)
        images = {
            "polaris_720p": rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8),
            "odd_padding": rng.integers(0, 256, size=(17, 29, 3), dtype=np.uint8),
            "odd_portrait": rng.integers(0, 256, size=(29, 17, 3), dtype=np.uint8),
        }

        for name, image in images.items():
            with self.subTest(name=name, shape=image.shape):
                expected = _tensorflow_training_wrist(image)
                actual = preprocess_lap_wrist_image(image, rotate_180=True)
                np.testing.assert_array_equal(actual, expected)

                wrong_order = resize_lap_image(rotate_image_180(image))
                self.assertFalse(
                    np.array_equal(wrong_order, expected),
                    msg=f"probe {name} does not distinguish the operation order",
                )


if __name__ == "__main__":
    unittest.main()
