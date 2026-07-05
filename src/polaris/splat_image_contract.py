"""Pixel-preserving conversion at the splat-renderer observation boundary."""

from typing import Any

import numpy as np


def splat_rgb_float_to_uint8(image: Any) -> np.ndarray:
    """Convert one renderer RGB tensor without changing its spatial sampling.

    Keep the historical PolaRiS clipping and truncating float-to-uint8 behavior
    exactly.  Training-matched policy preprocessing owns the only subsequent
    spatial resize.
    """

    image = image.detach().cpu().numpy()
    image = np.clip(image, 0, 1)
    return (image * 255).astype(np.uint8)
