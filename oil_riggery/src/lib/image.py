from typing import Tuple

import numpy as np


def pad_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Pad an image to a target size.
    """
    assert image.shape[0] <= target_size[0]
    assert image.shape[1] <= target_size[1]

    padded_image = np.zeros(target_size, dtype=image.dtype)
    padded_image[: image.shape[0], : image.shape[1], :] = image

    return padded_image