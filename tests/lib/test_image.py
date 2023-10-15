import cv2
import numpy as np
import pytest

from oil_riggery.src.lib.image import pad_image


def test_pad_image():
    one_channel = np.array([
        [1, 2],
        [3, 4],
    ])
    image = np.stack([one_channel, one_channel, one_channel], axis=-1)

    padded_image = pad_image(image, (4, 4, 3))

    assert padded_image.shape == (4, 4, 3)
    assert np.all(padded_image[:2, :2, :] == image)
    assert np.all(padded_image[2:, 2:, :] == 0)

@pytest.mark.skip(reason="this test is only for visual inspection")
def test_pad_image_visual():
    image = cv2.imread("tests/fixtures/sample.png")
    padded_image = pad_image(image, (1024, 1024, 3))
    cv2.imwrite("tests/fixtures/sample_padded.png", padded_image)