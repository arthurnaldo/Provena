import numpy as np
import pytest

from fingerprint import fingerprint_frame


def smooth_image(seed: int, size: int = 256) -> np.ndarray:
    """A deterministic, natural-ish RGB image: a sum of low-frequency waves."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size] / size
    gray = np.zeros((size, size))
    for _ in range(6):
        fx, fy = rng.uniform(0.5, 4, 2)
        gray += np.sin(2 * np.pi * (fx * x + fy * y) + rng.uniform(0, 2 * np.pi))
    gray = (gray - gray.min()) / (gray.max() - gray.min()) * 255
    return np.repeat(gray[..., None], 3, axis=2).astype(np.uint8)


def hamming(a: bytes, b: bytes) -> int:
    return int(np.unpackbits(np.frombuffer(a, np.uint8) ^ np.frombuffer(b, np.uint8)).sum())


def test_hash_shape_and_quality_range():
    pdq, quality = fingerprint_frame(smooth_image(0))
    assert len(pdq) == 32
    assert 0 <= quality <= 100


def test_deterministic():
    assert fingerprint_frame(smooth_image(0)) == fingerprint_frame(smooth_image(0))


def test_downscaled_image_is_close():
    image = smooth_image(0)
    half = image.reshape(128, 2, 128, 2, 3).mean(axis=(1, 3)).astype(np.uint8)
    assert hamming(fingerprint_frame(image)[0], fingerprint_frame(half)[0]) <= 31


def test_different_images_are_far():
    assert hamming(fingerprint_frame(smooth_image(0))[0], fingerprint_frame(smooth_image(1))[0]) > 31


def test_flat_image_has_low_quality():
    flat = np.full((256, 256, 3), 128, np.uint8)
    assert fingerprint_frame(flat)[1] <= 49


@pytest.mark.parametrize("bad", [np.zeros((64, 64), np.uint8), np.zeros((64, 64, 3), np.float32)])
def test_rejects_non_rgb_uint8(bad):
    with pytest.raises(ValueError):
        fingerprint_frame(bad)
