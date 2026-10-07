import numpy as np
import pdqhash


def fingerprint_frame(image: np.ndarray) -> tuple[bytes, int]:
    """Return the 32-byte PDQ hash and quality (0-100) of an RGB uint8 image of shape (H, W, 3).

    Bytes are in the reference implementation's order, so `.hex()` matches Meta's PDQ hex output.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"expected RGB uint8 image (H, W, 3), got {image.dtype} {image.shape}")
    bits, quality = pdqhash.compute(np.ascontiguousarray(image))
    return np.packbits(bits.astype(np.uint8)).tobytes(), int(quality)
