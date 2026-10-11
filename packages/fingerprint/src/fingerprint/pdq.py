import numpy as np
import pdqhash


def pdq_frame(image: np.ndarray) -> tuple[bytes, int]:
    """32-byte PDQ hash and quality (0-100) of one RGB uint8 frame.

    Bytes are in Meta's reference order (identical to vPDQ's output for the same frame).
    """
    bits, quality = pdqhash.compute(np.ascontiguousarray(image))
    return np.packbits(bits.astype(np.uint8)).tobytes(), int(quality)


def pdq_frames(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PDQ for a stack of frames → (hashes uint8 (n, 32), quality uint8 (n,))."""
    out = [pdq_frame(f) for f in frames]
    hashes = np.frombuffer(b"".join(h for h, _ in out), np.uint8).reshape(-1, 32)
    return hashes, np.array([q for _, q in out], np.uint8)


def pdq_dihedral(image: np.ndarray) -> list[bytes]:
    """PDQ of all 8 flips/rotations of a frame in one pass (for undoing mirrors/rotations at query time)."""
    vectors, _ = pdqhash.compute_dihedral(np.ascontiguousarray(image))
    return [np.packbits(v.astype(np.uint8)).tobytes() for v in vectors]
