import faiss
import numpy as np


def pdq_neighbours(hashes: np.ndarray, radius: int = 31, k: int = 100, batch: int = 200_000):
    """All-against-all PDQ search over (n, 32) uint8 hashes.

    Yields (query rows, hit rows) batch by batch for pairs within `radius` bits, excluding each row itself.
    k caps hits per frame (exact search; k-nearest, then the radius cut).
    """
    index = faiss.IndexBinaryFlat(256)
    index.add(np.ascontiguousarray(hashes))
    for i in range(0, len(hashes), batch):
        D, I = index.search(np.ascontiguousarray(hashes[i:i + batch]), min(k, len(hashes)))
        r, c = np.nonzero((D <= radius) & (I >= 0))
        q, h = r + i, I[r, c]
        keep = q != h
        yield q[keep], h[keep]
