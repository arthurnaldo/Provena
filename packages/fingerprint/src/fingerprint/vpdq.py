from pathlib import Path

import vpdq

from core import Frame

# Stored videos are sampled densely so a query frame is never far (in time) from a stored one;
# at 1/s, clips cut off the sampling grid failed to match. Queries only need 1/s.
STORED_SECONDS_PER_HASH = 0.25
QUERY_SECONDS_PER_HASH = 1.0


def fingerprint_video(path: str | Path, seconds_per_hash: float = STORED_SECONDS_PER_HASH) -> list[Frame]:
    """Hash one frame every `seconds_per_hash` seconds with Meta's vPDQ.

    Returns frames without IDs. Low-quality frames are kept; filtering happens downstream.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        Frame(offset=f.timestamp, pdq=bytes.fromhex(f.hex), quality=f.quality)
        for f in vpdq.computeHash(str(path), seconds_per_hash=seconds_per_hash)
    ]
