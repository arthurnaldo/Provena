from fingerprint.frames import iter_frames
from fingerprint.pdq import pdq_frame

__all__ = ["QUERY_SECONDS_PER_HASH", "STORED_SECONDS_PER_HASH", "fingerprint_video", "iter_frames", "pdq_frame"]


def __getattr__(name):
    # vPDQ (built from source against ffmpeg) is imported only when used, so the audit client needs just pdqhash.
    if name in ("QUERY_SECONDS_PER_HASH", "STORED_SECONDS_PER_HASH", "fingerprint_video"):
        from fingerprint import vpdq
        return getattr(vpdq, name)
    raise AttributeError(name)
