from dataclasses import dataclass
from datetime import datetime


@dataclass
class Video:
    platform: str                    # "bluesky", "youtube", ...
    native_id: str                   # identifies one video on the platform; unique with platform
    url: str
    channel_id: str                  # stable platform ID (e.g. Bluesky DID), never a handle
    posted_at: datetime              # platform-reported, tz-aware UTC
    deleted_at: datetime | None = None   # None = still live
    video_id: int | None = None          # internal key; None until stored


@dataclass
class Frame:
    offset: float               # seconds into the video
    pdq: bytes                  # 32-byte PDQ hash
    quality: int                # PDQ quality, 0-100; low = flat/blurry frame
    video_id: int | None = None # None until stored, or for query clips
    frame_id: int | None = None # None until stored


@dataclass
class Hit:                      # one stored frame that looks like one query frame
    query_offset: float
    video_id: int
    offset: float               # seconds into the stored video
    distance: int               # Hamming distance, 0-256


@dataclass
class Match:
    video_id: int
    clip_range: tuple[float, float]   # seconds in the query clip that match
    video_range: tuple[float, float]  # corresponding seconds in the stored video
    shift: float                      # video seconds - clip seconds (median over supporting hits)
    votes: int                        # distinct clip frames supporting this match
