from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Video:
    platform: str                    # "youtube", "telegram", ...
    native_id: str                   # identifies one video on the platform; unique with platform
    url: str
    channel_id: str                  # stable platform ID, never a handle
    channel_name: str                # display name as last seen; may change
    posted_at: datetime              # platform-reported, tz-aware UTC
    ingested_at: datetime            # when we first recorded it, tz-aware UTC
    last_seen_live_at: datetime      # tz-aware UTC
    deleted_at: datetime | None = None   # None = still live
    video_id: int | None = None          # internal key; None until stored

    @property
    def key(self) -> str:
        return f"{self.platform}:{self.native_id}"

@dataclass
class Frame:
    offset: float               # seconds into the video
    pdq: bytes                  # 32-byte PDQ hash
    quality: int                # PDQ quality, 0-100; low = flat/blurry frame
    video_id: int | None = None # FK → Video.video_id; None until stored, or for query clips
    frame_id: int | None = None # primary key; None until stored


@dataclass
class Match:                    # returned, not stored
    video_id: int
    clip_range: tuple[float, float]   # seconds in the query clip that match
    video_range: tuple[float, float]  # corresponding seconds in the stored video
    confidence: float


@dataclass
class QueryResult:
    clip_id: str
    matches: list[Match]        # empty = no match
