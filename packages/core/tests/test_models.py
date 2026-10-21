from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone

import pytest

from core import Frame, Video

NOW = datetime(2026, 10, 6, tzinfo=timezone.utc)


def make_video(**overrides) -> Video:
    fields = dict(
        platform="telegram",
        native_id="1234567890/123",
        url="https://t.me/c/1234567890/123",
        channel_id="1234567890",
        channel_name="Example News",
        posted_at=NOW,
        ingested_at=NOW,
        last_seen_live_at=NOW,
    )
    return Video(**(fields | overrides))


def test_video_defaults():
    v = make_video()
    assert v.video_id is None
    assert v.deleted_at is None


def test_video_key():
    assert make_video().key == "telegram:1234567890/123"


def test_video_is_immutable():
    v = make_video()
    with pytest.raises(FrozenInstanceError):
        v.deleted_at = NOW
    assert replace(v, deleted_at=NOW).deleted_at == NOW


def test_frame_ids_default_to_none():
    f = Frame(offset=1.0, pdq=bytes(32), quality=100)
    assert f.video_id is None
    assert f.frame_id is None
