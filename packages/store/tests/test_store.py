import random
from datetime import datetime, timezone

from core import Frame, Video
from store import Store

POSTED = datetime(2026, 10, 1, tzinfo=timezone.utc)


def video(native_id: str = "abc") -> Video:
    return Video(platform="bluesky", native_id=native_id, url="https://example.com", channel_id="did:plc:x", posted_at=POSTED)


def random_hash(seed: int) -> bytes:
    return random.Random(seed).randbytes(32)


def frames(n: int, seed: int = 0, quality: int = 100) -> list[Frame]:
    return [Frame(offset=i * 0.25, pdq=random_hash(seed * 1000 + i), quality=quality) for i in range(n)]


def test_add_and_search_exact(tmp_path):
    s = Store(tmp_path / "db")
    v = s.add_video(video(), frames(10))
    hits = s.search([Frame(offset=3.0, pdq=random_hash(4), quality=100)])
    assert [(h.video_id, h.offset, h.distance, h.query_offset) for h in hits] == [(v.video_id, 1.0, 0, 3.0)]


def test_near_hash_matches_far_does_not(tmp_path):
    s = Store(tmp_path / "db")
    s.add_video(video(), frames(1))
    original = bytearray(random_hash(0))
    near = bytes(b ^ 0x01 for b in original[:20]) + bytes(original[20:])   # 20 bits flipped
    far = bytes(b ^ 0x03 for b in original)                                 # 64 bits flipped
    assert [h.distance for h in s.search([Frame(0, near, 100)])] == [20]
    assert s.search([Frame(0, far, 100)]) == []


def test_reingest_is_noop(tmp_path):
    s = Store(tmp_path / "db")
    first = s.add_video(video(), frames(5))
    again = s.add_video(video(), frames(5, seed=1))
    assert again.video_id == first.video_id
    assert s.index.ntotal == 5


def test_low_quality_frames_stored_but_not_searchable(tmp_path):
    s = Store(tmp_path / "db")
    s.add_video(video(), frames(3, quality=10))
    assert s.db.execute("SELECT COUNT(*) FROM frames").fetchone()[0] == 3
    assert s.search([Frame(0, random_hash(0), 100)]) == []


def test_index_rebuilt_on_reopen(tmp_path):
    s = Store(tmp_path / "db")
    v = s.add_video(video(), frames(5))
    s.close()
    s = Store(tmp_path / "db")
    assert [h.video_id for h in s.search([Frame(0, random_hash(2), 100)])] == [v.video_id]
    assert s.get_videos([v.video_id]) == [v]
