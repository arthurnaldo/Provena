import random

from core import Hit
from matcher import match


def aligned(video_id: int, shift: float, clip_frames: int = 20) -> list[Hit]:
    return [Hit(query_offset=t, video_id=video_id, offset=t + shift, distance=10) for t in range(clip_frames)]


def test_clean_match():
    [m] = match(aligned(1, 20))
    assert (m.video_id, m.shift, m.votes) == (1, 20, 20)
    assert m.clip_range == (0, 19) and m.video_range == (20, 39)


def test_shift_on_bin_edge_is_not_split():
    # True shift 20.5: hits land at 20.4 and 20.6 alternately, i.e. bins 20 and 21.
    hits = [Hit(t, 1, t + (20.4 if t % 2 else 20.6), 10) for t in range(20)]
    [m] = match(hits)
    assert m.votes == 20 and m.shift in (20, 21)


def test_scattered_lookalikes_do_not_match():
    rng = random.Random(0)
    hits = [Hit(t, rng.randrange(5), rng.uniform(0, 600), 25) for t in range(20) for _ in range(3)]
    assert match(hits) == []


def test_original_and_repost_are_separate_matches():
    ms = match(aligned(1, 20) + aligned(2, 5))
    assert {(m.video_id, m.shift) for m in ms} == {(1, 20), (2, 5)}


def test_below_min_votes():
    assert match(aligned(1, 20, clip_frames=4)) == []
