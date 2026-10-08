import math
import statistics
from collections import defaultdict

from core import Hit, Match


def match(hits: list[Hit], min_votes: int = 5) -> list[Match]:
    """Hough voting on time shift: hits that agree on (video, stored - query offset) form a match.

    Shifts are binned to 1 s. A bin's score counts distinct query frames in it and its ±1 neighbours,
    so a true shift near a bin edge (e.g. 20.5 s) isn't split in half. Neighbouring bins of an
    accepted match are suppressed so one alignment is reported once.
    """
    bins: dict[tuple[int, int], list[Hit]] = defaultdict(list)
    for h in hits:
        bins[(h.video_id, math.floor(h.offset - h.query_offset + 0.5))].append(h)

    def support(video_id: int, shift: int) -> list[Hit]:
        return [h for s in (shift - 1, shift, shift + 1) for h in bins.get((video_id, s), [])]

    def votes(video_id: int, shift: int) -> int:
        return len({h.query_offset for h in support(video_id, shift)})

    # Ties between neighbouring bins go to the bin with the most votes of its own.
    candidates = sorted(bins, key=lambda key: (-votes(*key), -len({h.query_offset for h in bins[key]}), key))
    accepted: list[tuple[int, int]] = []
    matches = []
    for video_id, shift in candidates:
        n = votes(video_id, shift)
        if n < min_votes:
            break
        if any(v == video_id and abs(s - shift) <= 2 for v, s in accepted):
            continue
        accepted.append((video_id, shift))
        hs = support(video_id, shift)
        matches.append(Match(
            video_id=video_id,
            clip_range=(min(h.query_offset for h in hs), max(h.query_offset for h in hs)),
            video_range=(min(h.offset for h in hs), max(h.offset for h in hs)),
            shift=statistics.median(h.offset - h.query_offset for h in hs),
            votes=n,
        ))
    return matches
