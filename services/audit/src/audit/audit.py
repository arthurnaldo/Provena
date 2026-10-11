"""Our side: find duplicate footage across a customer's fingerprint bundle. Sees IDs and fingerprints only.

Duplicate time is counted conservatively: for each matched pair, the matched stretch counts against the
lesser copy only (shorter video; ties by ID), and overlapping stretches within a video are merged, so
nothing is counted twice and the best copy of any footage is never counted.
"""

import numpy as np

from audit import prints
from audit.storage import Location
from matcher import vote_pairs
from store import pdq_neighbours

MIN_QUALITY = 50


def find_pairs(data: dict, min_votes: int = 8, radius: int = 31) -> dict[str, np.ndarray]:
    """Undirected matched pairs (a < b) with votes, shift (b time − a time) and matched ranges."""
    rows = np.nonzero(data["quality"] >= MIN_QUALITY)[0]
    owner, offset = data["owner"], data["offset"].astype(np.float64)
    qs, hs = [], []
    for q, h in pdq_neighbours(data["pdq"][rows], radius=radius):
        q, h = rows[q], rows[h]
        keep = owner[q] != owner[h]  # repeats inside one video (loops) are a separate check, not built yet
        qs.append(q[keep]); hs.append(h[keep])
    q, h = (np.concatenate(qs), np.concatenate(hs)) if qs else (np.array([], np.int64),) * 2
    p = vote_pairs(q, owner[q], offset[q], owner[h], offset[h], min_votes=min_votes)

    # Both directions are searched; keep the stronger one per unordered pair, oriented a < b.
    flip = p["a"] > p["b"]
    a, b = np.where(flip, p["b"], p["a"]), np.where(flip, p["a"], p["b"])
    a_start, b_start = np.where(flip, p["b_start"], p["a_start"]), np.where(flip, p["a_start"], p["b_start"])
    a_end, b_end = np.where(flip, p["b_end"], p["a_end"]), np.where(flip, p["a_end"], p["b_end"])
    shift = np.where(flip, -p["shift"], p["shift"])
    order = np.lexsort((-p["votes"], b, a))
    first = np.ones(len(order), bool)
    first[1:] = (a[order][1:] != a[order][:-1]) | (b[order][1:] != b[order][:-1])
    o = order[first]
    return {"a": a[o], "b": b[o], "votes": p["votes"][o], "shift": shift[o],
            "a_start": a_start[o], "a_end": a_end[o], "b_start": b_start[o], "b_end": b_end[o]}


def _merged_length(intervals: list[tuple[float, float]]) -> float:
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + (cur_e - cur_s if cur_e is not None else 0.0)


def _groups(n: int, a: np.ndarray, b: np.ndarray) -> list[list[int]]:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for x, y in zip(a.tolist(), b.tolist()):
        parent[find(x)] = find(y)
    groups: dict[int, list[int]] = {}
    for v in sorted(set(a.tolist()) | set(b.tolist())):
        groups.setdefault(find(v), []).append(v)
    return sorted(groups.values(), key=len, reverse=True)


def run_audit(send: str, min_votes: int = 8, rate_per_hour: tuple[float, float] = (60.0, 240.0)) -> dict:
    data = prints.load(Location(send))
    fps = 4.0
    ids, durations = data["ids"], data["durations"].astype(np.float64)
    p = find_pairs(data, min_votes=min_votes)

    # Lesser copy of each pair carries the duplicate time.
    def rank(v):  # better copy sorts first
        return (-durations[v], ids[v])
    dup: dict[int, list[tuple[float, float]]] = {}
    for i in range(len(p["a"])):
        a, b = int(p["a"][i]), int(p["b"][i])
        if rank(a) <= rank(b):
            dup.setdefault(b, []).append((p["b_start"][i], p["b_end"][i] + 1 / fps))
        else:
            dup.setdefault(a, []).append((p["a_start"][i], p["a_end"][i] + 1 / fps))
    dup_seconds = {v: min(_merged_length(iv), durations[v]) for v, iv in dup.items()}

    total_h = durations.sum() / 3600
    dup_h = sum(dup_seconds.values()) / 3600
    pairs_by_video: dict[int, list[int]] = {}
    for i in range(len(p["a"])):
        pairs_by_video.setdefault(int(p["a"][i]), []).append(i)
    groups = []
    for g in _groups(len(ids), p["a"], p["b"]):
        gp = [i for v in g for i in pairs_by_video.get(v, [])]
        groups.append({
            "videos": [{"id": ids[v], "duration": round(float(durations[v]), 2),
                        "duplicate_seconds": round(float(dup_seconds.get(v, 0.0)), 2)} for v in g],
            "pairs": [{"a": ids[p["a"][i]], "b": ids[p["b"][i]], "votes": int(p["votes"][i]),
                       "shift": round(float(p["shift"][i]), 2),
                       "a_range": [round(float(p["a_start"][i]), 2), round(float(p["a_end"][i] + 1 / fps), 2)],
                       "b_range": [round(float(p["b_start"][i]), 2), round(float(p["b_end"][i] + 1 / fps), 2)]}
                      for i in sorted(gp, key=lambda i: -p["votes"][i])],
        })
    return {
        "summary": {
            "videos": len(ids), "hours": round(total_h, 2),
            "videos_with_duplicate_footage": len(dup_seconds),
            "duplicate_hours": round(dup_h, 2),
            "duplicate_share": round(dup_h / total_h, 4) if total_h else 0.0,
            "matched_pairs": len(p["a"]), "groups": len(groups), "min_votes": min_votes,
            "rate_per_hour": list(rate_per_hour),
            "duplicate_cost": [round(dup_h * r) for r in rate_per_hour],
        },
        "groups": groups,
    }
