"""Our side: find duplicate footage in a fingerprint folder. Sees random IDs and fingerprints only.

Writes pairs.csv (one row per copy; a clip used twice in another video gives two rows) and prints the
duplicate time, counted against the lesser (shorter) copy of each pair with overlaps merged.

Many machines: `audit --part i --of N` on each writes one part file; `combine` joins them and prints the summary.
Every machine loads all fingerprints but queries only its share, and each pair belongs to exactly one part.
"""

import csv
import os

import numpy as np

from provena_audit import _kernel, shards
from provena_audit.storage import Location

MIN_QUALITY = 50  # flat, low-detail frames match everything
FPS = 4.0
FIELDS = ["a", "b", "votes", "shift", "a_start", "a_end", "b_start", "b_end"]


def find_pairs(data: dict, min_votes: int = 8, radius: int = 31, part: int = 0, of: int = 1) -> list[dict]:
    """Copies between different videos, oriented a < b; votes = distinct frames agreeing on one time shift."""
    rows = np.nonzero(data["quality"] >= MIN_QUALITY)[0]
    rows = rows[np.argsort(data["owner"][rows], kind="stable")]
    owners, video = np.unique(data["owner"][rows], return_inverse=True)
    out = _kernel.find_pairs(np.ascontiguousarray(data["pdq"][rows]).view(np.uint64), video.astype(np.int64),
                             data["offset"][rows].astype(np.float64), radius, min_votes, os.cpu_count() or 1,
                             part, of)
    found = [dict(zip(FIELDS, r)) for r in zip(*out)]
    for r in found:  # owners is sorted, so a < b stays a < b
        r["a"], r["b"] = int(owners[r["a"]]), int(owners[r["b"]])
    return found


def _merged_length(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, -np.inf
    for s, e in sorted(intervals):
        total += max(0.0, e - max(s, end))
        end = max(end, e)
    return total


def _write(path: str, rows: list[list]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        w.writerows(rows)


def _summary(rows: list[list], ids: np.ndarray, durations: np.ndarray, out: str) -> dict:
    """Duplicate time: for each pair the shorter copy carries its matched stretch (ties: larger ID), merged."""
    duration = dict(zip(ids.tolist(), durations.tolist()))
    dup: dict[str, list[tuple[float, float]]] = {}
    for a, b, _votes, _shift, a_start, a_end, b_start, b_end in rows:
        if (-duration[a], a) <= (-duration[b], b):
            dup.setdefault(b, []).append((float(b_start), float(b_end)))
        else:
            dup.setdefault(a, []).append((float(a_start), float(a_end)))
    dup_hours = sum(min(_merged_length(iv), duration[v]) for v, iv in dup.items()) / 3600
    hours = float(durations.sum()) / 3600
    print(f"{len(ids):,} videos, {hours:.1f} h: {len(rows):,} copies, {dup_hours:.2f} h duplicate "
          f"({dup_hours / hours if hours else 0:.1%}) -> {out}")
    return {"videos": len(ids), "hours": hours, "pairs": len(rows), "duplicate_hours": dup_hours,
            "videos_with_duplicates": len(dup)}


def run_audit(send: str, out: str = "pairs.csv", min_votes: int = 8, part: int = 0, of: int = 1) -> dict | None:
    data = shards.load(Location(send))
    ids, durations = data["ids"], data["durations"].astype(np.float64)
    rows = [[ids[p["a"]], ids[p["b"]], p["votes"], round(p["shift"], 2)]
            + [round(p[k] + (1 / FPS if k.endswith("end") else 0), 2) for k in FIELDS[4:]]
            for p in find_pairs(data, min_votes, part=part, of=of)]
    _write(out, rows)
    if of > 1:
        print(f"part {part + 1}/{of}: {len(rows):,} copies -> {out} (join the parts with `provena-audit combine`)")
        return None
    return _summary(rows, ids, durations, out)


def combine(send: str, parts: list[str], out: str = "pairs.csv") -> dict:
    """Join the part files of a split audit and print the summary (needs only IDs and durations from send/)."""
    rows = []
    for path in parts:
        with open(path, newline="") as f:
            rows += [r for r in csv.reader(f)][1:]
    _write(out, rows)
    data = shards.load(Location(send))
    return _summary(rows, data["ids"], data["durations"].astype(np.float64), out)
