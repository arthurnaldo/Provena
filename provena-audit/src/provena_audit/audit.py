"""Our side: find duplicate footage in a fingerprint folder. Sees random IDs and fingerprints only.

Writes pairs.csv (one row per copy; a clip used twice in another video gives two rows) and prints the
duplicate time, counted against the lesser (shorter) copy of each pair with overlaps merged.
"""

import csv
import os

import numpy as np

from provena_audit import _kernel, shards
from provena_audit.storage import Location

MIN_QUALITY = 50  # flat, low-detail frames match everything
FPS = 4.0
FIELDS = ["a", "b", "votes", "shift", "a_start", "a_end", "b_start", "b_end"]


def find_pairs(data: dict, min_votes: int = 8, radius: int = 31) -> list[dict]:
    """Copies between different videos, oriented a < b; votes = distinct frames agreeing on one time shift."""
    rows = np.nonzero(data["quality"] >= MIN_QUALITY)[0]
    rows = rows[np.argsort(data["owner"][rows], kind="stable")]
    owners, video = np.unique(data["owner"][rows], return_inverse=True)
    out = _kernel.find_pairs(np.ascontiguousarray(data["pdq"][rows]).view(np.uint64), video.astype(np.int64),
                             data["offset"][rows].astype(np.float64), radius, min_votes, os.cpu_count() or 1)
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


def run_audit(send: str, out: str = "pairs.csv", min_votes: int = 8) -> dict:
    data = shards.load(Location(send))
    ids, durations = data["ids"], data["durations"].astype(np.float64)
    pairs = find_pairs(data, min_votes)

    dup: dict[int, list[tuple[float, float]]] = {}
    for p in pairs:  # the shorter copy carries the duplicate time (ties: larger ID)
        a, b = p["a"], p["b"]
        lesser = "b" if (-durations[a], ids[a]) <= (-durations[b], ids[b]) else "a"
        dup.setdefault(p[lesser], []).append((p[f"{lesser}_start"], p[f"{lesser}_end"] + 1 / FPS))
    dup_hours = sum(min(_merged_length(iv), durations[v]) for v, iv in dup.items()) / 3600

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for p in pairs:
            w.writerow([ids[p["a"]], ids[p["b"]], p["votes"], round(p["shift"], 2)]
                       + [round(p[k] + (1 / FPS if k.endswith("end") else 0), 2) for k in FIELDS[4:]])
    hours = durations.sum() / 3600
    summary = {"videos": len(ids), "hours": hours, "pairs": len(pairs), "duplicate_hours": dup_hours,
               "videos_with_duplicates": len(dup)}
    print(f"{len(ids):,} videos, {hours:.1f} h: {len(pairs):,} copies, {dup_hours:.2f} h duplicate "
          f"({dup_hours / hours if hours else 0:.1%}) -> {out}")
    return summary
