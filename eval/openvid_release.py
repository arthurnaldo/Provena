"""Build the deduplicated OpenVid stock release from the full audit: which clips to drop, and why.

    uv run python eval/openvid_release.py data/eval/openvid/onepass_cloud_all data/eval/openvid/audit_cloud/audit_pairs.csv

Writes to data/eval/openvid/release/:
  drop_list.csv   clip, duplicate_of, kind, votes   (one row per clip to drop)
  keep_list.txt   every stock clip not dropped
  README.md       numbers + method

Counted as duplicates:
  - same source ID + segment (one video listed twice), any tier: confirmed by fingerprints
  - different IDs, PDQ match with >= 8 agreeing frames (60/60 verified by eye)
Each pair drops at most one clip, and never a clip whose partner is already dropped, so static footage
can't chain unrelated segments of one video together.
"""

import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from openvid_full_audit import source_key  # noqa: E402

MIN_VOTES = 8


def segment(name: str) -> str:
    return re.search(r"_(\d+)\.mp4$", name).group(1)


def main(shards: Path, pairs_csv: Path, out: Path) -> None:
    frames: dict[str, int] = {}
    for f in sorted(shards.rglob("shard_*.npz")):
        z = np.load(f)
        counts = np.bincount(z["owner"], minlength=len(z["names"]))
        for n, c in zip(z["names"], counts):
            frames.setdefault(str(n), int(c))
    print(f"{len(frames):,} clips")

    def rank(c):  # better copy first: longer clip, then lower numeric ID, then name
        sid = source_key(c)[1]
        return (-frames.get(c, 0), int(sid) if sid.isdigit() else 1 << 62, c)

    pairs = []
    for r in csv.DictReader(open(pairs_csv)):
        a, b = r["clip_a"], r["clip_b"]
        same_id = source_key(a)[1] == source_key(b)[1]
        if same_id and segment(a) == segment(b):
            kind = "same video listed twice"
        elif not same_id and r["tier"] == "pdq" and int(r["votes"]) >= MIN_VOTES:
            kind = "hidden duplicate (different ID)"
        else:
            continue
        pairs.append((kind, int(r["votes"]), a, b))
    # Same-ID pairs first (certain), then hidden pairs strongest first.
    pairs.sort(key=lambda p: (p[0] != "same video listed twice", -p[1]))

    dropped: dict[str, tuple[str, str, int]] = {}
    for kind, votes, a, b in pairs:
        if a in dropped or b in dropped:
            continue
        keep, drop = sorted((a, b), key=rank)
        dropped[drop] = (keep, kind, votes)

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "drop_list.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "duplicate_of", "kind", "votes"])
        for c in sorted(dropped):
            w.writerow([c, *dropped[c]])
    keep = sorted(set(frames) - set(dropped))
    (out / "keep_list.txt").write_text("\n".join(keep) + "\n")

    n = len(frames)
    by_kind = {}
    for _, kind, _ in dropped.values():
        by_kind[kind] = by_kind.get(kind, 0) + 1
    print(f"drop {len(dropped):,} ({len(dropped) / n:.1%}), keep {len(keep):,}")
    for k, v in by_kind.items():
        print(f"  {k}: {v:,} ({v / n:.1%})")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else
         Path(__file__).resolve().parent.parent / "data" / "eval" / "openvid" / "release")
