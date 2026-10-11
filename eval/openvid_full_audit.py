"""Duplicate audit over the one-pass OpenVid shards (PDQ + SSCD), all clips against all clips.

    modal volume get provena-openvid onepass data/eval/openvid/onepass_cloud
    uv run python eval/openvid_full_audit.py data/eval/openvid/onepass_cloud

Tiers per pair of clips from different sources (segments of the same listing are never compared):
  pdq     confirmed copy: PDQ hits (<= 31 bits) agree on one time shift (>= 3 votes on 4/s frames)
  sscd    SSCD-only candidate (cosine >= 0.75, >= 3 votes on 2/s frames), split by alignment sharpness
Writes pairs to <dir>/audit_pairs.csv and prints totals by kind (same id two listings / different ids /
cross-site). No PyTorch here (FAISS only).
"""

import csv
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np

PDQ_RADIUS = 31
SSCD_THRESHOLD = 0.75
PDQ_K, SSCD_K = 100, 400


def source_key(name: str) -> tuple[str, str, str]:
    """(site, source id, listing). Segments of one listing are parts of one video."""
    if m := re.match(r"^(pixabay|pexels)_(.+)_(\d+)_\d+\.mp4$", name):
        return m.group(1), m.group(3), m.group(2)
    if m := re.match(r"^mixkit_([^_]+)_mixkit-.*-(\d+)_\d+\.mp4$", name):
        return "mixkit", m.group(2), m.group(1)
    return "other", name, ""


def kind(a, b) -> str:
    """Same numeric ID = one source video listed twice, even under different site prefixes: OpenVid files
    some Pexels videos under "pixabay_..." names, so a matching ID is a labelling duplicate, not two sites."""
    if a[1] == b[1]:
        return "same id, listed twice"
    return "different ids, different site label" if a[0] != b[0] else "different ids"


def load(root: Path):
    names, pdq, pdq_owner, pdq_off, qual, sscd, s_owner, s_off = [], [], [], [], [], [], [], []
    for f in sorted(root.rglob("shard_*.npz")):
        z = np.load(f)
        base = len(names)
        names += [str(n) for n in z["names"]]
        pdq.append(z["pdq"]); qual.append(z["quality"])
        pdq_owner.append(z["owner"] + base); pdq_off.append(z["offset"])
        sscd.append(z["sscd"]); s_owner.append(z["sscd_owner"] + base); s_off.append(z["sscd_offset"])
    # The same clip can appear in two shard sets after a resumed run: keep first occurrence.
    return (names, np.concatenate(pdq), np.concatenate(pdq_owner), np.concatenate(pdq_off), np.concatenate(qual),
            np.concatenate(sscd).astype(np.float32), np.concatenate(s_owner), np.concatenate(s_off))


def vote(q_owner, q_off, h_owner, h_off, min_votes: int) -> dict[tuple[int, int], int]:
    """Per (clip, other clip): distinct query frames agreeing on a 1 s shift bin, plus its ±1 neighbours."""
    if len(q_owner) == 0:
        return {}
    b = np.floor(h_off - q_off + 0.5).astype(np.int64)
    t = np.round(q_off * 4).astype(np.int64)
    keys = np.unique(np.stack([q_owner, h_owner, b, t], 1), axis=0)          # one vote per query frame per bin
    k3, counts = np.unique(keys[:, :3], axis=0, return_counts=True)
    lookup = {tuple(k): c for k, c in zip(map(tuple, k3), counts)}
    best: dict[tuple[int, int], int] = {}
    for (a, o, s), c in lookup.items():
        n = c + lookup.get((a, o, s - 1), 0) + lookup.get((a, o, s + 1), 0)
        if n >= min_votes and n > best.get((a, o), 0):
            best[(a, o)] = n
    return best


def search_batched(index, queries, k, keep, batch=200_000):
    """Yield (query rows, hit rows, scores) for hits passing `keep(scores)`, batch by batch."""
    for i in range(0, len(queries), batch):
        D, I = index.search(queries[i:i + batch], k)
        r, c = np.nonzero(keep(D) & (I >= 0))
        yield r + i, I[r, c], D[r, c]
        print(f"    searched {min(i + batch, len(queries)):,}/{len(queries):,}", flush=True)


def run_part(root: Path, part: int, of: int, out: Path) -> None:
    """Search with the frames of clips where clip_id % of == part (against ALL frames); write that slice's pairs."""
    names, pdq, p_owner, p_off, qual, sscd, s_owner, s_off = load(root)
    first = {}
    for i, n in enumerate(names):
        first.setdefault(n, i)
    keep_clip = np.array([first[n] == i for i, n in enumerate(names)])
    src = [source_key(n) for n in names]
    gid = {k: i for i, k in enumerate(sorted(set(src)))}
    group = np.array([gid[k] for k in src])
    mine = (np.arange(len(names)) % of) == part
    print(f"part {part}/{of}: {keep_clip.sum():,} clips, {len(pdq):,} PDQ frames, {len(sscd):,} SSCD frames", flush=True)
    faiss.omp_set_num_threads(max(1, os.cpu_count() or 1))

    # PDQ: index every frame with quality >= 50; query with this slice's frames.
    pidx = np.nonzero(keep_clip[p_owner] & (qual >= 50))[0]
    index = faiss.IndexBinaryFlat(256)
    index.add(np.ascontiguousarray(pdq[pidx]))
    qrows = pidx[mine[p_owner[pidx]]]
    qs, hs = [], []
    for r, hh, _ in search_batched(index, np.ascontiguousarray(pdq[qrows]), PDQ_K, lambda D: D <= PDQ_RADIUS):
        q, h = qrows[r], pidx[hh]
        ok = (p_owner[q] != p_owner[h]) & (group[p_owner[q]] != group[p_owner[h]])
        qs.append(q[ok]); hs.append(h[ok])
    q, h = np.concatenate(qs), np.concatenate(hs)
    pdq_pairs = vote(p_owner[q], p_off[q], p_owner[h], p_off[h], min_votes=3)
    print(f"PDQ: {len(pdq_pairs):,} directed pairs", flush=True)

    # SSCD: approximate index (same seeded training on every machine), query with this slice.
    sidx = np.nonzero(keep_clip[s_owner])[0]
    vecs = np.ascontiguousarray(sscd[sidx])
    nlist = int(4 * math.sqrt(len(vecs)))
    quant = faiss.IndexFlatIP(vecs.shape[1])
    sindex = faiss.IndexIVFFlat(quant, vecs.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
    sindex.train(vecs[np.random.default_rng(0).choice(len(vecs), min(len(vecs), 50 * nlist), replace=False)])
    sindex.add(vecs)
    sindex.nprobe = 32
    srows = np.nonzero(mine[s_owner[sidx]])[0]
    qs, hs = [], []
    for r, hh, _ in search_batched(sindex, np.ascontiguousarray(vecs[srows]), SSCD_K,
                                   lambda D: D >= SSCD_THRESHOLD, batch=50_000):
        q, h = sidx[srows[r]], sidx[hh]
        ok = (s_owner[q] != s_owner[h]) & (group[s_owner[q]] != group[s_owner[h]])
        qs.append(q[ok]); hs.append(h[ok])
    q, h = np.concatenate(qs), np.concatenate(hs)
    sscd_pairs = vote(s_owner[q], s_off[q], s_owner[h], s_off[h], min_votes=3)
    print(f"SSCD: {len(sscd_pairs):,} directed pairs", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"pairs_{part:03d}.csv", "w", newline="") as f:
        w = csv.writer(f)
        for tier, pairs in (("pdq", pdq_pairs), ("sscd", sscd_pairs)):
            for (a, b), v in pairs.items():
                w.writerow([names[a], names[b], tier, v])


def combine(root: Path, frags: Path) -> str:
    """Merge every slice's pairs: undirected, PDQ wins over SSCD, max votes. Write audit_pairs.csv + summary."""
    names = []
    for f in sorted(root.rglob("shard_*.npz")):
        names += [str(n) for n in np.load(f)["names"]]
    n_clips = len(set(names))
    pairs: dict[tuple[str, str], tuple[str, int]] = {}
    for f in sorted(frags.glob("pairs_*.csv")):
        for a, b, tier, v in csv.reader(open(f)):
            k, v = tuple(sorted((a, b))), int(v)
            old = pairs.get(k)
            if old is None or (tier == "pdq" and old[0] == "sscd") or (tier == old[0] and v > old[1]):
                pairs[k] = (tier, v)
    totals = defaultdict(lambda: defaultdict(int))
    dup_clips = defaultdict(set)
    with open(frags / "audit_pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_a", "clip_b", "tier", "votes", "kind"])
        for (a, b), (tier, v) in pairs.items():
            k = kind(source_key(a), source_key(b))
            totals[tier][k] += 1
            dup_clips[tier].update((a, b))
            w.writerow([a, b, tier, v, k])
    lines = [f"{n_clips:,} clips"]
    for tier in ("pdq", "sscd"):
        lines.append(f"{tier.upper()} pairs by kind: {dict(totals[tier])}")
        lines.append(f"  clips involved: {len(dup_clips[tier]):,} ({len(dup_clips[tier]) / n_clips:.1%} of clips)")
    either = dup_clips["pdq"] | dup_clips["sscd"]
    lines.append(f"clips with any match: {len(either):,} ({len(either) / n_clips:.1%})")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--combine", action="store_true")
    a = ap.parse_args()
    out = a.out or a.root / "audit"
    if a.combine:
        print(combine(a.root, out))
    else:
        run_part(a.root, a.part, a.of, out)
        if a.of == 1:
            print(combine(a.root, out))
