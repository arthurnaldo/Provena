"""SSCD on the OpenVid clips already audited with PDQ: what does SSCD find that PDQ missed, and vice versa?

    uv run --no-project --with torch --with numpy --with remotezip python eval/openvid_sscd.py embed
    uv run python eval/openvid_sscd.py compare

Two steps because PyTorch and FAISS can't share a process (duplicate OpenMP runtimes). `embed` re-fetches
each clip (range request), takes 4 frames/s like PDQ, stores SSCD vectors in shards, deletes the clip.
It resumes from existing shards.
"""

import csv
import json
import re
import sqlite3
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "eval"
OV = DATA / "openvid"
SHARDS = OV / "sscd_shards"
BASE = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/"
FPS, SIZE, THRESHOLD = 4, 320, 0.75
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def clip_names() -> list[str]:
    return [n for (n,) in sqlite3.connect(OV / "openvid.db").execute("SELECT native_id FROM videos ORDER BY native_id")]


def fetch_frames(part: str, members: list[tuple[str, str]]) -> list[tuple[str, np.ndarray]]:
    import subprocess
    from remotezip import RemoteZip
    out = []
    for attempt in range(3):
        try:
            with RemoteZip(BASE + part) as z, tempfile.TemporaryDirectory() as tmp:
                for name, member in members:
                    path = Path(tmp) / name
                    path.write_bytes(z.read(member))
                    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf",
                                          f"fps={FPS},scale={SIZE}:{SIZE}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                         capture_output=True, check=True).stdout
                    path.unlink()
                    out.append((name, np.frombuffer(raw, np.uint8).reshape(-1, SIZE, SIZE, 3)))
            return out
        except Exception:
            out = []
    return [(name, None) for name, _ in members]


def cmd_embed() -> None:
    import torch
    sys.path.insert(0, str(Path(__file__).parent))
    from sscd_compare import load_model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_model().to(device)
    SHARDS.mkdir(parents=True, exist_ok=True)
    done = {n for f in SHARDS.glob("*.npz") for n in np.load(f)["names"]}
    index = json.loads((OV / "zip_index.json").read_text())
    todo = [n for n in clip_names() if n not in done]
    by_part = defaultdict(list)
    for n in todo:
        part, member = index[n]
        by_part[part].append((n, member))
    batches = [(p, ms[i:i + 25]) for p, ms in by_part.items() for i in range(0, len(ms), 25)]
    print(f"{len(done):,} already embedded; {len(todo):,} to go in {len(batches)} batches on {device}", flush=True)

    count, failed, shard = len(done), 0, len(list(SHARDS.glob("*.npz")))
    with ThreadPoolExecutor(16) as pool:
        for fut in as_completed(pool.submit(fetch_frames, p, ms) for p, ms in batches):
            names, owners, offsets, vecs = [], [], [], []
            for name, imgs in fut.result():
                if imgs is None or len(imgs) == 0:
                    failed += 1
                    continue
                x = torch.from_numpy(((imgs.astype(np.float32) / 255 - MEAN) / STD)).permute(0, 3, 1, 2).to(device)
                with torch.no_grad():
                    v = torch.nn.functional.normalize(model(x), dim=1).cpu().numpy().astype(np.float16)
                names.append(name)
                owners += [len(names) - 1] * len(v)
                offsets += [i / FPS for i in range(len(v))]
                vecs.append(v)
            if names:
                np.savez(SHARDS / f"shard_{shard:05d}.npz", names=np.array(names), owner=np.array(owners),
                         offset=np.array(offsets, np.float32), vec=np.concatenate(vecs))
                shard += 1
                count += len(names)
            print(f"  {count:,}/{len(done) + len(todo):,} clips embedded ({failed} failed)", flush=True)


def kind(src_a: str, src_b: str) -> str:
    sa, ia, _ = src_a.split(":", 2)
    sb, ib, _ = src_b.split(":", 2)
    return "cross-site" if sa != sb else ("same id, two listings" if ia == ib else "different ids")


def cmd_compare() -> None:
    import faiss
    sys.path.insert(0, str(Path(__file__).parent))
    from core import Hit
    from matcher import match
    from openvid_audit import source_key

    names, owner, offset, vecs = [], [], [], []
    for f in sorted(SHARDS.glob("*.npz")):
        d = np.load(f)
        owner.append(d["owner"] + len(names))
        names += list(d["names"])
        offset.append(d["offset"])
        vecs.append(d["vec"].astype(np.float32))
    owner, offset, vec = np.concatenate(owner), np.concatenate(offset), np.concatenate(vecs)
    src = [source_key(n) for n in names]
    index = faiss.IndexFlatIP(vec.shape[1])
    index.add(vec)
    print(f"SSCD index: {len(names):,} clips, {len(vec):,} frames")

    starts = np.searchsorted(owner, np.arange(len(names)))
    ends = np.searchsorted(owner, np.arange(len(names)), side="right")
    sims, ids = index.search(vec, int(__import__("os").environ.get("K", "100")))
    pairs = {}
    for c in range(len(names)):
        hits = [Hit(query_offset=float(offset[q]), video_id=int(owner[j]), offset=float(offset[j]),
                    distance=int((1 - s) * 100))
                for q in range(starts[c], ends[c]) for s, j in zip(sims[q], ids[q])
                if j >= 0 and s >= THRESHOLD and owner[j] != c and src[owner[j]] != src[c]]
        for m in match(hits):
            key = tuple(sorted((names[c], names[m.video_id])))
            pairs[key] = max(pairs.get(key, 0), m.votes)

    pdq = {tuple(sorted((r["clip_a"], r["clip_b"]))) for r in csv.DictReader(open(OV / "duplicate_pairs.csv"))}
    # Only compare PDQ pairs among clips SSCD also saw.
    seen = set(names)
    pdq = {p for p in pdq if p[0] in seen and p[1] in seen}
    sscd = set(pairs)
    both, sscd_only, pdq_only = sscd & pdq, sscd - pdq, pdq - sscd
    print(f"\npairs: PDQ {len(pdq):,}, SSCD {len(sscd):,}, both {len(both):,}, "
          f"SSCD only {len(sscd_only):,}, PDQ only {len(pdq_only):,}")
    for label, group in (("SSCD only", sscd_only), ("PDQ only", pdq_only)):
        counts = defaultdict(int)
        for a, b in group:
            counts[kind(source_key(a), source_key(b))] += 1
        print(f"  {label}: {dict(counts)}")
    with open(OV / "sscd_only_pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_a", "clip_b", "votes", "kind"])
        for a, b in sorted(sscd_only, key=lambda p: -pairs[p]):
            w.writerow([a, b, pairs[(a, b)], kind(source_key(a), source_key(b))])
    print(f"SSCD-only pairs -> {OV / 'sscd_only_pairs.csv'}")


if __name__ == "__main__":
    {"embed": cmd_embed, "compare": cmd_compare}[sys.argv[1]]()
