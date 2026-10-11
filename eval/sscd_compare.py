"""PDQ vs SSCD on the cached Bluesky eval clips (AP + Reuters indexed, NYT as negatives).

    uv run --with torch python eval/sscd_compare.py embed   # SSCD vectors -> data/eval/sscd_vectors.npz
    uv run python eval/sscd_compare.py                      # PDQ + SSCD matching, report

Two steps because PyTorch and FAISS each bundle their own OpenMP runtime and crash in one process.

Run eval/run_bluesky.py first; this reuses its cached sources and edited clips. SSCD (Meta, MIT licence)
embeds frames into 512 numbers; frames with cosine similarity >= 0.75 count as copies (Meta's DISC
operating point, ~90% precision). Both methods feed the same temporal voting.
"""

import re
import subprocess
import urllib.request
from collections import defaultdict
from pathlib import Path

import sys

import numpy as np

from core import Hit
from matcher import match

DATA = Path(__file__).resolve().parent.parent / "data" / "eval"
MODEL_URL = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"
INDEXED = {"did_plc_a67zdrt4nl2tv2qojpngogbq", "did_plc_jbvnehrrdqoulco4rf5gxg5r"}  # AP, Reuters
EDITS = ["reencode", "half_size", "heavy_compress", "brightness", "crop_10pct",
         "logo_corner", "caption_bar", "letterbox", "flip"]
SSCD_THRESHOLD = 0.75
SIZE = 320
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


VECTORS = DATA / "sscd_vectors.npz"


def load_model():
    import torch
    path = DATA / "sscd_disc_mixup.torchscript.pt"
    if not path.exists():
        urllib.request.urlretrieve(MODEL_URL, path)
    return torch.jit.load(str(path)).eval()


def frames(path: Path, fps: float) -> tuple[list[float], np.ndarray]:
    """Decode at `fps`, resized to SIZE x SIZE (the model's square-input mode)."""
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf", f"fps={fps},scale={SIZE}:{SIZE}",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    imgs = np.frombuffer(raw, np.uint8).reshape(-1, SIZE, SIZE, 3)
    return [i / fps for i in range(len(imgs))], imgs


def embed(model, imgs: np.ndarray, device: str) -> np.ndarray:
    import torch
    x = (imgs.astype(np.float32) / 255 - MEAN) / STD
    x = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
    with torch.no_grad():
        out = torch.cat([model(x[i:i + 64]) for i in range(0, len(x), 64)])
    return torch.nn.functional.normalize(out, dim=1).cpu().numpy()


def indexed_sources() -> list[Path]:
    return [s for s in sorted((DATA / "sources").glob("*.mp4")) if any(s.stem.startswith(d) for d in INDEXED)]


def cmd_embed() -> None:
    """SSCD vectors for indexed sources (2 frames/s) and every eval clip (1 frame/s)."""
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_model().to(device)
    out = {}
    for vid, src in enumerate(indexed_sources()):
        t, imgs = frames(src, 2)
        out[f"src:{src.stem}"] = embed(model, imgs, device)
    for clip in sorted((DATA / "clips").glob("*.mp4")):
        t, imgs = frames(clip, 1)
        out[f"clip:{clip.stem}"] = embed(model, imgs, device)
    np.savez(VECTORS, **out)
    print(f"embedded {len(out)} videos on {device} -> {VECTORS}")


def main() -> None:
    from datetime import datetime, timezone
    from core import Video
    from fingerprint import QUERY_SECONDS_PER_HASH, fingerprint_video
    from store import Store

    vec = np.load(VECTORS)
    indexed = indexed_sources()

    # SSCD index: every indexed source at 2 frames/s.
    vecs, owner, offs = [], [], []
    for vid, src in enumerate(indexed):
        v = vec[f"src:{src.stem}"]
        vecs.append(v)
        owner += [vid] * len(v)
        offs += [i / 2 for i in range(len(v))]
    index = np.concatenate(vecs)
    owner, offs = np.array(owner), np.array(offs)
    print(f"SSCD index: {len(indexed)} videos, {len(index):,} frames")

    # PDQ: same index as the production pipeline (4 frames/s stored, 1 frame/s queries).
    pdq = Store(":memory:")
    pdq_ids = {}
    for vid, src in enumerate(indexed):
        v = pdq.add_video(Video(platform="eval", native_id=src.stem, url="", channel_id="",
                                posted_at=datetime.now(timezone.utc)), fingerprint_video(src))
        pdq_ids[v.video_id] = vid

    clip_re = re.compile(r"^(?P<src>.+)_(?P<edit>" + "|".join(EDITS) + r")_(?P<start>[\d.]+)$")
    results = defaultdict(lambda: defaultdict(int))
    for clip in sorted((DATA / "clips").glob("*.mp4")):
        m = clip_re.match(clip.stem)
        if not m:
            continue
        src_name = m["src"]
        truth = next((i for i, s in enumerate(indexed) if s.stem == src_name), None)
        kind = m["edit"] if truth is not None else "NEGATIVE (not indexed)"

        pdq_ms = match(pdq.search(fingerprint_video(clip, QUERY_SECONDS_PER_HASH)))
        pdq_top = pdq_ids[max(pdq_ms, key=lambda x: x.votes).video_id] if pdq_ms else None

        q = vec[f"clip:{clip.stem}"]
        t = [i for i in range(len(q))]
        sims = q @ index.T
        hits = [Hit(query_offset=t[q], video_id=int(owner[j]), offset=float(offs[j]),
                    distance=int((1 - sims[q, j]) * 100))
                for q in range(len(t)) for j in np.nonzero(sims[q] >= SSCD_THRESHOLD)[0]]
        sscd_ms = match(hits)
        sscd_top = max(sscd_ms, key=lambda x: x.votes).video_id if sscd_ms else None

        r = results[kind]
        r["n"] += 1
        for name, top in (("pdq", pdq_top), ("sscd", sscd_top)):
            if top is None:
                continue
            r[f"{name}_right" if top == truth else f"{name}_wrong"] += 1
        r["either"] += (pdq_top == truth and truth is not None) or (sscd_top == truth and truth is not None)

    print(f"\n{'edit':<24} {'n':>3} {'PDQ':>6} {'SSCD':>6} {'either':>7} {'wrong (PDQ/SSCD)':>17}")
    for kind in EDITS + ["NEGATIVE (not indexed)"]:
        r = results.get(kind)
        if not r:
            continue
        pct = lambda k: f"{r[k] / r['n']:.0%}" if not kind.startswith("NEG") else "-"
        print(f"{kind:<24} {r['n']:>3} {pct('pdq_right'):>6} {pct('sscd_right'):>6} {pct('either'):>7} "
              f"{r['pdq_wrong']:>8}/{r['sscd_wrong']}")


if __name__ == "__main__":
    cmd_embed() if sys.argv[1:] == ["embed"] else main()
