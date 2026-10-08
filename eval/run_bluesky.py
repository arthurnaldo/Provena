"""Eval on real Bluesky news video: ingest some accounts, query edited clips, score per edit type.

    uv run python eval/run_bluesky.py [--per-account 10]

Sources are cached in data/eval/ (gitignored). Clips from accounts that are never ingested are the
negatives: any match on them is a false positive.
"""

import argparse
import random
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

from fingerprint import QUERY_SECONDS_PER_HASH, fingerprint_video
from ingest import bluesky
from matcher import match
from store import Store

DATA = Path(__file__).resolve().parent.parent / "data" / "eval"
INDEXED = ["apnews.com", "reuters.com"]
NEGATIVE = ["nytimes.com"]
CLIP_SECONDS = 12

EDITS = {  # name → ffmpeg video filter (None = re-encode only)
    "reencode": None,
    "half_size": "scale=trunc(iw/4)*2:trunc(ih/4)*2",
    "heavy_compress": None,                      # crf 40 below
    "brightness": "eq=brightness=0.08:contrast=1.15",
    "crop_10pct": "crop=trunc(iw*0.45)*2:trunc(ih*0.45)*2",
    "logo_corner": "drawbox=x=iw*0.75:y=ih*0.03:w=iw*0.2:h=ih*0.08:color=white@0.9:t=fill",
    "caption_bar": "drawbox=y=ih*0.88:w=iw:h=ih*0.12:color=black@0.8:t=fill",
    "letterbox": "scale=trunc(iw/2)*2:trunc(ih/2)*2,pad=trunc(ih*16/18)*2:ih:(ow-iw)/2:0",
    "flip": "hflip",
}


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out)


def fetch(account: str, n: int) -> list[tuple]:
    """Download up to n video posts by account → [(Video, path)], cached."""
    out = []
    for post in bluesky.author_video_posts(account, n):
        parsed = bluesky.parse_post(post)
        if not parsed:
            continue
        video, playlist = parsed
        path = DATA / "sources" / (video.native_id.replace("/", "_").replace(":", "_") + ".mp4")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            bluesky.download(playlist, path)
        out.append((video, path))
    return out


def make_clip(src: Path, dest: Path, start: float, edit: str) -> None:
    vf = EDITS[edit]
    crf = "40" if edit == "heavy_compress" else "28"
    ffmpeg("-ss", f"{start:.3f}", "-i", str(src), "-t", str(CLIP_SECONDS), *(["-vf", vf] if vf else []),
           "-an", "-pix_fmt", "yuv420p", "-crf", crf, str(dest))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-account", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = random.Random(args.seed)
    DATA.mkdir(parents=True, exist_ok=True)

    store = Store(DATA / "bluesky.db")
    indexed = [vp for a in INDEXED for vp in fetch(a, args.per_account)]
    negatives = [vp for a in NEGATIVE for vp in fetch(a, args.per_account)]
    for video, path in indexed:
        if not store.has_video(video.platform, video.native_id):
            store.add_video(video, fingerprint_video(path))
    ids = {nid: vid for vid, nid in store.db.execute("SELECT video_id, native_id FROM videos")}
    print(f"indexed {len(indexed)} videos from {INDEXED}; negatives: {len(negatives)} from {NEGATIVE}\n")

    results = defaultdict(lambda: {"right": 0, "wrong": 0, "miss": 0, "n": 0, "shift_err": []})
    clips = DATA / "clips"
    clips.mkdir(exist_ok=True)
    for kind, pool in (("pos", indexed), ("neg", negatives)):
        for video, src in pool:
            length = duration(src)
            if length < CLIP_SECONDS + 1:
                continue
            start = round(rng.uniform(0, length - CLIP_SECONDS), 2)  # almost never on the sampling grid
            for edit in (EDITS if kind == "pos" else ["reencode"]):
                dest = clips / f"{src.stem}_{edit}_{start}.mp4"
                if not dest.exists():
                    make_clip(src, dest, start, edit)
                ms = match(store.search(fingerprint_video(dest, QUERY_SECONDS_PER_HASH)))
                top = max(ms, key=lambda m: m.votes) if ms else None
                r = results[edit if kind == "pos" else "NEGATIVE (not indexed)"]
                r["n"] += 1
                if top is None:
                    r["miss"] += 1
                elif kind == "pos" and top.video_id == ids[video.native_id]:
                    r["right"] += 1
                    r["shift_err"].append(abs(top.shift - start))
                else:
                    r["wrong"] += 1

    print(f"{'edit':<24} {'n':>3} {'right':>6} {'wrong':>6} {'miss':>5} {'recall':>7} {'median shift err':>17}")
    for edit, r in results.items():
        err = f"{statistics.median(r['shift_err']):.2f}s" if r["shift_err"] else "-"
        recall = f"{r['right'] / r['n']:.0%}" if not edit.startswith("NEG") else "-"
        print(f"{edit:<24} {r['n']:>3} {r['right']:>6} {r['wrong']:>6} {r['miss']:>5} {recall:>7} {err:>17}")


if __name__ == "__main__":
    main()
