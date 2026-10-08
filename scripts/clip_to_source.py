"""Experiment: map a clip's vPDQ frames back to source timestamps, then Hough-vote the shift.

    uv run python scripts/clip_to_source.py
    uv run python scripts/clip_to_source.py --source my_video.mp4 --clip-start 20 --clip-end 40
"""

import argparse
import csv
import math
import statistics
import subprocess
from collections import Counter
from pathlib import Path

from core import Frame
from fingerprint import QUERY_SECONDS_PER_HASH, STORED_SECONDS_PER_HASH, fingerprint_video

DATA = Path(__file__).resolve().parent.parent / "data"
MAX_DISTANCE = 31


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


def make_source(path: Path) -> None:
    """60 s synthetic footage whose content keeps changing (Mandelbrot zoom)."""
    ffmpeg("-f", "lavfi", "-i", "mandelbrot=size=640x360:rate=25", "-t", "60",
           "-pix_fmt", "yuv420p", "-c:v", "libx264", str(path))


def make_clip(source: Path, path: Path, start: float, end: float) -> None:
    """Accurate cut, re-encoded at half size so it isn't byte-identical to the source."""
    ffmpeg("-i", str(source), "-ss", str(start), "-to", str(end),
           "-vf", "scale=iw/2:ih/2", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "30", str(path))


def hamming(a: bytes, b: bytes) -> int:
    return (int.from_bytes(a) ^ int.from_bytes(b)).bit_count()


def dump(frames: list[Frame], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "quality", "pdq_hex"])
        for fr in frames:
            w.writerow([f"{fr.offset:.3f}", fr.quality, fr.pdq.hex()])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, help="source video (default: generated synthetic footage)")
    p.add_argument("--clip-start", type=float, default=20.0)
    p.add_argument("--clip-end", type=float, default=40.0)
    p.add_argument("--source-seconds-per-hash", type=float, default=STORED_SECONDS_PER_HASH, help="0 = every frame")
    p.add_argument("--clip-seconds-per-hash", type=float, default=QUERY_SECONDS_PER_HASH)
    args = p.parse_args()

    DATA.mkdir(exist_ok=True)
    source = args.source
    if source is None:
        source = DATA / "source.mp4"
        if not source.exists():
            make_source(source)
    clip = DATA / f"clip_{args.clip_start:g}-{args.clip_end:g}.mp4"
    if not clip.exists():
        make_clip(source, clip, args.clip_start, args.clip_end)

    src = fingerprint_video(source, args.source_seconds_per_hash)
    qry = fingerprint_video(clip, args.clip_seconds_per_hash)
    dump(src, DATA / "source_hashes.csv")
    dump(qry, DATA / "clip_hashes.csv")
    print(f"source: {source} → {len(src)} frames")
    print(f"clip:   {clip} → {len(qry)} frames (cut {args.clip_start:g}–{args.clip_end:g}s)\n")

    # Sanity check: source frames ~1 s apart must look different, or the test is meaningless.
    step = max(1, round(len(src) / src[-1].offset)) if src[-1].offset else 1
    adjacent = [hamming(a.pdq, b.pdq) for a, b in zip(src, src[step:])]
    print(f"median distance between source frames ~1 s apart: {statistics.median(adjacent)}"
          f" ({'OK' if statistics.median(adjacent) > MAX_DISTANCE else 'TOO SIMILAR, results not trustworthy'})\n")

    # Map each clip frame to its closest source frame.
    print(f"{'clip_t':>7} {'quality':>7} {'source_t':>9} {'distance':>8} {'shift':>6}")
    hits = []  # (clip_t, source_t) pairs within MAX_DISTANCE
    for q in qry:
        best = min(src, key=lambda s: hamming(q.pdq, s.pdq))
        d = hamming(q.pdq, best.pdq)
        print(f"{q.offset:7.2f} {q.quality:7d} {best.offset:9.2f} {d:8d} {best.offset - q.offset:6.2f}"
              + ("" if d <= MAX_DISTANCE else "  (no match)"))
        hits += [(q.offset, s.offset) for s in src if hamming(q.pdq, s.pdq) <= MAX_DISTANCE]

    # Hough vote on shift = source_t - clip_t, binned to 1 s (round half up; Python's round() is banker's).
    # A clip frame can hit several neighbouring source frames; it votes once per shift.
    votes = Counter(shift for _, shift in {(c, math.floor(s - c + 0.5)) for c, s in hits})
    print(f"\nvotes per shift (all hits ≤ {MAX_DISTANCE}): {dict(votes.most_common())}")
    if not votes:
        print("no match")
        return
    shift, n = votes.most_common(1)[0]
    support = [(c, s) for c, s in hits if math.floor(s - c + 0.5) == shift]
    print(f"winner: shift {shift}s with {n} votes")
    print(f"clip_range  = {min(c for c, _ in support):.2f}–{max(c for c, _ in support):.2f}s")
    print(f"video_range = {min(s for _, s in support):.2f}–{max(s for _, s in support):.2f}s")


if __name__ == "__main__":
    main()
