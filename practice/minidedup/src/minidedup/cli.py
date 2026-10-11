"""minidedup: find videos in a folder that share footage. Teaching version of provena-audit.

    minidedup scan ./videos
"""
import argparse
import math
import subprocess
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pdqhash

FPS = 4            # frames per second to sample
MAX_BITS = 31      # PDQ: <= 31 of 256 bits different = same picture
MIN_VOTES = 8      # this many frames must agree on the time shift
VIDEO = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def frames(path):
    """Yield RGB frames, FPS per second, straight from ffmpeg (no temp files, low memory)."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True, check=True).stdout
    w, h = map(int, probe.strip().split(",")[:2])
    proc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf", f"fps={FPS}",
                             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    while len(buf := proc.stdout.read(w * h * 3)) == w * h * 3:
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    proc.wait()


def fingerprint(path):
    """[(seconds, 256-bit PDQ hash)] for every usable frame."""
    out = []
    for i, frame in enumerate(frames(path)):
        bits, quality = pdqhash.compute(np.ascontiguousarray(frame))
        if quality >= 50:                      # skip blank/flat frames
            out.append((i / FPS, bits.astype(bool)))
    return out


def match(a, b):
    """Do two videos share footage? Count frame matches that agree on one time shift."""
    votes = defaultdict(set)                   # shift (s) -> frames of `a` that support it
    for ta, ha in a:
        for tb, hb in b:
            if (ha != hb).sum() <= MAX_BITS:   # Hamming distance
                votes[math.floor(tb - ta + 0.5)].add(ta)
    if not votes:
        return None
    shift, frames_ = max(votes.items(), key=lambda kv: len(kv[1]))
    if len(frames_) < MIN_VOTES:
        return None
    return {"shift": shift, "votes": len(frames_), "a_from": min(frames_), "a_to": max(frames_)}


def scan(folder):
    videos = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in VIDEO)
    prints = {}
    for p in videos:
        prints[p] = fingerprint(p)
        print(f"fingerprinted {p.name}: {len(prints[p])} frames")
    found = 0
    for a, b in combinations(videos, 2):       # every pair once
        m = match(prints[a], prints[b])
        if m:
            found += 1
            print(f"MATCH {a.name} {m['a_from']:.1f}-{m['a_to']:.1f}s = {b.name} "
                  f"(shift {m['shift']:+d}s, {m['votes']} frames agree)")
    print(f"{len(videos)} videos, {found} matching pairs")


def main():
    parser = argparse.ArgumentParser(prog="minidedup")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("scan", help="find duplicate footage in a folder")
    s.add_argument("folder")
    args = parser.parse_args()
    if args.command == "scan":
        scan(args.folder)


if __name__ == "__main__":
    main()
