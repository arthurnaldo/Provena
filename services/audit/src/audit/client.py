"""Customer side: fingerprint every video in a folder or bucket. Runs on the customer's own machines.

- Source, send and private can each be a local folder, s3://... or gs://...
- Many machines: each runs with --part i --of N and takes the videos whose path hashes to i. Defaults come
  from the job runner (Cloud Run Jobs, AWS Batch array jobs, Kubernetes indexed jobs), see `part_from_env`.
- Resumable per part: videos already in that part's names file are skipped.
- Remote videos are copied to a temp file, fingerprinted and deleted; nothing is kept.
"""

import hashlib
import multiprocessing
import os
import secrets
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from audit import prints
from audit.storage import Location

FPS = 4.0
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".ts", ".mts"}


def part_from_env() -> tuple[int | None, int | None]:
    """(part, of) from the job runner's environment, if any."""
    index = next((os.environ[k] for k in ("PROVENA_PART", "CLOUD_RUN_TASK_INDEX", "AWS_BATCH_JOB_ARRAY_INDEX",
                                          "JOB_COMPLETION_INDEX") if k in os.environ), None)
    count = next((os.environ[k] for k in ("PROVENA_PARTS", "CLOUD_RUN_TASK_COUNT") if k in os.environ), None)
    return (int(index) if index is not None else None), (int(count) if count is not None else None)


def in_part(rel: str, part: int, of: int) -> bool:
    """Stable split: a video always lands in the same part, even if files are added later."""
    return int(hashlib.md5(rel.encode()).hexdigest(), 16) % of == part


def _duration(path: Path) -> float | None:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return None


def fingerprint_file(path: Path) -> dict:
    """Stream frames at 4/s and PDQ each one. Returns hashes, quality and duration."""
    from fingerprint import iter_frames, pdq_frame
    hashes, quality = [], []
    for frame in iter_frames(path, FPS):
        h, q = pdq_frame(frame)
        hashes.append(h)
        quality.append(q)
    if not hashes:
        raise ValueError("no frames decoded")
    return {"pdq": np.frombuffer(b"".join(hashes), np.uint8).reshape(-1, 32),
            "quality": np.array(quality, np.uint8),
            "duration": _duration(path) or len(hashes) / FPS, "fps": FPS}


def _work(source_url: str, rel: str) -> dict:
    """Worker process: get the video (temp copy if remote), fingerprint it, drop the copy."""
    src = Location(source_url)
    with src.local_file(rel) as p:
        v = fingerprint_file(p)
    v["bytes"] = src.size(rel)
    return v


def fingerprint_folder(source: str, send: str, private: str, workers: int | None = None,
                       part: int = 0, of: int = 1, shard_size: int = 200) -> None:
    src, out, priv = Location(source), Location(send), Location(private)
    tag = prints.part_tag(part)
    names = prints.read_part_names(priv, tag)
    done = {path for _, path, _ in names}
    todo = [rel for rel in src.files(VIDEO_EXTENSIONS) if in_part(rel, part, of) and rel not in done]
    print(f"part {part + 1}/{of}: {len(done):,} videos already done, {len(todo):,} to fingerprint", flush=True)
    prints.write_manifest(out, FPS)
    shard_no = prints.shard_count(out, tag)
    batch, errors, n_ok, seconds, start = [], [], 0, 0.0, time.time()

    def flush():
        nonlocal batch, shard_no
        if batch:
            prints.write_shard(out, tag, shard_no, batch)  # shard first, then names: a crash can't mark unsaved work done
            shard_no += 1
            names.extend((v["id"], v["path"], v["bytes"]) for v in batch)
            prints.write_names(priv, tag, names)
            batch = []
        if errors:
            priv.write_bytes(f"errors_{tag}.txt", "".join(f"{r}\t{e}\n" for r, e in errors).encode())

    # spawn: cloud-storage clients don't survive fork
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(_work, src.url, rel): rel for rel in todo}
        for fut in as_completed(futures):
            rel = futures[fut]
            try:
                v = fut.result()
            except Exception as e:
                errors.append((rel, str(e).replace("\n", " ")[:300]))
                continue
            v["id"], v["path"] = "v" + secrets.token_hex(5), rel
            batch.append(v)
            n_ok += 1
            seconds += v["duration"]
            if len(batch) >= shard_size:
                flush()
            if n_ok % 50 == 0:
                speed = seconds / max(time.time() - start, 1e-9)
                print(f"  {n_ok:,}/{len(todo):,} videos, {seconds / 3600:.1f} h of footage "
                      f"({speed:.0f}x real time), {len(errors)} errors", flush=True)
    flush()
    print(f"done: {n_ok:,} videos ({seconds / 3600:.1f} h), {len(errors)} errors"
          + (f" (see errors_{tag}.txt in {priv.url})" if errors else ""), flush=True)
    print(f"send: {out.url}\nkeep private: {priv.url}", file=sys.stderr)
