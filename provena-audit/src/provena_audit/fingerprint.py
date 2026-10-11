"""Customer side: fingerprint every video in a folder or bucket. Runs on the customer's own machines.

- Many machines: each runs with --part i --of N and takes the videos whose path hashes to i.
- Resumable per part: videos already in that part's names file are skipped.
- Remote videos are copied to a temp file, fingerprinted and deleted; nothing is kept.
"""

import hashlib
import math
import multiprocessing
import secrets
from concurrent.futures import ProcessPoolExecutor, as_completed

import av
import numpy as np
import pdqhash

from provena_audit import shards
from provena_audit.storage import Location

FPS = 4.0
SHARD_SIZE = 200
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".ts", ".mts"}


def fingerprint_file(path) -> dict:
    """PDQ hash + quality of a full-size frame every 0.25 s, decoded in-process (PyAV) and streamed.

    Frames are picked like ffmpeg's fps filter: a frame at time t goes to slot round(t * 4), the last frame in
    a slot is used, a skipped slot repeats the previous frame and the video ends one frame after its last frame,
    so hashes match `ffmpeg -vf fps=4` exactly.
    PDQ must see the full-size frame: on a pre-shrunk frame it differs from standard PDQ by ~20 bits.
    """
    hashes, quality = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        start = float((stream.start_time or 0) * stream.time_base)
        pending, slot, last_t = None, 0, 0.0
        frame_time = 1 / float(stream.average_rate or 30)

        def emit(frame, copies):
            bits, q = pdqhash.compute(frame.to_ndarray(format="rgb24"))
            hashes.extend([np.packbits(bits.astype(np.uint8))] * copies)
            quality.extend([q] * copies)

        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            last_t = float(frame.pts * stream.time_base) - start
            s = math.floor(last_t * FPS + 0.5)
            if pending is not None and s > slot:
                emit(pending, s - slot)
            if pending is None or s >= slot:
                pending, slot = frame, max(s, slot)
        if pending is not None:  # like ffmpeg: the video ends one frame after its last frame
            end = math.floor((last_t + frame_time) * FPS + 0.5)
            emit(pending, max(end - slot, 0 if hashes else 1))
        duration = container.duration / av.time_base if container.duration else None
    if not hashes:
        raise RuntimeError("no frames decoded")
    return {"pdq": np.stack(hashes), "quality": np.array(quality, np.uint8),
            "duration": duration or len(hashes) / FPS, "fps": FPS}


def _work(source_url: str, rel: str) -> dict:
    with Location(source_url).local_file(rel) as p:
        return fingerprint_file(p)


def fingerprint_folder(source: str, send: str, private: str, part: int = 0, of: int = 1,
                       workers: int | None = None) -> None:
    src, out, priv = Location(source), Location(send), Location(private)
    tag = f"p{part:04d}"
    names = shards.read_names(priv, tag)
    done = {path for _, path in names}
    todo = [rel for rel in src.files(VIDEO_EXTENSIONS)
            if int(hashlib.md5(rel.encode()).hexdigest(), 16) % of == part and rel not in done]
    print(f"part {part + 1}/{of}: {len(done):,} videos already done, {len(todo):,} to fingerprint", flush=True)
    shard_no, batch, errors = shards.shard_count(out, tag), [], 0

    def flush():
        nonlocal batch, shard_no
        if batch:
            shards.write_shard(out, tag, shard_no, batch)  # shard first, then names: a crash can't mark unsaved work done
            shard_no += 1
            names.extend((v["id"], v["path"]) for v in batch)
            shards.write_names(priv, tag, names)
            batch = []

    # spawn: cloud-storage clients don't survive fork
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(_work, src.url, rel): rel for rel in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                v = fut.result()
            except Exception as e:
                errors += 1
                print(f"  failed: {futures[fut]}: {e}", flush=True)
                continue
            v["id"], v["path"] = "v" + secrets.token_hex(5), futures[fut]
            batch.append(v)
            if len(batch) >= SHARD_SIZE:
                flush()
            if i % 100 == 0:
                print(f"  {i:,}/{len(todo):,} videos", flush=True)
    flush()
    print(f"done: {len(todo) - errors:,} videos, {errors} failed\nsend: {out.url}\nkeep private: {priv.url}")
