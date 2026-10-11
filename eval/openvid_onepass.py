"""One-pass OpenVid fingerprinting: each clip is downloaded once, decoded once, and hashed with PDQ + SSCD.

    uv run --no-project --with torch --with numpy --with pdqhash --with requests --with stream-unzip \
        --with-editable packages/core --with-editable packages/fingerprint \
        python eval/openvid_onepass.py --prefix pixabay,pexels,mixkit [--limit N] [--parts P1 P2]

- Several zips stream at once (threads); each clip is written to a temp file, decoded + PDQ-hashed in a
  worker process, then embedded with SSCD on the GPU in batches, and deleted.
- Output: data/eval/openvid/onepass/shard_*.npz (names, owner, offset, pdq, quality, sscd), never video.
- Resumable: clips already in shards are skipped; finished parts are listed in done_parts.txt.
- Dropped connections retry the part (already-written clips are skipped while re-streaming).
"""

import argparse
import json
import os
import queue
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
OV = DATA / "eval" / "openvid"
OUT = Path(os.environ.get("PROVENA_ONEPASS_OUT", OV / "onepass"))
MODEL = Path(os.environ.get("PROVENA_MODEL", DATA / "models" / "sscd_disc_mixup.torchscript.pt"))
BASE = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/"
FPS = 4.0


SSCD_STRIDE = 2  # SSCD on every other frame (2/s): halves GPU work; PDQ keeps all 4/s
_sscd = None


def decode_and_hash(path: str):
    """Worker process: one decode → PDQ (all frames, full size) + SSCD (every other frame). Deletes the clip.

    Each worker holds its own SSCD model, so several clips share the GPU at once.
    """
    global _sscd
    from fingerprint.frames import decode_frames
    from fingerprint.pdq import pdq_frames
    try:
        if _sscd is None:
            from fingerprint.sscd import SSCD
            _sscd = SSCD(MODEL)
        full, small = decode_frames(path, FPS)
        hashes, quality = pdq_frames(full)  # PDQ on full-size frames (standard-compatible)
        return _sscd.embed(small[::SSCD_STRIDE]), hashes, quality
    except Exception as e:
        return f"error: {e}"
    finally:
        Path(path).unlink(missing_ok=True)


def ranged_bytes(url: str, log, chunk: int = 64 << 20):
    """Yield a remote file's bytes in order using HTTP range requests.

    A dropped connection resumes from the exact byte reached, instead of restarting a 40 GB download.
    """
    import requests
    with requests.head(url, allow_redirects=True, timeout=60) as h:
        size = int(h.headers["Content-Length"])
    pos, failures = 0, 0
    while pos < size:
        end = min(pos + chunk, size) - 1
        try:
            with requests.get(url, headers={"Range": f"bytes={pos}-{end}"}, stream=True, timeout=120) as r:
                r.raise_for_status()
                for block in r.iter_content(1 << 20):
                    pos += len(block)
                    yield block
            failures = 0
        except Exception as e:
            failures += 1
            if failures > 20:
                raise
            log(f"{url.rsplit('/', 1)[-1]}: {e.__class__.__name__} at {pos / 1e9:.1f} GB, resuming")
            time.sleep(min(60, 2 ** failures))


def stream_part(part, prefixes, done, pool, items, tmp, inflight, limit_reached, log):
    """Thread: stream one zip, hand matching clips to the worker pool. Retries on connection errors."""
    import requests
    from stream_unzip import stream_unzip
    queued = set()
    for attempt in range(6):
        try:
            if True:
                for raw, _size, chunks in stream_unzip(ranged_bytes(BASE + part, log)):
                    if limit_reached.is_set():
                        items.put((part, None, "partial"))  # stop downloading; part not finished
                        return
                    name = raw.decode().split("/")[-1]
                    wanted = (name.endswith(".mp4") and name.startswith(prefixes)
                              and name not in done and name not in queued and not limit_reached.is_set())
                    if not wanted:
                        for _ in chunks:
                            pass
                        continue
                    path = tmp / name
                    with path.open("wb") as f:
                        for c in chunks:
                            f.write(c)
                    inflight.acquire()  # bound temp files / memory
                    items.put((part, name, pool.submit(decode_and_hash, str(path))))
                    queued.add(name)
            items.put((part, None, None))  # part finished
            return
        except Exception as e:
            log(f"{part}: stream error ({e.__class__.__name__}), retry {attempt + 1}")
            time.sleep(min(60, 5 * 2 ** attempt))
    log(f"{part}: giving up after retries")
    items.put((part, None, "failed"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="pixabay,pexels,mixkit")
    ap.add_argument("--parts", nargs="*")
    ap.add_argument("--streams", type=int, default=3, help="zips downloaded at once")
    ap.add_argument("--workers", type=int, default=8, help="decode + PDQ processes")
    ap.add_argument("--limit", type=int, default=0, help="stop after N clips (testing)")
    ap.add_argument("--shard-size", type=int, default=500)
    args = ap.parse_args()
    prefixes = tuple(args.prefix.split(","))

    device = "worker processes (MPS/CUDA/CPU)"

    OUT.mkdir(parents=True, exist_ok=True)
    done = {str(n) for f in OUT.glob("shard_*.npz") for n in np.load(f)["names"]}
    done_parts = set((OUT / "done_parts.txt").read_text().split()) if (OUT / "done_parts.txt").exists() else set()
    index_file = OV / "zip_index.json"
    index = json.loads(index_file.read_text()) if index_file.exists() else {}
    parts = args.parts or sorted({p for n, (p, _) in index.items() if n.startswith(prefixes)})
    parts = [p for p in parts if p not in done_parts]
    target = sum(1 for n, (p, _) in index.items() if p in parts and n.startswith(prefixes) and n not in done) or 1
    if args.limit:
        target = min(target, args.limit)
    print(f"{len(done):,} clips already done; {target:,} to go from {len(parts)} parts using {device}", flush=True)

    items: queue.Queue = queue.Queue()
    inflight = threading.BoundedSemaphore(args.workers * 4)
    limit_reached = threading.Event()
    tmp = Path(tempfile.mkdtemp(prefix="openvid_"))
    log_lock = threading.Lock()
    def log(msg):
        with log_lock:
            print(msg, flush=True)

    shard_no = len(list(OUT.glob("shard_*.npz")))
    batch, written, errors, start = [], 0, 0, time.time()
    part_pending = {p: 0 for p in parts}
    part_streamed = set()

    def flush():
        nonlocal shard_no, batch
        if not batch:
            return
        # PDQ rows: every frame (FPS/s). SSCD rows: every SSCD_STRIDE-th frame.
        owner = np.concatenate([[i] * len(b[2]) for i, b in enumerate(batch)])
        offset = np.concatenate([np.arange(len(b[2]), dtype=np.float32) / FPS for b in batch])
        s_owner = np.concatenate([[i] * len(b[1]) for i, b in enumerate(batch)])
        s_offset = np.concatenate([np.arange(len(b[1]), dtype=np.float32) * SSCD_STRIDE / FPS for b in batch])
        np.savez(OUT / f"shard_{shard_no:05d}.npz", names=np.array([b[0] for b in batch]),
                 parts=np.array([b[4] for b in batch]), owner=owner, offset=offset,
                 pdq=np.concatenate([b[2] for b in batch]), quality=np.concatenate([b[3] for b in batch]),
                 sscd=np.concatenate([b[1] for b in batch]), sscd_owner=s_owner, sscd_offset=s_offset)
        shard_no += 1
        for b in batch:
            part_pending[b[4]] -= 1
        batch = []
        for p in list(part_streamed):
            if part_pending[p] == 0:
                with (OUT / "done_parts.txt").open("a") as f:
                    f.write(p + "\n")
                part_streamed.discard(p)

    with ProcessPoolExecutor(args.workers) as pool:
        threads = []
        part_iter = iter(parts)
        def start_next():
            p = next(part_iter, None)
            if p:
                t = threading.Thread(target=stream_part, daemon=True,
                                     args=(p, prefixes, done, pool, items, tmp, inflight, limit_reached, log))
                t.start()
                threads.append(t)
            return p is not None
        active = sum(start_next() for _ in range(args.streams))

        while active:
            part, name, fut = items.get()
            if name is None:
                active -= 1
                if fut is None:  # streamed to the end
                    part_streamed.add(part)
                if not limit_reached.is_set():
                    active += start_next()
                continue
            part_pending[part] += 1
            res = fut.result()
            inflight.release()
            if isinstance(res, str) or len(res[0]) == 0:
                errors += 1
                part_pending[part] -= 1
                continue
            vec, hashes, quality = res
            batch.append((name, vec, hashes, quality, part))
            written += 1
            if len(batch) >= args.shard_size:
                flush()
                rate = written / (time.time() - start)
                log(f"  {written:,}/{target:,} clips ({rate:.1f}/s, {errors} errors, "
                    f"eta {(target - written) / max(rate, 1e-9) / 3600:.1f} h)")
            if args.limit and written >= args.limit:
                limit_reached.set()
        flush()
    log(f"done: {written:,} clips in {(time.time() - start) / 60:.1f} min, {errors} errors")


if __name__ == "__main__":
    main()
