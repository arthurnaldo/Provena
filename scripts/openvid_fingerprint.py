"""Fingerprint OpenVid-1M zips without waiting on downloads: each zip is fetched as parallel range requests and
read in order, and clips are fingerprinted by worker processes while the download keeps going. Writes
provena-audit shards (clip names as IDs).

    uv run --project provena-audit --with requests --with stream-unzip \
        python scripts/openvid_fingerprint.py OpenVid_part1.zip --out data/openvid_send [--limit 200]

A name without .zip (e.g. OpenVid_part102) is a zip uploaded in pieces (_partaa, _partab, ...), read back to back.
Resumable: clips already in this zip's output shards are skipped. Prints where the time goes (download vs workers).
"""

import argparse
import io
import itertools
import queue
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from stream_unzip import stream_unzip

from provena_audit import shards
from provena_audit.fingerprint import fingerprint_file
from provena_audit.storage import Location

BASE = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/"


def fetch(url: str, start: int, end: int) -> bytes:
    """Bytes start..end (inclusive), retrying dropped connections."""
    for failures in range(20):
        try:
            r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            print(f"  {e.__class__.__name__} at {start / 1e9:.1f} GB, retrying", flush=True)
            time.sleep(min(60, 2 ** failures))
    raise RuntimeError(f"gave up on bytes {start}-{end}")


def size_of(url: str) -> int | None:
    r = requests.head(url, allow_redirects=True, timeout=60)
    return int(r.headers["Content-Length"]) if r.ok else None


def zip_urls(name: str) -> list[tuple[str, int]]:
    """(url, size) of a zip, or of its pieces in order when it was uploaded split (name without .zip)."""
    if name.endswith(".zip"):
        return [(BASE + name, size_of(BASE + name))]
    out = []
    for suffix in ("aa", "ab", "ac", "ad", "ae", "af"):
        if (size := size_of(f"{BASE}{name}_part{suffix}")) is None:
            break
        out.append((f"{BASE}{name}_part{suffix}", size))
    if not out:
        raise FileNotFoundError(name)
    return out


def ranged_bytes(urls: list[tuple[str, int]], stats: dict, connections: int, chunk: int = 64 << 20):
    """The files' bytes in order, back to back, `connections` 64 MB range requests in flight at once."""
    for url, size in urls:
        starts = iter(range(0, size, chunk))
        with ThreadPoolExecutor(connections) as pool:
            ahead = [pool.submit(fetch, url, s, min(s + chunk, size) - 1) for s in itertools.islice(starts, connections)]
            while ahead:
                data = ahead.pop(0).result()
                if (s := next(starts, None)) is not None:
                    ahead.append(pool.submit(fetch, url, s, min(s + chunk, size) - 1))
                stats["bytes"] += len(data)
                yield data


def work(path: str) -> dict | str:
    try:
        return fingerprint_file(path)
    except Exception as e:
        return f"error: {e}"
    finally:
        Path(path).unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("--out", default="data/openvid_send")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--connections", type=int, default=8, help="parallel download requests")
    ap.add_argument("--limit", type=int, default=0, help="stop after N clips (timing runs)")
    ap.add_argument("--shard-size", type=int, default=1000)
    args = ap.parse_args()

    out, tag = Location(args.out), args.zip.removesuffix(".zip")
    done = {str(i) for f in out.files({".npz"}) if f.startswith(f"shard_{tag}_")
            for i in np.load(io.BytesIO(out.read_bytes(f)))["ids"]}
    shard_no, batch, written, errors = shards.shard_count(out, tag), [], 0, 0
    stats = {"bytes": 0, "wait": 0.0, "error": None}
    results: queue.Queue = queue.Queue()
    slots = threading.BoundedSemaphore(64)  # clips on disk waiting for a worker
    stop = threading.Event()
    tmp = Path(tempfile.mkdtemp(prefix="openvid_"))
    start = time.time()

    def download(pool):
        try:
            for raw, _size, chunks in stream_unzip(ranged_bytes(zip_urls(args.zip), stats, args.connections)):
                name = raw.decode().rsplit("/", 1)[-1]
                if stop.is_set() or not name.endswith(".mp4") or name in done:
                    for _ in chunks:
                        pass
                    if stop.is_set():
                        break
                    continue
                path = tmp / name
                with path.open("wb") as f:
                    for c in chunks:
                        f.write(c)
                t = time.time()
                slots.acquire()  # time spent here = workers are the bottleneck, not the download
                stats["wait"] += time.time() - t
                results.put((name, pool.submit(work, str(path))))
        except Exception as e:  # finish what was downloaded, then fail so the zip is retried
            stats["error"] = e
        finally:
            results.put(None)

    def flush():
        nonlocal batch, shard_no
        if batch:
            shards.write_shard(out, tag, shard_no, batch)
            shard_no, batch = shard_no + 1, []

    with ProcessPoolExecutor(args.workers) as pool:
        threading.Thread(target=download, args=(pool,), daemon=True).start()
        while (item := results.get()) is not None:
            name, fut = item
            v = fut.result()
            slots.release()
            if isinstance(v, str):
                errors += 1
                continue
            v["id"] = name
            batch.append(v)
            written += 1
            if len(batch) >= args.shard_size:
                flush()
            if written % 200 == 0:
                el = time.time() - start
                print(f"  {written:,} clips, {written / el:.1f}/s, {stats['bytes'] / el / 1e6:.0f} MB/s download, "
                      f"download blocked on workers {stats['wait'] / el:.0%} of the time, {errors} errors", flush=True)
            if args.limit and written >= args.limit:
                stop.set()
        flush()
    if stats["error"] is not None:
        raise RuntimeError(f"download failed after {written:,} clips (rerun resumes): {stats['error']}")
    el = time.time() - start
    print(f"done: {written:,} clips in {el / 60:.1f} min ({written / el:.1f}/s), {stats['bytes'] / 1e9:.2f} GB at "
          f"{stats['bytes'] / el / 1e6:.0f} MB/s, download blocked on workers {stats['wait'] / el:.0%}, {errors} errors")


if __name__ == "__main__":
    main()
