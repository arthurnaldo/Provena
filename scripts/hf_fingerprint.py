"""Fingerprint Hugging Face video datasets: zip, tar(.gz), parquet-with-video and plain video files.

    python scripts/hf_fingerprint.py --plan plan.txt --part 3 --of 75 --out s3://bucket/index [--limit N]

plan.txt has one line per repo file: "<dataset repo>\t<file path>". This machine takes lines part, part+of, ...
Clips are streamed in, fingerprinted by worker processes and written as provena-audit shards to
<out>/<dataset slug>/send/ (clip IDs: the file path, plus "::<member>" for clips inside archives / parquet rows).
Resumable: clip IDs already in this machine's shards are skipped; finished files are marked <out>/_done/.
Gated datasets read HF_TOKEN from the environment.
"""

import argparse
import hashlib
import io
import itertools
import os
import queue
import shutil
import tarfile
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from stream_unzip import stream_unzip

from provena_audit import shards
from provena_audit.fingerprint import VIDEO_EXTENSIONS, fingerprint_file
from provena_audit.storage import Location

HEADERS = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
CHUNK = 64 << 20


def url(repo: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def fetch(u: str, start: int, end: int) -> bytes:
    for failures in range(20):
        try:
            r = requests.get(u, headers={**HEADERS, "Range": f"bytes={start}-{end}"}, timeout=120)
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            print(f"  {e.__class__.__name__} at {start / 1e9:.1f} GB, retrying", flush=True)
            time.sleep(min(60, 2 ** failures))
    raise RuntimeError(f"gave up on {u} bytes {start}-{end}")


def ranged_bytes(u: str, stats: dict, connections: int = 4):
    """A remote file's bytes in order, several 64 MB range requests in flight."""
    size = int(requests.head(u, headers=HEADERS, allow_redirects=True, timeout=60).headers["Content-Length"])
    starts = iter(range(0, size, CHUNK))
    with ThreadPoolExecutor(connections) as pool:
        ahead = [pool.submit(fetch, u, s, min(s + CHUNK, size) - 1) for s in itertools.islice(starts, connections)]
        while ahead:
            data = ahead.pop(0).result()
            if (s := next(starts, None)) is not None:
                ahead.append(pool.submit(fetch, u, s, min(s + CHUNK, size) - 1))
            stats["bytes"] += len(data)
            yield data


class Stream(io.RawIOBase):
    """File-like view of a byte generator (for tarfile's streaming mode)."""

    def __init__(self, gen):
        self.gen, self.buf = gen, b""

    def readable(self):
        return True

    def readinto(self, b):
        while not self.buf:
            self.buf = next(self.gen, b"")
            if not self.buf:
                return 0
        n = min(len(b), len(self.buf))
        b[:n], self.buf = self.buf[:n], self.buf[n:]
        return n


def is_video(name: str) -> bool:
    return Path(name).suffix.lower() in VIDEO_EXTENSIONS


def clips(repo: str, path: str, tmp: Path, stats: dict):
    """Yield (clip id, local temp file) for every video in one repo file; the caller deletes the file."""
    low = path.lower()
    if is_video(path):
        f = tmp / hashlib.md5(path.encode()).hexdigest()
        with f.open("wb") as out:
            for block in ranged_bytes(url(repo, path), stats):
                out.write(block)
        yield path, f
    elif low.endswith(".zip"):
        for raw, _size, chunks in stream_unzip(ranged_bytes(url(repo, path), stats)):
            name = raw.decode(errors="replace")
            if not is_video(name):
                for _ in chunks:
                    pass
                continue
            f = tmp / hashlib.md5(f"{path}::{name}".encode()).hexdigest()
            with f.open("wb") as out:
                for c in chunks:
                    out.write(c)
            yield f"{path}::{name}", f
    elif low.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(fileobj=io.BufferedReader(Stream(ranged_bytes(url(repo, path), stats)), 1 << 20),
                          mode="r|*") as tar:
            for m in tar:
                if not (m.isfile() and is_video(m.name)):
                    continue
                f = tmp / hashlib.md5(f"{path}::{m.name}".encode()).hexdigest()
                with f.open("wb") as out:
                    shutil.copyfileobj(tar.extractfile(m), out, 1 << 20)
                yield f"{path}::{m.name}", f
    elif low.endswith(".parquet"):
        import pyarrow.parquet as pq
        local = tmp / "file.parquet"
        with local.open("wb") as out:
            for block in ranged_bytes(url(repo, path), stats):
                out.write(block)
        pf = pq.ParquetFile(local)
        col = next((c for c in ("mp4", "video", "video_bytes") if c in pf.schema_arrow.names), None)
        if col is None:
            raise ValueError(f"{path}: no video column in {pf.schema_arrow.names}")
        row = 0
        for batch in pf.iter_batches(columns=[col], batch_size=16):
            for value in batch.column(0).to_pylist():
                if isinstance(value, dict):  # {"bytes": ..., "path": ...}
                    value = value.get("bytes")
                if value:
                    f = tmp / hashlib.md5(f"{path}::{row}".encode()).hexdigest()
                    f.write_bytes(value)
                    yield f"{path}::{row}", f
                row += 1
        local.unlink(missing_ok=True)


def work(path: str) -> dict | str:
    try:
        return fingerprint_file(path)
    except Exception as e:
        return f"error: {e}"
    finally:
        Path(path).unlink(missing_ok=True)


def slug(repo: str) -> str:
    return repo.replace("/", "__")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0, help="stop after N clips (timing runs)")
    ap.add_argument("--shard-size", type=int, default=500)
    args = ap.parse_args()

    lines = [l.split("\t") for l in Path(args.plan).read_text().splitlines() if l.strip()]
    mine = lines[args.part::args.of]
    out = Location(args.out)
    tag = f"m{args.part:04d}"
    tmp = Path(tempfile.mkdtemp(prefix="hf_"))
    stats = {"bytes": 0, "wait": 0.0}
    start, written, errors = time.time(), 0, 0
    print(f"part {args.part}/{args.of}: {len(mine)} files", flush=True)

    for repo, path in mine:
        marker = f"_done/{slug(repo)}/{hashlib.md5(path.encode()).hexdigest()}"
        if out.exists(marker):
            continue
        send = Location(f"{args.out}/{slug(repo)}/send")
        done = {str(i) for f in send.files({".npz"}) if f.startswith(f"shard_{tag}_")
                for i in np.load(io.BytesIO(send.read_bytes(f)))["ids"]}
        shard_no, batch = shards.shard_count(send, tag), []
        results: queue.Queue = queue.Queue()
        slots = threading.BoundedSemaphore(2 * (args.workers or os.cpu_count() or 1))  # clips waiting on disk
        failed, stop = {}, threading.Event()

        def download(pool):
            try:
                for cid, f in clips(repo, path, tmp, stats):
                    if stop.is_set():
                        f.unlink(missing_ok=True)
                        break
                    if cid in done:
                        f.unlink(missing_ok=True)
                        continue
                    t = time.time()
                    slots.acquire()
                    stats["wait"] += time.time() - t
                    results.put((cid, pool.submit(work, str(f))))
            except Exception as e:
                failed["error"] = e
            finally:
                results.put(None)

        try:
            with ProcessPoolExecutor(args.workers) as pool:
                threading.Thread(target=download, args=(pool,), daemon=True).start()
                while (item := results.get()) is not None:
                    cid, fut = item
                    v = fut.result()
                    slots.release()
                    if isinstance(v, str):
                        errors += 1
                        continue
                    v["id"] = cid
                    batch.append(v)
                    written += 1
                    if len(batch) >= args.shard_size:
                        shards.write_shard(send, tag, shard_no, batch)
                        shard_no, batch = shard_no + 1, []
                    if args.limit and written >= args.limit:
                        stop.set()
                    if written % 500 == 0:
                        el = time.time() - start
                        print(f"  {written:,} clips, {written / el:.1f}/s, {stats['bytes'] / el / 1e6:.0f} MB/s, "
                              f"download waited on workers {stats['wait'] / el:.0%}, {errors} errors", flush=True)
        except Exception as e:  # e.g. a worker killed for memory: save what we have, redo this file next run
            stop.set()
            failed["error"] = e
            for f in tmp.iterdir():
                f.unlink(missing_ok=True)
        if batch:
            shards.write_shard(send, tag, shard_no, batch)
        if "error" in failed:
            print(f"  {repo} {path}: download failed ({failed['error']}); will resume on rerun", flush=True)
            continue
        if not args.limit:
            out.write_bytes(marker, b"ok")
        if args.limit and written >= args.limit:
            break
    el = time.time() - start
    print(f"done: {written:,} clips in {el / 60:.1f} min, {stats['bytes'] / 1e9:.1f} GB "
          f"({stats['bytes'] / el / 1e6:.0f} MB/s), download waited on workers {stats['wait'] / el:.0%}, {errors} errors")


if __name__ == "__main__":
    main()
