"""Duplicate audit of OpenVid-1M, streamed: fetch one clip, fingerprint it, delete it. Only hashes are kept.

    uv run python eval/openvid_audit.py index                 # map clip -> zip part (reads zip directories only)
    uv run python eval/openvid_audit.py ingest --sample 2000  # stream + fingerprint a sample of stock clips
    uv run python eval/openvid_audit.py stream --prefix pixabay  # whole zips, streamed (faster at scale)
    uv run python eval/openvid_audit.py dedup                 # all-vs-all near-duplicate search, report

Clips are fetched with HTTP range requests straight out of the remote zips, so nothing but the clip
itself is downloaded. Segments cut from the same source video are not counted as duplicates.
"""

import argparse
import csv
import json
import random
import re
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from core import Video
from fingerprint import fingerprint_video
from matcher import match
from store import Store

BASE = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/"
DATA = Path(__file__).resolve().parent.parent / "data" / "eval" / "openvid"
EPOCH = datetime(2024, 7, 1, tzinfo=timezone.utc)  # placeholder posted_at; ordering is irrelevant here


def source_key(name: str) -> str:
    """"site:source_id:listing". Segments of one listing are parts of one video, so never counted as duplicates.

    The same source id under a different listing (category) is kept separate: whether those are really the
    same footage is one of the things the audit checks.
    """
    if m := re.match(r"^(pixabay|pexels)_(.+)_(\d+)_\d+\.mp4$", name):
        return f"{m.group(1)}:{m.group(3)}:{m.group(2)}"
    if m := re.match(r"^mixkit_([^_]+)_mixkit-.*-(\d+)_\d+\.mp4$", name):
        return f"mixkit:{m.group(2)}:{m.group(1)}"
    if m := re.match(r"^([A-Za-z0-9_-]{11})_\d+_\d+to\d+\.mp4$", name):
        return f"youtube:{m.group(1)}:"
    return f"other:{name}:"


def pair_kind(src_a: str, src_b: str) -> str:
    site_a, id_a, _ = src_a.split(":", 2)
    site_b, id_b, _ = src_b.split(":", 2)
    if site_a != site_b:
        return "cross-site (same footage on two sites)"
    if id_a == id_b:
        return "same source id, listed under two names"
    return "different source ids (hidden duplicate)"


def list_part(part: str) -> tuple[str, list[str]]:
    from remotezip import RemoteZip
    with RemoteZip(BASE + part) as z:
        return part, [i.filename for i in z.infolist() if i.filename.endswith(".mp4")]


def cmd_index(_args) -> None:
    import urllib.request
    with urllib.request.urlopen("https://huggingface.co/api/datasets/nkp37/OpenVid-1M/tree/main") as r:
        parts = [f["path"] for f in json.load(r) if re.fullmatch(r"OpenVid_part\d+\.zip", f["path"])]
    index = {}
    with ThreadPoolExecutor(8) as pool:
        for part, names in pool.map(list_part, parts):
            for n in names:
                index[n.split("/")[-1]] = (part, n)
            print(f"{part}: {len(names):,} clips")
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "zip_index.json").write_text(json.dumps(index))
    print(f"indexed {len(index):,} clips in {len(parts)} parts (split parts skipped)")


def fetch_and_fingerprint(part: str, members: list[tuple[str, str]]) -> list[tuple[str, list]]:
    """Runs in a worker process: one zip directory read, then each clip fetched, hashed and deleted."""
    from remotezip import RemoteZip
    out = []
    with RemoteZip(BASE + part) as z, tempfile.TemporaryDirectory() as tmp:
        for name, member in members:
            path = Path(tmp) / name
            try:
                path.write_bytes(z.read(member))
                out.append((name, fingerprint_video(path)))
            except Exception as e:  # a bad clip shouldn't stop the run
                out.append((name, f"error: {e}"))
            finally:
                path.unlink(missing_ok=True)
    return out


def cmd_ingest(args) -> None:
    index = json.loads((DATA / "zip_index.json").read_text())
    stock = [n for n in index if n.startswith(("pixabay", "pexels", "mixkit"))]
    rng = random.Random(args.seed)
    names = set(rng.sample(stock, min(args.sample, len(stock))))
    if args.targeted:  # every clip of N source ids that are listed under 2+ names
        listings = defaultdict(set)
        for n in stock:
            site, sid, listing = source_key(n).split(":", 2)
            listings[(site, sid)].add(listing)
        multi = [k for k, v in listings.items() if len(v) > 1]
        chosen = set(rng.sample(multi, min(args.targeted, len(multi))))
        names |= {n for n in stock if tuple(source_key(n).split(":", 2)[:2]) in chosen}
    names = sorted(names)
    store = Store(DATA / "openvid.db")
    names = [n for n in names if not store.has_video("openvid", n)]
    by_part = defaultdict(list)
    for n in names:
        part, member = index[n]
        by_part[part].append((n, member))
    batches = [(p, ms[i:i + 25]) for p, ms in by_part.items() for i in range(0, len(ms), 25)]
    print(f"ingesting {len(names):,} clips from {len(by_part)} parts in {len(batches)} batches")

    start, done, errors = time.time(), 0, 0
    with ProcessPoolExecutor(args.workers) as pool:
        for fut in as_completed(pool.submit(fetch_and_fingerprint, p, ms) for p, ms in batches):
            for name, frames in fut.result():
                if isinstance(frames, str):
                    errors += 1
                    continue
                store.add_video(Video(platform="openvid", native_id=name, url=BASE + index[name][0],
                                      channel_id=source_key(name), posted_at=EPOCH), frames)
                done += 1
            rate = done / (time.time() - start)
            print(f"  {done:,}/{len(names):,} clips  ({rate:.1f} clips/s, {errors} errors)", flush=True)
    store.close()


def fingerprint_and_delete(path: str) -> list | str:
    try:
        return fingerprint_video(path)
    except Exception as e:
        return f"error: {e}"
    finally:
        Path(path).unlink(missing_ok=True)


def cmd_stream(args) -> None:
    """Faster for whole zips: one sequential download per part, clips hashed as they arrive.

    Per-clip range requests spend most of their time waiting; a single stream keeps the connection busy.
    Only a bounded number of clips sit on disk at once.
    """
    import requests
    from stream_unzip import stream_unzip

    index = json.loads((DATA / "zip_index.json").read_text())
    parts = args.parts or sorted({part for n, (part, _) in index.items() if n.startswith(args.prefix)})
    store = Store(DATA / "openvid.db")
    errors_log = (DATA / "errors.log").open("a")
    tmp = Path(tempfile.mkdtemp())
    start, done = time.time(), 0
    with ProcessPoolExecutor(args.workers) as pool:
        for part in parts:
            inflight = {}

            def drain(limit: int) -> None:
                nonlocal done
                while len(inflight) > limit:
                    fut = next(as_completed(inflight))
                    name = inflight.pop(fut)
                    frames = fut.result()
                    if isinstance(frames, str):
                        errors_log.write(f"{name}\t{frames}\n")
                        continue
                    store.add_video(Video(platform="openvid", native_id=name, url=BASE + part,
                                          channel_id=source_key(name), posted_at=EPOCH), frames)
                    done += 1

            with requests.get(BASE + part, stream=True, timeout=60) as r:
                r.raise_for_status()
                for raw_name, _size, chunks in stream_unzip(r.iter_content(1 << 20)):
                    name = raw_name.decode().split("/")[-1]
                    if not name.startswith(args.prefix) or store.has_video("openvid", name):
                        for _ in chunks:  # must still consume the bytes to move through the stream
                            pass
                        continue
                    path = tmp / name
                    with path.open("wb") as f:
                        for c in chunks:
                            f.write(c)
                    inflight[pool.submit(fingerprint_and_delete, str(path))] = name
                    drain(args.workers * 2)
            drain(0)
            rate = done / (time.time() - start)
            print(f"{part}: total {done:,} clips ({rate:.1f} clips/s)", flush=True)
    store.close()


def cmd_dedup(args) -> None:
    store = Store(DATA / "openvid.db")
    vids = store.db.execute("SELECT video_id, native_id, channel_id FROM videos").fetchall()
    by_id = {vid: (name, src) for vid, name, src in vids}
    pairs = {}
    for vid, name, src in vids:
        rows = store.db.execute("SELECT offset, pdq, quality FROM frames WHERE video_id = ?", (vid,)).fetchall()
        from core import Frame
        frames = [Frame(offset=o, pdq=p, quality=q) for o, p, q in rows]
        for m in match(store.search(frames), min_votes=args.min_votes):
            other_name, other_src = by_id[m.video_id]
            if m.video_id == vid or other_src == src:
                continue  # itself, or another segment of the same source video
            key = tuple(sorted((vid, m.video_id)))
            pairs[key] = max(pairs.get(key, 0), m.votes)
    clusters = defaultdict(set)
    for a, b in pairs:
        clusters[a].add(b)
        clusters[b].add(a)
    dup_clips = len(clusters)
    print(f"\n{len(vids):,} clips fingerprinted; {len(pairs):,} near-duplicate pairs across different sources;")
    print(f"{dup_clips:,} clips ({dup_clips / len(vids):.1%}) have a near-duplicate under another source ID")
    with open(DATA / "duplicate_pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_a", "source_a", "clip_b", "source_b", "votes"])
        for (a, b), v in sorted(pairs.items(), key=lambda kv: -kv[1]):
            w.writerow([by_id[a][0], by_id[a][1], by_id[b][0], by_id[b][1], v])
    print(f"pairs written to {DATA / 'duplicate_pairs.csv'}")
    kinds = defaultdict(int)
    for a, b in pairs:
        kinds[pair_kind(by_id[a][1], by_id[b][1])] += 1
    for k, n in sorted(kinds.items(), key=lambda kv: -kv[1]):
        print(f"  {n:6,} pairs  {k}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    i = sub.add_parser("ingest")
    i.add_argument("--sample", type=int, default=2000)
    i.add_argument("--targeted", type=int, default=0, help="also take all clips of N multi-listed sources")
    i.add_argument("--workers", type=int, default=24, help="downloads dominate, so more than CPU count")
    i.add_argument("--seed", type=int, default=0)
    st = sub.add_parser("stream")
    st.add_argument("parts", nargs="*", help="zip parts to stream (default: every part containing --prefix clips)")
    st.add_argument("--prefix", default="", help="only fingerprint clips whose name starts with this")
    st.add_argument("--workers", type=int, default=8)
    d = sub.add_parser("dedup")
    d.add_argument("--min-votes", type=int, default=5)
    args = p.parse_args()
    {"index": cmd_index, "ingest": cmd_ingest, "stream": cmd_stream, "dedup": cmd_dedup}[args.cmd](args)


if __name__ == "__main__":
    main()
