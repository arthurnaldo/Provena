"""The fingerprint folder a customer sends us, and the private name table they keep.

send/     shard_<part>_<n>.npz: per video a random ID and duration; per frame (4/s) owner, offset,
          PDQ hash (32 bytes) and quality. No video, images, audio or file names.
private/  names_<part>.csv (id, path). Never sent; maps IDs in pairs.csv back to files.

Each machine (part) writes its own files, so many machines can share one send/ and private/ location.
"""

import csv
import io

import numpy as np

from provena_audit.storage import Location


def write_shard(send: Location, tag: str, number: int, videos: list[dict]) -> None:
    """videos: dicts with id, duration, pdq (n, 32) uint8, quality (n,) uint8, fps."""
    buf = io.BytesIO()
    np.savez(buf, ids=np.array([v["id"] for v in videos]),
             durations=np.array([v["duration"] for v in videos], np.float32),
             owner=np.concatenate([np.full(len(v["pdq"]), i, np.int32) for i, v in enumerate(videos)]),
             offset=np.concatenate([np.arange(len(v["pdq"]), dtype=np.float32) / v["fps"] for v in videos]),
             pdq=np.concatenate([v["pdq"] for v in videos]),
             quality=np.concatenate([v["quality"] for v in videos]))
    send.write_bytes(f"shard_{tag}_{number:05d}.npz", buf.getvalue())


def shard_count(send: Location, tag: str) -> int:
    return sum(1 for f in send.files({".npz"}) if f.startswith(f"shard_{tag}_"))


def load(send: Location) -> dict[str, np.ndarray]:
    """All shards from all parts as one set of arrays; `owner` indexes into `ids`."""
    ids, durations, owner, offset, pdq, quality = [], [], [], [], [], []
    for f in send.files({".npz"}):
        z = np.load(io.BytesIO(send.read_bytes(f)))
        owner.append(z["owner"].astype(np.int64) + len(ids))
        ids += [str(i) for i in z["ids"]]
        durations.append(z["durations"]); offset.append(z["offset"]); pdq.append(z["pdq"]); quality.append(z["quality"])
    if not ids:
        raise FileNotFoundError(f"no shard_*.npz in {send.url}")
    return {"ids": np.array(ids), "durations": np.concatenate(durations), "owner": np.concatenate(owner),
            "offset": np.concatenate(offset), "pdq": np.concatenate(pdq), "quality": np.concatenate(quality)}


def read_names(private: Location, tag: str | None = None) -> list[tuple[str, str]]:
    """(id, path) rows, for one part or all parts."""
    out = []
    for f in private.files({".csv"}):
        if f.startswith(f"names_{tag or ''}"):
            out += [(r["id"], r["path"]) for r in csv.DictReader(io.StringIO(private.read_bytes(f).decode()))]
    return out


def write_names(private: Location, tag: str, rows: list[tuple[str, str]]) -> None:
    """Rewrite this part's whole name table (works the same on local disk and object storage)."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "path"])
    w.writerows(rows)
    private.write_bytes(f"names_{tag}.csv", buf.getvalue().encode())
