"""Build the lookup metadata for the OpenVid-1M index (no re-fingerprinting).

    uv run --project provena-audit --with pyarrow python scripts/openvid_metadata.py data/index/openvid-1m

Reads the fingerprint shards in <dir>/send (in the order provena-audit loads them) and OpenVid's own
OpenVid-1M.csv, and writes:
  clips.parquet    one row per clip, in shard load order (row i = video i): id, zip, source_platform, source_id,
                   source_url, segment, duration
  captions.sqlite  table clips(id PRIMARY KEY, caption, aesthetic, motion, temporal_consistency, camera_motion,
                   frames, fps, seconds), from OpenVid-1M.csv
  manifest.json    dataset-level facts
"""

import csv
import io
import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from provena_audit.storage import Location

REVISION = "d8a63bd22989c80b5734ec2bb989f4e1b61a5807"  # Hugging Face commit the zips and CSV came from


def source(name: str) -> tuple[str, str, str, str]:
    """(platform, source id, source url, segment) from an OpenVid clip name."""
    if m := re.match(r"^pixabay_.+_(\d+)_(\d+)\.mp4$", name):
        return "pixabay", m[1], f"https://pixabay.com/videos/id-{m[1]}/", m[2]
    if m := re.match(r"^pexels_.+_(\d+)_(\d+)\.mp4$", name):
        return "pexels", m[1], f"https://www.pexels.com/video/{m[1]}/", m[2]
    if m := re.match(r"^mixkit_[^_]+_(mixkit-.*-(\d+))_(\d+)\.mp4$", name):
        return "mixkit", m[2], f"https://mixkit.co/free-stock-video/{m[1]}/", m[3]
    if name.startswith("celebv_"):  # celebv_<11-char YouTube ID>_<n>.mp4
        yt = name[7:18]
        return "celebv", yt, f"https://www.youtube.com/watch?v={yt}", name[19:].removesuffix(".mp4")
    yt = name[:11]  # <11-char YouTube ID>_<n>_<start>to<end>.mp4; IDs may contain "_"
    return "youtube", yt, f"https://www.youtube.com/watch?v={yt}", name[12:].removesuffix(".mp4")


def main(out: Path) -> None:
    send = Location(out / "send")
    ids, zips, durations = [], [], []
    for f in send.files({".npz"}):  # same order as shards.load
        z = np.load(io.BytesIO(send.read_bytes(f)))
        tag = re.match(r"shard_(.+)_\d+\.npz$", f)[1]
        ids += [str(i) for i in z["ids"]]
        zips += [tag] * len(z["ids"])
        durations.append(z["durations"])
    src = [source(i) for i in ids]
    table = pa.table({
        "id": ids, "zip": zips,
        "source_platform": [s[0] for s in src], "source_id": [s[1] for s in src],
        "source_url": [s[2] for s in src], "segment": [s[3] for s in src],
        "duration": np.concatenate(durations).astype(np.float32),
    })
    pq.write_table(table, out / "clips.parquet", compression="zstd")

    db = sqlite3.connect(out / "captions.sqlite")
    db.execute("DROP TABLE IF EXISTS clips")
    db.execute("CREATE TABLE clips (id TEXT PRIMARY KEY, caption TEXT, aesthetic REAL, motion REAL, "
               "temporal_consistency REAL, camera_motion TEXT, frames INTEGER, fps REAL, seconds REAL)")
    with open(out / "OpenVid-1M.csv", newline="") as f:
        rows = csv.reader(f)
        next(rows)
        db.executemany("INSERT OR REPLACE INTO clips VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    db.commit()
    n_caps = db.execute("SELECT COUNT(*) FROM clips").fetchone()[0]
    db.close()

    platforms = {p: int(n) for p, n in zip(*np.unique([s[0] for s in src], return_counts=True))}
    (out / "manifest.json").write_text(json.dumps({
        "dataset": "OpenVid-1M",
        "source": "https://huggingface.co/datasets/nkp37/OpenVid-1M",
        "revision": REVISION,
        "licence_stated": "CC BY 4.0",
        "clips": len(ids),
        "hours": round(float(table["duration"].to_numpy().sum()) / 3600, 1),
        "platforms": platforms,
        "fingerprinted": "2026-10-11",
        "fingerprint": "pdq, 4 frames/s, provena-audit 0.1.0",
        "zip_note": "zip is the Hugging Face file name without .zip; zips uploaded in pieces are <zip>_partaa, _partab",
    }, indent=1))
    print(f"{len(ids):,} clips -> clips.parquet; {n_caps:,} captions -> captions.sqlite; platforms {platforms}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
