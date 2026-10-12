"""Build lookup metadata (manifest.json + clips.parquet) for a Hugging Face dataset fingerprinted by hf_fingerprint.py.

    uv run --project provena-audit --with pyarrow python scripts/hf_metadata.py data/index/<repo with / as __>

Clip IDs look like "<repo file>" or "<repo file>::<member>". clips.parquet (one row per clip, shard load order):
id, file (repo file), member (inside an archive / parquet row), source_platform/source_id/source_url when the name
carries an 11-character YouTube ID, duration.
"""

import io
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from provena_audit.storage import Location

YT = re.compile(r"^(?:ytb_)?([A-Za-z0-9_-]{11})(?:-Scene-\d+)?$")  # <id>, ytb_<id>, <id>-Scene-012


def youtube_id(member: str) -> str | None:
    """The YouTube ID a clip's file name carries (e.g. 4EVVAEAMzbE.mp4, ytb_<id>.mp4, <id>-Scene-098.mp4), else None."""
    stem = member.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    m = YT.match(stem)
    return m[1] if m and re.search(r"[A-Za-z]", m[1]) else None  # all-digit names are stock IDs, not YouTube


def main(folder: Path) -> None:
    repo = folder.name.replace("__", "/", 1)
    send = Location(folder / "send")
    ids, durations = [], []
    for f in send.files({".npz"}):
        z = np.load(io.BytesIO(send.read_bytes(f)))
        ids += [str(i) for i in z["ids"]]
        durations.append(z["durations"])
    files, members, platform, sid, url = [], [], [], [], []
    for i in ids:
        file, _, member = i.partition("::")
        files.append(file)
        members.append(member)
        yt = youtube_id(member or file)
        platform.append("youtube" if yt else "")
        sid.append(yt or "")
        url.append(f"https://www.youtube.com/watch?v={yt}" if yt else "")
    dur = np.concatenate(durations).astype(np.float32)
    pq.write_table(pa.table({"id": ids, "file": files, "member": members, "source_platform": platform,
                             "source_id": sid, "source_url": url, "duration": dur}),
                   folder / "clips.parquet", compression="zstd")
    info = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}", timeout=60))
    lic = (info.get("cardData") or {}).get("license")
    (folder / "manifest.json").write_text(json.dumps({
        "dataset": repo.split("/", 1)[1],
        "source": f"https://huggingface.co/datasets/{repo}",
        "revision": info.get("sha"),
        "licence_stated": lic if isinstance(lic, str) else ("|".join(lic) if lic else "none stated"),
        "clips": len(ids),
        "hours": round(float(dur.sum()) / 3600, 1),
        "with_youtube_id": int(sum(1 for p in platform if p)),
        "fingerprinted": "2026-10-12",
        "fingerprint": "pdq, 4 frames/s, provena-audit 0.2.0",
        "complete": False,
    }, indent=1))
    print(f"{repo}: {len(ids):,} clips, {dur.sum() / 3600:,.0f} h, {sum(1 for p in platform if p):,} with a YouTube ID")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
