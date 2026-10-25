"""Side-by-side frames for a finding: three moments of the delivered clip next to the same moments in the match.

Our own clips (delivery, library) use the thumbnails saved when they were added. Another buyer's frames are never
shown. Public dataset frames are fetched from Hugging Face the first time they are asked for, then cached.
"""

import base64
import json
import os
import sqlite3
import subprocess
import tempfile
from functools import cache
from pathlib import Path

from remotezip import RemoteZip

from gate import engine
from gate.findings import report

ZIP_INDEX = Path(__file__).resolve().parents[2] / "data" / "eval" / "openvid" / "zip_index.json"  # OpenVid id -> zip, member
VIDEO = (".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v")


def _data_uri(path: Path) -> str | None:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode() if path.exists() else None


def _thumb(root: Path, clip_id: str, t: float) -> str | None:
    """The saved thumbnail nearest second t (clamped to the clip's last one)."""
    d = engine.thumbs_dir(root, clip_id)
    last = max((int(p.stem) for p in d.glob("*.jpg")), default=0)
    return _data_uri(d / f"{min(max(1, round(t) + 1), last):05d}.jpg") if last else None


@cache
def _openvid_members() -> dict:
    return json.loads(ZIP_INDEX.read_text()) if ZIP_INDEX.exists() else {}


def _public_frames(f: dict, times: list[float], out: Path) -> list[Path] | str:
    """Frames of a public dataset clip at the given seconds, or why they are not available."""
    if "dataset_url" not in f or "/datasets/" not in f["dataset_url"]:
        return "no download location for this dataset"
    repo = f["dataset_url"].split("/datasets/", 1)[1].strip("/")
    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/"
    file, _, member = f["match_ref"].partition("::")
    if f.get("zip"):  # OpenVid: clips live inside large zips
        file, member = _openvid_members().get(f["match_ref"], (None, None))
        if file is None:
            return "this clip's place in the dataset's zip files is not known"
    with tempfile.TemporaryDirectory() as tmp:
        if member and file.endswith(".zip"):
            src = Path(tmp) / "clip.mp4"
            with RemoteZip(url + file, headers=headers) as z:
                src.write_bytes(z.read(member))
            src = str(src)
        elif not member and file.lower().endswith(VIDEO):
            src = url + file  # ffmpeg seeks inside the remote file with range requests
        else:
            return "frames are not available for clips stored in this archive format"
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, t in enumerate(times):
            p = out / f"{i}.jpg"
            auth = ["-headers", f"Authorization: {headers['Authorization']}\r\n"] if headers and src.startswith("http") else []
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *auth, "-ss", f"{t:.2f}", "-i", src, "-frames:v", "1",
                            "-vf", "scale=240:-2", "-q:v", "5", str(p)], check=False, timeout=120)
            paths.append(p)
    return paths if any(p.exists() for p in paths) else "the dataset clip could not be read"


def evidence(con: sqlite3.Connection, root: str | Path, delivery_id: int, finding_id: int) -> dict | None:
    root = Path(root)
    f = next((x for x in report(con, delivery_id)["findings"] if x["id"] == finding_id), None)
    if f is None:
        return None
    start, end = f["clip_start"], f["clip_end"]
    times = [start + (end - start) * (i + 0.5) / 3 for i in range(3)]
    clip = [{"t": round(t, 1), "src": _thumb(root, f["clip_id"], t)} for t in times]
    shift = f.get("match_start", start) - start
    match, note = None, None
    if f["kind"] in ("repeat_in_delivery", "in_own_library"):
        match = [{"t": round(t + shift, 1), "src": _thumb(root, f["match_ref"], t + shift)} for t in times]
    elif f["kind"] == "sold_to_other_buyer":
        note = "Another buyer's footage is never shown. Only that it matched and when it was first seen are shared."
    else:
        cached = root / "deliveries" / str(delivery_id) / "evidence" / str(finding_id)
        frames = sorted(cached.glob("*.jpg")) or _public_frames(f, [t + shift for t in times], cached)
        if isinstance(frames, str):
            note = f"Dataset frames unavailable: {frames}."
        else:
            match = [{"t": round(t + shift, 1), "src": _data_uri(p)} for t, p in zip(times, frames)]
    return {"finding_id": finding_id, "clip": clip, "match": match, "note": note}
