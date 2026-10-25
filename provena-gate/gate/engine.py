"""Glue to the matching engine (provena-audit): account libraries as fingerprint folders.

Each account's library is a dataset folder <root>/accounts/<id>/library (send/ shards + manifest.json), so the
engine's Index labels its hits "acct:<id>". Video is fingerprinted and never stored.
"""

import json
import secrets
import sqlite3
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from provena_audit import shards
from provena_audit.fingerprint import fingerprint_file
from provena_audit.storage import Location

SHARD_SIZE = 500


def library_dir(root: str | Path, account_id: int) -> Path:
    d = Path(root) / "accounts" / str(account_id) / "library"
    if not (d / "manifest.json").exists():
        (d / "send").mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(json.dumps({"dataset": f"acct:{account_id}"}))
    return d


def _fingerprint(path: str) -> dict | str:
    try:
        return fingerprint_file(path)
    except Exception as e:
        return f"error: {e}"


def delivery_dir(root: str | Path, delivery_id: int) -> Path:
    return Path(root) / "deliveries" / str(delivery_id)


def thumbs_dir(root: str | Path, clip_id: str) -> Path:
    """Evidence thumbnails of one clip: <n>.jpg is second n-1, 240 px wide. Shown only to the clip's own account."""
    return Path(root) / "thumbs" / clip_id


def _thumbnails(job: tuple[str, str]) -> None:
    path, out = job
    Path(out).mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", path, "-vf", "fps=1,scale=240:-2", "-q:v", "5",
                    f"{out}/%05d.jpg"], check=False)


def fingerprint_into(folder: Path, tag: str, paths: list[str | Path],
                     root: str | Path | None = None) -> tuple[list[dict], dict[str, str]]:
    """Fingerprint video files into new shards in folder/send, in input order, plus evidence thumbnails under
    root when given. Returns (fingerprints with id, name and path, {path: error}). Files are not deleted here:
    the caller removes uploads, and a customer's own folder stays untouched."""
    paths = [str(p) for p in paths]
    send = Location(folder / "send")
    added, errors = [], {}
    with ProcessPoolExecutor() as pool:
        for path, fp in zip(paths, pool.map(_fingerprint, paths)):
            if isinstance(fp, str):
                errors[path] = fp
                continue
            fp["id"], fp["name"], fp["path"] = secrets.token_hex(8), Path(path).name, path
            added.append(fp)
        if root is not None:
            list(pool.map(_thumbnails, [(v["path"], str(thumbs_dir(root, v["id"]))) for v in added]))
    first = shards.shard_count(send, tag)
    for i in range(0, len(added), SHARD_SIZE):
        shards.write_shard(send, tag, first + i // SHARD_SIZE, added[i:i + SHARD_SIZE])
    return added, errors


def add_to_library(con: sqlite3.Connection, root: str | Path, account_id: int,
                   paths: list[str | Path]) -> tuple[list[str], dict[str, str]]:
    """Fingerprint video files into an account's library. Returns (new clip IDs, {path: error})."""
    added, errors = fingerprint_into(library_dir(root, account_id), "lib", paths, root)
    with con:
        con.executemany("INSERT INTO clips (id, account_id, name, duration) VALUES (?, ?, ?, ?)",
                        [(v["id"], account_id, v["name"], v["duration"]) for v in added])
    return [v["id"] for v in added], errors
