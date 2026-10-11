"""End to end on OpenVid-1M: download 20 known duplicate pairs (10 hidden, 10 listed twice), fingerprint, audit.

Needs internet (clips are pulled from Hugging Face with range requests) and ffmpeg.
"""

import csv
from pathlib import Path

from remotezip import RemoteZip

from provena_audit import shards
from provena_audit.audit import run_audit
from provena_audit.fingerprint import fingerprint_folder
from provena_audit.storage import Location

BASE = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/"


def test_openvid(tmp_path):
    pairs = list(csv.DictReader(open(Path(__file__).with_name("pairs.csv"))))
    videos = tmp_path / "videos"
    videos.mkdir()
    for p in pairs:
        for side in "ab":
            with RemoteZip(BASE + p[f"zip_{side}"]) as z:
                (videos / p[f"clip_{side}"]).write_bytes(z.read(p[f"member_{side}"]))

    send, private = tmp_path / "send", tmp_path / "private"
    fingerprint_folder(str(videos), str(send), str(private))
    run_audit(str(send), str(tmp_path / "pairs.csv"))

    name = dict(shards.read_names(Location(private)))
    found = {frozenset((name[r["a"]], name[r["b"]])) for r in csv.DictReader(open(tmp_path / "pairs.csv"))}
    known = {frozenset((p["clip_a"], p["clip_b"])) for p in pairs}
    print(f"found {len(known & found)}/{len(known)} known pairs, {len(found - known)} others: {found - known}")
    assert len(known & found) >= 19
    assert not found - known
