"""End to end on OpenVid-1M: download 20 known duplicate pairs (10 hidden, 10 listed twice), fingerprint, audit
(whole, and split over 3 parts + combine).

Needs internet (clips are pulled from Hugging Face with range requests).
"""

import csv
from pathlib import Path

from remotezip import RemoteZip

from provena_audit import shards
from provena_audit.audit import combine, run_audit
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
    whole = run_audit(str(send), str(tmp_path / "pairs.csv"))

    # The same audit split over 3 "machines" and combined gives the same pairs and summary.
    parts = [str(tmp_path / f"pairs_{i}.csv") for i in range(3)]
    for i, path in enumerate(parts):
        run_audit(str(send), path, part=i, of=3)
    assert combine(str(send), parts, str(tmp_path / "combined.csv")) == whole
    rows = lambda p: sorted(csv.reader(open(p)))  # noqa: E731
    assert rows(tmp_path / "combined.csv") == rows(tmp_path / "pairs.csv")

    name = dict(shards.read_names(Location(private)))
    found = {frozenset((name[r["a"]], name[r["b"]])) for r in csv.DictReader(open(tmp_path / "pairs.csv"))}
    known = {frozenset((p["clip_a"], p["clip_b"])) for p in pairs}
    print(f"found {len(known & found)}/{len(known)} known pairs, {len(found - known)} others: {found - known}")
    assert len(known & found) >= 19
    assert not found - known
