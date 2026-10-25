import json
import subprocess

import pytest
from provena_audit.audit import Index

from gate import db, engine, findings

SOURCES = {  # distinct synthetic footage, 8 s each
    "x": "testsrc2=size=640x360:rate=30",
    "y": "mandelbrot=size=640x360:rate=30",
    "z": "sierpinski=size=640x360:rate=30",
    "w": "life=size=640x360:rate=30:mold=10:ratio=0.5:death_color=#C83232:life_color=#00ff00",
    "v": "cellauto=size=640x360:rate=30:rule=110",
    "still": "smptehdbars=size=640x360:rate=30",
}


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *map(str, args)], check=True)


@pytest.fixture(scope="module")
def videos(tmp_path_factory):
    d = tmp_path_factory.mktemp("videos")
    for name, src in SOURCES.items():
        ffmpeg("-f", "lavfi", "-i", src, "-t", 8, "-pix_fmt", "yuv420p", d / f"{name}.mp4")
    ffmpeg("-ss", 1, "-i", d / "x.mp4", "-t", 6, "-vf", "scale=320:-2", "-crf", 35, d / "x_edited.mp4")
    ffmpeg("-i", d / "w.mp4", "-crf", 30, d / "w_again.mp4")
    return d


def setup(tmp_path, videos, b_shares):
    con = db.connect(tmp_path / "gate.sqlite")
    a = db.create_account(con, "acme", share_fingerprints=True)
    b = db.create_account(con, "orbit", share_fingerprints=b_shares)
    engine.add_to_library(con, tmp_path, a, [videos / "x.mp4", videos / "still.mp4"])
    engine.add_to_library(con, tmp_path, b, [videos / "y.mp4"])
    public = tmp_path / "public"
    public.mkdir()
    (public / "manifest.json").write_text(json.dumps({"dataset": "TestSet", "licence_stated": "cc-by-4.0"}))
    engine.fingerprint_into(public, "pub", [videos / "z.mp4"])
    return con, a, Index(str(public))


def test_check_delivery(tmp_path, videos):
    con, a, public = setup(tmp_path, videos, b_shares=True)
    names = ["x_edited", "y", "z", "w", "w_again", "still", "v"]
    r = findings.check_delivery(con, tmp_path, public, a, "Northlight", 150.0, 15.0,
                                [(videos / f"{n}.mp4", "maria" if n < "w" else "dev") for n in names] +
                                [(tmp_path / "missing.mp4", "dev")])
    kinds = {}
    for f in r["findings"]:
        kinds.setdefault(f["clip_name"], set()).add(f["kind"])
    assert kinds == {"x_edited.mp4": {"in_own_library"}, "y.mp4": {"sold_to_other_buyer"},
                     "z.mp4": {"public_dataset"}, "w_again.mp4": {"repeat_in_delivery"}, "still.mp4": {"in_own_library"}}
    by = {f["clip_name"]: f for f in r["findings"]}
    assert abs(by["x_edited.mp4"]["match_start"] - 1) < 0.5 and by["x_edited.mp4"]["match_name"] == "x.mp4"
    assert by["y.mp4"]["source"] == "another buyer" and by["y.mp4"]["match_ref"] is None and "match_start" not in by["y.mp4"]
    assert by["z.mp4"]["source"] == "TestSet" and by["z.mp4"]["licence"] == "cc-by-4.0"
    assert by["w_again.mp4"]["match_name"] == "w.mp4"
    assert by["still.mp4"]["static"] and not by["x_edited.mp4"]["static"]
    assert r["delivery"]["status"] == "checked" and r["delivery"]["clips"] == 7
    assert list(r["errors"]) == [str(tmp_path / "missing.mp4")]
    assert r["summary"]["flagged_clips"] == 5 and r["summary"]["static_findings"] == 1
    assert r["summary"]["flagged_amount"] == pytest.approx(r["summary"]["flagged_seconds"] / 3600 * 150, abs=0.01)
    assert findings.report(con, r["delivery"]["id"]) == r


def test_no_sharing(tmp_path, videos):
    con, a, public = setup(tmp_path, videos, b_shares=False)
    r = findings.check_delivery(con, tmp_path, None, a, "Northlight", 150.0, 1.0, [(videos / "y.mp4", "dev")])
    assert r["findings"] == []
