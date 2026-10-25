import json
import subprocess

import pytest
from provena_audit.audit import Index

from gate import db, engine

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


@pytest.fixture(scope="session")
def videos(tmp_path_factory):
    d = tmp_path_factory.mktemp("videos")
    for name, src in SOURCES.items():
        ffmpeg("-f", "lavfi", "-i", src, "-t", 8, "-pix_fmt", "yuv420p", d / f"{name}.mp4")
    ffmpeg("-ss", 1, "-i", d / "x.mp4", "-t", 6, "-vf", "scale=320:-2", "-crf", 35, d / "x_edited.mp4")
    ffmpeg("-i", d / "w.mp4", "-crf", 30, d / "w_again.mp4")
    return d


def world(tmp_path, videos, b_shares=True):
    """Account acme (library: x, still), account orbit (library: y), a public dataset TestSet (z)."""
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
