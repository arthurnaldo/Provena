"""Ingest generated videos, query an off-grid clip, expect the right video and shift."""

import subprocess
from datetime import datetime, timezone

import pytest

from core import Video
from fingerprint import QUERY_SECONDS_PER_HASH, fingerprint_video
from matcher import match
from store import Store


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def videos(tmp_path_factory):
    d = tmp_path_factory.mktemp("videos")
    sources = {
        "mandelbrot": "mandelbrot=size=320x180:rate=25",
        "life": "life=size=320x180:rate=25:mold=10:ratio=0.3:seed=1",
    }
    for name, src in sources.items():
        ffmpeg("-f", "lavfi", "-i", src, "-t", "40", "-pix_fmt", "yuv420p", str(d / f"{name}.mp4"))
    # Off the 1 s grid, re-encoded smaller.
    ffmpeg("-i", str(d / "mandelbrot.mp4"), "-ss", "10.5", "-to", "25.5", "-vf", "scale=iw/2:ih/2",
           "-pix_fmt", "yuv420p", "-crf", "30", str(d / "clip.mp4"))
    return d


def test_query_finds_source(videos, tmp_path):
    s = Store(tmp_path / "db")
    ids = {}
    for i, name in enumerate(["mandelbrot", "life"]):
        v = Video(platform="test", native_id=name, url=name, channel_id="c",
                  posted_at=datetime(2026, 10, 1 + i, tzinfo=timezone.utc))
        ids[name] = s.add_video(v, fingerprint_video(videos / f"{name}.mp4")).video_id

    clip = fingerprint_video(videos / "clip.mp4", QUERY_SECONDS_PER_HASH)
    [m] = match(s.search(clip))
    assert m.video_id == ids["mandelbrot"]
    assert m.shift in (10, 11)
    assert m.votes >= 0.8 * len(clip)
