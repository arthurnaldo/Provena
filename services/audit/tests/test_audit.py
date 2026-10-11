"""Fingerprint a folder on two "machines", audit it, render evidence: copies found, unrelated videos not."""

import json
import subprocess

import pytest

from audit import prints, report
from audit.audit import run_audit
from audit.client import fingerprint_folder
from audit.evidence import build_evidence
from audit.storage import Location


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    d = tmp_path_factory.mktemp("library")
    (d / "a").mkdir()
    (d / "b").mkdir()
    ffmpeg("-f", "lavfi", "-i", "mandelbrot=size=320x180:rate=25", "-t", "30", "-pix_fmt", "yuv420p",
           str(d / "a" / "source.mp4"))
    ffmpeg("-f", "lavfi", "-i", "life=size=320x180:rate=25:mold=10:ratio=0.3:seed=1", "-t", "20",
           "-pix_fmt", "yuv420p", str(d / "a" / "unrelated.mp4"))
    # A re-encoded, smaller cut of seconds 8.5-20.5 of the source.
    ffmpeg("-i", str(d / "a" / "source.mp4"), "-ss", "8.5", "-to", "20.5", "-vf", "scale=iw/2:ih/2",
           "-crf", "30", "-pix_fmt", "yuv420p", str(d / "b" / "copy.mov"))
    return d


def test_audit_end_to_end(library, tmp_path):
    send, private = str(tmp_path / "send"), str(tmp_path / "private")
    for part in (0, 1):
        fingerprint_folder(str(library), send, private, workers=2, part=part, of=2)
    names = prints.read_names(Location(private))
    assert sorted(names.values()) == ["a/source.mp4", "a/unrelated.mp4", "b/copy.mov"]
    assert not any(p in json.dumps(prints.load(Location(send))["ids"].tolist()) for p in names.values())

    rep = run_audit(send)
    by_name = {v: k for k, v in names.items()}
    assert rep["summary"]["groups"] == 1
    (pair,) = rep["groups"][0]["pairs"]
    assert {pair["a"], pair["b"]} == {by_name["a/source.mp4"], by_name["b/copy.mov"]}
    src_range = pair["a_range"] if pair["a"] == by_name["a/source.mp4"] else pair["b_range"]
    assert 7.5 <= src_range[0] <= 9.5 and 19.5 <= src_range[1] <= 21.5
    dup = {v["id"]: v["duplicate_seconds"] for v in rep["groups"][0]["videos"]}
    assert dup[by_name["a/source.mp4"]] == 0  # the longer copy is never counted
    assert 10 <= dup[by_name["b/copy.mov"]] <= 12.5

    out = tmp_path / "report"
    report.write(rep, out)
    page = build_evidence(out / "report.json", str(library), private, tmp_path / "evidence")
    html = page.read_text()
    assert "b/copy.mov" in html and "frames/" in html


def test_rerun_skips_done(library, tmp_path, capsys):
    send, private = str(tmp_path / "send"), str(tmp_path / "private")
    fingerprint_folder(str(library), send, private, workers=2)
    fingerprint_folder(str(library), send, private, workers=2)
    assert "3 videos already done, 0 to fingerprint" in capsys.readouterr().out
