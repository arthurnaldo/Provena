import subprocess

from provena_audit.audit import Index

from gate import db, engine


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *map(str, args)], check=True)


def test_add_to_library(tmp_path):
    a, b, broken = tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "broken.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=10", a)
    ffmpeg("-f", "lavfi", "-i", "mandelbrot=size=640x360:rate=30", "-t", 10, b)
    broken.write_bytes(b"not a video")
    con = db.connect(tmp_path / "gate.sqlite")
    acct = db.create_account(con, "acme")

    ids, errors = engine.add_to_library(con, tmp_path, acct, [a, b, broken])
    assert len(ids) == 2 and list(errors) == [str(broken)]
    assert {r["name"] for r in con.execute("SELECT name FROM clips WHERE account_id = ?", (acct,))} == {"a.mp4", "b.mp4"}

    edited = tmp_path / "edited.mp4"  # trimmed, half size, heavily compressed copy of a
    ffmpeg("-ss", 2, "-i", a, "-t", 6, "-vf", "scale=320:-2", "-crf", 35, edited)
    hits = Index(engine.library_dir(tmp_path, acct)).search(edited)
    assert hits and hits[0]["dataset"] == f"acct:{acct}" and hits[0]["id"] == ids[0]
    assert abs(hits[0]["b_start"] - 2) < 0.5
