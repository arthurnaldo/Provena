"""Customer side: turn our report (IDs only) into evidence with real file names and side-by-side frames,
rendered from the customer's own files (local folder or bucket)."""

import json
import subprocess
from pathlib import Path

from audit import prints, report
from audit.storage import Location


def side_by_side(a: Path, a_time: float, b: Path, b_time: float, out: Path, height: int = 240) -> bool:
    proc = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{a_time:.2f}", "-i", str(a), "-ss", f"{b_time:.2f}",
         "-i", str(b), "-filter_complex",
         f"[0]scale=-2:{height},setsar=1[l];[1]scale=-2:{height},setsar=1[r];[l][r]hstack",
         "-frames:v", "1", "-q:v", "4", str(out)], capture_output=True)
    return proc.returncode == 0 and out.exists()


def build_evidence(report_json: Path, source: str, private: str, out: Path, limit: int = 100) -> Path:
    rep = json.loads(report_json.read_text())
    src = Location(source)
    names = prints.read_names(Location(private))
    out.mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(exist_ok=True)
    images, done = {}, 0
    for g in rep["groups"]:
        for pr in g["pairs"][:20]:  # the HTML shows up to 20 matches per group
            if done >= limit:
                break
            a, b = pr["a"], pr["b"]
            if a not in names or b not in names:
                continue
            img = out / "frames" / f"{a}_{b}.jpg"
            with src.local_file(names[a]) as fa, src.local_file(names[b]) as fb:
                ok = side_by_side(fa, sum(pr["a_range"]) / 2, fb, sum(pr["b_range"]) / 2, img)
            if ok:
                images[(a, b)] = f"frames/{img.name}"
                done += 1
    page = out / "evidence.html"
    page.write_text(report.render(rep, names=names, images=images))
    return page
