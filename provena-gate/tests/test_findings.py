import pytest
from conftest import world

from gate import findings


def test_check_delivery(tmp_path, videos):
    con, a, public = world(tmp_path, videos)
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
    con, a, public = world(tmp_path, videos, b_shares=False)
    r = findings.check_delivery(con, tmp_path, None, a, "Northlight", 150.0, 1.0, [(videos / "y.mp4", "dev")])
    assert r["findings"] == []
