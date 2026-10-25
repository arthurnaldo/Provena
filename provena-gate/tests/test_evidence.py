from conftest import world

from gate import evidence, findings


def test_evidence(tmp_path, videos):
    con, a, public = world(tmp_path, videos)
    r = findings.check_delivery(con, tmp_path, public, a, "Northlight", 150.0, 1.0,
                                [(videos / f"{n}.mp4", "dev") for n in ["x_edited", "y", "z", "w", "w_again"]])
    by = {f["clip_name"]: f for f in r["findings"]}
    did = r["delivery"]["id"]
    for name in ("x_edited.mp4", "w_again.mp4"):  # own footage: frames on both sides, at the matching moments
        e = evidence.evidence(con, tmp_path, did, by[name]["id"])
        assert all(x["src"] for x in e["clip"] + e["match"]) and e["note"] is None
        assert all(abs(m["t"] - c["t"] - (by[name]["match_start"] - by[name]["clip_start"])) < 0.15
                   for c, m in zip(e["clip"], e["match"]))
    e = evidence.evidence(con, tmp_path, did, by["y.mp4"]["id"])  # another buyer: never their frames
    assert e["match"] is None and "never shown" in e["note"] and all(x["src"] for x in e["clip"])
    e = evidence.evidence(con, tmp_path, did, by["z.mp4"]["id"])  # test dataset has no download location
    assert e["match"] is None and "unavailable" in e["note"]
    assert evidence.evidence(con, tmp_path, did, 999) is None
