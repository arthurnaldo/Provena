import pytest
from conftest import world

from gate import findings, rules


def checked(tmp_path, videos, names, value_share=1.0):
    """Check a delivery, then set its invoice to value_share x the delivered footage's value at $150/h."""
    con, a, public = world(tmp_path, videos)
    r = findings.check_delivery(con, tmp_path, public, a, "Northlight", 150.0, 0.0,
                                [(videos / f"{n}.mp4", "dev") for n in names])
    seconds = sum(c["duration"] for c in con.execute("SELECT duration FROM clips WHERE delivery_id = ?",
                                                      (r["delivery"]["id"],)))
    with con:
        con.execute("UPDATE deliveries SET invoice_total = ? WHERE id = ?",
                    (round(seconds / 3600 * 150 * value_share, 2), r["delivery"]["id"]))
    return con, findings.report(con, r["delivery"]["id"])


def test_rules(tmp_path, videos, monkeypatch):
    monkeypatch.setattr(rules, "MAX_AUTO_HOLD", 1.0)
    con, r = checked(tmp_path, videos, ["x_edited", "y", "z", "w", "w_again", "still", "v"])
    out = rules.apply_rules(con, r["delivery"]["id"])
    by = {f["clip_name"]: a["action"] for f in r["findings"] for a in out["actions"] if a["finding_id"] == f["id"]}
    assert by == {"x_edited.mp4": "hold", "y.mp4": "hold", "z.mp4": "hold", "w_again.mp4": "hold", "still.mp4": "escalate"}
    still = next(f for f in r["findings"] if f["clip_name"] == "still.mp4")
    assert out["review"] == pytest.approx(still["amount"], abs=0.01)
    assert out["approved"] + out["held"] + out["review"] == pytest.approx(r["delivery"]["invoice_total"], abs=0.02)
    assert out["overbilled"] == 0 and not out["whole_delivery_to_human"]
    rows = con.execute("SELECT actor, action FROM decisions WHERE delivery_id = ?", (r["delivery"]["id"],)).fetchall()
    assert len(rows) == 6 and {tuple(x) for x in rows} == {("rules", "hold"), ("rules", "escalate"), ("rules", "split")}
    assert con.execute("SELECT status FROM deliveries WHERE id = ?", (r["delivery"]["id"],)).fetchone()[0] == "ruled"
    with pytest.raises(ValueError):
        rules.check(still, "hold")


def test_overbilled(tmp_path, videos):
    con, r = checked(tmp_path, videos, ["v"], value_share=1.2)  # billed 20% more than the footage is worth
    out = rules.apply_rules(con, r["delivery"]["id"])
    assert out["overbilled"] == pytest.approx(r["delivery"]["invoice_total"] / 6, abs=0.01)
    assert out["held"] == out["overbilled"] and out["review"] == 0 and not out["whole_delivery_to_human"]


def test_whole_delivery_to_human(tmp_path, videos):
    con, r = checked(tmp_path, videos, ["y", "z", "v"])
    out = rules.apply_rules(con, r["delivery"]["id"])
    assert out["whole_delivery_to_human"] and out["approved"] == 0
    assert out["held"] + out["review"] == pytest.approx(r["delivery"]["invoice_total"], abs=0.01)


def test_weak_match_needs_a_person():
    assert rules.allowed_actions({"static": False, "votes": 9, "clip_start": 0, "clip_end": 8}) == {"escalate"}
    assert rules.allowed_actions({"static": False, "votes": 30, "clip_start": 0, "clip_end": 8}) == {"hold", "escalate"}
    # a 2.8 s clip with every frame agreeing is strong; the same votes over 8 s are not
    assert rules.allowed_actions({"static": False, "votes": 11, "clip_start": 0, "clip_end": 2.75}) == {"hold", "escalate"}
    assert rules.allowed_actions({"static": False, "votes": 11, "clip_start": 0, "clip_end": 8}) == {"escalate"}
