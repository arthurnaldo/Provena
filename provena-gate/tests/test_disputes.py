import json
from types import SimpleNamespace

from conftest import world

from gate import agent, findings, rules


class Replies:
    """Fake Claude client returning one structured dispute answer."""

    def __init__(self, outcome):
        self.outcome, self.beta, self.seen = outcome, SimpleNamespace(messages=self), None

    def create(self, **kwargs):
        self.seen = kwargs
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(
            type="text", text=json.dumps({"reply": "The frames show the same footage.", "outcome": self.outcome}))])


def delivery(tmp_path, videos, monkeypatch):
    monkeypatch.setattr(rules, "MAX_AUTO_HOLD", 1.0)
    con, a, public = world(tmp_path, videos)
    r = findings.check_delivery(con, tmp_path, public, a, "Northlight", 150.0, 0.0,
                                [(videos / f"{n}.mp4", "maria") for n in ["x_edited", "v"]])
    with con:
        con.execute("UPDATE deliveries SET invoice_total = ? WHERE id = ?",
                    (round(r["delivery"]["seconds"] / 3600 * 150, 2), r["delivery"]["id"]))
    rules.apply_rules(con, r["delivery"]["id"])
    return con, findings.report(con, r["delivery"]["id"])


def test_dispute_then_person_releases(tmp_path, videos, monkeypatch):
    con, r = delivery(tmp_path, videos, monkeypatch)
    did, f = r["delivery"]["id"], r["findings"][0]
    client = Replies("escalate")
    out = agent.reply_to_dispute(con, tmp_path, did, f["id"], "maria", "I filmed this myself", client)
    assert out["outcome"] == "escalate" and out["reply"] == "The frames show the same footage."
    images = [c for c in client.seen["messages"][0]["content"] if c["type"] == "image"]
    assert len(images) == 6  # Claude saw both sides' frames
    assert rules.current_actions(con, did, r["findings"])[f["id"]] == "escalate"
    held_before = rules.split(r, {f["id"]: "hold"})["held"]
    assert rules.resplit(con, did, "test")["review"] == held_before
    with con:
        con.execute("INSERT INTO decisions (delivery_id, finding_id, actor, action, reason) VALUES (?, ?, 'human', 'release', 'ok')",
                    (did, f["id"]))
    released = rules.resplit(con, did, "human")
    assert released["held"] == 0 and released["review"] == 0
    assert released["approved"] == r["delivery"]["invoice_total"]


def test_upheld_dispute_keeps_hold(tmp_path, videos, monkeypatch):
    con, r = delivery(tmp_path, videos, monkeypatch)
    did, f = r["delivery"]["id"], r["findings"][0]
    agent.reply_to_dispute(con, tmp_path, did, f["id"], "maria", "That's not my clip", Replies("uphold"))
    assert rules.current_actions(con, did, r["findings"])[f["id"]] == "hold"


def test_dispute_goes_to_a_person_when_claude_is_down(tmp_path, videos, monkeypatch):
    con, r = delivery(tmp_path, videos, monkeypatch)
    did, f = r["delivery"]["id"], r["findings"][0]

    class Down:
        @property
        def beta(self):
            raise ValueError("offline")

    out = agent.reply_to_dispute(con, tmp_path, did, f["id"], "maria", "please check", Down())
    assert out["outcome"] == "escalate" and "person" in out["reply"]
    assert rules.current_actions(con, did, r["findings"])[f["id"]] == "escalate"


def test_decision_covers_same_kind_on_same_clip(tmp_path, videos, monkeypatch):
    con, r = delivery(tmp_path, videos, monkeypatch)
    f = r["findings"][0]
    twin = {**f, "id": f["id"] + 100}
    other = {**f, "id": f["id"] + 200, "kind": "public_dataset"}
    assert [x["id"] for x in rules.siblings([f, twin, other], f)] == [f["id"], twin["id"]]
