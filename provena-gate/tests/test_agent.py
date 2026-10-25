from types import SimpleNamespace

import anthropic
import httpx2
from conftest import world

from gate import agent, findings, rules

NAMES = ["x_edited", "y", "z", "w", "w_again", "still", "v"]


class FakeClient:
    """Stands in for anthropic.Anthropic: replays scripted turns, each a list of (tool name, input)."""

    def __init__(self, turns):
        self.turns, self.beta = list(turns), SimpleNamespace(messages=self)

    def create(self, **kwargs):
        calls = self.turns.pop(0)
        return SimpleNamespace(stop_reason="tool_use", content=[
            SimpleNamespace(type="tool_use", id=f"t{i}", name=name, input=args) for i, (name, args) in enumerate(calls)])


def ruled(tmp_path, videos, monkeypatch):
    monkeypatch.setattr(rules, "MAX_AUTO_HOLD", 1.0)
    con, a, public = world(tmp_path, videos)
    r = findings.check_delivery(con, tmp_path, public, a, "Northlight", 150.0, 0.0,
                                [(videos / f"{n}.mp4", "maria" if n < "w" else "dev") for n in NAMES])
    with con:
        con.execute("UPDATE deliveries SET invoice_total = ? WHERE id = ?",
                    (round(r["delivery"]["seconds"] / 3600 * 150, 2), r["delivery"]["id"]))
    rules.apply_rules(con, r["delivery"]["id"])
    return con, findings.report(con, r["delivery"]["id"])


def test_agent_decides_within_rules(tmp_path, videos, monkeypatch):
    con, r = ruled(tmp_path, videos, monkeypatch)
    by = {f["clip_name"]: f for f in r["findings"]}
    x, still = by["x_edited.mp4"]["id"], by["still.mp4"]["id"]
    client = FakeClient([
        [("get_delivery", {}), ("get_findings", {})],
        [("get_evidence", {"finding_id": still}), ("decide", {"finding_id": still, "action": "hold", "reason": "copy"})],
        [("decide", {"finding_id": x, "action": "escalate", "reason": "check whether the trim is a licensed edit"})],
        [("write_report", {"buyer_report": "4 copies held, 2 for review.",
                           "contributor_notes": [{"contributor": "maria", "note": "x_edited.mp4 is under review."}]})],
    ])
    out = agent.run_agent(con, r["delivery"]["id"], client)
    assert out["agent"] and out["buyer_report"] == "4 copies held, 2 for review."
    actions = {a["finding_id"]: a["action"] for a in out["actions"]}
    assert actions[still] == "escalate" and actions[x] == "escalate"  # holding a static clip was refused
    assert {k: out[k] for k in ("approved", "held", "review")} == {
        k: rules.split(r, actions)[k] for k in ("approved", "held", "review")}
    log = [tuple(row) for row in con.execute(
        "SELECT action, finding_id FROM decisions WHERE delivery_id = ? AND actor = 'claude' ORDER BY id",
        (r["delivery"]["id"],))]
    assert ("refused", still) in log and ("escalate", x) in log and ("report", None) in log and ("note", None) in log
    assert con.execute("SELECT status FROM deliveries WHERE id = ?", (r["delivery"]["id"],)).fetchone()[0] == "decided"


def test_rules_stand_when_claude_is_unavailable(tmp_path, videos, monkeypatch):
    con, r = ruled(tmp_path, videos, monkeypatch)

    class Down:
        beta = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(
            anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com")))))

    out = agent.run_agent(con, r["delivery"]["id"], Down())
    assert not out["agent"] and "unavailable" in out["reason"]
    assert {a["finding_id"]: a["action"] for a in out["actions"]} == {
        f["id"]: rules.default_action(f) for f in r["findings"]}
    assert con.execute("SELECT status FROM deliveries WHERE id = ?", (r["delivery"]["id"],)).fetchone()[0] == "ruled"
