"""Fixed payment rules. They set what may happen to each finding; the agent can only choose within them."""

import json
import sqlite3

from gate.findings import report

STRONG_VOTES = 12     # fewer matching frames than this: a weak match, a person decides
MAX_AUTO_HOLD = 0.30  # holding more than this share of an invoice sends the whole delivery to a person

REASONS = {
    "repeat_in_delivery": "same footage as {match_name} earlier in this delivery",
    "in_own_library": "already in your library as {match_name}",
    "sold_to_other_buyer": "already sold to another buyer (first seen {first_seen})",
    "public_dataset": "footage from the public dataset {source}",
}


def allowed_actions(finding: dict) -> set[str]:
    """Static or weak matches are never held without a person."""
    return {"escalate"} if finding["static"] or finding["votes"] < STRONG_VOTES else {"hold", "escalate"}


def default_action(finding: dict) -> str:
    return "hold" if "hold" in allowed_actions(finding) else "escalate"


def check(finding: dict, action: str) -> None:
    if action not in allowed_actions(finding):
        raise ValueError(f"{action} is not allowed for finding {finding['id']} "
                         f"(allowed: {', '.join(sorted(allowed_actions(finding)))})")


def _length(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, -1.0
    for s, e in sorted(intervals):
        total += max(0.0, e - max(s, end))
        end = max(end, e)
    return total


def current_actions(con: sqlite3.Connection, delivery_id: int, findings: list[dict]) -> dict[int, str]:
    """The latest hold / escalate / release per finding from the decision log, else the default."""
    latest = {row[0]: row[1] for row in con.execute(
        "SELECT finding_id, action FROM decisions WHERE delivery_id = ? AND finding_id IS NOT NULL "
        "AND action IN ('hold', 'escalate', 'release') ORDER BY id", (delivery_id,))}
    return {f["id"]: latest.get(f["id"], default_action(f)) for f in findings}


def siblings(findings: list[dict], finding: dict) -> list[dict]:
    """Findings of the same kind on the same clip (e.g. a dataset that holds both a video and a scene cut from it):
    a dispute or a person's decision covers them together."""
    return [f for f in findings if f["clip_id"] == finding["clip_id"] and f["kind"] == finding["kind"]]


def resplit(con: sqlite3.Connection, delivery_id: int, actor: str) -> dict:
    """Recalculate and log the split after a decision changed."""
    r = report(con, delivery_id)
    totals = split(r, current_actions(con, delivery_id, r["findings"]))
    with con:
        con.execute("INSERT INTO decisions (delivery_id, actor, action, amount, reason) VALUES (?, ?, 'split', ?, ?)",
                    (delivery_id, actor, totals["approved"], json.dumps(totals)))
    return totals


def split(r: dict, actions: dict[int, str]) -> dict:
    """Invoice split for {finding id: hold | escalate | release}: each second of a clip counted once, held before
    review; released findings (a person decided to pay) count as approved."""
    rate, billed = r["delivery"]["rate_per_hour"], r["delivery"]["invoice_total"]
    held_s = review_s = 0.0
    for clip in {f["clip_id"] for f in r["findings"]}:
        fs = [f for f in r["findings"] if f["clip_id"] == clip]
        held = [(f["clip_start"], f["clip_end"]) for f in fs if actions[f["id"]] == "hold"]
        both = held + [(f["clip_start"], f["clip_end"]) for f in fs if actions[f["id"]] == "escalate"]
        held_s += _length(held)
        review_s += _length(both) - _length(held)
    overbilled = max(0.0, round(billed - r["delivery"]["seconds"] / 3600 * rate, 2))
    held = held_s / 3600 * rate + overbilled
    review = review_s / 3600 * rate
    whole = held > MAX_AUTO_HOLD * billed
    approved = 0.0 if whole else max(0.0, billed - held - review)
    if whole:
        review = billed - held
    return {"approved": round(approved, 2), "held": round(held, 2), "review": round(review, 2),
            "overbilled": round(overbilled, 2), "whole_delivery_to_human": whole}


def apply_rules(con: sqlite3.Connection, delivery_id: int) -> dict:
    """Default action per finding (hold when allowed, else escalate), the invoice split, all logged as decisions."""
    r = report(con, delivery_id)
    actions, out = {}, []
    for f in r["findings"]:
        action = default_action(f)
        why = REASONS[f["kind"]].format(**f)
        if action == "escalate":
            why += ": static footage, weak evidence" if f["static"] else f": weak match ({f['votes']} frames)"
            why += ", a person decides"
        actions[f["id"]] = action
        out.append({"finding_id": f["id"], "action": action, "amount": f["amount"], "reason": why})
    totals = split(r, actions)
    note = (f"approve ${totals['approved']:,.2f}, hold ${totals['held']:,.2f}, review ${totals['review']:,.2f}"
            + (f"; billed ${totals['overbilled']:,.2f} more than the delivered footage is worth" if totals["overbilled"] else "")
            + ("; holds exceed the automatic limit, the whole delivery goes to a person" if totals["whole_delivery_to_human"] else ""))
    with con:
        con.execute("DELETE FROM decisions WHERE delivery_id = ? AND actor = 'rules'", (delivery_id,))
        con.executemany("INSERT INTO decisions (delivery_id, finding_id, actor, action, amount, reason) "
                        "VALUES (?, ?, 'rules', ?, ?, ?)",
                        [(delivery_id, a["finding_id"], a["action"], a["amount"], a["reason"]) for a in out])
        con.execute("INSERT INTO decisions (delivery_id, actor, action, amount, reason) VALUES (?, 'rules', 'split', ?, ?)",
                    (delivery_id, totals["approved"], note + " " + json.dumps(totals)))
        con.execute("UPDATE deliveries SET status = 'ruled' WHERE id = ?", (delivery_id,))
    return {**totals, "actions": out}
