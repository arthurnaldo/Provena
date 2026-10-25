"""Claude as the payment reviewer. It reviews the rules' default actions, may change them only within the rules,
explains each decision and writes the buyer report and contributor notes. Every tool call is logged as a decision
(actor "claude"); the money split is always recalculated by rules.split. If Claude is unavailable, refuses or does
not finish, the rules' decision stands."""

import json
import sqlite3

import anthropic

from gate import rules
from gate.findings import report

MODEL = "claude-opus-5-5"
MAX_TURNS = 20

SYSTEM = """You review a delivery of video footage before a buyer pays the vendor's invoice. A matching engine has \
compared every clip's fingerprints with the rest of the delivery, the buyer's own library, other buyers' libraries \
(only where both sides opted in; their details stay hidden) and public AI datasets. Fixed rules have proposed a \
default action for each finding: hold (don't pay for the matched seconds) or escalate (a person decides).

Your job:
1. Read the delivery and the findings. Look at the evidence for any finding where the default might be wrong.
2. Call decide for each finding you confirm or change, with a one-sentence reason a vendor would accept. Only use an \
action listed as allowed for that finding.
3. Call escalate when the evidence is ambiguous, with a specific question for the person reviewing it.
4. Finish by calling write_report once: a short report for the buyer and a plain note for each contributor whose \
pay is affected.

Use only facts from the tools; never invent matches, clips or numbers. A public-dataset match means the footage \
appears in that public dataset; never say a model was trained on it. Contributors are often paid by the hour, so be \
fair: when unsure, escalate rather than hold. Write plainly."""


def _tool(name: str, description: str, properties: dict) -> dict:
    return {"name": name, "description": description, "strict": True,
            "input_schema": {"type": "object", "properties": properties, "required": list(properties),
                             "additionalProperties": False}}


FINDING = {"finding_id": {"type": "integer"}}
TOOLS = [
    _tool("get_delivery", "The invoice, the delivered footage and the current split (approve / hold / review).", {}),
    _tool("get_findings", "Every finding with its current action, the actions allowed for it and the rules' reason.", {}),
    _tool("get_evidence", "Everything known about one finding: matched seconds on both sides, matching frames, "
          "static or not, source, licence.", FINDING),
    _tool("decide", "Confirm or change the action for one finding. Refused if the action is not allowed.",
          {**FINDING, "action": {"type": "string", "enum": ["hold", "escalate"]}, "reason": {"type": "string"}}),
    _tool("escalate", "Send one finding to a person with a specific question.", {**FINDING, "question": {"type": "string"}}),
    _tool("write_report", "Finish the review: a short report for the buyer and a note per affected contributor.",
          {"buyer_report": {"type": "string"},
           "contributor_notes": {"type": "array", "items": {
               "type": "object", "properties": {"contributor": {"type": "string"}, "note": {"type": "string"}},
               "required": ["contributor", "note"], "additionalProperties": False}}}),
]


def _log(con: sqlite3.Connection, delivery_id: int, action: str, reason: str, finding_id: int | None = None,
         amount: float | None = None) -> None:
    with con:
        con.execute("INSERT INTO decisions (delivery_id, finding_id, actor, action, amount, reason) "
                    "VALUES (?, ?, 'claude', ?, ?, ?)", (delivery_id, finding_id, action, amount, reason))


def run_agent(con: sqlite3.Connection, delivery_id: int, client: anthropic.Anthropic | None = None) -> dict:
    """Review a ruled delivery. Returns the split plus {agent, actions, buyer_report, contributor_notes}."""
    r = report(con, delivery_id)
    findings = {f["id"]: f for f in r["findings"]}
    actions = {i: rules.default_action(f) for i, f in findings.items()}
    reasons = {row["finding_id"]: row["reason"] for row in con.execute(
        "SELECT finding_id, reason FROM decisions WHERE delivery_id = ? AND actor = 'rules' AND finding_id IS NOT NULL",
        (delivery_id,))}
    done: dict = {}

    def run(name: str, args: dict) -> dict:
        if name == "get_delivery":
            return {"delivery": r["delivery"], "summary": r["summary"], "split": rules.split(r, actions)}
        if name == "get_findings":
            return {"findings": [{k: f[k] for k in ("id", "clip_name", "contributor", "kind", "source", "clip_start",
                                                    "clip_end", "votes", "static", "amount")}
                                 | {"current_action": actions[i], "allowed": sorted(rules.allowed_actions(f)),
                                    "rules_reason": reasons.get(i)} for i, f in findings.items()]}
        f = findings.get(args.get("finding_id"))
        if name in ("get_evidence", "decide", "escalate") and f is None:
            raise ValueError(f"no finding {args.get('finding_id')} in this delivery")
        if name == "get_evidence":
            return {**f, "current_action": actions[f["id"]], "allowed": sorted(rules.allowed_actions(f)),
                    "note": "static: the matched frames barely change, so the match is weak evidence" if f["static"]
                    else f"{f['votes']} frames agree on one time offset"}
        if name == "decide":
            rules.check(f, args["action"])
            actions[f["id"]] = args["action"]
            _log(con, delivery_id, args["action"], args["reason"], f["id"], f["amount"])
            return {"ok": True}
        if name == "escalate":
            actions[f["id"]] = "escalate"
            _log(con, delivery_id, "escalate", args["question"], f["id"], f["amount"])
            return {"ok": True}
        if name == "write_report":
            done.update(args)
            _log(con, delivery_id, "report", args["buyer_report"])
            for n in args["contributor_notes"]:
                _log(con, delivery_id, "note", f"{n['contributor']}: {n['note']}")
            return {"ok": True}
        raise ValueError(f"unknown tool {name}")

    client = client or anthropic.Anthropic()
    messages = [{"role": "user", "content": f"Review delivery {delivery_id}. Start with get_delivery and get_findings."}]
    failure = None
    try:
        for _ in range(MAX_TURNS):
            response = client.beta.messages.create(
                model=MODEL, max_tokens=16000, system=SYSTEM, tools=TOOLS, messages=messages,
                output_config={"effort": "medium"}, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
                if response.stop_reason != "end_turn":
                    failure = f"stopped: {response.stop_reason}"
                    break
                messages.append({"role": "user", "content": "Call write_report to finish the review."})
                continue
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                try:
                    out, error = run(block.name, block.input), False
                    if block.name.startswith("get_"):
                        _log(con, delivery_id, "look", f"{block.name} {json.dumps(block.input)}")
                except (ValueError, KeyError) as e:
                    out, error = {"error": str(e)}, True
                    _log(con, delivery_id, "refused", f"{block.name} {json.dumps(block.input)}: {e}",
                         block.input.get("finding_id"))
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(out),
                                "is_error": error})
            messages.append({"role": "user", "content": results})
            if done:
                break
        else:
            failure = f"did not finish in {MAX_TURNS} turns"
    except anthropic.APIError as e:
        failure = f"Claude unavailable: {e.__class__.__name__}"

    if failure or not done:
        reason = failure or "no report written"
        _log(con, delivery_id, "unavailable", reason + "; the rules' decision stands")
        defaults = {i: rules.default_action(f) for i, f in findings.items()}
        return {**rules.split(r, defaults), "agent": False, "reason": reason,
                "actions": [{"finding_id": i, "action": a} for i, a in defaults.items()]}
    totals = rules.split(r, actions)
    with con:
        con.execute("INSERT INTO decisions (delivery_id, actor, action, amount, reason) VALUES (?, 'claude', 'split', ?, ?)",
                    (delivery_id, totals["approved"], json.dumps(totals)))
        con.execute("UPDATE deliveries SET status = 'decided' WHERE id = ?", (delivery_id,))
    return {**totals, "agent": True, "actions": [{"finding_id": i, "action": a} for i, a in actions.items()],
            "buyer_report": done["buyer_report"], "contributor_notes": done["contributor_notes"]}
