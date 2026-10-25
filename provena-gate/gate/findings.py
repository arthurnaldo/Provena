"""Delivery check: a vendor's clips + invoice -> findings (what matched where, how strongly, what it is worth)."""

import json
import math
import sqlite3
from pathlib import Path

import numpy as np
from provena_audit import shards
from provena_audit.audit import FPS, Index, find_pairs
from provena_audit.storage import Location

from gate import engine

MIN_VOTES = 8
STATIC_BITS = 6  # median PDQ change between frames 0.25 s apart below this: barely moving footage (to tune)
KINDS = ("repeat_in_delivery", "in_own_library", "sold_to_other_buyer", "public_dataset")


def is_static(pdq: np.ndarray, start: float, end: float) -> bool:
    """True if the clip's frames in [start, end) barely change: weak evidence, every shift matches."""
    first = int(start * FPS)
    bits = np.unpackbits(pdq[first:max(math.ceil(end * FPS), first + 1)], axis=1)
    return len(bits) < 2 or float(np.median((bits[1:] != bits[:-1]).sum(1))) < STATIC_BITS


def check_delivery(con: sqlite3.Connection, root: str | Path, public: Index | None, account_id: int, vendor: str,
                   rate_per_hour: float, invoice_total: float, clips: list[tuple[str | Path, str]]) -> dict:
    """Fingerprint a delivery, find repeats inside it and matches in account libraries and public datasets,
    save the findings and return report(). clips: [(video file, contributor)]."""
    with con:
        did = con.execute("INSERT INTO deliveries (account_id, vendor, rate_per_hour, invoice_total) VALUES (?, ?, ?, ?)",
                          (account_id, vendor, rate_per_hour, invoice_total)).lastrowid
    contributor = {str(p): c for p, c in clips}
    folder = engine.delivery_dir(root, did)
    added, errors = engine.fingerprint_into(folder, "dlv", [p for p, _ in clips])
    with con:
        con.executemany("INSERT INTO clips (id, account_id, delivery_id, contributor, name, duration) VALUES (?, ?, ?, ?, ?, ?)",
                        [(v["id"], account_id, did, contributor[v["path"]], v["name"], v["duration"]) for v in added])
    found = []  # (clip, kind, source, match ref, clip start, clip end, match start, match end, votes, details)

    # The same footage twice in this delivery: the later clip is the repeat.
    if len(added) > 1:
        data = shards.load(Location(folder / "send"))
        for p in find_pairs(data, MIN_VOTES):
            found.append((str(data["ids"][p["b"]]), "repeat_in_delivery", "this delivery", str(data["ids"][p["a"]]),
                          p["b_start"], p["b_end"] + 1 / FPS, p["a_start"], p["a_end"] + 1 / FPS, p["votes"], {}))

    # The account's own library, other accounts' libraries when both sides share fingerprints, public datasets.
    sharing = con.execute("SELECT share_fingerprints FROM accounts WHERE id = ?", (account_id,)).fetchone()[0]
    libraries = [engine.library_dir(root, r["id"]) for r in con.execute(
        "SELECT id FROM accounts WHERE id = ? OR (? AND share_fingerprints)", (account_id, sharing))]
    libraries = [str(f) for f in libraries if any((f / "send").glob("*.npz"))]
    indexes = ([Index(*libraries)] if libraries else []) + ([public] if public is not None else [])
    for v in added:
        for ix in indexes:
            for h in ix.search(v["path"], min_votes=MIN_VOTES):
                src = h["dataset"]
                kind = ("in_own_library" if src == f"acct:{account_id}" else
                        "sold_to_other_buyer" if src.startswith("acct:") else "public_dataset")
                details = {k: h.get(f) for k, f in (("licence", "licence_stated"), ("dataset_url", "source"),
                                                    ("source_url", "source_url")) if h.get(f)}
                found.append((v["id"], kind, src, h["id"], h["clip_start"], h["clip_end"], h["b_start"], h["b_end"],
                              h["votes"], details))

    pdq = {v["id"]: v["pdq"] for v in added}
    rows, static_seen = [], set()
    for c, kind, src, ref, cs, ce, ms, me, votes, details in sorted(found, key=lambda f: -f[8]):
        static = is_static(pdq[c], cs, ce)
        if static and (c, ref) in static_seen:  # still footage matches at every shift: keep the strongest only
            continue
        if static:
            static_seen.add((c, ref))
        rows.append((did, c, kind, src, ref, cs, ce, ms, me, int(votes), int(static),
                     round((ce - cs) / 3600 * rate_per_hour, 2), json.dumps(details)))
    with con:
        con.executemany(
            "INSERT INTO findings (delivery_id, clip_id, kind, match_source, match_ref, clip_start, clip_end, "
            "match_start, match_end, votes, static, amount, details) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows)
        con.execute("UPDATE deliveries SET status = 'checked', errors = ? WHERE id = ?", (json.dumps(errors), did))
    return report(con, did)


def _seconds(findings: list[dict]) -> float:
    """Flagged seconds, each second of a clip counted once however many findings cover it."""
    total = 0.0
    for clip in {f["clip_id"] for f in findings}:
        end = -1.0
        for s, e in sorted((f["clip_start"], f["clip_end"]) for f in findings if f["clip_id"] == clip):
            total += max(0.0, e - max(s, end))
            end = max(end, e)
    return total


def report(con: sqlite3.Connection, delivery_id: int) -> dict:
    """A checked delivery as returned by the API. Matches in another buyer's library never show that buyer's
    clip, times or name."""
    d = dict(con.execute("SELECT * FROM deliveries WHERE id = ?", (delivery_id,)).fetchone())
    errors = json.loads(d.pop("errors"))
    clips = {r["id"]: r for r in con.execute("SELECT * FROM clips WHERE delivery_id = ?", (delivery_id,))}
    findings = []
    for r in con.execute("SELECT * FROM findings WHERE delivery_id = ? ORDER BY id", (delivery_id,)):
        f = {"id": r["id"], "clip_id": r["clip_id"], "clip_name": clips[r["clip_id"]]["name"],
             "contributor": clips[r["clip_id"]]["contributor"], "kind": r["kind"], "source": r["match_source"],
             "match_ref": r["match_ref"], "clip_start": r["clip_start"], "clip_end": r["clip_end"],
             "match_start": r["match_start"], "match_end": r["match_end"], "votes": r["votes"],
             "static": bool(r["static"]), "amount": r["amount"], **json.loads(r["details"])}
        if r["kind"] == "repeat_in_delivery":
            f["match_name"] = clips[r["match_ref"]]["name"]
        elif r["kind"] == "sold_to_other_buyer":
            seen = con.execute("SELECT created_at FROM clips WHERE id = ?", (r["match_ref"],)).fetchone()
            f.update(source="another buyer", match_ref=None, first_seen=seen[0][:10] if seen else None)
            del f["match_start"], f["match_end"]
        elif r["kind"] == "in_own_library":
            f["match_name"] = con.execute("SELECT name FROM clips WHERE id = ?", (r["match_ref"],)).fetchone()[0]
        findings.append(f)
    rate = d["rate_per_hour"]
    seconds = _seconds(findings)
    by_kind = {}
    for kind in KINDS:
        fs = [f for f in findings if f["kind"] == kind]
        if fs:
            s = _seconds(fs)
            by_kind[kind] = {"clips": len({f["clip_id"] for f in fs}), "seconds": round(s, 1),
                             "amount": round(s / 3600 * rate, 2)}
    d.update(clips=len(clips), hours=round(sum(c["duration"] for c in clips.values()) / 3600, 3))
    summary = {"flagged_clips": len({f["clip_id"] for f in findings}), "flagged_seconds": round(seconds, 1),
               "flagged_amount": round(seconds / 3600 * rate, 2), "static_findings": sum(f["static"] for f in findings),
               "by_kind": by_kind}
    return {"delivery": d, "summary": summary, "findings": findings, "errors": errors}
