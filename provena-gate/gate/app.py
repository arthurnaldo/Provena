"""HTTP API and dashboard. Run: uvicorn gate.app:app

GATE_DATA     data folder (default provena-gate/data): gate.sqlite, account libraries, delivery fingerprints
GATE_PUBLIC   public dataset folders, ":"-separated (default: every folder in ../data/index); "none" to skip
GATE_AGENT    "off" to skip the Claude review (the rules' decision stands)
"""

import json
import os
import shutil
import sqlite3
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from provena_audit.audit import Index

from gate import agent, db, engine, evidence, rules
from gate.findings import check_delivery, report

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("GATE_DATA", HERE.parent / "data"))
state: dict = {}


def public_folders() -> list[str]:
    spec = os.environ.get("GATE_PUBLIC")
    if spec == "none":
        return []
    if spec:
        return spec.split(":")
    index = HERE.parent.parent / "data" / "index"
    return sorted(str(d) for d in index.iterdir() if (d / "send").exists()) if index.exists() else []


@asynccontextmanager
async def lifespan(app: FastAPI):
    ROOT.mkdir(parents=True, exist_ok=True)
    folders = public_folders()
    state["public"] = Index(*folders) if folders else None
    state["datasets"] = [Path(f).name for f in folders]
    yield


app = FastAPI(title="Provena Gate", lifespan=lifespan)


def connect() -> sqlite3.Connection:
    return db.connect(ROOT / "gate.sqlite")


def save_uploads(files: list[UploadFile], tmp: Path) -> list[Path]:
    """Each upload in its own folder so clips keep their original names."""
    paths = []
    for i, f in enumerate(files):
        p = tmp / str(i) / Path(f.filename or f"clip_{i}.mp4").name
        p.parent.mkdir(parents=True)
        with p.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        paths.append(p)
    return paths


def account_or_404(con: sqlite3.Connection, account_id: int) -> sqlite3.Row:
    row = con.execute("SELECT * FROM accounts WHERE id = ?", (account_id,)).fetchone()
    if row is None:
        raise HTTPException(404, f"no account {account_id}")
    return row


def delivery_view(con: sqlite3.Connection, delivery_id: int) -> dict:
    """report() plus the current action per finding, the split, Claude's report and notes, and the decision log."""
    if con.execute("SELECT 1 FROM deliveries WHERE id = ?", (delivery_id,)).fetchone() is None:
        raise HTTPException(404, f"no delivery {delivery_id}")
    r = report(con, delivery_id)
    r["errors"] = {Path(k).name: v for k, v in r["errors"].items()}
    log = [dict(row) for row in con.execute(
        "SELECT actor, action, finding_id, amount, reason, created_at FROM decisions WHERE delivery_id = ? ORDER BY id",
        (delivery_id,))]
    actions, split, buyer_report, notes, agent_note = {}, None, None, [], None
    for row in log:
        if row["finding_id"] is not None and row["action"] in ("hold", "escalate"):
            actions[row["finding_id"]] = {"action": row["action"], "reason": row["reason"], "actor": row["actor"]}
        elif row["action"] == "split":
            split = json.loads(row["reason"][row["reason"].index("{"):]) | {"by": row["actor"]}
        elif row["action"] == "report":
            buyer_report = row["reason"]
        elif row["action"] == "note":
            notes.append(row["reason"])
        elif row["action"] == "unavailable":
            agent_note = row["reason"]
    for f in r["findings"]:
        f.update(actions.get(f["id"], {}))
    return {**r, "split": split, "buyer_report": buyer_report, "contributor_notes": notes,
            "agent_note": agent_note, "log": log}


class NewAccount(BaseModel):
    name: str
    share_fingerprints: bool = False


@app.get("/api/status")
def status() -> dict:
    ix = state.get("public")
    return {"public_datasets": state.get("datasets", []), "public_clips": len(ix.ids) if ix else 0,
            "agent": os.environ.get("GATE_AGENT") != "off" and bool(os.environ.get("ANTHROPIC_API_KEY"))}


@app.get("/api/accounts")
def list_accounts() -> list[dict]:
    with connect() as con:
        return [dict(r) | {"library_clips": con.execute(
            "SELECT COUNT(*) FROM clips WHERE account_id = ? AND delivery_id IS NULL", (r["id"],)).fetchone()[0]}
                for r in db.accounts(con)]


@app.post("/api/accounts", status_code=201)
def create_account(body: NewAccount) -> dict:
    with connect() as con:
        try:
            account_id = db.create_account(con, body.name.strip(), body.share_fingerprints)
        except sqlite3.IntegrityError:
            raise HTTPException(409, f"an account named {body.name!r} already exists")
        return dict(account_or_404(con, account_id))


@app.get("/api/accounts/{account_id}/library")
def library(account_id: int) -> list[dict]:
    with connect() as con:
        account_or_404(con, account_id)
        return [dict(r) for r in con.execute(
            "SELECT id, name, duration, created_at FROM clips WHERE account_id = ? AND delivery_id IS NULL ORDER BY created_at",
            (account_id,))]


@app.post("/api/accounts/{account_id}/library")
def add_library(account_id: int, files: list[UploadFile] = File(...)) -> dict:
    with connect() as con, tempfile.TemporaryDirectory() as tmp:
        account_or_404(con, account_id)
        ids, errors = engine.add_to_library(con, ROOT, account_id, save_uploads(files, Path(tmp)))
        return {"added": ids, "errors": {Path(k).name: v for k, v in errors.items()}}


@app.get("/api/deliveries")
def list_deliveries(account_id: int) -> list[dict]:
    with connect() as con:
        return [dict(r) for r in con.execute(
            "SELECT id, vendor, invoice_total, status, created_at FROM deliveries WHERE account_id = ? ORDER BY id DESC",
            (account_id,))]


@app.post("/api/deliveries", status_code=201)
def new_delivery(account_id: int = Form(...), vendor: str = Form(...), rate_per_hour: float = Form(...),
                 invoice_total: float = Form(...), contributors: str = Form("{}"),
                 files: list[UploadFile] = File(...)) -> dict:
    """Check a delivery, apply the rules, then let Claude review it. contributors: JSON {file name: contributor}."""
    names = json.loads(contributors or "{}")
    with connect() as con, tempfile.TemporaryDirectory() as tmp:
        account_or_404(con, account_id)
        paths = save_uploads(files, Path(tmp))
        r = check_delivery(con, ROOT, state.get("public"), account_id, vendor, rate_per_hour, invoice_total,
                           [(p, names.get(p.name) or "unknown") for p in paths])
        delivery_id = r["delivery"]["id"]
        rules.apply_rules(con, delivery_id)
        if status()["agent"]:
            agent.run_agent(con, delivery_id)
        return delivery_view(con, delivery_id)


@app.get("/api/deliveries/{delivery_id}")
def get_delivery(delivery_id: int) -> dict:
    with connect() as con:
        return delivery_view(con, delivery_id)


@app.get("/api/deliveries/{delivery_id}/findings/{finding_id}/evidence")
def finding_evidence(delivery_id: int, finding_id: int) -> dict:
    """Side-by-side frames; another buyer's frames are never included."""
    with connect() as con:
        if con.execute("SELECT 1 FROM deliveries WHERE id = ?", (delivery_id,)).fetchone() is None:
            raise HTTPException(404, f"no delivery {delivery_id}")
        e = evidence.evidence(con, ROOT, delivery_id, finding_id)
        if e is None:
            raise HTTPException(404, f"no finding {finding_id} in delivery {delivery_id}")
        return e


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(HERE / "static" / "index.html")
