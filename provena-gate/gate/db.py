"""SQLite store for results: accounts, clips, deliveries, findings, decisions, disputes.

Fingerprints never go here: they stay in provena-audit shard files, one library folder per account.
"""

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    share_fingerprints INTEGER NOT NULL DEFAULT 0,  -- opt-in: compare fingerprints (never footage) with other accounts
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS deliveries (
    id INTEGER PRIMARY KEY,
    account_id INTEGER NOT NULL REFERENCES accounts(id),
    vendor TEXT NOT NULL,
    rate_per_hour REAL NOT NULL,                    -- invoice price per hour of footage
    invoice_total REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'received',        -- received -> checked -> decided
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS clips (
    id TEXT PRIMARY KEY,                            -- the fingerprint ID in the shards
    account_id INTEGER NOT NULL REFERENCES accounts(id),
    delivery_id INTEGER REFERENCES deliveries(id),  -- NULL: part of the account's library
    contributor TEXT,
    name TEXT NOT NULL,                             -- original file name or path
    duration REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS findings (
    id INTEGER PRIMARY KEY,
    delivery_id INTEGER NOT NULL REFERENCES deliveries(id),
    clip_id TEXT NOT NULL REFERENCES clips(id),
    kind TEXT NOT NULL,                             -- repeat_in_delivery | in_own_library | sold_to_other_buyer | public_dataset
    match_source TEXT NOT NULL,                     -- account ID or dataset name
    match_ref TEXT NOT NULL,                        -- the matching clip's ID
    clip_start REAL NOT NULL, clip_end REAL NOT NULL,
    match_start REAL NOT NULL, match_end REAL NOT NULL,
    votes INTEGER NOT NULL,
    static INTEGER NOT NULL DEFAULT 0,              -- little motion: weak evidence, a human decides
    amount REAL NOT NULL                            -- invoice value of the matched seconds
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY,
    delivery_id INTEGER NOT NULL REFERENCES deliveries(id),
    finding_id INTEGER REFERENCES findings(id),     -- NULL: about the whole delivery
    actor TEXT NOT NULL,                            -- rules | claude | human
    action TEXT NOT NULL,                           -- approve | hold | escalate | report | ...
    amount REAL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS disputes (
    id INTEGER PRIMARY KEY,
    finding_id INTEGER NOT NULL REFERENCES findings(id),
    author TEXT NOT NULL,
    message TEXT NOT NULL,
    reply TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    return con


def create_account(con: sqlite3.Connection, name: str, share_fingerprints: bool = False) -> int:
    with con:
        return con.execute("INSERT INTO accounts (name, share_fingerprints) VALUES (?, ?)",
                           (name, int(share_fingerprints))).lastrowid


def accounts(con: sqlite3.Connection) -> list[sqlite3.Row]:
    return con.execute("SELECT * FROM accounts ORDER BY id").fetchall()
