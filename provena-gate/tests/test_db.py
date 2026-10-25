import sqlite3

import pytest

from gate import db


def test_accounts(tmp_path):
    con = db.connect(tmp_path / "gate.sqlite")
    a = db.create_account(con, "acme", share_fingerprints=True)
    b = db.create_account(con, "orbit")
    assert [(r["id"], r["name"], r["share_fingerprints"]) for r in db.accounts(con)] == [(a, "acme", 1), (b, "orbit", 0)]
    with pytest.raises(sqlite3.IntegrityError):
        db.create_account(con, "acme")
    con.close()
    assert len(db.accounts(db.connect(tmp_path / "gate.sqlite"))) == 2  # reopening keeps data, schema is idempotent
