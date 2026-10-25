import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GATE_DATA", str(tmp_path))
    monkeypatch.setenv("GATE_PUBLIC", "none")
    monkeypatch.setenv("GATE_AGENT", "off")
    import importlib
    from gate import app as app_module
    importlib.reload(app_module)
    with TestClient(app_module.app) as c:
        yield c


def upload(paths):
    return [("files", (p.name, p.read_bytes(), "video/mp4")) for p in paths]


def test_api_end_to_end(client, videos):
    acme = client.post("/api/accounts", json={"name": "acme"}).json()
    assert client.post("/api/accounts", json={"name": "acme"}).status_code == 409
    added = client.post(f"/api/accounts/{acme['id']}/library", files=upload([videos / "x.mp4", videos / "still.mp4"])).json()
    assert len(added["added"]) == 2 and added["errors"] == {}
    assert [c["name"] for c in client.get(f"/api/accounts/{acme['id']}/library").json()] == ["x.mp4", "still.mp4"]

    r = client.post("/api/deliveries", data={
        "account_id": acme["id"], "vendor": "Northlight", "rate_per_hour": 150, "invoice_total": 2.0,
        "contributors": json.dumps({"x_edited.mp4": "maria"})},
        files=upload([videos / "x_edited.mp4", videos / "v.mp4", videos / "still.mp4"]))
    assert r.status_code == 201
    d = r.json()
    by = {f["clip_name"]: f for f in d["findings"]}
    assert by["x_edited.mp4"]["kind"] == "in_own_library" and by["x_edited.mp4"]["contributor"] == "maria"
    assert by["x_edited.mp4"]["action"] == "hold" and by["still.mp4"]["action"] == "escalate"
    assert d["split"]["by"] == "rules" and d["delivery"]["status"] == "ruled"
    assert client.get(f"/api/deliveries/{d['delivery']['id']}").json() == d
    assert [x["id"] for x in client.get(f"/api/deliveries?account_id={acme['id']}").json()] == [d["delivery"]["id"]]
    assert client.get("/api/deliveries/999").status_code == 404
    assert client.get("/").status_code == 200
