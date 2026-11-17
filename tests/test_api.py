from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient

from app.database import Database
import app.database as db_mod
import app.main as main_mod
import app.paper_trader as pt_mod
import app.scanner as scanner_mod
import app.dhan_client as dc_mod


def test_api_end_to_end(tmp_path: Path, monkeypatch):
    test_db = Database(tmp_path / "api_test.db")
    monkeypatch.setattr(db_mod, "db", test_db)
    monkeypatch.setattr(main_mod, "db", test_db)
    monkeypatch.setattr(pt_mod, "db", test_db)
    monkeypatch.setattr(scanner_mod, "db", test_db)
    monkeypatch.setattr(dc_mod, "db", test_db)

    with TestClient(main_mod.app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "Dhan SMC Setup Desk" in r.text

        h = client.get("/api/health").json()
        assert h["universe_count"] >= 20

        setups_res = client.get("/api/setups").json()
        assert setups_res["total"] >= 1
        first = setups_res["items"][0]

        chart_res = client.get(f"/api/chart/{first['symbol']}?timeframe=15m").json()
        assert len(chart_res["candles"]) > 50
        assert "structure" in chart_res

        open_res = client.post("/api/paper/open", json={"setup_id": first["id"]})
        assert open_res.status_code == 200
        pos_id = open_res.json()["position"]["id"]

        close_res = client.post(f"/api/paper/positions/{pos_id}/close")
        assert close_res.status_code == 200
        assert close_res.json()["result"]["status"] == "CLOSED"
