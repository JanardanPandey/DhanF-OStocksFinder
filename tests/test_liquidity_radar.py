from __future__ import annotations

import datetime
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database import db
from app.liquidity_engine import get_available_liquidity_dates, scan_point_in_time_liquidity_setups, get_htf_liquidity_levels


def _seed_test_liquidity_candles(symbol: str = "RELIANCE"):
    """Seed 1D history and 5m intraday bars for testing HTF liquidity detection."""
    # Seed 1D candles (10 trading days ending 2026-11-17)
    base_epoch = 1731200000
    daily_recs = []
    for i in range(10):
        # 86400 seconds per day
        ts = base_epoch + (i * 86400)
        daily_recs.append({
            "ts": ts,
            "open": 1200.0 + i,
            "high": 1250.0 + i,
            "low": 1180.0 - i,
            "close": 1220.0 + i,
            "volume": 100000.0,
            "oi": 50000.0,
        })
    db.save_candles(symbol, "1D", daily_recs)

    # Seed 5m candles for trade date 2026-11-18
    # 2026-11-18 09:15 IST = 1731901500 (approx)
    tz_ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    d = datetime.date(2026, 11, 18)
    start_dt = datetime.datetime(d.year, d.month, d.day, 9, 15, tzinfo=tz_ist)
    start_ts = int(start_dt.timestamp())

    # Create 10 5m bars:
    # Bar 0: open at 1220
    # Bar 1: sweeps PDH (1259) with high=1262, then rejects down to 1245 (large upper wick)
    c5_recs = []
    for i in range(10):
        ts = start_ts + (i * 300)
        if i == 0:
            c5_recs.append({"ts": ts, "open": 1220.0, "high": 1240.0, "low": 1215.0, "close": 1235.0, "volume": 5000.0, "oi": 50000.0})
        elif i == 1:
            # Sweeps above PDH (1259), upper wick = 1262 - 1245 = 17 pts out of 22 pt range (>75% wick), closes below 1259
            c5_recs.append({"ts": ts, "open": 1240.0, "high": 1262.0, "low": 1240.0, "close": 1245.0, "volume": 15000.0, "oi": 52000.0})
        elif i == 2:
            # Rejection confirmation bar opens at 1245, drops to 1230
            c5_recs.append({"ts": ts, "open": 1245.0, "high": 1246.0, "low": 1228.0, "close": 1230.0, "volume": 12000.0, "oi": 53000.0})
        else:
            c5_recs.append({"ts": ts, "open": 1230.0 - i, "high": 1235.0 - i, "low": 1225.0 - i, "close": 1228.0 - i, "volume": 6000.0, "oi": 54000.0})

    db.save_candles(symbol, "5m", c5_recs)


def test_liquidity_empty_db_graceful():
    dates = get_available_liquidity_dates()
    assert isinstance(dates, list)
    setups = scan_point_in_time_liquidity_setups()
    assert setups == []


def test_liquidity_seeded_detection_and_api():
    _seed_test_liquidity_candles("RELIANCE")

    dates = get_available_liquidity_dates()
    assert "2026-11-18" in dates

    levels = get_htf_liquidity_levels("RELIANCE", "2026-11-18")
    assert levels is not None
    assert levels["daily_bsl"] > levels["daily_ssl"]
    assert levels["weekly_bsl"] >= levels["weekly_ssl"]

    setups = scan_point_in_time_liquidity_setups("2026-11-18", min_rr=1.5, min_reward_pts=5.0)
    assert isinstance(setups, list)
    assert len(setups) > 0

    first = setups[0]
    assert first["symbol"] == "RELIANCE"
    assert "Sweep" in first["setup_name"]
    assert first["direction"] == "SHORT"
    assert first["risk_pts"] > 0
    assert first["reward_pts"] > 0
    assert "outcome" in first
    assert "outcome_label" in first
    assert "current_pnl" in first

    # Test outcome filtering
    active_setups = scan_point_in_time_liquidity_setups("2026-11-18", min_rr=1.5, min_reward_pts=5.0, outcome_filter="ACTIVE")
    assert isinstance(active_setups, list)

    # API test
    client = TestClient(app)
    res = client.get("/api/liquidity/radar?trade_date=2026-11-18&outcome=ALL")
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert data["trade_date"] == "2026-11-18"
    assert "available_dates" in data
    assert "2026-11-18" in data["available_dates"]
    assert data["count"] >= 1
