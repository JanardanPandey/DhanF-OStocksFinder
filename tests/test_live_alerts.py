from __future__ import annotations

import struct
from fastapi.testclient import TestClient

from app.database import db
from app.dhan_websocket import dhan_feed
from app.main import app
from app.playbook_engine import evaluate_and_record_alerts


client = TestClient(app)


def test_live_alert_deduplication_and_timeline():
    db.clear_live_alerts()

    item_amber = {
        "symbol": "AMBER",
        "direction": "LONG",
        "option_safe": True,
        "score": 88.0,
        "status_tag": "CREAM_BULL_V2",
        "ltp": 6740.0,
        "vwap": 6680.0,
        "move_pct": 2.8,
        "order_block": {
            "status": "OB_BREAKOUT_CONFIRMED",
            "ob_zone": "₹6,680.00 - ₹6,720.00",
            "s1": 6670.0,
            "r1": 6820.0,
            "r2": 6940.0,
        },
        "option_strike": {
            "option_name": "AMBER 28 NOV 6900 CE",
            "est_premium": 127.9,
            "est_capital_inr": 12790.0,
            "sl_premium": 89.5,
            "tp1_premium": 185.5,
        },
        "oi_build_up": "CALL_UNWINDING_SQUEEZE",
        "vol_delta_shares": "+18.90L",
        "reasons": ["Clean OB Breakout", "VWAP hold"],
    }

    # First detection creates 1 alert
    alerts1 = evaluate_and_record_alerts([item_amber])
    assert len(alerts1) == 1
    assert alerts1[0]["event_type"] == "OB_BREAKOUT"
    assert alerts1[0]["symbol"] == "AMBER"

    # Duplicate tick must NOT create any new alert
    alerts2 = evaluate_and_record_alerts([item_amber])
    assert len(alerts2) == 0

    # Moving to Target R1 triggers a TARGET_R1_HIT alert without removing prior alert
    item_amber_hit = dict(item_amber)
    item_amber_hit["ltp"] = 6825.0
    alerts3 = evaluate_and_record_alerts([item_amber_hit])
    assert len(alerts3) == 1
    assert alerts3[0]["event_type"] == "TARGET_R1_HIT"

    # All alerts are saved in SQLite
    stored = db.get_live_alerts()
    assert len(stored) >= 2


def test_api_live_alerts_endpoint():
    res = client.get("/api/alerts/live")
    assert res.status_code == 200
    data = res.json()
    assert "alerts" in data
    assert "total" in data

    # Test filtering
    res_bo = client.get("/api/alerts/live?event_type=BREAKOUTS")
    assert res_bo.status_code == 200


def test_api_option_chain_amber():
    res = client.get("/api/option-chain/AMBER")
    assert res.status_code == 200
    data = res.json()
    assert data["symbol"] == "AMBER"
    assert "recommended_strike" in data
    assert "strikes" in data
    assert len(data["strikes"]) > 0


def test_dhan_websocket_quote_unpack():
    # 8-byte header + 42-byte quote payload
    header = struct.pack("<BhBI", 4, 50, 1, 1964)
    payload = struct.pack("<fHIffIIffff", 7450.0, 100, 1731518500, 7420.0, 890000, 12000, 9000, 7380.0, 7490.0, 7350.0, 7400.0)
    packet = header + payload

    dhan_feed.register_instrument_mapping("TRENT", "1964")
    dhan_feed._parse_binary_packet(packet)

    q = dhan_feed.get_quote("TRENT")
    assert q is not None
    assert q["last_price"] == 7450.0
    assert q["symbol"] == "TRENT"


def test_evaluate_live_momentum_board_filters_sideways_chop():
    from app.playbook_engine import evaluate_live_momentum_board

    symbols = ["LAURUSLABS", "CHOPSTOCK"]
    quotes_map = {
        "LAURUSLABS": {
            "last_price": 2280.0,
            "average_price": 2310.0,
            "open": 2350.0,
            "prev_close": 2350.0,
        },
        "CHOPSTOCK": {
            "last_price": 100.5,
            "average_price": 100.0,
            "open": 100.0,
            "prev_close": 100.0,
        },
    }
    breadth = {"abv_vwap_pct": 25.0, "source": "live", "trade_date": "2026-11-16"}

    board = evaluate_live_momentum_board(symbols, quotes_map, breadth, top_n=15)
    # CHOPSTOCK with tiny 0.5% move must be filtered out
    syms = [it["symbol"] for it in board["items"]]
    assert "CHOPSTOCK" not in syms

    # Any stock in CHOP_DECAY must NOT be marked option_safe
    for item in board["items"]:
        if item.get("option_phase") == "CHOP_DECAY":
            assert item["option_safe"] is False
