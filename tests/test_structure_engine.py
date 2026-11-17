from __future__ import annotations

import pandas as pd

from app.dhan_client import DhanClient
from app.structure_engine import analyze_structure


def test_structure_engine_detects_ob_sr_and_trend():
    df = DhanClient._generate_simulated_candles("RELIANCE", "15m", 220)
    res = analyze_structure(df, swing_window=3)

    assert res["trend"] in ("BULLISH", "BEARISH", "RANGING")
    assert res["ltp"] > 0
    assert res["atr"] > 0
    assert len(res["sr_zones"]) > 0
    assert isinstance(res["order_blocks"], list)
    assert isinstance(res["fvgs"], list)


def test_bullish_order_block_and_fvg_detection():
    # Craft an explicit bullish impulse leaving an OB and FVG
    rows = []
    px = 100.0
    ts = 1_700_000_000
    for i in range(30):
        op = px
        cl = px + (0.3 if i % 2 == 0 else -0.15)
        rows.append({"ts": ts + i * 300, "open": op, "high": max(op, cl) + 0.2, "low": min(op, cl) - 0.2, "close": cl, "volume": 10000.0})
        px = cl

    # Bearish OB candle at idx 30
    rows.append({"ts": ts + 30 * 300, "open": 103.0, "high": 103.2, "low": 101.8, "close": 102.0, "volume": 12000.0})
    # Impulsive displacement up at idx 31 & 32 leaving FVG (low[32] > high[30])
    rows.append({"ts": ts + 31 * 300, "open": 102.0, "high": 105.0, "low": 102.0, "close": 104.8, "volume": 35000.0})
    rows.append({"ts": ts + 32 * 300, "open": 104.8, "high": 107.2, "low": 104.5, "close": 107.0, "volume": 32000.0})
    for i in range(33, 45):
        rows.append({"ts": ts + i * 300, "open": 106.5, "high": 107.0, "low": 106.0, "close": 106.6, "volume": 10000.0})

    df = pd.DataFrame(rows)
    res = analyze_structure(df, swing_window=2)
    bull_obs = [ob for ob in res["order_blocks"] if ob["direction"] == "BULLISH"]
    assert len(bull_obs) >= 1
    assert len(res["fvgs"]) >= 1
