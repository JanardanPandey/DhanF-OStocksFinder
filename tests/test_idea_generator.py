from __future__ import annotations

from app.dhan_client import DhanClient
from app.idea_generator import generate_symbol_ideas


def test_idea_generator_respects_tight_sl_and_min_rr():
    cfg = {
        "risk_per_trade_inr": 1000.0,
        "max_sl_pct_intraday": 0.65,
        "max_sl_pct_swing": 1.80,
        "min_rr_ratio": 2.0,
    }
    found = 0
    for sym in ["RELIANCE", "HDFCBANK", "ICICIBANK", "SBIN", "TCS", "INFY", "HAL", "TRENT"]:
        df_1d = DhanClient._generate_simulated_candles(sym, "1D", 160)
        df_1h = DhanClient._generate_simulated_candles(sym, "1h", 160)
        df_15m = DhanClient._generate_simulated_candles(sym, "15m", 180)
        df_5m = DhanClient._generate_simulated_candles(sym, "5m", 180)
        out = generate_symbol_ideas(sym, 500, df_1d, df_1h, df_15m, df_5m, "demo", cfg)
        for idea in out["ideas"]:
            found += 1
            max_sl = cfg["max_sl_pct_intraday"] if idea["style"] == "INTRADAY" else cfg["max_sl_pct_swing"]
            assert idea["sl_pct"] <= max_sl + 0.05
            assert idea["rr_ratio"] >= cfg["min_rr_ratio"]
            assert idea["suggested_qty"] >= 1
            if idea["direction"] == "LONG":
                assert idea["stop_loss"] < idea["entry_price"] < idea["tp1"] < idea["tp2"]
            else:
                assert idea["stop_loss"] > idea["entry_price"] > idea["tp1"] > idea["tp2"]
    assert found >= 1
