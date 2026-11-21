from __future__ import annotations

import pandas as pd
import pytest

from app.database import db
from app.playbook_engine import (
    _classify_option_timing,
    compute_order_block,
    evaluate_and_record_alerts,
)


def test_database_candles_oi_column():
    """Verify that candles table stores and retrieves Open Interest (oi)."""
    sym = "TESTSYM_OI"
    recs = [
        {"ts": 1731816000, "open": 100.0, "high": 105.0, "low": 99.0, "close": 104.0, "volume": 1000.0, "oi": 45000.0},
        {"ts": 1731816300, "open": 104.0, "high": 106.0, "low": 103.0, "close": 105.5, "volume": 1200.0, "oi": 48500.0},
    ]
    db.save_candles(sym, "5m", recs)
    loaded = db.load_candles(sym, "5m", 10)
    assert len(loaded) >= 2
    assert float(loaded[-1].get("oi", 0)) == 48500.0

    with db.connect() as conn:
        conn.execute("DELETE FROM candles WHERE symbol = ?", (sym,))
        conn.commit()


def test_adverse_open_wick_filter():
    """Stock with large adverse open wick (>0.40%) must be flagged as 2-way whipsaw risk."""
    # 5m candles: opens at 1000, dumps to 992 (-0.8% adverse wick), then rallies to 1015
    dates = pd.date_range("2026-11-17 09:15", periods=6, freq="5min")
    day5 = pd.DataFrame({
        "dt": dates,
        "hhmm": [d.strftime("%H:%M") for d in dates],
        "open": [1000.0, 995.0, 1002.0, 1008.0, 1012.0, 1014.0],
        "high": [1002.0, 1004.0, 1010.0, 1014.0, 1016.0, 1017.0],
        "low": [992.0, 993.0, 1000.0, 1006.0, 1010.0, 1013.0],  # 09:15 dipped to 992 (0.80% adverse wick)
        "close": [996.0, 1003.0, 1009.0, 1013.0, 1015.0, 1016.0],
        "volume": [50000.0] * 6,
    })

    res = _classify_option_timing(day5, "LONG", 1016.0, 1005.0, 1.60, atr14=20.0, atr_pct=2.0)
    assert res["adverse_open_wick"] >= 0.70
    assert res["option_phase"] == "CHOP_DECAY"
    assert "ADVERSE OPEN WICK" in res["option_label"]
    assert res["option_safe"] is False


def test_overexpansion_gate_blocks_chasing():
    """Stock already up +4.7% at midday running stretched from 9EMA must be WAIT_PULLBACK."""
    dates = pd.date_range("2026-11-17 09:15", periods=20, freq="5min")
    # Simulate LODHA-like run from 945 to 988 (+4.55%)
    prices = [945.0 + (i * 2.3) for i in range(20)]
    day5 = pd.DataFrame({
        "dt": dates,
        "hhmm": [d.strftime("%H:%M") for d in dates],
        "open": prices,
        "high": [p + 1.5 for p in prices],
        "low": [p - 0.5 for p in prices],
        "close": [p + 1.0 for p in prices],
        "volume": [30000.0] * 20,
    })

    ltp = float(day5["close"].iloc[-1])
    vwap = float(day5["close"].mean())
    move_open_pct = (ltp - 945.0) / 945.0 * 100.0  # +4.7%

    res = _classify_option_timing(day5, "LONG", ltp, vwap, move_open_pct, atr14=25.0, atr_pct=2.6)
    assert res["option_safe"] is False
    assert res["option_phase"] == "WAIT_PULLBACK"
    assert "EXTENDED" in res["option_label"] or "STRETCHED" in res["option_label"]


def test_compute_order_block_anchors_targets_and_flags_extended():
    """When price is >= R1 from base, status must be OB_EXTENDED, not OB_BREAKOUT_CONFIRMED."""
    dates = pd.date_range("2026-11-17 09:15", periods=15, freq="5min")
    day5 = pd.DataFrame({
        "dt": dates,
        "hhmm": [d.strftime("%H:%M") for d in dates],
        "open": [945.0] * 5 + [955.0] * 5 + [985.0] * 5,
        "high": [948.0] * 5 + [958.0] * 5 + [989.0] * 5,
        "low": [943.0] * 5 + [953.0] * 5 + [984.0] * 5,
        "close": [946.0] * 5 + [957.0] * 5 + [988.0] * 5,
        "volume": [20000.0] * 15,
    })
    hourly_ob = {"low": 943.0, "high": 955.0}
    c1h = [{"open": 945.0, "high": 955.0, "low": 943.0, "close": 950.0, "volume": 100000.0, "ts": 1731816000}]

    ob = compute_order_block(day5, "LONG", 988.0, 960.0, c1h=c1h)
    assert ob["status"] == "OB_EXTENDED"
    assert ob["first_break_hhmm"] is not None
    # Target R1 should be anchored near 955 + 2 * risk (~975-980), so 988 >= R1
    assert 988.0 >= ob["r1"]


def test_late_startup_extended_stock_suppresses_alerts():
    """When scanner runs for an already extended stock, do NOT fire any alerts (prevents chasing)."""
    db.clear_live_alerts("2026-11-17")

    extended_lodha = {
        "symbol": "LODHA",
        "direction": "LONG",
        "option_safe": False,
        "score": 75.0,
        "status_tag": "LIVE MOMENTUM",
        "ltp": 988.60,
        "vwap": 965.0,
        "move_pct": 4.70,
        "order_block": {
            "status": "OB_EXTENDED",
            "ob_zone": "₹944.00 - ₹955.00",
            "s1": 942.50,
            "r1": 978.0,
            "r2": 995.0,
            "first_break_hhmm": "09:30",
            "first_break_px": 956.0,
        },
        "option_strike": {
            "option_name": "LODHA 980 CE",
            "est_premium": 25.0,
            "est_capital_inr": 25000.0,
            "sl_premium": 15.0,
            "tp1_premium": 40.0,
        },
        "oi_build_up": "LONG BUILD-UP",
        "vol_delta_shares": "+12.50L",
        "reasons": ["1H OB Extended (R1 ₹978.0 reached) — Exhaustion Zone"],
    }

    try:
        # Evaluate alerts
        new_alerts = evaluate_and_record_alerts([extended_lodha], "2026-11-17")

        # Must NOT fire any alerts for already-extended stocks
        assert len(new_alerts) == 0
    finally:
        with db.connect() as conn:
            conn.execute("DELETE FROM live_alerts WHERE symbol = 'LODHA' AND trade_date = '2026-11-17'")
            conn.commit()


def test_strike_oi_wall_blocks_breakout_and_bull_trap_triggers_exit():
    """Verify strike resistance blocks LONG breakouts and Bull Trap triggers fast exit."""
    db.clear_live_alerts("2026-11-20")

    # Item trapped under 1220 strike with strike_barrier_blocked = True
    trapped_item = {
        "symbol": "360ONE",
        "direction": "LONG",
        "option_safe": False,
        "score": 75.0,
        "status_tag": "LIVE MOMENTUM",
        "ltp": 1219.3,
        "vwap": 1214.0,
        "move_pct": 2.1,
        "strike_barrier_blocked": True,
        "strike_barrier_reason": "Trapped under ₹1,220 Call Strike (0.70 pts away) with multiple rejections",
        "order_block": {
            "status": "OB_STRIKE_RESISTANCE",
            "ob_zone": "₹1,191.55 - ₹1,216.00",
            "s1": 1189.76,
            "r1": 1268.48,
        },
        "option_strike": {
            "option_name": "360ONE 1240 CE",
            "est_premium": 22.6,
            "est_capital_inr": 11300.0,
            "sl_premium": 17.6,
            "tp1_premium": 32.8,
        },
        "oi_build_up": "LONG BUILD-UP",
        "vol_delta_shares": "+0.22L",
        "reasons": ["⛔ STRIKE RESISTANCE: Trapped under ₹1,220 Call Strike"],
    }

    alerts = evaluate_and_record_alerts([trapped_item], "2026-11-20")
    # Must NOT fire alert because strike wall is blocked!
    assert len(alerts) == 0

    # If an alert was previously active, a VWAP breakdown must trigger fast exit
    fake_prior_alert = {
        "alert_id": "ALT_360ONE_20261120_103634_OB_BREAKOUT",
        "trade_date": "2026-11-20",
        "timestamp_ist": "10:36:34 AM",
        "epoch_ts": 1795151194,
        "symbol": "360ONE",
        "direction": "LONG",
        "event_type": "OB_BREAKOUT",
        "price": 1219.3,
        "vwap": 1214.0,
        "move_pct": 2.1,
        "s1": 1189.76,
        "r1": 1268.48,
    }
    db.save_live_alert(fake_prior_alert)

    # Next tick: Price loses VWAP (1213.5 vs vwap 1215.0)
    reversal_item = dict(trapped_item)
    reversal_item["ltp"] = 1213.5
    reversal_item["vwap"] = 1215.0
    reversal_item["strike_barrier_blocked"] = False

    exit_alerts = evaluate_and_record_alerts([reversal_item], "2026-11-20")
    assert len(exit_alerts) == 1
    assert exit_alerts[0]["event_type"] in ("VWAP_BREAKDOWN_EXIT", "BULL_TRAP_EXIT", "BREAKOUT_FAILED_EXIT")


def test_htf_bsl_breakout_with_pattern_and_volume_triggers_alert():
    """Verify Daily BSL Breakout confirmed by Bullish Engulfing & RVOL surge generates valid alert."""
    db.clear_live_alerts("2026-11-20")

    hindalco_item = {
        "symbol": "HINDALCO",
        "direction": "LONG",
        "option_safe": True,
        "score": 86.0,
        "status_tag": "CREAM_BULL_V2",
        "ltp": 912.15,
        "vwap": 909.50,
        "move_pct": 2.2,
        "candle_pattern": "Bullish Engulfing",
        "rvol": 1.6,
        "net_cvd_shares": 350000.0,
        "order_block": {
            "status": "OB_BREAKOUT_CONFIRMED",
            "ob_zone": "₹910.00",
            "htf_level_name": "Daily BSL (PDH ₹910.00)",
            "htf_level_px": 910.0,
            "s1": 905.0,
            "r1": 940.0,
            "r2": 955.0,
        },
        "opt_chain_sr": {
            "max_call_oi_strike": 950,
            "max_put_oi_strike": 880,
            "dist_to_ce_pts": 37.85,
            "dist_to_pe_pts": 32.15,
            "option_chain_sr": "Max CE ₹950 · Max PE ₹880",
        },
        "option_strike": {
            "option_name": "HINDALCO 920 CE",
            "est_premium": 18.5,
            "est_capital_inr": 18500.0,
            "sl_premium": 14.0,
            "tp1_premium": 27.0,
        },
        "oi_build_up": "LONG BUILD-UP",
        "vol_delta_shares": "+3.50L",
        "reasons": ["Daily BSL Breakout", "VWAP Hold"],
    }

    alerts = evaluate_and_record_alerts([hindalco_item], "2026-11-20")
    assert len(alerts) == 1
    alt = alerts[0]
    assert alt["event_type"] == "OB_BREAKOUT"
    assert "Daily BSL" in alt["htf_level_name"]
    assert alt["candle_pattern"] == "Bullish Engulfing"
    assert alt["rvol"] == 1.6
    assert alt["max_call_oi_strike"] == 950
    assert alt["reward_pts"] >= 20.0
    # Reward = 940 - 912.15 = 27.85 pts, Risk = 912.15 - 905 = 7.15 pts (RR ~ 3.9:1)
    assert float(alt["rr_ratio"].split(":")[1]) >= 1.4


def test_bull_trap_upper_wick_rejection_blocks_alert():
    """Candle with heavy upper wick (>=35%) or shooting star must be blocked from LONG entry."""
    db.clear_live_alerts("2026-11-20")

    shooting_star_item = {
        "symbol": "HINDALCO",
        "direction": "LONG",
        "option_safe": True,
        "score": 82.0,
        "status_tag": "CREAM_BULL_V2",
        "ltp": 912.0,
        "vwap": 910.0,
        "move_pct": 2.1,
        "candle_pattern": "Shooting Star (Bearish Pin) [48% Wick]",
        "rvol": 1.4,
        "net_cvd_shares": 100000.0,
        "order_block": {
            "status": "OB_BREAKOUT_CONFIRMED",
            "ob_zone": "₹910.00",
            "s1": 905.0,
            "r1": 940.0,
        },
    }

    alerts = evaluate_and_record_alerts([shooting_star_item], "2026-11-20")
    assert len(alerts) == 0


def test_option_chain_call_wall_insufficient_headroom_blocks_alert():
    """Stock trapped < 20 pts from Max Call OI resistance must be blocked (e.g. 360ONE @ 1219.3 vs 1220 CE)."""
    db.clear_live_alerts("2026-11-20")

    trapped_call_wall = {
        "symbol": "360ONE",
        "direction": "LONG",
        "option_safe": True,
        "score": 80.0,
        "status_tag": "CREAM_BULL_V2",
        "ltp": 1219.3,
        "vwap": 1215.0,
        "move_pct": 2.0,
        "candle_pattern": "Bullish Pin Bar",
        "rvol": 1.5,
        "net_cvd_shares": 50000.0,
        "order_block": {
            "status": "OB_BREAKOUT_CONFIRMED",
            "ob_zone": "₹1,215.00",
            "s1": 1210.0,
            "r1": 1240.0,
        },
        "opt_chain_sr": {
            "max_call_oi_strike": 1220,  # Only 0.70 pts headroom!
            "max_put_oi_strike": 1180,
            "dist_to_ce_pts": 0.70,
        },
    }

    alerts = evaluate_and_record_alerts([trapped_call_wall], "2026-11-20")
    # Must be blocked by Option Chain Resistance Wall!
    assert len(alerts) == 0


def test_sluggish_volume_blocks_alert():
    """Breakout without volume expansion (RVOL < 0.70x and no CVD surge) must be suppressed."""
    db.clear_live_alerts("2026-11-20")

    sluggish_item = {
        "symbol": "HINDALCO",
        "direction": "LONG",
        "option_safe": True,
        "score": 78.0,
        "status_tag": "CREAM_BULL_V2",
        "ltp": 912.0,
        "vwap": 910.0,
        "move_pct": 2.1,
        "candle_pattern": "Bullish Displacement",
        "rvol": 0.45,  # Anemic dry volume
        "net_cvd_shares": -20000.0,
        "order_block": {
            "status": "OB_BREAKOUT_CONFIRMED",
            "ob_zone": "₹910.00",
            "s1": 905.0,
            "r1": 940.0,
        },
    }

    alerts = evaluate_and_record_alerts([sluggish_item], "2026-11-20")
    assert len(alerts) == 0


def test_inside_htf_range_strictly_blocks_alerts():
    """Stocks merely consolidating inside HTF range (no BSL/SSL touch) must NEVER fire alerts."""
    db.clear_live_alerts("2026-11-20")

    inside_range_item = {
        "symbol": "TCS",
        "direction": "LONG",
        "option_safe": True,
        "score": 85.0,
        "status_tag": "CREAM_BULL_V2",
        "ltp": 4150.0,
        "vwap": 4140.0,
        "move_pct": 1.5,
        "candle_pattern": "Bullish Engulfing",
        "rvol": 1.5,
        "net_cvd_shares": 50000.0,
        "order_block": {
            "status": "OB_BREAKOUT_CONFIRMED",
            "ob_zone": "₹4,140.00",
            "htf_level_name": "Inside HTF Range (BSL ₹4,220.00 · SSL ₹4,080.00)",
            "s1": 4120.0,
            "r1": 4220.0,
        },
    }

    alerts = evaluate_and_record_alerts([inside_range_item], "2026-11-20")
    assert len(alerts) == 0


def test_strict_no_reentry_after_target_or_exit():
    """Once Target R1 or Exit triggers, symbol is closed for the day — no subsequent alerts!"""
    db.clear_live_alerts("2026-11-20")

    entry_item = {
        "symbol": "INFY",
        "direction": "LONG",
        "option_safe": True,
        "score": 88.0,
        "status_tag": "CREAM_BULL_V2",
        "ltp": 1850.0,
        "vwap": 1840.0,
        "move_pct": 2.2,
        "candle_pattern": "Bullish Engulfing",
        "rvol": 1.6,
        "net_cvd_shares": 100000.0,
        "order_block": {
            "status": "OB_BREAKOUT_CONFIRMED",
            "ob_zone": "₹1,845.00",
            "htf_level_name": "Daily BSL (PDH ₹1,845.00)",
            "s1": 1835.0,
            "r1": 1875.0,
        },
    }

    # 1. Entry triggers
    alts1 = evaluate_and_record_alerts([entry_item], "2026-11-20")
    assert len(alts1) == 1
    assert alts1[0]["event_type"] == "OB_BREAKOUT"

    # 2. Target R1 hit
    hit_item = dict(entry_item)
    hit_item["ltp"] = 1878.0
    alts2 = evaluate_and_record_alerts([hit_item], "2026-11-20")
    assert len(alts2) == 1
    assert alts2[0]["event_type"] == "TARGET_R1_HIT"

    # 3. Subsequent breakout tick — STRICT NO RE-ENTRY: 0 alerts!
    reentry_attempt = dict(entry_item)
    reentry_attempt["ltp"] = 1890.0
    alts3 = evaluate_and_record_alerts([reentry_attempt], "2026-11-20")
    assert len(alts3) == 0


def test_sweep_reversal_generates_sweep_reversal_alert():
    """Daily SSL Sweep with bullish confirmation generates SWEEP_REVERSAL event."""
    db.clear_live_alerts("2026-11-20")

    sweep_item = {
        "symbol": "AXISBANK",
        "direction": "LONG",
        "option_safe": True,
        "score": 85.0,
        "status_tag": "CREAM_BULL_V2",
        "ltp": 1152.0,
        "vwap": 1150.0,
        "move_pct": 1.8,
        "candle_pattern": "Bullish Pin Bar",
        "rvol": 1.4,
        "net_cvd_shares": 80000.0,
        "order_block": {
            "status": "OB_RETEST_BOUNCE",
            "ob_zone": "₹1,148.00",
            "htf_level_name": "Daily SSL Sweep (PDL ₹1,148.00)",
            "s1": 1144.0,
            "r1": 1175.0,
        },
    }

    alts = evaluate_and_record_alerts([sweep_item], "2026-11-20")
    assert len(alts) == 1
    assert alts[0]["event_type"] == "SWEEP_REVERSAL"
    assert "Sweep & Reversal" in alts[0]["reasons"][0]


def test_100_session_major_swing_pivot_detection():
    """Verify that 100-session scan detects unmitigated turning points that occurred >20 days ago."""
    from app.liquidity_engine import find_major_swings_100d

    # Synthetic 50-day series: day 15 has a major peak at 1120.0 that dropped to 1050 (-6.25%)
    n = 50
    highs = [1000.0 + i for i in range(n)]
    lows = [h - 8.0 for h in highs]
    closes = [h - 4.0 for h in highs]

    # Peak on day 15 (which is n - 1 - 15 = 34 days ago from end)
    highs[15] = 1120.0
    lows[15] = 1090.0
    closes[15] = 1110.0
    # Following 5 bars drop heavily to 1040 (displacement)
    for k in range(16, 22):
        highs[k] = 1060.0
        lows[k] = 1040.0
        closes[k] = 1045.0
    # Subsequent bars stay below 1120 (unmitigated)
    for k in range(22, n):
        highs[k] = 1060.0 + (k - 22) * 0.5
        lows[k] = highs[k] - 6.0
        closes[k] = highs[k] - 3.0

    df_synth = pd.DataFrame({"high": highs, "low": lows, "close": closes})
    cmp_px = float(closes[-1]) # ~1071.0

    nearest_bsl, nearest_ssl = find_major_swings_100d(df_synth, cmp_px, min_reversal_pct=0.020)
    assert nearest_bsl is not None
    assert nearest_bsl[0] == 1120.0
    assert "D-34 Major Swing High" in nearest_bsl[1]


def test_nearest_barrier_priority_rule_for_targets():
    """Verify Target R1 anchors to the nearest obstacle (min of overhead barriers for Long)."""
    # 5m candles consolidating at 1045, breaking out above 1050
    dates = pd.date_range("2026-11-20 09:15", periods=5, freq="5min")
    day5 = pd.DataFrame({
        "dt": dates,
        "hhmm": [d.strftime("%H:%M") for d in dates],
        "open": [1045.0, 1046.0, 1048.0, 1050.0, 1052.0],
        "high": [1048.0, 1049.0, 1051.0, 1053.0, 1055.0],
        "low": [1044.0, 1045.0, 1047.0, 1049.0, 1051.0],
        "close": [1046.0, 1048.0, 1050.0, 1052.0, 1054.0],
        "volume": [25000.0] * 5,
    })

    # Obstacles: Major Swing High at 1075, Max Call OI at 1090, 100D Macro High at 1150
    htf = {
        "daily_bsl": 1050.0,
        "daily_ssl": 1030.0,
        "weekly_bsl": 1065.0,
        "weekly_ssl": 1020.0,
        "swing_bsl_20d": 1070.0,
        "swing_ssl_20d": 1015.0,
        "major_bsl_px": 1075.0,
        "major_bsl_desc": "D-42 Major Swing High",
        "swing_bsl_100d": 1150.0,
    }
    opt = {
        "max_call_oi_strike": 1090.0,
        "max_put_oi_strike": 1000.0,
    }

    ob = compute_order_block(day5, "LONG", 1054.0, 1049.0, htf_levels=htf, opt_chain_sr=opt)
    # Target R1 should be anchored near nearest overhead obstacle (weekly_bsl 1065 / swing_bsl_20d 1070), NOT 1150!
    assert ob["r1"] <= 1075.0
    assert ob["status"] in ("OB_BREAKOUT_CONFIRMED", "OB_RETEST_BOUNCE")

