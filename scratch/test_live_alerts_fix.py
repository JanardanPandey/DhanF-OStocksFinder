import asyncio
from app.database import db
from app.playbook_engine import (
    _candles_to_df, _resample_5m_records, is_valid_fno_symbol, IST,
    recommend_option_strike, _compute_volume_footprint
)
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

tz_ist = timezone(timedelta(hours=5, minutes=30))
today_epoch = int(datetime(2026, 11, 20, 9, 15, tzinfo=tz_ist).timestamp())

with db.connect() as conn:
    syms = [r[0] for r in conn.execute(
        "SELECT DISTINCT symbol FROM candles WHERE timeframe='5m' AND ts >= ?",
        (today_epoch,)
    ).fetchall() if is_valid_fno_symbol(r[0])]

print(f"Total symbols with 5m candles today: {len(syms)}")

alerts_found = []

for sym in syms:
    with db.connect() as conn:
        c5 = conn.execute(
            "SELECT ts, open, high, low, close, volume FROM candles WHERE symbol=? AND timeframe='5m' AND ts >= ? ORDER BY ts ASC",
            (sym, today_epoch)
        ).fetchall()
        c1d = conn.execute(
            "SELECT ts, open, high, low, close, volume FROM candles WHERE symbol=? AND timeframe='1D' ORDER BY ts ASC LIMIT 60",
            (sym,)
        ).fetchall()

    if len(c5) < 3:
        continue

    df5 = pd.DataFrame([dict(r) for r in c5])
    df5["datetime"] = pd.to_datetime(df5["ts"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
    df5["hhmm"] = df5["datetime"].dt.strftime("%H:%M")
    
    l_b = df5["low"].to_numpy(dtype=float)
    h_b = df5["high"].to_numpy(dtype=float)
    c_b = df5["close"].to_numpy(dtype=float)
    ltp = c_b[-1]
    
    cum_vol = df5["volume"].cumsum()
    cum_vol_px = (df5["volume"] * (df5["high"] + df5["low"] + df5["close"]) / 3.0).cumsum()
    vwap = float((cum_vol_px / cum_vol.replace(0, 1)).iloc[-1])
    
    # Intraday 5m base
    shelf_len = min(6, max(3, len(df5)))
    base_low = float(np.min(l_b[:shelf_len]))
    base_high = float(np.max(h_b[:shelf_len]))
    
    min_break_dist = max(0.02, ltp * 0.0025)
    
    # Check Long Breakout
    is_long_breakout = (c_b[-1] >= base_high + min_break_dist) and (c_b[-1] > vwap)
    # Check Short Breakdown
    is_short_breakdown = (c_b[-1] <= base_low - min_break_dist) and (c_b[-1] < vwap)
    
    if is_long_breakout:
        direction = "LONG"
        s1 = round(base_low * 0.9985, 2)
        base_risk = max(base_high * 0.005, base_high - s1)
        r1 = round(base_high + base_risk * 2.0, 2)
        r2 = round(base_high + base_risk * 3.5, 2)
        risk_pts = round(ltp - s1, 2)
        reward_pts = round(r1 - ltp, 2)
        
        if 0 < risk_pts <= 0.025 * ltp and reward_pts >= (20.0 if ltp >= 1000 else ltp * 0.01):
            alerts_found.append({
                "symbol": sym,
                "dir": "LONG",
                "event": "OB_BREAKOUT",
                "ltp": ltp,
                "s1": s1,
                "r1": r1,
                "risk": risk_pts,
                "reward": reward_pts,
                "rr": round(reward_pts / risk_pts, 2),
                "time": df5["hhmm"].iloc[-1],
            })
    elif is_short_breakdown:
        direction = "SHORT"
        s1 = round(base_high * 1.0015, 2)
        base_risk = max(base_low * 0.005, s1 - base_low)
        r1 = round(base_low - base_risk * 2.0, 2)
        r2 = round(base_low - base_risk * 3.5, 2)
        risk_pts = round(s1 - ltp, 2)
        reward_pts = round(ltp - r1, 2)
        
        if 0 < risk_pts <= 0.025 * ltp and reward_pts >= (20.0 if ltp >= 1000 else ltp * 0.01):
            alerts_found.append({
                "symbol": sym,
                "dir": "SHORT",
                "event": "OB_BREAKDOWN",
                "ltp": ltp,
                "s1": s1,
                "r1": r1,
                "risk": risk_pts,
                "reward": reward_pts,
                "rr": round(reward_pts / risk_pts, 2),
                "time": df5["hhmm"].iloc[-1],
            })

print(f"\nAlerts Found for Today ({len(alerts_found)}):")
for a in alerts_found:
    print(f"[{a['time']} IST] {a['symbol']} | {a['dir']} ({a['event']}) | LTP: ₹{a['ltp']} | S1: ₹{a['s1']} | R1: ₹{a['r1']} | RR: 1:{a['rr']}")
