import asyncio
from app.database import db
from app.playbook_engine import compute_order_block, evaluate_and_record_alerts, IST
import pandas as pd
import numpy as np
import datetime

tz_ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
today_epoch = int(datetime.datetime(2026, 11, 20, 9, 15, tzinfo=tz_ist).timestamp())

symbols = ["360ONE", "HDFCBANK", "HEROMOTOCO", "KOTAKBANK", "NESTLEIND", "BHARATFORG"]

for sym in symbols:
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT ts, open, high, low, close, volume FROM candles WHERE symbol=? AND timeframe='5m' AND ts >= ? ORDER BY ts ASC",
            (sym, today_epoch)
        ).fetchall()
        c1d = conn.execute(
            "SELECT ts, open, high, low, close, volume FROM candles WHERE symbol=? AND timeframe='1D' ORDER BY ts ASC LIMIT 60",
            (sym,)
        ).fetchall()

    if not rows:
        continue
    df = pd.DataFrame([dict(r) for r in rows])
    df["datetime"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
    df["hhmm"] = df["datetime"].dt.strftime("%H:%M")
    ltp = float(df.iloc[-1]["close"])
    cum_vol = df["volume"].cumsum()
    cum_vol_px = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3.0).cumsum()
    vwap = float((cum_vol_px / cum_vol.replace(0, 1)).iloc[-1])
    
    # Intraday 5m base:
    shelf_len = min(6, max(3, len(df)))
    base_low = float(df["low"].iloc[:shelf_len].min())
    base_high = float(df["high"].iloc[:shelf_len].max())
    
    direction = "LONG"
    s1 = round(base_low * 0.9985, 2)
    base_risk = max(base_high * 0.005, base_high - s1)
    r1 = round(base_high + base_risk * 2.0, 2)
    r2 = round(base_high + base_risk * 3.5, 2)
    
    min_break_dist = max(0.02, ltp * 0.0025)
    c_b = df["close"].to_numpy(dtype=float)
    is_breakout = (c_b[-1] >= base_high + min_break_dist) and (c_b[-1] > vwap)
    is_forming = base_low <= ltp <= base_high
    is_extended = (ltp >= r1) or (ltp > base_high * 1.025)
    
    status = "OB_EXTENDED" if is_extended else ("OB_BREAKOUT_CONFIRMED" if is_breakout else "OB_ACTIVE")
    
    reward_pts = abs(r1 - ltp)
    risk_pts = abs(ltp - s1)
    
    print(f"\n{sym}: Status={status} | LTP={ltp} | Base=[{base_low:.2f}, {base_high:.2f}] | S1={s1} | R1={r1}")
    print(f"  Risk: {risk_pts:.2f} ({risk_pts/ltp*100:.2f}%) | Reward: {reward_pts:.2f} | RR: {reward_pts/max(0.1, risk_pts):.2f}")
