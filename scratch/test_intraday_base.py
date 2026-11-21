import asyncio
from app.database import db
from app.playbook_engine import _candles_to_df
import numpy as np
import datetime
import pandas as pd

tz_ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
today_epoch = int(datetime.datetime(2026, 11, 20, 9, 15, tzinfo=tz_ist).timestamp())

symbols = ["360ONE", "HDFCBANK", "HEROMOTOCO", "KOTAKBANK", "NESTLEIND", "BHARATFORG"]
for sym in symbols:
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT ts, open, high, low, close, volume FROM candles WHERE symbol=? AND timeframe='5m' AND ts >= ? ORDER BY ts ASC",
            (sym, today_epoch)
        ).fetchall()
    if not rows:
        continue
    df = pd.DataFrame([dict(r) for r in rows])
    l_b = df["low"].to_numpy(dtype=float)
    h_b = df["high"].to_numpy(dtype=float)
    c_b = df["close"].to_numpy(dtype=float)
    ltp = c_b[-1]
    
    # Intraday opening shelf (first 3-6 bars)
    shelf_len = min(6, max(3, len(df)))
    intraday_base_low = float(np.min(l_b[:shelf_len]))
    intraday_base_high = float(np.max(h_b[:shelf_len]))
    
    # Recent 5m swing low (last 6 bars)
    recent_swing_low = float(np.min(l_b[-6:]))
    recent_swing_high = float(np.max(h_b[-6:]))
    
    intraday_risk = abs(ltp - intraday_base_low)
    intraday_risk_pct = (intraday_risk / ltp) * 100.0
    
    swing_risk = abs(ltp - recent_swing_low)
    swing_risk_pct = (swing_risk / ltp) * 100.0
    
    print(f"{sym} | LTP: {ltp:.2f}")
    print(f"  Intraday Base: Low={intraday_base_low:.2f}, High={intraday_base_high:.2f} | Risk={intraday_risk:.2f} ({intraday_risk_pct:.2f}%)")
    print(f"  Recent Swing:  Low={recent_swing_low:.2f}, High={recent_swing_high:.2f} | Risk={swing_risk:.2f} ({swing_risk_pct:.2f}%)")
