import asyncio
from app.database import db
from app.playbook_engine import compute_order_block, evaluate_and_record_alerts, IST
import pandas as pd
import datetime

tz_ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
today_epoch = int(datetime.datetime(2026, 11, 20, 9, 15, tzinfo=tz_ist).timestamp())

symbols = ["360ONE", "HDFCBANK", "HEROMOTOCO", "KOTAKBANK", "NESTLEIND", "BHARATFORG"]
mock_board = []

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
        c1h = conn.execute(
            "SELECT ts, open, high, low, close, volume FROM candles WHERE symbol=? AND timeframe='1h' ORDER BY ts ASC LIMIT 60",
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
    
    # Compute OB without hourly overwriting intraday base
    shelf_len = min(6, max(3, len(df)))
    base_low = float(df["low"].iloc[:shelf_len].min())
    base_high = float(df["high"].iloc[:shelf_len].max())
    
    # Let's test compute_order_block with intraday base
    ob_res = compute_order_block(df, "LONG", ltp, vwap, [dict(r) for r in c1d], [dict(r) for r in c1h])
    print(f"\n{sym}: Status={ob_res['status']} | OB Zone={ob_res['ob_zone']} | S1={ob_res['s1']} | R1={ob_res['r1']}")
