import datetime
from app.database import db
from app.liquidity_engine import get_htf_liquidity_levels
import pandas as pd

def run_smart_backtest(trade_date_str):
    trade_date = datetime.date.fromisoformat(trade_date_str)
    tz_ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    start_dt = datetime.datetime(trade_date.year, trade_date.month, trade_date.day, 9, 15, tzinfo=tz_ist)
    end_dt = datetime.datetime(trade_date.year, trade_date.month, trade_date.day, 15, 30, tzinfo=tz_ist)
    start_epoch = int(start_dt.timestamp())
    end_epoch = int(end_dt.timestamp())

    with db.connect() as conn:
        symbols = [r[0] for r in conn.execute(
            "SELECT DISTINCT symbol FROM candles WHERE timeframe = '5m' AND ts >= ? AND ts <= ?",
            (start_epoch, end_epoch)
        ).fetchall()]

    print(f"\n==========================================")
    print(f"ANALYSIS FOR {trade_date_str} ({len(symbols)} symbols)")
    print(f"==========================================")

    # Check POLYCAB specifically
    if "POLYCAB" in symbols:
        levels = get_htf_liquidity_levels("POLYCAB", trade_date)
        with db.connect() as conn:
            c5 = conn.execute(
                "SELECT ts, open, high, low, close, volume FROM candles WHERE symbol = 'POLYCAB' AND timeframe = '5m' AND ts >= ? AND ts <= ? ORDER BY ts ASC",
                (start_epoch, end_epoch)
            ).fetchall()
        df5 = pd.DataFrame([dict(r) for r in c5])
        df5["datetime"] = pd.to_datetime(df5["ts"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
        df5["hhmm"] = df5["datetime"].dt.strftime("%H:%M")
        day_open = df5.iloc[0]["open"]
        pdl = levels["daily_ssl"]
        pwl = levels["weekly_ssl"]
        print(f"POLYCAB: Day Open: {day_open} | PDL: {pdl} (Gap: {round(((day_open-pdl)/pdl)*100, 2)}%) | PWL: {pwl}")
        print(f"Candle 09:20: O={df5.iloc[1]['open']}, H={df5.iloc[1]['high']}, L={df5.iloc[1]['low']}, C={df5.iloc[1]['close']}")
        print(f"Candle 09:25: O={df5.iloc[2]['open']}, H={df5.iloc[2]['high']}, L={df5.iloc[2]['low']}, C={df5.iloc[2]['close']}")

run_smart_backtest("2026-11-18")
