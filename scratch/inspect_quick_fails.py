from app.database import db
from app.liquidity_engine import get_htf_liquidity_levels
import pandas as pd
import datetime

trade_date = datetime.date(2026, 11, 18)
tz_ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
start_epoch = int(datetime.datetime(2026, 11, 18, 9, 15, tzinfo=tz_ist).timestamp())
end_epoch = int(datetime.datetime(2026, 11, 18, 15, 30, tzinfo=tz_ist).timestamp())

symbols_to_check = ["SUPREMEIND", "ANGELONE", "ADANIGREEN", "ICICIGI", "ABB", "BHARTIARTL", "M&M", "NAVINFLUOR"]

for sym in symbols_to_check:
    levels = get_htf_liquidity_levels(sym, trade_date)
    with db.connect() as conn:
        c5 = conn.execute(
            "SELECT ts, open, high, low, close, volume FROM candles WHERE symbol = ? AND timeframe = '5m' AND ts >= ? AND ts <= ? ORDER BY ts ASC",
            (sym, start_epoch, end_epoch)
        ).fetchall()
    df = pd.DataFrame([dict(r) for r in c5])
    df["datetime"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
    df["time"] = df["datetime"].dt.strftime("%H:%M")
    
    # Calculate VWAP
    cum_vol = df["volume"].cumsum()
    cum_vol_px = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3.0).cumsum()
    df["vwap"] = (cum_vol_px / cum_vol.replace(0, 1)).round(2)
    
    print(f"\n=======================================================")
    print(f"{sym} | PDL: {levels['daily_ssl']} | PDH: {levels['daily_bsl']} | PWL: {levels['weekly_ssl']} | PWH: {levels['weekly_bsl']}")
    print(f"=======================================================")
    # Print around entry time
    for idx, r in df.iterrows():
        if sym == "SUPREMEIND" and "10:45" <= r["time"] <= "11:15":
            print(f"{r['time']} | O: {r['open']:.2f} | H: {r['high']:.2f} | L: {r['low']:.2f} | C: {r['close']:.2f} | Vol: {int(r['volume'])} | VWAP: {r['vwap']}")
        elif sym == "ANGELONE" and "11:50" <= r["time"] <= "12:15":
            print(f"{r['time']} | O: {r['open']:.2f} | H: {r['high']:.2f} | L: {r['low']:.2f} | C: {r['close']:.2f} | Vol: {int(r['volume'])} | VWAP: {r['vwap']}")
        elif sym == "ADANIGREEN" and "12:00" <= r["time"] <= "12:35":
            print(f"{r['time']} | O: {r['open']:.2f} | H: {r['high']:.2f} | L: {r['low']:.2f} | C: {r['close']:.2f} | Vol: {int(r['volume'])} | VWAP: {r['vwap']}")
        elif sym == "ICICIGI" and "10:10" <= r["time"] <= "10:45":
            print(f"{r['time']} | O: {r['open']:.2f} | H: {r['high']:.2f} | L: {r['low']:.2f} | C: {r['close']:.2f} | Vol: {int(r['volume'])} | VWAP: {r['vwap']}")
        elif sym == "ABB" and "11:40" <= r["time"] <= "12:20":
            print(f"{r['time']} | O: {r['open']:.2f} | H: {r['high']:.2f} | L: {r['low']:.2f} | C: {r['close']:.2f} | Vol: {int(r['volume'])} | VWAP: {r['vwap']}")
        elif sym == "BHARTIARTL" and "09:40" <= r["time"] <= "10:20":
            print(f"{r['time']} | O: {r['open']:.2f} | H: {r['high']:.2f} | L: {r['low']:.2f} | C: {r['close']:.2f} | Vol: {int(r['volume'])} | VWAP: {r['vwap']}")
        elif sym == "M&M" and "12:25" <= r["time"] <= "13:05":
            print(f"{r['time']} | O: {r['open']:.2f} | H: {r['high']:.2f} | L: {r['low']:.2f} | C: {r['close']:.2f} | Vol: {int(r['volume'])} | VWAP: {r['vwap']}")
        elif sym == "NAVINFLUOR" and "09:45" <= r["time"] <= "10:35":
            print(f"{r['time']} | O: {r['open']:.2f} | H: {r['high']:.2f} | L: {r['low']:.2f} | C: {r['close']:.2f} | Vol: {int(r['volume'])} | VWAP: {r['vwap']}")
