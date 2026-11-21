from app.database import db
import pandas as pd
import datetime

trade_date = datetime.date(2026, 11, 18)
tz_ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
start_epoch = int(datetime.datetime(2026, 11, 18, 9, 15, tzinfo=tz_ist).timestamp())
end_epoch = int(datetime.datetime(2026, 11, 18, 15, 30, tzinfo=tz_ist).timestamp())

with db.connect() as conn:
    c5 = conn.execute(
        "SELECT ts, open, high, low, close, volume FROM candles WHERE symbol = 'MCX' AND timeframe = '5m' AND ts >= ? AND ts <= ? ORDER BY ts ASC",
        (start_epoch, end_epoch)
    ).fetchall()
df5 = pd.DataFrame([dict(r) for r in c5])
df5["datetime"] = pd.to_datetime(df5["ts"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
df5["hhmm"] = df5["datetime"].dt.strftime("%H:%M")

cum_vol = df5["volume"].cumsum()
cum_vol_px = (df5["volume"] * (df5["high"] + df5["low"] + df5["close"]) / 3.0).cumsum()
df5["vwap"] = (cum_vol_px / cum_vol.replace(0, 1)).round(2)

print("MCX Morning Candles:")
for i in range(8):
    r = df5.iloc[i]
    print(f"{r['hhmm']} | O: {r['open']} | H: {r['high']} | L: {r['low']} | C: {r['close']} | VWAP: {r['vwap']}")
