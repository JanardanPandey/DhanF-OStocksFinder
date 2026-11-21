from app.database import db
import pandas as pd
import datetime

tz_ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
start_dt = datetime.datetime(2026, 11, 18, 9, 15, tzinfo=tz_ist)
end_dt = datetime.datetime(2026, 11, 18, 15, 30, tzinfo=tz_ist)

with db.connect() as conn:
    rows = conn.execute(
        "SELECT ts, open, high, low, close, volume FROM candles WHERE symbol='ADANIPORTS' AND timeframe='5m' AND ts >= ? AND ts <= ? ORDER BY ts ASC",
        (int(start_dt.timestamp()), int(end_dt.timestamp()))
    ).fetchall()

df = pd.DataFrame([dict(r) for r in rows])
df["datetime"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
df["time"] = df["datetime"].dt.strftime("%H:%M")

print("ADANIPORTS Key Inflection Points:")
for idx, r in df.iterrows():
    if r["time"] in ["09:15", "09:25", "09:30", "09:40", "10:30", "11:00", "11:30", "12:00", "12:30", "13:00", "13:30", "14:00", "14:30", "15:00", "15:25"]:
        print(f"{r['time']} | O: {r['open']:.2f} | H: {r['high']:.2f} | L: {r['low']:.2f} | C: {r['close']:.2f} | Vol: {int(r['volume'])}")
