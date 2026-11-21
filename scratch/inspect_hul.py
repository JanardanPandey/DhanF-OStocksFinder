from app.database import db
from app.liquidity_engine import get_htf_liquidity_levels
import pandas as pd
import datetime

trade_date = datetime.date(2026, 11, 19)
levels = get_htf_liquidity_levels("HINDUNILVR", trade_date)
print("HINDUNILVR HTF Levels Today:", levels)

tz_ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
start_epoch = int(datetime.datetime(2026, 11, 19, 9, 15, tzinfo=tz_ist).timestamp())
end_epoch = int(datetime.datetime(2026, 11, 19, 15, 30, tzinfo=tz_ist).timestamp())

with db.connect() as conn:
    rows = conn.execute(
        "SELECT ts, open, high, low, close, volume FROM candles WHERE symbol='HINDUNILVR' AND timeframe='5m' AND ts >= ? AND ts <= ? ORDER BY ts ASC",
        (start_epoch, end_epoch)
    ).fetchall()

df = pd.DataFrame([dict(r) for r in rows])
df["datetime"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
df["time"] = df["datetime"].dt.strftime("%H:%M")

cum_vol = df["volume"].cumsum()
cum_vol_px = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3.0).cumsum()
df["vwap"] = (cum_vol_px / cum_vol.replace(0, 1)).round(2)

print("\nHINDUNILVR 5m Candles Today (19-Nov-2026):")
for idx, r in df.iterrows():
    if "10:30" <= r["time"] <= "11:55":
        print(f"{r['time']} | O: {r['open']:.2f} | H: {r['high']:.2f} | L: {r['low']:.2f} | C: {r['close']:.2f} | Vol: {int(r['volume'])} | VWAP: {r['vwap']}")
