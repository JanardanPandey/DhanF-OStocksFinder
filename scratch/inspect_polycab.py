from app.database import db
from app.liquidity_engine import get_htf_liquidity_levels
import pandas as pd
import datetime

trade_date = datetime.date(2026, 11, 18)
levels = get_htf_liquidity_levels("POLYCAB", trade_date)
print("POLYCAB HTF Levels on 18-Nov-2026:")
for k, v in levels.items():
    print(f"  {k}: {v}")

tz_ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
start_dt = datetime.datetime(2026, 11, 18, 9, 15, tzinfo=tz_ist)
end_dt = datetime.datetime(2026, 11, 18, 15, 30, tzinfo=tz_ist)

with db.connect() as conn:
    rows = conn.execute(
        "SELECT ts, open, high, low, close, volume FROM candles WHERE symbol='POLYCAB' AND timeframe='5m' AND ts >= ? AND ts <= ? ORDER BY ts ASC",
        (int(start_dt.timestamp()), int(end_dt.timestamp()))
    ).fetchall()

df = pd.DataFrame([dict(r) for r in rows])
df["datetime"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
df["time"] = df["datetime"].dt.strftime("%H:%M")
df["cum_vol"] = df["volume"].cumsum()
df["cum_vol_px"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3.0).cumsum()
df["vwap"] = (df["cum_vol_px"] / df["cum_vol"].replace(0, 1)).round(2)
df["vol_med"] = df["volume"].rolling(10, min_periods=1).median().replace(0, 1.0)
df["rvol"] = (df["volume"] / df["vol_med"]).round(2)

print("\nPOLYCAB First 15 Candles (09:15 - 10:30):")
print(f"{'Time':<6} | {'Open':<8} | {'High':<8} | {'Low':<8} | {'Close':<8} | {'Vol':<8} | {'RVOL':<5} | {'VWAP':<8}")
print("-" * 75)
for idx, r in df.head(16).iterrows():
    print(f"{r['time']:<6} | {r['open']:<8.2f} | {r['high']:<8.2f} | {r['low']:<8.2f} | {r['close']:<8.2f} | {int(r['volume']):<8} | {r['rvol']:<5.1f} | {r['vwap']:<8.2f}")

print("\nPOLYCAB Full Day Summary:")
print(f"Day Open: {df.iloc[0]['open']} | Day High: {df['high'].max()} | Day Low: {df['low'].min()} | Day Close: {df.iloc[-1]['close']}")
