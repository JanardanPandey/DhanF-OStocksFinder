from app.database import db
from app.liquidity_engine import get_htf_liquidity_levels, identify_candle_pattern
import pandas as pd
import datetime

trade_date = datetime.date(2026, 11, 18)
levels = get_htf_liquidity_levels("POLYCAB", trade_date)
pwl = levels["weekly_ssl"]

tz_ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
start_epoch = int(datetime.datetime(2026, 11, 18, 9, 15, tzinfo=tz_ist).timestamp())
end_epoch = int(datetime.datetime(2026, 11, 18, 15, 30, tzinfo=tz_ist).timestamp())

with db.connect() as conn:
    c5 = conn.execute(
        "SELECT ts, open, high, low, close, volume FROM candles WHERE symbol = 'POLYCAB' AND timeframe = '5m' AND ts >= ? AND ts <= ? ORDER BY ts ASC",
        (start_epoch, end_epoch)
    ).fetchall()

df5 = pd.DataFrame([dict(r) for r in c5])
df5["datetime"] = pd.to_datetime(df5["ts"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
df5["hhmm"] = df5["datetime"].dt.strftime("%H:%M")

print(f"Weekly SSL: {pwl}")
# Check where 5m broke below Weekly SSL and retested
for i in range(len(df5)):
    bar = df5.iloc[i]
    if bar["close"] < pwl and bar["open"] > pwl:
        print(f"Breakdown candle @ {bar['hhmm']}: O={bar['open']}, C={bar['close']}")
    # Check retests
    if i > 2 and bar["high"] >= pwl * 0.998 and bar["close"] < pwl:
        print(f"Retest candle @ {bar['hhmm']}: H={bar['high']}, C={bar['close']}")
