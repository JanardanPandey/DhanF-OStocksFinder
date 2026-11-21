from app.database import db
from app.liquidity_engine import scan_point_in_time_liquidity_setups
import pandas as pd
import datetime

def identify_candle_pattern(bar, prev_bar=None):
    o, h, l, c = float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])
    rng = max(0.05, h - l)
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    upper_wick_pct = round((upper_wick / rng) * 100, 1)
    lower_wick_pct = round((lower_wick / rng) * 100, 1)
    body_pct = round((body / rng) * 100, 1)

    is_green = c >= o
    is_red = c < o

    if lower_wick_pct >= 35.0 and upper_wick_pct <= 25.0 and (c >= l + 0.5 * rng):
        return f"Bullish Pin Bar (Hammer) [{lower_wick_pct}% Wick]"
    elif upper_wick_pct >= 35.0 and lower_wick_pct <= 25.0 and (c <= l + 0.5 * rng):
        return f"Shooting Star (Bearish Pin) [{upper_wick_pct}% Wick]"
    elif prev_bar is not None and is_green and float(prev_bar["close"]) < float(prev_bar["open"]):
        if c > float(prev_bar["open"]) and o <= float(prev_bar["close"]):
            return "Bullish Engulfing"
    elif prev_bar is not None and is_red and float(prev_bar["close"]) > float(prev_bar["open"]):
        if c < float(prev_bar["open"]) and o >= float(prev_bar["close"]):
            return "Bearish Engulfing"
    elif upper_wick_pct >= 28.0:
        return f"Bearish Rejection Wick [{upper_wick_pct}%]"
    elif lower_wick_pct >= 28.0:
        return f"Bullish Rejection Wick [{lower_wick_pct}%]"
    elif body_pct >= 60.0:
        return "Bullish Displacement" if is_green else "Bearish Displacement"
    return "Neutral Rejection"

setups = scan_point_in_time_liquidity_setups("2026-11-18")
start_epoch = int(datetime.datetime(2026, 11, 18, 9, 15, tzinfo=datetime.timezone(datetime.timedelta(hours=5, minutes=30))).timestamp())
end_epoch = int(datetime.datetime(2026, 11, 18, 15, 30, tzinfo=datetime.timezone(datetime.timedelta(hours=5, minutes=30))).timestamp())

print(f"Total setups on 2026-11-18: {len(setups)}")
for s in setups[:12]:
    sym = s["symbol"]
    t_str = s["trigger_time"].split()[0]
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT ts, open, high, low, close, volume FROM candles WHERE symbol=? AND timeframe='5m' AND ts >= ? AND ts <= ? ORDER BY ts ASC",
            (sym, start_epoch, end_epoch)
        ).fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    df["hhmm"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.strftime("%H:%M")
    m = df[df["hhmm"] == t_str]
    if len(m) > 0:
        idx = m.index[0]
        bar = df.iloc[idx - 1] if idx > 0 else df.iloc[0]
        prev = df.iloc[idx - 2] if idx > 1 else None
        pat = identify_candle_pattern(bar, prev)
        direction = s['direction']
        setup_name = s['setup_name']
        print(f"[{t_str}] {sym} ({direction} - {setup_name}): Pattern -> {pat}")
