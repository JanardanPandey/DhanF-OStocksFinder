from app.database import db
from app.liquidity_engine import scan_point_in_time_liquidity_setups
import pandas as pd
import datetime

setups = scan_point_in_time_liquidity_setups("2026-11-18")
start_epoch = int(datetime.datetime(2026, 11, 18, 9, 15, tzinfo=datetime.timezone(datetime.timedelta(hours=5, minutes=30))).timestamp())
end_epoch = int(datetime.datetime(2026, 11, 18, 15, 30, tzinfo=datetime.timezone(datetime.timedelta(hours=5, minutes=30))).timestamp())

counts = {"TARGET_HIT": 0, "STOPPED_OUT": 0, "RUNNING": 0, "LIVE_TRACKING": 0}
sample_results = []

for s in setups:
    if "Approaching" in s["setup_name"]:
        counts["LIVE_TRACKING"] += 1
        continue

    sym = s["symbol"]
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT ts, open, high, low, close FROM candles WHERE symbol=? AND timeframe='5m' AND ts >= ? AND ts <= ? ORDER BY ts ASC",
            (sym, start_epoch, end_epoch)
        ).fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    df["hhmm"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.strftime("%H:%M")

    entry_time = s["trigger_time"].split()[0]
    match_idx = df[df["hhmm"] == entry_time].index
    if len(match_idx) == 0:
        continue
    e_idx = match_idx[0]

    direction = s["direction"]
    entry_px = s["entry_price"]
    sl_px = s["stop_loss"]
    tgt_px = s["target_price"]

    outcome = "RUNNING"
    hit_time = None
    for j in range(e_idx + 1, len(df)):
        bar = df.iloc[j]
        if direction == "LONG":
            if bar["low"] <= sl_px:
                outcome = "STOPPED_OUT"
                hit_time = bar["hhmm"]
                break
            elif bar["high"] >= tgt_px:
                outcome = "TARGET_HIT"
                hit_time = bar["hhmm"]
                break
        elif direction == "SHORT":
            if bar["high"] >= sl_px:
                outcome = "STOPPED_OUT"
                hit_time = bar["hhmm"]
                break
            elif bar["low"] <= tgt_px:
                outcome = "TARGET_HIT"
                hit_time = bar["hhmm"]
                break

    counts[outcome] += 1
    cmp = float(df.iloc[-1]["close"])
    pnl = round((cmp - entry_px) if direction == "LONG" else (entry_px - cmp), 2)
    sample_results.append({
        "time": s["trigger_time"],
        "symbol": sym,
        "dir": direction,
        "entry": entry_px,
        "sl": sl_px,
        "target": tgt_px,
        "outcome": outcome,
        "hit_time": hit_time or "Still Active",
        "cmp": cmp,
        "pnl": pnl
    })

print("Outcomes summary:", counts)
print("\nSample entries reconstructed when server starts late:")
for r in sample_results[:10]:
    t = r['time']
    sym = r['symbol']
    d = r['dir']
    ent = r['entry']
    out = r['outcome']
    ht = r['hit_time']
    cmp = r['cmp']
    pnl = r['pnl']
    print(f"[{t}] {sym} {d} @ Rs {ent} -> Status: {out} ({ht}), CMP: Rs {cmp} (PnL: {pnl:+0.2f} pts)")
