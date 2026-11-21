from scratch.test_adani_fix import *
# Let's run and print symbols and outcomes
trade_date_str = "2026-11-18"
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

for sym in ["MCX", "TATACONSUM", "CIPLA", "TCS", "ADANIPORTS"]:
    levels = get_htf_liquidity_levels(sym, trade_date)
    with db.connect() as conn:
        c5 = conn.execute(
            "SELECT ts, open, high, low, close, volume FROM candles WHERE symbol = ? AND timeframe = '5m' AND ts >= ? AND ts <= ? ORDER BY ts ASC",
            (sym, start_epoch, end_epoch)
        ).fetchall()
    df5 = pd.DataFrame([dict(r) for r in c5])
    df5["datetime"] = pd.to_datetime(df5["ts"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
    df5["hhmm"] = df5["datetime"].dt.strftime("%H:%M")
    pdh = levels["daily_bsl"]
    pdl = levels["daily_ssl"]
    print(f"\n{sym} (PDL={pdl}, PDH={pdh}):")
    for i in range(1, min(10, len(df5))):
        b = df5.iloc[i]
        pb = df5.iloc[i-1]
        print(f"  {b['hhmm']} | O={b['open']}, H={b['high']}, L={b['low']}, C={b['close']}")
