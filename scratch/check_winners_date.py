from app.liquidity_engine import get_htf_liquidity_levels
from app.database import db
import pandas as pd
import datetime

trade_date = datetime.date(2026, 11, 18)
tz_ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
start_epoch = int(datetime.datetime(2026, 11, 18, 9, 15, tzinfo=tz_ist).timestamp())
end_epoch = int(datetime.datetime(2026, 11, 18, 15, 30, tzinfo=tz_ist).timestamp())

for sym in ["CIPLA", "TATACONSUM", "MCX", "POLYCAB"]:
    levels = get_htf_liquidity_levels(sym, trade_date)
    with db.connect() as conn:
        row = conn.execute(
            "SELECT open, high, low, close FROM candles WHERE symbol=? AND timeframe='5m' AND ts >= ? AND ts <= ? ORDER BY ts ASC LIMIT 1",
            (sym, start_epoch, end_epoch)
        ).fetchone()
    open_px, high_px, low_px, close_px = row
    pdl = levels["daily_ssl"]
    pdh = levels["daily_bsl"]
    pwl = levels["weekly_ssl"]
    pwh = levels["weekly_bsl"]
    print(f"{sym} on 18-Nov-2026:")
    print(f"  09:15 Candle: O: {open_px}, H: {high_px}, L: {low_px}, C: {close_px}")
    print(f"  Daily SSL (PDL): {pdl} | Daily BSL (PDH): {pdh}")
    print(f"  Weekly SSL (PWL): {pwl} | Weekly BSL (PWH): {pwh}")
