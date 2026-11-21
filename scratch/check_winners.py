from app.liquidity_engine import get_htf_liquidity_levels
from app.database import db
import pandas as pd
import datetime

trade_date = datetime.date(2026, 11, 18)
for sym in ["CIPLA", "TATACONSUM", "MCX", "POLYCAB"]:
    levels = get_htf_liquidity_levels(sym, trade_date)
    with db.connect() as conn:
        row = conn.execute(
            "SELECT open FROM candles WHERE symbol=? AND timeframe='5m' ORDER BY ts ASC LIMIT 1",
            (sym,)
        ).fetchone()
    open_px = row[0]
    pdl = levels["daily_ssl"]
    pdh = levels["daily_bsl"]
    pwl = levels["weekly_ssl"]
    pwh = levels["weekly_bsl"]
    print(f"{sym}: Open: {open_px} | Daily SSL (PDL): {pdl} | Daily BSL (PDH): {pdh} | Weekly SSL: {pwl} | Weekly BSL: {pwh}")
