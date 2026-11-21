import datetime
from app.database import db
from app.liquidity_engine import get_htf_liquidity_levels
import pandas as pd

trade_date = datetime.date(2026, 11, 18)
tz_ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
start_dt = datetime.datetime(2026, 11, 18, 9, 15, tzinfo=tz_ist)
end_dt = datetime.datetime(2026, 11, 18, 15, 30, tzinfo=tz_ist)

with db.connect() as conn:
    rows = conn.execute(
        """
        SELECT symbol, open, high, low, close, volume, ts FROM candles 
        WHERE timeframe = '5m' AND ts >= ? AND ts <= ?
        ORDER BY symbol, ts ASC
        """,
        (int(start_dt.timestamp()), int(end_dt.timestamp()))
    ).fetchall()

df = pd.DataFrame([dict(r) for r in rows])
symbols = df["symbol"].unique()

gap_data = []
for sym in symbols:
    sym_df = df[df["symbol"] == sym]
    if len(sym_df) < 5:
        continue
    open_px = sym_df.iloc[0]["open"]
    close_px = sym_df.iloc[-1]["close"]
    levels = get_htf_liquidity_levels(sym, trade_date)
    if not levels:
        continue
    pdl = levels["daily_ssl"]
    pdh = levels["daily_bsl"]
    
    # Gap relative to PDL / PDH
    gap_below_pdl_pct = round(((pdl - open_px) / pdl) * 100, 2) if open_px < pdl else 0.0
    gap_above_pdh_pct = round(((open_px - pdh) / pdh) * 100, 2) if open_px > pdh else 0.0
    day_return_pct = round(((close_px - open_px) / open_px) * 100, 2)

    gap_data.append({
        "symbol": sym,
        "open": open_px,
        "close": close_px,
        "pdl": pdl,
        "pdh": pdh,
        "gap_below_pdl": gap_below_pdl_pct,
        "gap_above_pdh": gap_above_pdh_pct,
        "day_return": day_return_pct,
    })

gdf = pd.DataFrame(gap_data)
deep_gap_down = gdf[gdf["gap_below_pdl"] > 0.8].sort_values("gap_below_pdl", ascending=False)
print("Stocks that opened deep gap down (> 0.8% below PDL):")
print(deep_gap_down[["symbol", "open", "pdl", "gap_below_pdl", "day_return"]].to_string())
