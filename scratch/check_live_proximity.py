from app.liquidity_engine import scan_point_in_time_liquidity_setups
import pandas as pd

setups = scan_point_in_time_liquidity_setups("2026-11-19")
appr = [s for s in setups if "Approaching" in s.get("setup_name", "")]

print(f"Top 10 Live Proximity Stocks right now (10:45 AM):")
print(f"{'Symbol':<12} | {'Direction':<11} | {'CMP':<8} | {'Level':<8} | {'Dist %':<8} | {'Pattern':<35} | {'RVOL':<5}")
print("-" * 95)
for s in sorted(appr, key=lambda x: float(x['outcome_label'].split('(')[1].split('%')[0]))[:10]:
    dist = x_str = s['outcome_label'].split('(')[1].split('%')[0] + "%"
    cmp_px = s['entry_price']
    lvl_px = s['level_price']
    pat = s['candle_pattern']
    vol = s['volume_signature']
    print(f"{s['symbol']:<12} | {s['direction']:<11} | {cmp_px:<8.2f} | {lvl_px:<8.2f} | {dist:<8} | {pat:<35} | {vol:<5}")
