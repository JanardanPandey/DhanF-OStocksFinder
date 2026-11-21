from app.liquidity_engine import scan_point_in_time_liquidity_setups
setups = scan_point_in_time_liquidity_setups("2026-11-18")
print(f"Total Setups: {len(setups)}")
th = [s for s in setups if s["outcome"] == "TARGET_HIT"]
sl = [s for s in setups if s["outcome"] == "STOPPED_OUT"]
rn = [s for s in setups if s["outcome"] == "RUNNING"]
print(f"Target Hit: {len(th)} | SL Hit: {len(sl)} | Running: {len(rn)}")
print("\nFirst 10 setups:")
for s in setups[:10]:
    print(f"{s['symbol']:<12} | {s['direction']:<5} | {s['trigger_time']:<9} | {s['candle_pattern']:<35} | {s['outcome_label']}")
