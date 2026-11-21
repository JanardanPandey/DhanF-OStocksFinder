from app.liquidity_engine import scan_point_in_time_liquidity_setups
import pandas as pd
import datetime

# Audit all setups for 18-Nov-2026
setups = scan_point_in_time_liquidity_setups("2026-11-18")
print(f"Total Setups on 18-Nov-2026: {len(setups)}")

sl_hits = [s for s in setups if s["outcome"] == "STOPPED_OUT"]
print(f"\nTotal SL Hits: {len(sl_hits)}")

# Inspect every SL hit to see which ones failed immediately
immediate_fails = []
for s in sl_hits:
    t_entry = s["trigger_time"].split()[0] # e.g. "09:50"
    t_hit = s["outcome_time"].split()[0] if s["outcome_time"] else "N/A" # e.g. "10:15"
    
    # Calculate minutes between entry and SL hit
    try:
        e_h, e_m = map(int, t_entry.split(":"))
        h_h, h_m = map(int, t_hit.split(":"))
        mins = (h_h * 60 + h_m) - (e_h * 60 + e_m)
    except Exception:
        mins = -1

    immediate_fails.append({
        "symbol": s["symbol"],
        "setup": s["setup_name"],
        "dir": s["direction"],
        "entry_time": t_entry,
        "hit_time": t_hit,
        "mins_to_sl": mins,
        "entry_px": s["entry_price"],
        "sl_px": s["stop_loss"],
        "loss_pts": s["risk_pts"],
        "rvol": s["rvol"],
        "pattern": s["candle_pattern"],
        "vol_sig": s["volume_signature"],
    })

f_df = pd.DataFrame(immediate_fails).sort_values("mins_to_sl")
print("\nALL STOPPED OUT STOCKS (Sorted by Time to SL):")
print(f_df[["symbol", "setup", "dir", "entry_time", "hit_time", "mins_to_sl", "loss_pts", "rvol", "pattern"]].to_string(index=False))
