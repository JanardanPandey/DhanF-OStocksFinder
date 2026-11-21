from app.liquidity_engine import scan_point_in_time_liquidity_setups
import pandas as pd
import datetime

trade_date = "2026-11-19"
setups = scan_point_in_time_liquidity_setups(trade_date)
print(f"Total Setups Scanned for Today ({trade_date}): {len(setups)}")

if not setups:
    print("No setups found yet.")
else:
    th = [s for s in setups if s.get("outcome") == "TARGET_HIT"]
    sl = [s for s in setups if s.get("outcome") == "STOPPED_OUT"]
    rn = [s for s in setups if s.get("outcome") == "RUNNING"]
    appr = [s for s in setups if "Approaching" in s.get("setup_name", "")]

    print(f"\n--- PERFORMANCE SUMMARY FOR TODAY (09:15 - 10:45 AM IST) ---")
    print(f"Total Setups Triggered: {len(setups)}")
    print(f"Target Hit: {len(th)}")
    print(f"Stop Loss Hit: {len(sl)}")
    print(f"Running (Active): {len(rn)}")
    print(f"Proximity Alerts (Watching): {len(appr)}")

    print("\n--- ALL SETUPS DETAILS ---")
    for s in setups:
        sym = s["symbol"]
        setup = s["setup_name"]
        dir_ = s["direction"]
        t_trig = s["trigger_time"]
        entry = s.get("entry_price", "N/A")
        sl_px = s.get("stop_loss", "N/A")
        tgt_px = s.get("target_price", "N/A")
        rr = s.get("rr_ratio", "N/A")
        pat = s.get("candle_pattern", "N/A")
        vol = s.get("volume_signature", "N/A")
        outcome = s.get("outcome_label", "N/A")
        pnl = s.get("current_pnl", 0.0)
        cmp = s.get("cmp", "N/A")
        print(f"[{t_trig}] {sym} ({dir_} - {setup}) | Entry: ₹{entry} | SL: ₹{sl_px} | Tgt: ₹{tgt_px} | RR: {rr} | Pat: {pat} | Vol: {vol} | Status: {outcome}")
