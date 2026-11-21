from app.liquidity_engine import scan_point_in_time_liquidity_setups
import pandas as pd
import datetime

trade_date = "2026-11-19"
setups = scan_point_in_time_liquidity_setups(trade_date)

triggered = [s for s in setups if "Approaching" not in s.get("setup_name", "")]
proximity = [s for s in setups if "Approaching" in s.get("setup_name", "")]

th = [s for s in triggered if s.get("outcome") == "TARGET_HIT"]
sl = [s for s in triggered if s.get("outcome") == "STOPPED_OUT"]
rn = [s for s in triggered if s.get("outcome") == "RUNNING"]

rn_pos = [s for s in rn if s.get("current_pnl", 0) > 0]
rn_neg = [s for s in rn if s.get("current_pnl", 0) < 0]
rn_flat = [s for s in rn if s.get("current_pnl", 0) == 0]

print("================================================================================")
print(f"AUDIT SUMMARY FOR TODAY (19-NOV-2026) AT 12:22 PM IST")
print("================================================================================")
print(f"Total Triggered Setups: {len(triggered)}")
print(f"🎯 Target Hit: {len(th)}")
print(f"❌ Stop Loss Hit: {len(sl)}")
print(f"⚡ Running (Active): {len(rn)}")
print(f"   ↳ In Profit (Green): {len(rn_pos)}")
print(f"   ↳ In Minor Drawdown (Red): {len(rn_neg)}")
print(f"   ↳ At Breakeven: {len(rn_flat)}")
print(f"👀 Proximity Radar Alerts: {len(proximity)}")

print("\n" + "=" * 80)
print("DETAILED BREAKDOWN OF ALL TRIGGERED TRADES:")
print("=" * 80)
for idx, s in enumerate(triggered, 1):
    sym = s["symbol"]
    setup = s["setup_name"]
    dir_ = s["direction"]
    t_trig = s["trigger_time"]
    entry = s["entry_price"]
    sl_px = s["stop_loss"]
    tgt_px = s["target_price"]
    rr = s["rr_ratio"]
    pat = s["candle_pattern"]
    vol = s["volume_signature"]
    outcome_lbl = s["outcome_label"]
    pnl = s.get("current_pnl", 0.0)
    cmp_px = s.get("cmp", "N/A")
    risk_pts = s["risk_pts"]
    reward_pts = s["reward_pts"]
    print(f"\n{idx}. {sym} | {dir_} ({setup}) @ {t_trig}")
    print(f"   Entry: ₹{entry} | SL: ₹{sl_px} (Risk: {risk_pts} pts) | Tgt: ₹{tgt_px} (Reward: {reward_pts} pts) | RR: {rr}")
    print(f"   Pattern: {pat} | Volume: {vol}")
    print(f"   Status: {outcome_lbl}")
