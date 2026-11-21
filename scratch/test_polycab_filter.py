import datetime
from app.database import db
from app.liquidity_engine import get_htf_liquidity_levels
import pandas as pd

# Let's inspect all setups on 18-Nov and see the impact of:
# 1. RVOL >= 1.2x on sweeps
# 2. Deep gap filter
from app.liquidity_engine import scan_point_in_time_liquidity_setups

all_setups = scan_point_in_time_liquidity_setups("2026-11-18")
print(f"Total setups before filter: {len(all_setups)}")

target_hits = [s for s in all_setups if s["outcome"] == "TARGET_HIT"]
sl_hits = [s for s in all_setups if s["outcome"] == "STOPPED_OUT"]
running = [s for s in all_setups if s["outcome"] == "RUNNING"]

print(f"Before: Target Hits: {len(target_hits)} | SL Hits: {len(sl_hits)} | Running: {len(running)}")

# Check with RVOL >= 1.2x for Sweeps
sweeps_high_rvol = [s for s in all_setups if ("Sweep" not in s["setup_name"]) or (s["rvol"] >= 1.2)]
th_rvol = [s for s in sweeps_high_rvol if s["outcome"] == "TARGET_HIT"]
sl_rvol = [s for s in sweeps_high_rvol if s["outcome"] == "STOPPED_OUT"]
rn_rvol = [s for s in sweeps_high_rvol if s["outcome"] == "RUNNING"]

print(f"\nAfter RVOL >= 1.2x on Sweeps: Total: {len(sweeps_high_rvol)}")
print(f"Target Hits: {len(th_rvol)} | SL Hits: {len(sl_rvol)} | Running: {len(rn_rvol)}")

print("\nWhat happened to POLYCAB?")
polycab_s = [s for s in all_setups if s["symbol"] == "POLYCAB"]
for p in polycab_s:
    print(p)
