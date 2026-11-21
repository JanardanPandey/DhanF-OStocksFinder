from app.database import db
from app.liquidity_engine import get_htf_liquidity_levels, scan_point_in_time_liquidity_setups
import datetime

# Test what happens across 18-Nov and 17-Nov
for dt_str in ["2026-11-18", "2026-11-17"]:
    setups = scan_point_in_time_liquidity_setups(dt_str)
    print(f"\n--- Date: {dt_str} ---")
    print(f"Total Setups: {len(setups)}")
    th = [s for s in setups if s["outcome"] == "TARGET_HIT"]
    sl = [s for s in setups if s["outcome"] == "STOPPED_OUT"]
    rn = [s for s in setups if s["outcome"] == "RUNNING"]
    print(f"Target Hit: {len(th)} | SL Hit: {len(sl)} | Running: {len(rn)}")
