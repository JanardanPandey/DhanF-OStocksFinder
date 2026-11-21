import asyncio
from app.scanner import scanner

async def check_rr():
    board_res = await scanner.refresh_live_momentum()
    items = board_res.get("items", [])
    for idx, item in enumerate(items[:6]):
        sym = item.get("symbol")
        ltp = float(item.get("ltp", 0.0))
        ob_info = item.get("order_block", {})
        s1 = float(ob_info.get("s1", 0.0))
        r1 = float(ob_info.get("r1", 0.0))
        direction = item.get("direction", "LONG")
        
        reward_pts = abs(r1 - ltp)
        risk_pts = abs(ltp - s1)
        min_reward_pts = 25.0 if ltp >= 1000.0 else max(3.0, ltp * 0.012)
        
        print(f"\n{sym} ({direction}) | LTP: {ltp} | S1: {s1} | R1: {r1}")
        print(f"  Risk: {risk_pts:.2f} (max allowed: {0.012*ltp:.2f}) -> {risk_pts <= 0.012*ltp and risk_pts > 0}")
        print(f"  Reward: {reward_pts:.2f} (min required: {min_reward_pts:.2f}) -> {reward_pts >= min_reward_pts}")
        print(f"  RR: {reward_pts/max(0.1, risk_pts):.2f} (min 1.1) -> {reward_pts/max(0.1, risk_pts) >= 1.1}")

asyncio.run(check_rr())
