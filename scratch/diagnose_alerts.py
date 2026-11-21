import asyncio
from app.scanner import scanner
from app.playbook_engine import IST
from datetime import datetime

async def diagnose():
    board_res = await scanner.refresh_live_momentum()
    items = board_res.get("items", [])
    print(f"Total board items: {len(items)}")
    
    now_ist = datetime.now(IST)
    curr_hhmm = now_ist.strftime("%H:%M")
    print(f"Current time IST: {curr_hhmm}")

    for idx, item in enumerate(items):
        sym = item.get("symbol")
        ltp = float(item.get("ltp", 0.0))
        score = float(item.get("score", 0.0))
        is_opt_safe = bool(item.get("option_safe"))
        ob_info = item.get("order_block", {})
        ob_status = str(ob_info.get("status", "OB_ACTIVE"))
        s1 = float(ob_info.get("s1", ltp * 0.99))
        r1 = float(ob_info.get("r1", ltp * 1.015))
        anti_chasing = bool(item.get("anti_chasing"))
        status_tag = item.get("status_tag")
        
        # Check reasons
        reasons_drop = []
        if ob_status in ("EARLY_OPEN", "NO_DATA", "OB_ACTIVE", "OB_BASE_FORMING"):
            reasons_drop.append(f"ob_status={ob_status}")
        if abs(r1 - s1) < 0.005 * ltp or r1 <= 0 or s1 <= 0:
            reasons_drop.append(f"structural distance failed (r1={r1}, s1={s1})")
        if anti_chasing:
            reasons_drop.append("anti_chasing=True")
        if not is_opt_safe or score < 70.0:
            reasons_drop.append(f"opt_safe={is_opt_safe}, score={score} (<70)")
        
        # Check Entry status
        entry_ok = False
        if ob_status == "OB_BREAKOUT_CONFIRMED":
            entry_ok = True
        elif ob_status == "OB_RETEST_BOUNCE":
            entry_ok = True
        elif status_tag in ("CREAM_BULL_V2", "STRICT_BEAR_V2") and score >= 80.0:
            entry_ok = True
        else:
            reasons_drop.append(f"no entry trigger (ob_status={ob_status}, status_tag={status_tag}, score={score})")
            
        print(f"[{idx+1}] {sym:<12} | LTP: ₹{ltp:<8.2f} | Score: {score:<4.1f} | OB: {ob_status:<22} | Tag: {status_tag} | Drops: {reasons_drop}")

asyncio.run(diagnose())
