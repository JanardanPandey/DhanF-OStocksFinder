from app.liquidity_engine import get_available_liquidity_dates, scan_point_in_time_liquidity_setups, get_htf_liquidity_levels

dates = get_available_liquidity_dates()
for d in dates:
    levels = get_htf_liquidity_levels("UPL", d)
    setups = scan_point_in_time_liquidity_setups(d)
    upl_setups = [s for s in setups if s["symbol"] == "UPL"]
    if upl_setups:
        print(f"=== Date: {d} ===")
        print("Levels:", levels)
        for s in upl_setups:
            t = s['trigger_time']
            direction = s['direction']
            setup_name = s['setup_name']
            lvl = s['level_type']
            lvl_px = s['level_price']
            entry = s['entry_price']
            sl = s['stop_loss']
            tgt = s['target_price']
            out = s['outcome_label']
            print(f"  Trigger: {t}, Dir: {direction}, Setup: {setup_name}, Level: {lvl} {lvl_px}, Entry: {entry}, SL: {sl}, Target: {tgt}, Outcome: {out}")
