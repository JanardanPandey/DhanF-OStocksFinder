from app.database import db
from app.liquidity_engine import get_htf_liquidity_levels, _evaluate_trade_outcome, identify_candle_pattern
import pandas as pd
import datetime

def test_engine_retest_and_vwap(trade_date_str):
    trade_date = datetime.date.fromisoformat(trade_date_str)
    tz_ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    start_dt = datetime.datetime(trade_date.year, trade_date.month, trade_date.day, 9, 15, tzinfo=tz_ist)
    end_dt = datetime.datetime(trade_date.year, trade_date.month, trade_date.day, 15, 30, tzinfo=tz_ist)
    start_epoch = int(start_dt.timestamp())
    end_epoch = int(end_dt.timestamp())

    with db.connect() as conn:
        symbols = [r[0] for r in conn.execute(
            "SELECT DISTINCT symbol FROM candles WHERE timeframe = '5m' AND ts >= ? AND ts <= ?",
            (start_epoch, end_epoch)
        ).fetchall()]

    setups = []
    min_reward_pts = 10.0
    min_rr = 2.0

    for sym in symbols:
        levels = get_htf_liquidity_levels(sym, trade_date)
        if not levels:
            continue
        pdh = levels["daily_bsl"]
        pdl = levels["daily_ssl"]
        pwh = levels["weekly_bsl"]
        pwl = levels["weekly_ssl"]

        with db.connect() as conn:
            c5 = conn.execute(
                "SELECT ts, open, high, low, close, volume FROM candles WHERE symbol = ? AND timeframe = '5m' AND ts >= ? AND ts <= ? ORDER BY ts ASC",
                (sym, start_epoch, end_epoch)
            ).fetchall()

        if len(c5) < 2:
            continue

        df5 = pd.DataFrame([dict(r) for r in c5])
        df5["datetime"] = pd.to_datetime(df5["ts"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
        df5["hhmm"] = df5["datetime"].dt.strftime("%H:%M")
        df5["vol_med"] = df5["volume"].rolling(10, min_periods=1).median().replace(0, 1.0)
        df5["rvol"] = df5["volume"] / df5["vol_med"]

        # Intraday VWAP calculation
        cum_vol = df5["volume"].cumsum()
        cum_vol_px = (df5["volume"] * (df5["high"] + df5["low"] + df5["close"]) / 3.0).cumsum()
        df5["vwap"] = (cum_vol_px / cum_vol.replace(0, 1)).round(2)

        day_open = float(df5.iloc[0]["open"])
        is_deep_gap_down = day_open < (pdl * 0.992)
        is_deep_gap_up = day_open > (pdh * 1.008)

        found_for_sym = False
        for i in range(1, len(df5)):
            if found_for_sym:
                break
            bar = df5.iloc[i]
            prev_bar = df5.iloc[i - 1]
            has_next = (i + 1 < len(df5))
            next_bar = df5.iloc[i + 1] if has_next else None
            b_rng = max(0.05, float(bar["high"]) - float(bar["low"]))
            b_upper_wick = float(bar["high"]) - max(float(bar["open"]), float(bar["close"]))
            b_lower_wick = min(float(bar["open"]), float(bar["close"])) - float(bar["low"])
            b_vwap = float(bar["vwap"])
            b_close = float(bar["close"])

            # SETUP 1: BSL Sweep SHORT
            # Institutional rule: Must not be deep gap up, and price must reject below or near VWAP (close <= vwap * 1.002)
            swept_bsl = None
            if not is_deep_gap_up and b_close <= b_vwap * 1.002:
                if bar["high"] >= pdh and prev_bar["high"] < pdh:
                    swept_bsl = ("Daily BSL (PDH)", pdh)
                elif bar["high"] >= pwh and prev_bar["high"] < pwh:
                    swept_bsl = ("Weekly BSL (PWH)", pwh)

            if swept_bsl:
                lvl_name, lvl_px = swept_bsl
                is_rejection = (b_upper_wick / b_rng >= 0.28) and (
                    (bar["close"] <= bar["open"] and bar["close"] <= bar["low"] + 0.55 * b_rng and bar["close"] < lvl_px)
                    or (next_bar is not None and next_bar["close"] < bar["low"])
                )
                if is_rejection:
                    entry_px = round(float(next_bar["open"]) if next_bar is not None else float(bar["close"]), 2)
                    entry_time = next_bar["hhmm"] if next_bar is not None else bar["hhmm"]
                    entry_bar_idx = (i + 1) if has_next else i
                    sl_px = round(float(bar["high"]) + max(0.5, entry_px * 0.001), 2)
                    target_px = round(pdl if "Daily" in lvl_name else pwl, 2)
                    risk_pts = round(sl_px - entry_px, 2)
                    reward_pts = round(entry_px - target_px, 2)

                    if risk_pts > 0 and reward_pts >= min_reward_pts and (reward_pts / risk_pts) >= min_rr:
                        outcome_info = _evaluate_trade_outcome(
                            df5, entry_bar_idx, "SHORT", entry_px, sl_px, target_px, reward_pts, risk_pts
                        )
                        setups.append({"sym": sym, "setup": "BSL Sweep", "dir": "SHORT", "time": entry_time, **outcome_info})
                        found_for_sym = True
                        continue

            # SETUP 2: SSL Sweep LONG
            # Institutional rule: Must not be deep gap down, and price must hold above or near VWAP (close >= vwap * 0.998)
            swept_ssl = None
            if not is_deep_gap_down and b_close >= b_vwap * 0.998:
                if bar["low"] <= pdl and prev_bar["low"] > pdl:
                    swept_ssl = ("Daily SSL (PDL)", pdl)
                elif bar["low"] <= pwl and prev_bar["low"] > pwl:
                    swept_ssl = ("Weekly SSL (PWL)", pwl)

            if swept_ssl:
                lvl_name, lvl_px = swept_ssl
                is_rejection = (b_lower_wick / b_rng >= 0.28) and (
                    (bar["close"] >= bar["open"] and bar["close"] >= bar["low"] + 0.45 * b_rng and bar["close"] > lvl_px)
                    or (next_bar is not None and next_bar["close"] > bar["high"])
                )
                if is_rejection:
                    entry_px = round(float(next_bar["open"]) if next_bar is not None else float(bar["close"]), 2)
                    entry_time = next_bar["hhmm"] if next_bar is not None else bar["hhmm"]
                    entry_bar_idx = (i + 1) if has_next else i
                    sl_px = round(float(bar["low"]) - max(0.5, entry_px * 0.001), 2)
                    target_px = round(pdh if "Daily" in lvl_name else pwh, 2)
                    risk_pts = round(entry_px - sl_px, 2)
                    reward_pts = round(target_px - entry_px, 2)

                    if risk_pts > 0 and reward_pts >= min_reward_pts and (reward_pts / risk_pts) >= min_rr:
                        outcome_info = _evaluate_trade_outcome(
                            df5, entry_bar_idx, "LONG", entry_px, sl_px, target_px, reward_pts, risk_pts
                        )
                        setups.append({"sym": sym, "setup": "SSL Sweep", "dir": "LONG", "time": entry_time, **outcome_info})
                        found_for_sym = True
                        continue

            # SETUP 3: Breakdown & Retest Continuation (SHORT)
            lvl_break = None
            if prev_bar["close"] >= pdl and bar["close"] < pdl * 0.996 and bar["close"] < bar["open"]:
                lvl_break = ("Daily SSL (PDL)", pdl)
            elif prev_bar["close"] >= pwl and bar["close"] < pwl * 0.996 and bar["close"] < bar["open"]:
                lvl_break = ("Weekly SSL (PWL)", pwl)

            if lvl_break:
                lvl_name, lvl_px = lvl_break
                for j in range(i + 1, min(len(df5), i + 6)):
                    r_bar = df5.iloc[j]
                    # Upgraded Retest Filter:
                    # Retest candle must show BEARISH confirmation (Red body OR Upper rejection wick >= 25%)
                    # Must NOT be a strong green candle closing near high!
                    r_rng = max(0.05, float(r_bar["high"]) - float(r_bar["low"]))
                    r_uwick = float(r_bar["high"]) - max(float(r_bar["open"]), float(r_bar["close"]))
                    r_is_bearish = (float(r_bar["close"]) <= float(r_bar["open"])) or (r_uwick / r_rng >= 0.25)
                    
                    if r_bar["high"] >= lvl_px * 0.998 and r_bar["close"] < lvl_px and r_is_bearish:
                        entry_px = round(float(r_bar["close"]), 2)
                        entry_time = r_bar["hhmm"]
                        entry_bar_idx = j
                        sl_px = round(float(r_bar["high"]) + max(0.5, entry_px * 0.0015), 2)
                        day_rng = abs(pdh - pdl)
                        target_px = round(entry_px - max(day_rng, entry_px * 0.02), 2)
                        risk_pts = round(sl_px - entry_px, 2)
                        reward_pts = round(entry_px - target_px, 2)

                        if risk_pts > 0 and reward_pts >= min_reward_pts and (reward_pts / risk_pts) >= min_rr:
                            outcome_info = _evaluate_trade_outcome(
                                df5, entry_bar_idx, "SHORT", entry_px, sl_px, target_px, reward_pts, risk_pts
                            )
                            setups.append({"sym": sym, "setup": "Breakdown & Retest", "dir": "SHORT", "time": entry_time, **outcome_info})
                            found_for_sym = True
                            break

            # SETUP 4: Breakout & Retest Continuation (LONG)
            lvl_brout = None
            if prev_bar["close"] <= pdh and bar["close"] > pdh * 1.004 and bar["close"] > bar["open"]:
                lvl_brout = ("Daily BSL (PDH)", pdh)
            elif prev_bar["close"] <= pwh and bar["close"] > pwh * 1.004 and bar["close"] > pwh * 1.004:
                lvl_brout = ("Weekly BSL (PWH)", pwh)

            if lvl_brout:
                lvl_name, lvl_px = lvl_brout
                for j in range(i + 1, min(len(df5), i + 6)):
                    r_bar = df5.iloc[j]
                    # Upgraded Retest Filter:
                    # Retest candle must show BULLISH confirmation (Green body OR Lower rejection wick >= 25%)
                    # Must NOT be a strong red candle closing near low!
                    r_rng = max(0.05, float(r_bar["high"]) - float(r_bar["low"]))
                    r_lwick = min(float(r_bar["open"]), float(r_bar["close"])) - float(r_bar["low"])
                    r_is_bullish = (float(r_bar["close"]) >= float(r_bar["open"])) or (r_lwick / r_rng >= 0.25)

                    if r_bar["low"] <= lvl_px * 1.002 and r_bar["close"] > lvl_px and r_is_bullish:
                        entry_px = round(float(r_bar["close"]), 2)
                        entry_time = r_bar["hhmm"]
                        entry_bar_idx = j
                        sl_px = round(float(r_bar["low"]) - max(0.5, entry_px * 0.0015), 2)
                        day_rng = abs(pdh - pdl)
                        target_px = round(entry_px + max(day_rng, entry_px * 0.02), 2)
                        risk_pts = round(entry_px - sl_px, 2)
                        reward_pts = round(target_px - entry_px, 2)

                        if risk_pts > 0 and reward_pts >= min_reward_pts and (reward_pts / risk_pts) >= min_rr:
                            outcome_info = _evaluate_trade_outcome(
                                df5, entry_bar_idx, "LONG", entry_px, sl_px, target_px, reward_pts, risk_pts
                            )
                            setups.append({"sym": sym, "setup": "Breakout & Retest", "dir": "LONG", "time": entry_time, **outcome_info})
                            found_for_sym = True
                            break

    print(f"\n==========================================")
    print(f"Results for {trade_date_str} with Institutional Retest & VWAP Rules:")
    print(f"==========================================")
    th = [s for s in setups if s["outcome"] == "TARGET_HIT"]
    sl = [s for s in setups if s["outcome"] == "STOPPED_OUT"]
    rn = [s for s in setups if s["outcome"] == "RUNNING"]
    print(f"Total: {len(setups)} | Target Hit: {len(th)} | SL Hit: {len(sl)} | Running: {len(rn)}")
    print("\nAll Setups:")
    for s in setups:
        print(f"{s['sym']:<12} | {s['setup']:<20} | {s['dir']:<5} | {s['time']:<6} | {s['outcome_label']}")

test_engine_retest_and_vwap("2026-11-18")
test_engine_retest_and_vwap("2026-11-17")
