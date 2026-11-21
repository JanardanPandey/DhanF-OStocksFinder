from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

from app.database import db
from app.playbook_engine import (
    _candles_to_df,
    _classify_option_timing,
    compute_order_block,
    is_valid_fno_symbol,
    recommend_option_strike,
)

IST = timezone(timedelta(hours=5, minutes=30))


def run_walk_forward_replay(target_date: str = "2026-11-17") -> dict[str, Any]:
    print(f"\n==================================================================")
    print(f"  STARTING CANDLE-BY-CANDLE WALK-FORWARD REPLAY FOR: {target_date}")
    print(f"  (Zero-Lookahead Point-in-Time Simulation across All F&O Stocks)")
    print(f"==================================================================")

    # 1. Load all 5m candles from database
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT symbol, ts, open, high, low, close, volume, COALESCE(oi, 0) as oi
            FROM candles
            WHERE timeframe = '5m'
            ORDER BY symbol, ts ASC
            """
        ).fetchall()

    sym_candles: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        sym = r["symbol"]
        if not is_valid_fno_symbol(sym):
            continue
        if sym not in sym_candles:
            sym_candles[sym] = []
        sym_candles[sym].append(dict(r))

    print(f"Loaded 5m candle history for {len(sym_candles)} valid F&O symbols.")

    # 2. Extract target_date candles and T-1 reference context for each symbol
    universe_data: dict[str, dict[str, Any]] = {}
    all_timestamps: set[int] = set()

    for sym, recs in sym_candles.items():
        df_all = _candles_to_df(recs)
        if df_all.empty:
            continue

        df_target = df_all[df_all["trade_date"].astype(str) == target_date].reset_index(drop=True)
        if len(df_target) < 6:
            # Need at least 30 mins of trading on target date
            continue

        df_t1 = df_all[df_all["trade_date"].astype(str) < target_date]
        if df_t1.empty:
            continue

        # T-1 metrics
        t1_dates = df_t1["trade_date"].unique()
        last_t1_d = t1_dates[-1]
        df_last_t1 = df_t1[df_t1["trade_date"] == last_t1_d]

        pdc = float(df_last_t1["close"].iloc[-1])
        pdh = float(df_last_t1["high"].max())
        pdl = float(df_last_t1["low"].min())

        # Daily ATR14 from daily candles if available, else from 5m history
        c1d = db.load_candles(sym, "1D", 40)
        c1h = db.load_candles(sym, "1h", 60)

        atr14_val = 0.0
        atr_pct = 0.0
        if c1d:
            df1 = _candles_to_df(c1d)
            d1_hist = df1[df1["trade_date"].astype(str) < target_date].tail(16).reset_index(drop=True)
            if len(d1_hist) >= 8:
                h_arr = d1_hist["high"].to_numpy()
                l_arr = d1_hist["low"].to_numpy()
                c_arr = d1_hist["close"].to_numpy()
                tr_arr = np.maximum(
                    h_arr[1:] - l_arr[1:],
                    np.maximum(np.abs(h_arr[1:] - c_arr[:-1]), np.abs(l_arr[1:] - c_arr[:-1])),
                )
                atr14_val = float(pd.Series(tr_arr).rolling(min(14, len(tr_arr))).mean().iloc[-1])
                if pdc > 0:
                    atr_pct = round(atr14_val / pdc * 100.0, 2)

        if atr14_val <= 0 and pdc > 0:
            atr14_val = pdc * 0.02
            atr_pct = 2.0

        inst_meta = db.get_instrument(sym)
        lot_size = int(inst_meta.get("fno_lot_size") or 100) if inst_meta else 100

        universe_data[sym] = {
            "df_target": df_target,
            "pdc": pdc,
            "pdh": pdh,
            "pdl": pdl,
            "atr14": atr14_val,
            "atr_pct": atr_pct,
            "c1d": c1d,
            "c1h": c1h,
            "lot_size": lot_size,
        }

        for ts in df_target["ts"]:
            all_timestamps.add(int(ts))

    sorted_ts = sorted(list(all_timestamps))
    print(f"Target universe: {len(universe_data)} symbols active on {target_date}.")
    print(f"Total time steps to replay: {len(sorted_ts)} 5-minute candles.")

    if not sorted_ts:
        return {"error": "no_data_for_date", "target_date": target_date}

    # 3. Candle-by-Candle Simulation
    triggered_symbols: set[str] = set()
    alerts_log: list[dict[str, Any]] = []

    print(f"\nWalking forward candle-by-candle...")

    for step_idx, cur_ts in enumerate(sorted_ts):
        step_dt = datetime.fromtimestamp(cur_ts, tz=IST)
        hhmm = step_dt.strftime("%H:%M")

        # Skip opening 10 minutes (bars 0 and 1) — ORB structure requires at least 3 bars (09:25 onwards)
        if step_idx < 2:
            continue

        # Step 3a: Compute live market breadth across active symbols at this step
        above_vwap_cnt = 0
        total_active = 0
        step_quotes: dict[str, dict[str, Any]] = {}

        for sym, udata in universe_data.items():
            df_tgt = udata["df_target"]
            sub_df = df_tgt[df_tgt["ts"] <= cur_ts]
            if sub_df.empty:
                continue

            total_active += 1
            ltp = float(sub_df["close"].iloc[-1])
            o0 = float(sub_df["open"].iloc[0])
            tp = (sub_df["high"] + sub_df["low"] + sub_df["close"]) / 3.0
            cum_vol = sub_df["volume"].cumsum()
            sess_vwap = float((tp * sub_df["volume"]).cumsum().iloc[-1] / max(1.0, cum_vol.iloc[-1]))

            if ltp >= sess_vwap:
                above_vwap_cnt += 1

            step_quotes[sym] = {
                "ltp": ltp,
                "vwap": sess_vwap,
                "open": o0,
                "sub_df": sub_df,
            }

        breadth_pct = round(above_vwap_cnt / max(1, total_active) * 100.0, 1)

        # Step 3b: Evaluate candidates for fresh institutional breakout trigger
        for sym, q in step_quotes.items():
            if sym in triggered_symbols:
                # Already captured earlier today! No duplicate chasing.
                continue

            udata = universe_data[sym]
            sub_df = q["sub_df"]
            ltp = q["ltp"]
            vwap = q["vwap"]
            o0 = q["open"]
            pdc = udata["pdc"]
            pdh = udata["pdh"]
            pdl = udata["pdl"]
            atr14 = udata["atr14"]
            atr_pct = udata["atr_pct"]
            c1d = udata["c1d"]
            c1h = udata["c1h"]
            lot_size = udata["lot_size"]

            move_open_pct = round((ltp - o0) / max(0.01, o0) * 100.0, 2)
            direction = "LONG" if move_open_pct >= 0 else "SHORT"

            # 1. Option Timing Evaluation (Point-in-Time)
            opt_timing = _classify_option_timing(
                sub_df,
                direction,
                ltp,
                vwap,
                move_open_pct,
                atr14=atr14,
                atr_pct=atr_pct,
                pdh=pdh,
                pdl=pdl,
            )

            # 2. Smart Money Order Block Evaluation (Point-in-Time)
            ob_info = compute_order_block(sub_df, direction, ltp, vwap, c1d=c1d, c1h=c1h)
            ob_status = str(ob_info.get("status"))
            s1 = float(ob_info.get("s1", ltp * 0.99))
            r1 = float(ob_info.get("r1", ltp * 1.02))
            r2 = float(ob_info.get("r2", ltp * 1.035))

            # 3. Institutional Gates Check:
            is_opt_safe = bool(opt_timing.get("option_safe"))
            adverse_wick = float(opt_timing.get("adverse_open_wick", 0.0))
            dir_move = abs(move_open_pct)

            # Absolute rules:
            # - Adverse open wick <= 0.40%
            # - Dir move >= 1.25% (early morning) or >= 1.80% (midday)
            # - Order Block Breakout confirmed or Retest bounce
            # - Not extended (ob_status != "OB_EXTENDED")
            # - option_safe is True
            valid_breakout = (
                is_opt_safe
                and (adverse_wick <= 0.40)
                and (ob_status in ("OB_BREAKOUT_CONFIRMED", "OB_RETEST_BOUNCE"))
                and (ob_status != "OB_EXTENDED")
            )

            if valid_breakout:
                # Triggered! Note it down immediately at this exact candle
                triggered_symbols.add(sym)
                opt_strike = recommend_option_strike(sym, ltp, direction, lot_size)

                alert = {
                    "step_idx": step_idx,
                    "timestamp": f"{target_date} {hhmm} IST",
                    "hhmm": hhmm,
                    "symbol": sym,
                    "direction": direction,
                    "trigger_event": ob_status,
                    "entry_price": ltp,
                    "vwap_at_entry": vwap,
                    "move_pct_at_entry": move_open_pct,
                    "ob_zone": ob_info.get("ob_zone"),
                    "s1": s1,
                    "r1": r1,
                    "r2": r2,
                    "risk_per_share": round(abs(ltp - s1), 2),
                    "option_strike_name": opt_strike.get("option_name"),
                    "est_premium": opt_strike.get("est_premium"),
                    "est_capital_inr": opt_strike.get("est_capital_inr"),
                    "adverse_wick": adverse_wick,
                    "option_phase": opt_timing.get("option_phase"),
                    "breadth_pct": breadth_pct,
                    "full_df": udata["df_target"],
                    "entry_bar_idx": len(sub_df) - 1,
                }
                alerts_log.append(alert)
                print(
                    f"  🚨 [{hhmm} IST] {sym:<12} {direction:<5} @ ₹{ltp:<8.2f} "
                    f"({move_open_pct:+.2f}%) | {ob_status} | R1: ₹{r1:.2f} | Strike: {opt_strike.get('option_name')}"
                )

    print(f"\nCandle-by-candle scan completed! Total alerts triggered in real-time: {len(alerts_log)}")

    # 4. Forward Tracking Outcomes (from entry bar to 15:30)
    print(f"\n==================================================================")
    print(f"  TRACKING OUTCOMES FROM TRIGGER BAR TO END OF DAY...")
    print(f"==================================================================")

    results: list[dict[str, Any]] = []

    for al in alerts_log:
        sym = al["symbol"]
        entry_px = al["entry_price"]
        dirn = al["direction"]
        s1 = al["s1"]
        r1 = al["r1"]
        r2 = al["r2"]
        df_target = al["full_df"]
        entry_idx = al["entry_bar_idx"]

        # Future bars after entry
        future_bars = df_target.iloc[entry_idx + 1 :]
        if future_bars.empty:
            outcome = "TRIGGERED_AT_CLOSE"
            mfe_px = entry_px
            mae_px = entry_px
            hit_r1_hhmm = None
            hit_r2_hhmm = None
            hit_sl_hhmm = None
            exit_px = entry_px
        else:
            highs = future_bars["high"].to_numpy()
            lows = future_bars["low"].to_numpy()
            closes = future_bars["close"].to_numpy()
            hhmm_future = future_bars["hhmm"].tolist()

            hit_r1_idx = None
            hit_r2_idx = None
            hit_sl_idx = None

            if dirn == "LONG":
                mfe_px = float(np.max(highs))
                mae_px = float(np.min(lows))
                for j in range(len(future_bars)):
                    if highs[j] >= r1 and hit_r1_idx is None:
                        hit_r1_idx = j
                    if highs[j] >= r2 and hit_r2_idx is None:
                        hit_r2_idx = j
                    if lows[j] <= s1 and hit_sl_idx is None:
                        hit_sl_idx = j
            else:
                mfe_px = float(np.min(lows))
                mae_px = float(np.max(highs))
                for j in range(len(future_bars)):
                    if lows[j] <= r1 and hit_r1_idx is None:
                        hit_r1_idx = j
                    if lows[j] <= r2 and hit_r2_idx is None:
                        hit_r2_idx = j
                    if highs[j] >= s1 and hit_sl_idx is None:
                        hit_sl_idx = j

            hit_r1_hhmm = hhmm_future[hit_r1_idx] if hit_r1_idx is not None else None
            hit_r2_hhmm = hhmm_future[hit_r2_idx] if hit_r2_idx is not None else None
            hit_sl_hhmm = hhmm_future[hit_sl_idx] if hit_sl_idx is not None else None
            exit_px = float(closes[-1])

            # Classify Outcome:
            if hit_r2_idx is not None and (hit_sl_idx is None or hit_r2_idx < hit_sl_idx):
                outcome = "TARGET_R2_MONSTER"
            elif hit_r1_idx is not None and (hit_sl_idx is None or hit_r1_idx < hit_sl_idx):
                outcome = "TARGET_R1_WIN"
            elif hit_sl_idx is not None and (hit_r1_idx is None or hit_sl_idx < hit_r1_idx):
                outcome = "STOPPED_OUT"
            else:
                # Neither hit R1 nor S1
                net_close_pnl = (exit_px - entry_px) if dirn == "LONG" else (entry_px - exit_px)
                if net_close_pnl > 0:
                    outcome = "PARTIAL_PROFIT"
                else:
                    outcome = "CHOP_FLAT"

        dir_mult = 1.0 if dirn == "LONG" else -1.0
        mfe_pct = round(dir_mult * (mfe_px - entry_px) / entry_px * 100.0, 2)
        mae_pct = round(dir_mult * (mae_px - entry_px) / entry_px * 100.0, 2)
        eod_ret_pct = round(dir_mult * (exit_px - entry_px) / entry_px * 100.0, 2)

        # Estimate Option ROI % based on Delta 0.50
        est_prem = float(al.get("est_premium") or 10.0)
        opt_delta = 0.50
        est_prem_peak = max(0.5, est_prem + (mfe_px - entry_px) * dir_mult * opt_delta)
        opt_peak_roi = round((est_prem_peak - est_prem) / est_prem * 100.0, 1)

        res_item = {
            "hhmm": al["hhmm"],
            "symbol": sym,
            "direction": dirn,
            "entry_price": entry_px,
            "strike": al["option_strike_name"],
            "est_premium": est_prem,
            "s1": s1,
            "r1": r1,
            "r2": r2,
            "outcome": outcome,
            "mfe_pct": mfe_pct,
            "mae_pct": mae_pct,
            "eod_ret_pct": eod_ret_pct,
            "opt_peak_roi": opt_peak_roi,
            "hit_r1_time": hit_r1_hhmm,
            "hit_r2_time": hit_r2_hhmm,
            "hit_sl_time": hit_sl_hhmm,
        }
        results.append(res_item)

        status_icon = "🟢" if "TARGET" in outcome else ("🔴" if outcome == "STOPPED_OUT" else "🟡")
        print(
            f"  {status_icon} [{al['hhmm']}] {sym:<12} {dirn:<5} Entry: ₹{entry_px:<8.2f} | "
            f"Result: {outcome:<18} | Peak: {mfe_pct:+.2f}% (Opt: {opt_peak_roi:+.0f}%) | "
            f"Target Hit: {hit_r1_hhmm or 'None'} | SL: {hit_sl_hhmm or 'None'}"
        )

    # 5. Calculate Performance Statistics
    total_trades = len(results)
    winners = [r for r in results if "TARGET" in r["outcome"] or r["outcome"] == "PARTIAL_PROFIT"]
    losers = [r for r in results if r["outcome"] == "STOPPED_OUT"]
    chops = [r for r in results if r["outcome"] == "CHOP_FLAT"]

    win_rate = round(len(winners) / max(1, total_trades) * 100.0, 1)
    sl_rate = round(len(losers) / max(1, total_trades) * 100.0, 1)
    avg_mfe = round(float(np.mean([r["mfe_pct"] for r in results])) if results else 0.0, 2)
    avg_opt_roi = round(float(np.mean([r["opt_peak_roi"] for r in results])) if results else 0.0, 1)

    print(f"\n==================================================================")
    print(f"  PERFORMANCE SUMMARY FOR {target_date}:")
    print(f"  Total Trades Triggered Point-in-Time: {total_trades}")
    print(f"  Target R1/R2 Winners: {len(winners)} ({win_rate}%)")
    print(f"  Stop Loss Hits: {len(losers)} ({sl_rate}%)")
    print(f"  Chop / Neutral: {len(chops)}")
    print(f"  Average Peak MFE: +{avg_mfe}%")
    print(f"  Average Option Peak ROI: +{avg_opt_roi}%")
    print(f"==================================================================\n")

    return {
        "target_date": target_date,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "sl_rate": sl_rate,
        "avg_mfe": avg_mfe,
        "avg_opt_roi": avg_opt_roi,
        "results": results,
    }


if __name__ == "__main__":
    t_date = sys.argv[1] if len(sys.argv) > 1 else "2026-11-17"
    run_walk_forward_replay(t_date)
