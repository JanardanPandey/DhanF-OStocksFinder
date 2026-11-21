from __future__ import annotations

import datetime
from typing import Any
import numpy as np
import pandas as pd

from app.database import db
from app.playbook_engine import is_valid_fno_symbol


def get_available_liquidity_dates() -> list[str]:
    """Return distinct available trading dates formatted as YYYY-MM-DD from candles table, ensuring today is included."""
    tz_ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    now_ist = datetime.datetime.now(tz_ist)
    today_str = now_ist.date().isoformat()
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT date(datetime(ts, 'unixepoch', '+05:30')) as dt 
            FROM candles WHERE timeframe='5m' 
            ORDER BY dt DESC LIMIT 20
            """
        ).fetchall()
        dts = [r[0] for r in rows if r[0]]
        # Include today if it is a weekday (Monday-Friday) and not already present
        if now_ist.weekday() < 5 and today_str not in dts:
            dts.insert(0, today_str)
        return dts


def find_equal_highs_lows(df_window: pd.DataFrame, tolerance_pct: float = 0.0035) -> tuple[float | None, float | None]:
    """Find Equal Highs (EQH) and Equal Lows (EQL) in a multi-day window."""
    if len(df_window) < 5:
        return None, None
    highs = df_window["high"].to_numpy(dtype=float)
    lows = df_window["low"].to_numpy(dtype=float)

    swing_highs = []
    for i in range(1, len(highs) - 1):
        if highs[i] >= highs[i - 1] and highs[i] >= highs[i + 1]:
            swing_highs.append(highs[i])
    max_h = float(np.max(highs))
    if max_h not in swing_highs:
        swing_highs.append(max_h)

    eqh = None
    if len(swing_highs) >= 2:
        sorted_sh = sorted(swing_highs, reverse=True)
        if abs(sorted_sh[0] - sorted_sh[1]) / max(0.01, sorted_sh[0]) <= tolerance_pct:
            eqh = round((sorted_sh[0] + sorted_sh[1]) / 2.0, 2)

    swing_lows = []
    for i in range(1, len(lows) - 1):
        if lows[i] <= lows[i - 1] and lows[i] <= lows[i + 1]:
            swing_lows.append(lows[i])
    min_l = float(np.min(lows))
    if min_l not in swing_lows:
        swing_lows.append(min_l)

    eql = None
    if len(swing_lows) >= 2:
        sorted_sl = sorted(swing_lows)
        if abs(sorted_sl[1] - sorted_sl[0]) / max(0.01, sorted_sl[0]) <= tolerance_pct:
            eql = round((sorted_sl[0] + sorted_sl[1]) / 2.0, 2)

    return eqh, eql


def find_major_swings_100d(
    df_window: pd.DataFrame,
    cmp_px: float,
    min_reversal_pct: float = 0.020,
) -> tuple[tuple[float, str] | None, tuple[float, str] | None]:
    """
    Scans a 100-session daily window for unmitigated structural swing turning points.
    Identifies major pivots where price reversed by at least min_reversal_pct (>= 2%).
    Returns:
      (nearest_major_bsl_px, nearest_major_bsl_desc), (nearest_major_ssl_px, nearest_major_ssl_desc)
    """
    if len(df_window) < 5 or cmp_px <= 0:
        return None, None
    highs = df_window["high"].to_numpy(dtype=float)
    lows = df_window["low"].to_numpy(dtype=float)
    closes = df_window["close"].to_numpy(dtype=float)
    n = len(df_window)

    pivots_h: list[tuple[float, int, float]] = []
    pivots_l: list[tuple[float, int, float]] = []

    for i in range(2, n - 2):
        h = highs[i]
        l = lows[i]
        # Fractal Swing High: higher than 2 bars before and 2 bars after
        if h >= highs[i - 1] and h >= highs[i - 2] and h >= highs[i + 1] and h >= highs[i + 2]:
            sub_low = np.min(lows[i + 1 : min(n, i + 9)])
            drop = (h - sub_low) / max(0.01, h)
            if drop >= min_reversal_pct:
                later_max = np.max(closes[i + 1 :]) if i + 1 < n else 0
                if later_max < h:  # Unmitigated by subsequent closes
                    days_ago = n - 1 - i
                    pivots_h.append((h, days_ago, drop))

        # Fractal Swing Low: lower than 2 bars before and 2 bars after
        if l <= lows[i - 1] and l <= lows[i - 2] and l <= lows[i + 1] and l <= lows[i + 2]:
            sub_high = np.max(highs[i + 1 : min(n, i + 9)])
            rally = (sub_high - l) / max(0.01, l)
            if rally >= min_reversal_pct:
                later_min = np.min(closes[i + 1 :]) if i + 1 < n else 999999
                if later_min > l:  # Unmitigated by subsequent closes
                    days_ago = n - 1 - i
                    pivots_l.append((l, days_ago, rally))

    # Overhead BSL pivots: above cmp_px, sorted ascending (nearest first)
    overhead = sorted([p for p in pivots_h if p[0] > cmp_px * 1.001], key=lambda x: x[0])
    # Underlying SSL pivots: below cmp_px, sorted descending (nearest first)
    underlying = sorted([p for p in pivots_l if p[0] < cmp_px * 0.999], key=lambda x: x[0], reverse=True)

    nearest_bsl = (round(overhead[0][0], 2), f"D-{overhead[0][1]} Major Swing High") if overhead else None
    nearest_ssl = (round(underlying[0][0], 2), f"D-{underlying[0][1]} Major Swing Low") if underlying else None

    return nearest_bsl, nearest_ssl


def get_htf_liquidity_levels(symbol: str, target_date: datetime.date | str) -> dict[str, Any] | None:
    """
    Compute 4-Tier Institutional Liquidity Matrix:
    1. Layer 1 (Fresh Intraday): Daily BSL (PDH) & Daily SSL (PDL)
    2. Layer 2 (Weekly Macro): Weekly BSL (PWH) & Weekly SSL (PWL)
    3. Layer 3 (100-Session Dynamic Swings): Nearest Unmitigated Pivots, 100D Macro High/Low & EQH/EQL
    4. Layer 4 (Institutional Option Chain): Max Call OI (Ceiling) & Max Put OI (Floor)
    """
    if isinstance(target_date, str):
        target_date = datetime.date.fromisoformat(target_date[:10])

    c1d = db.load_candles(symbol, "1D", 120)
    if not c1d or len(c1d) < 3:
        return None

    df = pd.DataFrame(c1d)
    df["date"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.date
    hist = df[df["date"] < target_date].reset_index(drop=True)
    if len(hist) < 3:
        return None

    # Layer 1: Fresh Daily BSL / SSL = T-1 (most recent completed day)
    pdh = float(hist.iloc[-1]["high"])
    pdl = float(hist.iloc[-1]["low"])
    cmp_px = float(hist.iloc[-1]["close"])

    # Layer 2: Weekly BSL / SSL = Prior 5 trading days before the current week
    prev_week = hist.iloc[-6:-1] if len(hist) >= 6 else hist.iloc[:-1]
    pwh = float(prev_week["high"].max())
    pwl = float(prev_week["low"].min())

    # Layer 3: 20-Day & 100-Session Multi-Day Swing Liquidity (EQH / EQL & Major Pivots)
    lookback_20 = hist.iloc[-min(20, len(hist)):]
    swing_bsl_20d = float(lookback_20["high"].max())
    swing_ssl_20d = float(lookback_20["low"].min())

    lookback_100 = hist.iloc[-min(100, len(hist)):]
    swing_bsl_100d = float(lookback_100["high"].max())
    swing_ssl_100d = float(lookback_100["low"].min())
    eqh, eql = find_equal_highs_lows(lookback_100)

    nearest_bsl, nearest_ssl = find_major_swings_100d(lookback_100, cmp_px)
    major_bsl_px = nearest_bsl[0] if nearest_bsl else None
    major_bsl_desc = nearest_bsl[1] if nearest_bsl else None
    major_ssl_px = nearest_ssl[0] if nearest_ssl else None
    major_ssl_desc = nearest_ssl[1] if nearest_ssl else None

    # Layer 4: Institutional Option Chain Liquidity (Max Call OI & Max Put OI)
    try:
        from app.playbook_engine import get_option_chain_sr
        last_px = float(hist.iloc[-1]["close"])
        opt_sr = get_option_chain_sr(symbol, last_px, "LONG", 100)
        max_call_oi = float(opt_sr.get("max_call_oi_strike", 0.0))
        max_put_oi = float(opt_sr.get("max_put_oi_strike", 0.0))
    except Exception:
        max_call_oi = 0.0
        max_put_oi = 0.0

    return {
        "symbol": symbol,
        "daily_bsl": round(pdh, 2),
        "daily_ssl": round(pdl, 2),
        "weekly_bsl": round(pwh, 2),
        "weekly_ssl": round(pwl, 2),
        "swing_bsl_20d": round(swing_bsl_20d, 2),
        "swing_ssl_20d": round(swing_ssl_20d, 2),
        "swing_bsl_100d": round(swing_bsl_100d, 2),
        "swing_ssl_100d": round(swing_ssl_100d, 2),
        "major_bsl_px": round(major_bsl_px, 2) if major_bsl_px else None,
        "major_bsl_desc": major_bsl_desc,
        "major_ssl_px": round(major_ssl_px, 2) if major_ssl_px else None,
        "major_ssl_desc": major_ssl_desc,
        "eqh": round(eqh, 2) if eqh else None,
        "eql": round(eql, 2) if eql else None,
        "max_call_oi": round(max_call_oi, 2) if max_call_oi > 0 else None,
        "max_put_oi": round(max_put_oi, 2) if max_put_oi > 0 else None,
    }


def identify_candle_pattern(bar: pd.Series, prev_bar: pd.Series | None = None) -> str:
    """
    Identifies institutional candlestick patterns on 5m bars:
    - Bullish Pin Bar (Hammer)
    - Shooting Star (Bearish Pin Bar)
    - Bullish Engulfing
    - Bearish Engulfing
    - Long Wick Rejection
    - Displacement Marubozu
    """
    o, h, l, c = float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])
    rng = max(0.05, h - l)
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    upper_wick_pct = round((upper_wick / rng) * 100, 1)
    lower_wick_pct = round((lower_wick / rng) * 100, 1)
    body_pct = round((body / rng) * 100, 1)

    is_green = c >= o
    is_red = c < o

    if lower_wick_pct >= 35.0 and upper_wick_pct <= 25.0 and (c >= l + 0.45 * rng):
        return f"Bullish Pin Bar (Hammer) [{lower_wick_pct}% Wick]"
    elif upper_wick_pct >= 35.0 and lower_wick_pct <= 25.0 and (c <= l + 0.55 * rng):
        return f"Shooting Star (Bearish Pin) [{upper_wick_pct}% Wick]"
    elif prev_bar is not None and is_green and float(prev_bar["close"]) < float(prev_bar["open"]):
        if c > float(prev_bar["open"]) and o <= float(prev_bar["close"]):
            return "Bullish Engulfing"
    elif prev_bar is not None and is_red and float(prev_bar["close"]) > float(prev_bar["open"]):
        if c < float(prev_bar["open"]) and o >= float(prev_bar["close"]):
            return "Bearish Engulfing"
    elif upper_wick_pct >= 28.0:
        return f"Bearish Rejection Wick [{upper_wick_pct}%]"
    elif lower_wick_pct >= 28.0:
        return f"Bullish Rejection Wick [{lower_wick_pct}%]"
    elif body_pct >= 60.0:
        return "Bullish Displacement" if is_green else "Bearish Displacement"
    return "Institutional Rejection"


def _evaluate_trade_outcome(
    df5: pd.DataFrame,
    entry_bar_idx: int,
    direction: str,
    entry_px: float,
    sl_px: float,
    target_px: float,
    reward_pts: float,
    risk_pts: float,
) -> dict[str, Any]:
    """
    Evaluates post-entry candles from entry_bar_idx + 1 up to the latest available bar.
    Enables instant reconstruction of what happened earlier in the day when the server starts late.
    """
    last_bar = df5.iloc[-1]
    cmp_px = round(float(last_bar["close"]), 2)
    cur_pnl = round((cmp_px - entry_px) if direction == "LONG" else (entry_px - cmp_px), 2)

    outcome = "RUNNING"
    hit_time = None

    for k in range(entry_bar_idx + 1, len(df5)):
        sub_bar = df5.iloc[k]
        sub_h = float(sub_bar["high"])
        sub_l = float(sub_bar["low"])
        sub_t = sub_bar["hhmm"]

        if direction == "LONG":
            if sub_l <= sl_px:
                outcome = "STOPPED_OUT"
                hit_time = f"{sub_t} IST"
                break
            elif sub_h >= target_px:
                outcome = "TARGET_HIT"
                hit_time = f"{sub_t} IST"
                break
        elif direction == "SHORT":
            if sub_h >= sl_px:
                outcome = "STOPPED_OUT"
                hit_time = f"{sub_t} IST"
                break
            elif sub_l <= target_px:
                outcome = "TARGET_HIT"
                hit_time = f"{sub_t} IST"
                break

    if outcome == "TARGET_HIT":
        outcome_label = f"🎯 Target Hit (+{reward_pts} pts @ {hit_time})"
        outcome_badge = "badge-target-hit"
    elif outcome == "STOPPED_OUT":
        outcome_label = f"❌ SL Hit (-{risk_pts} pts @ {hit_time})"
        outcome_badge = "badge-stopped-out"
    else:
        sign = "+" if cur_pnl >= 0 else ""
        outcome_label = f"⚡ Running ({sign}{cur_pnl} pts | CMP ₹{cmp_px})"
        outcome_badge = "badge-running-pos" if cur_pnl >= 0 else "badge-running-neg"

    return {
        "outcome": outcome,
        "outcome_label": outcome_label,
        "outcome_badge": outcome_badge,
        "outcome_time": hit_time,
        "cmp": cmp_px,
        "current_pnl": cur_pnl,
    }


def _hhmm_to_epoch(t_str: str, d: datetime.date) -> int:
    """Convert HH:MM string and date to epoch seconds in IST."""
    try:
        parts = t_str.split()[0].split(":")
        hh = int(parts[0])
        mm = int(parts[1])
        tz_ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
        dt = datetime.datetime(d.year, d.month, d.day, hh, mm, tzinfo=tz_ist)
        return int(dt.timestamp())
    except Exception:
        return 0


def scan_point_in_time_liquidity_setups(
    trade_date: str | datetime.date | None = None,
    min_rr: float = 2.0,
    min_reward_pts: float = 10.0,
    outcome_filter: str | None = None,
) -> list[dict[str, Any]]:
    """
    Simulate or evaluate candle-by-candle point-in-time liquidity setups across all F&O stocks.
    Walks forward from 09:15 AM to evaluate setups, patterns, volume signatures, and post-entry outcomes.
    """
    avail = get_available_liquidity_dates()
    if not trade_date:
        trade_date = avail[0] if avail else datetime.date.today()
    if isinstance(trade_date, str):
        try:
            trade_date = datetime.date.fromisoformat(trade_date[:10])
        except Exception:
            trade_date = datetime.date.today()

    # Convert trade_date to epoch range (09:15 to 15:30 IST)
    tz_ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    today_date = datetime.datetime.now(tz_ist).date()
    is_today = (trade_date == today_date)
    start_dt = datetime.datetime(trade_date.year, trade_date.month, trade_date.day, 9, 15, tzinfo=tz_ist)
    end_dt = datetime.datetime(trade_date.year, trade_date.month, trade_date.day, 15, 30, tzinfo=tz_ist)
    start_epoch = int(start_dt.timestamp())
    end_epoch = int(end_dt.timestamp())

    with db.connect() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT symbol FROM candles 
            WHERE timeframe = '5m' AND ts >= ? AND ts <= ?
            """,
            (start_epoch, end_epoch),
        )
        symbols = [r[0] for r in cursor.fetchall() if is_valid_fno_symbol(r[0])]

    setups: list[dict[str, Any]] = []

    for sym in symbols:
        levels = get_htf_liquidity_levels(sym, trade_date)
        if not levels:
            continue

        pdh = levels["daily_bsl"]
        pdl = levels["daily_ssl"]
        pwh = levels["weekly_bsl"]
        pwl = levels["weekly_ssl"]
        s20h = float(levels.get("swing_bsl_20d") or pwh)
        s20l = float(levels.get("swing_ssl_20d") or pwl)
        s100h = float(levels.get("swing_bsl_100d") or s20h)
        s100l = float(levels.get("swing_ssl_100d") or s20l)
        major_bsl = levels.get("major_bsl_px")
        major_bsl_desc = levels.get("major_bsl_desc") or "Major Swing High"
        major_ssl = levels.get("major_ssl_px")
        major_ssl_desc = levels.get("major_ssl_desc") or "Major Swing Low"
        eqh = levels.get("eqh")
        eql = levels.get("eql")
        max_ce = levels.get("max_call_oi")
        max_pe = levels.get("max_put_oi")

        with db.connect() as conn:
            c5_rows = conn.execute(
                """
                SELECT ts, open, high, low, close, volume FROM candles 
                WHERE symbol = ? AND timeframe = '5m' AND ts >= ? AND ts <= ?
                ORDER BY ts ASC
                """,
                (sym, start_epoch, end_epoch),
            ).fetchall()

        if len(c5_rows) < 2:
            continue

        df5 = pd.DataFrame([dict(r) for r in c5_rows])
        df5["datetime"] = pd.to_datetime(df5["ts"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
        df5["hhmm"] = df5["datetime"].dt.strftime("%H:%M")

        # Rolling 10-bar median volume for RVOL
        df5["vol_med"] = df5["volume"].rolling(10, min_periods=1).median().replace(0, 1.0)
        df5["rvol"] = df5["volume"] / df5["vol_med"]

        # Intraday cumulative VWAP
        cum_vol = df5["volume"].cumsum()
        cum_vol_px = (df5["volume"] * (df5["high"] + df5["low"] + df5["close"]) / 3.0).cumsum()
        df5["vwap"] = (cum_vol_px / cum_vol.replace(0, 1.0)).round(2)

        day_open = float(df5.iloc[0]["open"])
        # Institutional Trend-Day Gap Filter:
        # A deep gap beyond PDH/PDL signals a one-way liquidation or squeeze regime.
        # Counter-trend sweeps on such days are dangerous retail traps (e.g. POLYCAB).
        is_deep_gap_down = day_open < (pdl * 0.992)
        is_deep_gap_up = day_open > (pdh * 1.008)

        # Candle-by-candle walk-forward evaluation from 09:15 AM
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
            b_close = float(bar["close"])
            b_vwap = float(bar["vwap"])

            # ==========================================================
            # SETUP 1: BSL Sweep & Reversal (SHORT)
            # Must not be runaway gap up and not extended > 0.4% above VWAP
            # ==========================================================
            swept_bsl = None
            if not is_deep_gap_up and b_close <= (b_vwap * 1.004):
                if eqh and bar["high"] >= eqh and prev_bar["high"] < eqh:
                    swept_bsl = (f"Major EQH Sweep (₹{eqh:,.2f})", eqh)
                elif max_ce and bar["high"] >= max_ce and prev_bar["high"] < max_ce:
                    swept_bsl = (f"Max Call OI Sweep (₹{max_ce:,.0f})", max_ce)
                elif major_bsl and bar["high"] >= major_bsl and prev_bar["high"] < major_bsl:
                    swept_bsl = (f"{major_bsl_desc} Sweep (₹{major_bsl:,.2f})", major_bsl)
                elif s100h > pwh and bar["high"] >= s100h and prev_bar["high"] < s100h:
                    swept_bsl = (f"100D Macro High Sweep (₹{s100h:,.2f})", s100h)
                elif s20h > pwh and bar["high"] >= s20h and prev_bar["high"] < s20h:
                    swept_bsl = (f"20D Swing BSL Sweep (₹{s20h:,.2f})", s20h)
                elif bar["high"] >= pwh and prev_bar["high"] < pwh:
                    swept_bsl = ("Weekly BSL (PWH)", pwh)
                elif bar["high"] >= pdh and prev_bar["high"] < pdh:
                    swept_bsl = ("Daily BSL (PDH)", pdh)

            if swept_bsl:
                lvl_name, lvl_px = swept_bsl
                # Institutional Reversal Confirmation:
                # Must have upper rejection wick >= 20% AND either:
                # 1. Bearish Red body closing in bottom half below level, OR
                # 2. Next bar confirms by closing below the sweep bar's low.
                is_rejection = (b_upper_wick / b_rng >= 0.20) and (
                    (bar["close"] <= bar["open"] and bar["close"] <= bar["low"] + 0.55 * b_rng and bar["close"] < lvl_px)
                    or (next_bar is not None and next_bar["close"] < bar["low"])
                )
                if is_rejection:
                    entry_px = round(float(next_bar["open"]) if next_bar is not None else float(bar["close"]), 2)
                    entry_time = next_bar["hhmm"] if next_bar is not None else bar["hhmm"]
                    entry_bar_idx = (i + 1) if has_next else i
                    sl_px = round(float(bar["high"]) + max(0.5, entry_px * 0.0025), 2)
                    # Nearest Underlying Barrier Rule:
                    underlying_barriers = [p for p in [max_pe, major_ssl, eql, pwl, pdl] if p and p < entry_px * 0.998]
                    target_px = round(max(underlying_barriers) if underlying_barriers else (pdl if "Daily" in lvl_name else pwl), 2)
                    risk_pts = round(sl_px - entry_px, 2)
                    reward_pts = round(entry_px - target_px, 2)

                    if risk_pts > 0 and reward_pts >= min_reward_pts and (reward_pts / risk_pts) >= min_rr:
                        outcome_info = _evaluate_trade_outcome(
                            df5, entry_bar_idx, "SHORT", entry_px, sl_px, target_px, reward_pts, risk_pts
                        )
                        pat_name = identify_candle_pattern(bar, prev_bar)
                        rvol_val = round(float(bar["rvol"]), 1)
                        vol_sig = f"RVOL {rvol_val}x (Absorption Surge)" if rvol_val >= 1.2 else f"RVOL {rvol_val}x"

                        setups.append({
                            "symbol": sym,
                            "trade_date": trade_date.isoformat(),
                            "direction": "SHORT",
                            "setup_name": "BSL Sweep & Reversal",
                            "level_type": lvl_name,
                            "level_price": lvl_px,
                            "trigger_time": f"{entry_time} IST",
                            "epoch_ts": _hhmm_to_epoch(entry_time, trade_date),
                            "entry_price": entry_px,
                            "stop_loss": sl_px,
                            "target_price": target_px,
                            "risk_pts": risk_pts,
                            "reward_pts": reward_pts,
                            "rr_ratio": f"1:{round(reward_pts / risk_pts, 1)}",
                            "rvol": rvol_val,
                            "candle_pattern": pat_name,
                            "volume_signature": vol_sig,
                            "notes": f"Swept {lvl_name} ₹{lvl_px} with {pat_name}. Target Opposite SSL.",
                            **outcome_info,
                        })
                        found_for_sym = True
                        continue

            # ==========================================================
            # SETUP 2: SSL Sweep & Reversal (LONG)
            # Must not be runaway gap down and not extended > 0.4% below VWAP (knife catch)
            # ==========================================================
            swept_ssl = None
            if not is_deep_gap_down and b_close >= (b_vwap * 0.996):
                if eql and bar["low"] <= eql and prev_bar["low"] > eql:
                    swept_ssl = (f"Major EQL Sweep (₹{eql:,.2f})", eql)
                elif max_pe and bar["low"] <= max_pe and prev_bar["low"] > max_pe:
                    swept_ssl = (f"Max Put OI Sweep (₹{max_pe:,.0f})", max_pe)
                elif major_ssl and bar["low"] <= major_ssl and prev_bar["low"] > major_ssl:
                    swept_ssl = (f"{major_ssl_desc} Sweep (₹{major_ssl:,.2f})", major_ssl)
                elif s100l < pwl and bar["low"] <= s100l and prev_bar["low"] > s100l:
                    swept_ssl = (f"100D Macro Low Sweep (₹{s100l:,.2f})", s100l)
                elif s20l < pwl and bar["low"] <= s20l and prev_bar["low"] > s20l:
                    swept_ssl = (f"20D Swing SSL Sweep (₹{s20l:,.2f})", s20l)
                elif bar["low"] <= pwl and prev_bar["low"] > pwl:
                    swept_ssl = ("Weekly SSL (PWL)", pwl)
                elif bar["low"] <= pdl and prev_bar["low"] > pdl:
                    swept_ssl = ("Daily SSL (PDL)", pdl)

            if swept_ssl:
                lvl_name, lvl_px = swept_ssl
                # Institutional Reversal Confirmation:
                # Must have lower rejection wick >= 20% AND either:
                # 1. Bullish Green body closing in top half above level, OR
                # 2. Next bar confirms by closing above the sweep bar's high.
                is_rejection = (b_lower_wick / b_rng >= 0.20) and (
                    (bar["close"] >= bar["open"] and bar["close"] >= bar["low"] + 0.45 * b_rng and bar["close"] > lvl_px)
                    or (next_bar is not None and next_bar["close"] > bar["high"])
                )
                if is_rejection:
                    entry_px = round(float(next_bar["open"]) if next_bar is not None else float(bar["close"]), 2)
                    entry_time = next_bar["hhmm"] if next_bar is not None else bar["hhmm"]
                    entry_bar_idx = (i + 1) if has_next else i
                    sl_px = round(float(bar["low"]) - max(0.5, entry_px * 0.0025), 2)
                    # Nearest Overhead Barrier Rule:
                    overhead_barriers = [p for p in [max_ce, major_bsl, eqh, pwh, pdh] if p and p > entry_px * 1.002]
                    target_px = round(min(overhead_barriers) if overhead_barriers else (pdh if "Daily" in lvl_name else pwh), 2)
                    risk_pts = round(entry_px - sl_px, 2)
                    reward_pts = round(target_px - entry_px, 2)
                    entry_bar_idx = (i + 1) if has_next else i
                    sl_px = round(float(bar["low"]) - max(0.5, entry_px * 0.0025), 2)
                    target_px = round(pdh if "Daily" in lvl_name else pwh, 2)
                    risk_pts = round(entry_px - sl_px, 2)
                    reward_pts = round(target_px - entry_px, 2)

                    if risk_pts > 0 and reward_pts >= min_reward_pts and (reward_pts / risk_pts) >= min_rr:
                        outcome_info = _evaluate_trade_outcome(
                            df5, entry_bar_idx, "LONG", entry_px, sl_px, target_px, reward_pts, risk_pts
                        )
                        pat_name = identify_candle_pattern(bar, prev_bar)
                        rvol_val = round(float(bar["rvol"]), 1)
                        vol_sig = f"RVOL {rvol_val}x (Absorption Surge)" if rvol_val >= 1.2 else f"RVOL {rvol_val}x"

                        setups.append({
                            "symbol": sym,
                            "trade_date": trade_date.isoformat(),
                            "direction": "LONG",
                            "setup_name": "SSL Sweep & Reversal",
                            "level_type": lvl_name,
                            "level_price": lvl_px,
                            "trigger_time": f"{entry_time} IST",
                            "epoch_ts": _hhmm_to_epoch(entry_time, trade_date),
                            "entry_price": entry_px,
                            "stop_loss": sl_px,
                            "target_price": target_px,
                            "risk_pts": risk_pts,
                            "reward_pts": reward_pts,
                            "rr_ratio": f"1:{round(reward_pts / risk_pts, 1)}",
                            "rvol": rvol_val,
                            "candle_pattern": pat_name,
                            "volume_signature": vol_sig,
                            "notes": f"Swept {lvl_name} ₹{lvl_px} with {pat_name}. Target Opposite BSL.",
                            **outcome_info,
                        })
                        found_for_sym = True
                        continue

            # ==========================================================
            # SETUP 3: Breakdown & Retest Continuation (SHORT)
            # Must have clean displacement (>= 0.4% below level with red body)
            # Retest must fail below level and show Bearish Rejection
            # ==========================================================
            lvl_break = None
            if max_pe and prev_bar["close"] >= max_pe and bar["close"] < max_pe * 0.996 and bar["close"] < bar["open"]:
                lvl_break = (f"Max Put OI Breakdown (₹{max_pe:,.0f})", max_pe)
            elif major_ssl and prev_bar["close"] >= major_ssl and bar["close"] < major_ssl * 0.996 and bar["close"] < bar["open"]:
                lvl_break = (f"{major_ssl_desc} (₹{major_ssl:,.2f})", major_ssl)
            elif s100l < pwl and prev_bar["close"] >= s100l and bar["close"] < s100l * 0.996 and bar["close"] < bar["open"]:
                lvl_break = (f"100D Macro Low (₹{s100l:,.2f})", s100l)
            elif s20l < pwl and prev_bar["close"] >= s20l and bar["close"] < s20l * 0.996 and bar["close"] < bar["open"]:
                lvl_break = (f"20D Swing SSL (₹{s20l:,.2f})", s20l)
            elif prev_bar["close"] >= pwl and bar["close"] < pwl * 0.996 and bar["close"] < bar["open"]:
                lvl_break = ("Weekly SSL (PWL)", pwl)
            elif prev_bar["close"] >= pdl and bar["close"] < pdl * 0.996 and bar["close"] < pdl * 0.996 and bar["close"] < bar["open"]:
                lvl_break = ("Daily SSL (PDL)", pdl)

            if lvl_break:
                lvl_name, lvl_px = lvl_break
                # Find retest within next 1 to 5 bars
                for j in range(i + 1, min(len(df5), i + 6)):
                    r_bar = df5.iloc[j]
                    r_rng = max(0.05, float(r_bar["high"]) - float(r_bar["low"]))
                    r_uwick = float(r_bar["high"]) - max(float(r_bar["open"]), float(r_bar["close"]))
                    r_is_bearish = (float(r_bar["close"]) <= float(r_bar["open"])) or (r_uwick / r_rng >= 0.20)

                    if r_bar["high"] >= lvl_px * 0.998 and r_bar["close"] < lvl_px and r_is_bearish:
                        entry_px = round(float(r_bar["close"]), 2)
                        entry_time = r_bar["hhmm"]
                        entry_bar_idx = j
                        sl_px = round(float(r_bar["high"]) + max(0.5, entry_px * 0.0025), 2)
                        day_rng = abs(pdh - pdl)
                        underlying_targets = [p for p in [max_pe, major_ssl, s100l, eql] if p and p < entry_px * 0.992]
                        target_px = round(max(underlying_targets) if underlying_targets else (entry_px - max(day_rng, entry_px * 0.02)), 2)
                        risk_pts = round(sl_px - entry_px, 2)
                        reward_pts = round(entry_px - target_px, 2)

                        if risk_pts > 0 and reward_pts >= min_reward_pts and (reward_pts / risk_pts) >= min_rr:
                            outcome_info = _evaluate_trade_outcome(
                                df5, entry_bar_idx, "SHORT", entry_px, sl_px, target_px, reward_pts, risk_pts
                            )
                            pat_name = identify_candle_pattern(r_bar, bar)
                            r_rvol = round(float(r_bar["rvol"]), 1)
                            bk_rvol = round(float(bar["rvol"]), 1)
                            vol_sig = f"Dry Retest ({r_rvol}x vs Bk {bk_rvol}x)" if r_rvol <= bk_rvol else f"RVOL {r_rvol}x"

                            setups.append({
                                "symbol": sym,
                                "trade_date": trade_date.isoformat(),
                                "direction": "SHORT",
                                "setup_name": "Breakdown & Retest",
                                "level_type": lvl_name,
                                "level_price": lvl_px,
                                "trigger_time": f"{entry_time} IST",
                                "epoch_ts": _hhmm_to_epoch(entry_time, trade_date),
                                "entry_price": entry_px,
                                "stop_loss": sl_px,
                                "target_price": target_px,
                                "risk_pts": risk_pts,
                                "reward_pts": reward_pts,
                                "rr_ratio": f"1:{round(reward_pts / risk_pts, 1)}",
                                "rvol": r_rvol,
                                "candle_pattern": pat_name,
                                "volume_signature": vol_sig,
                                "notes": f"Clean breakdown below {lvl_name} ₹{lvl_px} + {pat_name} retest.",
                                **outcome_info,
                            })
                            found_for_sym = True
                            break

            # ==========================================================
            # SETUP 4: Breakout & Retest Continuation (LONG)
            # Must have clean displacement (>= 0.4% above level with green body)
            # Retest must hold above level and show Bullish Rejection
            # ==========================================================
            lvl_brout = None
            if max_ce and prev_bar["close"] <= max_ce and bar["close"] > max_ce * 1.004 and bar["close"] > bar["open"]:
                lvl_brout = (f"Max Call OI Squeeze (₹{max_ce:,.0f})", max_ce)
            elif major_bsl and prev_bar["close"] <= major_bsl and bar["close"] > major_bsl * 1.004 and bar["close"] > bar["open"]:
                lvl_brout = (f"{major_bsl_desc} (₹{major_bsl:,.2f})", major_bsl)
            elif s100h > pwh and prev_bar["close"] <= s100h and bar["close"] > s100h * 1.004 and bar["close"] > bar["open"]:
                lvl_brout = (f"100D Macro High (₹{s100h:,.2f})", s100h)
            elif s20h > pwh and prev_bar["close"] <= s20h and bar["close"] > s20h * 1.004 and bar["close"] > bar["open"]:
                lvl_brout = (f"20D Swing BSL (₹{s20h:,.2f})", s20h)
            elif prev_bar["close"] <= pwh and bar["close"] > pwh * 1.004 and bar["close"] > bar["open"]:
                lvl_brout = ("Weekly BSL (PWH)", pwh)
            elif prev_bar["close"] <= pdh and bar["close"] > pdh * 1.004 and bar["close"] > bar["open"]:
                lvl_brout = ("Daily BSL (PDH)", pdh)

            if lvl_brout:
                lvl_name, lvl_px = lvl_brout
                for j in range(i + 1, min(len(df5), i + 6)):
                    r_bar = df5.iloc[j]
                    r_rng = max(0.05, float(r_bar["high"]) - float(r_bar["low"]))
                    r_lwick = min(float(r_bar["open"]), float(r_bar["close"])) - float(r_bar["low"])
                    r_is_bullish = (float(r_bar["close"]) >= float(r_bar["open"])) or (r_lwick / r_rng >= 0.20)

                    if r_bar["low"] <= lvl_px * 1.002 and r_bar["close"] > lvl_px and r_is_bullish:
                        entry_px = round(float(r_bar["close"]), 2)
                        entry_time = r_bar["hhmm"]
                        entry_bar_idx = j
                        sl_px = round(float(r_bar["low"]) - max(0.5, entry_px * 0.0025), 2)
                        day_rng = abs(pdh - pdl)
                        overhead_targets = [p for p in [max_ce, major_bsl, s100h, eqh] if p and p > entry_px * 1.008]
                        target_px = round(min(overhead_targets) if overhead_targets else (entry_px + max(day_rng, entry_px * 0.02)), 2)
                        risk_pts = round(entry_px - sl_px, 2)
                        reward_pts = round(target_px - entry_px, 2)

                        if risk_pts > 0 and reward_pts >= min_reward_pts and (reward_pts / risk_pts) >= min_rr:
                            outcome_info = _evaluate_trade_outcome(
                                df5, entry_bar_idx, "LONG", entry_px, sl_px, target_px, reward_pts, risk_pts
                            )
                            pat_name = identify_candle_pattern(r_bar, bar)
                            r_rvol = round(float(r_bar["rvol"]), 1)
                            bk_rvol = round(float(bar["rvol"]), 1)
                            vol_sig = f"Dry Retest ({r_rvol}x vs Bk {bk_rvol}x)" if r_rvol <= bk_rvol else f"RVOL {r_rvol}x"

                            setups.append({
                                "symbol": sym,
                                "trade_date": trade_date.isoformat(),
                                "direction": "LONG",
                                "setup_name": "Breakout & Retest",
                                "level_type": lvl_name,
                                "level_price": lvl_px,
                                "trigger_time": f"{entry_time} IST",
                                "epoch_ts": _hhmm_to_epoch(entry_time, trade_date),
                                "entry_price": entry_px,
                                "stop_loss": sl_px,
                                "target_price": target_px,
                                "risk_pts": risk_pts,
                                "reward_pts": reward_pts,
                                "rr_ratio": f"1:{round(reward_pts / risk_pts, 1)}",
                                "rvol": r_rvol,
                                "candle_pattern": pat_name,
                                "volume_signature": vol_sig,
                                "notes": f"Clean breakout above {lvl_name} ₹{lvl_px} + {pat_name} retest.",
                                **outcome_info,
                            })
                            found_for_sym = True
                            break

        # ==========================================================
        # SETUP 5: Live Market Approaching Radar Alert (4-Layer Matrix)
        # If no setup triggered yet, check latest bar proximity across all 4 layers (within 0.35%)
        # ==========================================================
        if not found_for_sym and len(df5) >= 1:
            last_bar = df5.iloc[-1]
            prev_last_bar = df5.iloc[-2] if len(df5) >= 2 else None
            last_px = float(last_bar["close"])
            t_now = last_bar["hhmm"]

            approaching_candidates = []
            if eqh:
                d = abs(last_px - eqh) / eqh
                if d <= 0.0035: approaching_candidates.append((f"Major EQH (₹{eqh:,.2f})", eqh, "WATCH SHORT", "BSL", round(d * 100, 2)))
            if max_ce and max_ce > 0:
                d = abs(last_px - max_ce) / max_ce
                if d <= 0.0035: approaching_candidates.append((f"Max Call OI (₹{max_ce:,.0f})", max_ce, "WATCH SHORT", "BSL", round(d * 100, 2)))
            if major_bsl and major_bsl > 0:
                d = abs(last_px - major_bsl) / major_bsl
                if d <= 0.0035: approaching_candidates.append((f"{major_bsl_desc} (₹{major_bsl:,.2f})", major_bsl, "WATCH SHORT", "BSL", round(d * 100, 2)))
            if s100h and s100h > pwh:
                d = abs(last_px - s100h) / s100h
                if d <= 0.0035: approaching_candidates.append((f"100D Macro High (₹{s100h:,.2f})", s100h, "WATCH SHORT", "BSL", round(d * 100, 2)))
            if s20h and s20h > pwh:
                d = abs(last_px - s20h) / s20h
                if d <= 0.0035: approaching_candidates.append((f"20D Swing High (₹{s20h:,.2f})", s20h, "WATCH SHORT", "BSL", round(d * 100, 2)))
            d_pwh = abs(last_px - pwh) / pwh
            if d_pwh <= 0.0035: approaching_candidates.append(("Weekly BSL (PWH)", pwh, "WATCH SHORT", "BSL", round(d_pwh * 100, 2)))
            d_pdh = abs(last_px - pdh) / pdh
            if d_pdh <= 0.0035: approaching_candidates.append(("Daily BSL (PDH)", pdh, "WATCH SHORT", "BSL", round(d_pdh * 100, 2)))

            if eql:
                d = abs(last_px - eql) / eql
                if d <= 0.0035: approaching_candidates.append((f"Major EQL (₹{eql:,.2f})", eql, "WATCH LONG", "SSL", round(d * 100, 2)))
            if max_pe and max_pe > 0:
                d = abs(last_px - max_pe) / max_pe
                if d <= 0.0035: approaching_candidates.append((f"Max Put OI (₹{max_pe:,.0f})", max_pe, "WATCH LONG", "SSL", round(d * 100, 2)))
            if major_ssl and major_ssl > 0:
                d = abs(last_px - major_ssl) / major_ssl
                if d <= 0.0035: approaching_candidates.append((f"{major_ssl_desc} (₹{major_ssl:,.2f})", major_ssl, "WATCH LONG", "SSL", round(d * 100, 2)))
            if s100l and s100l < pwl:
                d = abs(last_px - s100l) / s100l
                if d <= 0.0035: approaching_candidates.append((f"100D Macro Low (₹{s100l:,.2f})", s100l, "WATCH LONG", "SSL", round(d * 100, 2)))
            if s20l and s20l < pwl:
                d = abs(last_px - s20l) / s20l
                if d <= 0.0035: approaching_candidates.append((f"20D Swing Low (₹{s20l:,.2f})", s20l, "WATCH LONG", "SSL", round(d * 100, 2)))
            d_pwl = abs(last_px - pwl) / pwl
            if d_pwl <= 0.0035: approaching_candidates.append(("Weekly SSL (PWL)", pwl, "WATCH LONG", "SSL", round(d_pwl * 100, 2)))
            d_pdl = abs(last_px - pdl) / pdl
            if d_pdl <= 0.0035: approaching_candidates.append(("Daily SSL (PDL)", pdl, "WATCH LONG", "SSL", round(d_pdl * 100, 2)))

            approaching = min(approaching_candidates, key=lambda c: c[4]) if approaching_candidates else None

            if approaching:
                lvl_name, lvl_px, direction, l_kind, pct_dist = approaching
                pat_name = identify_candle_pattern(last_bar, prev_last_bar)
                rvol_val = round(float(last_bar.get("rvol", 1.0)), 1)

                setups.append({
                    "symbol": sym,
                    "trade_date": trade_date.isoformat(),
                    "direction": direction,
                    "setup_name": f"⚡ Approaching {l_kind}",
                    "level_type": lvl_name,
                    "level_price": lvl_px,
                    "trigger_time": f"{t_now} LIVE" if is_today else f"{t_now} EOD",
                    "epoch_ts": _hhmm_to_epoch(t_now, trade_date),
                    "entry_price": last_px,
                    "stop_loss": round(lvl_px * (1.003 if "BSL" in l_kind else 0.997), 2),
                    "target_price": round(pdl if "BSL" in l_kind else pdh, 2),
                    "risk_pts": round(abs(last_px - lvl_px), 2),
                    "reward_pts": round(abs(last_px - (pdl if "BSL" in l_kind else pdh)), 2),
                    "rr_ratio": "Live Tracking",
                    "rvol": rvol_val,
                    "candle_pattern": pat_name,
                    "volume_signature": f"Live RVOL {rvol_val}x",
                    "notes": f"Price within {pct_dist}% of {lvl_name} ₹{lvl_px}. Pattern: {pat_name}.",
                    "outcome": "APPROACHING",
                    "outcome_label": f"⚡ Proximity Alert ({pct_dist}% away)" if is_today else f"⚡ EOD Proximity ({pct_dist}% away)",
                    "outcome_badge": "badge-approaching",
                    "outcome_time": None,
                    "cmp": last_px,
                    "current_pnl": 0.0,
                })

    # Filter by outcome if requested
    if outcome_filter and outcome_filter != "ALL":
        u_filt = outcome_filter.upper()
        if u_filt == "ACTIVE":
            setups = [s for s in setups if s.get("outcome") in ("RUNNING", "APPROACHING")]
        else:
            setups = [s for s in setups if s.get("outcome") == u_filt]

    # Sort setups reverse-chronologically with newest / latest on top (LIVE alerts and latest triggers first)
    def _trigger_sort_key(s):
        epoch = int(s.get("epoch_ts", 0))
        is_live = 1 if "LIVE" in str(s.get("trigger_time", "")) else 0
        return (epoch, is_live)

    setups.sort(key=_trigger_sort_key, reverse=True)
    return setups
