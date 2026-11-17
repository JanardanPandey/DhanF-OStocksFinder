from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.indicators import add_indicators, detect_swings


def analyze_structure(df: pd.DataFrame, swing_window: int = 3) -> dict[str, Any]:
    """
    Analyze market structure, trend, S/R zones, Order Blocks, FVGs, and liquidity sweeps.
    """
    if df.empty or len(df) < 30:
        return {
            "trend": "RANGING",
            "trend_strength": 0,
            "structure_label": "INSUFFICIENT_DATA",
            "swing_highs": [],
            "swing_lows": [],
            "bos": [],
            "choch": [],
            "sr_zones": [],
            "order_blocks": [],
            "fvgs": [],
            "sweeps": [],
            "atr": 0.0,
            "rsi": 50.0,
            "vwap": 0.0,
            "ema20": 0.0,
            "ema50": 0.0,
            "ltp": 0.0,
        }

    work = add_indicators(df)
    swing_highs, swing_lows = detect_swings(work, left=swing_window, right=swing_window)

    atr_now = float(work["atr"].iloc[-1])
    ltp = float(work["close"].iloc[-1])
    ema20 = float(work["ema20"].iloc[-1])
    ema50 = float(work["ema50"].iloc[-1])
    rsi_now = float(work["rsi"].iloc[-1])
    vwap_now = float(work["vwap"].iloc[-1])

    bos_events, choch_events, struct_trend, struct_label = _classify_trend_and_breaks(
        work, swing_highs, swing_lows, ema20, ema50, ltp
    )
    sr_zones = _cluster_sr_zones(swing_highs, swing_lows, atr_now, ltp)
    fvgs = _detect_fvgs(work)
    order_blocks = _detect_order_blocks(work, swing_highs, swing_lows, fvgs)
    sweeps = _detect_liquidity_sweeps(work, swing_highs, swing_lows)

    trend_strength = _score_trend_strength(struct_trend, struct_label, ema20, ema50, ltp, bos_events)

    return {
        "trend": struct_trend,
        "trend_strength": trend_strength,
        "structure_label": struct_label,
        "swing_highs": swing_highs[-10:],
        "swing_lows": swing_lows[-10:],
        "bos": bos_events[-6:],
        "choch": choch_events[-4:],
        "sr_zones": sr_zones,
        "order_blocks": order_blocks,
        "fvgs": fvgs,
        "sweeps": sweeps,
        "atr": round(atr_now, 2),
        "rsi": round(rsi_now, 1),
        "vwap": round(vwap_now, 2),
        "ema20": round(ema20, 2),
        "ema50": round(ema50, 2),
        "ltp": round(ltp, 2),
    }


def _classify_trend_and_breaks(
    df: pd.DataFrame,
    swing_highs: list[dict],
    swing_lows: list[dict],
    ema20: float,
    ema50: float,
    ltp: float,
) -> tuple[list[dict], list[dict], str, str]:
    closes = df["close"].to_numpy()
    ts_arr = df["ts"].to_numpy()
    n = len(df)

    bos_events: list[dict] = []
    choch_events: list[dict] = []
    running_bias: str | None = None

    # Walk chronologically through swing breaks
    sh_idx = 0
    sl_idx = 0
    active_sh: dict | None = None
    active_sl: dict | None = None
    broken_sh_ids: set[int] = set()
    broken_sl_ids: set[int] = set()

    for i in range(n):
        while sh_idx < len(swing_highs) and swing_highs[sh_idx]["idx"] + 3 <= i:
            active_sh = swing_highs[sh_idx]
            sh_idx += 1
        while sl_idx < len(swing_lows) and swing_lows[sl_idx]["idx"] + 3 <= i:
            active_sl = swing_lows[sl_idx]
            sl_idx += 1

        if active_sh and active_sh["idx"] not in broken_sh_ids and closes[i] > active_sh["price"]:
            broken_sh_ids.add(active_sh["idx"])
            event = {
                "idx": i,
                "ts": int(ts_arr[i]),
                "direction": "BULLISH",
                "level": round(active_sh["price"], 2),
                "swing_idx": active_sh["idx"],
            }
            if running_bias == "BEARISH":
                choch_events.append({**event, "type": "CHoCH"})
            else:
                bos_events.append({**event, "type": "BOS"})
            running_bias = "BULLISH"

        if active_sl and active_sl["idx"] not in broken_sl_ids and closes[i] < active_sl["price"]:
            broken_sl_ids.add(active_sl["idx"])
            event = {
                "idx": i,
                "ts": int(ts_arr[i]),
                "direction": "BEARISH",
                "level": round(active_sl["price"], 2),
                "swing_idx": active_sl["idx"],
            }
            if running_bias == "BULLISH":
                choch_events.append({**event, "type": "CHoCH"})
            else:
                bos_events.append({**event, "type": "BOS"})
            running_bias = "BEARISH"

    # HH/HL vs LH/LL label from last 2 confirmed highs & lows
    label = "CONSOLIDATION"
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        hh = swing_highs[-1]["price"] > swing_highs[-2]["price"]
        hl = swing_lows[-1]["price"] > swing_lows[-2]["price"]
        lh = swing_highs[-1]["price"] < swing_highs[-2]["price"]
        ll = swing_lows[-1]["price"] < swing_lows[-2]["price"]
        if hh and hl:
            label = "HH/HL"
        elif lh and ll:
            label = "LH/LL"
        elif hh and not hl:
            label = "HH/LL_EXPANDING"
        elif lh and hl:
            label = "LH/HL_COIL"

    bull_votes = 0
    bear_votes = 0
    if label == "HH/HL":
        bull_votes += 2
    elif label == "LH/LL":
        bear_votes += 2

    if running_bias == "BULLISH":
        bull_votes += 1
    elif running_bias == "BEARISH":
        bear_votes += 1

    if ltp > ema20 > ema50:
        bull_votes += 1
    elif ltp < ema20 < ema50:
        bear_votes += 1

    if bull_votes >= 2 and bull_votes > bear_votes:
        trend = "BULLISH"
    elif bear_votes >= 2 and bear_votes > bull_votes:
        trend = "BEARISH"
    else:
        trend = "RANGING"

    return bos_events, choch_events, trend, label


def _score_trend_strength(
    trend: str, label: str, ema20: float, ema50: float, ltp: float, bos_events: list[dict]
) -> int:
    if trend == "RANGING":
        return 35
    score = 55
    if (trend == "BULLISH" and label == "HH/HL") or (trend == "BEARISH" and label == "LH/LL"):
        score += 20
    if (trend == "BULLISH" and ltp > ema20 > ema50) or (trend == "BEARISH" and ltp < ema20 < ema50):
        score += 15
    if bos_events and bos_events[-1]["direction"] == trend:
        score += 10
    return min(100, score)


def _cluster_sr_zones(
    swing_highs: list[dict], swing_lows: list[dict], atr: float, ltp: float
) -> list[dict[str, Any]]:
    tol = max(atr * 0.45, ltp * 0.0035)
    raw_points = [{"price": s["price"], "ts": s["ts"], "role": "R"} for s in swing_highs] + [
        {"price": s["price"], "ts": s["ts"], "role": "S"} for s in swing_lows
    ]
    if not raw_points:
        return []

    raw_points.sort(key=lambda x: x["price"])
    clusters: list[list[dict]] = []
    for pt in raw_points:
        if not clusters:
            clusters.append([pt])
            continue
        mean_px = float(np.mean([p["price"] for p in clusters[-1]]))
        if abs(pt["price"] - mean_px) <= tol:
            clusters[-1].append(pt)
        else:
            clusters.append([pt])

    zones: list[dict[str, Any]] = []
    for cl in clusters:
        prices = [p["price"] for p in cl]
        mid = float(np.mean(prices))
        half_band = max(tol * 0.35, (max(prices) - min(prices)) * 0.5, ltp * 0.001)
        kind = "RESISTANCE" if mid > ltp else "SUPPORT"
        touches = len(cl)
        zones.append(
            {
                "kind": kind,
                "low": round(mid - half_band, 2),
                "high": round(mid + half_band, 2),
                "mid": round(mid, 2),
                "touches": touches,
                "strength": min(100, 35 + touches * 20),
                "last_ts": max(p["ts"] for p in cl),
            }
        )

    # Keep strongest & closest zones
    zones.sort(key=lambda z: (-z["touches"], abs(z["mid"] - ltp)))
    return zones[:10]


def _detect_fvgs(df: pd.DataFrame) -> list[dict[str, Any]]:
    n = len(df)
    if n < 5:
        return []

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    atrs = df["atr"].to_numpy()
    ts_arr = df["ts"].to_numpy()

    fvgs: list[dict[str, Any]] = []
    for i in range(2, n):
        min_gap = max(atrs[i] * 0.12, closes[i] * 0.0008)
        # Bullish FVG
        if lows[i] > highs[i - 2] and (lows[i] - highs[i - 2]) >= min_gap:
            zone_low = float(highs[i - 2])
            zone_high = float(lows[i])
            filled = bool(np.any(lows[i + 1 :] <= zone_low)) if i + 1 < n else False
            if not filled:
                fvgs.append(
                    {
                        "direction": "BULLISH",
                        "idx": i - 1,
                        "ts": int(ts_arr[i - 1]),
                        "low": round(zone_low, 2),
                        "high": round(zone_high, 2),
                    }
                )
        # Bearish FVG
        elif highs[i] < lows[i - 2] and (lows[i - 2] - highs[i]) >= min_gap:
            zone_low = float(highs[i])
            zone_high = float(lows[i - 2])
            filled = bool(np.any(highs[i + 1 :] >= zone_high)) if i + 1 < n else False
            if not filled:
                fvgs.append(
                    {
                        "direction": "BEARISH",
                        "idx": i - 1,
                        "ts": int(ts_arr[i - 1]),
                        "low": round(zone_low, 2),
                        "high": round(zone_high, 2),
                    }
                )
    return fvgs[-8:]


def _detect_order_blocks(
    df: pd.DataFrame,
    swing_highs: list[dict],
    swing_lows: list[dict],
    fvgs: list[dict],
) -> list[dict[str, Any]]:
    n = len(df)
    if n < 15:
        return []

    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    atrs = df["atr"].to_numpy()
    rel_vols = df["rel_vol"].to_numpy()
    ts_arr = df["ts"].to_numpy()

    fvg_indices = {f["idx"] for f in fvgs}
    obs: list[dict[str, Any]] = []

    for i in range(2, n - 4):
        atr_i = max(float(atrs[i]), float(closes[i]) * 0.002)
        lookahead_end = min(n, i + 5)

        # Bullish OB: Bearish candle followed by impulsive upward displacement
        if closes[i] < opens[i]:
            fut_high = float(np.max(closes[i + 1 : lookahead_end]))
            displacement = fut_high - float(highs[i])
            max_vol = float(np.max(rel_vols[i + 1 : lookahead_end]))
            broke_swing = any(
                s["idx"] < i and fut_high > s["price"] and closes[i] <= s["price"]
                for s in swing_highs
            )
            has_fvg = any(idx in fvg_indices for idx in range(i + 1, lookahead_end))

            if displacement >= 1.15 * atr_i and (broke_swing or has_fvg or max_vol >= 1.3):
                ob_low = float(lows[i])
                ob_high = float(max(opens[i], highs[i] if (highs[i] - lows[i]) <= 1.8 * atr_i else opens[i]))
                # Check mitigation & invalidation after impulse
                post_slice_low = lows[i + 2 :]
                post_slice_close = closes[i + 2 :]
                if np.any(post_slice_close < ob_low):
                    continue  # Invalidated
                tapped = bool(np.any(post_slice_low <= ob_high))
                obs.append(
                    {
                        "direction": "BULLISH",
                        "idx": i,
                        "ts": int(ts_arr[i]),
                        "low": round(ob_low, 2),
                        "high": round(ob_high, 2),
                        "status": "TAPPED" if tapped else "FRESH",
                        "has_fvg": has_fvg,
                        "broke_swing": broke_swing,
                        "rel_vol": round(max_vol, 2),
                    }
                )

        # Bearish OB: Bullish candle followed by impulsive downward displacement
        elif closes[i] > opens[i]:
            fut_low = float(np.min(closes[i + 1 : lookahead_end]))
            displacement = float(lows[i]) - fut_low
            max_vol = float(np.max(rel_vols[i + 1 : lookahead_end]))
            broke_swing = any(
                s["idx"] < i and fut_low < s["price"] and closes[i] >= s["price"]
                for s in swing_lows
            )
            has_fvg = any(idx in fvg_indices for idx in range(i + 1, lookahead_end))

            if displacement >= 1.15 * atr_i and (broke_swing or has_fvg or max_vol >= 1.3):
                ob_high = float(highs[i])
                ob_low = float(min(opens[i], lows[i] if (highs[i] - lows[i]) <= 1.8 * atr_i else opens[i]))
                post_slice_high = highs[i + 2 :]
                post_slice_close = closes[i + 2 :]
                if np.any(post_slice_close > ob_high):
                    continue  # Invalidated
                tapped = bool(np.any(post_slice_high >= ob_low))
                obs.append(
                    {
                        "direction": "BEARISH",
                        "idx": i,
                        "ts": int(ts_arr[i]),
                        "low": round(ob_low, 2),
                        "high": round(ob_high, 2),
                        "status": "TAPPED" if tapped else "FRESH",
                        "has_fvg": has_fvg,
                        "broke_swing": broke_swing,
                        "rel_vol": round(max_vol, 2),
                    }
                )

    # Deduplicate overlapping OBs, keeping most recent
    deduped: list[dict[str, Any]] = []
    for ob in reversed(obs):
        overlap = any(
            d["direction"] == ob["direction"]
            and not (ob["high"] < d["low"] or ob["low"] > d["high"])
            for d in deduped
        )
        if not overlap:
            deduped.append(ob)
    return list(reversed(deduped[-6:]))


def _detect_liquidity_sweeps(
    df: pd.DataFrame, swing_highs: list[dict], swing_lows: list[dict]
) -> list[dict[str, Any]]:
    n = len(df)
    if n < 15:
        return []

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    ts_arr = df["ts"].to_numpy()

    sweeps: list[dict[str, Any]] = []
    start = max(5, n - 12)
    for i in range(start, n):
        # Bullish sweep: wick below a prior swing low, close back above it
        for sl in swing_lows:
            if sl["idx"] < i - 2 and lows[i] < sl["price"] and closes[i] > sl["price"]:
                sweeps.append(
                    {
                        "direction": "BULLISH",
                        "idx": i,
                        "ts": int(ts_arr[i]),
                        "swept_level": round(sl["price"], 2),
                        "wick_extreme": round(float(lows[i]), 2),
                    }
                )
                break
        # Bearish sweep: wick above a prior swing high, close back below it
        for sh in swing_highs:
            if sh["idx"] < i - 2 and highs[i] > sh["price"] and closes[i] < sh["price"]:
                sweeps.append(
                    {
                        "direction": "BEARISH",
                        "idx": i,
                        "ts": int(ts_arr[i]),
                        "swept_level": round(sh["price"], 2),
                        "wick_extreme": round(float(highs[i]), 2),
                    }
                )
                break
    return sweeps[-4:]
