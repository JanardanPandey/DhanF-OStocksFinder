from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.structure_engine import analyze_structure


def _compute_pdh_pdl(df_daily: pd.DataFrame) -> dict[str, float] | None:
    if len(df_daily) < 2:
        return None
    prev = df_daily.iloc[-2]
    return {
        "pdh": round(float(prev["high"]), 2),
        "pdl": round(float(prev["low"]), 2),
        "pdc": round(float(prev["close"]), 2),
    }


def _find_nearest_poi(
    ltp: float,
    direction: str,
    poi_struct: dict[str, Any],
    sr_zones: list[dict[str, Any]],
    pdh_pdl: dict[str, float] | None,
) -> dict[str, Any] | None:
    """Find the best unmitigated Order Block, FVG, or S/R zone near LTP."""
    candidates: list[dict[str, Any]] = []

    for ob in poi_struct.get("order_blocks", []):
        if ob["direction"] != direction:
            continue
        mid = (ob["low"] + ob["high"]) / 2.0
        dist_pct = abs(ltp - mid) / ltp * 100.0
        if dist_pct <= 2.8:
            # Prefer price inside or slightly above bullish OB / below bearish OB
            in_zone = ob["low"] * 0.996 <= ltp <= ob["high"] * 1.006
            score_bonus = 18 if ob["status"] == "FRESH" else 12
            if ob.get("has_fvg"):
                score_bonus += 8
            if in_zone:
                score_bonus += 10
            candidates.append(
                {
                    "type": "ORDER_BLOCK",
                    "low": ob["low"],
                    "high": ob["high"],
                    "status": ob["status"],
                    "has_fvg": ob.get("has_fvg", False),
                    "in_zone": in_zone,
                    "dist_pct": dist_pct,
                    "priority": score_bonus - dist_pct * 3.0,
                }
            )

    for fvg in poi_struct.get("fvgs", []):
        if fvg["direction"] != direction:
            continue
        mid = (fvg["low"] + fvg["high"]) / 2.0
        dist_pct = abs(ltp - mid) / ltp * 100.0
        if dist_pct <= 2.2:
            in_zone = fvg["low"] * 0.997 <= ltp <= fvg["high"] * 1.003
            candidates.append(
                {
                    "type": "FVG",
                    "low": fvg["low"],
                    "high": fvg["high"],
                    "status": "FRESH",
                    "has_fvg": True,
                    "in_zone": in_zone,
                    "dist_pct": dist_pct,
                    "priority": (18 if in_zone else 10) - dist_pct * 3.0,
                }
            )

    target_kind = "SUPPORT" if direction == "BULLISH" else "RESISTANCE"
    for sr in sr_zones:
        if sr["kind"] != target_kind and abs(sr["mid"] - ltp) / ltp * 100.0 > 1.2:
            continue
        dist_pct = abs(ltp - sr["mid"]) / ltp * 100.0
        if dist_pct <= 1.8:
            in_zone = sr["low"] * 0.997 <= ltp <= sr["high"] * 1.003
            candidates.append(
                {
                    "type": f"SR_{sr['kind']}",
                    "low": sr["low"],
                    "high": sr["high"],
                    "status": f"{sr['touches']}T",
                    "has_fvg": False,
                    "in_zone": in_zone,
                    "dist_pct": dist_pct,
                    "priority": (12 + sr["touches"] * 3) - dist_pct * 3.0,
                }
            )

    if pdh_pdl:
        lvl = pdh_pdl["pdl"] if direction == "BULLISH" else pdh_pdl["pdh"]
        dist_pct = abs(ltp - lvl) / ltp * 100.0
        if dist_pct <= 1.2:
            band = ltp * 0.002
            candidates.append(
                {
                    "type": "PDL" if direction == "BULLISH" else "PDH",
                    "low": round(lvl - band, 2),
                    "high": round(lvl + band, 2),
                    "status": "KEY_LEVEL",
                    "has_fvg": False,
                    "in_zone": dist_pct <= 0.35,
                    "dist_pct": dist_pct,
                    "priority": 14 - dist_pct * 3.0,
                }
            )

    if not candidates:
        return None
    candidates.sort(key=lambda c: c["priority"], reverse=True)
    return candidates[0]


def _pick_tight_sl_and_entry(
    direction: str,
    ltp: float,
    poi: dict[str, Any],
    ltf_struct: dict[str, Any],
    ltf_df: pd.DataFrame,
    max_sl_pct: float,
) -> tuple[float, float, str, list[str]]:
    """
    Use LTF structure (5m for Intraday, 1h for Swing) to achieve a tight invalidation SL.
    Returns (entry_price, stop_loss, entry_mode, trigger_notes).
    """
    triggers: list[str] = []
    atr_ltf = max(float(ltf_struct.get("atr") or 0.0), ltp * 0.0015)
    buffer = max(atr_ltf * 0.25, ltp * 0.0006)

    # Check if LTF already printed a matching sweep, CHoCH, or OB right near current price
    ltf_sweeps = [s for s in ltf_struct.get("sweeps", []) if s["direction"] == direction]
    ltf_choch = [c for c in ltf_struct.get("choch", []) if c["direction"] == direction]
    ltf_obs = [o for o in ltf_struct.get("order_blocks", []) if o["direction"] == direction]

    if ltf_sweeps:
        triggers.append(f"LTF Liquidity Sweep @ ₹{ltf_sweeps[-1]['swept_level']}")
    if ltf_choch:
        triggers.append(f"LTF CHoCH Shift @ ₹{ltf_choch[-1]['level']}")

    # Entry price: market/trigger if inside or just bounced off POI, else limit at POI edge
    if poi["in_zone"] or ltf_sweeps or ltf_choch:
        entry = ltp
        entry_mode = "READY_NOW"
    else:
        entry = round(poi["high"] if direction == "BULLISH" else poi["low"], 2)
        entry_mode = "WAIT_PULLBACK"

    # Determine tightest valid structural SL
    raw_sl: float | None = None
    if direction == "BULLISH":
        if ltf_sweeps:
            raw_sl = ltf_sweeps[-1]["wick_extreme"] - buffer
        elif ltf_obs and ltf_obs[-1]["low"] < entry:
            raw_sl = ltf_obs[-1]["low"] - buffer
            triggers.append(f"Refined LTF OB SL (₹{ltf_obs[-1]['low']})")
        else:
            recent_low = float(ltf_df["low"].tail(8).min())
            raw_sl = min(recent_low, poi["low"]) - buffer

        # Enforce small-SL guardrail: if structural SL > max_sl_pct, anchor to tight recent LTF low if viable
        sl_pct = (entry - raw_sl) / entry * 100.0
        if sl_pct > max_sl_pct:
            tight_low = float(ltf_df["low"].tail(5).min()) - buffer * 0.6
            tight_pct = (entry - tight_low) / entry * 100.0
            if 0.12 <= tight_pct <= max_sl_pct:
                raw_sl = tight_low
            else:
                # Cap at max_sl_pct only if entry is inside POI (so invalidation remains meaningful)
                raw_sl = entry * (1.0 - max_sl_pct / 100.0)
        min_sl = entry * (1.0 - 0.0015)
        stop_loss = min(raw_sl, min_sl)
    else:
        if ltf_sweeps:
            raw_sl = ltf_sweeps[-1]["wick_extreme"] + buffer
        elif ltf_obs and ltf_obs[-1]["high"] > entry:
            raw_sl = ltf_obs[-1]["high"] + buffer
            triggers.append(f"Refined LTF OB SL (₹{ltf_obs[-1]['high']})")
        else:
            recent_high = float(ltf_df["high"].tail(8).max())
            raw_sl = max(recent_high, poi["high"]) + buffer

        sl_pct = (raw_sl - entry) / entry * 100.0
        if sl_pct > max_sl_pct:
            tight_high = float(ltf_df["high"].tail(5).max()) + buffer * 0.6
            tight_pct = (tight_high - entry) / entry * 100.0
            if 0.12 <= tight_pct <= max_sl_pct:
                raw_sl = tight_high
            else:
                raw_sl = entry * (1.0 + max_sl_pct / 100.0)
        min_sl = entry * (1.0 + 0.0015)
        stop_loss = max(raw_sl, min_sl)

    return round(entry, 2), round(stop_loss, 2), entry_mode, triggers


def _compute_targets(
    direction: str,
    entry: float,
    stop_loss: float,
    htf_sr: list[dict[str, Any]],
    pdh_pdl: dict[str, float] | None,
    min_rr: float,
) -> tuple[float, float, float]:
    risk = abs(entry - stop_loss)
    if risk <= 0:
        return entry, entry, 0.0

    # Collect opposing structural liquidity levels
    opposing_levels: list[float] = []
    for z in htf_sr:
        if direction == "BULLISH" and z["mid"] > entry + risk * 1.1:
            opposing_levels.append(z["low"])
        elif direction == "BEARISH" and z["mid"] < entry - risk * 1.1:
            opposing_levels.append(z["high"])

    if pdh_pdl:
        if direction == "BULLISH" and pdh_pdl["pdh"] > entry + risk * 1.1:
            opposing_levels.append(pdh_pdl["pdh"])
        elif direction == "BEARISH" and pdh_pdl["pdl"] < entry - risk * 1.1:
            opposing_levels.append(pdh_pdl["pdl"])

    if direction == "BULLISH":
        opposing_levels.sort()
        tp1_default = entry + risk * 1.5
        tp2_default = entry + risk * max(2.6, min_rr + 0.5)
        tp1 = opposing_levels[0] if opposing_levels and 1.25 <= (opposing_levels[0] - entry) / risk <= 2.2 else tp1_default
        tp2_cand = [lvl for lvl in opposing_levels if (lvl - entry) / risk >= min_rr and lvl > tp1 + risk * 0.5]
        tp2 = tp2_cand[0] if tp2_cand else max(tp2_default, tp1 + risk * 0.9)
        rr = (tp2 - entry) / risk
    else:
        opposing_levels.sort(reverse=True)
        tp1_default = entry - risk * 1.5
        tp2_default = entry - risk * max(2.6, min_rr + 0.5)
        tp1 = opposing_levels[0] if opposing_levels and 1.25 <= (entry - opposing_levels[0]) / risk <= 2.2 else tp1_default
        tp2_cand = [lvl for lvl in opposing_levels if (entry - lvl) / risk >= min_rr and lvl < tp1 - risk * 0.5]
        tp2 = tp2_cand[0] if tp2_cand else min(tp2_default, tp1 - risk * 0.9)
        rr = (entry - tp2) / risk

    return round(tp1, 2), round(tp2, 2), round(rr, 2)


def generate_symbol_ideas(
    symbol: str,
    lot_size: int,
    df_1d: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_5m: pd.DataFrame,
    data_source: str,
    account_cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Evaluate both Intraday (1h+15m -> 5m) and Swing (1D -> 1h) setups for a single F&O stock.
    Returns analysis summary + list of qualified setups.
    """
    s_1d = analyze_structure(df_1d, swing_window=4)
    s_1h = analyze_structure(df_1h, swing_window=3)
    s_15m = analyze_structure(df_15m, swing_window=3)
    s_5m = analyze_structure(df_5m, swing_window=3)

    pdh_pdl = _compute_pdh_pdl(df_1d)
    ltp = float(s_5m["ltp"] or s_15m["ltp"] or s_1h["ltp"] or s_1d["ltp"])

    risk_inr = float(account_cfg.get("risk_per_trade_inr", 1000.0))
    max_sl_intraday = float(account_cfg.get("max_sl_pct_intraday", 0.65))
    max_sl_swing = float(account_cfg.get("max_sl_pct_swing", 1.80))
    min_rr = float(account_cfg.get("min_rr_ratio", 2.0))

    ideas: list[dict[str, Any]] = []

    # 1. INTRADAY IDEA (HTF: 1h trend + 1D S/R -> POI: 15m OB/FVG/S&R -> Trigger: 5m)
    intraday_idea = _evaluate_timeframe_setup(
        symbol=symbol,
        style="INTRADAY",
        lot_size=lot_size,
        ltp=ltp,
        htf_struct=s_1h,
        poi_struct=s_15m,
        ltf_struct=s_5m,
        ltf_df=df_5m,
        daily_sr=s_1d["sr_zones"],
        pdh_pdl=pdh_pdl,
        max_sl_pct=max_sl_intraday,
        min_rr=min_rr,
        risk_inr=risk_inr,
        data_source=data_source,
    )
    if intraday_idea:
        ideas.append(intraday_idea)

    # 2. SWING IDEA (HTF: 1D trend & S/R -> POI: 1D/1h OB/FVG -> Trigger: 1h/15m)
    swing_idea = _evaluate_timeframe_setup(
        symbol=symbol,
        style="SWING",
        lot_size=lot_size,
        ltp=ltp,
        htf_struct=s_1d,
        poi_struct=s_1h if s_1h.get("order_blocks") else s_1d,
        ltf_struct=s_1h,
        ltf_df=df_1h,
        daily_sr=s_1d["sr_zones"],
        pdh_pdl=pdh_pdl,
        max_sl_pct=max_sl_swing,
        min_rr=min_rr,
        risk_inr=risk_inr,
        data_source=data_source,
    )
    if swing_idea:
        ideas.append(swing_idea)

    return {
        "symbol": symbol,
        "ltp": ltp,
        "lot_size": lot_size,
        "data_source": data_source,
        "pdh_pdl": pdh_pdl,
        "structures": {
            "1D": s_1d,
            "1h": s_1h,
            "15m": s_15m,
            "5m": s_5m,
        },
        "ideas": ideas,
    }


def _evaluate_timeframe_setup(
    *,
    symbol: str,
    style: str,
    lot_size: int,
    ltp: float,
    htf_struct: dict[str, Any],
    poi_struct: dict[str, Any],
    ltf_struct: dict[str, Any],
    ltf_df: pd.DataFrame,
    daily_sr: list[dict[str, Any]],
    pdh_pdl: dict[str, float] | None,
    max_sl_pct: float,
    min_rr: float,
    risk_inr: float,
    data_source: str,
) -> dict[str, Any] | None:
    if ltp <= 0 or ltf_df.empty:
        return None

    htf_trend = htf_struct.get("trend", "RANGING")
    poi_trend = poi_struct.get("trend", "RANGING")

    # Determine direction from HTF trend, or POI trend if HTF is ranging with strong sweep
    if htf_trend in ("BULLISH", "BEARISH"):
        direction = htf_trend
    elif poi_trend in ("BULLISH", "BEARISH"):
        direction = poi_trend
    else:
        return None

    combined_sr = poi_struct.get("sr_zones", []) + daily_sr
    poi = _find_nearest_poi(ltp, direction, poi_struct, combined_sr, pdh_pdl)
    if poi is None:
        return None

    entry, stop_loss, entry_mode, trigger_notes = _pick_tight_sl_and_entry(
        direction=direction,
        ltp=ltp,
        poi=poi,
        ltf_struct=ltf_struct,
        ltf_df=ltf_df,
        max_sl_pct=max_sl_pct,
    )
    sl_pct = round(abs(entry - stop_loss) / entry * 100.0, 2)
    if sl_pct > max_sl_pct + 0.05 or sl_pct < 0.08:
        return None

    tp1, tp2, rr_ratio = _compute_targets(
        direction=direction,
        entry=entry,
        stop_loss=stop_loss,
        htf_sr=combined_sr,
        pdh_pdl=pdh_pdl,
        min_rr=min_rr,
    )
    if rr_ratio < min_rr:
        return None

    # Score confluence transparently (0..100)
    score = 40
    reasons: list[str] = []

    if htf_trend == direction:
        score += 18
        reasons.append(f"HTF {htf_trend} ({htf_struct['structure_label']})")
    if poi_trend == direction:
        score += 10
        reasons.append(f"POI TF {poi_trend} aligned")

    if poi["type"] == "ORDER_BLOCK":
        score += 14 if poi["status"] == "FRESH" else 9
        reasons.append(f"{poi['status']} Order Block (₹{poi['low']}–₹{poi['high']})")
    elif poi["type"] == "FVG":
        score += 10
        reasons.append(f"Unfilled FVG (₹{poi['low']}–₹{poi['high']})")
    else:
        score += 8
        reasons.append(f"Key S/R {poi['type']} (₹{poi['low']}–₹{poi['high']})")

    if poi.get("has_fvg") and poi["type"] == "ORDER_BLOCK":
        score += 6
        reasons.append("OB + FVG Displacement Imbalance")

    if poi["in_zone"]:
        score += 7
        reasons.append("Price Inside POI Zone")

    if trigger_notes:
        score += 8
        reasons.extend(trigger_notes)

    # VWAP & RSI momentum confluence
    vwap = float(ltf_struct.get("vwap") or ltp)
    rsi = float(ltf_struct.get("rsi") or 50.0)
    if style == "INTRADAY":
        if (direction == "BULLISH" and ltp >= vwap * 0.998) or (
            direction == "BEARISH" and ltp <= vwap * 1.002
        ):
            score += 5
            reasons.append(f"VWAP Aligned (₹{vwap})")
    if (direction == "BULLISH" and 35 <= rsi <= 65) or (
        direction == "BEARISH" and 35 <= rsi <= 65
    ):
        score += 4
        reasons.append(f"Healthy RSI ({rsi})")

    if rr_ratio >= 2.8:
        score += 6
    if sl_pct <= max_sl_pct * 0.7:
        score += 4
        reasons.append(f"Ultra-Tight SL ({sl_pct}%)")

    score = min(99, max(45, score))
    if score >= 82 and rr_ratio >= 2.5:
        grade = "A+"
    elif score >= 70:
        grade = "A"
    else:
        grade = "B"

    risk_per_share = abs(entry - stop_loss)
    suggested_qty = max(1, int(risk_inr // risk_per_share))
    fno_lots = max(1, round(suggested_qty / max(1, lot_size)))
    one_lot_risk = round(lot_size * risk_per_share, 0)

    setup_id = hashlib.md5(f"{symbol}:{style}:{direction}".encode()).hexdigest()[:12]
    return {
        "id": setup_id,
        "symbol": symbol,
        "style": style,
        "direction": "LONG" if direction == "BULLISH" else "SHORT",
        "grade": grade,
        "score": int(score),
        "entry_mode": entry_mode,
        "entry_price": entry,
        "stop_loss": stop_loss,
        "tp1": tp1,
        "tp2": tp2,
        "sl_pct": sl_pct,
        "rr_ratio": rr_ratio,
        "ltp": ltp,
        "lot_size": lot_size,
        "suggested_qty": suggested_qty,
        "fno_lots_equiv": fno_lots,
        "one_lot_risk_inr": int(one_lot_risk),
        "poi": poi,
        "htf_trend": htf_trend,
        "htf_structure": htf_struct.get("structure_label", ""),
        "rsi": rsi,
        "vwap": vwap,
        "reasons": reasons,
        "data_source": data_source,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
