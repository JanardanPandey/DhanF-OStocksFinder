from __future__ import annotations

import hashlib
import pickle
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.database import db
from app.structure_engine import analyze_structure

IST = timezone(timedelta(hours=5, minutes=30))

DEFAULT_CACHE_PKL = Path(
    "/home/janardan/.gemini/antigravity/brain/0250c9f9-10c9-466a-950c-5bc736703515/scratch/cache_2026_11_12.pkl"
)


def is_valid_fno_symbol(symbol: str) -> bool:
    """Exclude NSE mock symbols like 011NSETEST..181NSETEST."""
    s = symbol.upper().strip()
    return bool(s) and not s.endswith("NSETEST")


def next_nse_trading_date(d: date | str) -> str:
    """Return next weekday (Mon-Fri) date string YYYY-MM-DD after `d`."""
    if isinstance(d, str):
        d = date.fromisoformat(d[:10])
    nxt = d + timedelta(days=1)
    while nxt.weekday() >= 5:  # Sat=5, Sun=6
        nxt += timedelta(days=1)
    return nxt.isoformat()


def _candles_to_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).copy()
    if "ts" not in df.columns and "timestamp" in df.columns:
        df["ts"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["ts"].astype(int), unit="s", utc=True).dt.tz_convert(IST)
    df["trade_date"] = df["dt"].dt.date
    df["hhmm"] = df["dt"].dt.strftime("%H:%M")
    return df


def _resample_5m_records(df5: pd.DataFrame, rule: str) -> list[dict[str, Any]]:
    if df5.empty:
        return []
    work = df5.set_index("dt")
    # NSE intraday bars start at 09:15 IST -> 15min offset aligns 15m & 60m cleanly
    agg = (
        work.resample(rule, origin="start_day", offset="15min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["open", "high", "low", "close"])
    )
    out: list[dict[str, Any]] = []
    for ts_idx, row in agg.iterrows():
        out.append(
            {
                "ts": int(ts_idx.timestamp()),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
        )
    return out


def seed_sqlite_from_cache(pkl_path: Path | str = DEFAULT_CACHE_PKL, force: bool = False) -> dict[str, Any]:
    """
    Populate SQLite `candles` table from cached real F&O historical pickle
    if DB doesn't already hold 5m candles for >= 100 F&O symbols.
    """
    path = Path(pkl_path)
    if not path.exists():
        return {"seeded": False, "reason": "cache_not_found", "path": str(path)}

    with db.connect() as conn:
        cnt = conn.execute(
            "SELECT COUNT(DISTINCT symbol) AS c FROM candles WHERE timeframe = '5m'"
        ).fetchone()["c"]
    if cnt >= 100 and not force:
        return {"seeded": False, "reason": "already_populated", "symbols": int(cnt)}

    with path.open("rb") as fh:
        raw = pickle.load(fh)

    intraday_map: dict[str, dict[str, Any]] = raw.get("intraday", {})
    daily_map: dict[str, dict[str, Any]] = raw.get("daily", {})

    seeded_syms = 0
    for sym, d5 in intraday_map.items():
        if not is_valid_fno_symbol(sym):
            continue
        ts_list = d5.get("timestamp") or []
        if len(ts_list) < 30:
            continue
        recs_5m = [
            {
                "ts": int(float(ts_list[i])),
                "open": float(d5["open"][i]),
                "high": float(d5["high"][i]),
                "low": float(d5["low"][i]),
                "close": float(d5["close"][i]),
                "volume": float(d5["volume"][i]),
            }
            for i in range(len(ts_list))
        ]
        # Keep up to 1000 recent 5m bars (~13 trading sessions) + resampled 15m/1h
        recs_5m = recs_5m[-1000:]
        db.save_candles(sym, "5m", recs_5m)

        df5 = _candles_to_df(recs_5m)
        recs_15m = _resample_5m_records(df5, "15min")
        recs_1h = _resample_5m_records(df5, "60min")
        if recs_15m:
            db.save_candles(sym, "15m", recs_15m)
        if recs_1h:
            db.save_candles(sym, "1h", recs_1h)

        d1 = daily_map.get(sym)
        if d1 and d1.get("timestamp"):
            ts1 = d1["timestamp"]
            recs_1d = [
                {
                    "ts": int(float(ts1[i])),
                    "open": float(d1["open"][i]),
                    "high": float(d1["high"][i]),
                    "low": float(d1["low"][i]),
                    "close": float(d1["close"][i]),
                    "volume": float(d1["volume"][i]),
                }
                for i in range(len(ts1))
            ][-260:]
            db.save_candles(sym, "1D", recs_1d)
        seeded_syms += 1

    return {"seeded": True, "symbols": seeded_syms}


def compute_market_breadth(
    quotes_map: dict[str, dict[str, Any]],
    source: str = "cached",
) -> dict[str, Any]:
    """
    Playbook Concept #14: Compute F&O % Above Session VWAP, % Above Open,
    and NSE Advances % (C > PDC) across the F&O universe.
    """
    valid_items = [
        q for sym, q in quotes_map.items() if is_valid_fno_symbol(sym) and float(q.get("last_price") or 0) > 0
    ]
    n = len(valid_items)
    if n == 0:
        snapshot = {
            "trade_date": datetime.now(IST).date().isoformat(),
            "abv_vwap_pct": 50.0,
            "green_open_pct": 50.0,
            "nse_adv_pct": 50.0,
            "regime": "BALANCED",
            "sample_size": 0,
            "source": source,
            "updated_at": datetime.now(IST).isoformat(timespec="seconds"),
        }
        return snapshot

    abv_vwap = sum(1 for q in valid_items if float(q["last_price"]) >= float(q.get("average_price") or q["last_price"]))
    green_open = sum(1 for q in valid_items if float(q["last_price"]) >= float(q.get("open") or q["last_price"]))
    nse_adv = sum(1 for q in valid_items if float(q["last_price"]) >= float(q.get("prev_close") or q["last_price"]))

    abv_vwap_pct = round(abv_vwap * 100.0 / n, 1)
    green_open_pct = round(green_open * 100.0 / n, 1)
    nse_adv_pct = round(nse_adv * 100.0 / n, 1)

    bullish_count = abv_vwap
    bearish_count = n - abv_vwap

    if abv_vwap_pct >= 60.0:
        macro_bias = "BULLISH_DOMINANT"
        regime = "BULL_FAVORABLE"
    elif abv_vwap_pct <= 40.0:
        macro_bias = "BEARISH_DOMINANT"
        regime = "BEAR_FAVORABLE"
    elif abv_vwap_pct > 55.0:
        macro_bias = "MILD_BULLISH"
        regime = "BULL_FAVORABLE"
    elif abv_vwap_pct < 45.0:
        macro_bias = "MILD_BEARISH"
        regime = "BEAR_FAVORABLE"
    else:
        macro_bias = "BALANCED"
        regime = "BALANCED"

    trade_date = next((str(q["trade_date"]) for q in valid_items if q.get("trade_date")), datetime.now(IST).date().isoformat())
    snapshot = {
        "trade_date": trade_date,
        "abv_vwap_pct": abv_vwap_pct,
        "green_open_pct": green_open_pct,
        "nse_adv_pct": nse_adv_pct,
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "macro_bias": macro_bias,
        "regime": regime,
        "sample_size": n,
        "source": source,
        "updated_at": datetime.now(IST).isoformat(timespec="seconds"),
    }
    db.save_market_breadth(snapshot)
    return snapshot


def _session_vwap(day_5m: pd.DataFrame) -> float:
    tp = (day_5m["high"] + day_5m["low"] + day_5m["close"]) / 3.0
    vsum = float(day_5m["volume"].sum())
    if vsum <= 0:
        return float(tp.mean())
    return float((tp * day_5m["volume"]).sum() / vsum)


def _compute_atr14_pct(df_1d: pd.DataFrame) -> float:
    if len(df_1d) < 15:
        return 0.0
    h = df_1d["high"]
    l = df_1d["low"]
    pc = df_1d["close"].shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr14 = float(tr.rolling(14).mean().iloc[-1])
    close = float(df_1d["close"].iloc[-1])
    return round((atr14 / close) * 100.0, 2) if close > 0 else 0.0


def _compute_rsi14(series: pd.Series) -> float:
    if len(series) < 15:
        return 50.0
    delta = series.diff()
    gain = delta.clip(lower=0.0).rolling(14).mean()
    loss = (-delta.clip(upper=0.0)).rolling(14).mean()
    g = float(gain.iloc[-1])
    l = float(loss.iloc[-1])
    if l == 0:
        return 100.0 if g > 0 else 50.0
    rs = g / l
    return round(100.0 - (100.0 / (1.0 + rs)), 1)


def _pm_rel_volume(df_15m: pd.DataFrame, trade_d: date) -> tuple[float, str | None]:
    """Peak 14:00-15:15 15m relative volume vs trailing 20-bar median on `trade_d`."""
    day_15 = df_15m[df_15m["trade_date"] == trade_d]
    if day_15.empty:
        return 0.0, None
    pm_bars = day_15[(day_15["hhmm"] >= "14:00") & (day_15["hhmm"] <= "15:15")]
    if pm_bars.empty:
        return 0.0, None
    hist = df_15m[df_15m["trade_date"] < trade_d].tail(60)
    med_vol = float(hist["volume"].median()) if not hist.empty else float(df_15m["volume"].median())
    if med_vol <= 0:
        return 0.0, None
    ratios = pm_bars["volume"] / med_vol
    idx_max = ratios.idxmax()
    peak_rv = float(ratios.loc[idx_max])
    peak_time = str(pm_bars.loc[idx_max, "hhmm"]) + " IST"
    return round(peak_rv, 2), peak_time


def _nr_squeeze(df_1d: pd.DataFrame) -> tuple[bool, bool]:
    if len(df_1d) < 8:
        return False, False
    ranges = df_1d["high"] - df_1d["low"]
    last_r = float(ranges.iloc[-1])
    nr4 = last_r <= float(ranges.iloc[-4:].min()) * 1.01
    nr7 = last_r <= float(ranges.iloc[-7:].min()) * 1.01
    return bool(nr4), bool(nr7)


def _dry_shelf(day_5m: pd.DataFrame) -> bool:
    """Afternoon quiet coil: 13:30-15:00 tight range (<= 0.55%)."""
    aft = day_5m[(day_5m["hhmm"] >= "13:30") & (day_5m["hhmm"] <= "15:00")]
    if len(aft) < 6:
        return False
    hi = float(aft["high"].max())
    lo = float(aft["low"].min())
    mid = (hi + lo) / 2.0
    return mid > 0 and ((hi - lo) / mid * 100.0) <= 0.55


def _compute_volume_footprint(day_5m: pd.DataFrame, direction: str) -> dict[str, Any]:
    """
    Computes institutional Order-Flow Volume Footprint metrics:
    - Intra-candle Volume Delta & Cumulative Volume Delta (CVD)
    - First 1-hour CVD (H1 CVD %)
    - Volume Profile Point of Control (POC)
    - Total Session Traded Volume (Liquidity Check)
    - Delta Divergence Warning (Price moving in direction, but CVD opposing)
    """
    if day_5m.empty:
        return {
            "total_volume": 0.0,
            "cvd_pct": 0.0,
            "h1_cvd_pct": 0.0,
            "net_cvd_shares": 0.0,
            "poc_price": 0.0,
            "ltp_vs_poc_pct": 0.0,
            "delta_divergence": False,
            "is_liquid": False,
        }

    c_b = day_5m["close"].to_numpy(dtype=float)
    h_b = day_5m["high"].to_numpy(dtype=float)
    l_b = day_5m["low"].to_numpy(dtype=float)
    v_b = day_5m["volume"].to_numpy(dtype=float)

    rng = np.maximum(0.01, h_b - l_b)
    clv = (2.0 * c_b - h_b - l_b) / rng
    delta_vol = v_b * clv
    tot_vol = float(np.sum(v_b))
    cvd = np.cumsum(delta_vol)
    net_cvd = float(cvd[-1])
    cvd_pct = round(net_cvd / max(1.0, tot_vol) * 100.0, 1)

    n_h1 = min(len(day_5m), 12)
    h1_vol = float(np.sum(v_b[:n_h1]))
    h1_delta = float(np.sum(delta_vol[:n_h1]))
    h1_cvd_pct = round(h1_delta / max(1.0, h1_vol) * 100.0, 1)

    p_min, p_max = float(np.min(l_b)), float(np.max(h_b))
    if p_max > p_min:
        bins = np.linspace(p_min, p_max, 25)
        tp = (h_b + l_b + c_b) / 3.0
        vol_hist, bin_edges = np.histogram(tp, bins=bins, weights=v_b)
        poc_idx = int(np.argmax(vol_hist))
        poc_price = round(float((bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2.0), 2)
    else:
        poc_price = float(c_b[-1])

    ltp = float(c_b[-1])
    ltp_vs_poc = round((ltp - poc_price) / max(0.01, poc_price) * 100.0, 2)

    dir_sign = 1.0 if direction == "LONG" else -1.0
    dir_cvd = cvd_pct * dir_sign
    delta_divergence = dir_cvd < -8.0

    is_liquid = tot_vol >= 150_000

    return {
        "total_volume": tot_vol,
        "cvd_pct": cvd_pct,
        "h1_cvd_pct": h1_cvd_pct,
        "net_cvd_shares": net_cvd,
        "poc_price": poc_price,
        "ltp_vs_poc_pct": ltp_vs_poc,
        "delta_divergence": delta_divergence,
        "is_liquid": is_liquid,
    }


def get_strike_step(ltp: float) -> float:
    """Standard NSE F&O strike price step sizes."""
    if ltp < 50:
        return 1.0
    if ltp < 250:
        return 2.5
    if ltp < 500:
        return 5.0
    if ltp < 1000:
        return 10.0
    if ltp < 2500:
        return 20.0
    if ltp < 5000:
        return 50.0
    if ltp < 10000:
        return 100.0
    return 250.0


def recommend_option_strike(
    symbol: str,
    ltp: float,
    direction: str,
    lot_size: int = 1,
) -> dict[str, Any]:
    """
    Playbook v2 Smart Money Option Strike Selector:
    Selects near-OTM/ATM strike with optimal Gamma sensitivity (~0.35 - 0.45 Delta)
    and estimates realistic premium, capital requirement (~12k-15k budget), SL, and TP.
    """
    if ltp <= 0:
        return {
            "strike": 0,
            "option_name": f"{symbol} OPT",
            "option_type": "CE" if direction == "LONG" else "PE",
            "lot_size": lot_size,
            "est_premium": 0.0,
            "est_capital_inr": 0,
            "rec_lots": 1,
            "sl_premium": 0.0,
            "tp1_premium": 0.0,
            "tp2_premium": 0.0,
            "expected_roi": "+45% in 30-60m",
        }

    step = get_strike_step(ltp)
    if direction == "LONG":
        target_strike = ltp * 1.020  # ~2.0% OTM call option
        strike = round(target_strike / step) * step
        opt_type = "CE"
    else:
        target_strike = ltp * 0.980  # ~2.0% OTM put option
        strike = round(target_strike / step) * step
        opt_type = "PE"

    strike_disp = int(strike) if step >= 1.0 else strike
    option_name = f"{symbol} {strike_disp} {opt_type}"

    # Estimated near-month option premium: ~1.85% of cash LTP
    est_prem = round(ltp * 0.0185, 1)
    one_lot_cap = round(est_prem * max(1, lot_size))
    # Capital allocation target ~15,000 INR
    rec_lots = max(1, int(round(15000.0 / max(1.0, one_lot_cap))))
    total_cap = one_lot_cap * rec_lots

    sl_prem = round(est_prem * 0.78, 1)   # -22% risk
    tp1_prem = round(est_prem * 1.45, 1)  # +45% target 1 (matches video proof +48%)
    tp2_prem = round(est_prem * 1.70, 1)  # +70% target 2

    return {
        "strike": strike_disp,
        "option_name": option_name,
        "option_type": opt_type,
        "lot_size": lot_size,
        "est_premium": est_prem,
        "est_capital_inr": total_cap,
        "rec_lots": rec_lots,
        "sl_premium": sl_prem,
        "tp1_premium": tp1_prem,
        "tp2_premium": tp2_prem,
        "expected_roi": "+45% in 30-60m",
    }


def compute_order_block(
    day5: pd.DataFrame,
    direction: str,
    ltp: float,
    vwap: float,
    c1d: list[dict[str, Any]] | None = None,
    c1h: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Top-Down Multi-Timeframe Smart Money Order Block (OB) Hierarchy:
    1. WEEKLY OB (1W): Macro Supply & Demand Zones (Sets Macro Bias & Floor/Ceiling)
    2. DAILY OB (1D): Key Swing Structure & Invalidation S1 / Targets R1 & R2
    3. HOURLY OB (1H): Intermediate Intraday Base / Institutional Distribution Shelf
    4. 5m / 15m INTRADAY: Decisive Breakout / Retest Bounce Confirmation
    """
    n = len(day5)
    if n < 3 or ltp <= 0:
        return {
            "status": "EARLY_OPEN",
            "ob_low": round(ltp, 2),
            "ob_high": round(ltp, 2),
            "ob_zone": f"₹{ltp:,.2f}",
            "s1": round(ltp, 2),
            "r1": round(ltp, 2),
            "r2": round(ltp, 2),
            "rr_ratio": "1:2.0",
            "weekly_ob": {"zone": "None", "type": "NEUTRAL", "dist_pct": 0.0},
            "daily_ob": {"zone": "None", "type": "NEUTRAL", "dist_pct": 0.0},
            "hourly_ob": {"zone": "None", "type": "NEUTRAL"},
            "htf_warning": None,
            "trigger_hhmm": None,
        }

    h_b = day5["high"].to_numpy(dtype=float)
    l_b = day5["low"].to_numpy(dtype=float)
    c_b = day5["close"].to_numpy(dtype=float)
    hhmm_list = day5["hhmm"].tolist()

    weekly_ob: dict[str, Any] = {"zone": "None", "type": "NEUTRAL", "dist_pct": 0.0}
    daily_ob: dict[str, Any] = {"zone": "None", "type": "NEUTRAL", "dist_pct": 0.0}
    hourly_ob: dict[str, Any] = {"zone": "None", "type": "NEUTRAL"}
    htf_warning: str | None = None

    # 1. Weekly & 2. Daily Multi-Timeframe Order Block Detection
    if c1d:
        df1 = _candles_to_df(c1d)
        highs = df1["high"].to_numpy(dtype=float)
        lows = df1["low"].to_numpy(dtype=float)

        # Level 1: Weekly Order Block (aggregated from Daily)
        if "dt" in df1.columns:
            df1["datetime"] = pd.to_datetime(df1["dt"])
            df_w = (
                df1.set_index("datetime")
                .resample("W-FRI")
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                .dropna()
                .reset_index()
            )
            if len(df_w) >= 6:
                w_h = df_w["high"].to_numpy(dtype=float)
                w_l = df_w["low"].to_numpy(dtype=float)
                if direction == "SHORT":
                    demands = [(w_l[i], w_h[i]) for i in range(len(df_w) - 1) if w_l[i] <= ltp * 1.02]
                    if demands:
                        near_w = max(demands, key=lambda x: x[0])
                        d_pct = round((ltp - near_w[0]) / max(0.01, near_w[0]) * 100.0, 2)
                        weekly_ob = {
                            "zone": f"₹{near_w[0]:.2f} - ₹{near_w[1]:.2f}",
                            "type": "WEEKLY DEMAND",
                            "dist_pct": d_pct,
                            "low": round(near_w[0], 2),
                            "high": round(near_w[1], 2),
                        }
                        if abs(d_pct) <= 1.2:
                            htf_warning = f"At Weekly Demand Floor (₹{near_w[0]:.2f}) — Exhaustion Risk"
                else:
                    supplies = [(w_l[i], w_h[i]) for i in range(len(df_w) - 1) if w_h[i] >= ltp * 0.98]
                    if supplies:
                        near_w = min(supplies, key=lambda x: x[1])
                        d_pct = round((near_w[1] - ltp) / max(0.01, ltp) * 100.0, 2)
                        weekly_ob = {
                            "zone": f"₹{near_w[0]:.2f} - ₹{near_w[1]:.2f}",
                            "type": "WEEKLY SUPPLY",
                            "dist_pct": d_pct,
                            "low": round(near_w[0], 2),
                            "high": round(near_w[1], 2),
                        }
                        if abs(d_pct) <= 1.2:
                            htf_warning = f"At Weekly Supply Ceiling (₹{near_w[1]:.2f}) — Resistance Risk"

        # Level 2: Daily Order Block (Lookback 40 sessions)
        lookback_d = min(40, len(df1) - 1)
        if direction == "SHORT":
            d_zones = [(lows[i], highs[i]) for i in range(len(df1) - lookback_d, len(df1) - 1) if lows[i] <= ltp * 1.02]
            if d_zones:
                near_d = max(d_zones, key=lambda x: x[0])
                daily_ob = {
                    "zone": f"₹{near_d[0]:.2f} - ₹{near_d[1]:.2f}",
                    "type": "DAILY DEMAND",
                    "dist_pct": round((ltp - near_d[0]) / max(0.01, near_d[0]) * 100.0, 2),
                    "low": round(near_d[0], 2),
                    "high": round(near_d[1], 2),
                }
        else:
            d_zones = [(lows[i], highs[i]) for i in range(len(df1) - lookback_d, len(df1) - 1) if highs[i] >= ltp * 0.98]
            if d_zones:
                near_d = min(d_zones, key=lambda x: x[1])
                daily_ob = {
                    "zone": f"₹{near_d[0]:.2f} - ₹{near_d[1]:.2f}",
                    "type": "DAILY SUPPLY",
                    "dist_pct": round((near_d[1] - ltp) / max(0.01, ltp) * 100.0, 2),
                    "low": round(near_d[0], 2),
                    "high": round(near_d[1], 2),
                }

    # Level 3: Hourly Order Block (Lookback 6-12 bars = 6-12 hours)
    if c1h:
        df_h = _candles_to_df(c1h)
        h_h = df_h["high"].to_numpy(dtype=float)
        h_l = df_h["low"].to_numpy(dtype=float)
        h_shelf = min(12, max(6, len(df_h) - 1))
        h_l_shelf = float(np.min(h_l[-h_shelf - 1 : -1]))
        h_h_shelf = float(np.max(h_h[-h_shelf - 1 : -1]))
        hourly_ob = {
            "zone": f"₹{h_l_shelf:.2f} - ₹{h_h_shelf:.2f}",
            "type": "HOURLY OB",
            "low": round(h_l_shelf, 2),
            "high": round(h_h_shelf, 2),
        }

    # Level 4: Intraday Base Structure (use Hourly base limits if present, else 12-18 5m bars)
    if hourly_ob.get("low"):
        base_low = float(hourly_ob["low"])
        base_high = float(hourly_ob["high"])
    else:
        shelf_len = min(18, max(9, n - 1)) if n >= 12 else min(6, max(3, n - 1))
        shelf_h = h_b[-shelf_len - 1 : -1] if n > shelf_len else h_b[:n]
        shelf_l = l_b[-shelf_len - 1 : -1] if n > shelf_len else l_b[:n]
        base_low = float(np.min(shelf_l))
        base_high = float(np.max(shelf_h))

    trigger_hhmm = hhmm_list[-1] if hhmm_list else None
    min_break_dist = max(0.02, ltp * 0.0025)

    if direction == "LONG":
        s1 = round(base_low * 0.9985, 2)
        risk = max(ltp * 0.004, ltp - s1)
        r1 = round(ltp + risk * 2.0, 2)
        r2 = round(ltp + risk * 3.5, 2)

        is_breakout = (c_b[-1] >= base_high + min_break_dist) and (c_b[-1] > vwap)
        is_forming = base_low <= ltp <= base_high
        is_retest = (l_b[-1] <= base_low * 1.002) and (c_b[-1] > base_low)
        is_extended = ltp > base_high * 1.035
    else:
        s1 = round(base_high * 1.0015, 2)
        risk = max(ltp * 0.004, s1 - ltp)
        r1 = round(ltp - risk * 2.0, 2)
        r2 = round(ltp - risk * 3.5, 2)

        is_breakout = (c_b[-1] <= base_low - min_break_dist) and (c_b[-1] < vwap)
        is_forming = base_low <= ltp <= base_high
        is_retest = (h_b[-1] >= base_high * 0.998) and (c_b[-1] < base_high)
        is_extended = ltp < base_low * 0.965

    if is_extended:
        status = "OB_EXTENDED"
    elif is_breakout:
        status = "OB_BREAKOUT_CONFIRMED"
    elif is_retest:
        status = "OB_RETEST_BOUNCE"
    elif is_forming:
        status = "OB_BASE_FORMING"
    else:
        status = "OB_ACTIVE"

    return {
        "status": status,
        "ob_low": round(base_low, 2),
        "ob_high": round(base_high, 2),
        "ob_zone": f"₹{base_low:,.2f} - ₹{base_high:,.2f}",
        "s1": s1,
        "r1": r1,
        "r2": r2,
        "rr_ratio": "1:2.0",
        "weekly_ob": weekly_ob,
        "daily_ob": daily_ob,
        "hourly_ob": hourly_ob,
        "htf_warning": htf_warning,
        "trigger_hhmm": trigger_hhmm,
    }


def build_daily_watchlist(symbols: list[str], top_n: int = 40) -> dict[str, Any]:
    """
    Build Date-Wise T-1 -> Next-Day Watchlist from latest completed/session data.
    Ranks best picks #1..N with explicit `trigger_time` and `reasons`.
    """
    candidates: list[dict[str, Any]] = []
    latest_dates: list[date] = []

    for sym in symbols:
        sym_u = sym.upper()
        if not is_valid_fno_symbol(sym_u):
            continue
        c5 = db.load_candles(sym_u, "5m", 600)
        c15 = db.load_candles(sym_u, "15m", 240)
        c1d = db.load_candles(sym_u, "1D", 60)
        if len(c5) < 40 or len(c1d) < 15:
            continue

        df5 = _candles_to_df(c5)
        df15 = _candles_to_df(c15) if c15 else pd.DataFrame()
        if df15.empty:
            df15_recs = _resample_5m_records(df5, "15min")
            df15 = _candles_to_df(df15_recs)
        df1d = _candles_to_df(c1d)

        today_ist = datetime.now(IST).date()
        t1_d = None
        day5 = pd.DataFrame()
        for cand_d in sorted(df5["trade_date"].unique(), reverse=True):
            cand_slice = df5[df5["trade_date"] == cand_d]
            if len(cand_slice) < 10:
                continue
            # During live market before 14:45 IST, keep previous completed session as T-1 source
            if cand_d == today_ist and str(cand_slice["hhmm"].max()) < "14:45":
                continue
            t1_d = cand_d
            day5 = cand_slice
            break
        if t1_d is None or day5.empty:
            continue
        latest_dates.append(t1_d)

        d_open = float(day5["open"].iloc[0])
        d_high = float(day5["high"].max())
        d_low = float(day5["low"].min())
        d_close = float(day5["close"].iloc[-1])
        d_rng = max(0.01, d_high - d_low)
        t1_clv = round((d_close - d_low) / d_rng, 2)
        t1_vwap = round(_session_vwap(day5), 2)
        abv_vwap = d_close >= t1_vwap

        prev_1d = df1d[df1d["trade_date"] < t1_d]
        prev_close = float(prev_1d["close"].iloc[-1]) if not prev_1d.empty else d_open
        t1_ret_pct = round((d_close - prev_close) / prev_close * 100.0, 2) if prev_close > 0 else 0.0

        pm_rv, pm_time = _pm_rel_volume(df15, t1_d)
        atr_pct = _compute_atr14_pct(df1d)
        rsi14 = _compute_rsi14(df1d["close"])
        nr4, nr7 = _nr_squeeze(df1d)
        dry_coil = _dry_shelf(day5)

        df1_t1 = df1d[df1d["trade_date"] <= t1_d]
        ema20_1d = float(df1_t1["close"].ewm(span=20, adjust=False).mean().iloc[-1]) if not df1_t1.empty else d_close
        ema20_dist_pct = round((d_close - ema20_1d) / max(0.01, ema20_1d) * 100.0, 2)
        c5_slice = df1_t1.tail(5)
        c5_rng_pct = (
            round((float(c5_slice["high"].max()) - float(c5_slice["low"].min())) / max(0.01, d_close) * 100.0, 2)
            if len(c5_slice) >= 5
            else 99.0
        )

        reasons: list[str] = []
        tags: list[str] = []
        bull_pts = 0.0
        bear_pts = 0.0

        if pm_rv >= 2.0:
            reasons.append(f"T-1 Institutional PM Accum {pm_rv:.2f}x ({pm_time})")
            if abv_vwap:
                bull_pts += 26 + min(12.0, (pm_rv - 2.0) * 4.0)
            else:
                bear_pts += 22 + min(10.0, (pm_rv - 2.0) * 4.0)
        elif pm_rv >= 1.5:
            reasons.append(f"T-1 Afternoon Vol {pm_rv:.2f}x ({pm_time})")
            if abv_vwap:
                bull_pts += 14
            else:
                bear_pts += 12

        if abv_vwap and t1_clv >= 0.70:
            bull_pts += 24
            reasons.append(f"Strong Close > T-1 VWAP (₹{t1_vwap}, CLV {t1_clv:.2f})")
        elif abv_vwap and t1_clv >= 0.55:
            bull_pts += 14
            reasons.append(f"Closed > T-1 VWAP (CLV {t1_clv:.2f})")
        elif (not abv_vwap) and t1_clv <= 0.30:
            bear_pts += 26
            reasons.append(f"Weak Close < T-1 VWAP (₹{t1_vwap}, CLV {t1_clv:.2f})")

        if nr7:
            bull_pts += 14
            bear_pts += 10
            tags.append("NR7")
            reasons.append("NR7 Daily Volatility Squeeze")
        elif nr4:
            bull_pts += 9
            bear_pts += 6
            tags.append("NR4")
            reasons.append("NR4 Tight Range Coil")

        if dry_coil:
            bull_pts += 10
            tags.append("DRY SHELF")
            reasons.append("Quiet 13:30-15:00 Low-Vol Coil")

        if c5_rng_pct <= 5.5 and -2.5 <= ema20_dist_pct <= 2.5:
            bull_pts += 8
            bear_pts += 8
            tags.append("5D COIL")
            reasons.append(f"Tight 5D Coil ({c5_rng_pct:.1f}%) near 1D 20EMA")

        if ema20_dist_pct <= -3.0 and (t1_clv >= 0.70 or pm_rv >= 1.8):
            bull_pts += 14
            tags.append("20EMA RECLAIM")
            reasons.append(f"Oversold 20EMA Reclaim ({ema20_dist_pct:+.1f}% vs 20EMA, CLV {t1_clv:.2f})")

        if atr_pct >= 2.2 and (pm_rv >= 1.8 or nr4 or dry_coil or ema20_dist_pct <= -3.0):
            bull_pts += 10
            bear_pts += 8
            tags.append("MONSTER DNA")
            reasons.append(f"High-Beta ATR {atr_pct:.2f}% (One-Way DNA)")

        if t1_ret_pct <= -2.2 and t1_clv >= 0.45:
            bull_pts += 10
            tags.append("OVERSOLD BOUNCE")
            reasons.append(f"Deep T-1 Oversold Reclaim ({t1_ret_pct:+.2f}%, CLV {t1_clv:.2f})")
        elif rsi14 >= 72.0:
            bear_pts += 14
            tags.append("OVERBOUGHT")
            reasons.append(f"1D Overbought RSI {rsi14:.1f} (Gap-Up Fade risk)")

        anti_chasing = t1_ret_pct >= 2.0
        if t1_ret_pct >= 4.0:
            bull_pts -= 26
            tags.append("T-1 EXHAUSTION")
            reasons.append(f"Exhausted T-1 rally {t1_ret_pct:+.2f}% — High risk of next-day consolidation")
        elif anti_chasing:
            bull_pts -= 12
            tags.append("ANTI-CHASING WATCH")
            reasons.append(f"Extended T-1 {t1_ret_pct:+.2f}% -> Skip if next open Gap >= +0.40%")

        # Require meaningful Playbook qualification
        best_pts = max(bull_pts, bear_pts)
        if best_pts < 32.0 or (pm_rv < 1.5 and not nr7 and atr_pct < 2.5):
            continue

        bias_hint = "LONG" if bull_pts >= bear_pts else "SHORT"
        score = round(min(98.5, max(45.0, 38.0 + best_pts * 0.55)), 1)
        trigger_time = (
            "09:15-09:30 IST (Wait 15m0 >=2x & Gap>=0 or Open=Low)"
            if bias_hint == "LONG"
            else "09:15-09:30 IST (Wait 15m0 >=2.5x CLV<=0.30 & Breadth<=55%)"
        )

        candidates.append(
            {
                "symbol": sym_u,
                "bias_hint": bias_hint,
                "score": score,
                "trigger_time": trigger_time,
                "pm_rv": pm_rv,
                "pm_time": pm_time or "15:00 IST",
                "t1_clv": t1_clv,
                "t1_close": round(d_close, 2),
                "t1_vwap": t1_vwap,
                "t1_ret_pct": t1_ret_pct,
                "atr_pct": atr_pct,
                "rsi14": rsi14,
                "tags": ", ".join(tags) if tags else ("CREAM WATCH" if bias_hint == "LONG" else "BEAR WATCH"),
                "anti_chasing": anti_chasing,
                "reasons": reasons,
            }
        )

    if not latest_dates:
        return {"target_date": None, "t1_date": None, "count": 0, "items": []}

    t1_ref = max(latest_dates)
    target_date = next_nse_trading_date(t1_ref)
    candidates.sort(key=lambda x: (x["score"], x["pm_rv"], x["atr_pct"]), reverse=True)
    selected = candidates[:top_n]
    for idx, item in enumerate(selected, start=1):
        item["rank"] = idx

    db.save_daily_watchlist(target_date=target_date, t1_date=t1_ref.isoformat(), items=selected) if selected else None
    return {
        "target_date": target_date,
        "t1_date": t1_ref.isoformat(),
        "count": len(selected),
        "items": selected,
    }


def evaluate_playbook_setups(
    symbol: str,
    lot_size: int,
    df_1d: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_5m: pd.DataFrame,
    breadth: dict[str, Any],
    data_source: str,
    account_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Evaluate Playbook v2 intraday setups (Cream Bull v2, Strict Bear v2, P14B Reclaim)
    with explicit IST `trigger_time`, `MONSTER RUNNER` badge, Anti-Chasing guard, and reasons.
    """
    if df_5m.empty or len(df_5m) < 25 or df_15m.empty or len(df_15m) < 20 or df_1d.empty or len(df_1d) < 15:
        return []

    w5 = _candles_to_df(df_5m.to_dict("records"))
    w15 = _candles_to_df(df_15m.to_dict("records"))
    w1d = _candles_to_df(df_1d.to_dict("records"))

    cur_d = w5["trade_date"].max()
    prev_dates = sorted({d for d in w5["trade_date"].unique() if d < cur_d})
    if not prev_dates:
        return []
    t1_d = prev_dates[-1]

    day5 = w5[w5["trade_date"] == cur_d].reset_index(drop=True)
    t1_5 = w5[w5["trade_date"] == t1_d].reset_index(drop=True)
    day15 = w15[w15["trade_date"] == cur_d].reset_index(drop=True)
    if len(day5) < 4 or day15.empty or len(t1_5) < 5:
        return []

    # Previous close (prefer daily candle prior to cur_d, else T-1 5m close)
    d_prior = w1d[w1d["trade_date"] < cur_d]
    pdc = float(d_prior["close"].iloc[-1]) if not d_prior.empty else float(t1_5["close"].iloc[-1])
    t1_prior = w1d[w1d["trade_date"] < t1_d]
    t2c = float(t1_prior["close"].iloc[-1]) if not t1_prior.empty else float(t1_5["open"].iloc[0])
    t1_close = float(t1_5["close"].iloc[-1])
    t1_ret_pct = round((t1_close - t2c) / t2c * 100.0, 2) if t2c > 0 else 0.0

    # Today open, gap, 09:15 15m0 metrics
    open_today = float(day5["open"].iloc[0])
    gap_pct = round((open_today - pdc) / pdc * 100.0, 2) if pdc > 0 else 0.0

    hist_15 = w15[w15["trade_date"] < cur_d].tail(60)
    med_15_vol = float(hist_15["volume"].median()) if not hist_15.empty else float(w15["volume"].median())
    bar15_0 = day15.iloc[0]
    r15_0_rv = round(float(bar15_0["volume"]) / med_15_vol, 2) if med_15_vol > 0 else 1.0
    b15_rng = max(0.01, float(bar15_0["high"]) - float(bar15_0["low"]))
    b15_clv = round((float(bar15_0["close"]) - float(bar15_0["low"])) / b15_rng, 2)
    b15_open_low = (float(bar15_0["open"]) - float(bar15_0["low"])) / float(bar15_0["open"]) <= 0.0020
    b15_open_high = (float(bar15_0["high"]) - float(bar15_0["open"])) / float(bar15_0["open"]) <= 0.0015

    # Session VWAP series & 09:30 (or bar idx 3) state
    tp_all = (day5["high"] + day5["low"] + day5["close"]) / 3.0
    cum_v = day5["volume"].cumsum().replace(0, np.nan)
    vwap_series = (tp_all * day5["volume"]).cumsum() / cum_v
    vwap_series = vwap_series.fillna(tp_all)

    coil_idx = min(3, len(day5) - 1)  # 09:30 bar (4th 5m bar: 09:15, 09:20, 09:25, 09:30)
    px_0930 = float(day5["close"].iloc[coil_idx])
    vwap_0930 = float(vwap_series.iloc[coil_idx])
    dist_vwap_0930 = round((px_0930 - vwap_0930) / vwap_0930 * 100.0, 2) if vwap_0930 > 0 else 0.0

    ltp = float(day5["close"].iloc[-1])
    vwap_now = float(vwap_series.iloc[-1])
    atr_pct = _compute_atr14_pct(w1d)
    rsi14 = _compute_rsi14(w1d["close"])
    nr4, nr7 = _nr_squeeze(w1d)
    pm_rv, pm_time = _pm_rel_volume(w15, t1_d)
    dry_coil = _dry_shelf(t1_5)

    struct_15 = analyze_structure(df_15m, swing_window=3)
    struct_1d = analyze_structure(df_1d, swing_window=4)
    trend_15 = struct_15.get("trend", "RANGING")
    trend_1d = struct_1d.get("trend", "RANGING")

    abv_vwap_breadth = float(breadth.get("abv_vwap_pct", 50.0))
    anti_chasing = (t1_ret_pct >= 2.0) and (gap_pct >= 0.40)
    monster_dna = atr_pct >= 2.5 and r15_0_rv >= 3.0 and (nr4 or nr7 or dry_coil or pm_rv >= 2.0)

    risk_inr = float(account_cfg.get("risk_per_trade_inr", 1000.0))
    max_sl_pct = float(account_cfg.get("max_sl_pct_intraday", 0.65))
    min_rr = float(account_cfg.get("min_rr_ratio", 2.0))

    setups: list[dict[str, Any]] = []

    def _build_setup(
        direction: str,
        playbook_tag: str,
        base_score: float,
        trigger_time: str,
        reasons: list[str],
        sl_anchor: float,
    ) -> dict[str, Any] | None:
        is_long = direction == "LONG"
        entry = round(ltp, 2)
        if is_long:
            raw_sl = min(sl_anchor * 0.999, entry * (1.0 - max_sl_pct / 100.0))
            sl = round(max(raw_sl, entry * (1.0 - max_sl_pct / 100.0)), 2)
            # Keep tight Playbook SL <= max_sl_pct
            if (entry - sl) / entry * 100.0 < 0.25:
                sl = round(entry * (1.0 - 0.0035), 2)
            risk_ps = entry - sl
            if risk_ps <= 0:
                return None
            tp1 = round(entry + risk_ps * max(2.2, min_rr), 2)
            tp2 = round(entry + risk_ps * (3.5 if monster_dna else 3.0), 2)
        else:
            raw_sl = max(sl_anchor * 1.001, entry * (1.0 + max_sl_pct / 100.0))
            sl = round(min(raw_sl, entry * (1.0 + max_sl_pct / 100.0)), 2)
            if (sl - entry) / entry * 100.0 < 0.25:
                sl = round(entry * (1.0 + 0.0035), 2)
            risk_ps = sl - entry
            if risk_ps <= 0:
                return None
            tp1 = round(entry - risk_ps * max(2.2, min_rr), 2)
            tp2 = round(entry - risk_ps * (3.5 if monster_dna else 3.0), 2)

        sl_pct = round(risk_ps / entry * 100.0, 2)
        rr = round(abs(tp2 - entry) / risk_ps, 2)
        qty = max(1, int(risk_inr // risk_ps))
        score_val = int(round(min(99.0, max(45.0, base_score))))
        grade = "A+" if (score_val >= 82 and rr >= 2.5) else ("A" if score_val >= 70 else "B")
        sid = hashlib.md5(f"{symbol}:PLAYBOOK:{playbook_tag}:{direction}".encode()).hexdigest()[:12]
        return {
            "id": sid,
            "symbol": symbol,
            "style": "INTRADAY",
            "direction": direction,
            "grade": grade,
            "score": score_val,
            "conviction": round(base_score, 2),
            "entry_mode": "LIMIT_OB" if playbook_tag != "P14B" else "BREAKOUT_CONFIRM",
            "entry_price": entry,
            "stop_loss": sl,
            "tp1": tp1,
            "tp2": tp2,
            "sl_pct": sl_pct,
            "rr_ratio": rr,
            "ltp": entry,
            "lot_size": lot_size,
            "suggested_qty": qty,
            "fno_lots_equiv": max(1, round(qty / max(1, lot_size))),
            "one_lot_risk_inr": int(round(lot_size * risk_ps)),
            "poi": {
                "type": playbook_tag,
                "low": round(min(vwap_now, entry), 2),
                "high": round(max(vwap_now, entry), 2),
                "mid": round(vwap_now, 2),
                "timeframe": "15m/5m",
                "status": "ACTIVE",
                "in_zone": True,
                "has_fvg": monster_dna,
            },
            "htf_trend": trend_1d,
            "htf_structure": struct_1d.get("structure_label", ""),
            "rsi": rsi14,
            "vwap": round(vwap_now, 2),
            "trigger_time": trigger_time,
            "playbook_tag": playbook_tag,
            "monster_runner": monster_dna,
            "anti_chasing": anti_chasing,
            "reasons": reasons,
            "reason_summary": " | ".join(reasons[:4]),
            "data_source": data_source,
            "created_at": datetime.now(IST).isoformat(timespec="seconds"),
        }

    # 1) CREAM BULL v2 (Concepts #15 + #18)
    bull_open_guard = b15_open_low or gap_pct >= 0.0
    if pm_rv >= 1.8 and r15_0_rv >= 2.0 and b15_clv >= 0.55 and bull_open_guard and -0.40 <= dist_vwap_0930 <= 0.65 and trend_15 != "BEARISH":
        r_list = [
            f"CREAM BULL v2 (66.2% historical win)",
            f"T-1 PM Accum {pm_rv:.2f}x ({pm_time})",
            f"09:15 15m0 Vol {r15_0_rv:.2f}x, CLV {b15_clv:.2f}",
            f"09:30 VWAP Coil {dist_vwap_0930:+.2f}% (15m {trend_15})",
        ]
        if b15_open_low:
            r_list.append("Open = Low Bullish Control")
        else:
            r_list.append(f"Gap >= 0 Guard ({gap_pct:+.2f}%)")
        b_score = 73.0 + min(10.0, (r15_0_rv - 2.0) * 1.6) + min(5.0, max(0.0, pm_rv - 1.8) * 1.2)
        if ltp >= vwap_now * 0.998:
            b_score += 3.0
            r_list.append(f"Holding > Live VWAP (₹{vwap_now:.2f})")
        else:
            b_score -= 12.0
            r_list.append(f"Lost VWAP Guard (LTP ₹{ltp:.2f} < VWAP ₹{vwap_now:.2f})")
        if monster_dna:
            b_score += 4.0
            r_list.insert(0, f"MONSTER RUNNER DNA (ATR {atr_pct:.2f}% + Squeeze/Coil)")
        if anti_chasing:
            b_score -= 18.0
            r_list.append(f"ANTI-CHASING WARNING: T-1 {t1_ret_pct:+.2f}% + Gap {gap_pct:+.2f}%")

        trig = f"{cur_d.isoformat()} {day5['hhmm'].iloc[coil_idx]} IST (15m0 + 09:30 VWAP)"
        orb_low = float(bar15_0["low"])
        s = _build_setup("LONG", "CREAM_BULL_V2", b_score, trig, r_list, min(orb_low, vwap_0930))
        if s:
            setups.append(s)

    # 2) STRICT BEAR v2 (Concepts #14 + #16)
    bear_breadth_ok = abv_vwap_breadth <= 55.0
    if r15_0_rv >= 2.5 and b15_clv <= 0.30 and float(bar15_0["close"]) < float(vwap_0930) and (bear_breadth_ok or b15_open_high):
        r_list = [
            f"STRICT BEAR v2 (Breadth {abv_vwap_breadth:.1f}% >VWAP)",
            f"09:15 15m0 Bear Thrust {r15_0_rv:.2f}x, CLV {b15_clv:.2f}",
            f"Closed < VWAP (₹{vwap_0930:.2f})",
        ]
        if b15_open_high:
            r_list.append("Open = High Supply Exhaustion")
        if rsi14 >= 68.0:
            r_list.append(f"Overbought Fade RSI {rsi14:.1f}")
        b_score = 75.0 + min(8.0, (r15_0_rv - 2.5) * 2.5) + (5.0 if bear_breadth_ok else -12.0)
        if not bear_breadth_ok:
            r_list.append(f"Counter-Breadth Caution: F&O >VWAP {abv_vwap_breadth:.1f}% (>55%)")
        if monster_dna:
            b_score += 5.0
            r_list.insert(0, f"MONSTER BEAR DNA (ATR {atr_pct:.2f}%)")
        trig = f"{cur_d.isoformat()} {day5['hhmm'].iloc[coil_idx]} IST (15m0 Bear Thrust)"
        s = _build_setup("SHORT", "STRICT_BEAR_V2", b_score, trig, r_list, max(float(bar15_0["high"]), vwap_0930))
        if s:
            setups.append(s)

    # 3) P14B GAP-DOWN SHAKEOUT -> ORH + VWAP RECLAIM (Concept #17)
    if abv_vwap_breadth >= 55.0 and gap_pct <= -0.60 and len(day5) >= 8:
        orh = float(bar15_0["high"])
        post = day5.iloc[5:]
        hist_5_med = float(w5[w5["trade_date"] < cur_d]["volume"].tail(120).median()) or 1.0
        reclaim = post[(post["close"] > orh) & (post["close"] > vwap_series.iloc[post.index]) & (post["volume"] >= 1.8 * hist_5_med)]
        if not reclaim.empty:
            rec_bar = reclaim.iloc[0]
            r_list = [
                f"P14B Gap-Down Shakeout Reclaim (Gap {gap_pct:+.2f}%)",
                f"Reclaimed ORH ₹{orh:.2f} & VWAP on {float(rec_bar['volume']) / hist_5_med:.1f}x vol",
                f"Strong Bull Breadth ({abv_vwap_breadth:.1f}% >VWAP)",
            ]
            trig = f"{cur_d.isoformat()} {rec_bar['hhmm']} IST (ORH + VWAP Reclaim)"
            s = _build_setup("LONG", "P14B_RECLAIM", 76.0, trig, r_list, float(vwap_series.iloc[reclaim.index[0]]))
            if s:
                setups.append(s)

    return setups


def merge_and_rank_setups(
    playbook_setups: list[dict[str, Any]],
    smc_setups: list[dict[str, Any]],
    breadth: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Merge Playbook v2 setups with SMC engine setups, enforce Breadth & Anti-Chasing penalties,
    attach explicit `trigger_time`, `playbook_tag`, and `reasons`, and assign deterministic Ranks #1..N.
    """
    abv_vwap = float(breadth.get("abv_vwap_pct", 50.0))
    by_key: dict[tuple[str, str], dict[str, Any]] = {}

    for s in smc_setups:
        enriched = dict(s)
        enriched.setdefault("playbook_tag", "SMC_OB")
        enriched.setdefault("monster_runner", False)
        enriched.setdefault("anti_chasing", False)
        if not enriched.get("trigger_time"):
            created = str(enriched.get("created_at", ""))
            hhmm = created[11:16] if len(created) >= 16 else "09:30"
            enriched["trigger_time"] = f"{breadth.get('trade_date', '')} {hhmm} IST (SMC Zone)".strip()
        reasons = list(enriched.get("reasons", []))
        if enriched["direction"] == "SHORT" and abv_vwap > 55.0:
            enriched["score"] = max(45, int(enriched["score"]) - 15)
            reasons.append(f"Breadth Gate Caution: F&O >VWAP {abv_vwap:.1f}% (>55%)")
        enriched["reasons"] = reasons
        enriched["reason_summary"] = " | ".join(reasons[:4])
        by_key[(enriched["symbol"], enriched["direction"])] = enriched

    for p in playbook_setups:
        key = (p["symbol"], p["direction"])
        existing = by_key.get(key)
        if existing:
            # Combine SMC confluence + Playbook trigger
            merged = dict(p)
            smc_reasons = [r for r in existing.get("reasons", []) if r not in merged["reasons"]]
            merged["reasons"] = merged["reasons"] + smc_reasons[:2]
            merged["reason_summary"] = " | ".join(merged["reasons"][:4])
            merged["score"] = min(98, int(p["score"]) + 2)
            merged["conviction"] = round(float(p.get("conviction", p["score"])) + 2.0, 2)
            by_key[key] = merged
        else:
            by_key[key] = p

    ranked = list(by_key.values())
    tag_priority = {"CREAM_BULL_V2": 4, "STRICT_BEAR_V2": 3, "P14B_RECLAIM": 2, "SMC_OB": 1}
    for item in ranked:
        if item.get("playbook_tag") in ("CREAM_BULL_V2", "STRICT_BEAR_V2") and not item.get("anti_chasing"):
            item["score"] = min(98, int(item.get("score", 0)) + 2)

    ranked.sort(
        key=lambda x: (
            0 if not x.get("anti_chasing", False) else -1,
            tag_priority.get(str(x.get("playbook_tag", "SMC_OB")), 1) >= 2,
            float(x.get("score", 0)),
            float(x.get("conviction", x.get("score", 0))),
            int(bool(x.get("monster_runner", False))),
            float(x.get("rr_ratio", 0)),
            -float(x.get("sl_pct", 1.0)),
        ),
        reverse=True,
    )
    for idx, item in enumerate(ranked, start=1):
        item["rank"] = idx
    return ranked


def _classify_option_timing(
    day5: pd.DataFrame,
    direction: str,
    ltp: float,
    vwap: float,
    move_open_pct: float,
    atr14: float = 0.0,
    atr_pct: float = 0.0,
    pdh: float = 0.0,
    pdl: float = 0.0,
) -> dict[str, Any]:
    """
    Classify current intraday 5m structure into Option-Buyer Timing phases
    grounded in the 27-case True One-Way DNA:
    1) Absolute Volatility & VWAP Gates:
       - Rejects low daily ATR (< 1.65%) — slow gamma theta drain
       - Rejects early VWAP loss (< 70% hold) — OIL-style fakeout risk
    2) Early-Morning Engine (Bars 3..17: 09:30 to 10:45 IST):
       - Rejects Wicky Chop (Bar Efficiency < 0.20)
       - Rejects Erratic 2-Way Zigzags (Path Efficiency < 0.35)
       - Rejects Adverse Wick Rejections (Directional CLV < 0.65)
       - Rejects Failed ORB & PDH/PDL inside-range traps
       - Rejects Stalled Morning Spikes (No new extreme & range expansion < 0.05 ATR after 09:45)
       - Rejects Hugging VWAP without thrust (< 0.04 ATR)
    3) Midday / Afternoon Engine (Bars >= 18: 10:45 to 15:30 IST):
       - Requires sustained post-10:15 drift (post1h_net >= 0.70%, eff >= 0.35)
       - Requires active last 1h continuation (h1_net >= 0.15%, rng >= 0.55%)
       - Requires range expansion >= 1.15x ATR
       - Requires 5m 9EMA hold >= 65%
    """
    dir_sign = 1.0 if direction == "LONG" else -1.0
    dir_vwap_dist = round(dir_sign * (ltp - vwap) / max(0.01, vwap) * 100.0, 2)
    pd_brk = (ltp > pdh) if (direction == "LONG" and pdh > 0) else ((ltp < pdl) if (direction == "SHORT" and pdl > 0) else False)

    n_bars = len(day5)
    if day5.empty or n_bars < 3:
        if dir_vwap_dist > 1.35 or abs(move_open_pct) >= 3.2:
            return {
                "option_phase": "WAIT_PULLBACK",
                "option_label": "WAIT PB (STRETCHED)",
                "option_safe": False,
                "ema9_dist_pct": 0.0,
                "efficiency_5m": 0.5,
                "early_bar_eff": 0.5,
                "early_path_eff": 0.5,
                "dir_clv": 0.5,
                "post1h_net": 0.0,
                "h1_net": 0.0,
                "vwap_hold_pct": 100.0,
                "ema9_hold_pct": 100.0,
                "rng_vs_atr": 0.0,
                "pd_brk": pd_brk,
                "orb_broken": False,
                "last2_rv": 1.0,
                "coil_pct": 0.0,
                "pts_adj": -20.0,
                "note": f"Extended ({move_open_pct:+.2f}% Open, VWAP {dir_vwap_dist:+.2f}%) — wait for pullback",
            }
        return {
            "option_phase": "EARLY_OPEN",
            "option_label": "OPENING 10M",
            "option_safe": False,
            "ema9_dist_pct": 0.0,
            "efficiency_5m": 0.5,
            "early_bar_eff": 0.5,
            "early_path_eff": 0.5,
            "dir_clv": 0.5,
            "post1h_net": 0.0,
            "h1_net": 0.0,
            "vwap_hold_pct": 100.0,
            "ema9_hold_pct": 100.0,
            "rng_vs_atr": 0.0,
            "pd_brk": pd_brk,
            "orb_broken": False,
            "last2_rv": 1.0,
            "coil_pct": 0.0,
            "pts_adj": -6.0,
            "note": "Opening 10m — waiting for 15m ORB structure confirmation",
        }

    c_b = day5["close"].to_numpy()
    h_b = day5["high"].to_numpy()
    l_b = day5["low"].to_numpy()
    o_b = day5["open"].to_numpy()
    v_b = day5["volume"].to_numpy()
    o0 = float(o_b[0])

    tp_s = (day5["high"] + day5["low"] + day5["close"]) / 3.0
    cum_v = day5["volume"].cumsum().replace(0, 1)
    cum_vwap = (tp_s * day5["volume"]).cumsum() / cum_v
    ema9_s = day5["close"].ewm(span=9, adjust=False).mean()
    ema20_s = day5["close"].ewm(span=20, adjust=False).mean()

    vwap_hold_pct = round(float(((day5["close"] - cum_vwap) * dir_sign >= 0).mean()) * 100.0, 1)
    ema9_hold_pct = round(float(((day5["close"] - ema9_s) * dir_sign >= -0.0015 * day5["close"]).mean()) * 100.0, 1)

    ema9 = float(ema9_s.iloc[-1])
    ema20 = float(ema20_s.iloc[-1])
    dir_ema9_dist = round(dir_sign * (ltp - ema9) / max(0.01, ema9) * 100.0, 2)
    ema_sep_pct = round(dir_sign * (ema9 - ema20) / max(0.01, ema20) * 100.0, 2)

    hi_day = float(day5["high"].max())
    lo_day = float(day5["low"].min())
    day_rng = max(0.01, hi_day - lo_day)
    rng_vs_atr = round(day_rng / max(0.01, atr14), 2) if atr14 > 0 else 1.5

    # Early structural indicators (from 27-case true one-way DNA)
    net_dir = max(0.0, dir_sign * (ltp - o0))
    sum_bar_rng = float(np.sum(h_b - l_b))
    bar_eff = round(net_dir / max(1e-6, sum_bar_rng), 2)
    path = float(np.sum(np.abs(np.diff(np.concatenate(([o0], c_b))))))
    path_eff = round(net_dir / max(1e-6, path), 2)
    clv_dir = round((ltp - lo_day) / day_rng if direction == "LONG" else (hi_day - ltp) / day_rng, 2)
    vw_dist_atr = round((dir_sign * (ltp - float(cum_vwap.iloc[-1]))) / max(0.01, atr14), 2) if atr14 > 0 else 0.2

    # ORB 15m breakout (bars 0..2)
    orb_hi = float(np.max(h_b[:min(3, n_bars)]))
    orb_lo = float(np.min(l_b[:min(3, n_bars)]))
    orb_broken = (ltp > orb_hi) if direction == "LONG" else (ltp < orb_lo)

    # Expansion from 09:45 (bar 6) to 10:15 (bar 12)
    if n_bars >= 6:
        prev_ext = float(np.max(h_b[:n_bars-3])) if direction == "LONG" else float(np.min(l_b[:n_bars-3]))
        curr_ext = float(np.max(h_b[n_bars-3:])) if direction == "LONG" else float(np.min(l_b[n_bars-3:]))
        new_ext = (curr_ext > prev_ext) if direction == "LONG" else (curr_ext < prev_ext)
    else:
        new_ext = True

    if n_bars >= 12:
        r6 = (float(np.max(h_b[:6])) - float(np.min(l_b[:6]))) / max(1e-6, atr14)
        rng_expansion = round(rng_vs_atr - r6, 2)
    else:
        rng_expansion = 0.20

    c_now = float(day5["close"].iloc[-1])
    c_3 = float(day5["close"].iloc[-4]) if n_bars >= 4 else o0
    c_5 = float(day5["close"].iloc[-6]) if n_bars >= 6 else o0
    net3_pct = round(dir_sign * (c_now - c_3) / max(0.01, c_3) * 100.0, 2)
    net5_pct = round(dir_sign * (c_now - c_5) / max(0.01, c_5) * 100.0, 2)
    path5 = float(day5["close"].diff().abs().tail(5).sum()) / max(0.01, c_5) * 100.0 if n_bars >= 6 else 1.0
    eff5 = round(min(1.0, abs(net5_pct) / max(0.02, path5)), 2)

    # Post-opening-hour sustained trend (after 10:15 IST) & Last-1h active expansion
    post1h_net = 0.0
    post1h_eff = eff5
    h1_net = net5_pct
    h1_rng = 0.0
    h1_eff = eff5
    has_midday = n_bars >= 18
    if has_midday:
        a1h = day5.iloc[12:]
        a_rng = (float(a1h["high"].max()) - float(a1h["low"].min())) / max(0.01, o0) * 100.0
        post1h_net = round(dir_sign * (float(a1h["close"].iloc[-1]) - float(a1h["open"].iloc[0])) / max(0.01, o0) * 100.0, 2)
        post1h_eff = round(min(1.0, max(0.0, post1h_net) / max(0.05, a_rng)), 2)

        l12 = day5.tail(12)
        h1_rng = round((float(l12["high"].max()) - float(l12["low"].min())) / max(0.01, o0) * 100.0, 2)
        h1_net = round(dir_sign * (float(l12["close"].iloc[-1]) - float(l12["open"].iloc[0])) / max(0.01, o0) * 100.0, 2)
        h1_eff = round(min(1.0, max(0.0, h1_net) / max(0.05, h1_rng)), 2)

    vol_hist = day5["volume"].iloc[:-2] if n_bars >= 5 else day5["volume"]
    med_vol = float(vol_hist.median()) or 1.0
    last2_rv = round(float(day5["volume"].tail(2).mean()) / med_vol, 2)

    day_span = max(0.01, hi_day - lo_day)
    day_pos = (ltp - lo_day) / day_span if direction == "LONG" else (hi_day - ltp) / day_span

    last4_hi = float(day5["high"].tail(4).max())
    last4_lo = float(day5["low"].tail(4).min())
    coil_pct = round((last4_hi - last4_lo) / max(0.01, ltp) * 100.0, 2)

    last_bar = day5.iloc[-1]
    bar_rng = max(0.01, float(last_bar["high"]) - float(last_bar["low"]))
    body_ratio = abs(float(last_bar["close"]) - float(last_bar["open"])) / bar_rng
    close_loc = (
        (float(last_bar["close"]) - float(last_bar["low"])) / bar_rng
        if direction == "LONG"
        else (float(last_bar["high"]) - float(last_bar["close"])) / bar_rng
    )

    dna_bonus = (
        (6.0 if pd_brk else 0.0)
        + (4.0 if orb_broken else 0.0)
        + min(8.0, max(0.0, rng_vs_atr - 1.0) * 4.0)
        + min(6.0, max(0.0, vwap_hold_pct - 75.0) * 0.25)
    )

    fp = _compute_volume_footprint(day5, direction)
    dir_move_open = round(move_open_pct * dir_sign, 2)

    # Footprint Order-Flow Confluence Bonus
    if (
        (direction == "LONG" and fp["h1_cvd_pct"] >= 12.0 and ltp >= fp["poc_price"])
        or (direction == "SHORT" and fp["h1_cvd_pct"] <= -12.0 and ltp <= fp["poc_price"])
    ):
        dna_bonus += 6.0

    # 1) Absolute Volatility & VWAP & Footprint Gates
    if atr_pct > 0 and atr_pct < 1.65:
        phase = "CHOP_DECAY"
        label = "LOW ATR SLUGGISH"
        safe = False
        pts_adj = -24.0
        note = f"Low daily ATR {atr_pct:.2f}% (<1.65%) — slow option gamma move"
    elif n_bars >= 12 and not fp["is_liquid"]:
        phase = "CHOP_DECAY"
        label = "ILLIQUID TRAP (<150k VOL)"
        safe = False
        pts_adj = -26.0
        note = f"Illiquid volume only {fp['total_volume']/1000:.0f}k shares — wide spreads & option decay"
    elif n_bars >= 9 and fp["delta_divergence"]:
        phase = "CHOP_DECAY"
        label = "DELTA DIVERGENCE (TRAP)"
        safe = False
        pts_adj = -24.0
        note = f"Delta divergence (CVD {fp['cvd_pct']:+.1f}%) — smart money selling into retail rally"
    elif vwap_hold_pct < 70.0:
        phase = "CHOP_DECAY"
        label = "VWAP BROKEN (FAKEOUT)"
        safe = False
        pts_adj = -26.0
        note = f"Session VWAP hold only {vwap_hold_pct:.0f}% (OIL-style chop/fakeout risk)"

    # 2) EARLY MORNING ENGINE (Bars 3 to 17: 09:30 to 10:45 IST)
    elif not has_midday:
        if dir_move_open < 1.25:
            phase = "CHOP_DECAY"
            label = "LOW DISPLACEMENT (<1.25%)"
            safe = False
            pts_adj = -24.0
            note = f"Move from open only {dir_move_open:+.2f}% (<1.25%) — insufficient thrust for options"
        elif bar_eff < 0.25:
            phase = "CHOP_DECAY"
            label = "WICKY CHOP (LOW EFF)"
            safe = False
            pts_adj = -24.0
            note = f"Bar eff {bar_eff:.2f} (<0.25) — overlapping wicks/shadows"
        elif path_eff < 0.45:
            phase = "CHOP_DECAY"
            label = "ERRATIC PATH (CHOP)"
            safe = False
            pts_adj = -24.0
            note = f"Path eff {path_eff:.2f} (<0.45) — 2-way zigzag decay"
        elif clv_dir < 0.65:
            phase = "CHOP_DECAY"
            label = "WEAK CLV / WICK REJECT"
            safe = False
            pts_adj = -22.0
            note = f"Directional CLV {clv_dir:.2f} (<0.65) — adverse wick rejection"
        elif n_bars >= 9 and not orb_broken and not pd_brk:
            phase = "CHOP_DECAY"
            label = "INSIDE ORB & PD RANGE"
            safe = False
            pts_adj = -22.0
            note = "Failed both 15m ORB & PDH/PDL break — inside-range trap"
        elif n_bars >= 9 and vw_dist_atr < 0.04:
            phase = "CHOP_DECAY"
            label = "HUGGING VWAP (NO THRUST)"
            safe = False
            pts_adj = -20.0
            note = f"Hugging VWAP ({vw_dist_atr:+.2f}A) — lack of directional thrust"
        elif n_bars >= 12 and not new_ext and rng_expansion < 0.05 and (clv_dir < 0.85 or bar_eff < 0.25):
            phase = "CHOP_DECAY"
            label = "STALLED MORNING RANGE"
            safe = False
            pts_adj = -26.0
            note = f"Morning move stalled (expansion {rng_expansion:+.2f}A, CLV {clv_dir:.2f})"
        elif dir_ema9_dist > 1.30 or (dir_vwap_dist > 2.20 and dir_ema9_dist > 0.85):
            phase = "WAIT_PULLBACK"
            label = "WAIT PB (STRETCHED)"
            safe = False
            pts_adj = -18.0
            note = f"Stretched above 9EMA ({dir_ema9_dist:+.2f}%, VWAP {dir_vwap_dist:+.2f}%) — wait pullback to 9EMA"
        elif net3_pct < -0.25:
            phase = "CHOP_DECAY"
            label = "PULLING BACK (WAIT)"
            safe = False
            pts_adj = -18.0
            note = f"Active pullback (15m {net3_pct:+.2f}%) — wait for 9EMA bounce"
        else:
            # Clean Early-Morning One-Way Runner Confirmed!
            phase = "ONEWAY_CONFIRMED"
            label = "ONE-WAY RUNNER"
            safe = True
            pts_adj = 18.0 + dna_bonus + min(10.0, bar_eff * 25.0)
            brk_tag = "PDH/L Brk · " if pd_brk else ("ORB Brk · " if orb_broken else "")
            note = f"{brk_tag}One-Way: BarEff {bar_eff:.2f} · PathEff {path_eff:.2f} · VWAP {vwap_hold_pct:.0f}% · CLV {clv_dir:.2f}"

    # 3) MIDDAY / AFTERNOON ENGINE (Bars >= 18: 10:45 to 15:30 IST)
    elif dir_move_open < 1.80:
        phase = "CHOP_DECAY"
        label = "LOW DISPLACEMENT (<1.8%)"
        safe = False
        pts_adj = -26.0
        note = f"Move from open only {dir_move_open:+.2f}% (<1.8%) — option theta decay trap"
    elif ema9_hold_pct < 65.0:
        phase = "CHOP_DECAY"
        label = "NO 9EMA RIDE (CHOP)"
        safe = False
        pts_adj = -22.0
        note = f"5m 9EMA hold only {ema9_hold_pct:.0f}% — choppy two-way swing"
    elif rng_vs_atr < 1.15:
        phase = "CHOP_DECAY"
        label = "LOW EXPANSION (<1.15x ATR)"
        safe = False
        pts_adj = -20.0
        note = f"Range {rng_vs_atr:.2f}x ATR (<1.15x) — inside-range option bleed"
    elif post1h_net < 0.70 or post1h_eff < 0.35:
        phase = "CHOP_DECAY"
        label = "SIDEWAYS AFTER 10:15"
        safe = False
        pts_adj = -26.0
        note = f"Post-10:15 drift {post1h_net:+.2f}% (Eff {post1h_eff:.2f}) — morning move stalled sideways"
    elif (h1_net < 0.15 or h1_rng < 0.55) and not (dir_move_open >= 2.20 and post1h_net >= 1.0 and post1h_eff >= 0.40 and vwap_hold_pct >= 75.0 and day_pos >= 0.65):
        phase = "CHOP_DECAY"
        label = "FLAT LAST 1H (DECAY)"
        safe = False
        pts_adj = -24.0
        note = f"Last 1h net {h1_net:+.2f}% inside {h1_rng:.2f}% box — sideways option decay risk"
    elif dir_ema9_dist > 1.45 or (dir_vwap_dist > 2.50 and dir_ema9_dist > 0.95):
        phase = "WAIT_PULLBACK"
        label = "WAIT PB (STRETCHED)"
        safe = False
        pts_adj = -18.0
        note = f"Stretched above 9EMA ({dir_ema9_dist:+.2f}%, VWAP {dir_vwap_dist:+.2f}%) — wait pullback to 9EMA"
    elif net3_pct < -0.30 or (net5_pct < -0.35 and h1_net < 0.30):
        phase = "CHOP_DECAY"
        label = "PULLING BACK (WAIT)"
        safe = False
        pts_adj = -18.0
        note = f"Active pullback (15m {net3_pct:+.2f}%, 25m {net5_pct:+.2f}%) — wait for 9EMA bounce"
    elif (
        dir_move_open >= 1.80
        and post1h_net >= 1.00
        and post1h_eff >= 0.40
        and (h1_net >= 0.30 or (post1h_net >= 2.0 and post1h_eff >= 0.50))
        and dir_ema9_dist <= 1.45
        and ema_sep_pct >= 0.10
    ):
        phase = "CLEAN_TREND"
        label = "ACTIVE RUNNER"
        safe = True
        pts_adj = 16.0 + dna_bonus + min(12.0, post1h_net * 2.2)
        brk_tag = "PDH/L Brk · " if pd_brk else ""
        note = f"{brk_tag}VWAP {vwap_hold_pct:.0f}% · {rng_vs_atr:.1f}x ATR · Post1h {post1h_net:+.2f}% · 1h {h1_net:+.2f}%"
    elif (
        dir_move_open >= 2.20
        and post1h_net >= 1.0
        and post1h_eff >= 0.40
        and vwap_hold_pct >= 75.0
        and day_pos >= 0.65
        and (-0.40 <= dir_ema9_dist <= 0.65)
        and h1_rng <= 1.50
    ):
        phase = "FLAG_CONSOLIDATION"
        label = "HIGH TIGHT FLAG"
        safe = True
        pts_adj = 14.0 + dna_bonus + min(8.0, post1h_net * 1.8)
        brk_tag = "PDH/L Brk · " if pd_brk else ""
        note = f"{brk_tag}Flag at Day Highs: VWAP {vwap_hold_pct:.0f}% · Post1h {post1h_net:+.2f}% · Box {h1_rng:.2f}%"
    elif (
        dir_move_open >= 1.80
        and (-0.25 <= dir_ema9_dist <= 0.55)
        and ema_sep_pct >= 0.12
        and (post1h_net >= 0.80 and post1h_eff >= 0.45 and h1_net >= 0.25)
        and (net3_pct >= 0.05 or (body_ratio >= 0.35 and close_loc >= 0.55))
        and last2_rv >= 1.15
    ):
        phase = "PB_BOUNCE"
        label = "TREND PULLBACK"
        safe = True
        pts_adj = 14.0 + dna_bonus + min(8.0, post1h_net * 2.0)
        brk_tag = "PDH/L Brk · " if pd_brk else ""
        note = f"{brk_tag}VWAP {vwap_hold_pct:.0f}% · EMA9 {ema9_hold_pct:.0f}% · {rng_vs_atr:.1f}x ATR · Post1h {post1h_net:+.2f}%"
    elif (
        dir_move_open >= 1.80
        and coil_pct <= 0.85
        and day_pos >= 0.70
        and (post1h_net >= 0.75 and post1h_eff >= 0.40)
        and net3_pct >= 0.10
        and close_loc >= 0.55
        and last2_rv >= 1.25
    ):
        phase = "COIL_BREAK"
        label = "COIL BREAKOUT"
        safe = True
        pts_adj = 12.0 + dna_bonus + min(6.0, post1h_net * 1.8)
        brk_tag = "PDH/L Brk · " if pd_brk else ""
        note = f"{brk_tag}Coil {coil_pct:.2f}% Breakout · VWAP {vwap_hold_pct:.0f}% · {rng_vs_atr:.1f}x ATR"
    else:
        phase = "CHOP_DECAY"
        label = "LOW THRUST (DECAY)"
        safe = False
        pts_adj = -16.0
        note = f"Weak continuation (Post1h {post1h_net:+.2f}%, 1h {h1_net:+.2f}%, {rng_vs_atr:.1f}x ATR)"

    return {
        "option_phase": phase,
        "option_label": label,
        "option_safe": safe,
        "ema9_dist_pct": dir_ema9_dist,
        "efficiency_5m": bar_eff if not has_midday else (post1h_eff if has_midday else eff5),
        "early_bar_eff": bar_eff,
        "early_path_eff": path_eff,
        "dir_clv": clv_dir,
        "vw_dist_atr": vw_dist_atr,
        "orb_broken": orb_broken,
        "rng_expansion": rng_expansion,
        "post1h_net": post1h_net,
        "h1_net": h1_net,
        "vwap_hold_pct": vwap_hold_pct,
        "ema9_hold_pct": ema9_hold_pct,
        "rng_vs_atr": rng_vs_atr,
        "pd_brk": pd_brk,
        "last2_rv": last2_rv,
        "coil_pct": coil_pct,
        "pts_adj": pts_adj,
        "note": note,
        "cvd_pct": fp.get("cvd_pct", 0.0),
        "h1_cvd_pct": fp.get("h1_cvd_pct", 0.0),
        "net_cvd_shares": fp.get("net_cvd_shares", 0.0),
        "poc_price": fp.get("poc_price", 0.0),
        "total_volume": fp.get("total_volume", 0.0),
        "delta_divergence": fp.get("delta_divergence", False),
    }


def evaluate_live_momentum_board(
    symbols: list[str],
    quotes_map: dict[str, dict[str, Any]],
    breadth: dict[str, Any],
    watchlist_items: list[dict[str, Any]] | None = None,
    ranked_setups: list[dict[str, Any]] | None = None,
    top_n: int = 15,
) -> dict[str, Any]:
    """
    Build the Right-Side 'Live Momentum & All Conditions Matched' board.
    Keeps the Left T-1 Watchlist untouched while ranking stocks actively showing
    live momentum + Playbook v2 / VWAP / Volume confluence Top -> Bottom (#1..N).
    """
    wl_map: dict[str, dict[str, Any]] = {
        str(w["symbol"]).upper(): w for w in (watchlist_items or []) if w.get("symbol")
    }
    setup_by_sym: dict[str, dict[str, Any]] = {}
    for s in (ranked_setups or []):
        sym_u = str(s["symbol"]).upper()
        if sym_u not in setup_by_sym:
            setup_by_sym[sym_u] = s

    abv_vwap_breadth = float(breadth.get("abv_vwap_pct", 50.0))
    now_ist = datetime.now(IST)
    trade_date_str = str(breadth.get("trade_date") or now_ist.date().isoformat())

    rows: list[dict[str, Any]] = []
    for sym in symbols:
        sym_u = sym.upper()
        if not is_valid_fno_symbol(sym_u):
            continue
        q = quotes_map.get(sym_u)
        if not q:
            continue
        ltp = float(q.get("last_price") or 0.0)
        vwap = float(q.get("average_price") or ltp)
        open_px = float(q.get("open") or ltp)
        pdc = float(q.get("prev_close") or open_px)
        if ltp <= 0 or open_px <= 0 or vwap <= 0:
            continue

        move_open_pct = round((ltp - open_px) / open_px * 100.0, 2)
        move_pdc_pct = round((ltp - pdc) / pdc * 100.0, 2) if pdc > 0 else move_open_pct
        gap_pct = round((open_px - pdc) / pdc * 100.0, 2) if pdc > 0 else 0.0
        vwap_diff_pct = round((ltp - vwap) / vwap * 100.0, 2)

        setup = setup_by_sym.get(sym_u)
        wl = wl_map.get(sym_u)

        # Direction comes from confirmed setup if present, else live O->LTP momentum
        if setup and setup.get("playbook_tag") in ("CREAM_BULL_V2", "STRICT_BEAR_V2", "P14B_RECLAIM"):
            direction = str(setup["direction"])
        else:
            direction = "LONG" if move_open_pct >= 0 else "SHORT"

        vwap_aligned = (ltp >= vwap) if direction == "LONG" else (ltp <= vwap)
        dir_move = move_open_pct if direction == "LONG" else -move_open_pct

        # Inspect intraday 15m0 / recent volume surge & T-1 metrics from cached DB if available
        r15_0_rv = 1.0
        clv_15_0 = 0.5
        trig_time = f"{trade_date_str} {now_ist.strftime('%H:%M')} IST (Live Quote)"
        t1_ret_pct = float(wl.get("t1_ret_pct", 0.0)) if wl else 0.0
        pm_rv = float(wl.get("pm_rv", 0.0)) if wl else 0.0
        atr_pct = float(wl.get("atr_pct", 0.0)) if wl else 0.0

        c5 = db.load_candles(sym_u, "5m", 180)
        c15 = db.load_candles(sym_u, "15m", 80)
        c1h = db.load_candles(sym_u, "1h", 120)
        c1d = db.load_candles(sym_u, "1D", 250)
        day5 = pd.DataFrame()
        cur_d = now_ist.date()
        if c5 and c15:
            df5 = _candles_to_df(c5)
            df15 = _candles_to_df(c15)
            max_d = df5["trade_date"].max()
            is_live_mode = str(breadth.get("source") or breadth.get("data_source") or "").lower() == "live"
            if is_live_mode and max_d != now_ist.date():
                # In live mode, ignore stale candles from prior days so yesterday's winners aren't evaluated as today
                day15 = pd.DataFrame()
                day5 = pd.DataFrame()
                cur_d = now_ist.date()
            else:
                cur_d = max_d
                day15 = df15[df15["trade_date"] == cur_d]
                hist15 = df15[df15["trade_date"] < cur_d].tail(60)
                if not day15.empty and not hist15.empty:
                    med15 = float(hist15["volume"].median()) or 1.0
                    bar0 = day15.iloc[0]
                    r15_0_rv = round(float(bar0["volume"]) / med15, 2)
                    rng0 = max(0.01, float(bar0["high"]) - float(bar0["low"]))
                    clv_15_0 = round((float(bar0["close"]) - float(bar0["low"])) / rng0, 2)
                day5 = df5[df5["trade_date"] == cur_d]
                if not day5.empty:
                    trig_time = f"{cur_d.isoformat()} {day5['hhmm'].iloc[-1]} IST (Live Session)"

        atr14_val = 0.0
        pdh_val = 0.0
        pdl_val = 0.0
        if c1d:
            df1 = _candles_to_df(c1d)
            d1_hist = df1[df1["trade_date"] < cur_d].tail(16).reset_index(drop=True)
            if len(d1_hist) >= 12:
                h_arr = d1_hist["high"].to_numpy()
                l_arr = d1_hist["low"].to_numpy()
                c_arr = d1_hist["close"].to_numpy()
                tr_arr = np.maximum(h_arr[1:] - l_arr[1:], np.maximum(np.abs(h_arr[1:] - c_arr[:-1]), np.abs(l_arr[1:] - c_arr[:-1])))
                atr14_val = float(pd.Series(tr_arr).rolling(min(14, len(tr_arr))).mean().iloc[-1])
                ref_c = float(c_arr[-1])
                if atr_pct <= 0 and ref_c > 0:
                    atr_pct = round(atr14_val / ref_c * 100.0, 2)
                pdh_val = float(h_arr[-1])
                pdl_val = float(l_arr[-1])

        if atr14_val <= 0 and atr_pct > 0 and open_px > 0:
            atr14_val = open_px * (atr_pct / 100.0)
        if pdh_val <= 0 and pdc > 0:
            pdh_val = pdc * 1.008
            pdl_val = pdc * 0.992

        if setup and setup.get("trigger_time"):
            trig_time = str(setup["trigger_time"])

        opt_timing = _classify_option_timing(
            day5,
            direction,
            ltp,
            vwap,
            move_open_pct,
            atr14=atr14_val,
            atr_pct=atr_pct,
            pdh=pdh_val,
            pdl=pdl_val,
        )

        # Evaluate 6 core Playbook v2 live conditions + Option Anti-Decay timing
        matched = 0
        reasons: list[str] = []
        pts = 0.0

        pb_tag = str(setup.get("playbook_tag", "")) if setup else ""
        pb_confirmed = pb_tag in ("CREAM_BULL_V2", "STRICT_BEAR_V2", "P14B_RECLAIM")
        if pb_confirmed:
            matched += 1
            pts += 18.0
            reasons.append(f"{pb_tag} Playbook Confirmed")

        if wl:
            matched += 1
            pts += 10.0
            reasons.append(f"From T-1 Watchlist (#{wl.get('rank', '?')} · PM {pm_rv:.2f}x)")
        elif pm_rv >= 1.8:
            matched += 1
            pts += 7.0

        if vwap_aligned:
            matched += 1
            # Sweet spot VWAP hold (+0.1% to +1.0%); avoid rewarding over-stretched distance
            pts += 12.0 if abs(vwap_diff_pct) <= 1.1 else 5.0
            reasons.append(f"VWAP Aligned ({vwap_diff_pct:+.2f}% vs ₹{vwap:.2f})")
        else:
            pts -= 12.0

        if dir_move >= 0.60:
            matched += 1
            # Reward healthy intraday expansion; full thrust for strong trenders
            move_pts = 11.0 + min(5.0, (min(dir_move, 2.5) - 0.60) * 2.5) if dir_move <= 3.0 else 16.0
            pts += move_pts
            reasons.append(f"Move {move_open_pct:+.2f}% Open ({move_pdc_pct:+.2f}% Day)")

        vol_ok = (r15_0_rv >= 2.0 and ((direction == "LONG" and clv_15_0 >= 0.55) or (direction == "SHORT" and clv_15_0 <= 0.40)))
        if vol_ok:
            matched += 1
            pts += 9.0 + min(6.0, (r15_0_rv - 2.0) * 1.5)
            reasons.append(f"15m0 Thrust {r15_0_rv:.2f}x (CLV {clv_15_0:.2f})")

        counter_breadth = False
        if direction == "LONG":
            if abv_vwap_breadth >= 45.0 or r15_0_rv >= 3.0:
                matched += 1
                pts += 5.0
            elif abv_vwap_breadth < 35.0:
                counter_breadth = True
                pts -= 16.0
                reasons.append(f"Counter-Breadth Caution ({abv_vwap_breadth:.1f}% >VWAP Market Tide)")
            else:
                pts -= 5.0
        else:
            if abv_vwap_breadth <= 55.0 or r15_0_rv >= 3.0:
                matched += 1
                pts += 5.0
            elif abv_vwap_breadth > 65.0:
                counter_breadth = True
                pts -= 16.0
                reasons.append(f"Counter-Breadth Caution ({abv_vwap_breadth:.1f}% >VWAP Market Tide)")
            else:
                pts -= 5.0

        pts += float(opt_timing["pts_adj"])
        pts += min(8.0, max(0.0, (float(opt_timing["last2_rv"]) - 1.0) * 2.5))
        reasons.insert(0, opt_timing["note"])

        # T-1 Volume Footprint Rules:
        # 1. Springboard Accumulation: T-1 high PM volume into a tight base/pullback
        if pm_rv >= 2.5 and abs(t1_ret_pct) <= 2.5:
            pts += 8.0
            reasons.append(f"T-1 PM Springboard Accumulation ({pm_rv:.1f}x vol into base)")

        # 2. T-1 Exhaustion: Already ran > 4.0% on T-1 on peak volume (e.g. MOTILALOFS)
        t1_exhausted = (direction == "LONG" and t1_ret_pct >= 4.0) or (direction == "SHORT" and t1_ret_pct <= -4.0)
        anti_chasing = bool(setup.get("anti_chasing")) if setup else (t1_exhausted or (t1_ret_pct >= 2.0 and gap_pct >= 0.40))
        if anti_chasing:
            pts -= 18.0
            reasons.append(f"ANTI-CHASING / T-1 EXHAUSTION (T-1 {t1_ret_pct:+.2f}% + Gap {gap_pct:+.2f}%)")

        # Option Buyer Absolute Rule: A stock MUST have dir_move >= 1.80% (or 1.25% early morning)
        # to ever be considered Option-Ready (Safe). Below that, it's guaranteed theta decay!
        # Must pass option timing engine, must not be anti-chasing, and must not fight broader market tide.
        min_opt_move = 1.25 if (len(day5) < 18) else 1.80
        is_opt_safe = (
            bool(opt_timing["option_safe"])
            and (dir_move >= min_opt_move)
            and vwap_aligned
            and not anti_chasing
            and not counter_breadth
        )

        # Look up F&O lot size from DB instruments
        inst_meta = db.get_instrument(sym_u)
        lot_size = int(inst_meta.get("fno_lot_size") or 100) if inst_meta else 100

        # Smart Money Order Block & Strike Recommendation (Top-Down MTF: Weekly -> Daily -> Hourly -> 5m)
        ob_info = compute_order_block(day5, direction, ltp, vwap, c1d=c1d, c1h=c1h)
        opt_strike = recommend_option_strike(sym_u, ltp, direction, lot_size)

        # Volume Delta shares & OI build-up
        net_cvd = float(opt_timing.get("net_cvd_shares") or 0.0)
        cvd_lakhs = round(net_cvd / 100000.0, 2)
        vol_delta_shares = f"{cvd_lakhs:+.2f}L"
        vol_delta_status = "above VWAP" if ltp >= vwap else "below VWAP"

        if direction == "LONG":
            oi_build_up = "LONG BUILD-UP" if opt_timing.get("cvd_pct", 0) >= 0 else "SHORT COVERING"
        else:
            oi_build_up = "SHORT BUILD-UP" if opt_timing.get("cvd_pct", 0) <= 0 else "LONG UNWINDING"

        # Confluence bonus for Order Block Confirmation (never overrides option_safe timing)
        if ob_info.get("htf_warning"):
            pts -= 22.0
            is_opt_safe = False
            reasons.append(f"⚠️ HTF WARNING: {ob_info['htf_warning']}")
        elif ob_info["status"] == "OB_BREAKOUT_CONFIRMED":
            pts += 15.0
            reasons.append(f"1H OB Breakdown Confirmed < ₹{ob_info['ob_low']} (S1 ₹{ob_info['s1']} · R1 ₹{ob_info['r1']})")
            if not pb_confirmed and is_opt_safe:
                status_tag = "OB BREAKOUT"
        elif ob_info["status"] == "OB_RETEST_BOUNCE":
            pts += 12.0
            reasons.append(f"1H OB Retest Bounce from ₹{ob_info['ob_high']} (S1 ₹{ob_info['s1']} · R1 ₹{ob_info['r1']})")
            if not pb_confirmed and is_opt_safe:
                status_tag = "OB RETEST"
        elif ob_info["status"] == "OB_BASE_FORMING":
            reasons.append(f"1H OB Base Forming ({ob_info['ob_zone']}) — Watch for Breakout")

        if ob_info.get("weekly_ob", {}).get("zone") != "None":
            reasons.append(f"MTF OB Structure: Weekly {ob_info['weekly_ob']['zone']} · Daily {ob_info['daily_ob']['zone']}")

        # Strict Filter: Must be VWAP aligned
        if not vwap_aligned:
            continue

        # Strict Noise & Sideways Filter:
        # Discard stocks flagged in CHOP_DECAY / DEAD SIDEWAYS unless confirmed by high volume & Playbook v2
        is_chop_decay = opt_timing.get("option_phase") == "CHOP_DECAY"
        if is_chop_decay and not pb_confirmed and r15_0_rv < 2.5:
            continue

        if not (pb_confirmed or is_opt_safe or (matched >= 4 and dir_move >= 1.5)):
            continue

        score = round(min(98.5, max(35.0, 20.0 + pts)), 1)
        monster = bool((setup and setup.get("monster_runner")) or (atr_pct >= 2.5 and r15_0_rv >= 3.0))
        if pb_confirmed:
            status_tag = pb_tag
        elif wl and dir_move >= 1.2:
            status_tag = "WL + MOMENTUM"
        elif is_opt_safe:
            status_tag = "ONE-WAY CONFIRMED" if opt_timing["option_phase"] == "ONEWAY_CONFIRMED" else "TREND CONFIRMED"
        else:
            status_tag = "LIVE MOMENTUM"

        rows.append(
            {
                "symbol": sym_u,
                "direction": direction,
                "status_tag": status_tag,
                "option_phase": opt_timing["option_phase"],
                "option_label": opt_timing["option_label"],
                "option_safe": is_opt_safe,
                "ema9_dist_pct": opt_timing["ema9_dist_pct"],
                "efficiency_5m": opt_timing["efficiency_5m"],
                "early_bar_eff": opt_timing.get("early_bar_eff", 0.0),
                "early_path_eff": opt_timing.get("early_path_eff", 0.0),
                "dir_clv": opt_timing.get("dir_clv", 0.5),
                "post1h_net": opt_timing.get("post1h_net", 0.0),
                "h1_net": opt_timing.get("h1_net", 0.0),
                "vwap_hold_pct": opt_timing.get("vwap_hold_pct", 100.0),
                "ema9_hold_pct": opt_timing.get("ema9_hold_pct", 100.0),
                "rng_vs_atr": opt_timing.get("rng_vs_atr", 0.0),
                "pd_brk": bool(opt_timing.get("pd_brk")),
                "orb_broken": bool(opt_timing.get("orb_broken")),
                "last2_rv": opt_timing["last2_rv"],
                "in_watchlist": bool(wl),
                "watchlist_rank": int(wl["rank"]) if wl and wl.get("rank") else None,
                "monster_runner": monster,
                "anti_chasing": anti_chasing,
                "cvd_pct": opt_timing.get("cvd_pct", 0.0),
                "h1_cvd_pct": opt_timing.get("h1_cvd_pct", 0.0),
                "poc_price": opt_timing.get("poc_price", 0.0),
                "total_volume": opt_timing.get("total_volume", 0.0),
                "conditions_matched": matched,
                "total_conditions": 6,
                "score": score,
                "ltp": round(ltp, 2),
                "vwap": round(vwap, 2),
                "move_pct": move_open_pct,
                "day_pct": move_pdc_pct,
                "vwap_diff_pct": vwap_diff_pct,
                "rel_vol": r15_0_rv,
                "trigger_time": trig_time,
                "setup_id": setup.get("id") if setup else None,
                "lot_size": lot_size,
                "order_block": ob_info,
                "option_strike": opt_strike,
                "oi_build_up": oi_build_up,
                "vol_delta_shares": vol_delta_shares,
                "vol_delta_status": vol_delta_status,
                "net_cvd_shares": net_cvd,
                "reasons": reasons,
                "updated_at": now_ist.isoformat(timespec="seconds"),
            }
        )

    rows.sort(
        key=lambda x: (
            int(bool(x["option_safe"])),
            0 if not x["anti_chasing"] else -1,
            int(x.get("order_block", {}).get("status") == "OB_BREAKOUT_CONFIRMED"),
            int(x["status_tag"] in ("CREAM_BULL_V2", "STRICT_BEAR_V2", "P14B_RECLAIM")),
            x["score"],
            x["conditions_matched"],
            int(x["in_watchlist"]),
        ),
        reverse=True,
    )
    selected = rows[:top_n]
    total_sel = len(selected)
    for idx, item in enumerate(selected, start=1):
        item["rank"] = idx
        percentile = round((1.0 - (idx - 1) / max(1, total_sel)) * 100.0, 1)
        item["percentile_rank"] = f"p{percentile:.1f} #{idx}/{total_sel}"
        item["conviction_score"] = int(round(min(99.0, max(50.0, item["score"]))))

    db.save_live_momentum(trade_date_str, selected)
    new_alerts = evaluate_and_record_alerts(selected, trade_date_str)
    return {
        "trade_date": trade_date_str,
        "count": len(selected),
        "items": selected,
        "new_alerts_count": len(new_alerts),
    }


def evaluate_and_record_alerts(
    board_items: list[dict[str, Any]],
    trade_date: str | None = None,
) -> list[dict[str, Any]]:
    """
    Evaluates live momentum board candidates against historical alerts and
    appends new trigger events to SQLite `live_alerts`.

    Rules for deduplication and anti-spam:
    1. NEVER delete or overwrite existing alerts (append-only chronological log).
    2. FIRST DETECTION: If symbol has no alert today, and is option_safe (or conviction >= 70),
       record entry alert (OB_BREAKOUT, OB_RETEST_BOUNCE, or MOMENTUM_EXPANSION).
    3. SUBSEQUENT DETECTIONS (only record if significant new event):
       - TARGET_R1_HIT: LTP reached R1 expansion target.
       - INVALIDATION_EXIT: Support S1 breached.
       - OB_BREAKOUT: Promoted from base forming / retest to confirmed breakout.
       - NEW_HIGHER_OB / NEW_LOWER_OB: Fresh Order Block established >= 1.0% higher/lower
         and >= 8 minutes elapsed since last alert.
       - INSTITUTIONAL_SQUEEZE: Volume delta surge or Call Unwinding Squeeze confirmed.
       All other duplicate scans within cooldown are suppressed.
    """
    if not board_items:
        return []

    now_ist = datetime.now(IST)
    now_epoch = int(now_ist.timestamp())
    now_hhmmss = now_ist.strftime("%I:%M:%S %p")
    trade_date_str = trade_date or str(now_ist.date())

    new_alerts: list[dict[str, Any]] = []

    for item in board_items:
        sym = str(item.get("symbol", "")).upper()
        if not sym:
            continue

        is_opt_safe = bool(item.get("option_safe"))
        score = float(item.get("score", 0.0))
        # Strict Option Buyer Protection:
        # NEVER trigger entry alerts on choppy, stalled, or decaying stocks!
        if not is_opt_safe:
            continue

        direction = str(item.get("direction", "LONG")).upper()
        ltp = float(item.get("ltp", 0.0))
        vwap = float(item.get("vwap", 0.0))
        move_pct = float(item.get("move_pct", 0.0))
        ob_info = item.get("order_block", {})
        ob_status = str(ob_info.get("status", "OB_ACTIVE"))
        ob_zone = str(ob_info.get("ob_zone", ""))
        s1 = float(ob_info.get("s1", ltp * 0.99))
        r1 = float(ob_info.get("r1", ltp * 1.015))
        r2 = float(ob_info.get("r2", ltp * 1.03))

        opt_strike = item.get("option_strike") or {}
        opt_name = str(opt_strike.get("option_name", f"{sym} ATM"))
        opt_prem = float(opt_strike.get("est_premium", 0.0))
        opt_cap = float(opt_strike.get("est_capital_inr", 0.0))
        opt_sl = float(opt_strike.get("sl_premium", 0.0))
        opt_tp1 = float(opt_strike.get("tp1_premium", 0.0))
        oi_buildup = str(item.get("oi_build_up", ""))
        vol_delta = str(item.get("vol_delta_shares", ""))
        reasons = item.get("reasons", [])

        last_alert = db.get_last_alert_for_symbol(sym, trade_date_str)

        trigger_event: str | None = None
        event_reason: str = ""

        if last_alert is None:
            # Case 1: First detection of the day for this symbol
            if ob_status == "OB_BREAKOUT_CONFIRMED":
                trigger_event = "OB_BREAKOUT"
                event_reason = f"Order Block breakout confirmed above {ob_zone} with clean VWAP hold"
            elif ob_status == "OB_RETEST_BOUNCE":
                trigger_event = "OB_RETEST_BOUNCE"
                event_reason = f"Pullback retest and bounce inside Order Block {ob_zone}"
            elif item.get("status_tag") in ("CREAM_BULL_V2", "STRICT_BEAR_V2", "P14B_RECLAIM"):
                trigger_event = "MOMENTUM_EXPANSION"
                event_reason = f"Strong momentum expansion with {oi_buildup} & Volume Delta {vol_delta}"
            else:
                # Do NOT fire generic alerts when stock is merely resting/forming a base inside a range!
                continue
        else:
            # Case 2: Subsequent check - check for new catalysts or exits
            prior_epoch = int(last_alert.get("epoch_ts", 0))
            elapsed_sec = max(0, now_epoch - prior_epoch)
            prior_event = str(last_alert.get("event_type", ""))
            prior_price = float(last_alert.get("price", ltp))
            prior_r1 = float(last_alert.get("r1", 0.0))
            prior_s1 = float(last_alert.get("s1", 0.0))

            # a) Target R1 Reached (Profit Booking / Trail SL)
            if direction == "LONG" and ltp >= prior_r1 and prior_event != "TARGET_R1_HIT" and prior_r1 > 0:
                trigger_event = "TARGET_R1_HIT"
                event_reason = f"Target R1 ₹{prior_r1:,.2f} reached (+{((ltp - prior_price)/prior_price)*100.0:.1f}%), book partial or trail SL"
            elif direction == "SHORT" and ltp <= prior_r1 and prior_event != "TARGET_R1_HIT" and prior_r1 > 0:
                trigger_event = "TARGET_R1_HIT"
                event_reason = f"Target R1 ₹{prior_r1:,.2f} reached (-{((prior_price - ltp)/prior_price)*100.0:.1f}%), book partial or trail SL"

            # b) Invalidation S1 Breached (Cut Loss / Exit Alert)
            elif direction == "LONG" and ltp < prior_s1 and prior_event != "INVALIDATION_EXIT" and prior_s1 > 0:
                trigger_event = "INVALIDATION_EXIT"
                event_reason = f"Support S1 ₹{prior_s1:,.2f} breached, setup invalidated - exit immediately"
            elif direction == "SHORT" and ltp > prior_s1 and prior_event != "INVALIDATION_EXIT" and prior_s1 > 0:
                trigger_event = "INVALIDATION_EXIT"
                event_reason = f"Resistance S1 ₹{prior_s1:,.2f} breached, setup invalidated - exit immediately"

            # c) Promotion to Confirmed Breakout
            elif prior_event in ("ORDER_BLOCK_TRIGGER", "OB_RETEST_BOUNCE") and ob_status == "OB_BREAKOUT_CONFIRMED" and elapsed_sec >= 180:
                trigger_event = "OB_BREAKOUT"
                event_reason = f"Promoted from consolidation to Confirmed Breakout above {ob_zone}"

            # d) New Higher/Lower Order Block formed after continuation (min 8 mins & 1.0% shift)
            elif direction == "LONG" and ((ltp - prior_price) / max(1.0, prior_price) >= 0.010) and ob_zone != last_alert.get("ob_zone") and elapsed_sec >= 480:
                trigger_event = "NEW_HIGHER_OB"
                event_reason = f"Continuation impulse formed new higher Order Block at {ob_zone}"
            elif direction == "SHORT" and ((prior_price - ltp) / max(1.0, prior_price) >= 0.010) and ob_zone != last_alert.get("ob_zone") and elapsed_sec >= 480:
                trigger_event = "NEW_LOWER_OB"
                event_reason = f"Continuation impulse formed new lower Order Block at {ob_zone}"

            # e) Institutional Squeeze Surge
            elif oi_buildup in ("SHORT_COVERING", "CALL_UNWINDING_SQUEEZE") and last_alert.get("oi_build_up") not in ("SHORT_COVERING", "CALL_UNWINDING_SQUEEZE") and elapsed_sec >= 300:
                trigger_event = "INSTITUTIONAL_SQUEEZE"
                event_reason = f"Institutional Call Unwinding squeeze detected with Delta {vol_delta}"

        if trigger_event is not None:
            alert_id = f"ALT_{sym}_{now_ist.strftime('%Y%m%d_%H%M%S')}_{trigger_event}"
            reasons_full = [event_reason] + [r for r in reasons if r != event_reason]
            alert_record = {
                "alert_id": alert_id,
                "trade_date": trade_date_str,
                "timestamp_ist": now_hhmmss,
                "epoch_ts": now_epoch,
                "symbol": sym,
                "direction": direction,
                "event_type": trigger_event,
                "price": round(ltp, 2),
                "vwap": round(vwap, 2),
                "move_pct": round(move_pct, 2),
                "oi_build_up": oi_buildup,
                "vol_delta_shares": vol_delta,
                "ob_zone": ob_zone,
                "s1": round(s1, 2),
                "r1": round(r1, 2),
                "r2": round(r2, 2),
                "option_strike_name": opt_name,
                "option_est_premium": round(opt_prem, 2),
                "option_lot_capital": round(opt_cap, 2),
                "option_sl_premium": round(opt_sl, 2),
                "option_tp1_premium": round(opt_tp1, 2),
                "reasons": reasons_full,
                "created_at": now_ist.isoformat(timespec="seconds"),
            }
            db.save_live_alert(alert_record)
            new_alerts.append(alert_record)

    return new_alerts
