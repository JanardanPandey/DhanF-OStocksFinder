from __future__ import annotations

import numpy as np
import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute EMA(20/50/200), ATR(14), RSI(14), relative volume, and session VWAP."""
    if df.empty:
        return df.copy()

    out = df.copy()
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    vol = out["volume"].astype(float)

    out["ema20"] = close.ewm(span=20, adjust=False).mean()
    out["ema50"] = close.ewm(span=50, adjust=False).mean()
    out["ema200"] = close.ewm(span=200, adjust=False).mean()

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean().bfill()

    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0.0, np.nan)
    out["rsi"] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

    vol_sma = vol.rolling(20, min_periods=1).mean().replace(0.0, 1.0)
    out["rel_vol"] = (vol / vol_sma).fillna(1.0)

    # Intraday session VWAP grouped by IST trading day
    ts_ist = pd.to_datetime(out["ts"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
    out["trade_date"] = ts_ist.dt.date
    typical = (high + low + close) / 3.0
    tp_vol = typical * vol
    out["vwap"] = (
        tp_vol.groupby(out["trade_date"]).cumsum()
        / vol.groupby(out["trade_date"]).cumsum().replace(0.0, np.nan)
    ).fillna(close)

    return out


def detect_swings(df: pd.DataFrame, left: int = 3, right: int = 3) -> tuple[list[dict], list[dict]]:
    """
    Detect confirmed fractal swing highs and swing lows without lookahead beyond `right` bars.
    Returns (swing_highs, swing_lows) sorted chronologically.
    """
    n = len(df)
    if n < left + right + 1:
        return [], []

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    ts_arr = df["ts"].to_numpy()

    swing_highs: list[dict] = []
    swing_lows: list[dict] = []

    for i in range(left, n - right):
        win_h = highs[i - left : i + right + 1]
        win_l = lows[i - left : i + right + 1]
        if highs[i] == np.max(win_h) and np.sum(win_h == highs[i]) == 1:
            swing_highs.append({"idx": i, "ts": int(ts_arr[i]), "price": float(highs[i])})
        if lows[i] == np.min(win_l) and np.sum(win_l == lows[i]) == 1:
            swing_lows.append({"idx": i, "ts": int(ts_arr[i]), "price": float(lows[i])})

    return swing_highs, swing_lows
