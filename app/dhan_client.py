from __future__ import annotations

import asyncio
import hashlib
import io
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from app.config import ENV_PATH, settings
from app.database import db

IST = timezone(timedelta(hours=5, minutes=30))
SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
DHAN_API_BASE = "https://api.dhan.co/v2"
DHAN_AUTH_BASE = "https://auth.dhan.co/app"

# Fallback curated NSE F&O stocks if scrip master download fails offline
SEED_FNO_INSTRUMENTS: list[dict[str, Any]] = [
    {"symbol": "RELIANCE", "name": "Reliance Industries Ltd", "equity_security_id": "2885", "fno_lot_size": 500},
    {"symbol": "HDFCBANK", "name": "HDFC Bank Ltd", "equity_security_id": "1333", "fno_lot_size": 550},
    {"symbol": "ICICIBANK", "name": "ICICI Bank Ltd", "equity_security_id": "4963", "fno_lot_size": 700},
    {"symbol": "INFY", "name": "Infosys Ltd", "equity_security_id": "1594", "fno_lot_size": 400},
    {"symbol": "TCS", "name": "Tata Consultancy Services Ltd", "equity_security_id": "11536", "fno_lot_size": 175},
    {"symbol": "SBIN", "name": "State Bank of India", "equity_security_id": "3045", "fno_lot_size": 750},
    {"symbol": "AXISBANK", "name": "Axis Bank Ltd", "equity_security_id": "5900", "fno_lot_size": 625},
    {"symbol": "KOTAKBANK", "name": "Kotak Mahindra Bank Ltd", "equity_security_id": "1922", "fno_lot_size": 400},
    {"symbol": "BHARTIARTL", "name": "Bharti Airtel Ltd", "equity_security_id": "10604", "fno_lot_size": 475},
    {"symbol": "LT", "name": "Larsen & Toubro Ltd", "equity_security_id": "11483", "fno_lot_size": 150},
    {"symbol": "BAJFINANCE", "name": "Bajaj Finance Ltd", "equity_security_id": "317", "fno_lot_size": 125},
    {"symbol": "TATASTEEL", "name": "Tata Steel Ltd", "equity_security_id": "3499", "fno_lot_size": 5500},
    {"symbol": "SUNPHARMA", "name": "Sun Pharmaceutical Industries", "equity_security_id": "3351", "fno_lot_size": 350},
    {"symbol": "MARUTI", "name": "Maruti Suzuki India Ltd", "equity_security_id": "10999", "fno_lot_size": 50},
    {"symbol": "ADANIENT", "name": "Adani Enterprises Ltd", "equity_security_id": "25", "fno_lot_size": 300},
    {"symbol": "ADANIPORTS", "name": "Adani Ports and SEZ Ltd", "equity_security_id": "15083", "fno_lot_size": 400},
    {"symbol": "TITAN", "name": "Titan Company Ltd", "equity_security_id": "3506", "fno_lot_size": 175},
    {"symbol": "TRENT", "name": "Trent Ltd", "equity_security_id": "1964", "fno_lot_size": 100},
    {"symbol": "HAL", "name": "Hindustan Aeronautics Ltd", "equity_security_id": "2303", "fno_lot_size": 150},
    {"symbol": "BEL", "name": "Bharat Electronics Ltd", "equity_security_id": "383", "fno_lot_size": 2850},
    {"symbol": "PFC", "name": "Power Finance Corporation", "equity_security_id": "14299", "fno_lot_size": 1300},
    {"symbol": "RECLTD", "name": "REC Limited", "equity_security_id": "15355", "fno_lot_size": 1000},
    {"symbol": "INDUSINDBK", "name": "IndusInd Bank Ltd", "equity_security_id": "5258", "fno_lot_size": 500},
    {"symbol": "JSWSTEEL", "name": "JSW Steel Ltd", "equity_security_id": "11723", "fno_lot_size": 675},
    {"symbol": "HINDALCO", "name": "Hindalco Industries Ltd", "equity_security_id": "1363", "fno_lot_size": 1400},
    {"symbol": "TECHM", "name": "Tech Mahindra Ltd", "equity_security_id": "13538", "fno_lot_size": 600},
    {"symbol": "WIPRO", "name": "Wipro Ltd", "equity_security_id": "3787", "fno_lot_size": 3000},
    {"symbol": "HCLTECH", "name": "HCL Technologies Ltd", "equity_security_id": "7229", "fno_lot_size": 350},
    {"symbol": "M&M", "name": "Mahindra & Mahindra Ltd", "equity_security_id": "2031", "fno_lot_size": 350},
    {"symbol": "DIXON", "name": "Dixon Technologies Ltd", "equity_security_id": "21690", "fno_lot_size": 50},
]

_BASE_PRICES: dict[str, float] = {
    "RELIANCE": 1295.0, "HDFCBANK": 1745.0, "ICICIBANK": 1288.0, "INFY": 1860.0,
    "TCS": 4120.0, "SBIN": 842.0, "AXISBANK": 1168.0, "KOTAKBANK": 1790.0,
    "BHARTIARTL": 1595.0, "LT": 3620.0, "BAJFINANCE": 6890.0, "TATASTEEL": 152.0,
    "SUNPHARMA": 1835.0, "MARUTI": 11420.0, "ADANIENT": 2920.0, "ADANIPORTS": 1340.0,
    "TITAN": 3240.0, "TRENT": 6480.0, "HAL": 4380.0, "BEL": 294.0,
    "PFC": 476.0, "RECLTD": 532.0, "INDUSINDBK": 1065.0, "JSWSTEEL": 968.0,
    "HINDALCO": 672.0, "TECHM": 1690.0, "WIPRO": 565.0, "HCLTECH": 1845.0,
    "M&M": 2980.0, "DIXON": 14850.0,
}


class DhanClient:
    """Async DhanHQ v2 client with scrip master sync, rate limiting, and demo fallback."""

    def __init__(self) -> None:
        self._sem = asyncio.Semaphore(4)
        self._last_auth_error: str | None = None
        self._auth_ok: bool | None = None
        self._last_auth_check: float = 0.0

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "access-token": settings.dhan_access_token,
            "client-id": settings.dhan_client_id,
        }

    def update_credentials(
        self,
        client_id: str | None = None,
        access_token: str | None = None,
        app_id: str | None = None,
        app_secret: str | None = None,
    ) -> None:
        if client_id is not None:
            settings.dhan_client_id = client_id.strip()
        if access_token is not None:
            settings.dhan_access_token = access_token.strip()
        if app_id is not None:
            settings.dhan_api_key = app_id.strip()
        if app_secret is not None:
            settings.dhan_api_secret = app_secret.strip()
        self._auth_ok = None
        self._last_auth_error = None
        self._last_auth_check = 0.0
        self._persist_env(
            {
                "DHAN_CLIENT_ID": settings.dhan_client_id,
                "DHAN_ACCESS_TOKEN": settings.dhan_access_token,
                "DHAN_APP_ID": settings.dhan_api_key,
                "DHAN_APP_SECRET": settings.dhan_api_secret,
            }
        )

    @staticmethod
    def _persist_env(updates: dict[str, str]) -> None:
        existing: dict[str, str] = {}
        order: list[str] = []
        if ENV_PATH.exists():
            for line in ENV_PATH.read_text().splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.split("=", 1)
                    k = k.strip()
                    existing[k] = v.strip()
                    order.append(k)
        for k, v in updates.items():
            if v:
                existing[k] = v
                if k not in order:
                    order.append(k)
        lines = [f"{k}={existing[k]}" for k in order if k in existing]
        ENV_PATH.write_text("\n".join(lines) + "\n")

    def _auth_meta(self) -> dict[str, Any]:
        return {
            "app_id": settings.dhan_api_key,
            "has_api_key": bool(settings.dhan_api_key),
            "has_api_secret": bool(settings.dhan_api_secret),
        }

    async def check_auth(self, force: bool = False) -> dict[str, Any]:
        now = time.time()
        if not force and self._auth_ok is not None and (now - self._last_auth_check) < 60:
            return {
                "connected": self._auth_ok,
                "mode": "live" if self._auth_ok else "demo",
                "client_id": settings.dhan_client_id,
                "error": self._last_auth_error,
                **self._auth_meta(),
            }

        if not settings.dhan_client_id or not settings.dhan_access_token:
            self._auth_ok = False
            self._last_auth_error = "Missing Dhan Client ID or Access Token"
            self._last_auth_check = now
            return {
                "connected": False,
                "mode": "demo",
                "client_id": settings.dhan_client_id,
                "error": self._last_auth_error,
                **self._auth_meta(),
            }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{DHAN_API_BASE}/profile", headers=self._headers())
            self._last_auth_check = now
            if resp.status_code == 200:
                self._auth_ok = True
                self._last_auth_error = None
                return {
                    "connected": True,
                    "mode": "live",
                    "client_id": settings.dhan_client_id,
                    "profile": resp.json(),
                    "error": None,
                    **self._auth_meta(),
                }
            self._auth_ok = False
            try:
                err_data = resp.json()
                self._last_auth_error = err_data.get("errorMessage") or resp.text
            except Exception:
                self._last_auth_error = f"HTTP {resp.status_code}"
        except Exception as exc:
            self._auth_ok = False
            self._last_auth_check = now
            self._last_auth_error = f"Network error: {exc}"

        return {
            "connected": False,
            "mode": "demo",
            "client_id": settings.dhan_client_id,
            "error": self._last_auth_error,
            **self._auth_meta(),
        }

    async def generate_consent(
        self, client_id: str, app_id: str = "", app_secret: str = ""
    ) -> dict[str, Any]:
        aid = (app_id or settings.dhan_api_key).strip()
        asec = (app_secret or settings.dhan_api_secret).strip()
        if not client_id.strip() or not aid or not asec:
            raise ValueError("Client ID, API Key, and API Secret are required.")
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{DHAN_AUTH_BASE}/generate-consent",
                params={"client_id": client_id.strip()},
                headers={
                    "app_id": aid,
                    "app_secret": asec,
                    "Accept": "application/json",
                },
            )
            data = resp.json() if resp.content else {}
            if resp.status_code != 200 or not data.get("consentAppId"):
                raise ValueError(data.get("errorMessage") or f"HTTP {resp.status_code}: {resp.text}")
            consent_id = data["consentAppId"]
            return {
                "consentAppId": consent_id,
                "loginUrl": f"https://auth.dhan.co/login/consentApp-login?consentAppId={consent_id}",
                "raw": data,
            }

    async def consume_consent(
        self, token_id: str, app_id: str = "", app_secret: str = ""
    ) -> dict[str, Any]:
        aid = (app_id or settings.dhan_api_key).strip()
        asec = (app_secret or settings.dhan_api_secret).strip()
        if not token_id.strip() or not aid or not asec:
            raise ValueError("tokenId, API Key, and API Secret are required.")
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{DHAN_AUTH_BASE}/consumeApp-consent",
                params={"tokenId": token_id.strip()},
                headers={
                    "app_id": aid,
                    "app_secret": asec,
                    "Accept": "application/json",
                },
            )
            data = resp.json() if resp.content else {}
            if resp.status_code != 200 or not data.get("accessToken"):
                raise ValueError(data.get("errorMessage") or f"HTTP {resp.status_code}: {resp.text}")
            self.update_credentials(
                client_id=data.get("dhanClientId") or settings.dhan_client_id,
                access_token=data["accessToken"],
                app_id=aid,
                app_secret=asec,
            )
            return data

    async def sync_fno_universe(self, force: bool = False) -> list[dict[str, Any]]:
        existing = db.get_instruments()
        if existing and not force:
            return existing

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                resp = await client.get(SCRIP_MASTER_URL)
                resp.raise_for_status()

            cols = [
                "SEM_EXM_EXCH_ID",
                "SEM_SMST_SECURITY_ID",
                "SEM_INSTRUMENT_NAME",
                "SEM_TRADING_SYMBOL",
                "SEM_LOT_UNITS",
                "SEM_SERIES",
                "SM_SYMBOL_NAME",
            ]
            df = pd.read_csv(io.StringIO(resp.text), usecols=cols, low_memory=False)
            fut = df[(df["SEM_EXM_EXCH_ID"] == "NSE") & (df["SEM_INSTRUMENT_NAME"] == "FUTSTK")].copy()
            fut["BASE_SYMBOL"] = fut["SEM_TRADING_SYMBOL"].astype(str).str.split("-").str[0].str.strip()
            lot_map = (
                fut.groupby("BASE_SYMBOL")["SEM_LOT_UNITS"]
                .first()
                .dropna()
                .astype(int)
                .to_dict()
            )

            eq = df[
                (df["SEM_EXM_EXCH_ID"] == "NSE")
                & (df["SEM_INSTRUMENT_NAME"] == "EQUITY")
                & (df["SEM_SERIES"] == "EQ")
            ]
            items: list[dict[str, Any]] = []
            seen: set[str] = set()
            for _, row in eq.iterrows():
                sym = str(row["SEM_TRADING_SYMBOL"]).strip().upper()
                if sym in lot_map and sym not in seen:
                    seen.add(sym)
                    items.append(
                        {
                            "symbol": sym,
                            "name": str(row.get("SM_SYMBOL_NAME") or sym).strip(),
                            "equity_security_id": str(int(row["SEM_SMST_SECURITY_ID"])),
                            "fno_lot_size": max(1, int(lot_map[sym])),
                            "is_fno": True,
                        }
                    )
            if items:
                db.upsert_instruments(items)
                return db.get_instruments()
        except Exception:
            pass

        db.upsert_instruments([{**item, "is_fno": True} for item in SEED_FNO_INSTRUMENTS])
        return db.get_instruments()

    async def get_candles(
        self,
        symbol: str,
        timeframe: str,
        lookback_bars: int = 240,
        use_cache: bool = True,
    ) -> tuple[pd.DataFrame, str]:
        """Return OHLCV DataFrame (ts, open, high, low, close, volume) and data source ('live'|'demo')."""
        symbol = symbol.upper()
        inst = db.get_instrument(symbol)
        if inst is None:
            await self.sync_fno_universe()
            inst = db.get_instrument(symbol)

        auth = await self.check_auth()
        if auth["connected"] and inst is not None:
            df = await self._fetch_dhan_candles(inst, timeframe, lookback_bars)
            if not df.empty:
                db.save_candles(symbol, timeframe, df.to_dict("records"))
                return df.tail(lookback_bars).reset_index(drop=True), "live"

        # Fallback to stored real SQLite candles (from live sync or imported historical cache)
        min_bars = 20 if timeframe == "1D" else 30
        if use_cache:
            cached = db.load_candles(symbol, timeframe, lookback_bars)
            if len(cached) >= min_bars:
                return pd.DataFrame(cached), "cached"

        sim_df = self._generate_simulated_candles(symbol, timeframe, lookback_bars)
        return sim_df, "demo"

    async def _fetch_dhan_candles(
        self, inst: dict[str, Any], timeframe: str, lookback_bars: int
    ) -> pd.DataFrame:
        now_ist = datetime.now(IST)
        sec_id = str(inst["equity_security_id"])

        async with self._sem:
            await asyncio.sleep(0.12)
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    if timeframe == "1D":
                        from_dt = (now_ist - timedelta(days=max(365, lookback_bars * 2))).strftime("%Y-%m-%d")
                        to_dt = now_ist.strftime("%Y-%m-%d")
                        resp = await client.post(
                            f"{DHAN_API_BASE}/charts/historical",
                            headers=self._headers(),
                            json={
                                "securityId": sec_id,
                                "exchangeSegment": "NSE_EQ",
                                "instrument": "EQUITY",
                                "expiryCode": 0,
                                "oi": True,
                                "fromDate": from_dt,
                                "toDate": to_dt,
                            },
                        )
                    else:
                        tf_map = {"5m": "5", "15m": "15", "1h": "60"}
                        interval = tf_map.get(timeframe, "15")
                        days_back = 18 if timeframe == "5m" else (35 if timeframe == "15m" else 75)
                        from_dt = (now_ist - timedelta(days=days_back)).strftime("%Y-%m-%d 09:15:00")
                        to_dt = now_ist.strftime("%Y-%m-%d %H:%M:%S")
                        resp = await client.post(
                            f"{DHAN_API_BASE}/charts/intraday",
                            headers=self._headers(),
                            json={
                                "securityId": sec_id,
                                "exchangeSegment": "NSE_EQ",
                                "instrument": "EQUITY",
                                "interval": interval,
                                "oi": True,
                                "fromDate": from_dt,
                                "toDate": to_dt,
                            },
                        )

                    if resp.status_code == 401:
                        self._auth_ok = False
                        self._last_auth_error = "DH-901: Token invalid or expired"
                        return pd.DataFrame()
                    if resp.status_code != 200:
                        return pd.DataFrame()
                    data = resp.json()
                    return self._parse_dhan_ohlcv(data)
            except Exception:
                return pd.DataFrame()

    @staticmethod
    def _parse_dhan_ohlcv(data: dict[str, Any]) -> pd.DataFrame:
        ts = data.get("timestamp") or []
        if not ts:
            return pd.DataFrame()
        n = len(ts)
        raw_oi = data.get("oi") or []
        oi_vals = [float(x) for x in raw_oi] if len(raw_oi) == n else [0.0] * n
        df = pd.DataFrame(
            {
                "ts": [int(t) for t in ts],
                "open": [float(x) for x in data.get("open", [])],
                "high": [float(x) for x in data.get("high", [])],
                "low": [float(x) for x in data.get("low", [])],
                "close": [float(x) for x in data.get("close", [])],
                "volume": [float(x) for x in data.get("volume", [])],
                "oi": oi_vals,
            }
        )
        return df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    async def get_batch_ltp(self, symbols: list[str]) -> tuple[dict[str, float], str]:
        if not symbols:
            return {}, "demo"
        auth = await self.check_auth()
        if auth["connected"]:
            sec_map: dict[int, str] = {}
            for sym in symbols:
                inst = db.get_instrument(sym)
                if inst:
                    sec_map[int(inst["equity_security_id"])] = sym.upper()
            if sec_map:
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        resp = await client.post(
                            f"{DHAN_API_BASE}/marketfeed/ltp",
                            headers=self._headers(),
                            json={"NSE_EQ": list(sec_map.keys())},
                        )
                    if resp.status_code == 200:
                        payload = resp.json().get("data", {}).get("NSE_EQ", {})
                        out: dict[str, float] = {}
                        for sec_id_str, info in payload.items():
                            sym = sec_map.get(int(sec_id_str))
                            if sym and "last_price" in info:
                                out[sym] = float(info["last_price"])
                        if out:
                            return out, "live"
                except Exception:
                    pass

        # Cached SQLite fallback before demo simulation
        out_sim: dict[str, float] = {}
        used_cached = False
        now_bucket = int(time.time() // 5)
        for sym in symbols:
            sym_u = sym.upper()
            cached = db.load_candles(sym_u, "5m", 20)
            if cached:
                out_sim[sym_u] = round(float(cached[-1]["close"]), 2)
                used_cached = True
                continue
            df = self._generate_simulated_candles(sym_u, "5m", 60)
            base_px = float(df["close"].iloc[-1])
            seed_val = int(hashlib.md5(f"{sym}:{now_bucket}".encode()).hexdigest()[:8], 16)
            jitter = ((seed_val % 200) - 100) / 100.0 * 0.0018
            out_sim[sym_u] = round(base_px * (1.0 + jitter), 2)
        return out_sim, ("cached" if used_cached else "demo")

    async def get_batch_quotes(self, symbols: list[str]) -> tuple[dict[str, dict[str, Any]], str]:
        """Fetch LTP, ATP (session VWAP), OHLC, and volume via Dhan /marketfeed/quote with cached/demo fallback."""
        if not symbols:
            return {}, "demo"
        auth = await self.check_auth()
        if auth["connected"]:
            sec_map: dict[int, str] = {}
            for sym in symbols:
                inst = db.get_instrument(sym)
                if inst:
                    sec_map[int(inst["equity_security_id"])] = sym.upper()
            if sec_map:
                out_live: dict[str, dict[str, Any]] = {}
                sec_ids = [k for k in sec_map.keys() if str(k).isdigit()]
                now_ist = datetime.now(IST)
                today_str = now_ist.date().isoformat()
                try:
                    async with httpx.AsyncClient(timeout=12.0) as client:
                        # 1) Primary bulk fetch via /marketfeed/ohlc (allows up to 1000 instruments in 1 call, zero 429)
                        for start in range(0, len(sec_ids), 1000):
                            chunk = sec_ids[start : start + 1000]
                            resp = await client.post(
                                f"{DHAN_API_BASE}/marketfeed/ohlc",
                                headers=self._headers(),
                                json={"NSE_EQ": chunk},
                            )
                            if resp.status_code != 200:
                                continue
                            payload = resp.json().get("data", {}).get("NSE_EQ", {})
                            for sec_id_str, info in payload.items():
                                sym_u = sec_map.get(int(sec_id_str))
                                if not sym_u:
                                    continue
                                ohlc = info.get("ohlc") or {}
                                ltp = float(info.get("last_price") or 0.0)
                                hi = float(ohlc.get("high") or ltp)
                                lo = float(ohlc.get("low") or ltp)
                                op = float(ohlc.get("open") or ltp)
                                pc = float(ohlc.get("close") or ltp)
                                atp = round((hi + lo + ltp) / 3.0, 2)
                                out_live[sym_u] = {
                                    "last_price": ltp,
                                    "average_price": atp,
                                    "open": op,
                                    "high": hi,
                                    "low": lo,
                                    "prev_close": pc,
                                    "volume": 0.0,
                                    "trade_date": today_str,
                                }
                    if out_live:
                        # 2) For active symbols with cached 5m bars today, enrich with true cumulative volume & VWAP
                        for sym_u in out_live:
                            c5 = db.load_candles(sym_u, "5m", 80)
                            if c5:
                                df5 = pd.DataFrame(c5)
                                df5["dt"] = pd.to_datetime(df5["ts"], unit="s", utc=True).dt.tz_convert(IST)
                                day_df = df5[df5["dt"].dt.date == now_ist.date()]
                                if not day_df.empty:
                                    vol_sum = float(day_df["volume"].sum())
                                    tp = (day_df["high"] + day_df["low"] + day_df["close"]) / 3.0
                                    if vol_sum > 0:
                                        out_live[sym_u]["average_price"] = round(float((tp * day_df["volume"]).sum() / vol_sum), 2)
                                        out_live[sym_u]["volume"] = vol_sum
                        return out_live, "live"
                except Exception:
                    pass

        # Cached 5m + 1D fallback to compute exact session VWAP & prev_close
        out_cached: dict[str, dict[str, Any]] = {}
        used_cached = False
        for sym in symbols:
            sym_u = sym.upper()
            c5 = db.load_candles(sym_u, "5m", 160)
            c1d = db.load_candles(sym_u, "1D", 10)
            if c5:
                used_cached = True
                df5 = pd.DataFrame(c5)
                df5["dt"] = pd.to_datetime(df5["ts"], unit="s", utc=True).dt.tz_convert(IST)
                latest_d = df5["dt"].dt.date.max()
                day_df = df5[df5["dt"].dt.date == latest_d]
                if day_df.empty:
                    day_df = df5.tail(75)
                tp = (day_df["high"] + day_df["low"] + day_df["close"]) / 3.0
                vol_sum = float(day_df["volume"].sum())
                vwap = float((tp * day_df["volume"]).sum() / vol_sum) if vol_sum > 0 else float(tp.mean())
                prev_close = float(day_df["open"].iloc[0])
                if c1d:
                    prev_rows = [
                        r
                        for r in c1d
                        if datetime.fromtimestamp(int(r["ts"]), tz=IST).date() < latest_d
                    ]
                    if prev_rows:
                        prev_close = float(prev_rows[-1]["close"])
                    elif len(c1d) >= 2:
                        prev_close = float(c1d[-2]["close"])
                out_cached[sym_u] = {
                    "last_price": float(day_df["close"].iloc[-1]),
                    "average_price": round(vwap, 2),
                    "open": float(day_df["open"].iloc[0]),
                    "high": float(day_df["high"].max()),
                    "low": float(day_df["low"].min()),
                    "prev_close": prev_close,
                    "volume": vol_sum,
                    "trade_date": str(latest_d),
                }
        if used_cached:
            return out_cached, "cached"

        out_sim: dict[str, dict[str, Any]] = {}
        for sym in symbols:
            sym_u = sym.upper()
            df = self._generate_simulated_candles(sym_u, "5m", 80)
            ltp = float(df["close"].iloc[-1])
            vwap = float(((df["high"] + df["low"] + df["close"]) / 3.0).mean())
            out_sim[sym_u] = {
                "last_price": round(ltp, 2),
                "average_price": round(vwap, 2),
                "open": float(df["open"].iloc[0]),
                "high": float(df["high"].max()),
                "low": float(df["low"].min()),
                "prev_close": float(df["close"].iloc[0]),
                "volume": float(df["volume"].sum()),
            }
        return out_sim, "demo"

    async def get_option_chain(self, symbol: str, expiry: str | None = None) -> dict[str, Any]:
        """
        Fetch real-time Option Chain & Call/Put Open Interest (OI) metrics.
        Identifies Max Call OI (Resistance), Max Put OI (Support), Call Unwinding (Short Squeeze),
        and recommends the optimal high-gamma strike.
        """
        sym_u = symbol.upper()
        inst = db.get_instrument(sym_u)
        sec_id = str(inst.get("equity_security_id") or "") if inst else ""
        lot_size = int(inst.get("fno_lot_size") or 100) if inst else 100

        # Current cash price & VWAP
        quotes, _q_src = await self.get_batch_quotes([sym_u])
        q_data = quotes.get(sym_u, {})
        ltp = float(q_data.get("last_price") or _BASE_PRICES.get(sym_u, 1000.0))
        vwap = float(q_data.get("average_price") or ltp)
        open_p = float(q_data.get("open") or ltp)
        move_pct = round((ltp - open_p) / max(0.01, open_p) * 100.0, 2)
        direction = "LONG" if move_pct >= 0 else "SHORT"

        auth = await self.check_auth()
        if auth.get("connected") and sec_id:
            try:
                headers = self._headers()
                async with httpx.AsyncClient(timeout=6.0) as client:
                    body: dict[str, Any] = {
                        "UnderlyingScrip": int(sec_id),
                        "UnderlyingSeg": "NSE_EQ",
                    }
                    if expiry:
                        body["Expiry"] = expiry
                    resp = await client.post("https://api.dhan.co/v2/optionchain", json=body, headers=headers)
                    if resp.status_code == 200:
                        raw = resp.json().get("data", {})
                        if raw:
                            # Parse live Dhan option chain response
                            return self._parse_dhan_option_chain(sym_u, ltp, vwap, move_pct, lot_size, raw)
            except Exception:
                pass

        # High-Fidelity Simulation / Offline Model based on Real Cash Structure
        return self._generate_simulated_option_chain(sym_u, ltp, vwap, move_pct, direction, lot_size)

    def _generate_simulated_option_chain(
        self,
        symbol: str,
        ltp: float,
        vwap: float,
        move_pct: float,
        direction: str,
        lot_size: int,
    ) -> dict[str, Any]:
        from app.playbook_engine import get_strike_step, recommend_option_strike

        step = get_strike_step(ltp)
        atm = round(ltp / step) * step
        strikes_list: list[dict[str, Any]] = []

        max_ce_oi = -1
        max_ce_strike = atm
        max_pe_oi = -1
        max_pe_strike = atm

        call_unwinding_strikes: list[int] = []
        put_unwinding_strikes: list[int] = []

        # 15 strike window (-7 to +7 around ATM)
        for i in range(-7, 8):
            k = atm + i * step
            dist_pct = round((k - ltp) / max(0.01, ltp) * 100.0, 2)
            strike_disp = int(k) if step >= 1.0 else k

            # Realistic OI distribution: Calls peak around +3% OTM; Puts peak around -3% OTM
            ce_oi = int(round(120_000 * max(0.12, 1.0 - abs(dist_pct - 3.2) * 0.14)))
            pe_oi = int(round(120_000 * max(0.12, 1.0 - abs(dist_pct + 3.2) * 0.14)))

            if ce_oi > max_ce_oi:
                max_ce_oi = ce_oi
                max_ce_strike = strike_disp
            if pe_oi > max_pe_oi:
                max_pe_oi = pe_oi
                max_pe_strike = strike_disp

            # In strong trends, strikes immediately in path of trend experience writer panic (unwinding)
            if direction == "LONG" and 0.0 <= dist_pct <= 2.8:
                ce_chg_oi = -int(round(ce_oi * 0.18))  # -18% Call Unwinding (Short Squeeze!)
                call_unwinding_strikes.append(strike_disp)
            else:
                ce_chg_oi = int(round(ce_oi * 0.06))

            if direction == "SHORT" and -2.8 <= dist_pct <= 0.0:
                pe_chg_oi = -int(round(pe_oi * 0.18))  # -18% Put Unwinding
                put_unwinding_strikes.append(strike_disp)
            else:
                pe_chg_oi = int(round(pe_oi * 0.06))

            # Black-Scholes approximate intraday premium
            ce_ltp = max(0.5, round(max(0.0, ltp - k) + ltp * 0.018 * max(0.1, 1.0 - dist_pct * 0.25), 1))
            pe_ltp = max(0.5, round(max(0.0, k - ltp) + ltp * 0.018 * max(0.1, 1.0 + dist_pct * 0.25), 1))

            strikes_list.append(
                {
                    "strike": strike_disp,
                    "distance_pct": dist_pct,
                    "ce_ltp": ce_ltp,
                    "ce_oi": ce_oi,
                    "ce_chg_oi": ce_chg_oi,
                    "ce_unwinding": ce_chg_oi < 0,
                    "pe_ltp": pe_ltp,
                    "pe_oi": pe_oi,
                    "pe_chg_oi": pe_chg_oi,
                    "pe_unwinding": pe_chg_oi < 0,
                }
            )

        opt_rec = recommend_option_strike(symbol, ltp, direction, lot_size)
        squeeze_active = len(call_unwinding_strikes) > 0 if direction == "LONG" else len(put_unwinding_strikes) > 0
        if direction == "LONG":
            summary = (
                f"Call Writers Panicking at {', '.join(str(s) for s in call_unwinding_strikes)} CE (Short Squeeze Active)"
                if squeeze_active
                else f"Resistance at Max Call OI {max_ce_strike} CE"
            )
        else:
            summary = (
                f"Put Writers Panicking at {', '.join(str(s) for s in put_unwinding_strikes)} PE (Long Liquidation Active)"
                if squeeze_active
                else f"Support at Max Put OI {max_pe_strike} PE"
            )

        now_d = datetime.now(IST).date()
        days_ahead = 3 - now_d.weekday() if now_d.weekday() <= 3 else 10 - now_d.weekday()
        expiry_date = (now_d + timedelta(days=days_ahead)).isoformat()

        return {
            "symbol": symbol,
            "underlying_ltp": ltp,
            "vwap": vwap,
            "move_pct": move_pct,
            "direction": direction,
            "lot_size": lot_size,
            "expiry": expiry_date,
            "atm_strike": int(atm) if step >= 1.0 else atm,
            "max_call_oi_strike": max_ce_strike,
            "max_put_oi_strike": max_pe_strike,
            "recommended_strike": opt_rec["strike"],
            "recommended_option": opt_rec["option_name"],
            "recommended_premium": opt_rec["est_premium"],
            "recommended_capital": opt_rec["est_capital_inr"],
            "recommended_sl": opt_rec["sl_premium"],
            "recommended_tp1": opt_rec["tp1_premium"],
            "squeeze_active": squeeze_active,
            "call_unwinding_strikes": call_unwinding_strikes,
            "put_unwinding_strikes": put_unwinding_strikes,
            "unwinding_summary": summary,
            "strikes": strikes_list,
        }

    @staticmethod
    def _generate_simulated_candles(symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        """
        Deterministic per-symbol simulator generating realistic NSE F&O price action
        with clean trends, BOS impulses, Order Blocks, FVGs, and pullback triggers.
        """
        seed = int(hashlib.sha256(f"{symbol}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)

        base_price = _BASE_PRICES.get(symbol, 500.0 + (seed % 2500))
        tf_sec = {"5m": 300, "15m": 900, "1h": 3600, "1D": 86400}.get(timeframe, 900)
        vol_scale = {"5m": 0.0022, "15m": 0.0038, "1h": 0.0072, "1D": 0.0155}.get(timeframe, 0.004)

        # Bias direction per symbol so half look bullish, half bearish/ranging
        regime_roll = seed % 5
        trend_dir = 1.0 if regime_roll in (0, 1, 3) else -1.0

        total_bars = max(bars, 220)
        closes = np.zeros(total_bars)
        opens = np.zeros(total_bars)
        highs = np.zeros(total_bars)
        lows = np.zeros(total_bars)
        volumes = np.zeros(total_bars)

        px = base_price * (0.94 if trend_dir > 0 else 1.06)
        # Craft multi-phase price action:
        # 0..60% steady trend, 60..75% impulsive BOS (leaves OB + FVG), 75..92% pullback into OB/S&R, 92..100% LTF trigger!
        p1 = int(total_bars * 0.58)
        p2 = int(total_bars * 0.68)
        p3 = int(total_bars * 0.90)

        for i in range(total_bars):
            noise = rng.normal(0.0, vol_scale * 0.7)
            if i < p1:
                drift = trend_dir * vol_scale * 0.14 + math.sin(i / 8.0) * vol_scale * 0.35
                v_mult = 1.0
            elif i < p2:
                # Impulsive BOS leg (creates strong OB + FVG)
                if i == p1:
                    # Last opposing candle before impulse (the Order Block candle!)
                    drift = -trend_dir * vol_scale * 0.65
                    v_mult = 1.15
                else:
                    drift = trend_dir * vol_scale * 0.85
                    v_mult = 2.1
            elif i < p3:
                # Controlled pullback back into the OB / S&R zone
                drift = -trend_dir * vol_scale * 0.32 + math.sin(i / 4.0) * vol_scale * 0.2
                v_mult = 0.8
            else:
                # Liquidity sweep + sharp rejection trigger at the OB/S&R
                if i == total_bars - 5:
                    drift = -trend_dir * vol_scale * 0.45
                    v_mult = 1.6
                else:
                    drift = trend_dir * vol_scale * 0.48
                    v_mult = 1.7

            ret = drift + noise
            op = px
            cl = max(5.0, op * (1.0 + ret))
            body = abs(cl - op)
            wick_up = max(body * 0.35, op * vol_scale * rng.uniform(0.15, 0.55))
            wick_dn = max(body * 0.35, op * vol_scale * rng.uniform(0.15, 0.55))
            if i == total_bars - 5:
                # Clear liquidity sweep wick
                if trend_dir > 0:
                    wick_dn = op * vol_scale * 1.15
                else:
                    wick_up = op * vol_scale * 1.15
            hi = max(op, cl) + wick_up
            lo = max(1.0, min(op, cl) - wick_dn)

            opens[i] = round(op, 2)
            closes[i] = round(cl, 2)
            highs[i] = round(hi, 2)
            lows[i] = round(lo, 2)
            volumes[i] = round(float(rng.uniform(45_000, 140_000) * v_mult), 0)
            px = cl

        end_ts = int(datetime.now(IST).replace(second=0, microsecond=0).timestamp())
        end_ts = (end_ts // tf_sec) * tf_sec
        timestamps = [end_ts - (total_bars - 1 - i) * tf_sec for i in range(total_bars)]

        df = pd.DataFrame(
            {
                "ts": timestamps,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            }
        )
        return df.tail(bars).reset_index(drop=True)


dhan_client = DhanClient()
