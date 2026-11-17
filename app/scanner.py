from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from app.config import settings
from app.database import db
from app.dhan_client import dhan_client
from app.idea_generator import generate_symbol_ideas
from app.paper_trader import paper_trader
from app.playbook_engine import (
    build_daily_watchlist,
    compute_market_breadth,
    evaluate_live_momentum_board,
    evaluate_playbook_setups,
    is_valid_fno_symbol,
    merge_and_rank_setups,
    seed_sqlite_from_cache,
)
from app.structure_engine import analyze_structure


class MarketScanner:
    """Scans NSE F&O stocks for Playbook v2 + multi-timeframe OB + Trend + S/R ideas."""

    def __init__(self) -> None:
        self._scanning = False
        self._last_scan_at: str | None = None
        self._last_source: str = "demo"

    @property
    def status(self) -> dict[str, Any]:
        return {
            "scanning": self._scanning,
            "last_scan_at": self._last_scan_at,
            "data_source": self._last_source,
        }

    async def refresh_breadth(self, universe_syms: list[str] | None = None) -> dict[str, Any]:
        if universe_syms is None:
            instruments = await dhan_client.sync_fno_universe()
            universe_syms = [i["symbol"] for i in instruments if is_valid_fno_symbol(i["symbol"])]
        quotes_map, q_src = await dhan_client.get_batch_quotes(universe_syms)
        return compute_market_breadth(quotes_map, source=q_src)

    async def refresh_live_momentum(self) -> dict[str, Any]:
        instruments = await dhan_client.sync_fno_universe()
        universe_syms = [i["symbol"] for i in instruments if is_valid_fno_symbol(i["symbol"])]
        with db.connect() as conn:
            cached_syms = [
                r["symbol"]
                for r in conn.execute(
                    "SELECT DISTINCT symbol FROM candles WHERE timeframe = '5m'"
                ).fetchall()
                if is_valid_fno_symbol(r["symbol"])
            ]
        all_syms = sorted(set(universe_syms) | set(cached_syms))
        quotes_map, q_src = await dhan_client.get_batch_quotes(all_syms)
        if q_src == "live":
            auth = await dhan_client.check_auth()
            if auth.get("connected"):
                movers: list[tuple[float, str]] = []
                for s, q in quotes_map.items():
                    ltp = float(q.get("last_price") or 0.0)
                    op = float(q.get("open") or ltp)
                    if op > 0:
                        movers.append((abs(ltp - op) / op, s))
                movers.sort(reverse=True)
                top_to_sync = [s for mv, s in movers if mv >= 0.015][:45]
                sem = asyncio.Semaphore(8)

                async def _sync_c5(s: str) -> None:
                    async with sem:
                        try:
                            await dhan_client.get_candles(s, "5m", 80)
                        except Exception:
                            pass

                await asyncio.gather(*(_sync_c5(s) for s in top_to_sync))

        breadth = compute_market_breadth(quotes_map, source=q_src)
        wl_items = db.get_daily_watchlist()
        setups = db.get_setups()
        board = evaluate_live_momentum_board(
            symbols=all_syms,
            quotes_map=quotes_map,
            breadth=breadth,
            watchlist_items=wl_items,
            ranked_setups=setups,
        )
        board["breadth"] = breadth
        board["data_source"] = q_src
        return board

    async def build_watchlist(self, top_n: int = 40) -> dict[str, Any]:
        seed_sqlite_from_cache()
        instruments = await dhan_client.sync_fno_universe()
        universe_syms = [i["symbol"] for i in instruments if is_valid_fno_symbol(i["symbol"])]
        # Ensure instruments from cached pickle are also covered even if offline universe list is smaller
        with db.connect() as conn:
            cached_syms = [
                r["symbol"]
                for r in conn.execute(
                    "SELECT DISTINCT symbol FROM candles WHERE timeframe = '5m'"
                ).fetchall()
                if is_valid_fno_symbol(r["symbol"])
            ]
        all_syms = sorted(set(universe_syms) | set(cached_syms))
        return build_daily_watchlist(all_syms, top_n=top_n)

    async def scan_universe(self, symbols: list[str] | None = None) -> dict[str, Any]:
        if self._scanning:
            return {
                "status": "already_running",
                "setups": db.get_setups(),
                "breadth": db.get_market_breadth(),
            }

        self._scanning = True
        try:
            seed_sqlite_from_cache()
            instruments = await dhan_client.sync_fno_universe()
            valid_insts = [i for i in instruments if is_valid_fno_symbol(i["symbol"])]
            inst_map = {i["symbol"]: i for i in valid_insts}

            with db.connect() as conn:
                cached_syms = [
                    r["symbol"]
                    for r in conn.execute(
                        "SELECT DISTINCT symbol FROM candles WHERE timeframe = '5m'"
                    ).fetchall()
                    if is_valid_fno_symbol(r["symbol"])
                ]

            universe_syms = sorted(set(inst_map.keys()) | set(cached_syms))
            breadth = await self.refresh_breadth(universe_syms)
            watchlist_res = build_daily_watchlist(universe_syms, top_n=40)

            auth = await dhan_client.check_auth()
            if symbols:
                target_symbols = [s.upper() for s in symbols if is_valid_fno_symbol(s)]
            elif cached_syms and not auth.get("connected"):
                # Offline/sandbox cached mode: scan all 216 F&O stocks quickly from SQLite
                target_symbols = universe_syms
            else:
                wl_syms = [w["symbol"] for w in watchlist_res.get("items", [])]
                merged = list(dict.fromkeys(wl_syms + [s for s in settings.watchlist_symbols if s in inst_map]))
                target_symbols = merged[: max(settings.scan_universe_limit, 60)] or universe_syms[: settings.scan_universe_limit]

            account_cfg = db.get_account_state()
            smc_setups: list[dict[str, Any]] = []
            playbook_setups: list[dict[str, Any]] = []
            sources: set[str] = set()

            batch_size = 12 if (cached_syms and not auth.get("connected")) else 5
            for i in range(0, len(target_symbols), batch_size):
                batch = target_symbols[i : i + batch_size]
                results = await asyncio.gather(
                    *(self._scan_symbol(sym, inst_map.get(sym, {}), account_cfg, breadth) for sym in batch),
                    return_exceptions=True,
                )
                for res in results:
                    if isinstance(res, dict):
                        sources.add(res["data_source"])
                        smc_setups.extend(res.get("smc_ideas", []))
                        playbook_setups.extend(res.get("playbook_ideas", []))

            ranked_setups = merge_and_rank_setups(playbook_setups, smc_setups, breadth)
            db.replace_setups(ranked_setups)

            quotes_map, _q_src = await dhan_client.get_batch_quotes(universe_syms)
            mom_res = evaluate_live_momentum_board(
                symbols=universe_syms,
                quotes_map=quotes_map,
                breadth=breadth,
                watchlist_items=watchlist_res.get("items", []),
                ranked_setups=ranked_setups,
            )

            auto_opened = paper_trader.auto_take_setups(ranked_setups)
            self._last_scan_at = datetime.now(timezone.utc).isoformat()
            if "live" in sources:
                self._last_source = "live"
            elif "cached" in sources:
                self._last_source = "cached"
            else:
                self._last_source = "demo"

            return {
                "status": "completed",
                "scanned_count": len(target_symbols),
                "setups_count": len(ranked_setups),
                "auto_opened_count": len(auto_opened),
                "data_source": self._last_source,
                "last_scan_at": self._last_scan_at,
                "breadth": breadth,
                "watchlist_target_date": watchlist_res.get("target_date"),
                "watchlist_count": watchlist_res.get("count", 0),
                "setups": ranked_setups,
            }
        finally:
            self._scanning = False

    async def _scan_symbol(
        self,
        symbol: str,
        inst: dict[str, Any],
        account_cfg: dict[str, Any],
        breadth: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        lot_size = int(inst.get("fno_lot_size", 1))
        df_1d, src_1d = await dhan_client.get_candles(symbol, "1D", 160)
        df_1h, src_1h = await dhan_client.get_candles(symbol, "1h", 160)
        df_15m, src_15m = await dhan_client.get_candles(symbol, "15m", 180)
        df_5m, src_5m = await dhan_client.get_candles(symbol, "5m", 240)

        srcs = (src_1d, src_1h, src_15m, src_5m)
        if "live" in srcs:
            source = "live"
        elif "cached" in srcs:
            source = "cached"
        else:
            source = "demo"

        smc_res = generate_symbol_ideas(
            symbol=symbol,
            lot_size=lot_size,
            df_1d=df_1d,
            df_1h=df_1h,
            df_15m=df_15m,
            df_5m=df_5m,
            data_source=source,
            account_cfg=account_cfg,
        )
        pb_ideas = evaluate_playbook_setups(
            symbol=symbol,
            lot_size=lot_size,
            df_1d=df_1d,
            df_15m=df_15m,
            df_5m=df_5m,
            breadth=breadth or db.get_market_breadth() or {"abv_vwap_pct": 50.0},
            data_source=source,
            account_cfg=account_cfg,
        )
        smc_res["smc_ideas"] = smc_res.get("ideas", [])
        smc_res["playbook_ideas"] = pb_ideas
        return smc_res

    async def get_chart_bundle(self, symbol: str, timeframe: str = "15m") -> dict[str, Any]:
        symbol = symbol.upper()
        if timeframe not in ("5m", "15m", "1h", "1D"):
            timeframe = "15m"

        instruments = await dhan_client.sync_fno_universe()
        inst = next((i for i in instruments if i["symbol"] == symbol), {"fno_lot_size": 1, "name": symbol})

        df, source = await dhan_client.get_candles(symbol, timeframe, 220)
        swing_win = 4 if timeframe == "1D" else 3
        struct = analyze_structure(df, swing_window=swing_win)

        setups = [s for s in db.get_setups() if s["symbol"] == symbol]
        if not setups:
            account_cfg = db.get_account_state()
            analyzed = await self._scan_symbol(symbol, inst, account_cfg, db.get_market_breadth())
            setups = merge_and_rank_setups(analyzed["playbook_ideas"], analyzed["smc_ideas"], db.get_market_breadth() or {})

        candles = [
            {
                "time": int(row["ts"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
            for _, row in df.iterrows()
        ]

        return {
            "symbol": symbol,
            "name": inst.get("name", symbol),
            "lot_size": int(inst.get("fno_lot_size", 1)),
            "timeframe": timeframe,
            "data_source": source,
            "candles": candles,
            "structure": struct,
            "setups": setups,
        }

    async def refresh_paper_mtm(self) -> dict[str, Any]:
        open_positions = paper_trader.list_positions(status="OPEN")
        symbols = sorted({p["symbol"] for p in open_positions})
        if not symbols:
            return {"events": [], "portfolio": paper_trader.get_portfolio_summary()}
        ltp_map, source = await dhan_client.get_batch_ltp(symbols)
        events = paper_trader.process_price_updates(ltp_map)
        return {
            "data_source": source,
            "ltp_map": ltp_map,
            "events": events,
            "portfolio": paper_trader.get_portfolio_summary(),
            "positions": paper_trader.list_positions(status="OPEN"),
        }


scanner = MarketScanner()
