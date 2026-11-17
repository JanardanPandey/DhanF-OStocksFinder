from __future__ import annotations

import asyncio
import csv
import io
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import settings
from app.database import db
from app.dhan_client import dhan_client
from app.dhan_websocket import dhan_feed
from app.paper_trader import paper_trader
from app.playbook_engine import is_valid_fno_symbol
from app.scanner import scanner

STATIC_DIR = Path(__file__).resolve().parent / "static"


class ScanRequest(BaseModel):
    symbols: list[str] | None = None


class OpenPaperTradeRequest(BaseModel):
    setup_id: str | None = None
    symbol: str | None = None
    style: str = "INTRADAY"
    direction: str = "LONG"
    grade: str = "A"
    entry_price: float | None = None
    stop_loss: float | None = None
    tp1: float | None = None
    tp2: float | None = None
    qty: int | None = None
    lot_size: int = 1
    notes: str = ""


class UpdateLevelsRequest(BaseModel):
    stop_loss: float | None = None
    tp1: float | None = None
    tp2: float | None = None


class RiskSettingsRequest(BaseModel):
    initial_capital: float | None = None
    risk_per_trade_inr: float | None = None
    max_sl_pct_intraday: float | None = None
    max_sl_pct_swing: float | None = None
    min_rr_ratio: float | None = None
    auto_paper_trade: bool | None = None


class UpdateTokenRequest(BaseModel):
    client_id: str
    access_token: str


class ConsentRequest(BaseModel):
    client_id: str
    app_id: str = ""
    app_secret: str = ""


class ConsumeConsentRequest(BaseModel):
    token_id: str
    app_id: str = ""
    app_secret: str = ""


async def _mtm_loop() -> None:
    while True:
        await asyncio.sleep(8)
        try:
            await scanner.refresh_paper_mtm()
        except Exception:
            pass


async def _live_alert_monitor_loop() -> None:
    """Continuously evaluates live momentum candidates and records new alerts."""
    while True:
        await asyncio.sleep(25)
        try:
            await scanner.refresh_live_momentum()
        except Exception:
            pass


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await dhan_client.sync_fno_universe(force=False)
    # Refresh watchlist & breadth synchronously (<1s on cache/batch quote), then run full rank scan in background
    # so uvicorn binds port 8000 immediately even when fetching live Dhan candles.
    await scanner.build_watchlist(top_n=40)
    await scanner.refresh_live_momentum()
    await dhan_feed.start()
    startup_scan = asyncio.create_task(scanner.scan_universe())
    mtm_task = asyncio.create_task(_mtm_loop())
    alert_task = asyncio.create_task(_live_alert_monitor_loop())
    try:
        yield
    finally:
        startup_scan.cancel()
        mtm_task.cancel()
        alert_task.cancel()
        await dhan_feed.stop()


app = FastAPI(title="Dhan SMC & Price Action Setup Desk", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/health")
async def health() -> dict[str, Any]:
    auth = await dhan_client.check_auth()
    instruments = [i for i in db.get_instruments() if is_valid_fno_symbol(i["symbol"])]
    return {
        "auth": auth,
        "scanner": scanner.status,
        "universe_count": len(instruments),
        "setups_count": len(db.get_setups()),
        "breadth": db.get_market_breadth(),
        "watchlist_dates": db.list_watchlist_dates(),
    }


@app.get("/api/universe")
async def get_universe(q: str = "") -> dict[str, Any]:
    items = [i for i in await dhan_client.sync_fno_universe() if is_valid_fno_symbol(i["symbol"])]
    if q.strip():
        needle = q.strip().upper()
        items = [i for i in items if needle in i["symbol"] or needle in i["name"].upper()]
    return {"items": items, "total": len(items)}


@app.post("/api/universe/sync")
async def sync_universe() -> dict[str, Any]:
    items = [i for i in await dhan_client.sync_fno_universe(force=True) if is_valid_fno_symbol(i["symbol"])]
    return {"total": len(items)}


@app.post("/api/scan")
async def run_scan(req: ScanRequest | None = None) -> dict[str, Any]:
    symbols = req.symbols if req else None
    return await scanner.scan_universe(symbols=symbols)


@app.get("/api/setups")
async def list_setups(
    style: str = Query(default="ALL"),
    direction: str = Query(default="ALL"),
    grade: str = Query(default="ALL"),
    playbook: str = Query(default="ALL"),
) -> dict[str, Any]:
    setups = db.get_setups()
    if style != "ALL":
        setups = [s for s in setups if s["style"] == style.upper()]
    if direction != "ALL":
        setups = [s for s in setups if s["direction"] == direction.upper()]
    if grade != "ALL":
        setups = [s for s in setups if s["grade"] == grade.upper()]
    if playbook != "ALL":
        pb = playbook.upper()
        if pb == "MONSTER":
            setups = [s for s in setups if s.get("monster_runner")]
        else:
            setups = [s for s in setups if str(s.get("playbook_tag", "")).upper() == pb]
    return {
        "items": setups,
        "total": len(setups),
        "scanner": scanner.status,
        "breadth": db.get_market_breadth(),
    }


@app.get("/api/breadth")
async def get_breadth() -> dict[str, Any]:
    b = db.get_market_breadth()
    if not b:
        b = await scanner.refresh_breadth()
    return b


@app.post("/api/breadth/refresh")
async def refresh_breadth() -> dict[str, Any]:
    return await scanner.refresh_breadth()


@app.get("/api/watchlist")
async def get_watchlist(
    target_date: str | None = Query(default=None),
    bias: str = Query(default="ALL"),
) -> dict[str, Any]:
    dates = db.list_watchlist_dates()
    if not dates:
        await scanner.build_watchlist()
        dates = db.list_watchlist_dates()
    resolved_date = target_date or (dates[0]["target_date"] if dates else None)
    items = db.get_daily_watchlist(resolved_date) if resolved_date else []
    if bias != "ALL":
        b = bias.upper()
        if b == "MONSTER":
            items = [i for i in items if "MONSTER" in str(i.get("tags", "")).upper()]
        else:
            items = [i for i in items if str(i.get("bias_hint", "")).upper() == b]
    copy_text = ", ".join(f"NSE:{i['symbol']}" for i in items)
    return {
        "target_date": resolved_date,
        "t1_date": items[0].get("t1_date") if items else None,
        "items": items,
        "total": len(items),
        "dates": dates,
        "copy_text": copy_text,
        "breadth": db.get_market_breadth(),
    }


@app.post("/api/watchlist/build")
async def build_watchlist(top_n: int = Query(default=40, ge=5, le=100)) -> dict[str, Any]:
    res = await scanner.build_watchlist(top_n=top_n)
    dates = db.list_watchlist_dates()
    items = res.get("items", [])
    return {
        "target_date": res.get("target_date"),
        "t1_date": res.get("t1_date"),
        "items": items,
        "total": len(items),
        "dates": dates,
        "copy_text": ", ".join(f"NSE:{i['symbol']}" for i in items),
    }


@app.get("/api/watchlist/live")
async def get_live_momentum_watchlist(bias: str = Query(default="OPTION_READY")) -> dict[str, Any]:
    items = db.get_live_momentum()
    if not items or "early_bar_eff" not in items[0]:
        res = await scanner.refresh_live_momentum()
        items = res.get("items", [])
    if bias != "ALL":
        b = bias.upper()
        if b == "OPTION_READY":
            items = [i for i in items if i.get("option_safe")]
        elif b == "WL":
            items = [i for i in items if i.get("in_watchlist")]
        elif b == "MONSTER":
            items = [i for i in items if i.get("monster_runner")]
        else:
            items = [i for i in items if str(i.get("direction", "")).upper() == b]
    breadth = db.get_market_breadth()
    return {
        "trade_date": breadth.get("trade_date") if breadth else None,
        "items": items,
        "total": len(items),
        "breadth": breadth,
    }


@app.post("/api/watchlist/live/refresh")
async def refresh_live_momentum_watchlist() -> dict[str, Any]:
    return await scanner.refresh_live_momentum()


@app.get("/api/alerts/live")
async def get_live_alerts_feed(
    trade_date: str | None = Query(default=None),
    event_type: str | None = Query(default=None),
    limit: int = Query(default=150, ge=1, le=500),
) -> dict[str, Any]:
    alerts = db.get_live_alerts(trade_date=trade_date, limit=limit)
    if event_type and event_type != "ALL":
        ev_u = event_type.upper()
        if ev_u == "BREAKOUTS":
            alerts = [a for a in alerts if a.get("event_type") in ("OB_BREAKOUT", "MOMENTUM_EXPANSION")]
        elif ev_u == "EXITS":
            alerts = [a for a in alerts if a.get("event_type") in ("TARGET_R1_HIT", "INVALIDATION_EXIT")]
        elif ev_u == "SMC":
            alerts = [
                a
                for a in alerts
                if a.get("event_type") in ("OB_BREAKOUT", "OB_RETEST_BOUNCE", "NEW_HIGHER_OB", "NEW_LOWER_OB")
            ]
        elif ev_u == "SQUEEZE":
            alerts = [a for a in alerts if a.get("event_type") == "INSTITUTIONAL_SQUEEZE"]
        else:
            alerts = [a for a in alerts if str(a.get("event_type", "")).upper() == ev_u]
    return {
        "total": len(alerts),
        "trade_date": trade_date,
        "alerts": alerts,
    }


@app.post("/api/alerts/clear")
async def clear_live_alerts(trade_date: str | None = Query(default=None)) -> dict[str, Any]:
    db.clear_live_alerts(trade_date=trade_date)
    return {"ok": True, "message": "Live alerts cleared"}


@app.get("/api/option-chain/{symbol}")
async def get_symbol_option_chain(symbol: str) -> dict[str, Any]:
    sym_u = symbol.upper()
    if not is_valid_fno_symbol(sym_u):
        raise HTTPException(status_code=400, detail=f"Invalid F&O symbol: {symbol}")
    return await dhan_client.get_option_chain(sym_u)


@app.get("/api/websocket/status")
async def get_websocket_feed_status() -> dict[str, Any]:
    return dhan_feed.get_status()


@app.get("/api/watchlist/csv")
async def export_watchlist_csv(target_date: str | None = Query(default=None)) -> Response:
    dates = db.list_watchlist_dates()
    resolved_date = target_date or (dates[0]["target_date"] if dates else None)
    items = db.get_daily_watchlist(resolved_date) if resolved_date else []
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "Rank",
            "Symbol",
            "ExchangeSymbol",
            "TargetDate",
            "T1Date",
            "Bias",
            "Score",
            "TriggerTime",
            "Tags",
            "PM_RelVol",
            "T1_CLV",
            "ATR_Pct",
            "AntiChasing",
            "Reasons",
        ]
    )
    for row in items:
        writer.writerow(
            [
                row.get("rank", ""),
                row.get("symbol", ""),
                f"NSE:{row.get('symbol', '')}",
                row.get("target_date", resolved_date or ""),
                row.get("t1_date", ""),
                row.get("bias_hint", ""),
                row.get("score", ""),
                row.get("trigger_time", ""),
                row.get("tags", ""),
                row.get("pm_rv", ""),
                row.get("t1_clv", ""),
                row.get("atr_pct", ""),
                "YES" if row.get("anti_chasing") else "NO",
                " | ".join(row.get("reasons", [])),
            ]
        )
    filename = f"dhan_watchlist_{resolved_date or 'latest'}.csv"
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/chart/{symbol}")
async def get_chart(symbol: str, timeframe: str = Query(default="15m")) -> dict[str, Any]:
    return await scanner.get_chart_bundle(symbol=symbol, timeframe=timeframe)


@app.get("/api/paper/portfolio")
async def get_paper_portfolio() -> dict[str, Any]:
    return {
        "portfolio": paper_trader.get_portfolio_summary(),
        "positions": paper_trader.list_positions(status="OPEN"),
        "trades": paper_trader.list_trades(limit=100),
    }


@app.post("/api/paper/open")
async def open_paper_trade(req: OpenPaperTradeRequest) -> dict[str, Any]:
    try:
        if req.setup_id:
            setup = db.get_setup(req.setup_id)
            if not setup:
                raise ValueError("Setup not found; re-run scan or enter custom values.")
            pos = paper_trader.open_trade(
                symbol=setup["symbol"],
                style=setup["style"],
                direction=setup["direction"],
                grade=setup["grade"],
                entry_price=req.entry_price or setup["entry_price"],
                stop_loss=req.stop_loss or setup["stop_loss"],
                tp1=req.tp1 or setup["tp1"],
                tp2=req.tp2 or setup["tp2"],
                qty=req.qty or setup["suggested_qty"],
                lot_size=setup["lot_size"],
                setup_id=setup["id"],
                notes=req.notes or ", ".join(setup["reasons"][:3]),
            )
        else:
            if not req.symbol or req.entry_price is None or req.stop_loss is None:
                raise ValueError("symbol, entry_price, and stop_loss are required.")
            risk = abs(req.entry_price - req.stop_loss)
            mult = 1.0 if req.direction.upper() == "LONG" else -1.0
            tp1 = req.tp1 or round(req.entry_price + mult * risk * 1.5, 2)
            tp2 = req.tp2 or round(req.entry_price + mult * risk * 2.5, 2)
            inst = db.get_instrument(req.symbol.upper())
            pos = paper_trader.open_trade(
                symbol=req.symbol,
                style=req.style,
                direction=req.direction,
                grade=req.grade,
                entry_price=req.entry_price,
                stop_loss=req.stop_loss,
                tp1=tp1,
                tp2=tp2,
                qty=req.qty,
                lot_size=int(inst["fno_lot_size"]) if inst else req.lot_size,
                notes=req.notes,
            )
        return {"position": pos, "portfolio": paper_trader.get_portfolio_summary()}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/paper/positions/{pos_id}/close")
async def close_paper_position(pos_id: str) -> dict[str, Any]:
    try:
        res = paper_trader.close_position(pos_id)
        return {"result": res, "portfolio": paper_trader.get_portfolio_summary()}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/paper/positions/{pos_id}/breakeven")
async def move_breakeven(pos_id: str) -> dict[str, Any]:
    try:
        pos = paper_trader.move_sl_to_breakeven(pos_id)
        return {"position": pos}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/paper/positions/{pos_id}/levels")
async def update_levels(pos_id: str, req: UpdateLevelsRequest) -> dict[str, Any]:
    try:
        pos = paper_trader.update_sl_tp(pos_id, req.stop_loss, req.tp1, req.tp2)
        return {"position": pos}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/paper/refresh-mtm")
async def refresh_mtm() -> dict[str, Any]:
    return await scanner.refresh_paper_mtm()


@app.post("/api/paper/reset")
async def reset_paper() -> dict[str, Any]:
    return {"portfolio": paper_trader.reset_account()}


@app.post("/api/settings/risk")
async def update_risk(req: RiskSettingsRequest) -> dict[str, Any]:
    updated = db.update_risk_settings(req.model_dump())
    return {"settings": updated, "portfolio": paper_trader.get_portfolio_summary()}


@app.post("/api/dhan/auth/update-token")
async def update_dhan_token(req: UpdateTokenRequest) -> dict[str, Any]:
    dhan_client.update_credentials(client_id=req.client_id, access_token=req.access_token)
    auth = await dhan_client.check_auth(force=True)
    return {"auth": auth}


@app.post("/api/dhan/auth/generate-consent")
async def generate_dhan_consent(req: ConsentRequest) -> dict[str, Any]:
    try:
        return await dhan_client.generate_consent(req.client_id, req.app_id, req.app_secret)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/dhan/auth/consume-consent")
async def consume_dhan_consent(req: ConsumeConsentRequest) -> dict[str, Any]:
    try:
        data = await dhan_client.consume_consent(req.token_id, req.app_id, req.app_secret)
        auth = await dhan_client.check_auth(force=True)
        return {"auth": auth, "expiryTime": data.get("expiryTime")}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/dhan/callback")
async def dhan_oauth_callback(tokenId: str = Query(default="")) -> RedirectResponse:
    if tokenId.strip():
        try:
            await dhan_client.consume_consent(tokenId.strip())
        except Exception:
            pass
    return RedirectResponse(url="/")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port, reload=False)
