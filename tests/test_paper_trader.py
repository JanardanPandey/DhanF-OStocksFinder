from __future__ import annotations

from pathlib import Path

from app.database import Database
import app.paper_trader as pt_mod


def test_paper_trader_tp1_breakeven_and_tp2(tmp_path: Path, monkeypatch):
    test_db = Database(tmp_path / "test.db")
    monkeypatch.setattr(pt_mod, "db", test_db)
    trader = pt_mod.PaperTrader()

    pos = trader.open_trade(
        symbol="RELIANCE",
        style="INTRADAY",
        direction="LONG",
        grade="A+",
        entry_price=1000.0,
        stop_loss=995.0,
        tp1=1008.0,
        tp2=1015.0,
        qty=100,
        lot_size=500,
    )
    assert pos["remaining_qty"] == 100

    # Hit TP1 -> 50% partial exit + SL moved to breakeven
    events = trader.process_price_updates({"RELIANCE": 1009.0})
    assert len(events) == 1
    assert events[0]["event"] == "TP1_PARTIAL_BE"

    updated = trader.get_position(pos["id"])
    assert updated is not None
    assert updated["remaining_qty"] == 50
    assert updated["tp1_hit"] == 1
    assert updated["stop_loss"] == updated["entry_price"]

    # Hit TP2 -> closes remaining 50%
    events2 = trader.process_price_updates({"RELIANCE": 1016.0})
    assert len(events2) == 1
    assert events2[0]["event"] == "TP2_TARGET"

    summary = trader.get_portfolio_summary()
    assert summary["open_count"] == 0
    assert summary["closed_trades_count"] == 2
    assert summary["realized_pnl"] > 0
    assert summary["win_rate"] == 100.0
