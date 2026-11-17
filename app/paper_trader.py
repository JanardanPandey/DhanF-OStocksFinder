from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from app.config import settings
from app.database import db


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _estimate_charges(price: float, qty: int) -> float:
    turnover = abs(price * qty)
    return round(settings.brokerage_per_order + turnover * 0.00035, 2)


class PaperTrader:
    """Simulated execution & risk-managed paper trading ledger."""

    def open_trade(
        self,
        *,
        symbol: str,
        style: str,
        direction: str,
        grade: str,
        entry_price: float,
        stop_loss: float,
        tp1: float,
        tp2: float,
        qty: int | None = None,
        lot_size: int = 1,
        setup_id: str | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        symbol = symbol.upper()
        direction = direction.upper()
        account = db.get_account_state()
        risk_inr = float(account.get("risk_per_trade_inr", settings.default_risk_per_trade_inr))

        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            raise ValueError("Entry price and stop-loss must differ.")

        if qty is None or qty <= 0:
            qty = max(1, int(risk_inr // risk_per_share))

        # Apply tiny realistic entry slippage
        slip = entry_price * (settings.slippage_pct / 100.0)
        fill_entry = round(entry_price + slip if direction == "LONG" else entry_price - slip, 2)
        entry_charges = _estimate_charges(fill_entry, qty)

        pos_id = uuid.uuid4().hex[:10]
        now = _now_iso()
        with db.connect() as conn:
            conn.execute(
                """
                INSERT INTO paper_positions (
                    id, setup_id, symbol, style, direction, grade,
                    entry_price, current_price, stop_loss, initial_stop_loss,
                    tp1, tp2, initial_qty, remaining_qty, lot_size, tp1_hit,
                    status, realized_pnl, charges, opened_at, updated_at, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'OPEN', 0.0, ?, ?, ?, ?)
                """,
                (
                    pos_id,
                    setup_id,
                    symbol,
                    style,
                    direction,
                    grade,
                    fill_entry,
                    fill_entry,
                    round(stop_loss, 2),
                    round(stop_loss, 2),
                    round(tp1, 2),
                    round(tp2, 2),
                    int(qty),
                    int(qty),
                    int(lot_size),
                    entry_charges,
                    now,
                    now,
                    notes,
                ),
            )
            conn.execute(
                "UPDATE account_state SET total_charges = total_charges + ?, updated_at = ? WHERE id = 1",
                (entry_charges, now),
            )

        return self.get_position(pos_id)  # type: ignore[return-value]

    def get_position(self, pos_id: str) -> dict[str, Any] | None:
        with db.connect() as conn:
            row = conn.execute("SELECT * FROM paper_positions WHERE id = ?", (pos_id,)).fetchone()
        return self._enrich_position(dict(row)) if row else None

    def list_positions(self, status: str | None = "OPEN") -> list[dict[str, Any]]:
        with db.connect() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM paper_positions WHERE status = ? ORDER BY opened_at DESC",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM paper_positions ORDER BY opened_at DESC").fetchall()
        return [self._enrich_position(dict(r)) for r in rows]

    def list_trades(self, limit: int = 100) -> list[dict[str, Any]]:
        with db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM paper_trades ORDER BY closed_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def _enrich_position(pos: dict[str, Any]) -> dict[str, Any]:
        mult = 1.0 if pos["direction"] == "LONG" else -1.0
        unrealized = (pos["current_price"] - pos["entry_price"]) * pos["remaining_qty"] * mult
        risk_per_share = max(0.01, abs(pos["entry_price"] - pos["initial_stop_loss"]))
        pos["unrealized_pnl"] = round(unrealized, 2)
        pos["net_total_pnl"] = round(pos["realized_pnl"] + unrealized - pos["charges"], 2)
        pos["risk_inr"] = round(risk_per_share * pos["initial_qty"], 2)
        pos["current_r"] = round(((pos["current_price"] - pos["entry_price"]) * mult) / risk_per_share, 2)
        return pos

    def move_sl_to_breakeven(self, pos_id: str) -> dict[str, Any]:
        pos = self.get_position(pos_id)
        if not pos or pos["status"] != "OPEN":
            raise ValueError("Open position not found")
        with db.connect() as conn:
            conn.execute(
                "UPDATE paper_positions SET stop_loss = entry_price, updated_at = ? WHERE id = ?",
                (_now_iso(), pos_id),
            )
        return self.get_position(pos_id)  # type: ignore[return-value]

    def update_sl_tp(
        self, pos_id: str, stop_loss: float | None = None, tp1: float | None = None, tp2: float | None = None
    ) -> dict[str, Any]:
        pos = self.get_position(pos_id)
        if not pos or pos["status"] != "OPEN":
            raise ValueError("Open position not found")
        new_sl = round(stop_loss, 2) if stop_loss is not None else pos["stop_loss"]
        new_tp1 = round(tp1, 2) if tp1 is not None else pos["tp1"]
        new_tp2 = round(tp2, 2) if tp2 is not None else pos["tp2"]
        with db.connect() as conn:
            conn.execute(
                "UPDATE paper_positions SET stop_loss = ?, tp1 = ?, tp2 = ?, updated_at = ? WHERE id = ?",
                (new_sl, new_tp1, new_tp2, _now_iso(), pos_id),
            )
        return self.get_position(pos_id)  # type: ignore[return-value]

    def close_position(
        self, pos_id: str, exit_price: float | None = None, qty: int | None = None, reason: str = "MANUAL_EXIT"
    ) -> dict[str, Any]:
        pos = self.get_position(pos_id)
        if not pos or pos["status"] != "OPEN":
            raise ValueError("Open position not found")

        close_qty = pos["remaining_qty"] if qty is None else min(int(qty), pos["remaining_qty"])
        if close_qty <= 0:
            raise ValueError("Invalid close quantity")

        raw_exit = float(exit_price if exit_price is not None else pos["current_price"])
        slip = raw_exit * (settings.slippage_pct / 100.0)
        fill_exit = round(raw_exit - slip if pos["direction"] == "LONG" else raw_exit + slip, 2)

        mult = 1.0 if pos["direction"] == "LONG" else -1.0
        gross_pnl = round((fill_exit - pos["entry_price"]) * close_qty * mult, 2)
        exit_charges = _estimate_charges(fill_exit, close_qty)
        net_pnl = round(gross_pnl - exit_charges, 2)

        risk_per_share = max(0.01, abs(pos["entry_price"] - pos["initial_stop_loss"]))
        r_mult = round(((fill_exit - pos["entry_price"]) * mult) / risk_per_share, 2)

        now = _now_iso()
        new_rem = pos["remaining_qty"] - close_qty
        new_status = "CLOSED" if new_rem == 0 else "OPEN"
        trade_id = uuid.uuid4().hex[:10]

        with db.connect() as conn:
            conn.execute(
                """
                INSERT INTO paper_trades (
                    id, position_id, symbol, style, direction, grade,
                    entry_price, exit_price, initial_stop_loss, qty,
                    gross_pnl, charges, net_pnl, r_multiple, exit_reason,
                    opened_at, closed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade_id,
                    pos_id,
                    pos["symbol"],
                    pos["style"],
                    pos["direction"],
                    pos["grade"],
                    pos["entry_price"],
                    fill_exit,
                    pos["initial_stop_loss"],
                    close_qty,
                    gross_pnl,
                    exit_charges,
                    net_pnl,
                    r_mult,
                    reason,
                    pos["opened_at"],
                    now,
                ),
            )
            conn.execute(
                """
                UPDATE paper_positions
                SET remaining_qty = ?, status = ?,
                    realized_pnl = realized_pnl + ?, charges = charges + ?,
                    current_price = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_rem, new_status, gross_pnl, exit_charges, fill_exit, now, pos_id),
            )
            conn.execute(
                """
                UPDATE account_state
                SET realized_pnl = realized_pnl + ?,
                    total_charges = total_charges + ?,
                    updated_at = ?
                WHERE id = 1
                """,
                (net_pnl, exit_charges, now),
            )

        return {"trade_id": trade_id, "closed_qty": close_qty, "net_pnl": net_pnl, "status": new_status}

    def process_price_updates(self, ltp_map: dict[str, float]) -> list[dict[str, Any]]:
        """Update MTM and evaluate SL / TP1 (50% + BE) / TP2 triggers for all open positions."""
        events: list[dict[str, Any]] = []
        open_positions = self.list_positions(status="OPEN")
        now = _now_iso()

        for pos in open_positions:
            sym = pos["symbol"]
            if sym not in ltp_map:
                continue
            ltp = float(ltp_map[sym])
            with db.connect() as conn:
                conn.execute(
                    "UPDATE paper_positions SET current_price = ?, updated_at = ? WHERE id = ?",
                    (ltp, now, pos["id"]),
                )

            is_long = pos["direction"] == "LONG"
            sl_hit = (ltp <= pos["stop_loss"]) if is_long else (ltp >= pos["stop_loss"])
            tp2_hit = (ltp >= pos["tp2"]) if is_long else (ltp <= pos["tp2"])
            tp1_hit = (ltp >= pos["tp1"]) if is_long else (ltp <= pos["tp1"])

            if sl_hit:
                res = self.close_position(pos["id"], exit_price=pos["stop_loss"], reason="SL_HIT")
                events.append({"symbol": sym, "event": "SL_HIT", **res})
            elif tp2_hit:
                res = self.close_position(pos["id"], exit_price=pos["tp2"], reason="TP2_TARGET")
                events.append({"symbol": sym, "event": "TP2_TARGET", **res})
            elif tp1_hit and not pos["tp1_hit"]:
                partial_qty = max(1, pos["remaining_qty"] // 2)
                res = self.close_position(pos["id"], exit_price=pos["tp1"], qty=partial_qty, reason="TP1_PARTIAL")
                if pos["remaining_qty"] - partial_qty > 0:
                    with db.connect() as conn:
                        conn.execute(
                            """
                            UPDATE paper_positions
                            SET tp1_hit = 1, stop_loss = entry_price, updated_at = ?
                            WHERE id = ?
                            """,
                            (_now_iso(), pos["id"]),
                        )
                events.append({"symbol": sym, "event": "TP1_PARTIAL_BE", **res})

        return events

    def auto_take_setups(self, setups: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Automatically enter Grade A+ setups when Auto-Paper-Trade is enabled."""
        account = db.get_account_state()
        if not account.get("auto_paper_trade"):
            return []

        open_positions = self.list_positions(status="OPEN")
        if len(open_positions) >= settings.max_open_paper_positions:
            return []

        active_keys = {(p["symbol"], p["style"], p["direction"]) for p in open_positions}
        opened: list[dict[str, Any]] = []

        for s in setups:
            if len(open_positions) + len(opened) >= settings.max_open_paper_positions:
                break
            if s["grade"] != settings.auto_paper_min_grade:
                continue
            if s.get("entry_mode") != "READY_NOW":
                continue
            key = (s["symbol"], s["style"], s["direction"])
            if key in active_keys:
                continue
            pos = self.open_trade(
                symbol=s["symbol"],
                style=s["style"],
                direction=s["direction"],
                grade=s["grade"],
                entry_price=s["entry_price"],
                stop_loss=s["stop_loss"],
                tp1=s["tp1"],
                tp2=s["tp2"],
                qty=s["suggested_qty"],
                lot_size=s["lot_size"],
                setup_id=s["id"],
                notes="Auto-entered A+ setup",
            )
            active_keys.add(key)
            opened.append(pos)
        return opened

    def get_portfolio_summary(self) -> dict[str, Any]:
        account = db.get_account_state()
        open_positions = self.list_positions(status="OPEN")
        trades = self.list_trades(limit=500)

        unrealized_mtm = round(sum(p["unrealized_pnl"] for p in open_positions), 2)
        realized_pnl = round(float(account.get("realized_pnl", 0.0)), 2)
        initial_cap = float(account.get("initial_capital", settings.paper_initial_capital))
        equity = round(initial_cap + realized_pnl + unrealized_mtm, 2)

        wins = [t for t in trades if t["net_pnl"] > 0]
        losses = [t for t in trades if t["net_pnl"] <= 0]
        win_rate = round((len(wins) / len(trades) * 100.0), 1) if trades else 0.0

        gross_win = sum(t["net_pnl"] for t in wins)
        gross_loss = abs(sum(t["net_pnl"] for t in losses))
        profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else (9.99 if gross_win > 0 else 0.0)
        avg_r = round(sum(t["r_multiple"] for t in trades) / len(trades), 2) if trades else 0.0

        return {
            "initial_capital": initial_cap,
            "equity": equity,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_mtm,
            "total_pnl": round(realized_pnl + unrealized_mtm, 2),
            "return_pct": round(((equity - initial_cap) / initial_cap) * 100.0, 2) if initial_cap else 0.0,
            "total_charges": round(float(account.get("total_charges", 0.0)), 2),
            "open_count": len(open_positions),
            "closed_trades_count": len(trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_r_multiple": avg_r,
            "settings": {
                "risk_per_trade_inr": float(account.get("risk_per_trade_inr", 1000.0)),
                "max_sl_pct_intraday": float(account.get("max_sl_pct_intraday", 0.65)),
                "max_sl_pct_swing": float(account.get("max_sl_pct_swing", 1.80)),
                "min_rr_ratio": float(account.get("min_rr_ratio", 2.0)),
                "auto_paper_trade": bool(account.get("auto_paper_trade", 0)),
            },
        }

    def reset_account(self) -> dict[str, Any]:
        now = _now_iso()
        with db.connect() as conn:
            conn.execute("DELETE FROM paper_positions")
            conn.execute("DELETE FROM paper_trades")
            conn.execute(
                "UPDATE account_state SET realized_pnl = 0.0, total_charges = 0.0, updated_at = ? WHERE id = 1",
                (now,),
            )
        return self.get_portfolio_summary()


paper_trader = PaperTrader()
