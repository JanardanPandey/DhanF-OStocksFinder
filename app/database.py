from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.config import DB_PATH, settings


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    """SQLite storage for F&O instruments, cached candles, setups, and paper trades."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS instruments (
                    symbol TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    equity_security_id TEXT NOT NULL,
                    fno_lot_size INTEGER NOT NULL DEFAULT 1,
                    is_fno INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS candles (
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    ts INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    PRIMARY KEY (symbol, timeframe, ts)
                );

                CREATE INDEX IF NOT EXISTS idx_candles_lookup
                    ON candles (symbol, timeframe, ts);

                CREATE TABLE IF NOT EXISTS setups (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    style TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    grade TEXT NOT NULL,
                    score INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    tp1 REAL NOT NULL,
                    tp2 REAL NOT NULL,
                    sl_pct REAL NOT NULL,
                    rr_ratio REAL NOT NULL,
                    ltp REAL NOT NULL,
                    lot_size INTEGER NOT NULL,
                    suggested_qty INTEGER NOT NULL,
                    data_source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS account_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    initial_capital REAL NOT NULL,
                    realized_pnl REAL NOT NULL DEFAULT 0.0,
                    total_charges REAL NOT NULL DEFAULT 0.0,
                    risk_per_trade_inr REAL NOT NULL,
                    max_sl_pct_intraday REAL NOT NULL,
                    max_sl_pct_swing REAL NOT NULL,
                    min_rr_ratio REAL NOT NULL,
                    auto_paper_trade INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS paper_positions (
                    id TEXT PRIMARY KEY,
                    setup_id TEXT,
                    symbol TEXT NOT NULL,
                    style TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    grade TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    current_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    initial_stop_loss REAL NOT NULL,
                    tp1 REAL NOT NULL,
                    tp2 REAL NOT NULL,
                    initial_qty INTEGER NOT NULL,
                    remaining_qty INTEGER NOT NULL,
                    lot_size INTEGER NOT NULL DEFAULT 1,
                    tp1_hit INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    realized_pnl REAL NOT NULL DEFAULT 0.0,
                    charges REAL NOT NULL DEFAULT 0.0,
                    opened_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS paper_trades (
                    id TEXT PRIMARY KEY,
                    position_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    style TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    grade TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    initial_stop_loss REAL NOT NULL,
                    qty INTEGER NOT NULL,
                    gross_pnl REAL NOT NULL,
                    charges REAL NOT NULL,
                    net_pnl REAL NOT NULL,
                    r_multiple REAL NOT NULL,
                    exit_reason TEXT NOT NULL,
                    opened_at TEXT NOT NULL,
                    closed_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS daily_watchlist (
                    target_date TEXT NOT NULL,
                    t1_date TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    bias_hint TEXT NOT NULL,
                    score REAL NOT NULL,
                    trigger_time TEXT NOT NULL,
                    reasons_json TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    anti_chasing INTEGER NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (target_date, symbol)
                );

                CREATE INDEX IF NOT EXISTS idx_daily_watchlist_date_rank
                    ON daily_watchlist(target_date, rank);

                CREATE TABLE IF NOT EXISTS market_breadth (
                    trade_date TEXT PRIMARY KEY,
                    abv_vwap_pct REAL NOT NULL,
                    green_open_pct REAL NOT NULL,
                    nse_adv_pct REAL NOT NULL,
                    regime TEXT NOT NULL,
                    sample_size INTEGER NOT NULL,
                    source TEXT NOT NULL DEFAULT 'cached',
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS live_momentum (
                    symbol TEXT PRIMARY KEY,
                    rank INTEGER NOT NULL,
                    trade_date TEXT NOT NULL,
                    score REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS live_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE NOT NULL,
                    trade_date TEXT NOT NULL,
                    timestamp_ist TEXT NOT NULL,
                    epoch_ts INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    price REAL NOT NULL,
                    vwap REAL NOT NULL,
                    move_pct REAL NOT NULL,
                    oi_build_up TEXT NOT NULL,
                    vol_delta_shares TEXT NOT NULL,
                    ob_zone TEXT NOT NULL,
                    s1 REAL NOT NULL,
                    r1 REAL NOT NULL,
                    r2 REAL NOT NULL,
                    option_strike_name TEXT NOT NULL,
                    option_est_premium REAL NOT NULL,
                    option_lot_capital REAL NOT NULL,
                    option_sl_premium REAL NOT NULL,
                    option_tp1_premium REAL NOT NULL,
                    reasons_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_live_alerts_date_ts
                    ON live_alerts(trade_date, epoch_ts DESC);
                CREATE INDEX IF NOT EXISTS idx_live_alerts_symbol
                    ON live_alerts(symbol, trade_date);
                """
            )
            try:
                conn.execute("ALTER TABLE candles ADD COLUMN oi REAL DEFAULT 0")
            except Exception:
                pass
            row = conn.execute("SELECT id FROM account_state WHERE id = 1").fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO account_state (
                        id, initial_capital, realized_pnl, total_charges,
                        risk_per_trade_inr, max_sl_pct_intraday, max_sl_pct_swing,
                        min_rr_ratio, auto_paper_trade, updated_at
                    ) VALUES (?, ?, 0.0, 0.0, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        1,
                        settings.paper_initial_capital,
                        settings.default_risk_per_trade_inr,
                        settings.max_sl_pct_intraday,
                        settings.max_sl_pct_swing,
                        settings.min_rr_ratio,
                        1 if settings.auto_paper_trade else 0,
                        utc_now_iso(),
                    ),
                )

    def get_account_state(self) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM account_state WHERE id = 1").fetchone()
            return dict(row) if row else {}

    def update_risk_settings(self, updates: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "initial_capital",
            "risk_per_trade_inr",
            "max_sl_pct_intraday",
            "max_sl_pct_swing",
            "min_rr_ratio",
            "auto_paper_trade",
        }
        filtered = {k: v for k, v in updates.items() if k in allowed and v is not None}
        if not filtered:
            return self.get_account_state()
        if "auto_paper_trade" in filtered:
            filtered["auto_paper_trade"] = 1 if filtered["auto_paper_trade"] else 0
        filtered["updated_at"] = utc_now_iso()
        cols = ", ".join(f"{k} = ?" for k in filtered)
        vals = list(filtered.values())
        with self.connect() as conn:
            conn.execute(f"UPDATE account_state SET {cols} WHERE id = 1", vals)
        return self.get_account_state()

    def upsert_instruments(self, items: list[dict[str, Any]]) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO instruments (symbol, name, equity_security_id, fno_lot_size, is_fno, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    name = excluded.name,
                    equity_security_id = excluded.equity_security_id,
                    fno_lot_size = excluded.fno_lot_size,
                    is_fno = excluded.is_fno,
                    updated_at = excluded.updated_at
                """,
                [
                    (
                        item["symbol"],
                        item["name"],
                        str(item["equity_security_id"]),
                        int(item.get("fno_lot_size", 1)),
                        1 if item.get("is_fno", True) else 0,
                        now,
                    )
                    for item in items
                ],
            )

    def get_instruments(self, limit: int | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM instruments WHERE is_fno = 1 ORDER BY symbol ASC"
        if limit:
            query += f" LIMIT {int(limit)}"
        with self.connect() as conn:
            return [dict(r) for r in conn.execute(query).fetchall()]

    def get_instrument(self, symbol: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM instruments WHERE symbol = ?",
                (symbol.upper(),),
            ).fetchone()
            return dict(row) if row else None

    def save_candles(self, symbol: str, timeframe: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO candles (symbol, timeframe, ts, open, high, low, close, volume, oi)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        symbol.upper(),
                        timeframe,
                        int(r["ts"]),
                        float(r["open"]),
                        float(r["high"]),
                        float(r["low"]),
                        float(r["close"]),
                        float(r["volume"]),
                        float(r.get("oi") or 0.0),
                    )
                    for r in rows
                ],
            )

    def load_candles(self, symbol: str, timeframe: str, limit: int = 500) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT ts, open, high, low, close, volume, COALESCE(oi, 0) as oi
                FROM candles
                WHERE symbol = ? AND timeframe = ?
                ORDER BY ts DESC
                LIMIT ?
                """,
                (symbol.upper(), timeframe, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def replace_setups(self, setups: list[dict[str, Any]]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM setups")
            conn.executemany(
                """
                INSERT INTO setups (
                    id, symbol, style, direction, grade, score,
                    entry_price, stop_loss, tp1, tp2, sl_pct, rr_ratio,
                    ltp, lot_size, suggested_qty, data_source, created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        s["id"],
                        s["symbol"],
                        s["style"],
                        s["direction"],
                        s["grade"],
                        int(s["score"]),
                        float(s["entry_price"]),
                        float(s["stop_loss"]),
                        float(s["tp1"]),
                        float(s["tp2"]),
                        float(s["sl_pct"]),
                        float(s["rr_ratio"]),
                        float(s["ltp"]),
                        int(s["lot_size"]),
                        int(s["suggested_qty"]),
                        s["data_source"],
                        s["created_at"],
                        json.dumps(s),
                    )
                    for s in setups
                ],
            )

    def get_setups(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM setups ORDER BY score DESC, rr_ratio DESC"
            ).fetchall()
        items = [json.loads(r["payload_json"]) for r in rows]
        items.sort(key=lambda x: (int(x.get("rank") or 99999), -float(x.get("score") or 0), -float(x.get("rr_ratio") or 0)))
        for idx, payload in enumerate(items, start=1):
            payload["rank"] = idx
        return items

    def get_setup(self, setup_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM setups WHERE id = ?",
                (setup_id,),
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def save_daily_watchlist(
        self,
        target_date: str,
        t1_date: str,
        items: list[dict[str, Any]],
    ) -> None:
        from datetime import datetime, timedelta, timezone

        now_ist = datetime.now(timezone(timedelta(hours=5, minutes=30))).isoformat(timespec="seconds")
        with self.connect() as conn:
            conn.execute("DELETE FROM daily_watchlist WHERE target_date = ?", (target_date,))
            for idx, item in enumerate(items, start=1):
                rank = int(item.get("rank") or idx)
                payload = dict(item)
                payload["rank"] = rank
                payload["target_date"] = target_date
                payload["t1_date"] = t1_date
                payload.setdefault("updated_at", now_ist)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO daily_watchlist (
                        target_date, t1_date, rank, symbol, bias_hint, score,
                        trigger_time, reasons_json, tags, anti_chasing, payload_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        target_date,
                        t1_date,
                        rank,
                        str(payload["symbol"]),
                        str(payload.get("bias_hint", "BOTH")),
                        float(payload.get("score", 0.0)),
                        str(payload.get("trigger_time", "09:15-09:30 IST")),
                        json.dumps(payload.get("reasons", [])),
                        str(payload.get("tags", "")),
                        int(bool(payload.get("anti_chasing", False))),
                        json.dumps(payload),
                        str(payload.get("updated_at", now_ist)),
                    ),
                )

    def get_daily_watchlist(self, target_date: str | None = None) -> list[dict[str, Any]]:
        with self.connect() as conn:
            resolved_date = target_date
            if not resolved_date:
                row = conn.execute(
                    "SELECT target_date FROM daily_watchlist ORDER BY target_date DESC LIMIT 1"
                ).fetchone()
                if not row:
                    return []
                resolved_date = str(row["target_date"])
            rows = conn.execute(
                """
                SELECT payload_json FROM daily_watchlist
                WHERE target_date = ?
                ORDER BY rank ASC, score DESC
                """,
                (resolved_date,),
            ).fetchall()
        items: list[dict[str, Any]] = []
        for idx, r in enumerate(rows, start=1):
            payload = json.loads(r["payload_json"])
            payload["rank"] = idx
            items.append(payload)
        return items

    def list_watchlist_dates(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    target_date,
                    MAX(t1_date) AS t1_date,
                    COUNT(*) AS count,
                    MAX(updated_at) AS updated_at,
                    (
                        SELECT symbol FROM daily_watchlist dw2
                        WHERE dw2.target_date = dw1.target_date
                        ORDER BY dw2.rank ASC, dw2.score DESC
                        LIMIT 1
                    ) AS top_symbol
                FROM daily_watchlist dw1
                GROUP BY target_date
                ORDER BY target_date DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def save_market_breadth(self, snapshot: dict[str, Any]) -> None:
        from datetime import datetime, timedelta, timezone

        now_ist = datetime.now(timezone(timedelta(hours=5, minutes=30))).isoformat(timespec="seconds")
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO market_breadth (
                    trade_date, abv_vwap_pct, green_open_pct, nse_adv_pct,
                    regime, sample_size, source, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(snapshot["trade_date"]),
                    float(snapshot["abv_vwap_pct"]),
                    float(snapshot["green_open_pct"]),
                    float(snapshot["nse_adv_pct"]),
                    str(snapshot["regime"]),
                    int(snapshot["sample_size"]),
                    str(snapshot.get("source", "cached")),
                    str(snapshot.get("updated_at", now_ist)),
                ),
            )

    def get_market_breadth(self, trade_date: str | None = None) -> dict[str, Any] | None:
        with self.connect() as conn:
            if trade_date:
                row = conn.execute(
                    "SELECT * FROM market_breadth WHERE trade_date = ?",
                    (trade_date,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM market_breadth ORDER BY trade_date DESC LIMIT 1"
                ).fetchone()
        return dict(row) if row else None

    def save_live_momentum(self, trade_date: str, items: list[dict[str, Any]]) -> None:
        from datetime import datetime, timedelta, timezone

        now_ist = datetime.now(timezone(timedelta(hours=5, minutes=30))).isoformat(timespec="seconds")
        with self.connect() as conn:
            conn.execute("DELETE FROM live_momentum")
            for idx, item in enumerate(items, start=1):
                rank = int(item.get("rank") or idx)
                payload = dict(item)
                payload["rank"] = rank
                payload["trade_date"] = trade_date
                payload.setdefault("updated_at", now_ist)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO live_momentum (
                        symbol, rank, trade_date, score, payload_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(payload["symbol"]),
                        rank,
                        trade_date,
                        float(payload.get("score", 0.0)),
                        json.dumps(payload),
                        str(payload["updated_at"]),
                    ),
                )

    def get_live_momentum(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM live_momentum ORDER BY rank ASC, score DESC"
            ).fetchall()
        items: list[dict[str, Any]] = []
        for idx, r in enumerate(rows, start=1):
            payload = json.loads(r["payload_json"])
            payload["rank"] = idx
            items.append(payload)
        return items

    def save_live_alert(self, alert: dict[str, Any]) -> None:
        """Append-only insert for live trigger alert events."""
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO live_alerts (
                    alert_id, trade_date, timestamp_ist, epoch_ts, symbol, direction,
                    event_type, price, vwap, move_pct, oi_build_up, vol_delta_shares,
                    ob_zone, s1, r1, r2, option_strike_name, option_est_premium,
                    option_lot_capital, option_sl_premium, option_tp1_premium,
                    reasons_json, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(alert["alert_id"]),
                    str(alert["trade_date"]),
                    str(alert["timestamp_ist"]),
                    int(alert["epoch_ts"]),
                    str(alert["symbol"]).upper(),
                    str(alert["direction"]),
                    str(alert["event_type"]),
                    float(alert["price"]),
                    float(alert["vwap"]),
                    float(alert["move_pct"]),
                    str(alert.get("oi_build_up", "")),
                    str(alert.get("vol_delta_shares", "")),
                    str(alert.get("ob_zone", "")),
                    float(alert.get("s1", 0.0)),
                    float(alert.get("r1", 0.0)),
                    float(alert.get("r2", 0.0)),
                    str(alert.get("option_strike_name", "")),
                    float(alert.get("option_est_premium", 0.0)),
                    float(alert.get("option_lot_capital", 0.0)),
                    float(alert.get("option_sl_premium", 0.0)),
                    float(alert.get("option_tp1_premium", 0.0)),
                    json.dumps(alert.get("reasons", [])),
                    json.dumps(alert),
                    str(alert.get("created_at", utc_now_iso())),
                ),
            )

    def get_live_alerts(self, trade_date: str | None = None, limit: int = 150) -> list[dict[str, Any]]:
        """Get chronological alerts (latest first)."""
        with self.connect() as conn:
            if trade_date:
                rows = conn.execute(
                    """
                    SELECT payload_json FROM live_alerts
                    WHERE trade_date = ?
                    ORDER BY epoch_ts DESC, id DESC
                    LIMIT ?
                    """,
                    (trade_date, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT payload_json FROM live_alerts
                    ORDER BY epoch_ts DESC, id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [json.loads(r["payload_json"]) for r in rows]

    def get_last_alert_for_symbol(self, symbol: str, trade_date: str) -> dict[str, Any] | None:
        """Get the most recent alert for a specific symbol on trade_date."""
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT payload_json FROM live_alerts
                WHERE symbol = ? AND trade_date = ?
                ORDER BY epoch_ts DESC, id DESC
                LIMIT 1
                """,
                (symbol.upper(), trade_date),
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def clear_live_alerts(self, trade_date: str | None = None) -> None:
        with self.connect() as conn:
            if trade_date:
                conn.execute("DELETE FROM live_alerts WHERE trade_date = ?", (trade_date,))
            else:
                conn.execute("DELETE FROM live_alerts")


db = Database()
