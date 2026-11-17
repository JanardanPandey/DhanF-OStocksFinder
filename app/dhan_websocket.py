from __future__ import annotations

import asyncio
import json
import logging
import struct
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import websockets

from app.config import settings

logger = logging.getLogger("dhan_ws")
IST = timezone(timedelta(hours=5, minutes=30))
DHAN_WS_BASE = "wss://api-feed.dhan.co"


class DhanWebsocketFeed:
    """
    High-performance binary WebSocket client for DhanHQ Market Feed v2 (wss://api-feed.dhan.co).
    Streamlines Quote, Ticker, and OI packets without HTTP polling overhead.
    """

    def __init__(self) -> None:
        self._ws: Any = None
        self._running: bool = False
        self._task: asyncio.Task | None = None
        self._connected: bool = False
        self._last_tick_at: datetime | None = None
        self._ticks_received: int = 0
        self._subscribed_sec_ids: set[str] = set()
        self._live_quotes: dict[str, dict[str, Any]] = {}
        self._symbol_to_secid: dict[str, str] = {}
        self._secid_to_symbol: dict[str, str] = {}
        self._tick_listeners: list[Callable[[str, dict[str, Any]], None]] = []

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ws is not None

    def get_status(self) -> dict[str, Any]:
        return {
            "connected": self.is_connected,
            "running": self._running,
            "ticks_received": self._ticks_received,
            "subscribed_count": len(self._subscribed_sec_ids),
            "last_tick_at": self._last_tick_at.isoformat() if self._last_tick_at else None,
            "ws_url": DHAN_WS_BASE,
        }

    def register_instrument_mapping(self, symbol: str, security_id: str) -> None:
        sym_u = symbol.upper()
        sec_str = str(security_id)
        self._symbol_to_secid[sym_u] = sec_str
        self._secid_to_symbol[sec_str] = sym_u

    def get_quote(self, symbol_or_secid: str) -> dict[str, Any] | None:
        key = str(symbol_or_secid).upper()
        if key in self._symbol_to_secid:
            sec_id = self._symbol_to_secid[key]
            return self._live_quotes.get(sec_id)
        return self._live_quotes.get(key)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._connection_loop())

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._task and not self._task.done():
            self._task.cancel()
        self._connected = False

    async def subscribe(self, security_ids: list[str | int], segment: str = "NSE_EQ") -> None:
        new_ids = [str(sid) for sid in security_ids if str(sid) not in self._subscribed_sec_ids]
        if not new_ids:
            return
        for sid in new_ids:
            self._subscribed_sec_ids.add(sid)

        if self.is_connected:
            await self._send_subscription(new_ids, segment=segment)

    async def _send_subscription(self, sec_ids: list[str], segment: str = "NSE_EQ") -> None:
        if not self._ws or not sec_ids:
            return
        payload = {
            "RequestCode": 17,  # Quote packet (LTP, LTQ, Volume, OHLC)
            "InstrumentCount": len(sec_ids),
            "InstrumentList": [{"ExchangeSegment": segment, "SecurityId": sid} for sid in sec_ids],
        }
        try:
            await self._ws.send(json.dumps(payload))
            logger.info("Subscribed %d instruments to Dhan WebSocket feed", len(sec_ids))
        except Exception as e:
            logger.warning("Failed to send subscription: %s", e)

    async def _connection_loop(self) -> None:
        while self._running:
            client_id = settings.dhan_client_id.strip()
            access_token = settings.dhan_access_token.strip()

            if not client_id or not access_token:
                self._connected = False
                await asyncio.sleep(10)
                continue

            ws_url = f"{DHAN_WS_BASE}?version=2&token={access_token}&clientId={client_id}&authType=2"
            try:
                async with websockets.connect(
                    ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self._ws = ws
                    self._connected = True
                    logger.info("Dhan WebSocket feed connected.")

                    # Resubscribe instruments
                    if self._subscribed_sec_ids:
                        await self._send_subscription(list(self._subscribed_sec_ids))

                    async for message in ws:
                        if not self._running:
                            break
                        if isinstance(message, bytes):
                            self._parse_binary_packet(message)
                        elif isinstance(message, str):
                            try:
                                data = json.loads(message)
                                # JSON control messages
                                if data.get("type") == "disconnect":
                                    logger.warning("Dhan feed requested disconnect")
                                    break
                            except Exception:
                                pass
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._connected = False
                self._ws = None
                logger.debug("Dhan WebSocket connection error: %s. Reconnecting in 5s...", e)
                await asyncio.sleep(5)

    def _parse_binary_packet(self, data: bytes) -> None:
        """
        Dhan v2 Little-Endian Binary Feed Parser.
        Header: 8 bytes (<BhBI)
        """
        if len(data) < 8:
            return

        try:
            feed_code, msg_len, exchange_segment, security_id = struct.unpack("<BhBI", data[:8])
            sec_id_str = str(security_id)
            self._ticks_received += 1
            self._last_tick_at = datetime.now(IST)

            payload = data[8:]

            # Feed Code 4: Quote (LTP, LTQ, LTT, AvgPrice, Volume, TotalBuyQty, TotalSellQty, Open, High, Low, Close)
            if feed_code == 4 and len(payload) >= 42:
                (
                    ltp,
                    ltq,
                    ltt,
                    avg_price,
                    vol,
                    buy_qty,
                    sell_qty,
                    op,
                    hi,
                    lo,
                    cl,
                ) = struct.unpack("<fHIffIIffff", payload[:42])

                quote_info = {
                    "security_id": sec_id_str,
                    "symbol": self._secid_to_symbol.get(sec_id_str, sec_id_str),
                    "last_price": round(float(ltp), 2),
                    "ltq": int(ltq),
                    "volume": int(vol),
                    "open": round(float(op), 2),
                    "high": round(float(hi), 2),
                    "low": round(float(lo), 2),
                    "close": round(float(cl), 2),
                    "average_price": round(float(avg_price), 2),
                    "updated_at": self._last_tick_at.isoformat(timespec="seconds"),
                }
                self._live_quotes[sec_id_str] = quote_info

            # Feed Code 2: Ticker (LTP, LTT)
            elif feed_code == 2 and len(payload) >= 8:
                ltp, ltt = struct.unpack("<fI", payload[:8])
                prev = self._live_quotes.get(sec_id_str, {})
                prev.update(
                    {
                        "security_id": sec_id_str,
                        "symbol": self._secid_to_symbol.get(sec_id_str, sec_id_str),
                        "last_price": round(float(ltp), 2),
                        "updated_at": self._last_tick_at.isoformat(timespec="seconds"),
                    }
                )
                self._live_quotes[sec_id_str] = prev

            # Feed Code 5: Open Interest
            elif feed_code == 5 and len(payload) >= 4:
                (oi,) = struct.unpack("<I", payload[:4])
                prev = self._live_quotes.get(sec_id_str, {})
                prev["oi"] = int(oi)
                self._live_quotes[sec_id_str] = prev

        except Exception as e:
            logger.debug("Failed to unpack Dhan binary packet: %s", e)


dhan_feed = DhanWebsocketFeed()
