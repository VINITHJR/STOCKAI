"""Shared Upstox Market Data Feed V3 -> FastAPI WebSocket bridge.

One Upstox streamer is shared by all StockAI browser tabs/pages. Each browser
WebSocket subscribes to one instrument key and receives normalized JSON ticks.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid
from collections import defaultdict
from typing import Any

from dotenv import load_dotenv

load_dotenv("backend/.env")

try:
    import upstox_client
except ImportError:  # pragma: no cover
    upstox_client = None


class MarketStreamManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._streamer = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queues: dict[str, set[asyncio.Queue]] = defaultdict(set)
        self._subscribed: set[str] = set()
        self._pending: set[str] = set()
        self._connected = False
        self._last_error = ""
        self._last_status: dict[str, str] = {}
        self._started = False

    @staticmethod
    def _token() -> str:
        token = os.getenv("UPSTOX_ACCESS_TOKEN", "").strip()
        if not token:
            raise RuntimeError("UPSTOX_ACCESS_TOKEN is missing in backend/.env")
        return token

    @staticmethod
    def _segment_for(instrument_key: str) -> str:
        if instrument_key.startswith("NSE_EQ|"):
            return "NSE_EQ"
        if instrument_key.startswith("NSE_INDEX|"):
            return "NSE_INDEX"
        if instrument_key.startswith("BSE_EQ|"):
            return "BSE_EQ"
        if instrument_key.startswith("BSE_INDEX|"):
            return "BSE_INDEX"
        return instrument_key.split("|", 1)[0]

    def _publish(self, instrument_key: str | None, payload: dict[str, Any]) -> None:
        with self._lock:
            targets = set(self._queues.get(instrument_key or "", set()))
            if instrument_key is None:
                targets = {q for qs in self._queues.values() for q in qs}

        for queue in targets:
            try:
                if self._loop and not self._loop.is_closed():
                    self._loop.call_soon_threadsafe(queue.put_nowait, payload)
            except Exception:
                pass

    def _on_open(self) -> None:
        print("Upstox V3 market stream connected")
        with self._lock:
            self._connected = True
            pending = sorted(self._pending | set(self._queues.keys()))
            self._pending.clear()

        if pending and self._streamer is not None:
            try:
                self._streamer.subscribe(pending, "ltpc")
                with self._lock:
                    self._subscribed.update(pending)
                print(f"Upstox stream subscribed: {len(pending)} instrument(s)")
            except Exception as exc:
                self._last_error = str(exc)
                print("Upstox subscribe error:", repr(exc))
                self._publish(None, {"type": "stream_error", "message": str(exc)})

        self._publish(None, {"type": "stream_status", "status": "connected"})

    def _on_close(self, *args: Any) -> None:
        print("Upstox V3 market stream closed")
        with self._lock:
            self._connected = False
            self._subscribed.clear()
            self._pending.update(self._queues.keys())
        self._publish(None, {"type": "stream_status", "status": "disconnected"})

    def _on_error(self, error: Any) -> None:
        self._last_error = str(error)
        print("Upstox V3 market stream error:", repr(error))
        self._publish(None, {"type": "stream_error", "message": str(error)})

    @staticmethod
    def _get_ltpc(feed: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(feed, dict):
            return None
        if isinstance(feed.get("ltpc"), dict):
            return feed["ltpc"]
        # FULL mode compatibility if the mode is changed later.
        for key in ("fullFeed", "firstLevelWithGreeks"):
            nested = feed.get(key)
            if isinstance(nested, dict) and isinstance(nested.get("marketFF"), dict):
                nested = nested["marketFF"]
            if isinstance(nested, dict) and isinstance(nested.get("ltpc"), dict):
                return nested["ltpc"]
        return None

    def _on_message(self, message: Any) -> None:
        if not isinstance(message, dict):
            return

        message_type = message.get("type")
        current_ts = message.get("currentTs") or message.get("current_ts")

        if message_type == "market_info":
            market_info = message.get("marketInfo") or message.get("market_info") or {}
            statuses = market_info.get("segmentStatus") or market_info.get("segment_status") or {}
            if isinstance(statuses, dict):
                with self._lock:
                    self._last_status.update({str(k): str(v) for k, v in statuses.items()})
                for instrument_key in list(self._queues.keys()):
                    segment = self._segment_for(instrument_key)
                    status = statuses.get(segment)
                    if status:
                        self._publish(
                            instrument_key,
                            {
                                "type": "market_status",
                                "segment": segment,
                                "status": status,
                                "current_ts": current_ts,
                            },
                        )
            return

        feeds = message.get("feeds") or {}
        if not isinstance(feeds, dict):
            return

        for instrument_key, feed in feeds.items():
            ltpc = self._get_ltpc(feed)
            if not ltpc:
                continue
            ltp = ltpc.get("ltp")
            cp = ltpc.get("cp")
            if ltp is None:
                continue

            try:
                ltp = float(ltp)
                cp = float(cp) if cp is not None else None
            except (TypeError, ValueError):
                continue

            change = None
            net_change = None
            if cp is not None:
                net_change = ltp - cp
                change = (net_change / cp * 100) if cp else 0

            self._publish(
                instrument_key,
                {
                    "type": "tick",
                    "instrument_key": instrument_key,
                    "ltp": ltp,
                    "cp": cp,
                    "net_change": net_change,
                    "change": change,
                    "ltt": ltpc.get("ltt"),
                    "ltq": ltpc.get("ltq"),
                    "current_ts": current_ts,
                },
            )

    def _connect_worker(self) -> None:
        if upstox_client is None:
            self._last_error = "upstox-python-sdk is not installed"
            self._publish(None, {"type": "stream_error", "message": self._last_error})
            return

        try:
            configuration = upstox_client.Configuration()
            configuration.access_token = self._token()
            api_client = upstox_client.ApiClient(configuration)
            self._streamer = upstox_client.MarketDataStreamerV3(api_client)
            self._streamer.on("open", self._on_open)
            self._streamer.on("message", self._on_message)
            self._streamer.on("error", self._on_error)
            self._streamer.on("close", self._on_close)
            # Keep reconnecting while the backend is alive.
            try:
                self._streamer.auto_reconnect(True, 5, 20)
            except Exception:
                pass
            self._streamer.connect()
        except Exception as exc:
            self._last_error = str(exc)
            print("Could not start Upstox market stream:", repr(exc))
            self._publish(None, {"type": "stream_error", "message": str(exc)})
        finally:
            with self._lock:
                self._connected = False

    def _ensure_started(self, loop: asyncio.AbstractEventLoop) -> None:
        with self._lock:
            self._loop = loop
            if self._started and self._thread and self._thread.is_alive():
                return
            self._started = True
            self._thread = threading.Thread(
                target=self._connect_worker,
                name="upstox-market-stream",
                daemon=True,
            )
            self._thread.start()

    async def register(self, instrument_key: str) -> asyncio.Queue:
        if not instrument_key or "|" not in instrument_key:
            raise ValueError("A valid Upstox instrument_key is required")

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._ensure_started(loop)

        with self._lock:
            self._queues[instrument_key].add(queue)
            if instrument_key not in self._subscribed:
                self._pending.add(instrument_key)
                if self._connected and self._streamer is not None:
                    try:
                        self._streamer.subscribe([instrument_key], "ltpc")
                        self._subscribed.add(instrument_key)
                        self._pending.discard(instrument_key)
                    except Exception as exc:
                        self._last_error = str(exc)

            known_status = self._last_status.get(self._segment_for(instrument_key))

        if known_status:
            queue.put_nowait({
                "type": "market_status",
                "segment": self._segment_for(instrument_key),
                "status": known_status,
            })
        return queue

    def unregister(self, instrument_key: str, queue: asyncio.Queue) -> None:
        with self._lock:
            queues = self._queues.get(instrument_key)
            if not queues:
                return
            queues.discard(queue)
            if not queues:
                self._queues.pop(instrument_key, None)
                if self._streamer is not None and self._connected and instrument_key in self._subscribed:
                    try:
                        self._streamer.unsubscribe([instrument_key])
                    except Exception:
                        pass
                    self._subscribed.discard(instrument_key)
                self._pending.discard(instrument_key)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "connected": self._connected,
                "subscriptions": len(self._subscribed),
                "clients": sum(len(v) for v in self._queues.values()),
                "last_error": self._last_error,
                "market_status": dict(self._last_status),
            }


market_stream_manager = MarketStreamManager()
