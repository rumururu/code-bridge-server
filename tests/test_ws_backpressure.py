"""Unit tests for WS-4 backpressure guard on the serialized chat websocket.

Covers ``routes.chat_ws._SerializedWebSocket`` directly so we can drive the
inflight-bytes gauge and the slow-consumer timeout without spinning a full
FastAPI ASGI app.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import unittest
from pathlib import Path
from typing import Any, Optional

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import routes.chat_ws as chat_ws  # noqa: E402
from routes.chat_ws import (  # noqa: E402
    WS_BACKPRESSURE_HIGH_WATERMARK_BYTES,
    _SerializedWebSocket,
)


class _FakeWebSocket:
    """Minimal stand-in for the ASGI WebSocket with a controllable send delay."""

    def __init__(self, send_delay_s: float = 0.0) -> None:
        self.send_delay_s = send_delay_s
        self.sent: list[Any] = []
        self.close_calls: list[tuple[Optional[int], Optional[str]]] = []
        self._send_started = asyncio.Event()
        self._release_send = asyncio.Event()
        self._block_sends = False

    async def send_json(self, data: Any) -> None:
        self._send_started.set()
        if self._block_sends:
            await self._release_send.wait()
        if self.send_delay_s > 0:
            await asyncio.sleep(self.send_delay_s)
        self.sent.append(data)

    async def close(
        self,
        code: Optional[int] = None,
        reason: Optional[str] = None,
    ) -> None:
        self.close_calls.append((code, reason))

    def block_sends(self) -> None:
        self._block_sends = True

    def release_sends(self) -> None:
        self._release_send.set()


def _payload_of_bytes(n_bytes: int) -> dict[str, Any]:
    """Build a JSON payload whose UTF-8 serialization is at least ``n_bytes``."""
    # Account for the JSON wrapper overhead ({"v": "..."}).
    overhead = len(json.dumps({"v": ""}).encode("utf-8"))
    body_len = max(0, n_bytes - overhead)
    return {"v": "a" * body_len}


class SerializedWebSocketBackpressureTest(unittest.IsolatedAsyncioTestCase):
    async def test_normal_send_passes_through_without_warning(self) -> None:
        fake = _FakeWebSocket()
        ws = _SerializedWebSocket(fake)

        await ws.send_json({"type": "ping"})

        self.assertEqual(fake.sent, [{"type": "ping"}])
        self.assertEqual(ws._inflight_bytes, 0)
        self.assertIsNone(ws._high_watermark_since)

    async def test_high_watermark_logs_warning(self) -> None:
        fake = _FakeWebSocket()
        ws = _SerializedWebSocket(
            fake,
            high_watermark_bytes=1024,
            slow_consumer_timeout_s=10.0,
            backpressure_sleep_s=0.0,
        )

        with self.assertLogs(chat_ws.logger, level="WARNING") as captured:
            await ws.send_json(_payload_of_bytes(2048))

        joined = "\n".join(captured.output)
        self.assertIn("WS send buffer high", joined)
        self.assertEqual(len(fake.sent), 1)
        # gauge always returns to zero after the await completes
        self.assertEqual(ws._inflight_bytes, 0)
        self.assertIsNone(ws._high_watermark_since)

    async def test_inflight_accumulates_when_underlying_send_blocks(self) -> None:
        fake = _FakeWebSocket()
        fake.block_sends()

        ws = _SerializedWebSocket(
            fake,
            high_watermark_bytes=1024,
            slow_consumer_timeout_s=10.0,
            backpressure_sleep_s=0.0,
        )

        # Queue three concurrent sends, each ~700 bytes; combined they cross
        # the 1024-byte high-water mark while the first send is blocked.
        tasks = [
            asyncio.create_task(ws.send_json(_payload_of_bytes(700))) for _ in range(3)
        ]

        # Wait until the underlying send has started before checking the gauge.
        await fake._send_started.wait()
        # Give the other coroutines a chance to enqueue.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        self.assertGreaterEqual(ws._inflight_bytes, 1024)
        self.assertIsNotNone(ws._high_watermark_since)

        fake.release_sends()
        await asyncio.gather(*tasks)

        self.assertEqual(ws._inflight_bytes, 0)
        self.assertIsNone(ws._high_watermark_since)

    async def test_slow_consumer_closes_with_1011(self) -> None:
        fake = _FakeWebSocket()
        fake.block_sends()

        ws = _SerializedWebSocket(
            fake,
            high_watermark_bytes=1024,
            slow_consumer_timeout_s=0.05,
            backpressure_sleep_s=0.0,
        )

        # First task crosses the high-water mark and blocks on the underlying
        # send. The second task waits for the send lock and will be the one
        # that observes the elapsed-high timeout and triggers the close.
        first = asyncio.create_task(ws.send_json(_payload_of_bytes(2048)))
        await fake._send_started.wait()

        second = asyncio.create_task(ws.send_json(_payload_of_bytes(2048)))

        # Wait past the slow-consumer timeout, then release the blocked send
        # so the second task can acquire the lock and observe the timeout.
        await asyncio.sleep(0.15)
        fake.release_sends()

        await asyncio.gather(first, second)

        self.assertTrue(ws._slow_consumer_closed)
        self.assertEqual(len(fake.close_calls), 1)
        code, reason = fake.close_calls[0]
        self.assertEqual(code, 1011)
        self.assertEqual(reason, "slow_consumer")

    async def test_send_after_slow_consumer_close_is_dropped(self) -> None:
        fake = _FakeWebSocket()
        ws = _SerializedWebSocket(
            fake,
            high_watermark_bytes=1024,
            slow_consumer_timeout_s=0.05,
            backpressure_sleep_s=0.0,
        )

        # Force the slow-consumer state so subsequent sends are dropped.
        ws._slow_consumer_closed = True

        await ws.send_json({"type": "ping"})

        self.assertEqual(fake.sent, [])
        self.assertEqual(fake.close_calls, [])

    async def test_watermark_default_constant_matches_documented_4mib(self) -> None:
        # Guards against accidental drift of the documented threshold.
        self.assertEqual(WS_BACKPRESSURE_HIGH_WATERMARK_BYTES, 4 * 1024 * 1024)


class SerializedWebSocketBasicSendTest(unittest.IsolatedAsyncioTestCase):
    """Regression coverage: serialized-send semantics still hold."""

    async def test_concurrent_sends_are_serialized(self) -> None:
        fake = _FakeWebSocket(send_delay_s=0.01)
        ws = _SerializedWebSocket(fake)

        await asyncio.gather(
            ws.send_json({"i": 1}),
            ws.send_json({"i": 2}),
            ws.send_json({"i": 3}),
        )

        # All payloads delivered; order preserved relative to start order is
        # not guaranteed by asyncio.gather, but the count must match.
        self.assertEqual(len(fake.sent), 3)


if __name__ == "__main__":  # pragma: no cover - manual smoke
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
