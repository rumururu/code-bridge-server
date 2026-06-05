"""Unit tests for WS-1 periodic re-authentication helper.

Covers ``routes.deps.start_periodic_reauth_task`` directly so we exercise the
revocation -> close(4001) path without needing to spin a full ASGI app.
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from typing import Optional

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from routes.deps import start_periodic_reauth_task  # noqa: E402


class _StubResult:
    def __init__(self, success: bool) -> None:
        self.success = success


class _FakeWebSocket:
    """Minimal stand-in for ``starlette.websockets.WebSocket``."""

    def __init__(self) -> None:
        self.close_calls: list[tuple[Optional[int], Optional[str]]] = []
        self.closed_event = asyncio.Event()

    async def close(
        self,
        code: Optional[int] = None,
        reason: Optional[str] = None,
    ) -> None:
        self.close_calls.append((code, reason))
        self.closed_event.set()


class PeriodicReauthTest(unittest.IsolatedAsyncioTestCase):
    async def test_closes_with_4001_when_validation_flips_to_failure(self) -> None:
        websocket = _FakeWebSocket()
        results = iter([_StubResult(True), _StubResult(False)])
        calls: list[Optional[str]] = []

        def validator(api_key: Optional[str]) -> _StubResult:
            calls.append(api_key)
            try:
                return next(results)
            except StopIteration:
                return _StubResult(False)

        task = start_periodic_reauth_task(
            websocket,
            "key-123",
            interval_seconds=0.05,
            validator=validator,
        )

        try:
            await asyncio.wait_for(websocket.closed_event.wait(), timeout=2.0)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self.assertEqual(websocket.close_calls, [(4001, "auth_invalid")])
        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(calls[0], "key-123")

    async def test_transient_exception_does_not_close_socket(self) -> None:
        websocket = _FakeWebSocket()
        call_log: list[str] = []

        def validator(api_key: Optional[str]) -> _StubResult:
            call_log.append("call")
            if len(call_log) == 1:
                raise RuntimeError("transient failure")
            # subsequent ticks: still valid, loop should keep going
            return _StubResult(True)

        task = start_periodic_reauth_task(
            websocket,
            "key-abc",
            interval_seconds=0.05,
            validator=validator,
        )

        # Give the loop a few ticks (transient error + at least one healthy call).
        await asyncio.sleep(0.25)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        self.assertEqual(websocket.close_calls, [])
        self.assertGreaterEqual(len(call_log), 2)

    async def test_cancellation_does_not_close_socket(self) -> None:
        websocket = _FakeWebSocket()

        def validator(api_key: Optional[str]) -> _StubResult:
            return _StubResult(True)

        task = start_periodic_reauth_task(
            websocket,
            "key-xyz",
            interval_seconds=0.05,
            validator=validator,
        )
        # Cancel before any tick can complete.
        await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        self.assertEqual(websocket.close_calls, [])


if __name__ == "__main__":
    unittest.main()
