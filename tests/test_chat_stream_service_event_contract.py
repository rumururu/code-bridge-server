import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from chat.chat_stream_service import stream_claude_turn


class _FakeWebSocket:
    def __init__(self):
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)


class _FakeSession:
    provider_id = "openai"
    session_id = "sess-start"
    is_running = False
    has_pending_permission_denials = False

    def __init__(self, events):
        self._events = events

    async def send_message(self, message, permission_mode=None):
        for event in self._events:
            yield event

    async def approve_pending_permissions_and_retry(self):
        for event in self._events:
            yield event

    async def deny_pending_permissions(self, message="Permission denied by user."):
        for event in self._events:
            yield event

    async def close(self):
        return None

    async def set_model(self, model):
        return None

    async def abort_current_turn(self):
        return False


def _patch_usage():
    usage_db = MagicMock()
    usage_db.get_weekly_summary.return_value = {"weekly": "summary"}
    return (
        patch("chat.chat_stream_service.get_usage_db", return_value=usage_db),
        patch(
            "chat.chat_stream_service.get_config",
            return_value=SimpleNamespace(weekly_budget_usd=25.0, usage_window_days=7),
        ),
        patch(
            "chat.chat_stream_service.fetch_claude_usage_snapshot",
            new=AsyncMock(return_value={"claude": "usage"}),
        ),
        patch(
            "chat.chat_stream_service.merge_usage_for_display",
            return_value={"usage": "merged"},
        ),
    )


class ChatStreamServiceEventContractTest(unittest.IsolatedAsyncioTestCase):
    async def test_success_turn_app_event_schema_order_and_raw_provider_events(self):
        websocket = _FakeWebSocket()
        events = [
            {
                "type": "assistant",
                "provider_id": "openai",
                "session_id": "sess-raw",
                "raw_event": {"type": "raw.tool_start", "session_id": "sess-raw"},
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "bash",
                            "input": {"cmd": "pwd"},
                        }
                    ]
                },
            },
            {
                "type": "assistant",
                "provider_id": "openai",
                "session_id": "sess-raw",
                "raw_event": {"type": "raw.tool_result", "session_id": "sess-raw"},
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-1",
                            "content": "ok",
                        }
                    ]
                },
            },
            {
                "type": "result",
                "provider_id": "openai",
                "session_id": "sess-raw",
                "raw_event": {"type": "raw.result", "session_id": "sess-raw"},
                "result": "done",
                "usage": {"input_tokens": 3, "output_tokens": 4},
                "total_cost_usd": 0.02,
            },
        ]
        session = _FakeSession(events)

        usage_patches = _patch_usage()
        with (
            patch("chat.chat_stream_service._build_marionette_context", return_value=None),
            usage_patches[0],
            usage_patches[1],
            usage_patches[2],
            usage_patches[3],
        ):
            completed = await stream_claude_turn(websocket, session, "demo", user_message="hi")

        self.assertTrue(completed)

        app_events = [payload for payload in websocket.sent if payload["type"] == "app_event"]
        self.assertEqual(
            [payload["event"] for payload in app_events],
            [
                "turn.started",
                "tool.started",
                "tool.completed",
                "turn.metrics",
                "turn.completed",
            ],
        )
        self.assertEqual([payload["sequence"] for payload in app_events], [1, 2, 3, 4, 5])
        turn_ids = {payload["turn_id"] for payload in app_events}
        self.assertEqual(len(turn_ids), 1)
        for payload in app_events:
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["provider_id"], "openai")
            self.assertEqual(payload["provider"], "openai")
            self.assertIn("timestamp", payload)
            self.assertIn("title", payload)

        self.assertEqual(app_events[1]["raw_event"]["type"], "raw.tool_start")
        self.assertEqual(app_events[2]["raw_event"]["type"], "raw.tool_result")
        self.assertEqual(app_events[3]["raw_event"]["type"], "raw.result")

        provider_events = [
            payload for payload in websocket.sent if payload["type"] == "provider_event"
        ]
        self.assertEqual(
            [payload["event"]["type"] for payload in provider_events],
            ["raw.tool_start", "raw.tool_result", "raw.result"],
        )
        self.assertEqual(provider_events[0]["normalized"]["type"], "assistant")
        self.assertEqual(provider_events[0]["session_id"], "sess-raw")

        tool_result = next(
            payload for payload in websocket.sent if payload["type"] == "tool_result"
        )
        self.assertEqual(tool_result["raw_event"]["type"], "raw.tool_result")
        metrics = next(payload for payload in websocket.sent if payload["type"] == "turn_metrics")
        self.assertEqual(metrics["raw_event"]["type"], "raw.result")

    async def test_failure_turn_app_event_schema_and_raw_error(self):
        websocket = _FakeWebSocket()
        session = _FakeSession(
            [
                {
                    "type": "error",
                    "session_id": "sess-error",
                    "raw_event": {"type": "raw.error", "session_id": "sess-error"},
                    "error": {"message": "provider failed"},
                }
            ]
        )

        with patch("chat.chat_stream_service._build_marionette_context", return_value=None):
            completed = await stream_claude_turn(websocket, session, "demo", user_message="hi")

        self.assertFalse(completed)
        app_events = [payload for payload in websocket.sent if payload["type"] == "app_event"]
        self.assertEqual(
            [payload["event"] for payload in app_events],
            ["turn.started", "turn.failed"],
        )
        self.assertEqual([payload["sequence"] for payload in app_events], [1, 2])
        self.assertEqual(app_events[1]["level"], "error")
        self.assertEqual(app_events[1]["raw_event"]["type"], "raw.error")

        error = next(payload for payload in websocket.sent if payload["type"] == "error")
        self.assertEqual(error["raw_event"]["type"], "raw.error")

    async def test_unknown_provider_event_has_provider_neutral_and_legacy_passthrough(self):
        websocket = _FakeWebSocket()
        session = _FakeSession(
            [
                {
                    "type": "provider_event",
                    "session_id": "sess-unknown",
                    "event": {"type": "raw.unknown", "value": 1},
                    "legacy_type": "codex_event",
                }
            ]
        )

        with patch("chat.chat_stream_service._build_marionette_context", return_value=None):
            completed = await stream_claude_turn(websocket, session, "demo", user_message="hi")

        self.assertFalse(completed)
        provider_event = next(
            payload for payload in websocket.sent if payload["type"] == "provider_event"
        )
        self.assertEqual(provider_event["event"], {"type": "raw.unknown", "value": 1})
        legacy_event = next(
            payload for payload in websocket.sent if payload["type"] == "claude_event"
        )
        self.assertEqual(legacy_event["event"], {"type": "raw.unknown", "value": 1})
        self.assertEqual(legacy_event["provider_id"], "openai")


if __name__ == "__main__":
    unittest.main()
