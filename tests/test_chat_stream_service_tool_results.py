import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from chat.chat_stream_service import TurnState, _handle_assistant_event


class _FakeWebSocket:
    def __init__(self):
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)


class ChatStreamServiceToolResultsTest(unittest.IsolatedAsyncioTestCase):
    async def test_assistant_tool_result_blocks_emit_provider_neutral_events(self):
        websocket = _FakeWebSocket()
        state = TurnState()

        await _handle_assistant_event(
            websocket,
            state,
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-1",
                            "content": "done",
                        }
                    ]
                },
            },
        )

        self.assertEqual(len(websocket.sent), 2)
        tool_result = websocket.sent[0]
        self.assertEqual(tool_result["type"], "tool_result")
        self.assertEqual(tool_result["tool_use_id"], "call-1")
        self.assertFalse(tool_result["is_error"])
        self.assertEqual(tool_result["content"], "done")
        self.assertEqual(tool_result["raw_event"]["type"], "assistant")

        app_event = websocket.sent[1]
        self.assertEqual(app_event["type"], "app_event")
        self.assertEqual(app_event["schema_version"], 1)
        self.assertEqual(app_event["event"], "tool.completed")
        self.assertEqual(app_event["title"], "tool done")
        self.assertEqual(app_event["level"], "info")
        self.assertEqual(app_event["detail"], "done")
        self.assertEqual(app_event["sequence"], 1)
        self.assertIn("timestamp", app_event)
        self.assertEqual(
            app_event["data"],
            {
                "tool_use_id": "call-1",
                "is_error": False,
            },
        )
        self.assertEqual(app_event["raw_event"]["type"], "assistant")


if __name__ == "__main__":
    unittest.main()
