import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import routes.chat_ws as chat_ws
from chat_ws_service import ChatWebSocketMessageResult


class _DummyWebSocket:
    def __init__(self):
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)


class ChatWsHandlersTest(unittest.IsolatedAsyncioTestCase):
    async def test_dispatch_ping_sends_pong(self):
        websocket = _DummyWebSocket()

        await chat_ws._dispatch_chat_message(
            websocket,
            session=object(),
            project_name="demo",
            message={"type": "ping"},
            local_port=8080,
        )

        self.assertEqual(websocket.sent[-1], {"type": "pong"})

    async def test_dispatch_unknown_type_sends_error(self):
        websocket = _DummyWebSocket()

        await chat_ws._dispatch_chat_message(
            websocket,
            session=object(),
            project_name="demo",
            message={"type": "unknown"},
            local_port=8080,
        )

        self.assertEqual(websocket.sent[-1]["type"], "error")
        self.assertIn("Unknown message type", websocket.sent[-1]["message"])

    async def test_dispatch_routes_to_message_handler(self):
        websocket = _DummyWebSocket()
        session = object()
        message = {"type": "message", "content": "hi"}

        with patch.object(chat_ws, "_handle_user_message", new=AsyncMock()) as mock_handler:
            await chat_ws._dispatch_chat_message(
                websocket,
                session=session,
                project_name="demo",
                message=message,
                local_port=8080,
            )

            mock_handler.assert_awaited_once_with(websocket, session, "demo", message)

    async def test_firebase_auth_missing_id_token_sends_error(self):
        websocket = _DummyWebSocket()

        await chat_ws._handle_firebase_auth_message(
            websocket,
            message={"type": "firebase_auth"},
            local_port=8080,
        )

        self.assertEqual(websocket.sent[-1]["type"], "error")
        self.assertIn("id_token", websocket.sent[-1]["message"])

    async def test_firebase_auth_handler_relays_service_payload(self):
        websocket = _DummyWebSocket()

        with patch(
            "routes.chat_ws.process_firebase_auth_message_for_current_server",
            new=AsyncMock(
                return_value=ChatWebSocketMessageResult(
                    payload={"type": "status", "message": "Firebase authenticated"}
                )
            ),
        ):
            await chat_ws._handle_firebase_auth_message(
                websocket,
                message={"type": "firebase_auth"},
                local_port=8080,
            )

        self.assertEqual(websocket.sent[-1], {"type": "status", "message": "Firebase authenticated"})


if __name__ == "__main__":
    unittest.main()
