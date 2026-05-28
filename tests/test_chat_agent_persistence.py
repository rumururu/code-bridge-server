import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store
from core import database
from routes import chat_ws
from workspaces import workspace_store


class _FakeWebSocket:
    def __init__(self):
        self.sent = []

    async def send_json(self, data):
        self.sent.append(data)


class ChatAgentPersistenceTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "code_bridge_chat_agent_test.db"
        agent_store._agent_store = None
        workspace_store._workspace_store = None

    def tearDown(self):
        agent_store._agent_store = None
        workspace_store._workspace_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    async def test_user_message_creates_durable_agent_run(self):
        websocket = _FakeWebSocket()
        session = SimpleNamespace(
            provider_id="google",
            model="gemini-2.5-pro",
            session_id="native-session-1",
            project_path="/tmp/demo",
        )

        async def fake_stream(ws, session, project_name, user_message, **_):
            await ws.send_json(
                {
                    "type": "provider_event",
                    "provider_id": "google",
                    "normalized": {"type": "assistant", "text": "done"},
                }
            )
            await ws.send_json({"type": "complete", "content": "done"})
            return True

        with patch("routes.chat_ws.stream_claude_turn", new=AsyncMock(side_effect=fake_stream)):
            await chat_ws._handle_user_message(
                websocket,
                session,
                "demo",
                {"type": "message", "content": "Build a login screen"},
            )

        store = agent_store.get_agent_store()
        runs = store.list_runs(project_name="demo")
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["status"], "completed")
        self.assertTrue(runs[0]["workspace_id"].startswith("wsp_"))
        self.assertEqual(runs[0]["provider_id"], "google")
        self.assertEqual(runs[0]["model"], "gemini-2.5-pro")
        self.assertEqual(runs[0]["native_session_id"], "native-session-1")

        messages = store.list_messages(runs[0]["id"])
        self.assertEqual(messages[0]["content"], "Build a login screen")

        workspaces = workspace_store.get_workspace_store().list_workspaces(project_name="demo")
        self.assertEqual(workspaces[0]["id"], runs[0]["workspace_id"])
        self.assertEqual(workspaces[0]["permissions"]["roots"], ["/tmp/demo"])

        events = store.list_events(runs[0]["id"])
        self.assertEqual(
            [event["event_type"] for event in events],
            ["user_message", "provider_event", "complete"],
        )
        self.assertEqual(websocket.sent[-1], {"type": "complete", "content": "done"})


if __name__ == "__main__":
    unittest.main()
