import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store  # noqa: E402
from core import database  # noqa: E402
from routes import chat_ws  # noqa: E402
from workspaces import workspace_store  # noqa: E402


class _FakeWebSocket:
    def __init__(self):
        self.sent = []

    async def send_json(self, data):
        self.sent.append(data)


class ChatWsAgentIdTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "code_bridge_chat_ws_agent_id_test.db"
        agent_store._agent_store = None
        workspace_store._workspace_store = None

    def tearDown(self):
        agent_store._agent_store = None
        workspace_store._workspace_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    async def test_chat_turn_creates_run_for_adhoc_dev_agent(self):
        websocket = _FakeWebSocket()
        session = SimpleNamespace(
            provider_id="google",
            model="gemini-2.5-pro",
            session_id="native-session-1",
            project_path="/tmp/demo",
        )

        async def fake_stream(ws, session, project_name, user_message, **_):
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
        self.assertEqual(runs[0]["agent_id"], "agent_adhoc_dev")

    async def test_add_memory_from_event_noops_for_adhoc_dev_with_warning(self):
        store = agent_store.get_agent_store()
        before = store.count_memories("agent_adhoc_dev")

        with self.assertLogs("agent.agent_store", level="WARNING") as logs:
            memory = agent_store.add_memory_from_event(
                agent_id="agent_adhoc_dev",
                content="Do not save this.",
                source_run_id="run_adhoc",
            )

        after = store.count_memories("agent_adhoc_dev")
        self.assertIsNone(memory)
        self.assertEqual(before, 0)
        self.assertEqual(after, 0)
        self.assertTrue(
            any("pseudo-agent" in message for message in logs.output),
            logs.output,
        )


if __name__ == "__main__":
    unittest.main()
