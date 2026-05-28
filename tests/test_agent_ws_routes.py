import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store
from core import database
from routes import agent_ws


class AgentWsRoutesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "code_bridge_agent_ws_test.db"
        agent_store._agent_store = None

        app = FastAPI()
        app.include_router(agent_ws.router)
        self.client = TestClient(app)

    def tearDown(self):
        agent_store._agent_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def test_agent_run_timeline_streams_snapshot_and_events(self):
        store = agent_store.get_agent_store()
        run = store.create_run(project_name="demo", title="Timeline")
        event = store.append_event(
            run_id=run["id"],
            event_type="status",
            app_event={"stage": "created"},
        )

        with patch(
            "routes.agent_ws.validate_api_key_for_current_server",
            return_value=SimpleNamespace(success=True, api_key="test", error=None),
        ):
            with self.client.websocket_connect(f"/ws/agent/runs/{run['id']}") as ws:
                snapshot = ws.receive_json()
                streamed = ws.receive_json()

        self.assertEqual(snapshot["type"], "run_snapshot")
        self.assertEqual(snapshot["run"]["id"], run["id"])
        self.assertEqual(streamed["type"], "agent_event")
        self.assertEqual(streamed["event"]["id"], event["id"])


if __name__ == "__main__":
    unittest.main()
