import sys
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from core import database
from routes.deps import verify_api_key
from routes.usage import router as usage_router


class UsageRoutesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        self._original_usage_db = database._usage_db
        database.DB_PATH = Path(self._tmp.name) / "code_bridge_usage_test.db"
        database._usage_db = None

        app = FastAPI()
        app.include_router(usage_router)
        app.dependency_overrides[verify_api_key] = lambda: "test-api-key"
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        database._usage_db = self._original_usage_db
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def test_usage_summary_turns_and_breakdown(self):
        usage_db = database.get_usage_db()
        usage_db.record_event(
            source="chat",
            project_name="demo",
            workspace_id="wsp_1",
            task_id="task_1",
            run_id="run_1",
            provider_id="google",
            model="gemini",
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.25,
        )

        summary_response = self.client.get(
            "/api/usage/summary",
            params={"task_id": "task_1", "window_days": 7},
        )
        self.assertEqual(summary_response.status_code, 200)
        summary = summary_response.json()
        self.assertEqual(summary["turn_count"], 1)
        self.assertEqual(summary["input_tokens"], 10)

        turns_response = self.client.get("/api/usage/turns", params={"task_id": "task_1"})
        self.assertEqual(turns_response.status_code, 200)
        self.assertEqual(turns_response.json()["turns"][0]["run_id"], "run_1")

        breakdown_response = self.client.get(
            "/api/usage/breakdown",
            params={"group_by": "provider_id"},
        )
        self.assertEqual(breakdown_response.status_code, 200)
        self.assertEqual(breakdown_response.json()["items"][0]["key"], "google")


if __name__ == "__main__":
    unittest.main()
