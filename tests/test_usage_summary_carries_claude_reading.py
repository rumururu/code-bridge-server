"""The settings screen's summary has to carry Claude's own usage reading.

`GET /api/usage/summary` returned cost and budget only; the account's real
weekly percentage was merged in on the chat socket alone. The phone's settings
row therefore never had a real number, and showed cost / default budget.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from core import database  # noqa: E402
from routes import usage  # noqa: E402
from routes.deps import verify_api_key  # noqa: E402

SNAPSHOT = {"claude_usage_supported": True, "claude_usage_percent": 51.0, "claude_usage_error": None}


class UsageSummaryClaudeReadingTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._original = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "usage.db"
        self.addCleanup(lambda: setattr(database, "DB_PATH", self._original))
        database.init_db()
        # Inject the config rather than read whatever config.yaml the test
        # sandbox happens to hold: one generated from the old template still
        # says weekly_budget_usd: 100.0.
        config = patch.object(
            usage, "get_config",
            return_value=SimpleNamespace(weekly_budget_usd=0.0, usage_window_days=7),
        )
        config.start()
        self.addCleanup(config.stop)
        app = FastAPI()
        app.include_router(usage.router)
        app.dependency_overrides[verify_api_key] = lambda: "k"
        self.client = TestClient(app)

    def test_unfiltered_summary_carries_the_weekly_reading(self):
        with patch.object(usage, "fetch_claude_usage_snapshot", AsyncMock(return_value=SNAPSHOT)):
            body = self.client.get("/api/usage/summary").json()
        self.assertEqual(body["display_usage_percent"], 51.0)
        self.assertEqual(body["display_usage_source"], "claude_usage")
        # No budget is configured by default, so there is no cost ratio either.
        self.assertFalse(body["has_budget"])
        self.assertIsNone(body["usage_percent"])

    def test_a_failed_reading_is_reported_as_unavailable_not_invented(self):
        failed = {"claude_usage_supported": None, "claude_usage_percent": None, "claude_usage_error": "timeout"}
        with patch.object(usage, "fetch_claude_usage_snapshot", AsyncMock(return_value=failed)):
            body = self.client.get("/api/usage/summary").json()
        self.assertIsNone(body["display_usage_percent"])
        self.assertEqual(body["display_usage_source"], "unavailable")

    def test_a_filtered_slice_does_not_probe_the_cli(self):
        probe = AsyncMock(return_value=SNAPSHOT)
        with patch.object(usage, "fetch_claude_usage_snapshot", probe):
            body = self.client.get("/api/usage/summary", params={"task_id": "task_x"}).json()
        probe.assert_not_called()
        self.assertNotIn("display_usage_percent", body)


if __name__ == "__main__":
    unittest.main()
