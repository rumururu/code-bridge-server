import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store  # noqa: E402
from agent.browser_action_adapter import (  # noqa: E402
    PLAYWRIGHT_CHROMIUM_INSTALL_COMMAND,
    get_browser_runtime_readiness,
)
from core import database  # noqa: E402
from routes import agents  # noqa: E402
from routes.deps import verify_api_key  # noqa: E402


class BrowserRuntimeReadinessTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "browser_runtime_readiness.db"
        agent_store._agent_store = None

        app = FastAPI()
        app.include_router(agents.router)
        app.dependency_overrides[verify_api_key] = lambda: "test-api-key"
        self.client = TestClient(app)

    def tearDown(self):
        agent_store._agent_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def test_readiness_contract_reports_boolean_fields(self):
        readiness = asyncio.run(get_browser_runtime_readiness())

        self.assertIsInstance(readiness["ready"], bool)
        self.assertIsInstance(readiness["playwright_python"], bool)
        self.assertIsInstance(readiness["chromium_executable"], bool)
        self.assertEqual(
            readiness["install_command"],
            PLAYWRIGHT_CHROMIUM_INSTALL_COMMAND,
        )
        self.assertIn("message", readiness)
        self.assertIn("diagnostics", readiness)

    def test_readiness_route_returns_runtime_diagnostics(self):
        fake_readiness = {
            "ready": False,
            "playwright_python": True,
            "chromium_executable": False,
            "chromium_executable_path": "/missing/chromium",
            "install_command": PLAYWRIGHT_CHROMIUM_INSTALL_COMMAND,
            "message": "Playwright Chromium executable is missing.",
            "diagnostics": [
                {
                    "code": "chromium_executable_missing",
                    "message": "Install Chromium.",
                }
            ],
        }

        with patch(
            "routes.agents.get_browser_runtime_readiness",
            AsyncMock(return_value=fake_readiness),
        ):
            response = self.client.get("/api/agent/browser-runtime/readiness")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["ready"])
        self.assertTrue(payload["playwright_python"])
        self.assertFalse(payload["chromium_executable"])
        self.assertEqual(payload["install_command"], PLAYWRIGHT_CHROMIUM_INSTALL_COMMAND)
        self.assertEqual(payload["diagnostics"][0]["code"], "chromium_executable_missing")


if __name__ == "__main__":
    unittest.main()
