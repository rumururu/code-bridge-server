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
from agent import browser_action_adapter  # noqa: E402
from agent.browser_action_adapter import (  # noqa: E402
    BROWSER_READINESS_CACHE_TTL_SECONDS,
    PLAYWRIGHT_CHROMIUM_INSTALL_COMMAND,
    get_browser_runtime_readiness,
    get_cached_browser_readiness_sync,
    reset_browser_readiness_cache,
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
        reset_browser_readiness_cache()
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


class BrowserReadinessCacheTest(unittest.TestCase):
    """The probe launches the Playwright driver, so callers must not pay per call."""

    def setUp(self):
        reset_browser_readiness_cache()

    def tearDown(self):
        reset_browser_readiness_cache()

    @staticmethod
    def _fake_readiness(marker="first"):
        return {
            "ready": False,
            "playwright_python": False,
            "chromium_executable": False,
            "chromium_executable_path": None,
            "install_command": PLAYWRIGHT_CHROMIUM_INSTALL_COMMAND,
            "message": marker,
            "diagnostics": [],
        }

    def test_second_call_inside_ttl_does_not_reprobe(self):
        probe = AsyncMock(return_value=self._fake_readiness())

        with patch.object(browser_action_adapter, "_probe_browser_runtime_readiness", probe):
            first = asyncio.run(get_browser_runtime_readiness())
            second = asyncio.run(get_browser_runtime_readiness())

        self.assertEqual(probe.await_count, 1)
        self.assertEqual(first["message"], "first")
        self.assertEqual(second["message"], "first")

    def test_force_refresh_reprobes(self):
        probe = AsyncMock(
            side_effect=[self._fake_readiness("first"), self._fake_readiness("second")]
        )

        with patch.object(browser_action_adapter, "_probe_browser_runtime_readiness", probe):
            asyncio.run(get_browser_runtime_readiness())
            refreshed = asyncio.run(get_browser_runtime_readiness(force_refresh=True))

        self.assertEqual(probe.await_count, 2)
        self.assertEqual(refreshed["message"], "second")

    def test_expired_entry_reprobes(self):
        probe = AsyncMock(
            side_effect=[self._fake_readiness("first"), self._fake_readiness("second")]
        )

        with patch.object(browser_action_adapter, "_probe_browser_runtime_readiness", probe):
            asyncio.run(get_browser_runtime_readiness())
            browser_action_adapter._readiness_cached_at -= (
                BROWSER_READINESS_CACHE_TTL_SECONDS + 1
            )
            again = asyncio.run(get_browser_runtime_readiness())

        self.assertEqual(probe.await_count, 2)
        self.assertEqual(again["message"], "second")

    def test_sync_snapshot_is_none_until_probed_and_after_expiry(self):
        self.assertIsNone(get_cached_browser_readiness_sync())

        probe = AsyncMock(return_value=self._fake_readiness())
        with patch.object(browser_action_adapter, "_probe_browser_runtime_readiness", probe):
            asyncio.run(get_browser_runtime_readiness())

        snapshot = get_cached_browser_readiness_sync()
        self.assertIsNotNone(snapshot)
        self.assertFalse(snapshot["ready"])

        browser_action_adapter._readiness_cached_at -= (
            BROWSER_READINESS_CACHE_TTL_SECONDS + 1
        )
        self.assertIsNone(get_cached_browser_readiness_sync())

    def test_caller_mutation_does_not_poison_the_cache(self):
        probe = AsyncMock(return_value=self._fake_readiness())

        with patch.object(browser_action_adapter, "_probe_browser_runtime_readiness", probe):
            first = asyncio.run(get_browser_runtime_readiness())
            first["ready"] = True
            second = asyncio.run(get_browser_runtime_readiness())

        self.assertFalse(second["ready"])


class InstallCommandPathTest(unittest.TestCase):
    def test_install_command_points_at_the_running_interpreter(self):
        self.assertEqual(
            PLAYWRIGHT_CHROMIUM_INSTALL_COMMAND,
            f"{sys.executable} -m playwright install chromium",
        )
        # The old value was the fixed relative literal below, which resolved to
        # nothing on a real install (~/.code-bridge/venv) or the dev tree
        # (server/venv). What replaces it must be an absolute path to an
        # interpreter that exists.
        self.assertNotEqual(
            PLAYWRIGHT_CHROMIUM_INSTALL_COMMAND,
            "server/.venv/bin/python -m playwright install chromium",
        )
        interpreter = PLAYWRIGHT_CHROMIUM_INSTALL_COMMAND.split(" -m ")[0]
        self.assertTrue(Path(interpreter).is_absolute(), interpreter)
        self.assertTrue(Path(interpreter).is_file(), interpreter)


if __name__ == "__main__":
    unittest.main()
