import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from routes.deps import verify_api_key
from routes.system_settings import router as system_settings_router
from system_settings_service import SystemSettingsResult


class SystemSettingsRoutesTest(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(system_settings_router)
        app.dependency_overrides[verify_api_key] = lambda: True
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

    def test_get_heartbeat_settings_success(self):
        with patch(
            "routes.system_settings.get_heartbeat_settings_for_current_server",
            return_value=SystemSettingsResult(
                success=True,
                status_code=200,
                payload={"interval_minutes": 10, "min": 5, "max": 15},
            ),
        ):
            response = self.client.get("/api/system/heartbeat")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json().get("interval_minutes"), 10)

    def test_update_llm_selection_validation_error_returns_400(self):
        with patch(
            "routes.system_settings.update_llm_selection_for_current_server",
            return_value=SystemSettingsResult(
                success=False,
                status_code=400,
                payload={"error": "Unknown LLM provider"},
            ),
        ):
            response = self.client.put(
                "/api/system/llm/selection",
                json={"company_id": "unknown", "model": "x"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json().get("error"), "Unknown LLM provider")

    def test_update_codex_settings_success(self):
        with patch(
            "routes.system_settings.update_codex_settings_for_current_server",
            return_value=SystemSettingsResult(
                success=True,
                status_code=200,
                payload={"sandbox_mode": "workspace-write", "sandbox_modes": []},
            ),
        ):
            response = self.client.put(
                "/api/system/llm/codex/settings",
                json={"sandbox_mode": "workspace-write"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json().get("sandbox_mode"), "workspace-write")


if __name__ == "__main__":
    unittest.main()
