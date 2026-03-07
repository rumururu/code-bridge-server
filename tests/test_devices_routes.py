import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from routes.deps import verify_api_key
from routes.devices import router as devices_router


class DevicesRoutesTest(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(devices_router)
        app.dependency_overrides[verify_api_key] = lambda: True
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

    def test_start_scrcpy_success_returns_payload(self):
        with patch(
            "routes.devices.start_scrcpy_for_current_server",
            new=AsyncMock(return_value={"success": True, "url": "ws://localhost:3000"}),
        ):
            response = self.client.post("/api/scrcpy/start")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"success": True, "url": "ws://localhost:3000"})

    def test_start_scrcpy_failure_returns_400(self):
        with patch(
            "routes.devices.start_scrcpy_for_current_server",
            new=AsyncMock(return_value={"success": False, "error": "scrcpy failed"}),
        ):
            response = self.client.post("/api/scrcpy/start")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json().get("error"), "scrcpy failed")

    def test_stop_scrcpy_failure_returns_400(self):
        with patch(
            "routes.devices.stop_scrcpy_for_current_server",
            new=AsyncMock(return_value={"success": False, "error": "not running"}),
        ):
            response = self.client.post("/api/scrcpy/stop")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json().get("error"), "not running")
