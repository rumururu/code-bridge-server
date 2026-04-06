import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from routes.debug import router as debug_router
from routes.health import router as health_router


class HealthDebugRoutesTest(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(health_router)
        app.include_router(debug_router)
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

    def test_health_check_delegates_to_service(self):
        with patch(
            "routes.health.get_health_status_for_current_server",
            return_value={"status": "ok", "service": "claude-bridge"},
        ):
            response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json().get("status"), "ok")

    def test_debug_port_delegates_to_service(self):
        with patch(
            "routes.debug.get_debug_port_snapshot_for_current_server",
            return_value={"config_port": 8080, "env_port": "7000", "runtime_port": 9090},
        ):
            response = self.client.get("/api/debug/port")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json().get("runtime_port"), 9090)


if __name__ == "__main__":
    unittest.main()
