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
from routes.system_inspect import router as system_inspect_router
from system_inspect_service import SystemInspectResult


class SystemInspectRoutesTest(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(system_inspect_router)
        app.dependency_overrides[verify_api_key] = lambda: True
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

    def test_list_system_directories_error_returns_status(self):
        with patch(
            "routes.system_inspect.list_system_directories_for_current_server",
            return_value=SystemInspectResult(
                success=False,
                status_code=400,
                payload={"error": "Invalid path"},
            ),
        ):
            response = self.client.get("/api/system/directories", params={"path": "relative"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json().get("error"), "Invalid path")

    def test_list_project_candidates_success(self):
        with patch(
            "routes.system_inspect.list_project_candidates_for_current_server",
            return_value=SystemInspectResult(
                success=True,
                status_code=200,
                payload={
                    "root_path": "/tmp/root",
                    "excluded_dirs": [],
                    "candidates": [{"path": "/tmp/root/demo", "registered": False}],
                    "count": 1,
                },
            ),
        ):
            response = self.client.get(
                "/api/system/project-candidates",
                params={"root_path": "/tmp/root", "max_depth": 1},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json().get("count"), 1)

    def test_get_system_usage_success(self):
        with patch(
            "routes.system_inspect.get_system_usage_for_current_server",
            new=AsyncMock(
                return_value=SystemInspectResult(
                    success=True,
                    status_code=200,
                    payload={"total_spent": 0.0},
                )
            ),
        ):
            response = self.client.get("/api/system/usage")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"total_spent": 0.0})


if __name__ == "__main__":
    unittest.main()
