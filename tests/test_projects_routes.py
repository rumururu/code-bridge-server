import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from projects.project_action_service import ProjectRegistryResult
from routes.deps import verify_api_key
from routes.projects import router as projects_router


class ProjectsRoutesTest(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(projects_router)
        app.dependency_overrides[verify_api_key] = lambda: True
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

    def test_start_dev_server_failure_returns_400(self):
        with patch(
            "routes.projects.start_project_dev_server_for_current_server",
            new=AsyncMock(return_value={"success": False, "error": "cannot start"}),
        ):
            response = self.client.post("/api/projects/demo/start")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json().get("error"), "cannot start")

    def test_run_project_on_device_failure_returns_400(self):
        with patch(
            "routes.projects.run_project_on_device_for_current_server",
            new=AsyncMock(return_value={"success": False, "error": "device offline"}),
        ):
            response = self.client.post(
                "/api/projects/demo/run-device",
                json={"device_id": "emulator-5554"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json().get("error"), "device offline")

    def test_open_web_preview_failure_returns_400(self):
        mock_open = AsyncMock(return_value={"success": False, "message": "emulator missing"})
        with patch(
            "routes.projects.open_web_preview_on_device_for_current_server",
            new=mock_open,
        ):
            response = self.client.post(
                "/api/projects/demo/open-web-preview",
                json={
                    "device_id": "avd:Pixel_8_API_35",
                    "width": 390,
                    "height": 844,
                    "density": 420,
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json().get("message"), "emulator missing")
        mock_open.assert_awaited_once()
        self.assertEqual(mock_open.await_args.kwargs["width"], 390)
        self.assertEqual(mock_open.await_args.kwargs["height"], 844)
        self.assertEqual(mock_open.await_args.kwargs["density"], 420)
        self.assertFalse(mock_open.await_args.kwargs["reset_to_default"])

    def test_create_project_validation_error_returns_status(self):
        with patch(
            "routes.projects.create_project_record_for_current_server",
            return_value=ProjectRegistryResult(
                success=False,
                status_code=422,
                payload={"error": "Invalid project path"},
            ),
        ):
            response = self.client.post(
                "/api/projects",
                json={"path": "/tmp/invalid", "type": "flutter"},
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json().get("error"), "Invalid project path")

    def test_import_projects_success_returns_summary(self):
        with patch(
            "routes.projects.import_project_records_for_current_server",
            return_value=ProjectRegistryResult(
                success=True,
                status_code=200,
                payload={
                    "created": [{"name": "demo"}],
                    "skipped": [],
                    "failed": [],
                    "summary": {"created": 1, "skipped": 0, "failed": 0, "requested": 1},
                },
            ),
        ):
            response = self.client.post(
                "/api/projects/import",
                json={"paths": ["/tmp/demo"]},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["summary"]["created"], 1)

    def test_delete_project_not_found_returns_404(self):
        with patch(
            "routes.projects.delete_project_record_for_current_server",
            new=AsyncMock(
                return_value=ProjectRegistryResult(
                    success=False,
                    status_code=404,
                    payload={"error": "Project demo not found"},
                )
            ),
        ):
            response = self.client.delete("/api/projects/demo")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json().get("error"), "Project demo not found")
