import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store
from approvals import approval_store
from approvals.approval_service import decide_approval
from audit import audit_store
from core import database
from policy import policy_store
from projects.project_action_service import ProjectRegistryResult
from routes.deps import verify_api_key
from routes.projects import router as projects_router


class ProjectsRoutesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "code_bridge_projects_test.db"
        agent_store._agent_store = None
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None

        app = FastAPI()
        app.include_router(projects_router)
        app.dependency_overrides[verify_api_key] = lambda: True
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        agent_store._agent_store = None
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def _write_png(self, filename: str, color: tuple[int, int, int]) -> str:
        path = Path(self._tmp.name) / filename
        Image.new("RGB", (2, 2), color).save(path)
        return str(path)

    def test_start_dev_server_failure_returns_400(self):
        with patch(
            "routes.projects.start_project_dev_server_for_current_server",
            new=AsyncMock(return_value={"success": False, "error": "cannot start"}),
        ):
            response = self.client.post("/api/projects/demo/start")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json().get("error"), "cannot start")

    def test_start_dev_server_records_agent_artifact(self):
        run = agent_store.get_agent_store().create_run(project_name="demo", title="Start server")
        with patch(
            "routes.projects.start_project_dev_server_for_current_server",
            new=AsyncMock(return_value={"success": True, "status": "started"}),
        ):
            response = self.client.post(
                "/api/projects/demo/start",
                params={"run_id": run["id"]},
            )

        self.assertEqual(response.status_code, 200)
        artifacts = agent_store.get_agent_store().list_artifacts(run["id"])
        self.assertEqual(artifacts[0]["kind"], "process_devserver")
        self.assertEqual(artifacts[0]["metadata"]["details"]["action"], "start")

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

    def test_run_project_on_device_can_require_approval(self):
        run = agent_store.get_agent_store().create_run(project_name="demo", title="Run device")
        run_mock = AsyncMock(return_value={"success": True, "device_id": "emulator-5554"})
        with patch(
            "routes.projects.run_project_on_device_for_current_server",
            new=run_mock,
        ):
            blocked = self.client.post(
                "/api/projects/demo/run-device?require_approval=true",
                json={"device_id": "emulator-5554", "run_id": run["id"]},
            )

            self.assertEqual(blocked.status_code, 409)
            run_mock.assert_not_awaited()
            approval = approval_store.get_approval_store().list_pending(run_id=run["id"])[0]
            self.assertEqual(approval["operation"], "device.control")

            decide_approval(
                approval["id"],
                decision="approve_once",
                reason="test approval",
            )
            allowed = self.client.post(
                "/api/projects/demo/run-device?require_approval=true",
                json={
                    "device_id": "emulator-5554",
                    "run_id": run["id"],
                    "approval_id": approval["id"],
                },
            )

        self.assertEqual(allowed.status_code, 200)
        run_mock.assert_awaited_once_with("demo", "emulator-5554")

    def test_run_project_on_device_records_screenshot_artifact(self):
        run = agent_store.get_agent_store().create_run(project_name="demo", title="Run device")
        with patch(
            "routes.projects.run_project_on_device_for_current_server",
            new=AsyncMock(
                return_value={
                    "success": True,
                    "device_id": "emulator-5554",
                    "log_path": "/tmp/code_bridge_device_run.log",
                    "screenshot_path": "/tmp/code_bridge_flutter_demo.png",
                }
            ),
        ):
            response = self.client.post(
                "/api/projects/demo/run-device",
                json={
                    "device_id": "emulator-5554",
                    "run_id": run["id"],
                },
            )

        self.assertEqual(response.status_code, 200)
        artifacts = agent_store.get_agent_store().list_artifacts(run["id"])
        screenshot_artifacts = [
            artifact for artifact in artifacts if artifact["kind"] == "device_screenshot"
        ]
        self.assertEqual(len(screenshot_artifacts), 1)
        self.assertEqual(
            screenshot_artifacts[0]["path"],
            "/tmp/code_bridge_flutter_demo.png",
        )

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

    def test_open_web_preview_records_screenshot_artifact(self):
        run = agent_store.get_agent_store().create_run(project_name="demo", title="Preview")
        with patch(
            "routes.projects.open_web_preview_on_device_for_current_server",
            new=AsyncMock(
                return_value={
                    "success": True,
                    "device_id": "emulator-5554",
                    "preview_url": "http://localhost:3000",
                    "screenshot_path": "/tmp/code_bridge_preview_demo.png",
                }
            ),
        ):
            response = self.client.post(
                "/api/projects/demo/open-web-preview",
                json={
                    "device_id": "emulator-5554",
                    "run_id": run["id"],
                },
            )

        self.assertEqual(response.status_code, 200)
        artifacts = agent_store.get_agent_store().list_artifacts(run["id"])
        screenshot_artifacts = [
            artifact for artifact in artifacts if artifact["kind"] == "device_screenshot"
        ]
        self.assertEqual(len(screenshot_artifacts), 1)
        self.assertEqual(
            screenshot_artifacts[0]["path"],
            "/tmp/code_bridge_preview_demo.png",
        )

    def test_open_web_preview_records_visual_regression_artifact(self):
        run = agent_store.get_agent_store().create_run(project_name="demo", title="Preview")
        baseline_path = self._write_png("baseline.png", (0, 0, 0))
        current_path = self._write_png("current.png", (255, 0, 0))
        agent_store.get_agent_store().add_artifact(
            run_id=run["id"],
            kind="device_screenshot",
            path=baseline_path,
            mime_type="image/png",
            metadata={"project_name": "demo"},
        )
        with patch(
            "routes.projects.open_web_preview_on_device_for_current_server",
            new=AsyncMock(
                return_value={
                    "success": True,
                    "device_id": "emulator-5554",
                    "preview_url": "http://localhost:3000",
                    "screenshot_path": current_path,
                }
            ),
        ):
            response = self.client.post(
                "/api/projects/demo/open-web-preview",
                json={
                    "device_id": "emulator-5554",
                    "run_id": run["id"],
                },
            )

        self.assertEqual(response.status_code, 200)
        artifacts = agent_store.get_agent_store().list_artifacts(run["id"])
        visual_artifacts = [
            artifact for artifact in artifacts if artifact["kind"] == "visual_regression"
        ]
        self.assertEqual(len(visual_artifacts), 1)
        metadata = visual_artifacts[0]["metadata"]
        self.assertEqual(metadata["status"], "compared")
        self.assertEqual(metadata["baseline_path"], baseline_path)
        self.assertEqual(metadata["current_path"], current_path)
        self.assertTrue(metadata["changed"])
        self.assertGreater(metadata["diff_ratio"], 0)

    def test_open_web_preview_records_screenshot_error_artifact(self):
        run = agent_store.get_agent_store().create_run(project_name="demo", title="Preview")
        with patch(
            "routes.projects.open_web_preview_on_device_for_current_server",
            new=AsyncMock(
                return_value={
                    "success": True,
                    "device_id": "emulator-5554",
                    "preview_url": "http://localhost:3000",
                    "screenshot_path": None,
                    "screenshot_error": "adb screencap timed out",
                }
            ),
        ):
            response = self.client.post(
                "/api/projects/demo/open-web-preview",
                json={
                    "device_id": "emulator-5554",
                    "run_id": run["id"],
                },
            )

        self.assertEqual(response.status_code, 200)
        artifacts = agent_store.get_agent_store().list_artifacts(run["id"])
        error_artifacts = [
            artifact for artifact in artifacts if artifact["kind"] == "device_screenshot_error"
        ]
        self.assertEqual(len(error_artifacts), 1)
        self.assertEqual(error_artifacts[0]["metadata"]["error"], "adb screencap timed out")

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

    def test_create_project_folder_success_returns_created_project(self):
        with patch(
            "routes.projects.create_project_folder_for_current_server",
            return_value=ProjectRegistryResult(
                success=True,
                status_code=201,
                payload={"name": "demo", "path": "/tmp/demo", "type": "other"},
            ),
        ) as mock_create:
            response = self.client.post(
                "/api/projects/folder",
                json={"root_path": "/tmp", "folder_name": "demo"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json().get("name"), "demo")
        mock_create.assert_called_once()
        self.assertEqual(mock_create.call_args.kwargs["root_path"], "/tmp")
        self.assertEqual(mock_create.call_args.kwargs["folder_name"], "demo")

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
