import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store
from audit import audit_store
from core import database
from routes import app_builder
from routes.deps import verify_api_key
from workspaces import workspace_store


class AppBuilderRoutesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "apps"
        self.root.mkdir()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "code_bridge_app_builder_test.db"
        database._project_db = None
        agent_store._agent_store = None
        workspace_store._workspace_store = None
        audit_store._audit_store = None

        app = FastAPI()
        app.include_router(app_builder.router)
        app.dependency_overrides[verify_api_key] = lambda: "test-api-key"
        self.client = TestClient(app)

    def tearDown(self):
        database._project_db = None
        agent_store._agent_store = None
        workspace_store._workspace_store = None
        audit_store._audit_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def test_create_local_app_seeds_project_workspace_task_and_run(self):
        with patch("projects.project_action_service.validate_accessible_path", return_value=True):
            response = self.client.post(
                "/api/agent/apps",
                json={
                    "root_path": str(self.root),
                    "app_name": "Demo App",
                    "prompt": "Build a compact CRM dashboard.",
                    "provider_id": "google",
                    "model": "gemini-2.5-pro",
                },
            )

        self.assertEqual(response.status_code, 201)
        payload = response.json()
        project = payload["project"]
        workspace = payload["workspace"]
        task = payload["task"]
        tasks = payload["tasks"]
        run = payload["run"]

        project_path = Path(project["path"])
        self.assertEqual(project["name"], "demo-app")
        self.assertTrue((project_path / "package.json").exists())
        self.assertTrue((project_path / "codebridge.agent.json").exists())
        self.assertTrue((project_path / "docs" / "agent-brief.md").exists())
        self.assertTrue((project_path / "app" / "page.tsx").exists())
        self.assertEqual(workspace["project_name"], "demo-app")
        self.assertEqual(task["workspace_id"], workspace["id"])
        self.assertEqual(len(tasks), 3)
        self.assertEqual(tasks[0]["run_id"], run["id"])
        self.assertEqual(run["workspace_id"], workspace["id"])
        self.assertEqual(run["provider_id"], "google")
        self.assertIn("app/page.tsx", payload["files"])
        self.assertIn("codebridge.agent.json", payload["files"])

        artifacts = agent_store.get_agent_store().list_artifacts(run["id"])
        self.assertEqual(len(artifacts), len(payload["files"]))
        audit_events = audit_store.get_audit_store().list_events(project_name="demo-app")
        self.assertEqual(audit_events[0]["operation"], "app.create")

    def test_unsupported_template_returns_400(self):
        response = self.client.post(
            "/api/agent/apps",
            json={
                "root_path": str(self.root),
                "app_name": "Demo App",
                "prompt": "Build an app.",
                "template": "rails",
            },
        )

        self.assertEqual(response.status_code, 400)

    def test_create_vite_local_app_uses_react_project_seed(self):
        with patch("projects.project_action_service.validate_accessible_path", return_value=True):
            response = self.client.post(
                "/api/agent/apps",
                json={
                    "root_path": str(self.root),
                    "app_name": "Vite App",
                    "prompt": "Build a field ops dashboard.",
                    "template": "vite",
                },
            )

        self.assertEqual(response.status_code, 201)
        payload = response.json()
        project = payload["project"]
        project_path = Path(project["path"])
        self.assertEqual(project["type"], "react")
        self.assertTrue((project_path / "vite.config.ts").exists())
        self.assertTrue((project_path / "src" / "main.tsx").exists())
        self.assertIn("src/main.tsx", payload["files"])
        self.assertEqual(payload["tasks"][0]["run_id"], payload["run"]["id"])

    def test_create_flutter_local_app_uses_flutter_project_seed(self):
        with patch("projects.project_action_service.validate_accessible_path", return_value=True), \
             patch("app_builder.app_builder_service.shutil.which", return_value=None):
            response = self.client.post(
                "/api/agent/apps",
                json={
                    "root_path": str(self.root),
                    "app_name": "Mobile App",
                    "prompt": "Build a field service mobile app.",
                    "template": "flutter",
                },
            )

        self.assertEqual(response.status_code, 201)
        payload = response.json()
        project = payload["project"]
        project_path = Path(project["path"])
        self.assertEqual(project["type"], "flutter")
        self.assertTrue((project_path / "pubspec.yaml").exists())
        self.assertTrue((project_path / "lib" / "main.dart").exists())
        self.assertTrue((project_path / "test" / "widget_test.dart").exists())
        self.assertIn("lib/main.dart", payload["files"])
        self.assertEqual(payload["platform_scaffold"]["status"], "skipped")
        manifest = (project_path / "codebridge.agent.json").read_text(encoding="utf-8")
        self.assertIn("flutter create --platforms=android,ios .", manifest)
        artifacts = agent_store.get_agent_store().list_artifacts(payload["run"]["id"])
        platform_artifacts = [
            artifact for artifact in artifacts if artifact["kind"] == "platform_scaffold"
        ]
        self.assertEqual(len(platform_artifacts), 1)
        self.assertFalse(platform_artifacts[0]["metadata"]["available"])

    def test_create_flutter_local_app_materializes_platforms_when_flutter_exists(self):
        def fake_run(command, cwd, **kwargs):
            project_path = Path(cwd)
            (project_path / "android").mkdir()
            (project_path / "ios").mkdir()
            return subprocess.CompletedProcess(command, 0, "created\n", "")

        with patch("projects.project_action_service.validate_accessible_path", return_value=True), \
             patch("app_builder.app_builder_service.shutil.which", return_value="/usr/bin/flutter"), \
             patch("app_builder.app_builder_service.subprocess.run", side_effect=fake_run) as run_mock:
            response = self.client.post(
                "/api/agent/apps",
                json={
                    "root_path": str(self.root),
                    "app_name": "99 Mobile App",
                    "prompt": "Build a field service mobile app.",
                    "template": "flutter",
                },
            )

        self.assertEqual(response.status_code, 201)
        payload = response.json()
        project_path = Path(payload["project"]["path"])
        self.assertTrue((project_path / "android").is_dir())
        self.assertTrue((project_path / "ios").is_dir())
        self.assertEqual(payload["platform_scaffold"]["status"], "created")
        self.assertEqual(payload["platform_scaffold"]["platforms"], ["android", "ios"])
        run_mock.assert_called_once()
        command = run_mock.call_args.args[0]
        self.assertIn("--no-pub", command)
        self.assertIn("code_bridge_99_mobile_app", command)


if __name__ == "__main__":
    unittest.main()
