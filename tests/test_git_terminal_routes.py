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

from agent import agent_store
from approvals import approval_store
from approvals.approval_service import decide_approval
from policy import policy_store
from core.base_result import BaseRouteResult
from core import database
from audit import audit_store
from routes import git, terminal
from routes.deps import verify_api_key


class GitTerminalRoutesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "code_bridge_git_terminal_test.db"
        database._project_db = None
        agent_store._agent_store = None
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None

        app = FastAPI()
        app.include_router(git.router)
        app.include_router(terminal.router)
        app.dependency_overrides[verify_api_key] = lambda: "test-api-key"
        self.client = TestClient(app)

    def tearDown(self):
        agent_store._agent_store = None
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None
        database._project_db = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def test_git_status_route_delegates(self):
        with patch(
            "routes.git.get_git_status_for_current_server",
            new=AsyncMock(
                return_value=BaseRouteResult.ok(
                    {
                        "branch": "main",
                        "is_clean": True,
                        "ahead": 0,
                        "behind": 0,
                        "staged": [],
                        "modified": [],
                        "untracked": [],
                        "deleted": [],
                    }
                )
            ),
        ) as mock_status:
            response = self.client.get("/api/projects/demo/git/status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["branch"], "main")
        mock_status.assert_awaited_once_with("demo")

    def test_git_commit_route_returns_output(self):
        with patch(
            "routes.git.commit_git_changes_for_current_server",
            new=AsyncMock(
                return_value=BaseRouteResult.ok(
                    {
                        "success": True,
                        "output": "[main abc123] update",
                        "stdout": "[main abc123] update",
                        "stderr": "",
                        "exit_code": 0,
                        "error": None,
                    }
                )
            ),
        ) as mock_commit:
            response = self.client.post(
                "/api/projects/demo/git/commit",
                json={"message": "update"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["output"], "[main abc123] update")
        mock_commit.assert_awaited_once_with("demo", "update")
        audit_events = audit_store.get_audit_store().list_events(project_name="demo")
        self.assertEqual(audit_events[0]["operation"], "git.commit")
        self.assertEqual(audit_events[0]["decision"], "executed")

    def test_terminal_execute_route_returns_command_result(self):
        run = agent_store.get_agent_store().create_run(project_name="demo", title="Terminal run")
        with patch(
            "routes.terminal.execute_terminal_command_for_current_server",
            new=AsyncMock(
                return_value=BaseRouteResult.ok(
                    {
                        "stdout": "hello\n",
                        "stderr": "",
                        "exit_code": 0,
                        "error": None,
                        "timed_out": False,
                    }
                )
            ),
        ) as mock_execute:
            response = self.client.post(
                "/api/projects/demo/terminal/execute",
                json={"command": "echo hello", "timeout": 5, "run_id": run["id"]},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["stdout"], "hello\n")
        mock_execute.assert_awaited_once_with("demo", command="echo hello", timeout=5)
        audit_events = audit_store.get_audit_store().list_events(project_name="demo")
        self.assertEqual(audit_events[0]["operation"], "process.terminal")
        self.assertEqual(audit_events[0]["decision"], "executed")
        artifacts = agent_store.get_agent_store().list_artifacts(run["id"])
        self.assertEqual(artifacts[0]["kind"], "process_terminal")
        self.assertEqual(artifacts[0]["metadata"]["result"]["stdout"], "hello\n")
        log_artifacts = [
            artifact for artifact in artifacts if artifact["kind"] == "terminal_log"
        ]
        self.assertEqual(len(log_artifacts), 1)
        log_text = Path(log_artifacts[0]["path"]).read_text(encoding="utf-8")
        self.assertIn("[stdout]", log_text)
        self.assertIn("hello", log_text)

    def test_terminal_git_diff_records_source_diff_artifact(self):
        run = agent_store.get_agent_store().create_run(project_name="demo", title="Diff run")
        diff_text = "diff --git a/lib/main.dart b/lib/main.dart\n+hello\n"
        with patch(
            "routes.terminal.execute_terminal_command_for_current_server",
            new=AsyncMock(
                return_value=BaseRouteResult.ok(
                    {
                        "stdout": diff_text,
                        "stderr": "",
                        "exit_code": 0,
                        "error": None,
                        "timed_out": False,
                    }
                )
            ),
        ):
            response = self.client.post(
                "/api/projects/demo/terminal/execute",
                json={
                    "command": "git diff -- lib/main.dart",
                    "timeout": 5,
                    "run_id": run["id"],
                },
            )

        self.assertEqual(response.status_code, 200)
        artifacts = agent_store.get_agent_store().list_artifacts(run["id"])
        diff_artifacts = [
            artifact for artifact in artifacts if artifact["kind"] == "source_diff"
        ]
        self.assertEqual(len(diff_artifacts), 1)
        stored_diff = Path(diff_artifacts[0]["path"]).read_text(encoding="utf-8")
        self.assertEqual(stored_diff, diff_text)

    def test_terminal_build_records_build_output_artifact(self):
        project_root = Path(self._tmp.name) / "demo"
        output_path = project_root / "build" / "app" / "outputs" / "flutter-apk" / "app-release.apk"
        output_path.parent.mkdir(parents=True)
        output_path.write_bytes(b"apk")
        database.get_project_db().create(
            {
                "name": "demo",
                "path": str(project_root),
                "type": "flutter",
            }
        )
        run = agent_store.get_agent_store().create_run(project_name="demo", title="Build run")
        with patch(
            "routes.terminal.execute_terminal_command_for_current_server",
            new=AsyncMock(
                return_value=BaseRouteResult.ok(
                    {
                        "stdout": "built\n",
                        "stderr": "",
                        "exit_code": 0,
                        "error": None,
                        "timed_out": False,
                    }
                )
            ),
        ):
            response = self.client.post(
                "/api/projects/demo/terminal/execute",
                json={
                    "command": "flutter build apk",
                    "timeout": 120,
                    "run_id": run["id"],
                },
            )

        self.assertEqual(response.status_code, 200)
        artifacts = agent_store.get_agent_store().list_artifacts(run["id"])
        build_artifacts = [
            artifact for artifact in artifacts if artifact["kind"] == "build_output"
        ]
        self.assertEqual(len(build_artifacts), 2)
        self.assertEqual(build_artifacts[0]["path"], str(output_path))
        self.assertEqual(
            build_artifacts[0]["mime_type"],
            "application/vnd.android.package-archive",
        )
        self.assertEqual(build_artifacts[0]["metadata"]["size_bytes"], 3)
        self.assertIn(
            "app-release.apk",
            build_artifacts[1]["metadata"]["entry_sample"],
        )

    def test_terminal_execute_can_require_approval_before_running(self):
        run = agent_store.get_agent_store().create_run(project_name="demo", title="Terminal run")
        with patch(
            "routes.terminal.execute_terminal_command_for_current_server",
            new=AsyncMock(return_value=BaseRouteResult.ok({"stdout": "ok\n"})),
        ) as mock_execute:
            blocked = self.client.post(
                "/api/projects/demo/terminal/execute?require_approval=true",
                json={"command": "npm install", "timeout": 5, "run_id": run["id"]},
            )

            self.assertEqual(blocked.status_code, 409)
            mock_execute.assert_not_awaited()
            approval = approval_store.get_approval_store().list_pending(run_id=run["id"])[0]
            self.assertEqual(approval["operation"], "process.terminal")

            decide_approval(
                approval["id"],
                decision="approve_once",
                scope="once",
                reason="test approval",
            )
            allowed = self.client.post(
                "/api/projects/demo/terminal/execute?require_approval=true",
                json={
                    "command": "npm install",
                    "timeout": 5,
                    "run_id": run["id"],
                    "approval_id": approval["id"],
                },
            )

        self.assertEqual(allowed.status_code, 200)
        mock_execute.assert_awaited_once_with("demo", command="npm install", timeout=5)

    def test_terminal_missing_project_error_maps_status(self):
        with patch(
            "routes.terminal.get_terminal_history_for_current_server",
            return_value=BaseRouteResult.error(404, "Project 'demo' not found"),
        ):
            response = self.client.get("/api/projects/demo/terminal/history")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"], "Project 'demo' not found")


if __name__ == "__main__":
    unittest.main()
