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
from agent.tool_artifacts import ARTIFACT_ROOT
from approvals import approval_store
from approvals.approval_service import decide_approval
from audit import audit_store
from core.base_result import BaseRouteResult
from core import database
from policy import policy_store
from routes import agents
from routes.deps import verify_api_key


class AgentRoutesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "code_bridge_agent_test.db"
        agent_store._agent_store = None
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None

        app = FastAPI()
        app.include_router(agents.router)
        app.dependency_overrides[verify_api_key] = lambda: "test-api-key"
        self.client = TestClient(app)

    def tearDown(self):
        agent_store._agent_store = None
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def test_create_run_message_event_artifact_and_list(self):
        create_response = self.client.post(
            "/api/agent/runs",
            json={
                "project_name": "demo",
                "provider_id": "google",
                "model": "gemini-2.5-pro",
                "title": "Build login screen",
                "goal": "Create a login UI and run preview",
                "cwd": "/tmp/demo",
            },
        )

        self.assertEqual(create_response.status_code, 200)
        run = create_response.json()["run"]
        self.assertTrue(run["id"].startswith("run_"))
        self.assertEqual(run["project_name"], "demo")
        self.assertEqual(run["status"], "queued")

        message_response = self.client.post(
            f"/api/agent/runs/{run['id']}/message",
            json={
                "role": "user",
                "content": "Build a login screen",
                "attachments": [{"kind": "screenshot", "path": "screen.png"}],
            },
        )
        self.assertEqual(message_response.status_code, 200)
        self.assertEqual(message_response.json()["message"]["role"], "user")

        event_response = self.client.post(
            f"/api/agent/runs/{run['id']}/event",
            json={
                "event_type": "status",
                "provider_id": "google",
                "app_event": {"stage": "planning"},
            },
        )
        self.assertEqual(event_response.status_code, 200)
        self.assertEqual(event_response.json()["event"]["sequence"], 1)

        artifact_response = self.client.post(
            f"/api/agent/runs/{run['id']}/artifacts",
            json={
                "kind": "log",
                "path": "logs/build.txt",
                "mime_type": "text/plain",
                "metadata": {"lines": 20},
            },
        )
        self.assertEqual(artifact_response.status_code, 200)
        self.assertEqual(artifact_response.json()["artifact"]["metadata"]["lines"], 20)

        get_response = self.client.get(f"/api/agent/runs/{run['id']}")
        self.assertEqual(get_response.status_code, 200)
        payload = get_response.json()
        self.assertEqual(payload["run"]["id"], run["id"])
        self.assertEqual(len(payload["messages"]), 1)
        self.assertEqual(len(payload["artifacts"]), 1)

        events_response = self.client.get(f"/api/agent/runs/{run['id']}/events")
        self.assertEqual(events_response.status_code, 200)
        self.assertEqual(events_response.json()["events"][0]["app_event"]["stage"], "planning")

        list_response = self.client.get("/api/agent/runs?project_name=demo")
        self.assertEqual(list_response.status_code, 200)
        self.assertEqual([item["id"] for item in list_response.json()["runs"]], [run["id"]])

    def test_get_artifact_content_reads_generated_text_artifact(self):
        run = self.client.post(
            "/api/agent/runs",
            json={"project_name": "demo", "title": "Artifacts"},
        ).json()["run"]
        artifact_dir = ARTIFACT_ROOT / run["id"]
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "terminal_001.log"
        artifact_path.write_text("hello artifact\n", encoding="utf-8")
        artifact = agent_store.get_agent_store().add_artifact(
            run_id=run["id"],
            kind="terminal_log",
            path=str(artifact_path),
            mime_type="text/plain",
        )

        response = self.client.get(
            f"/api/agent/runs/{run['id']}/artifacts/{artifact['id']}/content",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["readable"])
        self.assertEqual(payload["content"], "hello artifact\n")

    def test_get_artifact_content_restricts_external_paths(self):
        run = self.client.post(
            "/api/agent/runs",
            json={"project_name": "demo", "title": "Artifacts"},
        ).json()["run"]
        external_path = Path(self._tmp.name) / "external.log"
        external_path.write_text("secret\n", encoding="utf-8")
        artifact = agent_store.get_agent_store().add_artifact(
            run_id=run["id"],
            kind="terminal_log",
            path=str(external_path),
            mime_type="text/plain",
        )

        response = self.client.get(
            f"/api/agent/runs/{run['id']}/artifacts/{artifact['id']}/content",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["readable"])
        self.assertIsNone(payload["content"])

    def test_get_artifact_raw_serves_generated_text_artifact(self):
        run = self.client.post(
            "/api/agent/runs",
            json={"project_name": "demo", "title": "Artifacts"},
        ).json()["run"]
        artifact_dir = ARTIFACT_ROOT / run["id"]
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "source_diff_001.diff"
        artifact_path.write_text("diff content\n", encoding="utf-8")
        artifact = agent_store.get_agent_store().add_artifact(
            run_id=run["id"],
            kind="source_diff",
            path=str(artifact_path),
            mime_type="text/x-diff",
        )

        response = self.client.get(
            f"/api/agent/runs/{run['id']}/artifacts/{artifact['id']}/raw",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "diff content\n")

    def test_get_artifact_raw_rejects_external_paths(self):
        run = self.client.post(
            "/api/agent/runs",
            json={"project_name": "demo", "title": "Artifacts"},
        ).json()["run"]
        external_path = Path(self._tmp.name) / "external.log"
        external_path.write_text("secret\n", encoding="utf-8")
        artifact = agent_store.get_agent_store().add_artifact(
            run_id=run["id"],
            kind="terminal_log",
            path=str(external_path),
            mime_type="text/plain",
        )

        response = self.client.get(
            f"/api/agent/runs/{run['id']}/artifacts/{artifact['id']}/raw",
        )

        self.assertEqual(response.status_code, 403)

    def test_list_runs_can_filter_by_workspace(self):
        first = self.client.post(
            "/api/agent/runs",
            json={"workspace_id": "wsp_one", "title": "First"},
        ).json()["run"]
        self.client.post(
            "/api/agent/runs",
            json={"workspace_id": "wsp_two", "title": "Second"},
        )

        response = self.client.get("/api/agent/runs?workspace_id=wsp_one")

        self.assertEqual(response.status_code, 200)
        self.assertEqual([item["id"] for item in response.json()["runs"]], [first["id"]])

    def test_task_lifecycle(self):
        create_response = self.client.post(
            "/api/agent/tasks",
            json={
                "title": "Add onboarding flow",
                "description": "Create screens and connect preview",
                "project_name": "demo",
                "kind": "general",
                "source": "manual",
                "goal": "Ship onboarding",
                "labels": ["product", "mobile"],
                "assignee": "owner",
                "priority": 5,
            },
        )
        self.assertEqual(create_response.status_code, 200)
        task = create_response.json()["task"]
        self.assertTrue(task["id"].startswith("task_"))
        self.assertEqual(task["status"], "queued")
        self.assertEqual(task["project_name"], "demo")
        self.assertEqual(task["kind"], "general")
        self.assertEqual(task["labels"], ["product", "mobile"])

        list_response = self.client.get("/api/agent/tasks")
        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(list_response.json()["tasks"][0]["id"], task["id"])

        update_response = self.client.patch(
            f"/api/agent/tasks/{task['id']}",
            json={"status": "in_progress", "priority": 3},
        )
        self.assertEqual(update_response.status_code, 200)
        self.assertEqual(update_response.json()["task"]["status"], "in_progress")

        cancel_response = self.client.post(f"/api/agent/tasks/{task['id']}/cancel")
        self.assertEqual(cancel_response.status_code, 200)
        self.assertEqual(cancel_response.json()["task"]["status"], "cancelled")

    def test_task_can_link_run_and_return_timeline(self):
        task = self.client.post(
            "/api/agent/tasks",
            json={"title": "Research competitors", "kind": "research"},
        ).json()["task"]
        run = self.client.post(
            "/api/agent/runs",
            json={"title": "Research run", "task_id": task["id"]},
        ).json()["run"]

        link_response = self.client.post(
            f"/api/agent/tasks/{task['id']}/runs",
            json={"run_id": run["id"], "role": "retry"},
        )

        self.assertEqual(link_response.status_code, 200)
        self.assertEqual(link_response.json()["link"]["task_id"], task["id"])
        timeline_response = self.client.get(f"/api/agent/tasks/{task['id']}/timeline")
        self.assertEqual(timeline_response.status_code, 200)
        timeline = timeline_response.json()
        self.assertEqual(timeline["task"]["id"], task["id"])
        self.assertEqual(timeline["runs"][0]["id"], run["id"])

    def test_task_start_creates_orchestration_run_steps_and_capabilities(self):
        task = self.client.post(
            "/api/agent/tasks",
            json={
                "title": "Research competitors",
                "kind": "research",
                "labels": ["github"],
                "goal": "Summarize market positioning",
            },
        ).json()["task"]

        response = self.client.post(
            f"/api/agent/tasks/{task['id']}/start",
            json={
                "provider_id": "google",
                "model": "gemini-test",
                "auto_start": False,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["run"]["task_id"], task["id"])
        self.assertEqual(payload["run"]["provider_id"], "google")
        self.assertGreaterEqual(len(payload["steps"]), 3)
        capability_names = {item["name"] for item in payload["capabilities"]}
        self.assertIn("browser", capability_names)
        self.assertIn("github", capability_names)
        self.assertIn("launch_message", payload)

        timeline_response = self.client.get(f"/api/agent/tasks/{task['id']}/timeline")
        self.assertEqual(timeline_response.status_code, 200)
        timeline = timeline_response.json()
        self.assertEqual(timeline["runs"][0]["id"], payload["run"]["id"])
        self.assertEqual(len(timeline["steps"]), len(payload["steps"]))
        self.assertTrue(timeline["capability_links"])
        self.assertTrue(
            any(
                step["input"].get("adapter", {}).get("adapter") in {"skill_registry", "mcp_registry"}
                for step in timeline["steps"]
            )
        )

        run_detail = self.client.get(f"/api/agent/runs/{payload['run']['id']}").json()
        self.assertEqual([message["role"] for message in run_detail["messages"]], ["system", "user"])

    def test_task_start_background_execution_updates_status(self):
        task = self.client.post(
            "/api/agent/tasks",
            json={"title": "Review code", "kind": "review"},
        ).json()["task"]

        async def fake_create_chat_session(**_kwargs):
            return object()

        async def fake_stream_turn(websocket, _session, **_kwargs):
            await websocket.send_json({"type": "complete", "content": "done"})
            return True

        with patch("agent.task_orchestrator.create_chat_session", fake_create_chat_session):
            with patch("agent.task_orchestrator.stream_claude_turn", fake_stream_turn):
                response = self.client.post(
                    f"/api/agent/tasks/{task['id']}/start",
                    json={"provider_id": "google", "auto_start": True},
                )

        self.assertEqual(response.status_code, 200)
        run = response.json()["run"]
        task_response = self.client.get(f"/api/agent/tasks/{task['id']}")
        self.assertEqual(task_response.json()["task"]["status"], "completed")
        run_response = self.client.get(f"/api/agent/runs/{run['id']}")
        self.assertEqual(run_response.json()["run"]["status"], "completed")

    def test_task_step_terminal_adapter_can_run_after_review(self):
        task = self.client.post(
            "/api/agent/tasks",
            json={"title": "Run smoke test", "kind": "ops", "project_name": "demo"},
        ).json()["task"]
        started = self.client.post(
            f"/api/agent/tasks/{task['id']}/start",
            json={"provider_id": "google", "auto_start": False},
        ).json()
        terminal_step = next(
            step for step in started["steps"] if step["input"].get("capability") == "process.terminal"
        )

        with patch(
            "agent.task_orchestrator.execute_terminal_command_streaming_for_current_server",
            new=AsyncMock(return_value=BaseRouteResult.ok({"stdout": "ok", "exit_code": 0})),
        ):
            response = self.client.post(
                f"/api/agent/tasks/{task['id']}/steps/{terminal_step['id']}/run",
                json={"input": {"command": "npm test", "timeout": 30}},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["step"]["status"], "completed")
        self.assertEqual(payload["output"]["commands"][0]["command"], "npm test")

    def test_deferred_skill_step_blocks_for_review(self):
        task = self.client.post(
            "/api/agent/tasks",
            json={"title": "Research market", "kind": "research"},
        ).json()["task"]
        started = self.client.post(
            f"/api/agent/tasks/{task['id']}/start",
            json={"provider_id": "google", "auto_start": False},
        ).json()
        skill_step = next(
            step for step in started["steps"] if step["input"].get("adapter", {}).get("adapter") == "skill_registry"
        )

        response = self.client.post(
            f"/api/agent/tasks/{task['id']}/steps/{skill_step['id']}/run",
            json={},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["step"]["status"], "blocked")
        self.assertTrue(payload["output"]["review_required"])
        request = payload["output"]["connector_request"]
        self.assertEqual(request["status"], "pending_review")

        list_response = self.client.get(f"/api/agent/tasks/{task['id']}/connector-requests")
        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(
            list_response.json()["connector_requests"][0]["id"],
            request["id"],
        )

        complete_response = self.client.patch(
            f"/api/agent/tasks/{task['id']}/connector-requests/{request['id']}",
            json={
                "status": "completed",
                "result": {"summary": "external skill completed"},
            },
        )
        self.assertEqual(complete_response.status_code, 200)
        self.assertEqual(complete_response.json()["connector_request"]["status"], "completed")
        self.assertEqual(complete_response.json()["step"]["status"], "completed")

    def test_capability_catalog_refreshes_builtins_and_providers(self):
        response = self.client.post("/api/agent/capabilities/refresh")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertGreater(payload["count"], 0)
        names = {item["name"] for item in payload["capabilities"]}
        self.assertIn("file.write", names)

        list_response = self.client.get("/api/agent/capabilities")
        self.assertEqual(list_response.status_code, 200)
        self.assertTrue(list_response.json()["capabilities"])

    def test_unknown_run_returns_404(self):
        response = self.client.get("/api/agent/runs/run_missing")

        self.assertEqual(response.status_code, 404)

    def test_run_preflight_requires_approval_and_records_summary(self):
        run = self.client.post(
            "/api/agent/runs",
            json={"project_name": "demo", "title": "Preflight"},
        ).json()["run"]

        blocked = self.client.post(
            f"/api/agent/runs/{run['id']}/preflight",
            json={"commands": ["npm run build"], "timeout": 120},
        )

        self.assertEqual(blocked.status_code, 409)
        approval = approval_store.get_approval_store().list_pending(run_id=run["id"])[0]
        self.assertEqual(approval["operation"], "process.terminal")
        decide_approval(
            approval["id"],
            decision="approve_once",
            reason="test approval",
            approver={"type": "desktop_app"},
        )

        with patch(
            "routes.agents.execute_terminal_command_for_current_server",
            new=AsyncMock(
                return_value=BaseRouteResult.ok(
                    {
                        "stdout": "built",
                        "stderr": "",
                        "exit_code": 0,
                        "error": None,
                        "timed_out": False,
                    }
                )
            ),
        ) as execute_mock:
            allowed = self.client.post(
                f"/api/agent/runs/{run['id']}/preflight",
                json={
                    "commands": ["npm run build"],
                    "timeout": 120,
                    "approval_id": approval["id"],
                },
            )

        self.assertEqual(allowed.status_code, 200)
        self.assertTrue(allowed.json()["passed"])
        execute_mock.assert_awaited_once_with(
            "demo",
            command="npm run build",
            timeout=120,
        )
        artifacts = agent_store.get_agent_store().list_artifacts(run["id"])
        preflight_artifacts = [
            artifact for artifact in artifacts if artifact["kind"] == "agent_preflight"
        ]
        self.assertEqual(len(preflight_artifacts), 1)
        self.assertTrue(preflight_artifacts[0]["metadata"]["passed"])
        events = agent_store.get_agent_store().list_events(run["id"])
        self.assertEqual(events[-1]["event_type"], "preflight.completed")

    def test_emergency_stop_cancels_active_runs_and_denies_approvals(self):
        run = self.client.post(
            "/api/agent/runs",
            json={"project_name": "demo", "title": "Active run"},
        ).json()["run"]
        approval = approval_store.get_approval_store().create_request(
            operation="process.terminal",
            run_id=run["id"],
            details={"command": "npm test"},
        )

        response = self.client.post("/api/agent/emergency-stop")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["summary"]["cancelled_runs"], 1)
        self.assertEqual(payload["summary"]["denied_approvals"], 1)
        self.assertEqual(payload["cancelled_runs"][0]["status"], "cancelled")
        self.assertEqual(
            approval_store.get_approval_store().get_request(approval["id"])["status"],
            "denied",
        )


if __name__ == "__main__":
    unittest.main()
