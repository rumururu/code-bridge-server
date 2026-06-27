import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store, browser_session_store  # noqa: E402
from agent.browser_action_adapter import BrowserActionAdapterResult  # noqa: E402
from core import database  # noqa: E402
from routes import agents  # noqa: E402
from routes.deps import verify_api_key  # noqa: E402


class AgentCheckpointRoutesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "checkpoint_routes.db"
        agent_store._agent_store = None
        browser_session_store._browser_session_store = None

        app = FastAPI()
        app.include_router(agents.router)
        app.dependency_overrides[verify_api_key] = lambda: "test-api-key"
        self.client = TestClient(app)
        self.store = agent_store.get_agent_store()

    def tearDown(self):
        agent_store._agent_store = None
        browser_session_store._browser_session_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def _waiting_fixture(self):
        agent = self.store.create_agent(
            name="checkpoint bot",
            system_prompt="Pause when the user must act.",
            provider_id="openai",
        )
        task = self.store.create_task(
            title="Resolve captcha",
            assigned_agent_id=agent["id"],
            goal="Wait for captcha",
        )
        run = self.store.create_run(
            task_id=task["id"],
            agent_id=agent["id"],
            provider_id="openai",
            title="Run captcha task",
        )
        self.store.update_task(task["id"], {"run_id": run["id"]})
        step = self.store.create_task_step(
            task_id=task["id"],
            run_id=run["id"],
            title="Captcha handoff",
            status="waiting_for_user",
            input={"workflow_step_id": "captcha", "workflow_type": "manual_handoff"},
            output={
                "checkpoint": {
                    "status": "waiting_for_user",
                    "reason": "manual_handoff",
                    "prompt": "Complete captcha, then continue.",
                    "workflow_step_id": "captcha",
                    "step_id": "step_captcha",
                    "step_title": "Captcha handoff",
                    "workflow_type": "manual_handoff",
                    "success_criteria": "Captcha is complete.",
                    "resume": "same_step",
                    "resume_step_id": None,
                    "resume_behavior": "complete_waiting_step_then_continue",
                    "resume_label": "응답을 저장하고 현재 대기 단계를 완료 처리한 뒤 다음 단계로 진행합니다.",
                    "required_user_action": "Complete captcha in browser.",
                    "allow_memory": False,
                    "created_at": "2026-06-09T00:00:00+00:00",
                }
            },
        )
        self.store.update_task(task["id"], {"status": "waiting_for_user"})
        self.store.update_run_status(run["id"], "waiting_for_user")
        return agent, task, run, step

    def test_task_checkpoint_route_returns_active_checkpoint(self):
        _agent, task, _run, step = self._waiting_fixture()

        response = self.client.get(f"/api/agent/tasks/{task['id']}/checkpoint")

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["step"]["id"], step["id"])
        self.assertEqual(payload["checkpoint"]["workflow_step_id"], "captcha")
        self.assertEqual(payload["checkpoint"]["step_title"], "Captcha handoff")
        self.assertEqual(
            payload["checkpoint"]["resume_behavior"],
            "complete_waiting_step_then_continue",
        )
        self.assertIn("완료 처리", payload["checkpoint"]["resume_label"])
        self.assertIn("allow_memory", payload["checkpoint"])
        self.assertEqual(payload["connector_requests"], [])

    def test_run_checkpoint_route_returns_active_checkpoint(self):
        _agent, _task, run, _step = self._waiting_fixture()

        response = self.client.get(f"/api/agent/runs/{run['id']}/checkpoint")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["run"]["id"], run["id"])
        self.assertEqual(response.json()["checkpoint"]["reason"], "manual_handoff")

    def test_respond_route_persists_response_and_memory(self):
        agent, task, run, step = self._waiting_fixture()

        response = self.client.post(
            f"/api/agent/tasks/{task['id']}/steps/{step['id']}/respond",
            json={
                "message": "Captcha completed.",
                "metadata": {"source": "route-test"},
                "remember": True,
                "resume": True,
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["resume_requested"])
        self.assertEqual(payload["response"]["message"], "Captcha completed.")
        self.assertEqual(payload["memory"]["agent_id"], agent["id"])
        updated_run = self.store.get_run(run["id"])
        updated_task = self.store.get_task(task["id"])
        assert updated_run is not None
        assert updated_task is not None
        self.assertEqual(updated_run["status"], "completed")
        self.assertEqual(updated_task["status"], "completed")

    def test_respond_route_rejects_non_waiting_step(self):
        task = self.store.create_task(title="Running task")
        step = self.store.create_task_step(
            task_id=task["id"],
            title="Running",
            status="running",
        )
        assert step is not None

        response = self.client.post(
            f"/api/agent/tasks/{task['id']}/steps/{step['id']}/respond",
            json={"message": "done"},
        )

        self.assertEqual(response.status_code, 409)

    def test_route_e2e_start_checkpoint_respond_and_resume_workflow(self):
        agent_response = self.client.post(
            "/api/agent/agents",
            json={
                "name": "route e2e workflow bot",
                "system_prompt": "Run workflow steps.",
                "provider_id": "openai",
                "flow_json": [
                    {
                        "id": "open_page",
                        "type": "browser_action",
                        "name": "Open page",
                        "actions": [
                            {
                                "type": "navigate",
                                "url": "data:text/html,<h1>Route E2E</h1>",
                            }
                        ],
                        "success_criteria": "Page opens.",
                    },
                    {
                        "id": "human_gate",
                        "type": "manual_handoff",
                        "name": "Manual confirmation",
                        "on_failure": {
                            "type": "manual_handoff",
                            "prompt": "Confirm the route E2E handoff.",
                            "resume": "same_step",
                        },
                    },
                    {
                        "id": "final_report",
                        "type": "llm",
                        "name": "Final report",
                    },
                ],
            },
        )
        self.assertEqual(agent_response.status_code, 200, agent_response.text)
        agent = agent_response.json()
        task_response = self.client.post(
            "/api/agent/tasks",
            json={
                "title": "Route E2E workflow",
                "assigned_agent_id": agent["id"],
                "goal": "Run route-level workflow E2E.",
            },
        )
        self.assertEqual(task_response.status_code, 200, task_response.text)
        task = task_response.json()["task"]

        async def fake_execute_browser_actions(actions, *, context):
            self.assertEqual(context["workflow_step_id"], "open_page")
            self.assertEqual(actions[0]["type"], "navigate")
            return BrowserActionAdapterResult(
                status="waiting_for_user",
                wait_reason="captcha_or_bot_challenge",
                prompt="Complete captcha, then resume.",
                observations=[{"title": "Route E2E"}],
            )

        async def fake_create_chat_session(**_kwargs):
            return object()

        async def fake_stream(_sink, _session, **_kwargs):
            return True

        with patch(
            "agent.task_orchestrator.execute_browser_actions",
            fake_execute_browser_actions,
        ), patch(
            "agent.task_orchestrator.create_chat_session",
            fake_create_chat_session,
        ), patch(
            "agent.task_orchestrator.stream_claude_turn",
            fake_stream,
        ):
            start_response = self.client.post(
                f"/api/agent/tasks/{task['id']}/start",
                json={"provider_id": "openai", "auto_start": True},
            )

        self.assertEqual(start_response.status_code, 200, start_response.text)
        run = start_response.json()["run"]
        checkpoint_response = self.client.get(
            f"/api/agent/tasks/{task['id']}/checkpoint"
        )
        self.assertEqual(checkpoint_response.status_code, 200, checkpoint_response.text)
        checkpoint_payload = checkpoint_response.json()
        self.assertEqual(
            checkpoint_payload["checkpoint"]["workflow_step_id"],
            "open_page",
        )
        self.assertTrue(
            checkpoint_payload["checkpoint"]["browser_session_id"].startswith("bs_")
        )
        handoff_response = self.client.get(
            f"/api/agent/tasks/{task['id']}/browser-handoff"
        )
        self.assertEqual(handoff_response.status_code, 200, handoff_response.text)
        handoff = handoff_response.json()
        self.assertEqual(
            handoff["browser_session"]["id"],
            checkpoint_payload["checkpoint"]["browser_session_id"],
        )
        self.assertEqual(handoff["browser_session"]["status"], "waiting_for_user")
        screenshot = Path(self._tmp.name) / "handoff.png"
        screenshot.write_bytes(b"\x89PNG\r\n\x1a\n")

        async def fake_handoff_actions(actions, *, context):
            self.assertEqual(context["browser_session_id"], handoff["browser_session"]["id"])
            storage_path = Path(context["browser_storage_state_path"])
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            storage_path.write_text("{}", encoding="utf-8")
            return BrowserActionAdapterResult(
                status="completed",
                observations=[
                    {
                        "url": "https://example.test/handoff",
                        "title": "Handoff",
                        "action_count": len(actions),
                    }
                ],
                screenshots=[str(screenshot)],
                storage_state_path=str(storage_path),
            )

        with patch("routes.agents.execute_browser_actions", fake_handoff_actions):
            snapshot_response = self.client.post(
                f"/api/agent/tasks/{task['id']}/browser-handoff/snapshot"
            )

        self.assertEqual(snapshot_response.status_code, 200, snapshot_response.text)
        snapshot = snapshot_response.json()
        self.assertEqual(snapshot["browser_session"]["current_url"], "https://example.test/handoff")
        self.assertEqual(snapshot["browser_session"]["title"], "Handoff")
        self.assertTrue(snapshot["browser_session"]["storage_state_path"].endswith("storage_state.json"))
        self.assertEqual(snapshot["artifact_ids"], [])
        self.assertTrue(snapshot["result"]["sensitive"])
        self.assertEqual(snapshot["result"]["screenshots"], [])

        with patch("routes.agents.execute_browser_actions", fake_handoff_actions):
            action_response = self.client.post(
                f"/api/agent/tasks/{task['id']}/browser-handoff/actions",
                json={"actions": [{"type": "click", "selector": "#continue"}]},
            )

        self.assertEqual(action_response.status_code, 200, action_response.text)
        action_payload = action_response.json()
        self.assertEqual(action_payload["result"]["status"], "completed")
        self.assertTrue(action_payload["result"]["sensitive"])
        self.assertEqual(action_payload["artifact_ids"], [])
        self.assertEqual(action_payload["browser_session"]["status"], "waiting_for_user")
        steps = self.client.get(f"/api/agent/tasks/{task['id']}/steps").json()["steps"]
        self.assertEqual(
            [
                (step["input"] or {}).get("workflow_step_id")
                for step in steps
            ],
            ["open_page", "human_gate", "final_report"],
        )
        self.assertEqual([step["status"] for step in steps], ["waiting_for_user", "queued", "queued"])

        with patch(
            "agent.task_orchestrator.execute_browser_actions",
            fake_execute_browser_actions,
        ), patch(
            "agent.task_orchestrator.create_chat_session",
            fake_create_chat_session,
        ), patch(
            "agent.task_orchestrator.stream_claude_turn",
            fake_stream,
        ):
            handoff_complete_response = self.client.post(
                f"/api/agent/tasks/{task['id']}/browser-handoff/complete",
                json={
                    "message": "Browser handoff completed.",
                    "remember": False,
                    "resume": True,
                },
            )

        self.assertEqual(
            handoff_complete_response.status_code,
            200,
            handoff_complete_response.text,
        )
        self.assertTrue(handoff_complete_response.json()["browser_handoff_completed"])
        self.assertEqual(
            handoff_complete_response.json()["browser_session"]["status"],
            "resumed",
        )
        second_checkpoint_response = self.client.get(
            f"/api/agent/tasks/{task['id']}/checkpoint"
        )
        self.assertEqual(second_checkpoint_response.status_code, 200)
        second_checkpoint = second_checkpoint_response.json()
        self.assertEqual(
            second_checkpoint["checkpoint"]["workflow_step_id"],
            "human_gate",
        )

        with patch(
            "agent.task_orchestrator.create_chat_session",
            fake_create_chat_session,
        ), patch(
            "agent.task_orchestrator.stream_claude_turn",
            fake_stream,
        ):
            second_respond_response = self.client.post(
                f"/api/agent/tasks/{task['id']}/steps/{second_checkpoint['step']['id']}/respond",
                json={
                    "message": "Manual confirmation completed.",
                    "remember": False,
                    "resume": True,
                },
            )

        self.assertEqual(second_respond_response.status_code, 200, second_respond_response.text)
        final_task = self.client.get(f"/api/agent/tasks/{task['id']}").json()["task"]
        final_run = self.client.get(f"/api/agent/runs/{run['id']}").json()["run"]
        final_checkpoint = self.client.get(
            f"/api/agent/tasks/{task['id']}/checkpoint"
        ).json()
        final_steps = self.client.get(f"/api/agent/tasks/{task['id']}/steps").json()["steps"]
        self.assertEqual(final_task["status"], "completed")
        self.assertEqual(final_run["status"], "completed")
        self.assertIsNone(final_checkpoint["checkpoint"])
        self.assertEqual(
            [step["status"] for step in final_steps],
            ["completed", "completed", "completed"],
        )
        sessions = browser_session_store.get_browser_session_store().list_for_run(run["id"])
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["workflow_step_id"], "open_page")


if __name__ == "__main__":
    unittest.main()
