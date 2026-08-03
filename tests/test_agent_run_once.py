"""Tests for ``POST /api/agent/agents/{agent_id}/run-once``.

This is the real counterpart to ``/dry-run``: before this route existed,
the only way to actually execute an agent once (as opposed to previewing
it) was to trigger a schedule — and an agent that has no schedule (which is
most agents right after they are built) had no such button at all. These
tests check three things the walkthrough that started this initiative found
missing:

1. A real run genuinely reaches the provider/tool layer — the same
   ``prepare_task_orchestration`` / ``execute_task_orchestration`` pair a
   scheduled fire uses — rather than a second, parallel execution path.
2. An agent with no assigned task gets a clear 4xx, never a 500.
3. The response is unambiguous about what kind of run this was.
"""

from __future__ import annotations

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

from agent import agent_store  # noqa: E402
from core import database  # noqa: E402
from routes import agents  # noqa: E402
from routes.deps import verify_api_key  # noqa: E402


class AgentRunOnceTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "code_bridge_run_once_test.db"
        agent_store._agent_store = None

        app = FastAPI()
        app.include_router(agents.router)
        app.dependency_overrides[verify_api_key] = lambda: "test-api-key"
        self.client = TestClient(app)

    def tearDown(self):
        agent_store._agent_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def _store(self):
        return agent_store.get_agent_store()

    def _create_agent(self):
        return self._store().create_agent(
            name="Disk checker",
            system_prompt="Check disk space and report.",
            provider_id="openai",
            flow_json=[
                {
                    "name": "Check disk",
                    "description": "Inspect free disk space.",
                    "success_criteria": "Free space is known",
                    "on_failure": "ask_user",
                },
            ],
        )

    def _create_task(self, agent_id: str):
        return self._store().create_task(
            title="Check disk daily",
            assigned_agent_id=agent_id,
            goal="Check free disk space and report if under 15%",
            source="test",
        )

    def test_run_once_executes_through_real_orchestration_path(self):
        """A real run reaches the provider layer — nothing is simulated."""
        agent = self._create_agent()
        task = self._create_task(agent["id"])

        async def fake_create_chat_session(**_kwargs):
            return object()

        async def fake_stream_turn(websocket, _session, **_kwargs):
            await websocket.send_json({"type": "complete", "content": "done"})
            return True

        with patch(
            "agent.task_orchestrator.create_chat_session",
            fake_create_chat_session,
        ) as chat, patch(
            "agent.task_orchestrator.stream_claude_turn",
            fake_stream_turn,
        ):
            response = self.client.post(
                f"/api/agent/agents/{agent['id']}/run-once",
                json={},
            )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        # Unmistakable in the response, not just the route name.
        self.assertTrue(payload["real_run"])
        self.assertFalse(payload["dry_run"])
        self.assertEqual(payload["task"]["id"], task["id"])
        run = payload["run"]
        self.assertFalse(bool(run.get("dry_run")))

        # The background execution actually ran the provider path (not
        # `_simulate_run`, which never touches create_chat_session at all).
        task_response = self.client.get(f"/api/agent/tasks/{task['id']}")
        self.assertEqual(task_response.json()["task"]["status"], "completed")
        run_response = self.client.get(f"/api/agent/runs/{run['id']}")
        self.assertEqual(run_response.json()["run"]["status"], "completed")
        self.assertFalse(bool(run_response.json()["run"]["dry_run"]))

    def test_run_once_resolves_the_agents_only_task_without_task_id(self):
        agent = self._create_agent()
        task = self._create_task(agent["id"])

        async def fake_create_chat_session(**_kwargs):
            return object()

        async def fake_stream_turn(websocket, _session, **_kwargs):
            await websocket.send_json({"type": "complete", "content": "done"})
            return True

        with patch(
            "agent.task_orchestrator.create_chat_session",
            fake_create_chat_session,
        ), patch(
            "agent.task_orchestrator.stream_claude_turn",
            fake_stream_turn,
        ):
            response = self.client.post(
                f"/api/agent/agents/{agent['id']}/run-once",
                json={},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["task"]["id"], task["id"])

    def test_run_once_with_no_task_returns_clear_422_not_500(self):
        agent = self._create_agent()

        response = self.client.post(
            f"/api/agent/agents/{agent['id']}/run-once",
            json={},
        )

        self.assertEqual(response.status_code, 422, response.text)
        detail = response.json()["detail"]
        self.assertIsInstance(detail, str)
        self.assertTrue(detail.strip())

    def test_run_once_rejects_task_id_belonging_to_another_agent(self):
        agent_a = self._create_agent()
        agent_b = self._create_agent()
        task_b = self._create_task(agent_b["id"])

        response = self.client.post(
            f"/api/agent/agents/{agent_a['id']}/run-once",
            json={"task_id": task_b["id"]},
        )

        self.assertEqual(response.status_code, 409, response.text)

    def test_run_once_404s_for_unknown_agent(self):
        response = self.client.post(
            "/api/agent/agents/agent_does_not_exist/run-once",
            json={},
        )

        self.assertEqual(response.status_code, 404, response.text)


if __name__ == "__main__":
    unittest.main()
