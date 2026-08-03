"""Dry-run support tests for persistent agents and task schedules."""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store, schedule_store  # noqa: E402
from agent.schedule_store import compute_next_fire_at  # noqa: E402
from agent.task_orchestrator import execute_task_orchestration  # noqa: E402
from core import database  # noqa: E402
from routes import agents  # noqa: E402
from routes.deps import verify_api_key  # noqa: E402


class DryRunTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "code_bridge_dry_run_test.db"
        agent_store._agent_store = None
        schedule_store._store = None

        app = FastAPI()
        app.include_router(agents.router)
        app.dependency_overrides[verify_api_key] = lambda: "test-api-key"
        self.client = TestClient(app)

    def tearDown(self):
        agent_store._agent_store = None
        schedule_store._store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def _store(self):
        return agent_store.get_agent_store()

    def _create_agent(self, *, flow=None, tools=None):
        return self._store().create_agent(
            name="Maintenance bot",
            system_prompt="Maintain safely.",
            provider_id="openai",
            tools_json=tools
            or [
                {
                    "mcp_id": "openai:filesystem",
                    "tool_names": ["read_file", "write_file"],
                    "user_examples": ["inspect files"],
                }
            ],
            flow_json=flow
            or [
                {
                    "name": "Inspect",
                    "description": "Check the current app state.",
                    "tool_hint": "openai:filesystem",
                    "success_criteria": "State is known",
                    "on_failure": "ask_user",
                },
                {
                    "name": "Report",
                    "description": "Summarize what would happen.",
                    "success_criteria": "Summary is ready",
                    "on_failure": "ask_user",
                },
            ],
        )

    def _create_task(self, agent_id: str):
        return self._store().create_task(
            title="Run maintenance",
            assigned_agent_id=agent_id,
            goal="Run the maintenance checklist",
            source="test",
        )

    def test_migration_adds_dry_run_columns_with_default_zero(self):
        with sqlite3.connect(database.DB_PATH) as conn:
            database._migrate_legacy_foundation(conn)
            database._migrate_agent_cockpit_foundation(conn)
            database._migrate_work_cockpit_foundation(conn)
            database._migrate_task_schedules(conn)
            conn.execute(
                """
                INSERT INTO agent_runs (id, status, title)
                VALUES ('run_before_dry_run', 'completed', 'legacy')
                """
            )
            conn.commit()

        database.init_db()

        with sqlite3.connect(database.DB_PATH) as conn:
            columns = {
                row[1]: row[2]
                for row in conn.execute("PRAGMA table_info(agent_runs)").fetchall()
            }
            dry_run_value = conn.execute(
                "SELECT dry_run FROM agent_runs WHERE id = 'run_before_dry_run'"
            ).fetchone()[0]

        self.assertIn("dry_run", columns)
        self.assertIn("simulated_summary_json", columns)
        self.assertEqual(dry_run_value, 0)

    def test_create_run_persists_dry_run_true(self):
        run = self._store().create_run(title="Preview", dry_run=True)

        self.assertTrue(run["dry_run"])
        with sqlite3.connect(database.DB_PATH) as conn:
            raw = conn.execute(
                "SELECT dry_run FROM agent_runs WHERE id = ?",
                (run["id"],),
            ).fetchone()[0]
        self.assertEqual(raw, 1)

    def test_simulate_run_from_workflow_appends_step_events(self):
        agent = self._create_agent()
        task = self._create_task(agent["id"])
        run = self._store().create_run(
            agent_id=agent["id"],
            task_id=task["id"],
            title="Dry run",
            dry_run=True,
        )

        summary = self._store().simulate_run_from_workflow(
            agent=agent,
            task=task,
            run_id=run["id"],
        )
        events = self._store().list_events(run["id"])

        self.assertEqual(summary["steps_simulated"], 2)
        self.assertEqual([event["event_type"] for event in events], ["step.simulated", "step.simulated"])
        self.assertEqual(events[0]["app_event"]["step_name"], "Inspect")
        self.assertEqual(
            events[0]["app_event"]["would_call_mcp"]["mcp_id"],
            "openai:filesystem",
        )

    def test_execute_task_orchestration_dry_run_skips_real_runtime_calls(self):
        agent = self._create_agent()
        task = self._create_task(agent["id"])
        run = self._store().create_run(
            agent_id=agent["id"],
            task_id=task["id"],
            title="Dry run",
            dry_run=True,
        )

        with patch("agent.task_orchestrator.create_chat_session", new_callable=AsyncMock) as chat, patch(
            "agent.task_orchestrator.stream_claude_turn",
            new_callable=AsyncMock,
        ) as stream, patch("agent.task_orchestrator.evaluate_direct_action_gate") as gate, patch(
            "agent.task_orchestrator.execute_terminal_command_streaming_for_current_server",
            new_callable=AsyncMock,
        ) as terminal:
            asyncio.run(
                execute_task_orchestration(
                    {
                        "dry_run": True,
                        "run_id": run["id"],
                        "task_id": task["id"],
                        "agent_id": agent["id"],
                        "agent": agent,
                        "task": task,
                    }
                )
            )

        chat.assert_not_called()
        stream.assert_not_called()
        gate.assert_not_called()
        terminal.assert_not_called()
        updated = self._store().get_run(run["id"])
        events = self._store().list_events(run["id"])
        self.assertEqual(updated["status"], "completed")
        self.assertEqual(events[-1]["event_type"], "run.complete")
        summary = events[-1]["app_event"]["summary"]
        self.assertIn("2 steps simulated", summary)
        # The fact a reader must not miss comes first: this is the exact
        # symptom agent-usability-repair started from — a dry run reporting
        # step successes read as a real run having happened.
        self.assertTrue(summary.startswith("Simulation only — nothing was executed"))

    def test_compute_next_fire_at_returns_nearest_enabled_schedule(self):
        agent = self._create_agent()
        task_a = self._create_task(agent["id"])
        task_b = self._create_task(agent["id"])
        other_agent = self._create_agent()
        other_task = self._create_task(other_agent["id"])
        store = schedule_store.get_schedule_store()
        soon = store.create(task_id=task_a["id"], expression={"kind": "interval", "seconds": 3600})
        later = store.create(task_id=task_b["id"], expression={"kind": "interval", "seconds": 7200})
        other = store.create(task_id=other_task["id"], expression={"kind": "interval", "seconds": 60})
        now = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
        with database.get_db_connection() as conn:
            conn.execute(
                "UPDATE task_schedules SET next_run_at = ? WHERE id = ?",
                ((now + timedelta(minutes=30)).isoformat(), soon["id"]),
            )
            conn.execute(
                "UPDATE task_schedules SET next_run_at = ? WHERE id = ?",
                ((now + timedelta(hours=3)).isoformat(), later["id"]),
            )
            conn.execute(
                "UPDATE task_schedules SET next_run_at = ? WHERE id = ?",
                ((now + timedelta(minutes=5)).isoformat(), other["id"]),
            )
            conn.commit()

        self.assertEqual(
            compute_next_fire_at(agent["id"], now=now),
            now + timedelta(minutes=30),
        )

    def test_compute_next_fire_at_returns_none_without_enabled_schedule(self):
        agent = self._create_agent()

        self.assertIsNone(compute_next_fire_at(agent["id"]))

    def test_get_agent_includes_next_fire_at_key(self):
        agent = self._create_agent()

        response = self.client.get(f"/api/agent/agents/{agent['id']}")

        self.assertEqual(response.status_code, 200)
        self.assertIn("next_fire_at", response.json())
        self.assertIsNone(response.json()["next_fire_at"])

    def test_dry_run_route_returns_run_id_and_records_events(self):
        agent = self._create_agent()
        task = self._create_task(agent["id"])

        response = self.client.post(
            f"/api/agent/agents/{agent['id']}/dry-run",
            json={"task_id": task["id"]},
        )

        self.assertEqual(response.status_code, 200, response.text)
        run_id = response.json()["run_id"]
        self.assertEqual(response.json()["run"]["id"], run_id)
        run = self._store().get_run(run_id)
        events = self._store().list_events(run_id)
        self.assertTrue(run["dry_run"])
        self.assertEqual(run["task_id"], task["id"])
        self.assertEqual(
            [event["event_type"] for event in events],
            ["run.started", "step.simulated", "step.simulated", "run.complete"],
        )

    def test_dry_run_route_uses_synthetic_task_when_agent_has_no_tasks(self):
        agent = self._create_agent()

        response = self.client.post(
            f"/api/agent/agents/{agent['id']}/dry-run",
            json={},
        )

        self.assertEqual(response.status_code, 200, response.text)
        run_id = response.json()["run_id"]
        run = self._store().get_run(run_id)
        events = self._store().list_events(run_id)
        self.assertIsNone(run["task_id"])
        self.assertEqual(events[1]["event_type"], "step.simulated")


if __name__ == "__main__":
    unittest.main()
