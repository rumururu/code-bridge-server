import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store, schedule_store  # noqa: E402
from agent.configurator import create_builder_session  # noqa: E402
from core import database  # noqa: E402
from routes import agents  # noqa: E402
from routes.deps import verify_api_key  # noqa: E402


class AgentsCrudTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "code_bridge_agents_crud_test.db"
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

    def _create_agent(self, name: str = "bot", **overrides):
        body = {
            "name": name,
            "description": "test agent",
            "system_prompt": "You are useful.",
            "provider_id": "openai",
            "tools_json": [{"mcp_id": "openai:terminal", "tool_names": ["run"]}],
            "flow_json": [{"id": "step_one", "name": "Plan"}],
            "policy_overrides_json": {"risk": "low"},
        }
        body.update(overrides)
        response = self.client.post("/api/agent/agents", json=body)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _prepare_legacy_db_with_rows(self):
        with sqlite3.connect(database.DB_PATH) as conn:
            database._migrate_legacy_foundation(conn)
            database._migrate_agent_cockpit_foundation(conn)
            database._migrate_work_cockpit_foundation(conn)
            database._migrate_task_schedules(conn)
            conn.execute(
                """
                INSERT INTO agent_runs (id, status, title)
                VALUES ('run_legacy', 'completed', 'legacy run')
                """
            )
            conn.execute(
                """
                INSERT INTO agent_tasks (id, title, status)
                VALUES ('task_legacy', 'legacy task', 'queued')
                """
            )
            conn.commit()

    def test_migration_seeds_two_pseudo_agents(self):
        database.init_db()

        with sqlite3.connect(database.DB_PATH) as conn:
            rows = conn.execute(
                """
                SELECT id FROM agents
                WHERE is_pseudo = 1
                ORDER BY id
                """
            ).fetchall()

        self.assertEqual(
            [row[0] for row in rows],
            ["agent_adhoc_dev", "agent_legacy_chat"],
        )

    def test_agent_list_excludes_pseudo_by_default(self):
        response = self.client.get("/api/agent/agents")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"agents": [], "total": 0})

    def test_agent_list_can_include_pseudo(self):
        response = self.client.get("/api/agent/agents?include_pseudo=true")

        self.assertEqual(response.status_code, 200)
        ids = {agent["id"] for agent in response.json()["agents"]}
        self.assertEqual(ids, {"agent_legacy_chat", "agent_adhoc_dev"})
        self.assertEqual(response.json()["total"], 2)

    def test_existing_agent_runs_are_backfilled_to_legacy_chat(self):
        self._prepare_legacy_db_with_rows()
        database.init_db()

        with sqlite3.connect(database.DB_PATH) as conn:
            agent_id = conn.execute(
                "SELECT agent_id FROM agent_runs WHERE id = 'run_legacy'"
            ).fetchone()[0]

        self.assertEqual(agent_id, "agent_legacy_chat")

    def test_existing_agent_tasks_are_backfilled_to_legacy_chat(self):
        self._prepare_legacy_db_with_rows()
        database.init_db()

        with sqlite3.connect(database.DB_PATH) as conn:
            assigned_agent_id = conn.execute(
                "SELECT assigned_agent_id FROM agent_tasks WHERE id = 'task_legacy'"
            ).fetchone()[0]

        self.assertEqual(assigned_agent_id, "agent_legacy_chat")

    def test_agents_migration_is_idempotent(self):
        database.init_db()
        database.init_db()

        with sqlite3.connect(database.DB_PATH) as conn:
            pseudo_count = conn.execute(
                "SELECT COUNT(*) FROM agents WHERE is_pseudo = 1"
            ).fetchone()[0]
            migration_count = conn.execute(
                """
                SELECT COUNT(*) FROM schema_migrations
                WHERE version = 2026053100 AND name = 'agents_and_memories'
                """
            ).fetchone()[0]

        self.assertEqual(pseudo_count, 2)
        self.assertEqual(migration_count, 1)

    def test_create_agent_ignores_request_id_and_parses_json_fields(self):
        agent = self._create_agent(id="client_supplied")

        self.assertTrue(agent["id"].startswith("agent_"))
        self.assertNotEqual(agent["id"], "client_supplied")
        self.assertEqual(agent["tools_json"][0]["mcp_id"], "openai:terminal")
        self.assertEqual(agent["flow_json"][0]["id"], "step_one")
        self.assertEqual(agent["policy_overrides_json"]["risk"], "low")
        self.assertFalse(agent["is_pseudo"])

    def test_create_agent_normalizes_legacy_workflow_failure_policy(self):
        agent = self._create_agent(
            flow_json=[
                {
                    "id": "step_one",
                    "name": "Plan",
                    "description": "Plan the work",
                    "on_failure": "retry_once",
                }
            ],
        )

        step = agent["flow_json"][0]
        self.assertEqual(step["type"], "llm")
        self.assertEqual(
            step["on_failure"],
            {"type": "retry", "max_attempts": 1, "then": {"type": "abort"}},
        )
        self.assertEqual(step["actions"], [])

    def test_create_agent_rejects_invalid_workflow(self):
        cases = [
            (
                [{"id": "dup", "name": "First"}, {"id": "dup", "name": "Second"}],
                "duplicate step id",
            ),
            (
                [
                    {
                        "id": "one",
                        "name": "Browse",
                        "actions": [{"type": "teleport"}],
                    }
                ],
                "unknown action type",
            ),
            (
                [
                    {
                        "id": "one",
                        "name": "Plan",
                        "on_failure": {
                            "type": "goto_step",
                            "target_step_id": "missing",
                        },
                    }
                ],
                "unknown goto_step target",
            ),
        ]

        for flow_json, expected_detail in cases:
            with self.subTest(expected_detail=expected_detail):
                response = self.client.post(
                    "/api/agent/agents",
                    json={
                        "name": "invalid workflow",
                        "system_prompt": "You are useful.",
                        "provider_id": "openai",
                        "flow_json": flow_json,
                    },
                )

                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn(expected_detail, response.json()["detail"])

    def test_create_agent_requires_name(self):
        response = self.client.post("/api/agent/agents", json={"name": ""})

        self.assertEqual(response.status_code, 422)

    def test_create_agent_validates_provider(self):
        response = self.client.post(
            "/api/agent/agents",
            json={"name": "bot", "provider_id": "invalid"},
        )

        self.assertEqual(response.status_code, 422)

    def test_get_agent_returns_created_agent(self):
        agent = self._create_agent(name="reader")

        response = self.client.get(f"/api/agent/agents/{agent['id']}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], agent["id"])
        self.assertEqual(response.json()["name"], "reader")
        self.assertIn("next_fire_at", response.json())

    def test_patch_agent_updates_fields_and_json(self):
        agent = self._create_agent()

        response = self.client.patch(
            f"/api/agent/agents/{agent['id']}",
            json={
                "name": "patched",
                "provider_id": "google",
                "tools_json": [{"mcp_id": "google:files", "tool_names": ["read"]}],
            },
        )

        self.assertEqual(response.status_code, 200)
        updated = response.json()
        self.assertEqual(updated["name"], "patched")
        self.assertEqual(updated["provider_id"], "google")
        self.assertEqual(updated["tools_json"][0]["mcp_id"], "google:files")

    def test_patch_agent_normalizes_workflow_failure_policy(self):
        agent = self._create_agent()

        response = self.client.patch(
            f"/api/agent/agents/{agent['id']}",
            json={
                "flow_json": [
                    {
                        "id": "step_one",
                        "name": "Ask",
                        "on_failure": "ask_user",
                    },
                    {
                        "id": "step_two",
                        "name": "Stop",
                        "on_failure": "abort",
                    },
                ],
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        steps = response.json()["flow_json"]
        self.assertEqual(steps[0]["on_failure"], {"type": "ask_user", "resume": "same_step"})
        self.assertEqual(steps[1]["on_failure"], {"type": "abort"})

    def test_patch_agent_rejects_invalid_goto_workflow(self):
        agent = self._create_agent()

        response = self.client.patch(
            f"/api/agent/agents/{agent['id']}",
            json={
                "flow_json": [
                    {
                        "id": "step_one",
                        "name": "Plan",
                        "on_failure": {
                            "type": "goto_step",
                            "target_step_id": "missing",
                        },
                    }
                ],
            },
        )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("unknown goto_step target", response.json()["detail"])

    def test_patch_agent_with_no_fields_returns_current_agent(self):
        agent = self._create_agent(name="steady")

        response = self.client.patch(f"/api/agent/agents/{agent['id']}", json={})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["name"], "steady")

    def test_builder_commit_normalizes_legacy_workflow_failure_policy(self):
        session = create_builder_session(system_prompt="test")

        response = self.client.post(
            "/api/agent/builder/commit",
            json={
                "session_id": session.session_id,
                "draft": {
                    "name": "Builder normalized",
                    "description": "Builder commit test",
                    "system_prompt": "You are useful.",
                    "provider_id": "openai",
                    "tools": [],
                    "flow": [
                        {
                            "name": "Retry once",
                            "description": "Try once more before aborting.",
                            "tool_hint": None,
                            "success_criteria": "Done",
                            "on_failure": "retry_once",
                        }
                    ],
                    "memory_seeds": [],
                },
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        step = response.json()["agent"]["flow_json"][0]
        self.assertEqual(step["id"], "step_1")
        self.assertEqual(step["type"], "llm")
        self.assertEqual(
            step["on_failure"],
            {"type": "retry", "max_attempts": 1, "then": {"type": "abort"}},
        )

    def test_patch_pseudo_agent_is_forbidden(self):
        response = self.client.patch(
            "/api/agent/agents/agent_legacy_chat",
            json={"name": "new name"},
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json(), {"error": "pseudo_agent_protected"})

    def test_delete_pseudo_agent_is_forbidden(self):
        response = self.client.delete("/api/agent/agents/agent_adhoc_dev")

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json(), {"error": "pseudo_agent_protected"})

    def test_post_with_pseudo_id_conflict_returns_409(self):
        response = self.client.post(
            "/api/agent/agents",
            json={"id": "agent_adhoc_dev", "name": "collision"},
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"], "agent_id_conflict")

    def test_duplicate_agent_names_are_allowed(self):
        first = self._create_agent(name="bot")
        second = self._create_agent(name="bot")

        self.assertNotEqual(first["id"], second["id"])
        response = self.client.get("/api/agent/agents")
        self.assertEqual(response.json()["total"], 2)

    def test_soft_delete_hides_agent_by_default(self):
        agent = self._create_agent(name="archive me")

        delete_response = self.client.delete(f"/api/agent/agents/{agent['id']}")
        list_response = self.client.get("/api/agent/agents")

        self.assertEqual(delete_response.status_code, 200)
        self.assertEqual(delete_response.json(), {"id": agent["id"], "archived": True})
        self.assertEqual(list_response.json()["agents"], [])

    def test_soft_delete_disables_assigned_task_schedules(self):
        agent = self._create_agent(name="archive scheduled")
        task = agent_store.get_agent_store().create_task(
            title="Scheduled task",
            assigned_agent_id=agent["id"],
            goal="Run every hour",
        )
        schedule = schedule_store.get_schedule_store().create(
            task_id=task["id"],
            expression={"kind": "interval", "seconds": 3600},
        )
        with database.get_db_connection() as conn:
            conn.execute(
                "UPDATE task_schedules SET next_run_at = ? WHERE id = ?",
                ("2020-01-01T00:00:00+00:00", schedule["id"]),
            )
            conn.commit()

        delete_response = self.client.delete(f"/api/agent/agents/{agent['id']}")
        updated = schedule_store.get_schedule_store().get(schedule["id"])
        due_ids = {
            item["id"] for item in schedule_store.get_schedule_store().list_due()
        }

        self.assertEqual(delete_response.status_code, 200)
        self.assertIsNotNone(updated)
        self.assertFalse(updated["enabled"])
        self.assertNotIn(schedule["id"], due_ids)

    def test_include_archived_shows_soft_deleted_agent(self):
        agent = self._create_agent(name="archived visible")
        self.client.delete(f"/api/agent/agents/{agent['id']}")

        response = self.client.get("/api/agent/agents?include_archived=true")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["agents"][0]["id"], agent["id"])
        self.assertIsNotNone(response.json()["agents"][0]["archived_at"])

    def test_hard_delete_agent_without_active_run_removes_it(self):
        agent = self._create_agent()

        response = self.client.delete(f"/api/agent/agents/{agent['id']}?archive=false")
        get_response = self.client.get(f"/api/agent/agents/{agent['id']}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"id": agent["id"], "archived": False})
        self.assertEqual(get_response.status_code, 404)

    def test_hard_delete_disables_assigned_task_schedules(self):
        agent = self._create_agent(name="delete scheduled")
        task = agent_store.get_agent_store().create_task(
            title="Scheduled task",
            assigned_agent_id=agent["id"],
            goal="Run every hour",
        )
        schedule = schedule_store.get_schedule_store().create(
            task_id=task["id"],
            expression={"kind": "interval", "seconds": 3600},
        )

        response = self.client.delete(f"/api/agent/agents/{agent['id']}?archive=false")
        updated = schedule_store.get_schedule_store().get(schedule["id"])

        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(updated)
        self.assertFalse(updated["enabled"])

    def test_hard_delete_agent_with_active_run_returns_409(self):
        agent = self._create_agent()
        run_response = self.client.post(
            "/api/agent/runs",
            json={"agent_id": agent["id"], "title": "active"},
        )
        self.assertEqual(run_response.status_code, 200)

        response = self.client.delete(f"/api/agent/agents/{agent['id']}?archive=false")

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"], "active_runs_present")

    def test_list_runs_filters_by_agent_and_task(self):
        first = self._create_agent(name="first")
        second = self._create_agent(name="second")
        task = agent_store.get_agent_store().create_task(
            title="First task",
            assigned_agent_id=first["id"],
            goal="Run first",
        )
        first_run = self.client.post(
            "/api/agent/runs",
            json={
                "agent_id": first["id"],
                "task_id": task["id"],
                "title": "first run",
            },
        ).json()["run"]
        self.client.post(
            "/api/agent/runs",
            json={"agent_id": second["id"], "title": "second run"},
        )

        by_agent = self.client.get(f"/api/agent/runs?agent_id={first['id']}")
        by_task = self.client.get(f"/api/agent/runs?task_id={task['id']}")

        self.assertEqual(by_agent.status_code, 200)
        self.assertEqual([run["id"] for run in by_agent.json()["runs"]], [first_run["id"]])
        self.assertEqual(by_task.status_code, 200)
        self.assertEqual([run["id"] for run in by_task.json()["runs"]], [first_run["id"]])

    def test_memory_add_and_list_orders_pinned_first(self):
        agent = self._create_agent()
        plain = self.client.post(
            f"/api/agent/agents/{agent['id']}/memories",
            json={"content": "plain"},
        ).json()
        pinned = self.client.post(
            f"/api/agent/agents/{agent['id']}/memories",
            json={"content": "pinned", "pinned": True},
        ).json()

        response = self.client.get(f"/api/agent/agents/{agent['id']}/memories")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["total"], 2)
        self.assertEqual([memory["id"] for memory in response.json()["memories"]], [pinned["id"], plain["id"]])

    def test_memory_update_pin_reorders_list(self):
        agent = self._create_agent()
        first = self.client.post(
            f"/api/agent/agents/{agent['id']}/memories",
            json={"content": "first"},
        ).json()
        second = self.client.post(
            f"/api/agent/agents/{agent['id']}/memories",
            json={"content": "second"},
        ).json()

        update_response = self.client.patch(
            f"/api/agent/agents/{agent['id']}/memories/{first['id']}",
            json={"pinned": True},
        )
        list_response = self.client.get(f"/api/agent/agents/{agent['id']}/memories")

        self.assertEqual(update_response.status_code, 200)
        self.assertTrue(update_response.json()["pinned"])
        self.assertEqual(list_response.json()["memories"][0]["id"], first["id"])
        self.assertIn(second["id"], [item["id"] for item in list_response.json()["memories"]])

    def test_memory_delete_removes_item(self):
        agent = self._create_agent()
        memory = self.client.post(
            f"/api/agent/agents/{agent['id']}/memories",
            json={"content": "temporary"},
        ).json()

        delete_response = self.client.delete(
            f"/api/agent/agents/{agent['id']}/memories/{memory['id']}"
        )
        list_response = self.client.get(f"/api/agent/agents/{agent['id']}/memories")

        self.assertEqual(delete_response.status_code, 204)
        self.assertEqual(list_response.json()["memories"], [])

    def test_pseudo_agent_memories_get_returns_empty_list(self):
        response = self.client.get("/api/agent/agents/agent_adhoc_dev/memories")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"memories": [], "total": 0})

    def test_pseudo_agent_memory_post_is_forbidden(self):
        response = self.client.post(
            "/api/agent/agents/agent_adhoc_dev/memories",
            json={"content": "must not store"},
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json(), {"error": "pseudo_agent_protected"})

    def test_memory_missing_agent_returns_404(self):
        response = self.client.get("/api/agent/agents/agent_missing/memories")

        self.assertEqual(response.status_code, 404)

    def test_get_agent_exposes_next_fire_at_when_scheduled(self):
        agent = self._create_agent(name="scheduled")
        task = agent_store.get_agent_store().create_task(
            title="Scheduled task",
            assigned_agent_id=agent["id"],
            goal="Run every hour",
        )
        schedule = schedule_store.get_schedule_store().create(
            task_id=task["id"],
            expression={"kind": "interval", "seconds": 3600},
        )
        with database.get_db_connection() as conn:
            conn.execute(
                "UPDATE task_schedules SET next_run_at = ? WHERE id = ?",
                ("2099-06-01T12:30:00+00:00", schedule["id"]),
            )
            conn.commit()

        response = self.client.get(f"/api/agent/agents/{agent['id']}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["next_fire_at"], "2099-06-01T12:30:00+00:00")

    def test_get_agent_next_fire_at_null_without_schedule(self):
        agent = self._create_agent(name="manual only")

        response = self.client.get(f"/api/agent/agents/{agent['id']}")

        self.assertEqual(response.status_code, 200)
        self.assertIn("next_fire_at", response.json())
        self.assertIsNone(response.json()["next_fire_at"])

    def test_get_agent_exposes_activation_summary(self):
        agent = self._create_agent(name="activation summary")
        unscheduled_task = agent_store.get_agent_store().create_task(
            title="Manual activation task",
            assigned_agent_id=agent["id"],
            goal="Run manually",
        )
        scheduled_task = agent_store.get_agent_store().create_task(
            title="Scheduled activation task",
            assigned_agent_id=agent["id"],
            goal="Run on schedule",
        )
        schedule_store.get_schedule_store().create(
            task_id=scheduled_task["id"],
            expression={"kind": "interval", "seconds": 3600},
        )
        schedule_store.get_schedule_store().create(
            task_id=scheduled_task["id"],
            expression={"kind": "interval", "seconds": 7200},
            enabled=False,
        )
        other_agent = self._create_agent(name="other activation summary")
        other_task = agent_store.get_agent_store().create_task(
            title="Other task",
            assigned_agent_id=other_agent["id"],
            goal="Ignore this task",
        )
        schedule_store.get_schedule_store().create(
            task_id=other_task["id"],
            expression={"kind": "interval", "seconds": 1800},
        )
        self.assertNotEqual(unscheduled_task["id"], scheduled_task["id"])

        response = self.client.get(f"/api/agent/agents/{agent['id']}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["activation"],
            {
                "has_assigned_tasks": True,
                "has_schedules": True,
                "has_enabled_schedules": True,
                "assigned_task_count": 2,
                "scheduled_task_count": 1,
                "schedule_count": 2,
                "enabled_schedule_count": 1,
            },
        )

    def test_list_agents_exposes_activation_summary(self):
        agent = self._create_agent(name="listed activation summary")
        task = agent_store.get_agent_store().create_task(
            title="Listed scheduled task",
            assigned_agent_id=agent["id"],
            goal="Run from list",
        )
        schedule_store.get_schedule_store().create(
            task_id=task["id"],
            expression={"kind": "interval", "seconds": 3600},
        )

        response = self.client.get("/api/agent/agents")

        self.assertEqual(response.status_code, 200)
        listed = next(
            item for item in response.json()["agents"] if item["id"] == agent["id"]
        )
        self.assertEqual(listed["activation"]["assigned_task_count"], 1)
        self.assertEqual(listed["activation"]["schedule_count"], 1)
        self.assertTrue(listed["activation"]["has_enabled_schedules"])

    def test_list_agents_says_how_the_last_run_went_and_who_is_waiting(self):
        """The list is where the user decides which agent to open.

        Without these two fields it can only say "활성 N개", which reads the
        same whether the agent is working fine, has been failing every morning,
        or is parked waiting for an answer only the user can give.
        """
        agent = self._create_agent(name="listed run status")
        store = agent_store.get_agent_store()
        task = store.create_task(title="Run", assigned_agent_id=agent["id"])
        for status in ("completed", "waiting_for_user"):
            run = store.create_run(
                task_id=task["id"], agent_id=agent["id"], title="Run task"
            )
            store.update_run_status(run["id"], status)

        response = self.client.get("/api/agent/agents")

        self.assertEqual(response.status_code, 200, response.text)
        listed = next(
            item for item in response.json()["agents"] if item["id"] == agent["id"]
        )
        self.assertEqual(listed["last_run_status"], "waiting_for_user")
        self.assertEqual(listed["waiting_run_count"], 1)
        self.assertEqual(listed["active_run_count"], 0)
        self.assertIsNotNone(listed["last_fire_at"])


if __name__ == "__main__":
    unittest.main()
