"""Route tests for /api/agent/{tasks/{id}/schedules, schedules, ...}."""

from __future__ import annotations

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

from agent import agent_store, schedule_store
from core import database
from routes import agents
from routes.deps import verify_api_key


class ScheduleRoutesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "code_bridge_schedule_test.db"
        agent_store._agent_store = None
        schedule_store._store = None
        # Some routes (GET /schedules/{id}) touch the schedule_store before any
        # task is created, so init_db must run independently of agent_store
        # construction.
        database.init_db()

        app = FastAPI()
        app.include_router(agents.router)
        app.dependency_overrides[verify_api_key] = lambda: "test-api-key"
        self.client = TestClient(app)

    def tearDown(self):
        agent_store._agent_store = None
        schedule_store._store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    # ----------------------------------------------------------------- helpers

    def _create_task(self, title: str = "Sched test"):
        response = self.client.post(
            "/api/agent/tasks",
            json={"title": title, "kind": "general"},
        )
        self.assertEqual(response.status_code, 200)
        return response.json()["task"]

    def _create_schedule(
        self,
        task_id: str,
        *,
        expression=None,
        name: str | None = None,
    ):
        body = {
            "expression": expression or {"kind": "interval", "seconds": 3600},
        }
        if name is not None:
            body["name"] = name
        response = self.client.post(
            f"/api/agent/tasks/{task_id}/schedules",
            json=body,
        )
        self.assertEqual(response.status_code, 200)
        return response.json()["schedule"]

    # ------------------------------------------------------------------- tests

    def test_post_schedule_persists_all_fields(self):
        task = self._create_task()

        response = self.client.post(
            f"/api/agent/tasks/{task['id']}/schedules",
            json={
                "expression": {"kind": "interval", "seconds": 3600},
                "name": "Hourly",
                "provider_id": "claude",
                "model": "claude-sonnet",
                "cwd": "/tmp",
                "prompt": "Run it",
                "capabilities": ["cap1"],
                "enabled": True,
                "skip_if_active": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        schedule = response.json()["schedule"]
        self.assertEqual(schedule["task_id"], task["id"])
        self.assertEqual(schedule["name"], "Hourly")
        self.assertEqual(
            schedule["expression"],
            {"kind": "interval", "seconds": 3600},
        )
        self.assertEqual(schedule["provider_id"], "claude")
        self.assertEqual(schedule["capabilities"], ["cap1"])
        self.assertTrue(schedule["enabled"])

    def test_post_schedule_404_for_unknown_task(self):
        response = self.client.post(
            "/api/agent/tasks/missing/schedules",
            json={"expression": {"kind": "interval", "seconds": 3600}},
        )
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.json()["detail"])

    def test_post_schedule_400_for_invalid_expression(self):
        task = self._create_task()
        # missing expression dict
        response = self.client.post(
            f"/api/agent/tasks/{task['id']}/schedules",
            json={"expression": "not a dict"},
        )
        self.assertEqual(response.status_code, 400)

    def test_post_schedule_400_for_too_short_interval(self):
        task = self._create_task()
        response = self.client.post(
            f"/api/agent/tasks/{task['id']}/schedules",
            json={"expression": {"kind": "interval", "seconds": 10}},
        )
        self.assertEqual(response.status_code, 400)

    def test_list_for_task_returns_only_that_task(self):
        task_a = self._create_task("A")
        task_b = self._create_task("B")
        self._create_schedule(task_a["id"], name="schedA1")
        self._create_schedule(task_a["id"], name="schedA2")
        self._create_schedule(task_b["id"], name="schedB1")

        response = self.client.get(f"/api/agent/tasks/{task_a['id']}/schedules")
        self.assertEqual(response.status_code, 200)
        names = sorted(s["name"] for s in response.json()["schedules"])
        self.assertEqual(names, ["schedA1", "schedA2"])

    def test_list_for_unknown_task_returns_404(self):
        response = self.client.get("/api/agent/tasks/missing/schedules")
        self.assertEqual(response.status_code, 404)

    def test_list_all_returns_every_schedule(self):
        task = self._create_task()
        self._create_schedule(task["id"], name="s1")
        self._create_schedule(task["id"], name="s2")

        response = self.client.get("/api/agent/schedules")
        self.assertEqual(response.status_code, 200)
        names = sorted(s["name"] for s in response.json()["schedules"])
        self.assertEqual(names, ["s1", "s2"])

    def test_list_all_enabled_only_filters_disabled(self):
        task = self._create_task()
        on = self._create_schedule(task["id"], name="on")
        off = self._create_schedule(task["id"], name="off")
        self.client.patch(
            f"/api/agent/schedules/{off['id']}",
            json={"enabled": False},
        )

        response = self.client.get(
            "/api/agent/schedules",
            params={"enabled_only": "true"},
        )
        self.assertEqual(response.status_code, 200)
        names = [s["name"] for s in response.json()["schedules"]]
        self.assertIn("on", names)
        self.assertNotIn("off", names)

    def test_get_schedule_by_id(self):
        task = self._create_task()
        created = self._create_schedule(task["id"], name="get me")

        response = self.client.get(f"/api/agent/schedules/{created['id']}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["schedule"]["id"], created["id"])

    def test_get_schedule_404_for_missing(self):
        response = self.client.get("/api/agent/schedules/missing")
        self.assertEqual(response.status_code, 404)

    def test_patch_schedule_updates_fields(self):
        task = self._create_task()
        created = self._create_schedule(task["id"], name="original")

        response = self.client.patch(
            f"/api/agent/schedules/{created['id']}",
            json={"name": "renamed", "enabled": False},
        )
        self.assertEqual(response.status_code, 200)
        updated = response.json()["schedule"]
        self.assertEqual(updated["name"], "renamed")
        self.assertFalse(updated["enabled"])

    def test_patch_schedule_404_for_missing(self):
        response = self.client.patch(
            "/api/agent/schedules/missing",
            json={"enabled": False},
        )
        self.assertEqual(response.status_code, 404)

    def test_patch_schedule_400_for_unknown_field(self):
        task = self._create_task()
        created = self._create_schedule(task["id"])

        response = self.client.patch(
            f"/api/agent/schedules/{created['id']}",
            json={"forbidden_field": "x"},
        )
        self.assertEqual(response.status_code, 400)

    def test_delete_schedule_removes_row(self):
        task = self._create_task()
        created = self._create_schedule(task["id"])

        response = self.client.delete(f"/api/agent/schedules/{created['id']}")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["deleted"])

        # second delete is 404
        response = self.client.delete(f"/api/agent/schedules/{created['id']}")
        self.assertEqual(response.status_code, 404)

    def test_trigger_now_404_for_missing(self):
        response = self.client.post("/api/agent/schedules/missing/trigger")
        self.assertEqual(response.status_code, 404)

    def test_trigger_now_invokes_fire_path(self):
        task = self._create_task()
        created = self._create_schedule(task["id"])

        with patch(
            "agent.scheduler._fire_schedule",
            new_callable=lambda: __import__(
                "unittest.mock", fromlist=["AsyncMock"]
            ).AsyncMock(),
        ) as mocked:
            response = self.client.post(
                f"/api/agent/schedules/{created['id']}/trigger"
            )

        self.assertEqual(response.status_code, 200)
        mocked.assert_called_once()
        # Returned schedule is the post-fire state pulled from the store
        self.assertEqual(response.json()["schedule"]["id"], created["id"])

    def test_scheduler_tick_endpoint_returns_zero_when_no_due(self):
        response = self.client.post("/api/agent/scheduler/tick")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["fired"], 0)
