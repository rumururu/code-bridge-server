"""A run's detail must show that run's steps, and nothing else.

An agent owns exactly one task, and every run of that agent appends its steps
to that same task. The dashboard's run detail read `/tasks/{task_id}/steps`,
so clicking any single run rendered the agent's entire history: measured on a
live install, one click on a two-step run drew 386 steps from 193 runs going
back six weeks, with the oldest run's stale `running` status at the top. The
phone's runs panel (`agent_runs_panel.dart`) read the same task-scoped path.

`run_id` was already stored on every step row; only a read scoped to it was
missing. These tests pin the scoped read and the two callers that must use it.
"""

from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

REPO_ROOT = SERVER_DIR.parent

from agent import agent_store  # noqa: E402
from core import database  # noqa: E402
from routes import agents  # noqa: E402
from routes.deps import verify_api_key  # noqa: E402


class RunScopedStepsRouteTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "run_scoped_steps_test.db"
        agent_store._agent_store = None

        app = FastAPI()
        app.include_router(agents.router)
        app.dependency_overrides[verify_api_key] = lambda: "test-api-key"
        self.client = TestClient(app)

        # One task, two runs of it — the shape every scheduled agent has.
        self.store = agent_store.get_agent_store()
        task = self.store.create_task(title="Daily cycle")
        self.task_id = task["id"]
        self.first_run = self.store.create_run(title="run 1", task_id=self.task_id)["id"]
        self.second_run = self.store.create_run(title="run 2", task_id=self.task_id)["id"]
        for run_id, titles in (
            (self.first_run, ("yesterday: collect", "yesterday: publish")),
            (self.second_run, ("today: collect", "today: publish", "today: notify")),
        ):
            for title in titles:
                self.store.create_task_step(
                    task_id=self.task_id, run_id=run_id, title=title
                )

    def tearDown(self):
        agent_store._agent_store = None
        database.DB_PATH = self._original_db_path

    def test_run_steps_returns_only_that_runs_steps(self):
        body = self.client.get(f"/api/agent/runs/{self.second_run}/steps").json()
        titles = [step["title"] for step in body["steps"]]

        self.assertEqual(
            titles, ["today: collect", "today: publish", "today: notify"]
        )
        self.assertTrue(all(step["run_id"] == self.second_run for step in body["steps"]))
        self.assertEqual(body["run"]["id"], self.second_run)

    def test_the_other_run_is_unaffected(self):
        body = self.client.get(f"/api/agent/runs/{self.first_run}/steps").json()
        self.assertEqual(
            [step["title"] for step in body["steps"]],
            ["yesterday: collect", "yesterday: publish"],
        )

    def test_task_scoped_read_still_returns_the_whole_history(self):
        """The task path is not the bug and keeps its meaning — the run
        detail simply must not be the thing that reads it."""
        body = self.client.get(f"/api/agent/tasks/{self.task_id}/steps").json()
        self.assertEqual(len(body["steps"]), 5)

    def test_unknown_run_is_a_404_not_an_empty_list(self):
        response = self.client.get("/api/agent/runs/run_does_not_exist/steps")
        self.assertEqual(response.status_code, 404)

    def test_a_run_with_no_recorded_steps_answers_empty(self):
        bare = self.store.create_run(title="never started", task_id=self.task_id)["id"]
        body = self.client.get(f"/api/agent/runs/{bare}/steps").json()
        self.assertEqual(body["steps"], [])


class RunScopedStepIndexTest(unittest.TestCase):
    """The filter needs an index, and it has to reach existing databases.

    The index was first added to the block that creates the table — which is
    inside a migration every existing database has already recorded, so it
    would have reached nothing. Pinned here against a database built the way a
    real one was: migrated, not created fresh from today's code.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "index_test.db"
        agent_store._agent_store = None
        self.addCleanup(setattr, database, "DB_PATH", self._original_db_path)

    def _indexes(self):
        with database.get_db_connection() as conn:
            return {
                row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type = 'index' AND tbl_name = 'agent_task_steps'"
                )
            }

    def test_index_exists_after_init(self):
        database.init_db()
        self.assertIn("idx_agent_task_steps_run_sequence", self._indexes())

    def test_a_database_that_predates_the_index_still_gains_it(self):
        database.init_db()
        with database.get_db_connection() as conn:
            conn.execute("DROP INDEX idx_agent_task_steps_run_sequence")
            conn.execute(
                "DELETE FROM schema_migrations WHERE version = ?",
                (database.RUN_SCOPED_STEP_INDEX_SCHEMA_VERSION,),
            )
            conn.commit()
        self.assertNotIn("idx_agent_task_steps_run_sequence", self._indexes())

        database.init_db()
        self.assertIn("idx_agent_task_steps_run_sequence", self._indexes())


class DashboardMirrorsTheRunScopedRead(unittest.TestCase):
    def test_dashboard_router_exposes_runs_steps(self):
        from routes import dashboard_agents

        paths = {route.path for route in dashboard_agents.router.routes}
        self.assertIn("/api/dashboard/agent/runs/{run_id}/steps", paths)


class CallersUseTheRunScopedPath(unittest.TestCase):
    """The two run-detail views must not go back to the task-scoped read.

    Both defects were one string each, in code whose surrounding names all say
    "run" — which is exactly why they survived review.
    """

    def test_dashboard_run_detail_reads_by_run(self):
        source = (
            REPO_ROOT / "server/dashboard/templates/agents.html"
        ).read_text(encoding="utf-8")
        select_run = re.search(
            r"async function selectRun\(runId\) \{(.*?)\n        \}", source, re.S
        )
        self.assertIsNotNone(select_run, "selectRun() not found in agents.html")
        # Comments are allowed to name the old path (this one does, to say
        # what went wrong); the code must not call it.
        code = "\n".join(
            line for line in select_run.group(1).splitlines()
            if not line.strip().startswith("//")
        )
        self.assertIn("api(`/runs/${encodeURIComponent(run.id)}/steps`)", code)
        self.assertNotIn("/tasks/", code)

    def test_phone_runs_panel_reads_by_run(self):
        service = (
            REPO_ROOT / "lib/services/agent_management_service.dart"
        ).read_text(encoding="utf-8")
        self.assertIn("'/api/agent/runs/$runId/steps'", service)
        self.assertNotIn("_taskStepsPath", service)

        panel = (
            REPO_ROOT / "lib/screens/agent/widgets/agent_runs_panel.dart"
        ).read_text(encoding="utf-8")
        self.assertIn("listRunSteps(run.id)", panel)
        self.assertNotIn("listRunSteps(taskId)", panel)


if __name__ == "__main__":
    unittest.main()
