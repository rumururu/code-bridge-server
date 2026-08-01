"""A restart must not retire a schedule.

Orchestration runs inside the server process, so a run marked `running` is
being driven by a coroutine here. Stop the server and that row is frozen
forever — and `skip_if_active` reads it as work in progress. Unlike a run
waiting on a human, a progressing run has no grace period, so the schedule
that owns it skips every fire from then on. One restart at the wrong moment
silently retires it.

Nothing has had a chance to run when the process starts, so anything still
progressing is stale by definition.
"""

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import tempfile

from agent import agent_store, run_reconciliation
from agent.run_reconciliation import reconcile_interrupted_runs
from core import database


class RunReconciliationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._original = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "reconcile.db"
        agent_store._agent_store = None
        database.init_db()
        self.addCleanup(self._restore)
        self.store = agent_store.get_agent_store()

    def _restore(self) -> None:
        agent_store._agent_store = None
        database.DB_PATH = self._original

    def _run_with_status(self, status: str):
        task = self.store.create_task(title="Nightly cycle", kind="ops")
        run = self.store.create_run(task_id=task["id"], title="Run task: Nightly cycle")
        self.store.update_run_status(run["id"], status)
        return task, run

    def test_running_run_is_closed(self):
        _, run = self._run_with_status("running")
        closed = reconcile_interrupted_runs()
        self.assertEqual([r["id"] for r in closed], [run["id"]])
        self.assertEqual(self.store.get_run(run["id"])["status"], "failed")

    def test_queued_and_starting_are_closed_too(self):
        for status in ("queued", "starting"):
            with self.subTest(status=status):
                _, run = self._run_with_status(status)
                reconcile_interrupted_runs()
                self.assertEqual(self.store.get_run(run["id"])["status"], "failed")

    def test_finished_runs_are_left_alone(self):
        _, completed = self._run_with_status("completed")
        _, failed = self._run_with_status("failed")
        self.assertEqual(reconcile_interrupted_runs(), [])
        self.assertEqual(self.store.get_run(completed["id"])["status"], "completed")
        self.assertEqual(self.store.get_run(failed["id"])["status"], "failed")

    def test_a_run_waiting_on_a_human_is_not_touched(self):
        # That one has its own grace period in the scheduler and may still be
        # answered from the phone; failing it here would throw away work.
        _, waiting = self._run_with_status("waiting_for_user")
        self.assertEqual(reconcile_interrupted_runs(), [])
        self.assertEqual(self.store.get_run(waiting["id"])["status"], "waiting_for_user")

    def test_unfinished_steps_are_closed_with_the_run(self):
        task, run = self._run_with_status("running")
        step = self.store.create_task_step(
            task_id=task["id"], run_id=run["id"], title="Cycle", status="running"
        )

        reconcile_interrupted_runs()

        steps = self.store.list_task_steps(task["id"])
        self.assertEqual(steps[0]["status"], "failed")
        self.assertIn("Interrupted", str(steps[0].get("output")))

    def test_it_says_why_in_the_timeline(self):
        _, run = self._run_with_status("running")
        reconcile_interrupted_runs()
        events = self.store.list_events(run_id=run["id"])
        reasons = [str(event) for event in events]
        self.assertTrue(
            any("Interrupted" in reason for reason in reasons),
            "the run gives no reason for having failed",
        )

    def test_running_twice_is_harmless(self):
        _, run = self._run_with_status("running")
        reconcile_interrupted_runs()
        self.assertEqual(reconcile_interrupted_runs(), [])
        self.assertEqual(self.store.get_run(run["id"])["status"], "failed")


if __name__ == "__main__":
    unittest.main()
