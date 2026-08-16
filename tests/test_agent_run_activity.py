"""An agent's run history has to reach the client.

The payload carried `next_fire_at` and an activation summary but never
`last_fire_at`, so a phone rendered "no runs yet" for an agent that had run
sixty times. The history was there the whole time — it just stopped at the
server.
"""

import sys
import tempfile
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store
from core import database
from routes.agents import _agent_run_activity


class AgentRunActivityTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._original = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "activity.db"
        agent_store._agent_store = None
        database.init_db()
        self.addCleanup(self._restore)
        self.store = agent_store.get_agent_store()
        self.agent = self.store.create_agent(name="Cycle", system_prompt="")

    def _restore(self) -> None:
        agent_store._agent_store = None
        database.DB_PATH = self._original

    def _run(self, status: str):
        task = self.store.create_task(title="Cycle", assigned_agent_id=self.agent["id"])
        run = self.store.create_run(
            task_id=task["id"], agent_id=self.agent["id"], title="Run task: Cycle"
        )
        self.store.update_run_status(run["id"], status)
        return run

    def test_never_run_reports_nothing(self):
        activity = _agent_run_activity(self.agent["id"])
        self.assertIsNone(activity["last_fire_at"])
        self.assertEqual(activity["active_run_count"], 0)

    def test_a_completed_run_becomes_last_fire_at(self):
        self._run("completed")
        self.assertIsNotNone(_agent_run_activity(self.agent["id"])["last_fire_at"])

    def test_a_failed_run_still_counts_as_having_run(self):
        # It ran; it just did not work. "No runs yet" would be a lie.
        self._run("failed")
        self.assertIsNotNone(_agent_run_activity(self.agent["id"])["last_fire_at"])

    def test_active_runs_are_counted(self):
        self._run("running")
        self._run("completed")
        self.assertEqual(_agent_run_activity(self.agent["id"])["active_run_count"], 1)

    def test_never_run_reports_no_status(self):
        activity = _agent_run_activity(self.agent["id"])
        self.assertIsNone(activity["last_run_status"])
        self.assertEqual(activity["waiting_run_count"], 0)

    def test_the_last_run_reports_how_it_went(self):
        """"Ran 3 hours ago" and "ran 3 hours ago and failed" are different agents.

        `last_fire_at` alone cannot tell sixty clean runs from sixty failures,
        so a list built only on it shows a broken agent exactly as it shows a
        working one.
        """
        for status in ("completed", "failed"):
            with self.subTest(status=status):
                self._run(status)
                activity = _agent_run_activity(self.agent["id"])
                self.assertEqual(activity["last_run_status"], status)
                self.assertIsNotNone(activity["last_fire_at"])

    def test_a_run_waiting_on_the_user_is_not_counted_as_active(self):
        """Waiting is the state the user has to act on; active is the one they don't.

        Counting them together is what made the one agent that needed an answer
        indistinguishable from every agent that was simply busy — it read as
        "활성 1개", the same words a healthy run produces, so nobody opened it.
        """
        self._run("running")
        self._run("waiting_for_user")

        activity = _agent_run_activity(self.agent["id"])

        self.assertEqual(activity["waiting_run_count"], 1)
        self.assertEqual(activity["active_run_count"], 1)
        self.assertEqual(activity["last_run_status"], "waiting_for_user")

    def test_every_waiting_status_the_scheduler_knows_is_counted(self):
        # The two sets describe the same runs; a status parked by the scheduler
        # but unknown here would be a run waiting invisibly.
        from agent.scheduler import _WAITING_RUN_STATUSES as scheduler_waiting

        from routes.agents import _WAITING_RUN_STATUSES as routes_waiting

        self.assertEqual(set(routes_waiting), set(scheduler_waiting))

    def test_another_agents_runs_are_not_borrowed(self):
        other = self.store.create_agent(name="Other", system_prompt="")
        self._run("completed")
        self.assertIsNone(_agent_run_activity(other["id"])["last_fire_at"])


if __name__ == "__main__":
    unittest.main()
