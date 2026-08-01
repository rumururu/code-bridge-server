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

    def test_another_agents_runs_are_not_borrowed(self):
        other = self.store.create_agent(name="Other", system_prompt="")
        self._run("completed")
        self.assertIsNone(_agent_run_activity(other["id"])["last_fire_at"])


if __name__ == "__main__":
    unittest.main()
