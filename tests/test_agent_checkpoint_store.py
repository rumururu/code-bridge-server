import sys
import tempfile
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store  # noqa: E402
from core import database  # noqa: E402


class AgentCheckpointStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "checkpoint_store.db"
        agent_store._agent_store = None
        database.init_db()
        self.store = agent_store.get_agent_store()

    def tearDown(self):
        agent_store._agent_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def _waiting_fixture(self):
        agent = self.store.create_agent(
            name="handoff bot",
            system_prompt="Ask for help when blocked.",
            provider_id="openai",
        )
        task = self.store.create_task(
            title="Manual login",
            assigned_agent_id=agent["id"],
            goal="Check login",
        )
        run = self.store.create_run(
            task_id=task["id"],
            agent_id=agent["id"],
            provider_id="openai",
            title="Run manual login",
        )
        self.store.update_task(task["id"], {"run_id": run["id"]})
        step = self.store.create_task_step(
            task_id=task["id"],
            run_id=run["id"],
            title="Login handoff",
            status="waiting_for_user",
            input={
                "workflow_step_id": "login_check",
                "workflow_type": "manual_handoff",
                "on_failure": {"type": "ask_user", "resume": "same_step"},
            },
            output={
                "checkpoint": {
                    "status": "waiting_for_user",
                    "reason": "manual_handoff",
                    "prompt": "Log in, then continue.",
                    "workflow_step_id": "login_check",
                    "resume": "same_step",
                    "resume_step_id": None,
                    "required_user_action": "Complete login in the browser.",
                    "created_at": "2026-06-09T00:00:00+00:00",
                }
            },
        )
        self.store.update_task(task["id"], {"status": "waiting_for_user"})
        self.store.update_run_status(run["id"], "waiting_for_user")
        return agent, task, run, step

    def test_get_task_checkpoint_returns_waiting_step_contract(self):
        _agent, task, run, step = self._waiting_fixture()

        checkpoint = self.store.get_task_checkpoint(task["id"])

        self.assertIsNotNone(checkpoint)
        assert checkpoint is not None
        self.assertEqual(checkpoint["task"]["id"], task["id"])
        self.assertEqual(checkpoint["run"]["id"], run["id"])
        self.assertEqual(checkpoint["step"]["id"], step["id"])
        self.assertEqual(checkpoint["checkpoint"]["workflow_step_id"], "login_check")
        self.assertEqual(checkpoint["checkpoint"]["resume"], "same_step")

    def test_get_task_checkpoint_returns_null_checkpoint_when_not_waiting(self):
        task = self.store.create_task(title="No checkpoint")

        checkpoint = self.store.get_task_checkpoint(task["id"])

        self.assertIsNotNone(checkpoint)
        assert checkpoint is not None
        self.assertIsNone(checkpoint["checkpoint"])
        self.assertIsNone(checkpoint["step"])

    def test_append_step_user_response_persists_event_and_optional_memory(self):
        agent, task, run, step = self._waiting_fixture()

        result = self.store.append_step_user_response(
            task_id=task["id"],
            step_id=step["id"],
            message="Login completed.",
            metadata={"source": "test"},
            remember=True,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["response"]["message"], "Login completed.")
        self.assertEqual(result["memory"]["agent_id"], agent["id"])
        updated_step = self.store.get_task_step(step["id"])
        assert updated_step is not None
        self.assertEqual(
            updated_step["output"]["last_user_response"]["metadata"],
            {"source": "test"},
        )
        events = self.store.list_events(run["id"])
        self.assertIn(
            "task.step.user_response",
            [event["event_type"] for event in events],
        )

    def test_append_step_user_response_rejects_non_waiting_step(self):
        task = self.store.create_task(title="Running task")
        step = self.store.create_task_step(
            task_id=task["id"],
            title="Running",
            status="running",
        )
        assert step is not None

        with self.assertRaises(agent_store.AgentStoreConflictError):
            self.store.append_step_user_response(
                task_id=task["id"],
                step_id=step["id"],
                message="done",
            )


if __name__ == "__main__":
    unittest.main()
