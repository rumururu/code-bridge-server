"""An unattended agent needs a way to say something that outlives the run.

"Check the disk every 6 hours and tell me if it is nearly full" has two halves.
The checking was already possible — a shell step runs a registered script for
no tokens. The telling was not: nothing carried a message off the machine, so
the Configurator wrote "sends an alert" into a system prompt and the run had no
way to honour it.

Delivery is the server's own capability rather than a script on purpose. A
script would make every agent that needs to speak depend on a notifier the user
must first write, vet and register, and the message would be exactly as
reliable as that script.
"""

import sys
import tempfile
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import notification_store
from agent.workflow_v2 import ALLOWED_STEP_TYPES, normalize_workflow_step
from core import database


class NotificationStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._original = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "notify.db"
        notification_store._notification_store = None
        database.init_db()
        self.addCleanup(self._restore)
        self.store = notification_store.get_notification_store()

    def _restore(self) -> None:
        notification_store._notification_store = None
        database.DB_PATH = self._original

    def test_a_notification_survives_the_run(self):
        created = self.store.create(title="Disk almost full", body="8% left")
        self.assertEqual(self.store.get(created["id"])["title"], "Disk almost full")

    def test_it_starts_unread(self):
        self.store.create(title="Disk almost full")
        self.assertEqual(self.store.unread_count(), 1)
        self.assertIsNone(self.store.list_notifications()[0]["read_at"])

    def test_reading_stamps_a_time_once(self):
        created = self.store.create(title="Disk almost full")
        first = self.store.mark_read(created["id"])["read_at"]
        second = self.store.mark_read(created["id"])["read_at"]
        # Re-reading must not move it, or "when did I see this" means nothing.
        self.assertEqual(first, second)
        self.assertEqual(self.store.unread_count(), 0)

    def test_unread_only_filters(self):
        first = self.store.create(title="One")
        self.store.create(title="Two")
        self.store.mark_read(first["id"])
        unread = self.store.list_notifications(unread_only=True)
        self.assertEqual([n["title"] for n in unread], ["Two"])

    def test_it_can_be_filtered_by_agent(self):
        self.store.create(title="From A", agent_id="agent_a")
        self.store.create(title="From B", agent_id="agent_b")
        mine = self.store.list_notifications(agent_id="agent_a")
        self.assertEqual([n["title"] for n in mine], ["From A"])

    def test_mark_all_read(self):
        self.store.create(title="One")
        self.store.create(title="Two")
        self.assertEqual(self.store.mark_all_read(), 2)
        self.assertEqual(self.store.unread_count(), 0)

    def test_a_titleless_notification_is_refused(self):
        # It would show as a blank row in the inbox, which is worse than the
        # step failing where the author can see it.
        with self.assertRaises(ValueError):
            self.store.create(title="   ")

    def test_an_unknown_level_falls_back_rather_than_failing(self):
        created = self.store.create(title="Disk", level="catastrophic")
        self.assertEqual(created["level"], "info")

    def test_timestamps_carry_a_zone(self):
        # SQLite writes naive UTC; a phone in +09:00 would read it as local and
        # place the notification nine hours in the past.
        created = self.store.create(title="Disk")
        self.assertTrue(created["created_at"].endswith("+00:00"))

    def test_run_and_task_are_recorded(self):
        created = self.store.create(
            title="Disk", run_id="run_1", task_id="task_1", agent_id="agent_1"
        )
        self.assertEqual(created["run_id"], "run_1")
        self.assertEqual(created["task_id"], "task_1")


class NotifyStepTest(unittest.TestCase):
    def test_notify_is_an_allowed_step_type(self):
        self.assertIn("notify", ALLOWED_STEP_TYPES)

    def test_the_payload_is_normalized(self):
        step = normalize_workflow_step(
            {
                "id": "tell_user",
                "type": "notify",
                "name": "Warn about disk",
                "notify": {"title": "Disk almost full", "body": "8% left", "level": "warning"},
            },
            index=1,
        )
        self.assertEqual(step["notify"]["title"], "Disk almost full")
        self.assertEqual(step["notify"]["level"], "warning")

    def test_a_missing_title_falls_back_to_the_step_name(self):
        # The author already wrote a name; an empty inbox row helps nobody.
        step = normalize_workflow_step(
            {"id": "tell", "type": "notify", "name": "Warn about disk"}, index=1
        )
        self.assertEqual(step["notify"]["title"], "Warn about disk")

    def test_an_unknown_level_is_not_passed_through(self):
        step = normalize_workflow_step(
            {
                "id": "tell",
                "type": "notify",
                "name": "Warn",
                "notify": {"level": "shouting"},
            },
            index=1,
        )
        self.assertEqual(step["notify"]["level"], "info")

    def test_a_shell_step_still_needs_a_registered_script(self):
        """The other half of the pair, and the reason it cannot be a command.

        A workflow that could carry a command line is remote code execution
        behind whatever standing rule lets it run unattended.
        """
        from agent.workflow_v2 import WorkflowNormalizationError

        with self.assertRaises(WorkflowNormalizationError):
            normalize_workflow_step({"id": "run", "type": "shell", "name": "Run"}, index=1)


if __name__ == "__main__":
    unittest.main()
