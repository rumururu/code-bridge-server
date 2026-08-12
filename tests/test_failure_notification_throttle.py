"""An agent that fails the same way every six hours must not flood the phone.

This file exists because of a real, live pattern: two scheduled agents were
each failing on every firing for days — the root cause was a UI automation
script that had stopped finding a button, nothing this server could do
anything about — and every failure pushed an identical notification. Four
times a day, every day. That is worse than useless: it trains the one reflex
that must never apply to a Code Bridge notification, which is to swipe it
away without reading it, and the swipe that eats today's fourth "X failed
again" is the same reflex that eats the one push that actually mattered —
a run genuinely parked waiting for a human, or an agent's source file gone
missing.

`_failure_notification_gate` (agent/task_orchestrator.py) is the fix: while
an agent keeps failing *consecutively*, it notifies at most once per day.
What "consecutively" means is the whole feature, and is exactly what this
file pins down:

- A first failure always notifies — nobody should wait a day to learn an
  agent just broke.
- A second failure of the *same* agent inside the 24h window stays quiet.
- The same failure after the window notifies again.
- A run that succeeds ends the streak: a failure right after a success is
  new information, not a repeat, and must notify immediately even inside
  what would otherwise be the quiet window.
- The throttle is per agent. One agent's ongoing trouble must never make a
  different agent's first failure go quiet.
- A read failure in the throttle's own state (run history or notification
  history) must fail *open* — notify — rather than silently eating a
  possibly-genuine first failure.
- Suppression must never touch the run's own recorded outcome: the run is
  still "failed" in the store either way, and the fact that a push was
  withheld is recorded as an event on the run rather than vanishing.

If this regresses toward "never throttles": the four-a-day flood comes back.
If it regresses toward "throttles too eagerly" (e.g. keying on error text, or
failing closed on a read error, or never resetting on success): a genuine
first failure, a fresh regression, or an unrelated agent's trouble goes
silently missing, which is worse than the flood it replaces.
"""

import os
import stat
import sys
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import mock

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import (  # noqa: E402
    agent_store,
    browser_session_store,
    push_notifier,
    script_store as script_store_module,
)
from agent.notification_store import get_notification_store  # noqa: E402
from agent.task_orchestrator import (  # noqa: E402
    execute_task_orchestration,
    prepare_task_orchestration,
)
from core import database  # noqa: E402
from core.database import get_db_connection  # noqa: E402
from pairing import pairing_service as pairing_service_module  # noqa: E402
from pairing.pairing_service import PairingService  # noqa: E402

_FAILING_SCRIPT = """#!/bin/bash
echo "still can't find the button" >&2
exit 1
"""

_OK_SCRIPT = """#!/bin/bash
echo "found it"
exit 0
"""


class _ThrottleTestBase(unittest.TestCase):
    """Same isolation strategy as test_run_terminal_state.py / test_wait_for_user_notifies.py."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self._original_db_path = database.DB_PATH
        database.DB_PATH = self.dir / "failure_throttle.db"
        agent_store._agent_store = None
        browser_session_store._browser_session_store = None
        script_store_module._script_store = None
        database.init_db()
        self.store = agent_store.get_agent_store()
        self.scripts = script_store_module.get_script_store()

        self._original_pairing_service = pairing_service_module._pairing_service
        pairing_service_module._pairing_service = PairingService(config_dir=self.dir / "pairing")

        self._env_patch = mock.patch.dict(
            os.environ,
            {push_notifier.SERVICE_ACCOUNT_ENV: str(self.dir / "no_such_key.json")},
        )
        self._env_patch.start()
        push_notifier.reset_for_tests()

        self.addCleanup(self._restore)

    def _restore(self):
        self._env_patch.stop()
        push_notifier.reset_for_tests()
        pairing_service_module._pairing_service = self._original_pairing_service
        agent_store._agent_store = None
        browser_session_store._browser_session_store = None
        script_store_module._script_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def _register_script(self, body: str, name: str) -> dict:
        path = self.dir / name
        path.write_text(body)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        return self.scripts.register(name=name, path=str(path))

    def _agent_and_task(self, flow_json: list[dict], *, agent_name: str) -> tuple[dict, dict]:
        agent = self.store.create_agent(
            name=agent_name,
            system_prompt="Do the thing.",
            provider_id="openai",
            flow_json=flow_json,
        )
        task = self.store.create_task(
            title=f"Run for {agent_name}",
            assigned_agent_id=agent["id"],
            goal="Do the thing.",
        )
        return agent, task

    def _failing_flow(self, script_name: str) -> list[dict]:
        script = self._register_script(_FAILING_SCRIPT, script_name)
        return [
            {
                "id": "cycle",
                "type": "shell",
                "name": "cycle",
                "script_id": script["id"],
                "on_failure": "abort",
            }
        ]

    def _ok_flow(self, script_name: str) -> list[dict]:
        script = self._register_script(_OK_SCRIPT, script_name)
        return [
            {
                "id": "cycle",
                "type": "shell",
                "name": "cycle",
                "script_id": script["id"],
            }
        ]

    def _fire(self, task_id: str) -> dict:
        """One firing of a task: prepare a run, then execute it to completion."""
        import asyncio

        prepared = prepare_task_orchestration(task_id, provider_id="openai", auto_start=False)
        assert prepared is not None
        with mock.patch(
            "agent.task_orchestrator.create_chat_session", self._fake_create_chat_session
        ), mock.patch("agent.task_orchestrator.stream_claude_turn", self._fake_stream):
            asyncio.run(execute_task_orchestration(prepared["execution"]))
        return prepared["run"]

    @staticmethod
    async def _fake_create_chat_session(**_kwargs):
        return object()

    @staticmethod
    async def _fake_stream(_sink, _session, **_kwargs):
        return True

    def _error_notifications(self) -> list[dict]:
        return get_notification_store().list_notifications(level="error")

    def _rewind_last_notification(self, *, hours: float) -> None:
        """Push the most recent notification's created_at back in time.

        Direct SQL, same pattern as test_schedule_store.py's `next_run_at`
        manipulation — there is no production path that back-dates a
        notification, so a test simulating "24h ago" has to reach into the
        DB directly.
        """
        notifications = get_notification_store().list_notifications(limit=1)
        assert notifications, "expected a notification to rewind"
        stamp = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
        with get_db_connection() as conn:
            conn.execute(
                "UPDATE agent_notifications SET created_at = ? WHERE id = ?",
                (stamp, notifications[0]["id"]),
            )
            conn.commit()


class FirstFailureAlwaysNotifiesTest(_ThrottleTestBase):
    def test_a_first_failure_notifies(self):
        _agent, task = self._agent_and_task(
            self._failing_flow("cycle.sh"), agent_name="lonely failer"
        )
        run = self.store.get_run(self._fire(task["id"])["id"])

        assert run is not None
        self.assertEqual(run["status"], "failed")
        self.assertEqual(len(self._error_notifications()), 1)


class ConsecutiveFailureThrottleTest(_ThrottleTestBase):
    def test_a_second_failure_within_24h_does_not_notify_again(self):
        _agent, task = self._agent_and_task(
            self._failing_flow("cycle.sh"), agent_name="broken script"
        )
        first_run = self.store.get_run(self._fire(task["id"])["id"])
        self.assertEqual(len(self._error_notifications()), 1)

        second_run = self.store.get_run(self._fire(task["id"])["id"])

        # The run itself is unaffected by the suppression: it still ends up
        # correctly recorded as failed, exactly as if the push had gone out.
        assert second_run is not None
        self.assertEqual(second_run["status"], "failed")
        self.assertIsNotNone(second_run["ended_at"])
        self.assertNotEqual(first_run["id"], second_run["id"])

        self.assertEqual(
            len(self._error_notifications()),
            1,
            "a second consecutive failure inside 24h must not push again",
        )

        # Suppression is discoverable on the run it happened to, not silent.
        events = [
            event
            for event in self.store.list_events(second_run["id"])
            if event["event_type"] == "notification.suppressed"
        ]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["app_event"]["kind"], "failed")

    def test_the_same_failure_after_24h_notifies_again(self):
        _agent, task = self._agent_and_task(
            self._failing_flow("cycle.sh"), agent_name="broken script"
        )
        self._fire(task["id"])
        self.assertEqual(len(self._error_notifications()), 1)

        self._rewind_last_notification(hours=25)

        third_run = self.store.get_run(self._fire(task["id"])["id"])
        assert third_run is not None
        self.assertEqual(third_run["status"], "failed")
        self.assertEqual(
            len(self._error_notifications()),
            2,
            "the same failure a day later is news again, not a repeat",
        )

    def test_a_success_between_two_failures_notifies_immediately(self):
        script = self._register_script(_FAILING_SCRIPT, "cycle.sh")
        ok_script = self._register_script(_OK_SCRIPT, "cycle_ok.sh")
        agent = self.store.create_agent(
            name="flaky agent",
            system_prompt="Do the thing.",
            provider_id="openai",
            flow_json=[
                {
                    "id": "cycle",
                    "type": "shell",
                    "name": "cycle",
                    "script_id": script["id"],
                    "on_failure": "abort",
                }
            ],
        )
        task = self.store.create_task(
            title="Run for flaky agent",
            assigned_agent_id=agent["id"],
            goal="Do the thing.",
        )

        first_run = self.store.get_run(self._fire(task["id"])["id"])
        assert first_run is not None
        self.assertEqual(first_run["status"], "failed")
        self.assertEqual(len(self._error_notifications()), 1)

        # Fix it: same agent, now succeeds. Still well inside the 24h window.
        self.store.update_agent(
            agent["id"],
            {
                "flow_json": [
                    {"id": "cycle", "type": "shell", "name": "cycle", "script_id": ok_script["id"]}
                ]
            },
        )
        second_run = self.store.get_run(self._fire(task["id"])["id"])
        assert second_run is not None
        self.assertEqual(second_run["status"], "completed")

        # Break it again, immediately. This is a new streak, not a repeat of
        # the first failure's quiet window, even though under a day has
        # passed since that first notification.
        self.store.update_agent(
            agent["id"],
            {
                "flow_json": [
                    {
                        "id": "cycle",
                        "type": "shell",
                        "name": "cycle",
                        "script_id": script["id"],
                        "on_failure": "abort",
                    }
                ]
            },
        )
        third_run = self.store.get_run(self._fire(task["id"])["id"])
        assert third_run is not None
        self.assertEqual(third_run["status"], "failed")
        self.assertEqual(
            len(self._error_notifications()),
            2,
            "a failure right after a success is new information, not a repeat",
        )


class PerAgentThrottleTest(_ThrottleTestBase):
    def test_two_different_agents_failing_do_not_throttle_each_other(self):
        _agent_one, task_one = self._agent_and_task(
            self._failing_flow("cycle_one.sh"), agent_name="agent one"
        )
        _agent_two, task_two = self._agent_and_task(
            self._failing_flow("cycle_two.sh"), agent_name="agent two"
        )

        self._fire(task_one["id"])
        self._fire(task_two["id"])

        self.assertEqual(
            len(self._error_notifications()),
            2,
            "one agent's failure streak must never silence another agent's first failure",
        )


class ThrottleStateReadFailureTest(_ThrottleTestBase):
    def test_a_run_history_read_failure_still_notifies(self):
        """The throttle's own state read failing must fail open, not closed.

        Failing toward "notify" merely restores today's known (and
        acceptable) behavior; failing toward "stay quiet" could silently
        eat a genuine first failure, which is the one outcome this whole
        feature exists to prevent.
        """
        _agent, task = self._agent_and_task(
            self._failing_flow("cycle.sh"), agent_name="broken script"
        )
        self._fire(task["id"])
        self.assertEqual(len(self._error_notifications()), 1)

        # Simulate the throttle's run-history read breaking on the *second*
        # failure, which would otherwise be suppressed.
        with mock.patch.object(
            self.store, "list_runs", side_effect=RuntimeError("db is on fire")
        ):
            second_run = self.store.get_run(self._fire(task["id"])["id"])

        assert second_run is not None
        self.assertEqual(
            second_run["status"],
            "failed",
            "a throttle-state read failure must not change the run's own outcome",
        )
        self.assertEqual(
            len(self._error_notifications()),
            2,
            "a broken throttle read must notify rather than silently suppress",
        )


if __name__ == "__main__":
    unittest.main()
