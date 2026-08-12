"""A run that parks for a human must actually tell someone.

Before this, `_wait_for_user_step` (agent/task_orchestrator.py) parked a run
by writing DB rows and a `task.step.waiting_for_user` event, full stop. It
never touched `notification_store` or the push path that `notify` workflow
steps already use. The only way to learn a scheduled 3am run had stalled on
a login prompt was to open the app and happen to notice a badge in the task
list — which defeats the entire point of running it unattended in the first
place.

If this regresses: a parked run goes invisible again, exactly as above. Three
narrower regressions this file also guards against, because each one made an
earlier draft of the fix worse than doing nothing:

- The notification/push failing must never fail or further stall the run —
  it is already parked; making that outcome worse is the one unacceptable
  failure mode (this mirrors test_notify_step_push.py's coverage of the
  `notify` step's own push path).
- The message must say *why* it is waiting (login vs. captcha vs. approval,
  etc.) — "a task needs you" with no reason is not actionable at 3am.
- Resuming a run that immediately re-parks on the same step for the same
  reason (e.g. a browser handoff where the login page is still showing) must
  not re-notify — that would turn one stuck run into a stream of pings.

A fourth regression, added later, needed a second look rather than reusing
the third one's fix: the same-run dedup above only compares against the task
metadata's `active_checkpoint`, which is cleared when a run finishes through
`_finish_workflow_execution` — but a *scheduled* run that stalls waiting on a
human instead gets abandoned by `agent.scheduler._abandon_stalled_run`, which
marks it failed directly and never clears that checkpoint or goes through
that finish path. A schedule that lands on the same wait reason every time it
fires, getting abandoned in between, is therefore a *new* run every time as
far as the same-run dedup is concerned, and would notify on every firing —
the identical flood `_failure_notification_gate`
(see `test_failure_notification_throttle.py`) exists to stop on the failure
side, just triggered by a stuck human handoff instead of a stuck script.
`_wait_notification_gate` throttles that cross-run repeat the same way: once
per agent per day while the streak holds, immediately again once the
previous run for that agent actually finished successfully.
"""

import asyncio
import os
import sys
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import mock

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store, browser_session_store, push_notifier  # noqa: E402
from agent.notification_store import get_notification_store  # noqa: E402
from agent.task_orchestrator import (  # noqa: E402
    _wait_reason_label,
    execute_task_orchestration,
    prepare_task_orchestration,
    resume_task_orchestration,
)
from core import database  # noqa: E402
from core.database import get_db_connection  # noqa: E402
from pairing import pairing_service as pairing_service_module  # noqa: E402
from pairing.pairing_service import PairingService  # noqa: E402


class _OrchestratorTestBase(unittest.TestCase):
    """Same isolation strategy as test_notify_step_push.py: fresh DB + pairing
    singleton per test, push forced off unless a test explicitly registers a
    token, so no test can touch a real operator's data or a real FCM key."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self._original_db_path = database.DB_PATH
        database.DB_PATH = self.dir / "wait_for_user.db"
        agent_store._agent_store = None
        browser_session_store._browser_session_store = None
        database.init_db()
        self.store = agent_store.get_agent_store()

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
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def _start_manual_handoff_task(self, agent_name: str = "handoff bot") -> dict:
        agent = self.store.create_agent(
            name=agent_name,
            system_prompt="Hand off to a human.",
            provider_id="openai",
            flow_json=[
                {
                    "id": "need_a_human",
                    "type": "manual_handoff",
                    "name": "Log into the portal",
                }
            ],
        )
        task = self.store.create_task(
            title="Run handoff workflow",
            assigned_agent_id=agent["id"],
            goal="Log in and continue.",
        )
        result = prepare_task_orchestration(task["id"], provider_id="openai", auto_start=False)
        assert result is not None
        asyncio.run(execute_task_orchestration(result["execution"]))
        return task

    def _park_a_new_run_for_agent(self, agent_id: str, *, title: str) -> dict:
        """Fire a brand-new task/run for an *existing* agent, and let it park.

        Distinct from `_start_manual_handoff_task`: that creates a fresh
        agent every time, which is the wrong shape for testing a throttle
        that is keyed on the agent — this reuses one agent across several
        independent runs, the way a schedule re-fires the same agent's task
        over and over.
        """
        task = self.store.create_task(
            title=title,
            assigned_agent_id=agent_id,
            goal="Log in and continue.",
        )
        result = prepare_task_orchestration(task["id"], provider_id="openai", auto_start=False)
        assert result is not None
        asyncio.run(execute_task_orchestration(result["execution"]))
        return task

    def _resume_without_answering(self, task_id: str) -> None:
        resumed = resume_task_orchestration(task_id)
        assert resumed is not None
        asyncio.run(execute_task_orchestration(resumed["execution"]))


class ParkedRunStoresANotificationTest(_OrchestratorTestBase):
    def test_the_park_stores_a_notification_naming_the_reason(self):
        task = self._start_manual_handoff_task()

        steps = self.store.list_task_steps(task["id"])
        self.assertEqual(steps[0]["status"], "waiting_for_user")

        notifications = get_notification_store().list_notifications()
        self.assertEqual(len(notifications), 1)
        notification = notifications[0]

        # Says why, not just "needs you" — the reason for a manual_handoff
        # step is the literal step type name.
        self.assertIn(_wait_reason_label("manual_handoff"), notification["title"])
        self.assertEqual(notification["task_id"], task["id"])
        self.assertIsNotNone(notification["body"])

    def test_a_parked_run_is_not_info_level(self):
        # It is not "here is an FYI" — it is a run that has stopped making
        # progress until a human acts.
        self._start_manual_handoff_task()
        notification = get_notification_store().list_notifications()[0]
        self.assertNotEqual(notification["level"], "info")

    def test_known_reasons_get_a_readable_label_not_a_bare_code(self):
        for reason in ("login_required", "captcha_or_bot_challenge", "approval_gate"):
            with self.subTest(reason=reason):
                label = _wait_reason_label(reason)
                self.assertNotEqual(label, reason)
                self.assertNotIn("_", label)

    def test_an_unknown_reason_still_reads_as_words_not_a_raw_code(self):
        label = _wait_reason_label("some_brand_new_wait_reason")
        self.assertNotIn("_", label)


class ParkNotificationPushFailureTest(_OrchestratorTestBase):
    def test_a_push_failure_does_not_fail_or_further_stall_the_park(self):
        pairing = pairing_service_module.get_pairing_service()
        pairing._api_keys["client_1"] = {"device_name": "Phone", "api_key_sha256": "hash"}
        pairing.register_push_token("client_1", "fcm-token-1")

        with mock.patch(
            "agent.push_notifier.send_to_tokens",
            side_effect=RuntimeError("FCM is unreachable"),
        ):
            task = self._start_manual_handoff_task()

        # The run is parked exactly as it would be without push configured at
        # all — a push failure must not additionally fail the step or run.
        steps = self.store.list_task_steps(task["id"])
        self.assertEqual(steps[0]["status"], "waiting_for_user")
        updated_task = self.store.get_task(task["id"])
        self.assertEqual(updated_task["status"], "waiting_for_user")
        self.assertEqual(len(get_notification_store().list_notifications()), 1)

    def test_a_notification_store_failure_does_not_escape_and_stall_the_run(self):
        with mock.patch(
            "agent.notification_store.get_notification_store",
            side_effect=RuntimeError("db is on fire"),
        ):
            task = self._start_manual_handoff_task()

        # Best-effort means best-effort even when the store itself is the
        # part that breaks: the park must still have happened.
        steps = self.store.list_task_steps(task["id"])
        self.assertEqual(steps[0]["status"], "waiting_for_user")


class RepeatParkDoesNotSpamTest(_OrchestratorTestBase):
    def test_resuming_into_the_same_unanswered_step_does_not_renotify(self):
        task = self._start_manual_handoff_task()
        self.assertEqual(len(get_notification_store().list_notifications()), 1)

        # Resume without ever answering the checkpoint: the workflow loop
        # finds the step still `waiting_for_user` with no user response and
        # re-runs the same manual_handoff branch, parking on the same
        # run/step/reason a second time.
        self._resume_without_answering(task["id"])
        steps = self.store.list_task_steps(task["id"])
        self.assertEqual(steps[0]["status"], "waiting_for_user")

        self.assertEqual(
            len(get_notification_store().list_notifications()),
            1,
            "re-parking the same step for the same reason must not send a second notification",
        )

    def test_a_genuinely_new_park_on_a_different_task_still_notifies(self):
        # Dedup is scoped to a run/step, not global — an unrelated task
        # parking must not be swallowed by another task's history.
        self._start_manual_handoff_task(agent_name="handoff bot one")
        self._start_manual_handoff_task(agent_name="handoff bot two")
        self.assertEqual(len(get_notification_store().list_notifications()), 2)


class CrossRunParkThrottleTest(_OrchestratorTestBase):
    """The gap `RepeatParkDoesNotSpamTest` cannot see: separate *runs*.

    `test_resuming_into_the_same_unanswered_step_does_not_renotify` above
    proves the same run re-parking on the same step doesn't spam. These
    tests are about the case the module docstring describes: a schedule that
    keeps firing a brand-new run for the same agent, and that run parking on
    the same wait every time. `_wait_notification_gate` is what stops that
    from becoming the identical four-a-day flood the failure throttle exists
    to stop.
    """

    def _create_handoff_agent(self, name: str = "handoff bot") -> dict:
        return self.store.create_agent(
            name=name,
            system_prompt="Hand off to a human.",
            provider_id="openai",
            flow_json=[
                {
                    "id": "need_a_human",
                    "type": "manual_handoff",
                    "name": "Log into the portal",
                }
            ],
        )

    def test_a_second_park_of_the_same_agent_in_a_new_run_does_not_renotify_within_24h(self):
        agent = self._create_handoff_agent()
        self._park_a_new_run_for_agent(agent["id"], title="first firing")
        self.assertEqual(len(get_notification_store().list_notifications()), 1)

        second_task = self._park_a_new_run_for_agent(agent["id"], title="second firing")

        # The park itself is unaffected — the run and step still end up
        # correctly recorded as waiting, exactly as if the push had gone out.
        steps = self.store.list_task_steps(second_task["id"])
        self.assertEqual(steps[0]["status"], "waiting_for_user")
        self.assertEqual(
            len(get_notification_store().list_notifications()),
            1,
            "a same-agent repeat park in a new run must not push again inside 24h",
        )

        run_id = self.store.get_task(second_task["id"])["run_id"]
        events = [
            event
            for event in self.store.list_events(run_id)
            if event["event_type"] == "notification.suppressed"
        ]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["app_event"]["kind"], "waiting_for_user")

    def test_the_same_agent_parking_again_after_24h_notifies_again(self):
        agent = self._create_handoff_agent()
        self._park_a_new_run_for_agent(agent["id"], title="first firing")
        self.assertEqual(len(get_notification_store().list_notifications()), 1)

        notifications = get_notification_store().list_notifications(limit=1)
        stamp = (datetime.now(UTC) - timedelta(hours=25)).isoformat()
        with get_db_connection() as conn:
            conn.execute(
                "UPDATE agent_notifications SET created_at = ? WHERE id = ?",
                (stamp, notifications[0]["id"]),
            )
            conn.commit()

        self._park_a_new_run_for_agent(agent["id"], title="third firing")
        self.assertEqual(
            len(get_notification_store().list_notifications()),
            2,
            "the same agent parking again a day later is news again, not a repeat",
        )

    def test_two_different_agents_parking_do_not_throttle_each_other(self):
        agent_one = self._create_handoff_agent(name="handoff bot one")
        agent_two = self._create_handoff_agent(name="handoff bot two")

        self._park_a_new_run_for_agent(agent_one["id"], title="agent one's task")
        self._park_a_new_run_for_agent(agent_two["id"], title="agent two's task")

        self.assertEqual(
            len(get_notification_store().list_notifications()),
            2,
            "one agent's stuck streak must never silence another agent's first park",
        )


if __name__ == "__main__":
    unittest.main()
