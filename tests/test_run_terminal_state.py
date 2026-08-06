"""A run has to end, and someone has to be told when it ends badly.

This file exists because of a real morning. A 6-hourly agent ("스몰뎁 사이클")
fired at 00:11, its `shell` step drove an Android phone that had gone offline,
and 8h45m later the run was still `running`. It was not hung: it was re-running
the same 52-minute script over and over. `execute_task_orchestration` walked
`list_task_steps(task_id)` — every step row the *task* had ever had, 170 of
them across 85 earlier fires — instead of the steps belonging to this run. It
skipped every `completed` row, landed on a shell step left `running` by a run
three days earlier, ran it, and when it failed its `on_failure: goto_step:
diagnose` resolved to the *first* `diagnose` row on the task, completed back in
May. So the loop walked forward from there, met the same stale shell step
again, and went round. The run never reached a terminal state, and because
`skip_if_active` gives a progressing run no grace period at all, every later
firing of that schedule was skipped. Nobody was told any of this.

What breaks for the user if this regresses:

- A scheduled agent silently stops running, forever, after one bad night —
  the schedule's `skip_count` rises and nothing else happens.
- The same expensive script runs on a loop nobody asked for (here: ten
  52-minute device cycles against a phone that was not there).
- A failure at 3am produces no notification at all, so the first sign of
  trouble is noticing days-old red in the app.

The fix is *not* a timeout on `running` runs: a genuinely long job must not be
killed, which is what `test_scheduler_stalled_run.py` asserts on purpose. The
fix is that a run only ever executes and resolves its own steps, and that
every exit from the loop leaves it terminal.
"""

import asyncio
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import (  # noqa: E402
    agent_store,
    browser_session_store,
    push_notifier,
    scheduler,
    script_store as script_store_module,
    task_orchestrator,
)
from agent.notification_store import get_notification_store  # noqa: E402
from agent.task_orchestrator import (  # noqa: E402
    execute_task_orchestration,
    prepare_task_orchestration,
)
from core import database  # noqa: E402
from pairing import pairing_service as pairing_service_module  # noqa: E402
from pairing.pairing_service import PairingService  # noqa: E402

_OFFLINE_DEVICE_SCRIPT = """#!/bin/bash
echo "[smalldev-cycle] R59N3035LQL open com.mkideabox.devfeedbackhub"
echo "adb: device offline" >&2
exit 1
"""


class _RunTerminalStateTestBase(unittest.TestCase):
    """Fresh DB, fresh pairing singleton, push forced off.

    Same isolation as test_wait_for_user_notifies.py: no test may touch a real
    operator's runs or a real FCM key.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self._original_db_path = database.DB_PATH
        database.DB_PATH = self.dir / "run_terminal_state.db"
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

    def _failing_script(self, name: str = "cycle.sh") -> dict:
        path = self.dir / name
        path.write_text(_OFFLINE_DEVICE_SCRIPT)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        return self.scripts.register(name=name, path=str(path))

    def _agent_and_task(self, flow_json: list[dict], *, agent_name: str = "스몰뎁 사이클 · M205N"):
        agent = self.store.create_agent(
            name=agent_name,
            system_prompt="Drive the nightly device cycle.",
            provider_id="openai",
            flow_json=flow_json,
        )
        task = self.store.create_task(
            title="Nightly device cycle",
            assigned_agent_id=agent["id"],
            goal="Run the exchange cycle on the phone.",
        )
        return agent, task

    def _fire(self, task_id: str) -> dict:
        """One firing of the schedule: prepare a run, then execute it."""
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

    def _shell_starts(self, run_id: str) -> int:
        return len(
            [
                event
                for event in self.store.list_events(run_id)
                if event["event_type"] == "task.step.shell.started"
            ]
        )

    def _steps_of(self, task_id: str, run_id: str) -> list[dict]:
        return [
            step
            for step in self.store.list_task_steps(task_id)
            if step.get("run_id") == run_id
        ]


class ShellFailureEndsTheRunTest(_RunTerminalStateTestBase):
    def test_a_failing_shell_step_with_nowhere_to_go_ends_the_run_failed(self):
        script = self._failing_script()
        _agent, task = self._agent_and_task(
            [
                {
                    "id": "cycle",
                    "type": "shell",
                    "name": "M205N 사이클",
                    "script_id": script["id"],
                    "on_failure": "abort",
                }
            ]
        )

        run = self.store.get_run(self._fire(task["id"])["id"])

        assert run is not None
        self.assertEqual(run["status"], "failed")
        self.assertIsNotNone(run["ended_at"], "a finished run must record when it ended")
        self.assertEqual(self.store.get_task(task["id"])["status"], "failed")
        self.assertEqual(self._shell_starts(run["id"]), 1, "the script must not be re-run")

    def test_the_default_policy_parks_the_run_rather_than_wedging_it(self):
        # A step written with no `on_failure` at all normalizes to `ask_user`
        # (agent/workflow_v2.py::normalize_failure_policy), which is a resting
        # place, not a terminal one — deliberately, because the common case is
        # a human who can unstick it. What must never happen is the third
        # option: still `running`, driven by nobody. So this asserts the two
        # things that make parking survivable — someone is told, and the
        # schedule can still recover on its own.
        script = self._failing_script()
        _agent, task = self._agent_and_task(
            [
                {
                    "id": "cycle",
                    "type": "shell",
                    "name": "M205N 사이클",
                    "script_id": script["id"],
                }
            ]
        )

        run = self.store.get_run(self._fire(task["id"])["id"])

        assert run is not None
        self.assertEqual(run["status"], "waiting_for_user")
        self.assertEqual(len(get_notification_store().list_notifications()), 1)
        self.assertEqual(self._shell_starts(run["id"]), 1)

        with mock.patch.object(scheduler, "_stall_grace_seconds", return_value=0):
            blocking, reason = scheduler._blocking_run_for_task(task["id"])
        self.assertIsNone(blocking, f"schedule still blocked: {reason}")

    def test_an_on_failure_goto_still_routes_to_the_diagnosis_step(self):
        # The escalation the shell step type exists for: script fails, an LLM
        # step gets to look at why. Ending the run on the first failed step
        # would be a cheap way to guarantee terminality and would break this.
        script = self._failing_script()
        _agent, task = self._agent_and_task(
            [
                {
                    "id": "cycle",
                    "type": "shell",
                    "name": "M205N 사이클",
                    "script_id": script["id"],
                    "on_failure": "goto_step:diagnose",
                    "on_success": {"type": "end"},
                },
                {"id": "diagnose", "type": "llm", "name": "실패 진단"},
            ]
        )

        run_id = self._fire(task["id"])["id"]
        run = self.store.get_run(run_id)
        steps = {
            step["input"]["workflow_step_id"]: step
            for step in self._steps_of(task["id"], run_id)
        }

        self.assertEqual(steps["cycle"]["status"], "failed")
        self.assertEqual(
            steps["diagnose"]["status"],
            "completed",
            "the on_failure goto target must actually run",
        )
        goto_events = [
            event
            for event in self.store.list_events(run_id)
            if event["event_type"] == "task.step.goto"
        ]
        self.assertEqual(len(goto_events), 1)
        self.assertEqual(goto_events[0]["app_event"]["target_step_id"], "diagnose")

        # ...and the run still ends, failed, because a step failed on the way.
        assert run is not None
        self.assertEqual(run["status"], "failed")
        self.assertIsNotNone(run["ended_at"])
        self.assertEqual(self._shell_starts(run_id), 1)


class StaleStepsFromEarlierRunsTest(_RunTerminalStateTestBase):
    def test_a_new_run_ignores_steps_left_behind_by_an_earlier_run(self):
        """The production defect, reproduced in miniature.

        An earlier run leaves a shell step at `running` (a crash, a restart, a
        kill). Every later firing used to pick that stale row up, re-run its
        script, follow its `on_failure` goto to a *completed* step from the
        first run, walk forward into the stale row again, and loop.
        """
        script = self._failing_script()
        _agent, task = self._agent_and_task(
            [
                {
                    "id": "cycle",
                    "type": "shell",
                    "name": "M205N 사이클",
                    "script_id": script["id"],
                    "on_failure": "goto_step:diagnose",
                    "on_success": {"type": "end"},
                },
                {"id": "diagnose", "type": "llm", "name": "실패 진단"},
            ]
        )

        first_run_id = self._fire(task["id"])["id"]
        first_steps = self._steps_of(task["id"], first_run_id)
        stale_step = next(
            step for step in first_steps if step["input"]["workflow_step_id"] == "cycle"
        )
        # Freeze it mid-flight, the way an interrupted process would.
        self.store.update_task_step(stale_step["id"], {"status": "running"})

        second_run_id = self._fire(task["id"])["id"]
        second_run = self.store.get_run(second_run_id)
        second_steps = {
            step["input"]["workflow_step_id"]: step
            for step in self._steps_of(task["id"], second_run_id)
        }

        assert second_run is not None
        self.assertEqual(second_run["status"], "failed")
        self.assertIsNotNone(second_run["ended_at"])

        # The second run ran its own steps...
        self.assertEqual(second_steps["cycle"]["status"], "failed")
        self.assertEqual(second_steps["diagnose"]["status"], "completed")
        # ...exactly once each, rather than looping on the stale row.
        self.assertEqual(self._shell_starts(second_run_id), 1)
        # ...and left the first run's history alone.
        self.assertEqual(
            self.store.get_task_step(stale_step["id"])["status"],
            "running",
            "a later run must not reach back into an earlier run's steps",
        )

    def test_the_scheduler_is_free_to_fire_again_once_the_run_is_terminal(self):
        script = self._failing_script()
        _agent, task = self._agent_and_task(
            [
                {
                    "id": "cycle",
                    "type": "shell",
                    "name": "M205N 사이클",
                    "script_id": script["id"],
                    "on_failure": "abort",
                }
            ]
        )
        run_id = self._fire(task["id"])["id"]

        blocking, reason = scheduler._blocking_run_for_task(task["id"])
        self.assertIsNone(blocking, f"schedule still blocked: {reason}")

        # And the contrast that says why the fix is not a timeout: while a run
        # really is progressing, it still blocks, at any age. See
        # test_scheduler_stalled_run.py::test_running_run_blocks_regardless_of_age.
        self.store.update_run_status(run_id, "running")
        blocking, reason = scheduler._blocking_run_for_task(task["id"])
        assert blocking is not None
        self.assertEqual(blocking["id"], run_id)
        self.assertEqual(reason, "previous run still active")


class RunAlwaysReachesATerminalStateTest(_RunTerminalStateTestBase):
    def test_an_unexpected_error_inside_the_loop_ends_the_run_failed(self):
        script = self._failing_script()
        _agent, task = self._agent_and_task(
            [
                {
                    "id": "cycle",
                    "type": "shell",
                    "name": "M205N 사이클",
                    "script_id": script["id"],
                }
            ]
        )
        prepared = prepare_task_orchestration(task["id"], provider_id="openai", auto_start=False)
        assert prepared is not None

        async def explode(**_kwargs):
            raise RuntimeError("adb server died")

        with mock.patch.object(task_orchestrator, "_execute_shell_workflow_step", explode):
            asyncio.run(execute_task_orchestration(prepared["execution"]))

        run = self.store.get_run(prepared["run"]["id"])
        assert run is not None
        self.assertEqual(run["status"], "failed")
        self.assertIsNotNone(run["ended_at"])
        self.assertIn("adb server died", self.store.get_task(task["id"])["error"]["message"])

    def test_a_loop_branch_that_returns_without_finishing_still_closes_the_run(self):
        """The safety net, exercised directly.

        If some future branch of the step loop returns without finishing or
        parking the run, the run must not be left looking alive — that is what
        wedges the schedule.
        """
        script = self._failing_script()
        _agent, task = self._agent_and_task(
            [
                {
                    "id": "cycle",
                    "type": "shell",
                    "name": "M205N 사이클",
                    "script_id": script["id"],
                }
            ]
        )
        prepared = prepare_task_orchestration(task["id"], provider_id="openai", auto_start=False)
        assert prepared is not None

        async def do_nothing(**_kwargs):
            return None

        with mock.patch.object(task_orchestrator, "_drive_workflow_steps", do_nothing):
            asyncio.run(execute_task_orchestration(prepared["execution"]))

        run = self.store.get_run(prepared["run"]["id"])
        assert run is not None
        self.assertEqual(run["status"], "failed")
        self.assertIsNotNone(run["ended_at"])

    def test_a_parked_run_is_left_parked(self):
        # waiting_for_user is a legitimate resting place: a human still has to
        # answer it, and the parked-run notification has already fired. The
        # close-out must not convert it into a failure.
        _agent, task = self._agent_and_task(
            [{"id": "login", "type": "manual_handoff", "name": "Log into the portal"}]
        )

        run = self.store.get_run(self._fire(task["id"])["id"])

        assert run is not None
        self.assertEqual(run["status"], "waiting_for_user")


class FailedRunNotifiesTest(_RunTerminalStateTestBase):
    def test_a_failed_run_stores_a_notification_naming_the_agent_and_the_reason(self):
        script = self._failing_script()
        _agent, task = self._agent_and_task(
            [
                {
                    "id": "cycle",
                    "type": "shell",
                    "name": "M205N 사이클",
                    "script_id": script["id"],
                    "on_failure": "abort",
                }
            ]
        )

        run_id = self._fire(task["id"])["id"]

        notifications = get_notification_store().list_notifications()
        self.assertEqual(len(notifications), 1)
        notification = notifications[0]

        self.assertEqual(notification["level"], "error")
        self.assertIn("스몰뎁 사이클 · M205N", notification["title"])
        self.assertIn("failed", notification["title"])
        self.assertEqual(notification["run_id"], run_id)
        self.assertEqual(notification["task_id"], task["id"])
        # Why, not just that: the step that failed and the evidence it kept.
        self.assertIn("Nightly device cycle", notification["body"])
        self.assertIn("M205N 사이클", notification["body"])
        self.assertIn("device offline", notification["body"])

    def test_a_notification_failure_leaves_the_runs_terminal_state_intact(self):
        # The run is already over by the time we try to tell anyone. Losing the
        # message is bad; letting it rewrite what the run says it did is worse.
        script = self._failing_script()
        _agent, task = self._agent_and_task(
            [
                {
                    "id": "cycle",
                    "type": "shell",
                    "name": "M205N 사이클",
                    "script_id": script["id"],
                    "on_failure": "abort",
                }
            ]
        )

        def boom():
            raise RuntimeError("notification store is down")

        with mock.patch("agent.notification_store.get_notification_store", boom):
            run_id = self._fire(task["id"])["id"]

        run = self.store.get_run(run_id)
        assert run is not None
        self.assertEqual(run["status"], "failed")
        self.assertIsNotNone(run["ended_at"])
        self.assertEqual(self.store.get_task(task["id"])["status"], "failed")
        self.assertEqual(get_notification_store().list_notifications(), [])

    def test_a_completed_run_says_nothing(self):
        # Notifying on success would train the user to ignore the channel.
        _agent, task = self._agent_and_task(
            [{"id": "think", "type": "llm", "name": "Think about it"}]
        )

        run = self.store.get_run(self._fire(task["id"])["id"])

        assert run is not None
        self.assertEqual(run["status"], "completed")
        self.assertEqual(get_notification_store().list_notifications(), [])


class StepScopingFallbackTest(unittest.TestCase):
    """The legacy fallback must not become a way back into the original bug.

    `_steps_for_run` falls back to the task's whole step list so rows written
    before steps carried a run_id still execute. If that fallback triggers
    whenever *this run* has no rows — rather than when *no row on the task* is
    stamped — then any run that reaches the loop without its own steps
    inherits every earlier firing's history and starts replaying it, which is
    exactly how a nightly agent came to run the same 52-minute device script
    ten times in one fire.
    """

    @staticmethod
    def _store(rows):
        store = mock.Mock()
        store.list_task_steps.return_value = rows
        return store

    def test_a_run_sees_only_its_own_steps(self):
        rows = [
            {"id": "s1", "run_id": "run_old"},
            {"id": "s2", "run_id": "run_new"},
        ]
        scoped = task_orchestrator._steps_for_run(self._store(rows), "task_1", "run_new")
        self.assertEqual([s["id"] for s in scoped], ["s2"])

    def test_unstamped_rows_still_run(self):
        """A task whose rows predate run stamping has to keep working."""
        rows = [{"id": "s1"}, {"id": "s2"}]
        scoped = task_orchestrator._steps_for_run(self._store(rows), "task_1", "run_new")
        self.assertEqual([s["id"] for s in scoped], ["s1", "s2"])

    def test_a_run_with_no_rows_on_a_stamped_task_gets_nothing(self):
        """Running nothing beats running another run's work.

        The task clearly stamps its steps, so a run owning none of them is a
        bug somewhere else — inheriting 170 rows of history would turn that
        bug into a replay loop.
        """
        rows = [
            {"id": "s1", "run_id": "run_old"},
            {"id": "s2", "run_id": "run_older"},
        ]
        scoped = task_orchestrator._steps_for_run(self._store(rows), "task_1", "run_new")
        self.assertEqual(scoped, [])


if __name__ == "__main__":
    unittest.main()
