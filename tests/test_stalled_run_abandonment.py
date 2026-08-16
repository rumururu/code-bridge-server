"""Giving up on a run nobody answered has exactly one shape.

There used to be two. The expiry sweep (`agent/approval_resume.py`) ended an
unanswered approval by *denying* it: the parked provider turn was handed a
refusal, the step recorded which tool call had been refused, and the step's
failure policy decided what happened next. The scheduler had its own version
(`_abandon_stalled_run`) that expired the approval rows and wrote `failed`
straight onto the run — and because its grace period (1h) was shorter than the
approval deadline (24h), that version won nearly every time.

Measured on the running server, with a 90s grace and a 100000s approval
deadline, at the moment the scheduler gave up on a real parked run:

    run.status                 : failed              <- flipped directly
    step.status                : waiting_for_user    <- never touched
    step.output                : ['checkpoint', 'reason']
    denied_tool_calls          : absent
    task.step.failed           : absent from the event log
    task.step.approval_resumed : absent
    last event                 : task.step.waiting_for_user

Three separate harms in one row. The run and its step *disagree* — the app's
run detail reads the step, so it showed a finished run still asking a question.
Nothing explains the failure: the panel added in wave 5 reads
`step.output.denied_tool_calls`, which this path never wrote, so this was the
one failure the app could not describe. And the event log simply stopped
mid-air, with no terminal event at all.

Underneath all of that, the parked `can_use_tool` future was never settled, so
the provider session leaked — the exact leak the expiry work existed to close.

What is pinned here:

1. An unanswered approval past the grace is settled through the deny path: the
   parked turn is handed `deny_from_permission_message`, the step records
   `denied_tool_calls`, and the approval row is recorded as expired.
2. A park with no approval in it (`ask_user`) is still abandoned — but through
   the non-approval path, which fails the step and the run *saying why*, rather
   than being forced into an approval-shaped ending it has no approval for.
3. The schedule fires again afterwards, which is what the grace exists for.
4. A run genuinely progressing still blocks, at any age.
5. Nothing is abandoned twice, and nothing is abandoned and resumed on the same
   tick: the sweep claims a run before the schedule check looks at it.
"""

import asyncio
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import (  # noqa: E402
    agent_store,
    approval_resume,
    browser_session_store,
    schedule_store,
    scheduler,
)
from agent.approval_resume import abandon_waiting_run  # noqa: E402
from agent.task_orchestrator import (  # noqa: E402
    ABANDONED_WAIT_REASON,
    STEP_DENIED_TOOL_CALLS_KEY,
    execute_task_orchestration,
    prepare_task_orchestration,
)
from approvals import approval_store  # noqa: E402
from audit import audit_store  # noqa: E402
from core import database  # noqa: E402
from policy import policy_store  # noqa: E402


def _future(minutes: int = 60) -> str:
    return (datetime.now(tz=timezone.utc) + timedelta(minutes=minutes)).isoformat()


async def _drain_resumes() -> None:
    for _ in range(20):
        pending = [
            task for task in approval_resume._BACKGROUND_TASKS if not task.done()
        ]
        if not pending:
            return
        await asyncio.gather(*pending, return_exceptions=True)


class _ParkedSession:
    """A provider session holding an unanswered permission callback."""

    provider_id = "anthropic"
    session_id = "native-1"
    has_pending_permission = True


class _FakeSessionManager:
    def __init__(self, session=None):
        self._session = session

    def get_session_if_exists(self, _project_name: str):
        return self._session


def _permission_event(*, approval_id: str) -> dict:
    return {
        "type": "permission_required",
        "approval_id": approval_id,
        "denials": [
            {
                "request_id": "provider-req-1",
                "approval_id": approval_id,
                "tool_name": "Write",
                "tool_use_id": "tool-1",
                "input": {"file_path": "/etc/hosts"},
            }
        ],
        "message": "Tool 'Write' requires approval to continue.",
    }


class _AbandonmentTestBase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "stalled_run_abandonment.db"
        agent_store._agent_store = None
        browser_session_store._browser_session_store = None
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None
        database.init_db()
        self.store = agent_store.get_agent_store()
        self.approvals = approval_store.get_approval_store()
        approval_resume._SETTLING_RUNS.clear()

    def tearDown(self):
        approval_resume._SETTLING_RUNS.clear()
        agent_store._agent_store = None
        browser_session_store._browser_session_store = None
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def _prepare(self, *, flow_json=None):
        agent = self.store.create_agent(
            name="approval bot",
            system_prompt="Run workflow steps.",
            provider_id="anthropic",
            flow_json=flow_json
            or [
                {
                    "id": "inspect",
                    "type": "llm",
                    "name": "Inspect",
                    "on_failure": {
                        "type": "ask_user",
                        "prompt": "What should I do next?",
                    },
                }
            ],
        )
        task = self.store.create_task(
            title="Run workflow",
            assigned_agent_id=agent["id"],
            goal="Finish the workflow.",
        )
        result = prepare_task_orchestration(
            task["id"], provider_id="anthropic", auto_start=False
        )
        assert result is not None
        return agent, task, result

    async def _park_on_approval(self, result, *, approval_id: str):
        async def fake_create_chat_session(**_kwargs):
            return object()

        async def fake_stream(sink, _session, **_kwargs):
            await sink.send_json(_permission_event(approval_id=approval_id))
            return False

        with patch(
            "agent.task_orchestrator.create_chat_session", fake_create_chat_session
        ), patch("agent.task_orchestrator.stream_claude_turn", fake_stream):
            await execute_task_orchestration(result["execution"])

    async def _park_on_ask_user(self, result):
        """Park with no approval anywhere: the step fails, `on_failure` asks."""

        async def fake_create_chat_session(**_kwargs):
            return object()

        async def fake_stream(_sink, _session, **_kwargs):
            raise RuntimeError("the device was not there")

        with patch(
            "agent.task_orchestrator.create_chat_session", fake_create_chat_session
        ), patch("agent.task_orchestrator.stream_claude_turn", fake_stream):
            await execute_task_orchestration(result["execution"])

    def _event_types(self, run_id: str) -> list[str]:
        return [event["event_type"] for event in self.store.list_events(run_id)]


class UnansweredApprovalIsDeniedTest(_AbandonmentTestBase):
    async def test_the_scheduler_settles_an_unanswered_approval_via_the_deny_path(self):
        _agent, task, result = self._prepare()
        run_id = result["run"]["id"]
        # A deadline far in the future — the reproduction's 100000s. The sweep
        # will not touch this row; only the scheduler's grace can end it, which
        # is exactly the case that used to bypass everything.
        approval = self.approvals.create_request(
            operation="file.write",
            run_id=run_id,
            details={"tool_name": "Write"},
            expires_at=_future(minutes=1800),
        )
        await self._park_on_approval(result, approval_id=approval["id"])
        self.assertEqual(self.store.get_run(run_id)["status"], "waiting_for_user")

        seen_kwargs: list[dict] = []

        async def fake_stream(_sink, _session, **kwargs):
            seen_kwargs.append(kwargs)
            return True

        manager = _FakeSessionManager(_ParkedSession())
        with patch(
            "agent.task_orchestrator.get_session_manager", lambda: manager
        ), patch("agent.task_orchestrator.stream_claude_turn", fake_stream), patch.object(
            scheduler, "_stall_grace_seconds", return_value=0
        ):
            await scheduler.TaskScheduler()._sweep_stalled_parks()
            blocking, reason = scheduler._blocking_run_for_task(task["id"])
            await _drain_resumes()

        # The deny is delivered to the parked session — not "the run is failed".
        self.assertEqual(len(seen_kwargs), 1, "the parked turn must be answered")
        self.assertIn("deny_from_permission_message", seen_kwargs[0])
        self.assertNotIn("retry_from_permission", seen_kwargs[0])

        # Settling is asynchronous, so the firing later in this same tick is
        # still blocked; the schedule fires on the next tick rather than over
        # the top of a run still being wound down.
        self.assertIsNotNone(blocking)
        self.assertEqual(reason, "previous run is being settled")

        # The approval is recorded as expired rather than silently dropped...
        self.assertEqual(
            self.approvals.get_request(approval["id"])["status"], "expired"
        )
        decisions = [
            event["decision"] for event in audit_store.get_audit_store().list_events(limit=50)
        ]
        self.assertIn("approval_expired", decisions)

        # ...and the run/step agree, which is the thing the measurement showed
        # broken: the step is no longer sitting at `waiting_for_user` under an
        # `approval_required` checkpoint while the run says it is over.
        step = self.store.list_task_steps(task["id"])[0]
        self.assertNotEqual(step["output"]["checkpoint"]["reason"], "approval_required")

        # And the app can explain it: what was refused is on the step.
        denied = step["output"]["denied_tool_calls"]
        self.assertEqual(denied[0]["source"], "approval_decision")
        self.assertEqual(denied[0]["approval_id"], approval["id"])
        self.assertEqual(denied[0]["tool_name"], "Write")

        event_types = self._event_types(run_id)
        self.assertEqual(event_types.count("task.step.approval_resumed"), 1)
        self.assertEqual(event_types.count("task.step.failed"), 1)
        resumed = [
            event
            for event in self.store.list_events(run_id)
            if event["event_type"] == "task.step.approval_resumed"
        ][0]
        self.assertEqual(resumed["app_event"]["decision"], "deny")

    async def test_the_schedule_fires_again_once_the_settled_run_is_over(self):
        # The whole point of the grace period. `on_failure: ask_user` re-parks
        # the denied step, so the run is still waiting — until that park in
        # turn goes unanswered, and then the schedule is free.
        _agent, task, result = self._prepare()
        run_id = result["run"]["id"]
        approval = self.approvals.create_request(
            operation="file.write",
            run_id=run_id,
            details={"tool_name": "Write"},
            expires_at=_future(minutes=1800),
        )
        await self._park_on_approval(result, approval_id=approval["id"])

        async def fake_stream(_sink, _session, **_kwargs):
            return True

        manager = _FakeSessionManager(_ParkedSession())
        with patch(
            "agent.task_orchestrator.get_session_manager", lambda: manager
        ), patch("agent.task_orchestrator.stream_claude_turn", fake_stream), patch.object(
            scheduler, "_stall_grace_seconds", return_value=0
        ):
            tick = scheduler.TaskScheduler()
            await tick._sweep_stalled_parks()
            await _drain_resumes()
            # Next tick: the run is parked on `ask_user` now, with no approval
            # in it, so the non-approval path ends it.
            await tick._sweep_stalled_parks()
            blocking, reason = scheduler._blocking_run_for_task(task["id"])

        self.assertIsNone(blocking, f"schedule still blocked: {reason}")
        run = self.store.get_run(run_id)
        self.assertEqual(run["status"], "failed")
        self.assertIsNotNone(run["ended_at"])

    async def test_a_missing_approval_row_still_goes_down_the_deny_path(self):
        # The checkpoint knows which approval the run stopped for even when the
        # row is gone. Losing the row must not downgrade the ending to a bare
        # `failed` that leaves the provider turn parked.
        _agent, task, result = self._prepare()
        run_id = result["run"]["id"]
        await self._park_on_approval(result, approval_id="apr_never_stored")

        seen_kwargs: list[dict] = []

        async def fake_stream(_sink, _session, **kwargs):
            seen_kwargs.append(kwargs)
            return True

        manager = _FakeSessionManager(_ParkedSession())
        with patch(
            "agent.task_orchestrator.get_session_manager", lambda: manager
        ), patch("agent.task_orchestrator.stream_claude_turn", fake_stream):
            record = await abandon_waiting_run(self.store.get_run(run_id))
            await _drain_resumes()

        self.assertEqual(record["path"], "deny")
        self.assertEqual(len(seen_kwargs), 1)
        self.assertIn("deny_from_permission_message", seen_kwargs[0])


class NonApprovalParkIsAbandonedSanelyTest(_AbandonmentTestBase):
    async def test_an_ask_user_park_is_abandoned_without_the_approval_path(self):
        _agent, task, result = self._prepare()
        run_id = result["run"]["id"]
        await self._park_on_ask_user(result)

        step = self.store.list_task_steps(task["id"])[0]
        self.assertEqual(step["output"]["checkpoint"]["reason"], "ask_user")

        async def fake_stream(_sink, _session, **_kwargs):
            raise AssertionError("a park with no approval must not resume a turn")

        manager = _FakeSessionManager(_ParkedSession())
        with patch(
            "agent.task_orchestrator.get_session_manager", lambda: manager
        ), patch("agent.task_orchestrator.stream_claude_turn", fake_stream):
            record = await abandon_waiting_run(
                self.store.get_run(run_id), waited_seconds=7200
            )
            await _drain_resumes()

        self.assertEqual(record["path"], "abandon")

        # The run ends, and every reader agrees about why.
        run = self.store.get_run(run_id)
        self.assertEqual(run["status"], "failed")
        self.assertIsNotNone(run["ended_at"])

        step = self.store.list_task_steps(task["id"])[0]
        self.assertEqual(step["status"], "failed")
        self.assertEqual(step["output"]["reason"], ABANDONED_WAIT_REASON)
        self.assertEqual(step["output"]["waiting_for"], "ask_user")
        self.assertEqual(step["output"]["waited_seconds"], 7200)
        self.assertTrue(step["output"]["message"])
        # No approval was invented for a park that had none.
        self.assertNotIn("denied_tool_calls", step["output"])

        task_row = self.store.get_task(task["id"])
        self.assertEqual(task_row["error"]["reason"], ABANDONED_WAIT_REASON)
        self.assertIsNone((task_row["metadata"] or {}).get("active_checkpoint"))

        # The timeline ends somewhere instead of stopping mid-air.
        event_types = self._event_types(run_id)
        self.assertEqual(event_types.count("task.run.abandoned"), 1)
        self.assertEqual(event_types[-1], "task.execution.failed")

    async def test_an_earlier_refusal_survives_being_abandoned_later(self):
        """The refusal is the useful half of the story; abandonment must not erase it.

        A step can be refused a tool, fail, park under `on_failure: ask_user`,
        and then have *that* park go unanswered too. Measured live on
        2026-08-16, the second ending overwrote the first: the run's final
        record said nobody answered and no longer said that `Read` had been
        denied on a named path — so the panel, which reads
        `denied_tool_calls`, could name the timeout but not the refusal that
        caused it.
        """
        _agent, task, result = self._prepare()
        run_id = result["run"]["id"]
        await self._park_on_ask_user(result)

        step = self.store.list_task_steps(task["id"])[0]
        earlier_denial = [
            {
                "source": "approval_decision",
                "tool_name": "Read",
                "tool_target": "/Users/someone/.ssh/id_rsa",
                "decision": "expired",
            }
        ]
        self.store.update_task_step(
            step["id"],
            {"output": {**step["output"], STEP_DENIED_TOOL_CALLS_KEY: earlier_denial}},
        )

        manager = _FakeSessionManager(_ParkedSession())
        with patch("agent.task_orchestrator.get_session_manager", lambda: manager):
            await abandon_waiting_run(self.store.get_run(run_id), waited_seconds=7200)
            await _drain_resumes()

        step = self.store.list_task_steps(task["id"])[0]
        self.assertEqual(step["status"], "failed")
        # Both facts, not one: why it ended, and what had been refused.
        self.assertEqual(step["output"]["reason"], ABANDONED_WAIT_REASON)
        self.assertEqual(step["output"][STEP_DENIED_TOOL_CALLS_KEY], earlier_denial)

        task_row = self.store.get_task(task["id"])
        self.assertEqual(
            task_row["error"][STEP_DENIED_TOOL_CALLS_KEY], earlier_denial
        )

    async def test_a_run_that_moved_on_is_left_alone(self):
        _agent, _task, result = self._prepare()
        run_id = result["run"]["id"]
        await self._park_on_ask_user(result)
        # Somebody answered between the decision to abandon and the attempt.
        self.store.update_run_status(run_id, "running")

        record = await abandon_waiting_run({"id": run_id, "status": "waiting_for_user"})

        self.assertFalse(record["abandoned"])
        self.assertEqual(record["reason"], "no_longer_waiting")
        self.assertEqual(self.store.get_run(run_id)["status"], "running")


class NothingIsSettledTwiceTest(_AbandonmentTestBase):
    async def test_a_second_abandon_of_the_same_run_is_refused(self):
        _agent, _task, result = self._prepare()
        run_id = result["run"]["id"]
        await self._park_on_ask_user(result)

        first = await abandon_waiting_run(self.store.get_run(run_id))
        self.assertTrue(first["abandoned"])

        second = await abandon_waiting_run({"id": run_id, "status": "waiting_for_user"})
        self.assertFalse(second["abandoned"])
        # It is refused because the run is no longer waiting, and the record
        # was written exactly once.
        self.assertEqual(self._event_types(run_id).count("task.run.abandoned"), 1)

    async def test_a_claimed_run_survives_a_whole_tick_untouched(self):
        # The approval expiry sweep, or a decision the user is making right
        # now, holds the claim. Neither half of the rest of the tick may touch
        # the run: not the park sweep, not the schedule check.
        _agent, task, result = self._prepare()
        run_id = result["run"]["id"]
        await self._park_on_ask_user(result)

        self.assertTrue(approval_resume._claim_settlement(run_id))
        try:
            with patch.object(scheduler, "_stall_grace_seconds", return_value=0):
                await scheduler.TaskScheduler()._sweep_stalled_parks()
                blocking, reason = scheduler._blocking_run_for_task(task["id"])
        finally:
            approval_resume._release_settlement(run_id)

        self.assertIsNotNone(blocking)
        self.assertEqual(reason, "previous run is being settled")
        self.assertEqual(self.store.get_run(run_id)["status"], "waiting_for_user")
        self.assertEqual(self._event_types(run_id).count("task.run.abandoned"), 0)

    async def test_one_tick_abandons_a_run_exactly_once(self):
        # The sweep ends it; the schedule check that follows in the same tick
        # must find nothing left to do rather than write a second ending.
        _agent, task, result = self._prepare()
        run_id = result["run"]["id"]
        await self._park_on_ask_user(result)

        with patch.object(scheduler, "_stall_grace_seconds", return_value=0):
            await scheduler.TaskScheduler()._sweep_stalled_parks()
            blocking, reason = scheduler._blocking_run_for_task(task["id"])

        self.assertIsNone(blocking, f"schedule still blocked: {reason}")
        self.assertEqual(self._event_types(run_id).count("task.run.abandoned"), 1)
        self.assertEqual(
            self._event_types(run_id).count("task.execution.failed"), 1
        )

    async def test_a_decision_cannot_drive_a_run_that_is_already_settling(self):
        _agent, _task, result = self._prepare()
        run_id = result["run"]["id"]
        await self._park_on_approval(result, approval_id="apr_1")

        self.assertTrue(approval_resume._claim_settlement(run_id))
        try:
            resumed = await approval_resume.maybe_resume_run_for_decision(
                {"id": "apr_1", "run_id": run_id}, decision="approve_once"
            )
        finally:
            approval_resume._release_settlement(run_id)

        self.assertIsNone(resumed)


class ProgressingRunStillBlocksTest(_AbandonmentTestBase):
    async def test_a_working_run_is_never_abandoned_however_old(self):
        _agent, task, result = self._prepare()
        run_id = result["run"]["id"]
        self.store.update_run_status(run_id, "running")

        with patch.object(scheduler, "_stall_grace_seconds", return_value=0):
            await scheduler.TaskScheduler()._sweep_stalled_parks()
            blocking, reason = scheduler._blocking_run_for_task(task["id"])

        self.assertEqual(blocking["id"], run_id)
        self.assertEqual(reason, "previous run still active")
        self.assertEqual(self.store.get_run(run_id)["status"], "running")


class AParkWithNoScheduleIsStillNoticedTest(_AbandonmentTestBase):
    """The gap this class exists for, in one sentence.

    Giving up on a stalled run used to happen only inside
    `_blocking_run_for_task`, whose one caller is `_fire_schedule`
    (`agent/scheduler.py`). So a run on a task with **no schedule** — a
    `run once` agent, a disabled schedule, anything started by hand — was never
    looked at by it, and the approval expiry sweep does not cover it either
    because an `ask_user` park has no approval row to expire.

    Measured live on 2026-08-16 with a 60s grace: five `ask_user`-parked runs on
    scheduleless tasks were still `waiting_for_user` three minutes later, and
    had to be ended by hand.
    """

    async def test_an_ask_user_park_on_a_scheduleless_task_is_abandoned(self):
        _agent, task, result = self._prepare()
        run_id = result["run"]["id"]
        await self._park_on_ask_user(result)
        self.assertEqual(self.store.get_run(run_id)["status"], "waiting_for_user")
        # The premise, asserted rather than assumed: no schedule exists for
        # this task, so `_fire_schedule` — the only caller of
        # `_blocking_run_for_task` — will never look at this run.
        self.assertEqual(
            schedule_store.get_schedule_store().list_for_task(task["id"]), []
        )

        with patch.object(scheduler, "_stall_grace_seconds", return_value=0):
            await scheduler.TaskScheduler()._sweep_stalled_parks()

        run = self.store.get_run(run_id)
        self.assertEqual(run["status"], "failed")
        self.assertIsNotNone(run["ended_at"])
        step = self.store.list_task_steps(task["id"])[0]
        self.assertEqual(step["status"], "failed")
        self.assertEqual(step["output"]["reason"], ABANDONED_WAIT_REASON)
        self.assertEqual(self._event_types(run_id).count("task.run.abandoned"), 1)

    async def test_a_park_still_inside_the_grace_is_left_alone(self):
        _agent, task, result = self._prepare()
        run_id = result["run"]["id"]
        await self._park_on_ask_user(result)

        with patch.object(scheduler, "_stall_grace_seconds", return_value=3600):
            await scheduler.TaskScheduler()._sweep_stalled_parks()

        self.assertEqual(self.store.get_run(run_id)["status"], "waiting_for_user")
        step = self.store.list_task_steps(task["id"])[0]
        self.assertEqual(step["output"]["checkpoint"]["reason"], "ask_user")
        self.assertEqual(self._event_types(run_id).count("task.run.abandoned"), 0)

    async def test_a_whole_tick_ends_it_with_no_schedule_due_at_all(self):
        # End to end through `trigger_once`, with nothing due to fire: the
        # gap was that a tick with no due schedule did nothing about parks.
        _agent, task, result = self._prepare()
        run_id = result["run"]["id"]
        await self._park_on_ask_user(result)

        with patch.object(scheduler, "_stall_grace_seconds", return_value=0), patch(
            "agent.scheduler.TaskScheduler._sweep_cli_agents_if_due",
            lambda _self: asyncio.sleep(0),
        ):
            fired = await scheduler.TaskScheduler().trigger_once()

        self.assertEqual(fired, 0, "no schedule exists to fire")
        self.assertEqual(self.store.get_run(run_id)["status"], "failed")
        self.assertEqual(self._event_types(run_id).count("task.run.abandoned"), 1)


class TickOrderTest(_AbandonmentTestBase):
    async def test_the_sweeps_run_before_any_schedule_fires(self):
        """Ordering, stated as an ordering.

        Three things on one tick can decide to end the same run — the approval
        expiry sweep, the park sweep, the schedule check — and by default the
        first two share one deadline. Approvals go first so a run past its
        approval deadline ends *as an expired approval* and takes the claim;
        the park sweep goes next so anything it settles is already settled by
        the time the schedule check reports on it.
        """
        order: list[str] = []

        async def fake_approvals():
            order.append("approvals")
            return []

        async def fake_parks(_self):
            order.append("parks")

        async def fake_fire(_schedule):
            order.append("fire")

        due = [{"id": "sch_1", "task_id": "task_1"}]
        with patch(
            "agent.approval_resume.sweep_expired_approvals", fake_approvals
        ), patch(
            "agent.scheduler.TaskScheduler._sweep_stalled_parks", fake_parks
        ), patch("agent.scheduler._fire_schedule", fake_fire), patch(
            "agent.scheduler.TaskScheduler._sweep_cli_agents_if_due",
            lambda _self: asyncio.sleep(0),
        ), patch(
            "agent.schedule_store.get_schedule_store"
        ) as store_mock:
            store_mock.return_value.list_due.return_value = due
            with patch("agent.scheduler.get_schedule_store", store_mock):
                await scheduler.TaskScheduler().trigger_once()

        self.assertEqual(order, ["approvals", "parks", "fire"])


if __name__ == "__main__":
    unittest.main()
