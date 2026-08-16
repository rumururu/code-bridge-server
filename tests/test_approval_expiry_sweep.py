"""Approvals expire, and an expired approval ends the run it parked.

B1/B2/B3 made a run stop for an approval and start again when someone answers
it. Nobody has to answer it, though — and until this, nobody answering meant
the run sat in `waiting_for_user` for good, holding its task, its schedule slot
and a provider session still parked on an unanswered `can_use_tool` callback.

What is pinned here:

1. An approval is created with a deadline (24h by default, configurable), and
   switching expiry off is an explicit choice rather than the default.
2. The sweep expires only rows past their deadline, records each one as
   expired, and resumes the run each was blocking down the *deny* path.
3. A resume that meets an approval which is past its deadline but not yet swept
   denies too — and writes the expiry down rather than silently denying a row
   the pending queue still shows as answerable.
4. An approval that is not due yet is left strictly alone.
5. The parking notification names the tool and its target.

Deadlines are written directly (past/future timestamps); nothing sleeps.
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

from agent import agent_store, approval_resume, browser_session_store, scheduler  # noqa: E402
from agent.approval_resume import sweep_expired_approvals  # noqa: E402
from agent.task_orchestrator import (  # noqa: E402
    _approval_wait_prompt,
    execute_task_orchestration,
    prepare_task_orchestration,
    resume_task_orchestration,
)
from approvals import approval_store  # noqa: E402
from approvals.approval_service import (  # noqa: E402
    approval_expiry_seconds,
    default_approval_expires_at,
    expire_stale_approvals,
)
from audit import audit_store  # noqa: E402
from core import database  # noqa: E402
from policy import policy_store  # noqa: E402


async def _drain_resumes() -> None:
    """Wait for the resumes the sweep spawned in the background.

    Deterministic on purpose: `asyncio.sleep(0)` a couple of times happens to
    work today and would start failing the moment the resume path gains an
    await.
    """
    for _ in range(20):
        pending = [
            task for task in approval_resume._BACKGROUND_TASKS if not task.done()
        ]
        if not pending:
            return
        await asyncio.gather(*pending, return_exceptions=True)


def _past(minutes: int = 5) -> str:
    return (datetime.now(tz=timezone.utc) - timedelta(minutes=minutes)).isoformat()


def _future(minutes: int = 60) -> str:
    return (datetime.now(tz=timezone.utc) + timedelta(minutes=minutes)).isoformat()


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


def _permission_event(*, approval_id: str, tool_name: str = "Read", target: str = "/etc/hosts") -> dict:
    return {
        "type": "permission_required",
        "approval_id": approval_id,
        "denials": [
            {
                "request_id": "provider-req-1",
                "approval_id": approval_id,
                "tool_name": tool_name,
                "tool_use_id": "tool-1",
                "input": {"file_path": target},
            }
        ],
        "message": f"Tool '{tool_name}' requires approval to continue.",
    }


class _DbBackedTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "approval_expiry.db"
        agent_store._agent_store = None
        browser_session_store._browser_session_store = None
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None
        database.init_db()
        self.store = agent_store.get_agent_store()
        self.approvals = approval_store.get_approval_store()

    def tearDown(self):
        agent_store._agent_store = None
        browser_session_store._browser_session_store = None
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def _prepare(self):
        agent = self.store.create_agent(
            name="approval bot",
            system_prompt="Run workflow steps.",
            provider_id="anthropic",
            flow_json=[
                {
                    "id": "inspect",
                    "type": "llm",
                    "name": "Inspect",
                    "on_failure": {"type": "ask_user", "prompt": "What should I do next?"},
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


class ApprovalExpiryConfigTest(unittest.TestCase):
    """The deadline itself: a default, an override, and an explicit opt-out."""

    def test_default_is_twenty_four_hours(self):
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("CODEBRIDGE_APPROVAL_EXPIRY_SECONDS", None)
            self.assertEqual(approval_expiry_seconds(), 24 * 60 * 60)
            now = datetime(2026, 8, 16, 9, 0, tzinfo=timezone.utc)
            self.assertEqual(
                default_approval_expires_at(now=now),
                (now + timedelta(hours=24)).isoformat(),
            )

    def test_override_is_honoured(self):
        with patch.dict(
            "os.environ", {"CODEBRIDGE_APPROVAL_EXPIRY_SECONDS": "60"}, clear=False
        ):
            self.assertEqual(approval_expiry_seconds(), 60)
            now = datetime(2026, 8, 16, 9, 0, tzinfo=timezone.utc)
            self.assertEqual(
                default_approval_expires_at(now=now),
                (now + timedelta(seconds=60)).isoformat(),
            )

    def test_zero_means_no_deadline(self):
        with patch.dict(
            "os.environ", {"CODEBRIDGE_APPROVAL_EXPIRY_SECONDS": "0"}, clear=False
        ):
            self.assertIsNone(default_approval_expires_at())

    def test_garbage_falls_back_to_the_default(self):
        with patch.dict(
            "os.environ", {"CODEBRIDGE_APPROVAL_EXPIRY_SECONDS": "soon"}, clear=False
        ):
            self.assertEqual(approval_expiry_seconds(), 24 * 60 * 60)


class ApprovalNotificationPromptTest(unittest.TestCase):
    """B7: the parking prompt has to say what is being asked."""

    def test_prompt_names_the_tool_and_the_target(self):
        self.assertEqual(
            _approval_wait_prompt("Read", "/etc/hosts"),
            "Read needs your approval: /etc/hosts",
        )

    def test_prompt_names_the_tool_when_the_target_is_unknown(self):
        self.assertEqual(
            _approval_wait_prompt("MysteryTool", None),
            "MysteryTool needs your approval before this step can continue.",
        )

    def test_prompt_falls_back_when_nothing_is_known(self):
        self.assertEqual(
            _approval_wait_prompt(None, None),
            "Approval is required before this workflow step can continue.",
        )


class ApprovalStoreExpiryTest(_DbBackedTest):
    def test_list_expired_pending_is_the_complement_of_list_pending(self):
        due = self.approvals.create_request(
            operation="provider.tool", run_id="run_1", expires_at=_past()
        )
        fresh = self.approvals.create_request(
            operation="provider.tool", run_id="run_1", expires_at=_future()
        )
        forever = self.approvals.create_request(operation="provider.tool", run_id="run_1")

        pending_ids = {row["id"] for row in self.approvals.list_pending()}
        expired_ids = {row["id"] for row in self.approvals.list_expired_pending()}

        self.assertEqual(expired_ids, {due["id"]})
        self.assertEqual(pending_ids, {fresh["id"], forever["id"]})

    def test_expire_stale_approvals_marks_and_audits_only_the_stale_one(self):
        due = self.approvals.create_request(
            operation="provider.tool", run_id="run_1", expires_at=_past()
        )
        fresh = self.approvals.create_request(
            operation="provider.tool", run_id="run_1", expires_at=_future()
        )

        expired = expire_stale_approvals()

        self.assertEqual([row["id"] for row in expired], [due["id"]])
        self.assertEqual(self.approvals.get_request(due["id"])["status"], "expired")
        self.assertEqual(self.approvals.get_request(fresh["id"])["status"], "pending")

        decisions = [
            event["decision"]
            for event in audit_store.get_audit_store().list_events(limit=50)
        ]
        self.assertIn("approval_expired", decisions)


class ApprovalExpirySweepTest(_DbBackedTest):
    """The sweep: a parked run must leave `waiting_for_user` via deny."""

    async def test_expired_approval_sweeps_the_parked_run_down_the_deny_path(self):
        _agent, task, result = self._prepare()
        approval = self.approvals.create_request(
            operation="provider.tool",
            run_id=result["run"]["id"],
            details={"tool_name": "Read"},
            expires_at=_past(),
        )
        await self._park_on_approval(result, approval_id=approval["id"])

        run = self.store.get_run(result["run"]["id"])
        assert run is not None
        self.assertEqual(run["status"], "waiting_for_user")

        seen_kwargs: list[dict] = []

        async def fake_stream(_sink, _session, **kwargs):
            seen_kwargs.append(kwargs)
            return True

        manager = _FakeSessionManager(_ParkedSession())
        with patch(
            "agent.task_orchestrator.get_session_manager", lambda: manager
        ), patch("agent.task_orchestrator.stream_claude_turn", fake_stream):
            swept = await sweep_expired_approvals()
            await _drain_resumes()

        self.assertEqual(len(swept), 1)
        self.assertEqual(swept[0]["approval_id"], approval["id"])
        self.assertTrue(swept[0]["resumed"])
        self.assertEqual(swept[0]["decision"], "expired")

        # Recorded as expired, not silently dropped.
        self.assertEqual(
            self.approvals.get_request(approval["id"])["status"], "expired"
        )

        # The parked turn was answered with a denial, not retried.
        self.assertEqual(len(seen_kwargs), 1)
        self.assertIn("deny_from_permission_message", seen_kwargs[0])
        self.assertNotIn("retry_from_permission", seen_kwargs[0])

        # And the run is no longer parked on the approval.
        run = self.store.get_run(result["run"]["id"])
        step = self.store.list_task_steps(task["id"])[0]
        assert run is not None
        self.assertNotEqual(run["status"], "waiting_for_user")
        self.assertNotEqual(step["status"], "waiting_for_user")
        # The park is gone: the step's output is a turn result, not a checkpoint.
        self.assertNotIn("checkpoint", step["output"])
        resumed_events = [
            event
            for event in self.store.list_events(result["run"]["id"])
            if event["event_type"] == "task.step.approval_resumed"
        ]
        self.assertEqual(len(resumed_events), 1)
        self.assertEqual(resumed_events[0]["app_event"]["decision"], "deny")

    async def test_expired_approval_without_a_session_fails_the_step(self):
        _agent, task, result = self._prepare()
        approval = self.approvals.create_request(
            operation="provider.tool",
            run_id=result["run"]["id"],
            details={"tool_name": "Read"},
            expires_at=_past(),
        )
        await self._park_on_approval(result, approval_id=approval["id"])

        async def fake_stream(_sink, _session, **_kwargs):
            raise AssertionError("an expired approval must not be re-run")

        manager = _FakeSessionManager(None)
        with patch(
            "agent.task_orchestrator.get_session_manager", lambda: manager
        ), patch("agent.task_orchestrator.stream_claude_turn", fake_stream):
            await sweep_expired_approvals()
            await _drain_resumes()

        event_types = [
            event["event_type"] for event in self.store.list_events(result["run"]["id"])
        ]
        self.assertEqual(event_types.count("task.step.failed"), 1)
        step = self.store.list_task_steps(task["id"])[0]
        # `on_failure` is ask_user, so the failed step parks for input — the
        # point is it failed first and left the approval park behind.
        self.assertEqual(step["output"]["checkpoint"]["reason"], "ask_user")

    async def test_a_fresh_approval_is_left_strictly_alone(self):
        _agent, task, result = self._prepare()
        approval = self.approvals.create_request(
            operation="provider.tool",
            run_id=result["run"]["id"],
            details={"tool_name": "Read"},
            expires_at=_future(),
        )
        await self._park_on_approval(result, approval_id=approval["id"])

        async def fake_stream(_sink, _session, **_kwargs):
            raise AssertionError("a live approval must not resume anything")

        manager = _FakeSessionManager(_ParkedSession())
        with patch(
            "agent.task_orchestrator.get_session_manager", lambda: manager
        ), patch("agent.task_orchestrator.stream_claude_turn", fake_stream):
            swept = await sweep_expired_approvals()
            await _drain_resumes()

        self.assertEqual(swept, [])
        self.assertEqual(self.approvals.get_request(approval["id"])["status"], "pending")

        run = self.store.get_run(result["run"]["id"])
        step = self.store.list_task_steps(task["id"])[0]
        assert run is not None
        self.assertEqual(run["status"], "waiting_for_user")
        self.assertEqual(step["output"]["checkpoint"]["reason"], "approval_required")

    async def test_expired_approval_with_no_parked_run_is_still_recorded(self):
        approval = self.approvals.create_request(
            operation="provider.tool",
            run_id="run_that_never_parked",
            details={"tool_name": "Read"},
            expires_at=_past(),
        )

        swept = await sweep_expired_approvals()

        self.assertEqual(len(swept), 1)
        self.assertFalse(swept[0]["resumed"])
        self.assertEqual(self.approvals.get_request(approval["id"])["status"], "expired")

    async def test_scheduler_tick_runs_the_sweep(self):
        calls: list[int] = []

        async def fake_sweep():
            calls.append(1)
            return []

        with patch("agent.approval_resume.sweep_expired_approvals", fake_sweep), patch(
            "agent.scheduler.TaskScheduler._sweep_cli_agents_if_due",
            lambda _self: asyncio.sleep(0),
        ):
            await scheduler.TaskScheduler().trigger_once()

        self.assertEqual(len(calls), 1)


class ResumeMeetsExpiredApprovalTest(_DbBackedTest):
    """A resume must not proceed on an approval whose deadline already passed."""

    async def test_resume_denies_an_approval_that_is_past_its_deadline(self):
        _agent, task, result = self._prepare()
        approval = self.approvals.create_request(
            operation="provider.tool",
            run_id=result["run"]["id"],
            details={"tool_name": "Read"},
            expires_at=_past(),
        )
        await self._park_on_approval(result, approval_id=approval["id"])

        seen_kwargs: list[dict] = []

        async def fake_stream(_sink, _session, **kwargs):
            seen_kwargs.append(kwargs)
            return True

        manager = _FakeSessionManager(_ParkedSession())
        # A manual resume from the app: no decision passed, the approval row is
        # still `pending` on disk, and its deadline has passed.
        resume = resume_task_orchestration(task["id"])
        assert resume is not None
        self.assertNotIn("permission_decision", resume["execution"])

        with patch(
            "agent.task_orchestrator.get_session_manager", lambda: manager
        ), patch("agent.task_orchestrator.stream_claude_turn", fake_stream):
            await execute_task_orchestration(resume["execution"])

        self.assertEqual(len(seen_kwargs), 1)
        self.assertIn("deny_from_permission_message", seen_kwargs[0])
        self.assertNotIn("retry_from_permission", seen_kwargs[0])
        # The denial is recorded on the row, not just acted on.
        self.assertEqual(self.approvals.get_request(approval["id"])["status"], "expired")

        step = self.store.list_task_steps(task["id"])[0]
        self.assertNotEqual(step["status"], "waiting_for_user")
        self.assertNotIn("checkpoint", step["output"])

    async def test_resume_stays_parked_for_an_approval_that_is_still_live(self):
        _agent, task, result = self._prepare()
        approval = self.approvals.create_request(
            operation="provider.tool",
            run_id=result["run"]["id"],
            details={"tool_name": "Read"},
            expires_at=_future(),
        )
        await self._park_on_approval(result, approval_id=approval["id"])

        async def fake_stream(_sink, _session, **_kwargs):
            raise AssertionError("an undecided, unexpired approval runs nothing")

        manager = _FakeSessionManager(_ParkedSession())
        resume = resume_task_orchestration(task["id"])
        assert resume is not None

        with patch(
            "agent.task_orchestrator.get_session_manager", lambda: manager
        ), patch("agent.task_orchestrator.stream_claude_turn", fake_stream):
            await execute_task_orchestration(resume["execution"])

        step = self.store.list_task_steps(task["id"])[0]
        self.assertEqual(step["status"], "waiting_for_user")
        self.assertEqual(step["output"]["checkpoint"]["reason"], "approval_required")
        self.assertEqual(self.approvals.get_request(approval["id"])["status"], "pending")


class ApprovalParkNotificationTest(_DbBackedTest):
    """The push body has to carry the tool name and target (B7)."""

    async def test_checkpoint_prompt_and_notification_name_the_tool(self):
        _agent, task, result = self._prepare()
        approval = self.approvals.create_request(
            operation="provider.tool",
            run_id=result["run"]["id"],
            details={"tool_name": "Read"},
            expires_at=_future(),
        )

        pushes: list[dict] = []

        def fake_push(**kwargs):
            pushes.append(kwargs)

        with patch(
            "agent.task_orchestrator._push_notification_best_effort", fake_push
        ):
            await self._park_on_approval(result, approval_id=approval["id"])

        checkpoint = self.store.get_task_checkpoint(task["id"])["checkpoint"]
        self.assertEqual(
            checkpoint["prompt"], "Read needs your approval: /etc/hosts"
        )
        self.assertEqual(len(pushes), 1)
        self.assertEqual(pushes[0]["body"], "Read needs your approval: /etc/hosts")

    async def test_non_workflow_block_uses_the_same_words(self):
        """The `blocked` path describes the same park with the same vocabulary."""
        agent = self.store.create_agent(
            name="plain bot",
            system_prompt="Just answer.",
            provider_id="anthropic",
        )
        task = self.store.create_task(
            title="Plain task",
            assigned_agent_id=agent["id"],
            goal="Answer something.",
        )
        prepared = prepare_task_orchestration(
            task["id"], provider_id="anthropic", auto_start=False
        )
        assert prepared is not None

        async def fake_create_chat_session(**_kwargs):
            return object()

        async def fake_stream(sink, _session, **_kwargs):
            await sink.send_json(_permission_event(approval_id="apr_plain"))
            return False

        with patch(
            "agent.task_orchestrator.create_chat_session", fake_create_chat_session
        ), patch("agent.task_orchestrator.stream_claude_turn", fake_stream):
            await execute_task_orchestration(prepared["execution"])

        updated = self.store.get_task(task["id"])
        assert updated is not None
        error = updated["error"] or {}
        self.assertEqual(updated["status"], "blocked")
        self.assertEqual(error["reason"], "approval_required")
        self.assertEqual(error["message"], "Read needs your approval: /etc/hosts")
        self.assertEqual(error["tool_name"], "Read")
        self.assertEqual(error["tool_target"], "/etc/hosts")
        self.assertTrue(error["required_user_action"])


if __name__ == "__main__":
    unittest.main()
