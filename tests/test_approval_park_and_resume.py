"""Approval parking and resume: the run must stop, and it must start again.

Three behaviours are pinned here, one per defect the walkthrough exposed:

1. A turn that stops for an approval is *parked*, not failed. The step's
   `on_failure` policy must not run, because its `ask_user` branch overwrote
   the approval checkpoint a second after it was written and the phone showed
   two conflicting "waiting for you" cards.
2. Resuming a parked run continues the *same* provider turn when the session is
   still holding the permission callback, and only re-executes the step when
   that session is gone (server restarted). Re-sending the message to a live
   parked session dies with "Another Claude turn is already in progress".
3. Deciding an approval resumes the run it was blocking — and nothing else.
"""

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store, browser_session_store  # noqa: E402
from agent.approval_resume import maybe_resume_run_for_decision  # noqa: E402
from agent.task_orchestrator import (  # noqa: E402
    execute_task_orchestration,
    prepare_task_orchestration,
    resume_task_orchestration,
)
from approvals import approval_store  # noqa: E402
from audit import audit_store  # noqa: E402
from core import database  # noqa: E402
from policy import policy_store  # noqa: E402
from routes import approvals as approvals_routes  # noqa: E402
from routes.deps import verify_api_key  # noqa: E402


class _ParkedSession:
    """A provider session holding an unanswered permission callback."""

    provider_id = "anthropic"
    session_id = "native-1"
    has_pending_permission = True


class _FakeSessionManager:
    def __init__(self, session=None):
        self._session = session
        self.asked_for: list[str] = []

    def get_session_if_exists(self, project_name: str):
        self.asked_for.append(project_name)
        return self._session


def _permission_event(
    *, approval_id: str = "apr_test", tool_name: str = "Read", target: str = "/etc/hosts"
) -> dict:
    """The payload `chat_stream_service` sends when a tool call is gated."""
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


class ApprovalParkAndResumeTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "approval_resume.db"
        agent_store._agent_store = None
        browser_session_store._browser_session_store = None
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None
        database.init_db()
        self.store = agent_store.get_agent_store()

    def tearDown(self):
        agent_store._agent_store = None
        browser_session_store._browser_session_store = None
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    # -- helpers -------------------------------------------------------

    def _prepare(self, flow_json):
        agent = self.store.create_agent(
            name="approval bot",
            system_prompt="Run workflow steps.",
            provider_id="anthropic",
            flow_json=flow_json,
        )
        task = self.store.create_task(
            title="Run workflow",
            assigned_agent_id=agent["id"],
            goal="Finish the workflow.",
        )
        result = prepare_task_orchestration(
            task["id"],
            provider_id="anthropic",
            auto_start=False,
        )
        assert result is not None
        return agent, task, result

    def _single_llm_step_flow(self):
        return [
            {
                "id": "inspect",
                "type": "llm",
                "name": "Inspect",
                "on_failure": {"type": "ask_user", "prompt": "What should I do next?"},
            }
        ]

    async def _park_on_approval(self, task, result, *, approval_id="apr_test"):
        """Run the task until its llm step parks on a permission prompt."""

        async def fake_create_chat_session(**_kwargs):
            return object()

        async def fake_stream(sink, _session, **_kwargs):
            await sink.send_json(_permission_event(approval_id=approval_id))
            return False

        with patch(
            "agent.task_orchestrator.create_chat_session", fake_create_chat_session
        ), patch("agent.task_orchestrator.stream_claude_turn", fake_stream):
            await execute_task_orchestration(result["execution"])

    # -- B1: parking is not failing ------------------------------------

    async def test_permission_park_does_not_run_the_failure_policy(self):
        _agent, task, result = self._prepare(self._single_llm_step_flow())

        failure_policy = Mock(name="_apply_workflow_failure_policy")
        with patch(
            "agent.task_orchestrator._apply_workflow_failure_policy", failure_policy
        ):
            await self._park_on_approval(task, result)

        failure_policy.assert_not_called()

        step = self.store.list_task_steps(task["id"])[0]
        run = self.store.get_run(result["run"]["id"])
        checkpoint = self.store.get_task_checkpoint(task["id"])
        assert run is not None and checkpoint is not None

        self.assertEqual(step["status"], "waiting_for_user")
        self.assertEqual(run["status"], "waiting_for_user")
        self.assertEqual(checkpoint["checkpoint"]["reason"], "approval_required")
        # The ask_user prompt never got a chance to overwrite this one.
        self.assertNotEqual(
            checkpoint["checkpoint"]["prompt"], "What should I do next?"
        )
        self.assertEqual(checkpoint["checkpoint"]["approval_id"], "apr_test")
        self.assertEqual(checkpoint["checkpoint"]["tool_name"], "Read")
        self.assertEqual(checkpoint["checkpoint"]["tool_target"], "/etc/hosts")

        updated_task = self.store.get_task(task["id"])
        assert updated_task is not None
        self.assertEqual(
            updated_task["metadata"]["active_checkpoint"]["reason"],
            "approval_required",
        )

    async def test_only_one_waiting_for_user_event_is_recorded(self):
        _agent, task, result = self._prepare(self._single_llm_step_flow())

        await self._park_on_approval(task, result)

        event_types = [
            event["event_type"]
            for event in self.store.list_events(result["run"]["id"])
        ]
        self.assertEqual(event_types.count("task.step.waiting_for_user"), 1)
        self.assertNotIn("task.step.failed", event_types)

    # -- B2: resuming a parked turn ------------------------------------

    async def test_resume_continues_the_same_turn_when_the_session_is_parked(self):
        _agent, task, result = self._prepare(self._single_llm_step_flow())
        await self._park_on_approval(task, result)

        seen_kwargs: list[dict] = []
        created_sessions: list[dict] = []

        async def fake_create_chat_session(**kwargs):
            created_sessions.append(kwargs)
            return object()

        async def fake_stream(_sink, _session, **kwargs):
            seen_kwargs.append(kwargs)
            return True

        manager = _FakeSessionManager(_ParkedSession())
        resume = resume_task_orchestration(
            task["id"], permission_decision="approve_once"
        )
        assert resume is not None
        self.assertEqual(resume["execution"]["permission_decision"], "approve_once")

        with patch(
            "agent.task_orchestrator.get_session_manager", lambda: manager
        ), patch(
            "agent.task_orchestrator.create_chat_session", fake_create_chat_session
        ), patch("agent.task_orchestrator.stream_claude_turn", fake_stream):
            await execute_task_orchestration(resume["execution"])

        self.assertEqual(manager.asked_for, [f"task:{task['id']}"])
        self.assertEqual(len(seen_kwargs), 1)
        self.assertTrue(seen_kwargs[0].get("retry_from_permission"))
        self.assertNotIn("user_message", seen_kwargs[0])
        # A parked turn is continued, never re-opened.
        self.assertEqual(created_sessions, [])

        step = self.store.list_task_steps(task["id"])[0]
        run = self.store.get_run(result["run"]["id"])
        assert run is not None
        self.assertEqual(step["status"], "completed")
        self.assertEqual(run["status"], "completed")
        resume_events = [
            event
            for event in self.store.list_events(result["run"]["id"])
            if event["event_type"] == "task.step.approval_resumed"
        ]
        self.assertEqual(len(resume_events), 1)
        self.assertEqual(resume_events[0]["app_event"]["mode"], "same_turn")
        self.assertEqual(resume_events[0]["app_event"]["decision"], "allow")

    async def test_resume_falls_back_to_send_message_when_the_session_is_gone(self):
        _agent, task, result = self._prepare(self._single_llm_step_flow())
        await self._park_on_approval(task, result)

        seen_kwargs: list[dict] = []
        created_sessions: list[dict] = []

        async def fake_create_chat_session(**kwargs):
            created_sessions.append(kwargs)
            return object()

        async def fake_stream(_sink, _session, **kwargs):
            seen_kwargs.append(kwargs)
            return True

        # Server restarted: the session cache no longer holds the turn.
        manager = _FakeSessionManager(None)
        resume = resume_task_orchestration(
            task["id"], permission_decision="approve_once"
        )
        assert resume is not None

        with patch(
            "agent.task_orchestrator.get_session_manager", lambda: manager
        ), patch(
            "agent.task_orchestrator.create_chat_session", fake_create_chat_session
        ), patch("agent.task_orchestrator.stream_claude_turn", fake_stream):
            await execute_task_orchestration(resume["execution"])

        self.assertEqual(len(seen_kwargs), 1)
        self.assertNotIn("retry_from_permission", seen_kwargs[0])
        self.assertIn("user_message", seen_kwargs[0])
        self.assertEqual(len(created_sessions), 1)

        step = self.store.list_task_steps(task["id"])[0]
        self.assertEqual(step["status"], "completed")
        resume_events = [
            event
            for event in self.store.list_events(result["run"]["id"])
            if event["event_type"] == "task.step.approval_resumed"
        ]
        self.assertEqual(resume_events[0]["app_event"]["mode"], "re_execute")

    async def test_deny_is_sent_into_the_parked_turn(self):
        _agent, task, result = self._prepare(self._single_llm_step_flow())
        await self._park_on_approval(task, result)

        seen_kwargs: list[dict] = []

        async def fake_stream(_sink, _session, **kwargs):
            seen_kwargs.append(kwargs)
            return True

        manager = _FakeSessionManager(_ParkedSession())
        resume = resume_task_orchestration(task["id"], permission_decision="deny")
        assert resume is not None

        with patch(
            "agent.task_orchestrator.get_session_manager", lambda: manager
        ), patch("agent.task_orchestrator.stream_claude_turn", fake_stream):
            await execute_task_orchestration(resume["execution"])

        self.assertEqual(len(seen_kwargs), 1)
        self.assertIn("deny_from_permission_message", seen_kwargs[0])
        self.assertNotIn("retry_from_permission", seen_kwargs[0])

    async def test_deny_without_a_parked_session_fails_instead_of_re_running(self):
        _agent, task, result = self._prepare(self._single_llm_step_flow())
        await self._park_on_approval(task, result)

        async def fake_stream(_sink, _session, **_kwargs):
            raise AssertionError("the denied step must not be re-run")

        manager = _FakeSessionManager(None)
        resume = resume_task_orchestration(task["id"], permission_decision="deny")
        assert resume is not None

        with patch(
            "agent.task_orchestrator.get_session_manager", lambda: manager
        ), patch("agent.task_orchestrator.stream_claude_turn", fake_stream):
            await execute_task_orchestration(resume["execution"])

        step = self.store.list_task_steps(task["id"])[0]
        # on_failure is `ask_user`, so the failed step parks for input — but it
        # failed first, which is the point: nothing the user denied was run.
        self.assertEqual(
            [
                event["event_type"]
                for event in self.store.list_events(result["run"]["id"])
            ].count("task.step.failed"),
            1,
        )
        self.assertEqual(step["output"]["checkpoint"]["reason"], "ask_user")

    async def test_resume_without_a_decision_stays_parked(self):
        _agent, task, result = self._prepare(self._single_llm_step_flow())
        approval = approval_store.get_approval_store().create_request(
            operation="provider.tool",
            run_id=result["run"]["id"],
            details={"tool_name": "Read"},
        )
        await self._park_on_approval(task, result, approval_id=approval["id"])

        async def fake_stream(_sink, _session, **_kwargs):
            raise AssertionError("an undecided approval must not run anything")

        manager = _FakeSessionManager(_ParkedSession())
        resume = resume_task_orchestration(task["id"])
        assert resume is not None
        self.assertNotIn("permission_decision", resume["execution"])

        with patch(
            "agent.task_orchestrator.get_session_manager", lambda: manager
        ), patch("agent.task_orchestrator.stream_claude_turn", fake_stream):
            await execute_task_orchestration(resume["execution"])

        step = self.store.list_task_steps(task["id"])[0]
        run = self.store.get_run(result["run"]["id"])
        assert run is not None
        self.assertEqual(step["status"], "waiting_for_user")
        self.assertEqual(run["status"], "waiting_for_user")
        self.assertEqual(step["output"]["checkpoint"]["reason"], "approval_required")
        self.assertEqual(step["output"]["checkpoint"]["approval_id"], approval["id"])
        event_types = [
            event["event_type"]
            for event in self.store.list_events(result["run"]["id"])
        ]
        self.assertIn("task.step.approval_still_pending", event_types)

    async def test_decided_approval_row_is_read_when_no_decision_is_passed(self):
        _agent, task, result = self._prepare(self._single_llm_step_flow())
        approvals = approval_store.get_approval_store()
        approval = approvals.create_request(
            operation="provider.tool",
            run_id=result["run"]["id"],
            details={"tool_name": "Read"},
        )
        await self._park_on_approval(task, result, approval_id=approval["id"])
        approvals.create_decision(approval_id=approval["id"], decision="approve_once")

        seen_kwargs: list[dict] = []

        async def fake_stream(_sink, _session, **kwargs):
            seen_kwargs.append(kwargs)
            return True

        manager = _FakeSessionManager(_ParkedSession())
        resume = resume_task_orchestration(task["id"])
        assert resume is not None

        with patch(
            "agent.task_orchestrator.get_session_manager", lambda: manager
        ), patch("agent.task_orchestrator.stream_claude_turn", fake_stream):
            await execute_task_orchestration(resume["execution"])

        self.assertEqual(len(seen_kwargs), 1)
        self.assertTrue(seen_kwargs[0].get("retry_from_permission"))
        self.assertEqual(
            self.store.list_task_steps(task["id"])[0]["status"], "completed"
        )

    async def test_a_user_message_does_not_complete_an_approval_park(self):
        _agent, task, result = self._prepare(self._single_llm_step_flow())
        await self._park_on_approval(task, result)
        step = self.store.list_task_steps(task["id"])[0]
        self.store.append_step_user_response(
            task_id=task["id"],
            step_id=step["id"],
            message="go ahead",
            remember=False,
        )

        async def fake_stream(_sink, _session, **_kwargs):
            raise AssertionError("nothing should run: the approval is undecided")

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

    # -- B3: the decision endpoint resumes the run ---------------------

    async def test_decision_spawns_a_resume_for_the_matching_checkpoint(self):
        _agent, task, result = self._prepare(self._single_llm_step_flow())
        approval = approval_store.get_approval_store().create_request(
            operation="provider.tool",
            run_id=result["run"]["id"],
            details={"tool_name": "Read"},
        )
        await self._park_on_approval(task, result, approval_id=approval["id"])

        executed: list[dict] = []

        async def fake_execute(execution):
            executed.append(execution)

        with patch("agent.approval_resume.execute_task_orchestration", fake_execute):
            resumed = await maybe_resume_run_for_decision(
                approval, decision="approve_once"
            )
            await asyncio.sleep(0)

        assert resumed is not None
        self.assertTrue(resumed["resumed"])
        self.assertEqual(resumed["run_id"], result["run"]["id"])
        self.assertEqual(resumed["task_id"], task["id"])
        self.assertEqual(len(executed), 1)
        self.assertEqual(executed[0]["permission_decision"], "approve_once")
        self.assertEqual(executed[0]["run_id"], result["run"]["id"])

        # The park this decision answered is over, so the next one is not a
        # "repeat" that gets its notification suppressed.
        updated_task = self.store.get_task(task["id"])
        assert updated_task is not None
        self.assertNotIn("active_checkpoint", updated_task["metadata"] or {})

    async def test_decision_for_a_different_approval_resumes_nothing(self):
        _agent, task, result = self._prepare(self._single_llm_step_flow())
        approvals = approval_store.get_approval_store()
        parked = approvals.create_request(
            operation="provider.tool",
            run_id=result["run"]["id"],
            details={"tool_name": "Read"},
        )
        other = approvals.create_request(
            operation="process.terminal",
            run_id=result["run"]["id"],
            details={"command": "rm -rf /"},
        )
        await self._park_on_approval(task, result, approval_id=parked["id"])

        executed: list[dict] = []

        async def fake_execute(execution):
            executed.append(execution)

        with patch("agent.approval_resume.execute_task_orchestration", fake_execute):
            resumed = await maybe_resume_run_for_decision(other, decision="approve_once")
            await asyncio.sleep(0)

        self.assertIsNone(resumed)
        self.assertEqual(executed, [])

    async def test_decision_for_a_run_that_is_not_parked_resumes_nothing(self):
        run = self.store.create_run(
            project_name="demo",
            provider_id="anthropic",
            model="sonnet",
            title="Chat run",
            goal="Answer a question",
            cwd="/tmp/demo",
        )
        approval = approval_store.get_approval_store().create_request(
            operation="provider.tool",
            run_id=run["id"],
            details={"tool_name": "Read"},
        )

        executed: list[dict] = []

        async def fake_execute(execution):
            executed.append(execution)

        with patch("agent.approval_resume.execute_task_orchestration", fake_execute):
            resumed = await maybe_resume_run_for_decision(
                approval, decision="approve_once"
            )
            await asyncio.sleep(0)

        self.assertIsNone(resumed)
        self.assertEqual(executed, [])


class ApprovalDecisionRouteResumeTest(unittest.IsolatedAsyncioTestCase):
    """The wiring itself: POST /decision has to reach the resume."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "approval_decision_route.db"
        agent_store._agent_store = None
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None
        database.init_db()

        app = FastAPI()
        app.include_router(approvals_routes.router)
        app.dependency_overrides[verify_api_key] = lambda: "test-api-key"
        self.client = TestClient(app)

    def tearDown(self):
        agent_store._agent_store = None
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    async def test_decision_endpoint_calls_the_resume_hand_off(self):
        approval = approval_store.get_approval_store().create_request(
            operation="provider.tool",
            run_id="run_demo",
            details={"tool_name": "Read"},
        )

        calls: list[tuple] = []

        async def fake_resume(approval_payload, *, decision):
            calls.append((approval_payload, decision))
            return {"resumed": True, "run_id": "run_demo"}

        with patch("routes.approvals.maybe_resume_run_for_decision", fake_resume):
            response = self.client.post(
                f"/api/approvals/{approval['id']}/decision",
                json={"decision": "approve_once", "scope": "once"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["resume"], {"resumed": True, "run_id": "run_demo"})
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0]["id"], approval["id"])
        self.assertEqual(calls[0][1], "approve_once")


if __name__ == "__main__":
    unittest.main()
