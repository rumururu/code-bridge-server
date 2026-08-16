"""A denied tool call must not be reported as a completed step.

Live run `run_af371e422a034dda9f4b83fa20a16541` (2026-08-16) is the defect this
module pins. An approval expired, the sweep resumed the parked turn down the
deny path, the SDK delivered the denial, and the model said in plain Korean
that it could not finish. The provider turn nonetheless ended
`{"type": "result", "subtype": "success"}`, and the runtime read that as the
step having succeeded: `task.step.completed` and `task.execution.completed`
fired with "Workflow completed." The user was told a refusal had worked.

The rule pinned here is fail-closed and stated in
`task_orchestrator._DENIED_STEP_MESSAGE`: **a step in which any tool call was
refused ends `failed`**, and its `on_failure` policy runs like any other
failure. The runtime cannot tell whether the model met the step's
`success_criteria` some other way without reading its prose, which is a guess;
and of the two possible mistakes, a false "failed" puts a person back in the
loop while a false "completed" is never corrected.

The last test here is the deliberate cost of that rule: a denial followed by an
approved tool that finishes the work *still* fails the step. That is the choice,
written down so it cannot drift silently.
"""

import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store, browser_session_store  # noqa: E402
from agent.approval_resume import sweep_expired_approvals  # noqa: E402
from agent.task_orchestrator import (  # noqa: E402
    _DENIED_STEP_MESSAGE,
    execute_task_orchestration,
    prepare_task_orchestration,
    resume_task_orchestration,
)
from approvals import approval_store  # noqa: E402
from audit import audit_store  # noqa: E402
from core import database  # noqa: E402
from policy import policy_store  # noqa: E402


class _ParkedSession:
    provider_id = "anthropic"
    session_id = "native-1"
    has_pending_permission = True


class _FakeSessionManager:
    def __init__(self, session=None):
        self._session = session

    def get_session_if_exists(self, project_name: str):
        return self._session


def _permission_event(*, approval_id: str, tool_name: str = "Read", target: str = "/etc/hosts") -> dict:
    """What `chat_stream_service` sends when a tool call parks on approval."""
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


def _policy_denied_event(*, tool_name: str = "Bash", reason: str = "Denied by standing rule.") -> dict:
    """What `chat_stream_service._handle_control_request` emits for a policy deny.

    This one never parks the run: the SDK is answered with a deny result and
    the same turn keeps going, which is exactly why the turn's clean ending
    says nothing about whether the step did its work.
    """
    return {
        "type": "app_event",
        "schema_version": 1,
        "event": "permission.policy_denied",
        "provider_id": "anthropic",
        "session_id": "native-1",
        "turn_id": "turn-1",
        "sequence": 3,
        "title": "permission denied by policy",
        "level": "warning",
        "detail": tool_name,
        "data": {"request_id": "provider-req-1", "reason": reason},
    }


def _past() -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()


#: What the model says after being handed the denial. The runtime must never
#: need to read this to decide the outcome — it is asserted only as *evidence*
#: carried onto the failed step.
_MODEL_REPORT = (
    "파일 읽기 단계를 완료하지 못했습니다. Read 도구로 읽으려 했으나 "
    "도구 호출이 거부되어 마지막 줄을 확인할 수 없었습니다."
)


class DeniedToolCallStepOutcomeTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "denied_tool_call.db"
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
            task["id"], provider_id="anthropic", auto_start=False
        )
        assert result is not None
        return agent, task, result

    def _single_llm_step_flow(self):
        return [
            {
                "id": "inspect",
                "type": "llm",
                "name": "Inspect",
                "success_criteria": "the last line of the file is reported",
                "on_failure": {"type": "ask_user", "prompt": "What should I do next?"},
            }
        ]

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

    def _event_types(self, run_id: str) -> list[str]:
        return [event["event_type"] for event in self.store.list_events(run_id)]

    async def _run_turn(self, execution, *, stream, session=_ParkedSession()):
        manager = _FakeSessionManager(session)

        async def fake_create_chat_session(**_kwargs):
            return object()

        with patch(
            "agent.task_orchestrator.get_session_manager", lambda: manager
        ), patch(
            "agent.task_orchestrator.create_chat_session", fake_create_chat_session
        ), patch("agent.task_orchestrator.stream_claude_turn", stream):
            await execute_task_orchestration(execution)

    # -- the defect ----------------------------------------------------

    async def test_denied_tool_call_fails_the_step_instead_of_completing(self):
        _agent, task, result = self._prepare(self._single_llm_step_flow())
        approval = self.approvals.create_request(
            operation="provider.tool",
            run_id=result["run"]["id"],
            details={"tool_name": "Read"},
        )
        await self._park_on_approval(result, approval_id=approval["id"])

        async def fake_stream(sink, _session, **_kwargs):
            # The live shape: the model is handed the denial, reports it could
            # not do the work, and the provider turn ends *cleanly*.
            await sink.send_json({"type": "result", "result": _MODEL_REPORT})
            return True

        resume = resume_task_orchestration(task["id"], permission_decision="deny")
        assert resume is not None
        await self._run_turn(resume["execution"], stream=fake_stream)

        step = self.store.list_task_steps(task["id"])[0]
        run = self.store.get_run(result["run"]["id"])
        assert run is not None

        event_types = self._event_types(result["run"]["id"])
        self.assertEqual(event_types.count("task.step.failed"), 1)
        self.assertNotIn("task.step.completed", event_types)
        # The line the live run got wrong: "Workflow completed." on a refusal.
        self.assertNotIn("task.execution.completed", event_types)

        # `on_failure` ran, and its ask_user branch parked the step for a person
        # — which is what the model was asking for with its `<ask_user/>` tag.
        self.assertEqual(step["status"], "waiting_for_user")
        self.assertEqual(step["output"]["checkpoint"]["reason"], "ask_user")
        self.assertEqual(step["output"]["message"], _DENIED_STEP_MESSAGE)
        self.assertEqual(step["output"]["reason"], "ask_user")

        denied = step["output"]["denied_tool_calls"]
        self.assertEqual(len(denied), 1)
        self.assertEqual(denied[0]["source"], "approval_decision")
        self.assertEqual(denied[0]["approval_id"], approval["id"])
        self.assertEqual(denied[0]["tool_name"], "Read")
        self.assertEqual(denied[0]["decision"], "deny")
        # The model's own account is kept as evidence, never parsed for a verdict.
        self.assertEqual(step["output"]["result"], _MODEL_REPORT)

    async def test_expiry_sweep_deny_lands_the_same_way(self):
        _agent, task, result = self._prepare(self._single_llm_step_flow())
        approval = self.approvals.create_request(
            operation="provider.tool",
            run_id=result["run"]["id"],
            details={"tool_name": "Read"},
            expires_at=_past(),
        )
        await self._park_on_approval(result, approval_id=approval["id"])

        async def fake_stream(sink, _session, **_kwargs):
            await sink.send_json({"type": "result", "result": _MODEL_REPORT})
            return True

        manager = _FakeSessionManager(_ParkedSession())
        with patch(
            "agent.task_orchestrator.get_session_manager", lambda: manager
        ), patch("agent.task_orchestrator.stream_claude_turn", fake_stream):
            swept = await sweep_expired_approvals()
            # The sweep spawns the resume as a background task.
            import asyncio

            for _ in range(20):
                await asyncio.sleep(0)

        self.assertEqual(len(swept), 1)
        self.assertEqual(swept[0]["decision"], "expired")

        step = self.store.list_task_steps(task["id"])[0]
        event_types = self._event_types(result["run"]["id"])
        self.assertEqual(event_types.count("task.step.failed"), 1)
        self.assertNotIn("task.execution.completed", event_types)
        self.assertEqual(step["output"]["checkpoint"]["reason"], "ask_user")
        self.assertEqual(
            step["output"]["denied_tool_calls"][0]["decision"], "expired"
        )

    async def test_a_policy_denied_tool_fails_the_step(self):
        """A standing rule refusing a tool never parks the run — same rule."""
        _agent, task, result = self._prepare(self._single_llm_step_flow())

        async def fake_stream(sink, _session, **_kwargs):
            await sink.send_json(_policy_denied_event())
            await sink.send_json({"type": "result", "result": "Could not run it."})
            return True

        async def fake_create_chat_session(**_kwargs):
            return object()

        with patch(
            "agent.task_orchestrator.create_chat_session", fake_create_chat_session
        ), patch("agent.task_orchestrator.stream_claude_turn", fake_stream):
            await execute_task_orchestration(result["execution"])

        step = self.store.list_task_steps(task["id"])[0]
        event_types = self._event_types(result["run"]["id"])
        self.assertEqual(event_types.count("task.step.failed"), 1)
        self.assertNotIn("task.step.completed", event_types)
        denied = step["output"]["denied_tool_calls"]
        self.assertEqual(denied[0]["source"], "policy")
        self.assertEqual(denied[0]["tool_name"], "Bash")
        self.assertEqual(denied[0]["reason"], "Denied by standing rule.")

    async def test_a_policy_denied_tool_fails_a_workflow_less_run(self):
        """The single-turn path said "Provider turn completed." just as wrongly."""
        _agent, task, result = self._prepare([])

        async def fake_stream(sink, _session, **_kwargs):
            await sink.send_json(_policy_denied_event())
            return True

        async def fake_create_chat_session(**_kwargs):
            return object()

        with patch(
            "agent.task_orchestrator.create_chat_session", fake_create_chat_session
        ), patch("agent.task_orchestrator.stream_claude_turn", fake_stream):
            await execute_task_orchestration(result["execution"])

        run = self.store.get_run(result["run"]["id"])
        updated_task = self.store.get_task(task["id"])
        assert run is not None and updated_task is not None
        self.assertEqual(run["status"], "failed")
        self.assertEqual(updated_task["error"]["message"], _DENIED_STEP_MESSAGE)
        self.assertIn("task.execution.failed", self._event_types(result["run"]["id"]))
        self.assertNotIn(
            "task.execution.completed", self._event_types(result["run"]["id"])
        )

    # -- the rule's boundaries -----------------------------------------

    async def test_an_ordinary_step_still_completes(self):
        """No denial anywhere: nothing about a clean run changes."""
        _agent, task, result = self._prepare(self._single_llm_step_flow())

        async def fake_stream(sink, _session, **_kwargs):
            await sink.send_json({"type": "result", "result": "Last line: EOF."})
            return True

        async def fake_create_chat_session(**_kwargs):
            return object()

        with patch(
            "agent.task_orchestrator.create_chat_session", fake_create_chat_session
        ), patch("agent.task_orchestrator.stream_claude_turn", fake_stream):
            await execute_task_orchestration(result["execution"])

        step = self.store.list_task_steps(task["id"])[0]
        run = self.store.get_run(result["run"]["id"])
        assert run is not None
        self.assertEqual(step["status"], "completed")
        self.assertEqual(run["status"], "completed")
        self.assertNotIn("denied_tool_calls", step["output"])
        self.assertNotIn("task.step.failed", self._event_types(result["run"]["id"]))

    async def test_an_approved_step_after_a_park_still_completes(self):
        """Parking is not denial: an approval that is granted completes as before."""
        _agent, task, result = self._prepare(self._single_llm_step_flow())
        approval = self.approvals.create_request(
            operation="provider.tool",
            run_id=result["run"]["id"],
            details={"tool_name": "Read"},
        )
        await self._park_on_approval(result, approval_id=approval["id"])

        async def fake_stream(sink, _session, **_kwargs):
            await sink.send_json({"type": "result", "result": "Last line: EOF."})
            return True

        resume = resume_task_orchestration(
            task["id"], permission_decision="approve_once"
        )
        assert resume is not None
        await self._run_turn(resume["execution"], stream=fake_stream)

        step = self.store.list_task_steps(task["id"])[0]
        self.assertEqual(step["status"], "completed")
        self.assertNotIn("denied_tool_calls", step["output"])

    async def test_a_retry_attempt_does_not_inherit_the_previous_denial(self):
        """The record is scoped to one attempt, not to the step forever.

        `on_failure: retry` re-runs the step from the top. Carrying the last
        attempt's denial into it would fail an attempt in which nothing was
        refused — the same lie, pointing the other way.
        """
        _agent, task, result = self._prepare(
            [
                {
                    "id": "inspect",
                    "type": "llm",
                    "name": "Inspect",
                    "on_failure": {"type": "retry", "max_attempts": 1},
                }
            ]
        )

        turns: list[dict] = []

        async def deny_then_succeed(sink, _session, **kwargs):
            turns.append(kwargs)
            if len(turns) == 1:
                await sink.send_json(_policy_denied_event())
                await sink.send_json({"type": "result", "result": "Refused."})
            else:
                await sink.send_json({"type": "result", "result": "Last line: EOF."})
            return True

        async def fake_create_chat_session(**_kwargs):
            return object()

        with patch(
            "agent.task_orchestrator.create_chat_session", fake_create_chat_session
        ), patch("agent.task_orchestrator.stream_claude_turn", deny_then_succeed):
            await execute_task_orchestration(result["execution"])

        self.assertEqual(len(turns), 2)
        step = self.store.list_task_steps(task["id"])[0]
        event_types = self._event_types(result["run"]["id"])
        self.assertEqual(event_types.count("task.step.failed"), 1)
        self.assertIn("task.step.retry_scheduled", event_types)
        self.assertEqual(step["status"], "completed")
        self.assertNotIn("denied_tool_calls", step["output"])

    async def test_a_denial_still_fails_the_step_when_a_later_tool_is_approved(self):
        """The pinned cost of fail-closed.

        Tool A is denied, the model asks for tool B instead, B is approved, and
        the model finishes the work. The step still fails. The runtime has no
        trustworthy way to tell "finished it another way" from "wrote a
        confident summary of work it never did", and the recoverable mistake is
        the one that asks a person.
        """
        _agent, task, result = self._prepare(self._single_llm_step_flow())
        first = self.approvals.create_request(
            operation="provider.tool",
            run_id=result["run"]["id"],
            details={"tool_name": "Read"},
        )
        await self._park_on_approval(result, approval_id=first["id"])

        second = self.approvals.create_request(
            operation="process.terminal",
            run_id=result["run"]["id"],
            details={"tool_name": "Bash"},
        )

        async def deny_then_park_again(sink, _session, **_kwargs):
            await sink.send_json(
                _permission_event(
                    approval_id=second["id"], tool_name="Bash", target="tail -1 f.txt"
                )
            )
            return False

        deny = resume_task_orchestration(task["id"], permission_decision="deny")
        assert deny is not None
        await self._run_turn(deny["execution"], stream=deny_then_park_again)

        parked = self.store.list_task_steps(task["id"])[0]
        self.assertEqual(parked["status"], "waiting_for_user")
        self.assertEqual(parked["output"]["checkpoint"]["approval_id"], second["id"])
        # The first denial is on the step, not in a local variable that the
        # next process would never see.
        self.assertEqual(len(parked["output"]["denied_tool_calls"]), 1)

        async def finish_the_work(sink, _session, **_kwargs):
            await sink.send_json(
                {"type": "result", "result": "Used Bash instead. Last line: EOF."}
            )
            return True

        allow = resume_task_orchestration(
            task["id"], permission_decision="approve_once"
        )
        assert allow is not None
        await self._run_turn(allow["execution"], stream=finish_the_work)

        step = self.store.list_task_steps(task["id"])[0]
        event_types = self._event_types(result["run"]["id"])
        self.assertNotIn("task.step.completed", event_types)
        self.assertEqual(event_types.count("task.step.failed"), 1)
        self.assertEqual(step["output"]["message"], _DENIED_STEP_MESSAGE)
        self.assertEqual(step["output"]["checkpoint"]["reason"], "ask_user")
        self.assertEqual(len(step["output"]["denied_tool_calls"]), 1)


if __name__ == "__main__":
    unittest.main()
