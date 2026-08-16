"""A permission granted to an agent has to survive that agent's schedule.

``approve_rule`` writes a standing policy rule so an unattended run stops
asking — that is the only reason the phone's "항상 허용" button exists. But the
scope it picked made it useless for exactly the case it was built for.

Measured on the running server: an agent built purely by conversation,
scheduled every two minutes, parked on ``file.write``. Approving with a
standing rule wrote ``operation=file.write, scope=run:run_83d2cc9d…``. The very
next fire parked on the same approval — a new fire is a new run, so the rule
matched nothing.

The cause is that a conversation-built agent has no project: its
``details.project_name`` is the ``__global__`` sentinel, which the write side
refuses (rightly — every project-less agent shares it), and its runs carry no
``workspace_id``. So the scope fell all the way through to ``run:``.

These tests pin the fix end to end: the agent reaches the approval details, the
scope becomes the agent, and — the whole bug — the rule written on one run is
*found* on a different run of the same agent. A rule that is written and never
found is worse than no feature, so both sides are asserted, never just the
write.
"""

import sys
import tempfile
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store  # noqa: E402
from approvals import approval_service, approval_store  # noqa: E402
from audit import audit_store  # noqa: E402
from chat.chat_stream_service import (  # noqa: E402
    TurnState,
    _handle_control_request,
    _run_agent_identity,
)
from core import database  # noqa: E402
from policy import policy_store  # noqa: E402


class _FakeWebSocket:
    def __init__(self, *, agent_run_id: str | None = None):
        self.sent: list[dict] = []
        if agent_run_id is not None:
            self.agent_run_id = agent_run_id

    async def send_json(self, data):
        self.sent.append(data)


class _FakeSession:
    provider_id = "claude"
    session_id = "native-1"

    async def approve_pending_permissions_and_retry(self):
        yield {"type": "result", "result": "allowed", "total_cost_usd": 0}

    async def deny_pending_permissions(self, message="Permission denied by user."):
        yield {"type": "result", "result": message, "total_cost_usd": 0}


class _StoreFixture(unittest.IsolatedAsyncioTestCase):
    """Temp-database plumbing shared by both halves of the story."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "agent_scoped_rule_test.db"
        self._reset_stores()
        self.addCleanup(self._restore)
        database.init_db()
        self.agents = agent_store.get_agent_store()
        self.rules = policy_store.get_policy_rule_store()

    @staticmethod
    def _reset_stores() -> None:
        agent_store._agent_store = None
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None

    def _restore(self) -> None:
        self._reset_stores()
        database.DB_PATH = self._original_db_path

    def _agent_run(self, *, name: str = "Nightly report bot", agent_id: str | None = None):
        """One conversation-built agent with a task and a run, no project."""
        if agent_id is None:
            agent = self.agents.create_agent(name=name, system_prompt="Do the thing.")
            agent_id = agent["id"]
        task = self.agents.create_task(title="Fire", assigned_agent_id=agent_id)
        run = self.agents.create_run(
            project_name=None,
            agent_id=agent_id,
            provider_id="claude",
            title="Scheduled fire",
            task_id=task["id"],
        )
        return agent_id, run["id"]


class AgentIdentityReachesDetailsTest(_StoreFixture):
    """``agent_id`` has to get into the approval details, or nothing matches."""

    async def _park(self, run_id: str, *, project_name: str = "__global__") -> dict:
        websocket = _FakeWebSocket(agent_run_id=run_id)
        state = TurnState(provider_id="claude", provider="claude", session_id="native-1")
        await _handle_control_request(
            websocket,
            _FakeSession(),
            state,
            {
                "request_id": "provider-req-1",
                "request": {
                    "subtype": "can_use_tool",
                    "tool_name": "Write",
                    "tool_use_id": "tool-1",
                    "input": {"file_path": "/tmp/report.md", "content": "hi"},
                },
            },
            project_name,
        )
        prompt = next(m for m in websocket.sent if m.get("type") == "permission_required")
        request = approval_store.get_approval_store().get_request(prompt["approval_id"])
        assert request is not None
        return request

    async def test_the_approval_names_the_agent_that_asked(self):
        agent_id, run_id = self._agent_run(name="Nightly report bot")

        request = await self._park(run_id)

        self.assertEqual(request["details"]["agent_id"], agent_id)
        # Display only, so the confirmation can name the agent rather than
        # print an opaque `agent_…` id.
        self.assertEqual(request["details"]["agent_name"], "Nightly report bot")
        self.assertEqual(
            approval_service.standing_rule_scope_for_request(request),
            f"agent:{agent_id}",
        )

    async def test_a_chat_run_is_unchanged(self):
        # Interactive chat files every run under one pseudo-agent
        # (`routes/chat_ws.py:285`). An `agent:` rule there would grant the
        # operation to every chat on the server, so the details must not carry
        # one and the scope must stay exactly where it was: the run.
        _, run_id = self._agent_run(agent_id="agent_adhoc_dev")

        request = await self._park(run_id)

        self.assertNotIn("agent_id", request["details"])
        self.assertNotIn("agent_name", request["details"])
        self.assertEqual(
            approval_service.standing_rule_scope_for_request(request),
            f"run:{run_id}",
        )

    async def test_a_project_run_still_scopes_to_the_project(self):
        agent_id, run_id = self._agent_run()

        request = await self._park(run_id, project_name="code_bridge")

        # The agent is recorded either way — it is true, and useful in the
        # audit trail — but it must not take the scope away from the project.
        self.assertEqual(request["details"]["agent_id"], agent_id)
        self.assertEqual(
            approval_service.standing_rule_scope_for_request(request),
            "project:code_bridge",
        )

    def test_an_unresolvable_agent_falls_back_rather_than_guessing(self):
        # Fail closed: no agent means today's behaviour (`run:`), never
        # something broader.
        self.assertIsNone(_run_agent_identity("run_does_not_exist"))
        orphan = self.agents.create_run(provider_id="claude", title="No agent")
        self.assertIsNone(_run_agent_identity(orphan["id"]))


class AgentScopedRuleSurvivesTheScheduleTest(_StoreFixture):
    """The bug itself: the rule has to be found on the *next* fire."""

    def _ask(self, *, run_id: str, details: dict, operation: str = "file.write"):
        return approval_service.request_approval_for_operation(
            operation=operation,
            run_id=run_id,
            actor={"type": "agent_session"},
            details=details,
        )

    @staticmethod
    def _details(agent_id: str, **extra) -> dict:
        return {"project_name": "__global__", "tool_name": "Write", "agent_id": agent_id, **extra}

    def test_the_rule_granted_on_one_fire_answers_the_next_run(self):
        agent_id = "agent_5f3a"
        first = self._ask(run_id="run_first", details=self._details(agent_id))
        self.assertTrue(first["approval_required"])

        result = approval_service.decide_approval(
            first["approval"]["id"], decision="approve_rule"
        )
        self.assertEqual(result["rule"]["scope"], f"agent:{agent_id}")

        # The next scheduled fire is a different run. This is the assertion the
        # whole change exists for: under the old `run:` scope it parked again.
        second = self._ask(run_id="run_second", details=self._details(agent_id))
        self.assertFalse(second["approval_required"])
        self.assertTrue(second["allowed"])
        self.assertNotIn("approval", second)

    def test_the_rule_does_not_leak_to_another_agent(self):
        first = self._ask(run_id="run_first", details=self._details("agent_a"))
        approval_service.decide_approval(first["approval"]["id"], decision="approve_rule")

        other = self._ask(run_id="run_other", details=self._details("agent_b"))
        self.assertTrue(other["approval_required"])

    def test_a_workspace_still_wins_over_the_agent(self):
        # Nothing that resolves today may change. A workspace-bearing request
        # keeps its workspace scope.
        pending = self._ask(
            run_id="run_first",
            details=self._details("agent_a", workspace_id="ws_1"),
        )
        result = approval_service.decide_approval(
            pending["approval"]["id"], decision="approve_rule"
        )
        self.assertEqual(result["rule"]["scope"], "workspace:ws_1")

    def test_a_desktop_only_request_ignores_rules_entirely(self):
        # `decide_policy_with_rules` short-circuits before any lookup when the
        # built-in policy says desktop_only, so a standing rule can never
        # unlock one — the card offers no standing option for the same reason.
        self.rules.create_rule(
            scope="agent:agent_a", operation="tunnel.start", effect="allow"
        )
        pending = self._ask(
            run_id="run_first",
            details=self._details("agent_a"),
            operation="tunnel.start",
        )
        self.assertTrue(pending["approval_required"])
        self.assertTrue(pending["policy"]["desktop_only"])
        self.assertNotIn("rule", pending["policy"])

    def test_the_global_project_sentinel_is_refused_on_both_sides(self):
        # The write side has always refused to create `project:__global__`.
        # The lookup side used to still match one, so a rule written there by
        # hand — the dashboard's free-text scope box takes any string — applied
        # to every project-less run on the server. Closing that is what makes
        # the sentinel actually mean "no project".
        self.rules.create_rule(
            scope="project:__global__", operation="file.write", effect="allow"
        )
        pending = self._ask(run_id="run_first", details=self._details("agent_a"))
        self.assertTrue(pending["approval_required"])

    def test_a_named_project_rule_is_unaffected(self):
        self.rules.create_rule(
            scope="project:code_bridge", operation="file.write", effect="allow"
        )
        allowed = self._ask(
            run_id="run_first",
            details={"project_name": "code_bridge", "tool_name": "Write"},
        )
        self.assertFalse(allowed["approval_required"])
        self.assertTrue(allowed["allowed"])


if __name__ == "__main__":
    unittest.main()
