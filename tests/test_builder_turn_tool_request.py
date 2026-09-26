"""A tool call in the Agent Builder used to stall the whole turn.

`ClaudeSession` gates every tool call through `can_use_tool`: it publishes a
`control_request` event and then *parks on a future* until something answers
(`llm/claude_session.py`). The chat surface answers it — it shows the user a
permission card over its websocket and calls
`approve_pending_permissions_and_retry` / `deny_pending_permissions`.

The Agent Builder never did. `_collect_llm_response_text` reads `assistant`,
`result` and `error` and ignores everything else, so a `control_request` fell
on the floor and the parked callback was never settled. The stream had no more
events to give, the turn produced nothing, and the only thing that ended it was
the job timeout — 240 seconds of silence for what should be an immediate
"no".

The Configurator runs on the `claude_code` preset, so it *has* tools it can try
to call. This is not hypothetical: the moment it decides to check something
with Bash — read a project file, probe a page — the design conversation stops
dead.

These tests pin both halves: the turn ends promptly, and it ends by telling the
model why, so the answer it produces is written knowing the tool was refused
rather than trailing off mid-thought.
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from routes.agents import _collect_llm_response_text  # noqa: E402


class _SessionThatAsksForATool:
    """A session that requests a tool and can only continue once answered.

    Mirrors `ClaudeSession`: `send_message` yields the `control_request` and
    then has nothing further, exactly as the real stream does while its
    `can_use_tool` callback is parked. The rest of the turn is only reachable
    through `deny_pending_permissions`.
    """

    def __init__(self) -> None:
        self.denied_with: str | None = None
        self.deny_calls = 0

    async def send_message(self, _prompt: str):
        yield {
            "type": "control_request",
            "request_id": "req_1",
            "request": {
                "subtype": "can_use_tool",
                "tool_name": "Bash",
                "input": {"command": "python -m playwright ..."},
                "tool_use_id": "tu_1",
            },
        }
        # Nothing else. The real session is blocked on its future here.

    async def deny_pending_permissions(self, message: str = ""):
        self.deny_calls += 1
        self.denied_with = message
        yield {"type": "assistant", "message": {"content": "I cannot check that here."}}
        yield {"type": "result", "result": "I cannot check that here."}


class ATurnEndsEvenWhenTheModelReachesForAToolTest(unittest.TestCase):
    def test_the_turn_produces_an_answer_instead_of_stalling(self):
        session = _SessionThatAsksForATool()
        text = asyncio.run(
            asyncio.wait_for(_collect_llm_response_text(session, "design me an agent", timeout=30), timeout=5)
        )
        self.assertEqual(text, "I cannot check that here.")

    def test_the_request_is_answered_exactly_once(self):
        # Settling twice would race the SDK's own callback; leaving it
        # unsettled is the stall this file exists for.
        session = _SessionThatAsksForATool()
        asyncio.run(asyncio.wait_for(_collect_llm_response_text(session, "x", timeout=30), timeout=5))
        self.assertEqual(session.deny_calls, 1)

    def test_the_refusal_says_why_and_what_to_do_instead(self):
        """The model has to be able to act on the refusal.

        A bare "denied" leaves it guessing whether to retry, and a retry loops
        straight back into another refusal. The message names the surface and
        points at the two things that *do* work: ask the person, or put the
        check in the agent's own steps, where a browser step runs it at
        runtime with a human able to answer.
        """
        session = _SessionThatAsksForATool()
        asyncio.run(asyncio.wait_for(_collect_llm_response_text(session, "x", timeout=30), timeout=5))
        message = session.denied_with or ""
        self.assertIn("Agent Builder", message)
        for expected in ("ask", "step"):
            self.assertIn(expected, message.lower())
        # Never encourage a retry: the next attempt is refused identically.
        self.assertNotIn("try again", message.lower())


class AnOrdinaryTurnIsUnaffectedTest(unittest.TestCase):
    class _PlainSession:
        async def send_message(self, _prompt: str):
            yield {"type": "assistant", "message": {"content": "part one. "}}
            yield {"type": "result", "result": "the whole answer"}

    def test_a_turn_with_no_tool_call_behaves_exactly_as_before(self):
        text = asyncio.run(
            _collect_llm_response_text(self._PlainSession(), "hello", timeout=30)
        )
        self.assertEqual(text, "the whole answer")


if __name__ == "__main__":
    unittest.main()


class _SessionAskingForAnMcpTool(_SessionThatAsksForATool):
    """The same shape, but the tool is one a configured MCP server provides."""

    def __init__(self) -> None:
        super().__init__()
        self.approved = 0

    async def send_message(self, _prompt: str):
        yield {
            "type": "control_request",
            "request_id": "req_1",
            "request": {
                "subtype": "can_use_tool",
                "tool_name": "mcp__playwright__browser_navigate",
                "input": {"url": "https://cafe.naver.com/x"},
                "tool_use_id": "tu_1",
            },
        }

    async def approve_pending_permissions_and_retry(self):
        self.approved += 1
        yield {"type": "result", "result": "I looked, and the field is #subject."}


class TheBoundaryIsDrawnByPrefixNotBySettingTest(unittest.TestCase):
    """A setting must not be able to hand a background job a shell.

    `Bash`, `Write` and `Edit` are arbitrary code and arbitrary writes on the
    user's machine. The Agent Builder turn runs unattended-by-default, so
    those are refused whatever the setting says, and only `mcp__*` — servers
    the user configured, each for a stated purpose — is ever eligible to be
    asked about.
    """

    def test_a_shell_tool_is_refused_even_with_asking_switched_on(self):
        from agent import builder_tool_policy

        with patch.object(builder_tool_policy, "builder_mcp_enabled", lambda: True):
            self.assertFalse(builder_tool_policy.may_ask("Bash"))
            self.assertFalse(builder_tool_policy.may_ask("Write"))
            # ...and an MCP tool is, with the same setting.
            self.assertTrue(builder_tool_policy.may_ask("mcp__playwright__navigate"))

    def test_an_mcp_tool_is_refused_while_asking_is_switched_off(self):
        from agent import builder_tool_policy

        with patch.object(builder_tool_policy, "builder_mcp_enabled", lambda: False):
            self.assertFalse(builder_tool_policy.may_ask("mcp__playwright__navigate"))

    def test_the_two_refusals_say_different_things(self):
        # "never available here" and "switched off" send the reader to
        # different places; collapsing them would have someone hunting for a
        # setting that could not have helped.
        from agent import builder_tool_policy

        never = builder_tool_policy.refusal_reason("Bash")
        off = builder_tool_policy.refusal_reason("mcp__playwright__navigate")
        self.assertNotEqual(never, off)
        self.assertIn("never available", never)
        self.assertIn("switched off", off)


class AnEligibleCallIsPutToThePersonTest(unittest.TestCase):
    def _job(self):
        from routes.agents import BuilderConverseJob

        return BuilderConverseJob(id="job_1", session_id="s_1", user_message="hi")

    def _drive(self, *, answer: bool):
        """Run the turn and answer the permission request it parks on."""
        from agent import builder_tool_policy
        from routes.agents import _collect_llm_response_text

        session = _SessionAskingForAnMcpTool()
        job = self._job()

        async def scenario():
            turn = asyncio.ensure_future(
                _collect_llm_response_text(session, "x", timeout=30, job=job)
            )
            # Wait for the job to park, the way a polling client would see it.
            for _ in range(200):
                if job.status == "waiting_for_permission":
                    break
                await asyncio.sleep(0.01)
            parked = job.permission
            assert parked is not None, "the job never asked"
            parked.decision.set_result(answer)
            return await asyncio.wait_for(turn, timeout=5), job, session, parked

        with patch.object(builder_tool_policy, "builder_mcp_enabled", lambda: True), \
             patch("routes.agents.record_api_action"):
            return asyncio.run(scenario())

    def test_the_job_parks_and_the_request_rides_the_poll(self):
        from routes.agents import _job_payload

        _text, job, _session, parked = self._drive(answer=True)
        # The view a client polls is what carries the question out.
        payload = _job_payload(job)
        self.assertEqual(parked.tool_name, "mcp__playwright__browser_navigate")
        self.assertEqual(parked.tool_input, {"url": "https://cafe.naver.com/x"})
        # Settled by now, so the poll no longer shows it.
        self.assertNotIn("permission_request", payload)
        self.assertEqual(job.status, "running")

    def test_yes_runs_the_tool_and_the_turn_finishes(self):
        text, _job, session, _parked = self._drive(answer=True)
        self.assertEqual(session.approved, 1)
        self.assertEqual(session.deny_calls, 0)
        self.assertIn("#subject", text)

    def test_no_declines_and_says_it_was_the_person_who_declined(self):
        # Not the same message as the policy's: "you were asked and the answer
        # was no" must not read as "this is never available", or the model
        # goes looking for a setting to change.
        from routes.agents import BUILDER_TOOL_DECLINED

        _text, _job, session, _parked = self._drive(answer=False)
        self.assertEqual(session.approved, 0)
        self.assertEqual(session.deny_calls, 1)
        self.assertEqual(session.denied_with, BUILDER_TOOL_DECLINED)
        self.assertIn("declined", BUILDER_TOOL_DECLINED)


class TheModelsClockStopsWhileAPersonDecidesTest(unittest.TestCase):
    def test_a_slow_human_does_not_spend_the_models_budget(self):
        """A design conversation must not die because someone stepped away.

        The model gets `timeout` seconds of its *own*. Here the model needs a
        moment after approval and the person takes longer than the whole
        budget; if the wait counted, the turn would time out having done
        nothing wrong.
        """
        from agent import builder_tool_policy
        from routes.agents import BuilderConverseJob, _collect_llm_response_text

        class _SlowAfterApproval(_SessionAskingForAnMcpTool):
            async def approve_pending_permissions_and_retry(self):
                self.approved += 1
                await asyncio.sleep(0.15)
                yield {"type": "result", "result": "done"}

        session = _SlowAfterApproval()
        job = BuilderConverseJob(id="job_2", session_id="s_2", user_message="hi")

        async def scenario():
            turn = asyncio.ensure_future(
                _collect_llm_response_text(session, "x", timeout=0.4, job=job)
            )
            for _ in range(300):
                if job.status == "waiting_for_permission":
                    break
                await asyncio.sleep(0.01)
            # The person takes longer than the model's entire budget.
            await asyncio.sleep(0.5)
            job.permission.decision.set_result(True)
            return await asyncio.wait_for(turn, timeout=5)

        with patch.object(builder_tool_policy, "builder_mcp_enabled", lambda: True), \
             patch("routes.agents.record_api_action"):
            text = asyncio.run(scenario())
        self.assertEqual(text, "done")


class EveryDecisionIsAuditedTest(unittest.TestCase):
    def test_a_refusal_nobody_was_asked_about_is_recorded_too(self):
        """An audit that kept only approvals would make a conversation that
        quietly tried twenty tools look like one that tried none."""
        from routes.agents import _collect_llm_response_text

        session = _SessionThatAsksForATool()
        with patch("routes.agents.record_api_action") as audit:
            asyncio.run(
                asyncio.wait_for(
                    _collect_llm_response_text(session, "x", timeout=30), timeout=5
                )
            )
        self.assertEqual(audit.call_count, 1)
        details = audit.call_args.kwargs["details"]
        self.assertEqual(details["surface"], "agent_builder")
        self.assertEqual(details["tool_name"], "Bash")
        self.assertIs(details["asked_user"], False)
        self.assertIs(audit.call_args.kwargs["success"], False)
