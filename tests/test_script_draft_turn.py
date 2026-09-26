"""Script drafting had no test that actually ran it, and it broke.

`draft_script` calls the Agent Builder's turn reader,
`_collect_llm_response_text`. When that reader grew a *required* `timeout`
keyword — so the model's clock could be stopped while a person answers a
permission card — two of its three call sites were updated. This one was not,
because it is reached through a lazy import inside the function body and no
test ever executed it. The whole suite stayed green.

What the user saw instead: asking the Configurator for an agent that needed a
new script produced a proposal card reading

    Script drafting failed: _collect_llm_response_text() missing 1 required
    keyword-only argument: 'timeout'

`draft_script` catches every exception and reports it as a 502, so a plain
TypeError in the signature arrived looking like the model had refused.

These tests run the real function end to end against a fake session, so the
call site is exercised rather than assumed. The second one covers the reason
the keyword exists at all: the script writer runs on a tool-capable preset, so
it can try to call a tool mid-draft, and that has to be refused and the draft
still finished.
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from routes.scripts import ScriptDraftRequest, draft_script  # noqa: E402


class _PlainSession:
    """Answers in one turn, the way a well-behaved provider does."""

    def __init__(self, text: str) -> None:
        self._text = text

    async def send_message(self, _prompt: str):
        yield {"type": "result", "result": self._text}

    async def abort_current_turn(self) -> None:  # pragma: no cover - not reached
        raise AssertionError("a completed turn must not be aborted")


class _SessionThatReachesForATool:
    """Requests a tool, then continues only once the refusal settles it."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.denied_with: str | None = None

    async def send_message(self, _prompt: str):
        yield {
            "type": "control_request",
            "request_id": "req_1",
            "request": {
                "subtype": "can_use_tool",
                "tool_name": "Bash",
                "input": {"command": "ls /"},
                "tool_use_id": "tu_1",
            },
        }

    async def deny_pending_permissions(self, message: str = ""):
        self.denied_with = message
        yield {"type": "result", "result": self._text}


class ScriptDraftTurnTests(unittest.IsolatedAsyncioTestCase):
    async def _draft(self, session) -> dict:
        with (
            patch(
                "chat.chat_session_service.create_chat_session",
                return_value=session,
            ) as create,
            patch(
                "chat.chat_session_service.get_chat_provider_selection",
                return_value=None,
            ),
        ):
            async def _create(*args, **kwargs):
                return session

            create.side_effect = _create
            return await draft_script(
                ScriptDraftRequest(intent="check git status and run the tests")
            )

    async def test_a_draft_comes_back_whole(self) -> None:
        """The call site is real: a wrong signature here fails the test."""

        draft = await self._draft(
            _PlainSession("```bash\n#!/usr/bin/env bash\ngit status\n```")
        )

        self.assertEqual(draft["interpreter"], "bash")
        self.assertIn("git status", draft["body"])
        # The fences are the model's packaging, not part of the script.
        self.assertNotIn("```", draft["body"])

    async def test_a_tool_call_mid_draft_is_refused_and_the_draft_still_lands(
        self,
    ) -> None:
        """The reason `timeout` exists: this path answers tool calls.

        Before the reader learned to answer them the turn simply stalled here
        until the outer deadline, and the user got a timeout instead of a
        script.
        """

        session = _SessionThatReachesForATool(
            "```bash\n#!/usr/bin/env bash\npytest -q\n```"
        )

        draft = await asyncio.wait_for(self._draft(session), timeout=10)

        self.assertIn("pytest -q", draft["body"])
        self.assertIsNotNone(
            session.denied_with, "the parked tool call must be settled, not ignored"
        )


class EveryTurnReaderCallSiteTests(unittest.TestCase):
    """The same drift, guarded across the tree rather than one call site.

    `_collect_llm_response_text` is reached from three modules, one of them
    through a lazy import inside a function body. A required keyword can be
    added to the definition and land green while a caller three files away is
    still passing the old signature — that is exactly what happened. Running
    each path would be better, but this at least fails at the moment the
    signatures diverge instead of the moment a user asks for a script.
    """

    def test_no_caller_omits_the_model_budget(self) -> None:
        import ast

        offenders: list[str] = []
        for path in sorted(SERVER_DIR.rglob("*.py")):
            if "venv" in path.parts or path.parts[-2:-1] == ("tests",):
                continue
            try:
                tree = ast.parse(path.read_text("utf-8"), str(path))
            except (SyntaxError, UnicodeDecodeError):
                # Fixtures deliberately hold unparseable text; they hold no
                # call sites either.
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = getattr(func, "id", None) or getattr(func, "attr", None)
                if name != "_collect_llm_response_text":
                    continue
                if not any(kw.arg == "timeout" for kw in node.keywords):
                    offenders.append(
                        f"{path.relative_to(SERVER_DIR)}:{node.lineno}"
                    )

        self.assertEqual(
            offenders,
            [],
            "these callers read a turn with no deadline of their own: "
            + ", ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
