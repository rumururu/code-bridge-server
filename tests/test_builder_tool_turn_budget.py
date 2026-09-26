"""A turn that drives a browser died on a budget sized for a turn that writes.

240s was measured against a Configurator turn that reads the draft, thinks,
and answers. Once MCP tools are attached the same turn also loads pages, waits
for editors to render, and runs scripts inside them — and that time is spent
against the model's clock, because the stream gives one gap covering both the
tool call and the thinking after it.

Measured: asked to find out why a form fill failed on a Naver post editor, the
Configurator made thirteen tool calls and died at 240s with the answer
unwritten. The permission waits were not the cause — those are already added
back in `_collect_llm_response_text`, and each was answered in about two
seconds. The page loads were.

So the budget widens when the tools are actually attached, and only then. The
rule about clients still holds and is the reason this is not simply set to a
large number everywhere: every client's give-up point must *outlast* the
server's, or the client abandons a job the server is still honouring and the
finished answer is dropped. See `BUILDER_POLL_LIMIT` (dashboard) and
`_converseJobPollAttempts` (app), both raised alongside this.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from routes.agents import (  # noqa: E402
    BUILDER_CONVERSE_JOB_TIMEOUT_SECONDS,
    BUILDER_TOOL_TURN_TIMEOUT_SECONDS,
    builder_turn_timeout_seconds,
)

REPO_ROOT = SERVER_DIR.parent


class BudgetSelectionTests(unittest.TestCase):
    def test_a_turn_with_tools_gets_the_wider_budget(self) -> None:
        with patch("agent.builder_tool_policy.builder_mcp_enabled", return_value=True):
            self.assertEqual(
                builder_turn_timeout_seconds(), BUILDER_TOOL_TURN_TIMEOUT_SECONDS
            )

    def test_a_plain_design_turn_keeps_the_tighter_one(self) -> None:
        """Not widened for everyone: a wedged provider is still cut off."""

        with patch("agent.builder_tool_policy.builder_mcp_enabled", return_value=False):
            self.assertEqual(
                builder_turn_timeout_seconds(), BUILDER_CONVERSE_JOB_TIMEOUT_SECONDS
            )

    def test_the_wider_budget_is_actually_wider(self) -> None:
        self.assertGreater(
            BUILDER_TOOL_TURN_TIMEOUT_SECONDS, BUILDER_CONVERSE_JOB_TIMEOUT_SECONDS
        )


class ClientsOutlastTheServerTests(unittest.TestCase):
    """The rule the constants' own comments state, checked against the clients.

    Both clients poll; the job keeps running after a client stops asking. A
    client that gives up first throws away a reply the server went on to
    produce — which is exactly what happened at 120s against a 120s ceiling
    before any of these numbers were written down.
    """

    def test_the_dashboard_waits_longer_than_the_widest_turn(self) -> None:
        html = (SERVER_DIR / "dashboard" / "templates" / "agents.html").read_text("utf-8")
        limit = int(re.search(r"BUILDER_POLL_LIMIT = (\d+)", html).group(1))
        interval_ms = int(re.search(r"BUILDER_POLL_MS = (\d+)", html).group(1))

        self.assertGreater(
            limit * interval_ms / 1000,
            BUILDER_TOOL_TURN_TIMEOUT_SECONDS,
            "the page gives up while the server is still working",
        )

    def test_the_app_waits_longer_than_the_widest_turn(self) -> None:
        dart = (REPO_ROOT / "lib" / "providers" / "builder_provider.dart")
        if not dart.is_file():  # pragma: no cover - server-only checkout
            self.skipTest("app sources are not present in this checkout")
        text = dart.read_text("utf-8")
        attempts = int(
            re.search(r"_converseJobPollAttempts = (\d+)", text).group(1)
        )

        # The app polls once a second (`Duration(seconds: 1)` in the loop).
        self.assertGreater(
            attempts,
            BUILDER_TOOL_TURN_TIMEOUT_SECONDS,
            "the app gives up while the server is still working",
        )


class UnreadableSettingTests(unittest.TestCase):
    """The lookup is new on this path and can fail before anything else runs.

    `builder_turn_timeout_seconds` is the first thing the job path asks for —
    ahead of the turn, the session, and everything that would otherwise have
    opened the settings database. Two suites reached it with no `app_settings`
    table and the whole job failed, having previously never touched storage at
    all because they patch `run_configurator_turn` wholesale.

    Falling back is right, but only in one direction: not knowing whether the
    tools are on must never buy the wider budget, or a wedged provider is held
    open for ten minutes on the strength of a failed lookup.
    """

    def test_it_falls_back_to_the_tighter_budget(self) -> None:
        with patch(
            "agent.builder_tool_policy.builder_mcp_enabled",
            side_effect=RuntimeError("no such table: app_settings"),
        ):
            self.assertEqual(
                builder_turn_timeout_seconds(), BUILDER_CONVERSE_JOB_TIMEOUT_SECONDS
            )


if __name__ == "__main__":
    unittest.main()
