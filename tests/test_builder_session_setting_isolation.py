"""The toggle promised a question that a settings file was answering for it.

`builder.allow_mcp_tools` is documented as "On does not grant anything by
itself — each call is still put to the person, one at a time", and
`_apply_mcp_servers` deliberately leaves `allowed_tools` empty so the round
trip in `_on_can_use_tool` is the only way a tool runs.

None of that survives `~/.claude/settings.json`. `setting_sources=None` means
"load every settings file the CLI would", and a `permissions.allow` entry
there — `"mcp__playwright"` — pre-approves the tool *inside the CLI*, before
any of this code runs. The callback is never consulted, no control request is
published, no permission card is shown, and the audit trail records nothing.

Measured on this machine, not theorised: with that entry present and
`defaultMode: auto`, a Configurator turn given the Playwright server drove a
browser through four navigations to an external site with nobody asked. The
browser profile's own history is the evidence; the audit table holds only the
turning-on of the toggle.

So the builder session refuses filesystem settings outright. It belongs to no
project, and a person's editor configuration is not an answer to "may this
server act on my behalf".
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from llm.claude_session import ClaudeSession  # noqa: E402
from routes.agents import _isolate_builder_session  # noqa: E402


class _Options:
    """Only the fields these paths touch."""

    def __init__(self) -> None:
        self.setting_sources = None
        self.mcp_servers = None
        self.allowed_tools: list[str] = []


class SettingIsolationTests(unittest.TestCase):
    def _session(self) -> ClaudeSession:
        return ClaudeSession(project_path=str(SERVER_DIR))

    def test_a_plain_session_still_reads_the_users_settings(self) -> None:
        """Unchanged for chat: this is a builder-only decision."""

        options = _Options()
        self._session()._apply_setting_sources(options)

        self.assertIsNone(options.setting_sources)

    def test_an_isolated_session_loads_no_settings_file(self) -> None:
        session = self._session()
        session.isolate_filesystem_settings = True
        options = _Options()

        session._apply_setting_sources(options)

        # `[]` is the SDK's isolation mode; `None` would mean "load them all".
        self.assertEqual(options.setting_sources, [])

    def test_allowed_tools_is_still_left_empty(self) -> None:
        """The other half of the same promise, guarded next to it."""

        session = self._session()
        session.mcp_servers = {"playwright": {"type": "stdio", "command": "npx"}}
        options = _Options()

        session._apply_mcp_servers(options)

        self.assertEqual(options.mcp_servers, session.mcp_servers)
        self.assertEqual(
            options.allowed_tools,
            [],
            "listing a tool here pre-approves it and skips the round trip",
        )


class BuilderTurnIsolatesItsSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_builder_turns_it_on(self) -> None:
        session = ClaudeSession(project_path=str(SERVER_DIR))

        await _isolate_builder_session(session)

        self.assertTrue(session.isolate_filesystem_settings)

    async def test_it_does_not_depend_on_the_mcp_toggle(self) -> None:
        """A builder session takes no permissions from disk either way.

        Tying this to `builder.allow_mcp_tools` would leave the hole open for
        every tool the CLI allows that is not an MCP tool.
        """

        from agent import builder_tool_policy

        session = ClaudeSession(project_path=str(SERVER_DIR))
        was = builder_tool_policy.builder_mcp_enabled()
        try:
            builder_tool_policy.set_builder_mcp_enabled(False)
            await _isolate_builder_session(session)
        finally:
            builder_tool_policy.set_builder_mcp_enabled(was)

        self.assertTrue(session.isolate_filesystem_settings)

    async def test_a_provider_without_the_notion_is_left_alone(self) -> None:
        class _Codex:
            pass

        codex = _Codex()
        await _isolate_builder_session(codex)  # must not raise

        self.assertFalse(hasattr(codex, "isolate_filesystem_settings"))

    async def test_changing_it_closes_a_live_client(self) -> None:
        """Options are built once per connection, as with `set_mcp_servers`."""

        closed: list[bool] = []

        class _Live(ClaudeSession):
            @property
            def is_running(self) -> bool:
                return True

            async def close(self) -> None:
                closed.append(True)

        session = _Live(project_path=str(SERVER_DIR))
        await session.set_setting_isolation(True)
        self.assertEqual(closed, [True])

        # Same value again is not a change, so the turn is not interrupted.
        await session.set_setting_isolation(True)
        self.assertEqual(closed, [True])


if __name__ == "__main__":
    unittest.main()
