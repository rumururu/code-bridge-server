"""A declared MCP tool must be a real one, and a real one must reach the run.

Two halves of the same defect. The builder would put `mcp_id: "gmail"` into a
draft on a machine with no gmail server anywhere, commit it to
`agents.tools_json`, and show it in the app as something the agent can do —
and the runtime never read `tools_json` at all, so even the servers that *did*
exist were never handed to the Claude session. Declared and real were two
disconnected sets, and nothing reported the gap in either direction.

So: the builder drops what it cannot verify and says so in the same reply
(C6-1); the orchestrator passes the verified ones to
`ClaudeAgentOptions.mcp_servers` and writes a run event naming everything it
could not pass (C6-2). The tests below assert on the constructed options and
the recorded events — no Claude session is ever started.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store, browser_session_store  # noqa: E402
from agent import capability_registry  # noqa: E402
from agent import task_orchestrator  # noqa: E402
from agent.agent_models import AgentDraft, AgentToolDraft  # noqa: E402
from agent.capability_registry import (  # noqa: E402
    detected_mcp_server_configs,
    verify_declared_mcp_ids,
)
from agent.configurator import (  # noqa: E402
    _drop_unverifiable_tools,
    create_builder_session,
)
from agent.task_orchestrator import (  # noqa: E402
    MCP_TOOLS_EVENT_TYPE,
    _apply_declared_mcp_servers,
    execute_task_orchestration,
    prepare_task_orchestration,
)
from core import database  # noqa: E402


class _TmpMcpConfig:
    """A temporary home/cwd pair holding whatever MCP config a test needs."""

    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name) / "home"
        self.home.mkdir()
        self.project = Path(self._tmp.name) / "project"
        self.project.mkdir()

    def write(self, servers: dict) -> None:
        (self.home / ".claude.json").write_text(
            json.dumps({"mcpServers": servers}), encoding="utf-8"
        )

    def patched(self):
        return (
            patch.object(capability_registry.Path, "home", staticmethod(lambda: self.home)),
            patch.object(capability_registry.Path, "cwd", staticmethod(lambda: self.project)),
        )

    def cleanup(self) -> None:
        self._tmp.cleanup()


class _McpConfigTestCase(unittest.TestCase):
    def setUp(self):
        self.config = _TmpMcpConfig()
        for context in self.config.patched():
            self.enterContext(context)

    def tearDown(self):
        self.config.cleanup()


class VerifyDeclaredMcpIdsTest(_McpConfigTestCase):
    """What counts as "this tool exists on this machine"."""

    def test_a_server_the_config_does_not_declare_is_not_verified(self):
        self.config.write({"marionette": {"command": "/bin/marionette_mcp"}})

        verdicts = {v.mcp_id: v for v in verify_declared_mcp_ids(["marionette", "gmail"])}

        self.assertTrue(verdicts["marionette"].verified)
        self.assertEqual(verdicts["marionette"].source, "mcp_config")
        self.assertFalse(verdicts["gmail"].verified)
        self.assertEqual(verdicts["gmail"].source, "")
        # The reason names what was looked for and where, so the sentence the
        # user reads is actionable rather than "unavailable".
        self.assertIn("gmail", verdicts["gmail"].detail)
        self.assertIn(str(self.config.home / ".claude.json"), verdicts["gmail"].detail)

    def test_builtin_runtimes_are_verified_without_any_mcp_config(self):
        """`app_action` / `browser` are executors of this server, not servers."""
        verdicts = {
            v.mcp_id: v for v in verify_declared_mcp_ids(["app_action", "browser"])
        }

        for mcp_id in ("app_action", "browser"):
            self.assertTrue(verdicts[mcp_id].verified, mcp_id)
            self.assertEqual(verdicts[mcp_id].source, "builtin_runtime", mcp_id)

    def test_a_configured_server_wins_over_the_builtin_alias(self):
        """A real `playwright` server must not be masked by the built-in name."""
        self.config.write({"playwright": {"command": "npx", "args": ["-y", "@playwright/mcp@latest"]}})

        (verdict,) = verify_declared_mcp_ids(["playwright"])

        self.assertEqual(verdict.source, "mcp_config")

    def test_without_a_configured_server_playwright_falls_back_to_the_builtin(self):
        (verdict,) = verify_declared_mcp_ids(["playwright"])

        self.assertTrue(verdict.verified)
        self.assertEqual(verdict.source, "builtin_runtime")

    def test_blanks_and_duplicates_are_collapsed(self):
        verdicts = verify_declared_mcp_ids(["app_action", " app_action ", "", "  "])

        self.assertEqual([v.mcp_id for v in verdicts], ["app_action"])

    def test_an_unreadable_config_verifies_nothing_rather_than_everything(self):
        (self.config.home / ".claude.json").write_text("{broken", encoding="utf-8")

        (verdict,) = verify_declared_mcp_ids(["marionette"])

        self.assertFalse(verdict.verified)


class DetectedMcpServerConfigsTest(_McpConfigTestCase):
    """The launch configs handed to the SDK come from the file, verbatim."""

    def test_stdio_entry_keeps_command_args_and_env(self):
        self.config.write(
            {
                "marionette": {
                    "type": "stdio",
                    "command": "/bin/marionette_mcp",
                    "args": ["--quiet"],
                    "env": {"TOKEN": "abc"},
                }
            }
        )

        configs = detected_mcp_server_configs()

        self.assertEqual(
            configs["marionette"],
            {
                "type": "stdio",
                "command": "/bin/marionette_mcp",
                "args": ["--quiet"],
                "env": {"TOKEN": "abc"},
            },
        )

    def test_http_entry_keeps_url_and_headers(self):
        self.config.write(
            {"remote": {"type": "http", "url": "https://mcp.example", "headers": {"A": "b"}}}
        )

        self.assertEqual(
            detected_mcp_server_configs()["remote"],
            {"type": "http", "url": "https://mcp.example", "headers": {"A": "b"}},
        )

    def test_unlaunchable_entries_are_omitted_not_guessed(self):
        self.config.write(
            {
                "no_command": {"type": "stdio"},
                "no_url": {"type": "http"},
                "in_process": {"type": "sdk", "name": "x"},
                "fine": {"command": "/bin/ok"},
            }
        )

        self.assertEqual(set(detected_mcp_server_configs()), {"fine"})


class BuilderDropsUnverifiableToolsTest(_McpConfigTestCase):
    """C6-1: a draft must not carry a tool this machine cannot run."""

    @staticmethod
    def _tool(mcp_id: str) -> AgentToolDraft:
        return AgentToolDraft(mcp_id=mcp_id, tool_names=["do_thing"])

    def test_an_undetected_server_is_removed_from_the_draft(self):
        self.config.write({"marionette": {"command": "/bin/marionette_mcp"}})
        draft = AgentDraft(
            name="Mailer",
            tools=[self._tool("gmail"), self._tool("marionette")],
        )

        next_draft, dropped = _drop_unverifiable_tools(draft)

        self.assertEqual([tool.mcp_id for tool in next_draft.tools], ["marionette"])
        self.assertEqual([verdict.mcp_id for verdict in dropped], ["gmail"])

    def test_builtin_runtime_tools_survive_with_no_mcp_config_at_all(self):
        """The regression this must not cause: dropping the server's own runtimes."""
        draft = AgentDraft(
            name="Phone bot", tools=[self._tool("app_action"), self._tool("playwright")]
        )

        next_draft, dropped = _drop_unverifiable_tools(draft)

        self.assertEqual(
            [tool.mcp_id for tool in next_draft.tools], ["app_action", "playwright"]
        )
        self.assertEqual(dropped, [])

    def test_a_draft_with_nothing_to_drop_is_returned_unchanged(self):
        draft = AgentDraft(name="Phone bot", tools=[self._tool("app_action")])

        next_draft, dropped = _drop_unverifiable_tools(draft)

        self.assertIs(next_draft, draft)
        self.assertEqual(dropped, [])

    def test_the_turn_tells_the_user_which_tool_it_removed(self):
        """Dropping silently would be the same defect wearing a different coat."""
        self.config.write({"marionette": {"command": "/bin/marionette_mcp"}})
        session = create_builder_session(system_prompt="build an agent")

        parsed = session.apply_llm_response(
            """Here is the agent.

```draft
{
  "name": "Inbox Triage",
  "description": "Sorts mail",
  "system_prompt": "You triage email.",
  "provider_id": "anthropic",
  "tools": [{"mcp_id": "gmail", "tool_names": ["search_threads"]}],
  "flow": [
    {"id": "read", "type": "llm", "name": "Read mail",
     "description": "Read the newest threads.", "success_criteria": "Threads read."}
  ],
  "memory_seeds": []
}
```
""",
            user_message="이메일 정리해주는 에이전트 만들어줘",
        )

        self.assertEqual(
            [tool.mcp_id for tool in session.current_draft.tools],
            [],
            "an MCP server that exists nowhere stayed in the committed draft",
        )
        self.assertIn("'gmail'", parsed.assistant_message)
        self.assertIn("MCP 서버가 등록되어 있지 않아", parsed.assistant_message)

    def test_the_user_is_told_once_not_on_every_later_turn(self):
        self.config.write({})
        session = create_builder_session(system_prompt="build an agent")
        first = session.apply_llm_response(
            """```draft
{"name": "A", "tools": [{"mcp_id": "gmail", "tool_names": ["x"]}], "flow": [
  {"id": "s", "type": "llm", "name": "Step", "description": "d", "success_criteria": "c"}]}
```
""",
            user_message="메일 에이전트",
        )
        self.assertIn("'gmail'", first.assistant_message)

        second = session.apply_llm_response(
            "이름만 바꿀게요.", user_message="이름 바꿔줘"
        )

        self.assertNotIn("'gmail'", second.assistant_message)


class ClaudeSessionMcpOptionsTest(unittest.TestCase):
    """C6-2, session half: the servers land on the options object."""

    def _session(self, **kwargs):
        from llm.claude_session import ClaudeSession  # noqa: PLC0415

        return ClaudeSession(project_path=tempfile.gettempdir(), **kwargs)

    def test_the_sdk_option_this_relies_on_exists(self):
        """Pinned deliberately: the whole feature is one SDK field."""
        import dataclasses  # noqa: PLC0415

        from claude_agent_sdk import ClaudeAgentOptions  # noqa: PLC0415

        names = {f.name for f in dataclasses.fields(ClaudeAgentOptions)}
        self.assertIn("mcp_servers", names)

    def test_declared_servers_reach_the_options_object(self):
        servers = {"marionette": {"type": "stdio", "command": "/bin/marionette_mcp"}}
        options = self._session(mcp_servers=servers)._build_options()

        self.assertEqual(options.mcp_servers, servers)

    def test_no_declaration_leaves_the_cli_configuration_alone(self):
        options = self._session()._build_options()

        self.assertFalse(options.mcp_servers)

    def test_allowed_tools_is_not_widened_by_injection(self):
        """Listing mcp__* there would pre-approve calls past the approval round-trip."""
        options = self._session(
            mcp_servers={"marionette": {"type": "stdio", "command": "/bin/m"}}
        )._build_options()

        self.assertEqual(options.allowed_tools, [])

    def test_setting_new_servers_closes_a_live_session_so_options_rebuild(self):
        session = self._session()
        closed: list[bool] = []

        async def fake_close():
            closed.append(True)

        session._client = object()  # is_running
        session.close = fake_close  # type: ignore[method-assign]

        asyncio.run(session.set_mcp_servers({"a": {"type": "stdio", "command": "/bin/a"}}))
        self.assertEqual(closed, [True])

        # Same set again: nothing changed, so nothing is torn down.
        asyncio.run(session.set_mcp_servers({"a": {"type": "stdio", "command": "/bin/a"}}))
        self.assertEqual(closed, [True])


class _RecordingSession:
    """Stands in for a provider session; records what was pushed into it."""

    provider_id = "anthropic"
    has_pending_permission = False

    def __init__(self) -> None:
        self.received: dict | None = None

    async def set_mcp_servers(self, servers):
        self.received = servers


class _NoMcpSession:
    """A provider session with no MCP option at all (codex / gemini)."""

    provider_id = "openai"
    has_pending_permission = False


class OrchestratorMcpInjectionTest(unittest.IsolatedAsyncioTestCase):
    """C6-2, run half: injection happens, and every gap is written to the run."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "declared_mcp.db"
        agent_store._agent_store = None
        browser_session_store._browser_session_store = None
        database.init_db()
        self.store = agent_store.get_agent_store()

        self.config = _TmpMcpConfig()
        for context in self.config.patched():
            self.enterContext(context)

    def tearDown(self):
        agent_store._agent_store = None
        browser_session_store._browser_session_store = None
        database.DB_PATH = self._original_db_path
        self.config.cleanup()
        self._tmp.cleanup()

    def _agent_task_run(self, tools_json):
        agent = self.store.create_agent(
            name="tooled bot",
            system_prompt="Use the declared tools.",
            provider_id="anthropic",
            tools_json=tools_json,
            flow_json=[{"id": "think", "type": "llm", "name": "Think"}],
        )
        task = self.store.create_task(
            title="Do the thing",
            assigned_agent_id=agent["id"],
            goal="Do the thing.",
        )
        run = self.store.create_run(task_id=task["id"], agent_id=agent["id"])
        return agent, task, run

    def _mcp_event(self, run_id):
        events = [
            event
            for event in self.store.list_events(run_id)
            if event["event_type"] == MCP_TOOLS_EVENT_TYPE
        ]
        self.assertEqual(len(events), 1, "expected exactly one MCP tools event")
        return events[0]["app_event"]

    async def test_a_detected_server_reaches_the_session(self):
        self.config.write(
            {"marionette": {"command": "/bin/marionette_mcp", "env": {"TOKEN": "s3cret"}}}
        )
        agent, task, run = self._agent_task_run([{"mcp_id": "marionette", "tool_names": ["tap"]}])
        session = _RecordingSession()

        await _apply_declared_mcp_servers(
            session,
            agent_id=agent["id"],
            run_id=run["id"],
            task_id=task["id"],
            provider_id="anthropic",
            step_id="think",
        )

        self.assertEqual(
            session.received,
            {"marionette": {"type": "stdio", "command": "/bin/marionette_mcp", "env": {"TOKEN": "s3cret"}}},
        )
        payload = self._mcp_event(run["id"])
        self.assertEqual(payload["injected"], ["marionette"])
        self.assertEqual(payload["missing"], [])
        self.assertNotIn("s3cret", json.dumps(payload), "credentials leaked into the run log")

    async def test_a_declared_server_that_is_not_there_is_reported_not_dropped(self):
        self.config.write({"marionette": {"command": "/bin/marionette_mcp"}})
        agent, task, run = self._agent_task_run(
            [{"mcp_id": "marionette", "tool_names": ["tap"]}, {"mcp_id": "gmail", "tool_names": ["send"]}]
        )
        session = _RecordingSession()

        await _apply_declared_mcp_servers(
            session,
            agent_id=agent["id"],
            run_id=run["id"],
            task_id=task["id"],
            provider_id="anthropic",
        )

        payload = self._mcp_event(run["id"])
        self.assertEqual(payload["injected"], ["marionette"])
        self.assertEqual([entry["mcp_id"] for entry in payload["missing"]], ["gmail"])
        self.assertIn("gmail", payload["message"])
        self.assertEqual(session.received, {"marionette": {"type": "stdio", "command": "/bin/marionette_mcp"}})

    async def test_a_builtin_runtime_is_named_as_such_and_not_injected(self):
        agent, task, run = self._agent_task_run([{"mcp_id": "app_action", "tool_names": ["launch_app"]}])
        session = _RecordingSession()

        await _apply_declared_mcp_servers(
            session,
            agent_id=agent["id"],
            run_id=run["id"],
            task_id=task["id"],
            provider_id="anthropic",
        )

        payload = self._mcp_event(run["id"])
        self.assertEqual(payload["builtin_runtime"], ["app_action"])
        self.assertEqual(payload["injected"], [])
        self.assertIsNone(session.received)

    async def test_a_provider_that_cannot_take_mcp_servers_says_so(self):
        self.config.write({"marionette": {"command": "/bin/marionette_mcp"}})
        agent, task, run = self._agent_task_run([{"mcp_id": "marionette", "tool_names": ["tap"]}])

        await _apply_declared_mcp_servers(
            _NoMcpSession(),
            agent_id=agent["id"],
            run_id=run["id"],
            task_id=task["id"],
            provider_id="openai",
        )

        payload = self._mcp_event(run["id"])
        self.assertIs(payload["provider_supports_mcp_injection"], False)
        self.assertEqual(payload["injected"], [])
        self.assertEqual([entry["mcp_id"] for entry in payload["not_injected"]], ["marionette"])
        self.assertIn("openai", payload["message"])

    async def test_an_agent_with_no_declared_tools_records_nothing(self):
        agent, task, run = self._agent_task_run([])
        session = _RecordingSession()

        await _apply_declared_mcp_servers(
            session,
            agent_id=agent["id"],
            run_id=run["id"],
            task_id=task["id"],
            provider_id="anthropic",
        )

        self.assertEqual(
            [e for e in self.store.list_events(run["id"]) if e["event_type"] == MCP_TOOLS_EVENT_TYPE],
            [],
        )
        self.assertIsNone(session.received)

    async def test_the_real_workflow_step_path_injects_before_the_turn(self):
        """End to end through `_execute_llm_workflow_step`, no Claude started."""
        self.config.write({"marionette": {"command": "/bin/marionette_mcp"}})
        agent = self.store.create_agent(
            name="tooled bot",
            system_prompt="Use the declared tools.",
            provider_id="anthropic",
            tools_json=[
                {"mcp_id": "marionette", "tool_names": ["tap"]},
                {"mcp_id": "gmail", "tool_names": ["send"]},
            ],
            flow_json=[{"id": "think", "type": "llm", "name": "Think"}],
        )
        task = self.store.create_task(
            title="Run workflow", assigned_agent_id=agent["id"], goal="Finish it."
        )
        prepared = prepare_task_orchestration(
            task["id"], provider_id="anthropic", auto_start=False
        )
        assert prepared is not None
        session = _RecordingSession()
        order: list[str] = []

        async def fake_create_chat_session(**_kwargs):
            return session

        async def fake_stream(_sink, _session, **_kwargs):
            order.append(f"turn:{sorted(session.received or {})}")
            return True

        with patch.object(task_orchestrator, "create_chat_session", fake_create_chat_session), patch.object(
            task_orchestrator, "stream_claude_turn", fake_stream
        ):
            await execute_task_orchestration(prepared["execution"])

        # The servers were already on the session by the time the turn ran.
        self.assertEqual(order, ["turn:['marionette']"])
        payload = self._mcp_event(prepared["run"]["id"])
        self.assertEqual(payload["injected"], ["marionette"])
        self.assertEqual([entry["mcp_id"] for entry in payload["missing"]], ["gmail"])
        # Recorded against the step, so this is the workflow executor's path
        # and not the workflow-less fallback.
        self.assertEqual(
            payload["step_id"], self.store.list_task_steps(task["id"])[0]["id"]
        )


    async def test_a_workflow_less_run_injects_too(self):
        """An agent without a flow runs its tools on the same promise."""
        self.config.write({"marionette": {"command": "/bin/marionette_mcp"}})
        agent = self.store.create_agent(
            name="planner bot",
            system_prompt="Plan from the goal.",
            provider_id="anthropic",
            tools_json=[{"mcp_id": "marionette", "tool_names": ["tap"]}],
            flow_json=[],
        )
        task = self.store.create_task(
            title="Plan it", assigned_agent_id=agent["id"], goal="Plan it."
        )
        prepared = prepare_task_orchestration(
            task["id"], provider_id="anthropic", auto_start=False
        )
        assert prepared is not None
        session = _RecordingSession()

        async def fake_create_chat_session(**_kwargs):
            return session

        async def fake_stream(_sink, _session, **_kwargs):
            return True

        with patch.object(task_orchestrator, "create_chat_session", fake_create_chat_session), patch.object(
            task_orchestrator, "stream_claude_turn", fake_stream
        ):
            await execute_task_orchestration(prepared["execution"])

        self.assertEqual(
            session.received, {"marionette": {"type": "stdio", "command": "/bin/marionette_mcp"}}
        )
        payload = self._mcp_event(prepared["run"]["id"])
        self.assertEqual(payload["injected"], ["marionette"])
        self.assertIsNone(payload["step_id"], "the workflow-less path has no step id")


if __name__ == "__main__":
    unittest.main()
