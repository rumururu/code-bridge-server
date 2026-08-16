"""An MCP server is started for a run only when the agent actually needs it.

C6-2 gave `ClaudeAgentOptions.mcp_servers` whatever an agent's `tools_json`
named. That was one step too generous: `configurator._ensure_playwright_tool`
puts a `playwright` entry into any draft whose *wording* looked web-ish
("click", "form", "url", "웹"), so on a machine where a `playwright` MCP server
is configured, an agent that never opens a browser — a two-minute scheduled
one, say — spawned `npx @playwright/mcp@latest` on every single llm turn.

The rule these tests pin down: a server reaches the session when the agent
declared it itself, or when a workflow step runs on it. A builder-added entry
with no step behind it does not, and the run event says so with its own reason
rather than letting a configured server look missing.
"""

from __future__ import annotations

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
from agent.agent_models import AgentDraft  # noqa: E402
from agent.configurator import (  # noqa: E402
    BUILDER_ADDED_TOOL_TEMPLATES,
    _ensure_app_action_tool,
    _ensure_playwright_tool,
    is_builder_added_tool,
)
from agent.task_orchestrator import (  # noqa: E402
    _STEP_TYPES_REQUIRING_MCP_ID,
    _apply_declared_mcp_servers,
    _flow_need_for_mcp_id,
    _is_app_action_workflow_type,
    MCP_TOOLS_EVENT_TYPE,
)
from core import database  # noqa: E402


def _builder_added_entry(mcp_id: str) -> dict:
    """The stored `tools_json` shape of a tool the builder added by itself."""
    ensure = _ensure_playwright_tool if mcp_id == "playwright" else _ensure_app_action_tool
    tool = ensure(AgentDraft()).tools[0]
    entry = tool.model_dump(mode="json")
    assert entry["mcp_id"] == mcp_id
    # Round-trip through JSON, because that is what the database hands back.
    return json.loads(json.dumps(entry))


class BuilderAddedToolMarkerTest(unittest.TestCase):
    """What the builder writes must stay recognisable as what the builder writes."""

    def test_the_recogniser_matches_what_the_injectors_produce(self):
        for mcp_id in BUILDER_ADDED_TOOL_TEMPLATES:
            with self.subTest(mcp_id=mcp_id):
                self.assertTrue(is_builder_added_tool(_builder_added_entry(mcp_id)))

    def test_a_model_written_entry_for_the_same_server_is_not_builder_added(self):
        self.assertFalse(
            is_builder_added_tool(
                {
                    "mcp_id": "playwright",
                    "tool_names": ["browser_navigate"],
                    "user_capability": "우리 회사 관리자 페이지에 로그인해 주문을 확인한다.",
                    "user_examples": ["관리자 페이지 로그인"],
                }
            )
        )

    def test_an_entry_from_an_older_template_degrades_to_model_declared(self):
        """Unrecognised means "the model declared it", i.e. keep injecting.

        The opposite default would take a capability away from an agent we
        cannot prove was auto-fitted, on nothing but a template edit.
        """
        stale = _builder_added_entry("playwright")
        stale["user_capability"] = "웹사이트를 연다."
        self.assertFalse(is_builder_added_tool(stale))

    def test_nonsense_entries_are_not_builder_added(self):
        for entry in (None, "playwright", {}, {"mcp_id": "gmail"}):
            with self.subTest(entry=entry):
                self.assertFalse(is_builder_added_tool(entry))


class FlowNeedDetectionTest(unittest.TestCase):
    def test_a_browser_step_needs_the_browser_server(self):
        need = _flow_need_for_mcp_id(
            [{"id": "open", "type": "browser_action", "name": "Open the page"}],
            "playwright",
        )
        self.assertIsNotNone(need)
        self.assertIn("Open the page", need)

    def test_an_llm_steps_tool_hint_is_not_evidence(self):
        """`_add_playwright_hints` writes that hint from keywords alone."""
        self.assertIsNone(
            _flow_need_for_mcp_id(
                [{"id": "think", "type": "llm", "name": "Think", "tool_hint": "playwright"}],
                "playwright",
            )
        )

    def test_an_mcp_tool_step_naming_the_server_needs_it(self):
        for hint in ("gmail",):
            with self.subTest(hint=hint):
                self.assertIsNotNone(
                    _flow_need_for_mcp_id(
                        [{"id": "send", "type": "mcp_tool", "name": "Send", "tool_hint": hint}],
                        "gmail",
                    )
                )

    def test_an_mcp_tool_steps_runtime_alias_resolves(self):
        for hint in ("playwright", "browser"):
            with self.subTest(hint=hint):
                self.assertIsNotNone(
                    _flow_need_for_mcp_id(
                        [{"id": "call", "type": "mcp_tool", "name": "Call", "tool_hint": hint}],
                        "playwright",
                    )
                )

    def test_the_legacy_step_type_key_is_read_too(self):
        self.assertIsNotNone(
            _flow_need_for_mcp_id(
                [{"id": "open", "step_type": "browser_action", "name": "Open"}], "playwright"
            )
        )

    def test_the_app_step_types_agree_with_the_dispatcher(self):
        """One list of app step types, not two that can drift apart."""
        for step_type in _STEP_TYPES_REQUIRING_MCP_ID["app_action"]:
            with self.subTest(step_type=step_type):
                self.assertTrue(_is_app_action_workflow_type(step_type))

    def test_an_empty_flow_proves_no_need(self):
        self.assertIsNone(_flow_need_for_mcp_id([], "playwright"))


class _RecordingSession:
    provider_id = "anthropic"
    has_pending_permission = False

    def __init__(self) -> None:
        self.received: dict | None = None

    async def set_mcp_servers(self, servers):
        self.received = servers


class _NoMcpSession:
    provider_id = "openai"
    has_pending_permission = False


class InjectionNeedTest(unittest.IsolatedAsyncioTestCase):
    """The decision, end to end through `_apply_declared_mcp_servers`."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "mcp_need.db"
        agent_store._agent_store = None
        browser_session_store._browser_session_store = None
        database.init_db()
        self.store = agent_store.get_agent_store()

        self.home = Path(self._tmp.name) / "home"
        self.home.mkdir()
        self.project = Path(self._tmp.name) / "project"
        self.project.mkdir()
        (self.home / ".claude.json").write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "playwright": {
                            "command": "npx",
                            "args": ["@playwright/mcp@latest"],
                        },
                        "gmail": {"command": "/bin/gmail_mcp"},
                    }
                }
            ),
            encoding="utf-8",
        )
        self.enterContext(
            patch.object(capability_registry.Path, "home", staticmethod(lambda: self.home))
        )
        self.enterContext(
            patch.object(capability_registry.Path, "cwd", staticmethod(lambda: self.project))
        )

    def tearDown(self):
        agent_store._agent_store = None
        browser_session_store._browser_session_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def _run_for(self, *, tools_json, flow_json):
        agent = self.store.create_agent(
            name="scheduled bot",
            system_prompt="Do the two-minute job.",
            provider_id="anthropic",
            tools_json=tools_json,
            flow_json=flow_json,
        )
        task = self.store.create_task(
            title="Two minute job", assigned_agent_id=agent["id"], goal="Do it."
        )
        run = self.store.create_run(task_id=task["id"], agent_id=agent["id"])
        return agent, task, run

    async def _apply(self, *, tools_json, flow_json, session=None, provider_id="anthropic"):
        agent, task, run = self._run_for(tools_json=tools_json, flow_json=flow_json)
        session = session if session is not None else _RecordingSession()
        await _apply_declared_mcp_servers(
            session,
            agent_id=agent["id"],
            run_id=run["id"],
            task_id=task["id"],
            provider_id=provider_id,
        )
        events = [
            event
            for event in self.store.list_events(run["id"])
            if event["event_type"] == MCP_TOOLS_EVENT_TYPE
        ]
        self.assertEqual(len(events), 1, "expected exactly one MCP tools event")
        return session, events[0]["app_event"]

    async def test_a_browser_step_still_gets_the_browser_server(self):
        session, payload = await self._apply(
            tools_json=[_builder_added_entry("playwright")],
            flow_json=[{"id": "open", "type": "browser_action", "name": "Open the page"}],
        )

        self.assertEqual(payload["injected"], ["playwright"])
        self.assertEqual(payload["not_injected"], [])
        self.assertEqual(sorted(session.received or {}), ["playwright"])

    async def test_a_flow_of_llm_steps_does_not_spawn_the_browser(self):
        """The reported defect: web-ish wording, no browser, npx every turn."""
        session, payload = await self._apply(
            tools_json=[_builder_added_entry("playwright")],
            flow_json=[
                {"id": "think", "type": "llm", "name": "Check the queue", "tool_hint": "playwright"},
                {"id": "tell", "type": "notify", "name": "Report", "notify": {"title": "done"}},
            ],
        )

        self.assertIsNone(session.received, "a browser MCP server was started anyway")
        self.assertEqual(payload["injected"], [])
        # Declared, so it is still reported — and reported as present but
        # unused, not as something that could not be found.
        self.assertEqual(payload["declared"], ["playwright"])
        self.assertEqual(payload["missing"], [])
        self.assertEqual(
            [(entry["mcp_id"], entry["reason"]) for entry in payload["not_injected"]],
            [("playwright", "not_required_by_agent")],
        )
        self.assertIn("no workflow step uses it", payload["message"])

    async def test_an_mcp_tool_step_naming_the_server_still_injects_it(self):
        session, payload = await self._apply(
            tools_json=[_builder_added_entry("playwright")],
            flow_json=[
                {"id": "call", "type": "mcp_tool", "name": "Call it", "tool_hint": "playwright"}
            ],
        )

        self.assertEqual(payload["injected"], ["playwright"])
        self.assertEqual(sorted(session.received or {}), ["playwright"])

    async def test_a_stored_agent_with_no_marker_keeps_its_server(self):
        """Legacy degradation: an entry we cannot prove was auto-fitted injects."""
        session, payload = await self._apply(
            tools_json=[
                {
                    "mcp_id": "playwright",
                    "tool_names": ["browser_navigate"],
                    "user_capability": "우리 회사 관리자 페이지를 연다.",
                }
            ],
            flow_json=[{"id": "think", "type": "llm", "name": "Think"}],
        )

        self.assertEqual(payload["injected"], ["playwright"])
        self.assertEqual(sorted(session.received or {}), ["playwright"])

    async def test_a_model_declared_server_is_never_gated(self):
        session, payload = await self._apply(
            tools_json=[{"mcp_id": "gmail", "tool_names": ["send"]}],
            flow_json=[{"id": "think", "type": "llm", "name": "Think"}],
        )

        self.assertEqual(payload["injected"], ["gmail"])
        self.assertEqual(sorted(session.received or {}), ["gmail"])

    async def test_a_builtin_runtime_is_still_named_as_such(self):
        """`app_action` runs here, spawns nothing, and must not read as withheld."""
        session, payload = await self._apply(
            tools_json=[_builder_added_entry("app_action")],
            flow_json=[{"id": "think", "type": "llm", "name": "Think"}],
        )

        self.assertEqual(payload["builtin_runtime"], ["app_action"])
        self.assertEqual(payload["not_injected"], [])
        self.assertIsNone(session.received)

    async def test_the_non_anthropic_path_is_unchanged(self):
        _session, payload = await self._apply(
            tools_json=[{"mcp_id": "gmail", "tool_names": ["send"]}],
            flow_json=[{"id": "think", "type": "llm", "name": "Think"}],
            session=_NoMcpSession(),
            provider_id="openai",
        )

        self.assertIs(payload["provider_supports_mcp_injection"], False)
        self.assertEqual(payload["injected"], [])
        self.assertEqual(
            [(entry["mcp_id"], entry["reason"]) for entry in payload["not_injected"]],
            [("gmail", "session_cannot_take_mcp_servers")],
        )
        self.assertIn("openai", payload["message"])


if __name__ == "__main__":
    unittest.main()
