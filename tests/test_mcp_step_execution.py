"""An `mcp_tool` step must run, and must not pretend to when it cannot.

Every `mcp_tool` step parked unconditionally: the Configurator could author
one, the runtime handed it to a person, and a scheduled agent containing one
never ran unattended at all. The capability was already there — the agent's
declared MCP servers are injected into the turn by
`_apply_declared_mcp_servers` — so only the dispatch was missing.

Parking is still correct for the cases where running would be a lie, and those
are what these tests pin: a step that never says which server it needs, a
server the agent never declared, and a server that is not configured on this
machine. All three are things a person fixes; none is something a model can
work around, and letting the turn proceed would let it satisfy the instruction
by some other route and report success on a server it never touched.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import agent.task_orchestrator as orchestrator  # noqa: E402


def _step(tool_hint=None, workflow_type="mcp_tool"):
    step_input = {"workflow_type": workflow_type}
    if tool_hint is not None:
        step_input["tool_hint"] = tool_hint
    return {"id": "step_1", "title": "Send the summary", "input": step_input}


def _agent(*mcp_ids):
    return {"id": "agent_1", "tools_json": [{"mcp_id": value} for value in mcp_ids]}


class McpStepGatingTest(unittest.TestCase):
    def _blocker(self, step, agent, *, configured=("n8n",)):
        with patch.object(
            orchestrator,
            "detected_mcp_server_configs",
            return_value={name: {"type": "stdio"} for name in configured},
        ):
            return orchestrator._mcp_step_blocker(step, agent)

    def test_a_configured_declared_server_runs(self) -> None:
        self.assertIsNone(self._blocker(_step("n8n"), _agent("n8n")))

    def test_the_server_name_is_matched_case_insensitively(self) -> None:
        self.assertIsNone(self._blocker(_step("N8N"), _agent("n8n")))

    def test_a_step_with_no_server_asks(self) -> None:
        blocker = self._blocker(_step(None), _agent("n8n"))
        self.assertIsNotNone(blocker)
        self.assertIn("tool_hint", blocker)

    def test_a_server_the_agent_never_declared_asks(self) -> None:
        """Otherwise the turn runs on whatever the CLI happens to have, which
        is the silent mismatch the declaration machinery exists to end."""
        blocker = self._blocker(_step("slack"), _agent("n8n"))
        self.assertIsNotNone(blocker)
        self.assertIn("slack", blocker)

    def test_a_server_not_installed_on_this_machine_asks(self) -> None:
        blocker = self._blocker(_step("n8n"), _agent("n8n"), configured=())
        self.assertIsNotNone(blocker)
        self.assertIn("n8n", blocker)

    def test_an_agent_with_no_declarations_still_runs_a_configured_server(self) -> None:
        """Agents stored before tools_json was populated must not all break;
        the machine's own configuration is still checked."""
        self.assertIsNone(self._blocker(_step("n8n"), _agent()))

    def test_an_unreadable_config_asks_rather_than_running_blind(self) -> None:
        with patch.object(
            orchestrator,
            "detected_mcp_server_configs",
            side_effect=RuntimeError("config unreadable"),
        ):
            blocker = orchestrator._mcp_step_blocker(_step("n8n"), _agent("n8n"))
        self.assertIsNotNone(blocker)


class TheTurnIsToldToUseThatServerTest(unittest.TestCase):
    def _message(self, step):
        return orchestrator._workflow_step_message(step, "Run the workflow.")

    def test_an_mcp_step_names_the_server_and_forbids_a_detour(self) -> None:
        message = self._message(_step("n8n"))
        self.assertIn("mcp_tool step", message)
        self.assertIn("'n8n'", message)
        self.assertIn("fail the step", message)

    def test_an_ordinary_step_gets_no_such_instruction(self) -> None:
        message = self._message(_step("n8n", workflow_type="llm"))
        self.assertNotIn("mcp_tool step", message)


class AnMcpStepCanSayWhatToDoTest(unittest.TestCase):
    """The deeper reason none of these had ever run.

    `mcp_tool` allowed exactly one field, `tool_hint`. It could name a server
    and had nowhere to put the task, so a flow containing a usable step was
    rejected outright — "'instruction' is not a field of a mcp_tool step" —
    and the run silently fell back to a generic plan that ignored the workflow
    entirely. Measured: an agent whose flow was two mcp_tool steps ran four
    steps named "Prepare task context" / "Execute provider turn" instead.
    """

    def test_a_usable_step_survives_normalization(self) -> None:
        from agent.workflow_v2 import normalize_workflow

        steps = normalize_workflow(
            [
                {
                    "id": "call_it",
                    "type": "mcp_tool",
                    "name": "Run the n8n workflow",
                    "tool_hint": "n8n",
                    "instruction": "Trigger the daily digest workflow.",
                    "success_criteria": "The workflow reports success.",
                }
            ]
        )
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["tool_hint"], "n8n")
        self.assertEqual(steps[0]["instruction"], "Trigger the daily digest workflow.")

    def test_the_published_schema_offers_those_fields(self) -> None:
        """Authoring surfaces render from the published schema; a field the
        normalizer accepts but no client can enter is not reachable."""
        from agent.workflow_v2 import WORKFLOW_STEP_SCHEMA

        self.assertIn("instruction", WORKFLOW_STEP_SCHEMA["mcp_tool"])
        self.assertIn("tool_hint", WORKFLOW_STEP_SCHEMA["mcp_tool"])


class OtherHandoffTypesStillParkTest(unittest.TestCase):
    """Splitting `mcp_tool` out of the shared branch must not change the two
    step types that are *supposed* to stop for a person."""

    def test_the_dispatch_still_parks_manual_handoff_and_approval_gate(self) -> None:
        source = (SERVER_DIR / "agent" / "task_orchestrator.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('workflow_type in {"manual_handoff", "approval_gate"}', source)
        self.assertNotIn(
            'workflow_type in {"manual_handoff", "mcp_tool", "approval_gate"}',
            source,
            "an mcp_tool step must no longer park unconditionally",
        )


if __name__ == "__main__":
    unittest.main()
