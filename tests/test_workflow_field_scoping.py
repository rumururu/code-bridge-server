"""A step editor cannot be trusted to show only fields that do something.

Before this, `normalize_workflow_step` accepted every field on every step
type: a `shell` step could carry `memory_write` or a `notify` block, an `llm`
step could carry `device_id`. None of it did anything at execution time —
`shell_step_executor` only ever looks at `script_id`/`script_args`, the
notify executor only fires for `notify` steps, device adapters only read
`device_id` for device/app/browser steps. The authoring surfaces (dashboard,
phone, Configurator) had no way to know this, so they showed the field
anyway, a user filled it in believing it would change what the step does, and
the step silently did the same thing it would have done with the field left
blank. That is worse than the field not existing: it is a lie the UI told
without meaning to.

Scoping fields per type here means a mismatched field is refused at
authoring time, with a message that says which type rejected it and why it
doesn't belong — instead of being accepted, stored, and quietly ignored
forever. The one deliberate exception is `tool_hint` / `success_criteria` /
`actions`: the pre-scoping normalizer wrote all three onto every step
regardless of type, so real agents saved before this change — including a
live scheduled agent in this project's own database — carry them on types
that never read them. `normalize_workflow` re-normalizes an agent's stored
`flow_json` from scratch on every run rather than persisting the normalized
shape back, so rejecting those three outside their scoped types would not
refuse a bad new draft — it would make an already-working agent stop running,
silently, the next time its schedule fires. The tests below pin both halves:
genuinely foreign fields are refused, and the fields real stored agents
already carry keep normalizing.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent.workflow_v2 import (  # noqa: E402
    WorkflowNormalizationError,
    normalize_workflow,
    normalize_workflow_step,
)


class PerTypeFieldsAreAcceptedTest(unittest.TestCase):
    """Each type's own fields normalize without complaint."""

    def test_shell_accepts_script_id_and_args(self) -> None:
        step = normalize_workflow_step(
            {
                "id": "run",
                "type": "shell",
                "name": "Run cycle",
                "script_id": "script_1",
                "script_args": ["a", "b"],
            },
            index=1,
        )
        self.assertEqual(step["script_id"], "script_1")
        self.assertEqual(step["script_args"], ["a", "b"])

    def test_llm_accepts_instruction_observation_memory_and_success_criteria(self) -> None:
        step = normalize_workflow_step(
            {
                "id": "diagnose",
                "type": "llm",
                "name": "Diagnose",
                "instruction": "Work out why the cycle failed.",
                "observation": "Read the shell step's stderr first.",
                "memory_read": "prior failures on this device",
                "memory_write": "what blocked this run",
                "success_criteria": "Root cause named",
                "tool_hint": "openai",
            },
            index=1,
        )
        self.assertEqual(step["instruction"], "Work out why the cycle failed.")
        self.assertEqual(step["memory_read"], "prior failures on this device")
        self.assertEqual(step["memory_write"], "what blocked this run")
        self.assertEqual(step["success_criteria"], "Root cause named")

    def test_notify_accepts_the_notify_block(self) -> None:
        step = normalize_workflow_step(
            {
                "id": "tell",
                "type": "notify",
                "name": "Tell the user",
                "notify": {"title": "Disk almost full", "level": "warning"},
            },
            index=1,
        )
        self.assertEqual(step["notify"]["title"], "Disk almost full")

    def test_browser_action_accepts_actions_and_tool_hint(self) -> None:
        step = normalize_workflow_step(
            {
                "id": "open",
                "type": "browser_action",
                "name": "Open page",
                "actions": [{"type": "navigate", "url": "https://example.test"}],
                "tool_hint": "playwright",
            },
            index=1,
        )
        self.assertEqual(step["actions"][0]["type"], "navigate")

    def test_app_action_accepts_device_fields(self) -> None:
        step = normalize_workflow_step(
            {
                "id": "read",
                "type": "app_action",
                "name": "Read screen",
                "android_device_id": "emulator-5554",
                "actions": [{"type": "read_screen"}],
            },
            index=1,
        )
        self.assertEqual(step["android_device_id"], "emulator-5554")

    def test_mcp_tool_accepts_tool_hint(self) -> None:
        step = normalize_workflow_step(
            {"id": "call", "type": "mcp_tool", "name": "Call tool", "tool_hint": "filesystem"},
            index=1,
        )
        self.assertEqual(step["tool_hint"], "filesystem")

    def test_approval_gate_and_manual_handoff_and_condition_need_no_extra_field(self) -> None:
        for step_type in ("approval_gate", "manual_handoff", "condition"):
            step = normalize_workflow_step(
                {"id": f"gate_{step_type}", "type": step_type, "name": "Gate"},
                index=1,
            )
            self.assertEqual(step["type"], step_type)


class ForeignFieldsAreRejectedTest(unittest.TestCase):
    """A field that belongs to a different type is refused, not stored."""

    def test_shell_rejects_memory_write(self) -> None:
        # The example from the spec this change closes: a shell step has no
        # judgement to record, so memory_write on it can only ever be a lie.
        with self.assertRaisesRegex(
            WorkflowNormalizationError,
            r"'memory_write' is not a field of a shell step",
        ):
            normalize_workflow_step(
                {
                    "id": "run",
                    "type": "shell",
                    "name": "Run",
                    "script_id": "script_1",
                    "memory_write": "what the shell step learned",
                },
                index=2,
            )

    def test_shell_rejects_notify_block(self) -> None:
        with self.assertRaisesRegex(
            WorkflowNormalizationError,
            r"'notify' is not a field of a shell step",
        ):
            normalize_workflow_step(
                {
                    "id": "run",
                    "type": "shell",
                    "name": "Run",
                    "script_id": "script_1",
                    "notify": {"title": "Done"},
                },
                index=1,
            )

    def test_llm_rejects_script_id(self) -> None:
        with self.assertRaisesRegex(
            WorkflowNormalizationError,
            r"'script_id' is not a field of a llm step",
        ):
            normalize_workflow_step(
                {"id": "think", "type": "llm", "name": "Think", "script_id": "script_1"},
                index=1,
            )

    def test_llm_rejects_device_id(self) -> None:
        with self.assertRaisesRegex(
            WorkflowNormalizationError,
            r"'device_id' is not a field of a llm step",
        ):
            normalize_workflow_step(
                {"id": "think", "type": "llm", "name": "Think", "device_id": "emulator-5554"},
                index=1,
            )

    def test_llm_rejects_notify_block(self) -> None:
        with self.assertRaisesRegex(
            WorkflowNormalizationError,
            r"'notify' is not a field of a llm step",
        ):
            normalize_workflow_step(
                {
                    "id": "think",
                    "type": "llm",
                    "name": "Think",
                    "notify": {"title": "x"},
                },
                index=1,
            )

    def test_notify_rejects_memory_read(self) -> None:
        with self.assertRaisesRegex(
            WorkflowNormalizationError,
            r"'memory_read' is not a field of a notify step",
        ):
            normalize_workflow_step(
                {
                    "id": "tell",
                    "type": "notify",
                    "name": "Tell",
                    "notify": {"title": "x"},
                    "memory_read": "recent runs",
                },
                index=1,
            )

    def test_browser_action_rejects_script_args(self) -> None:
        with self.assertRaisesRegex(
            WorkflowNormalizationError,
            r"'script_args' is not a field of a browser_action step",
        ):
            normalize_workflow_step(
                {
                    "id": "open",
                    "type": "browser_action",
                    "name": "Open",
                    "actions": [],
                    "script_args": ["x"],
                },
                index=1,
            )

    def test_approval_gate_rejects_instruction(self) -> None:
        with self.assertRaisesRegex(
            WorkflowNormalizationError,
            r"'instruction' is not a field of a approval_gate step",
        ):
            normalize_workflow_step(
                {
                    "id": "gate",
                    "type": "approval_gate",
                    "name": "Gate",
                    "instruction": "Approve this",
                },
                index=1,
            )

    def test_error_message_names_the_step_index(self) -> None:
        with self.assertRaisesRegex(WorkflowNormalizationError, r"^step 4:"):
            normalize_workflow_step(
                {"id": "think", "type": "llm", "name": "Think", "device_id": "x"},
                index=4,
            )


class SuccessCriteriaScopingTest(unittest.TestCase):
    """success_criteria is scoped to llm going forward (spec item 1.3).

    It reaches only the prompt an llm step opens
    (task_orchestrator._workflow_step_message); on a shell step the exit code
    decides pass/fail and overrules whatever this field says, silently. It is
    accepted, not rejected, on every *other* type too — the legacy-compat
    exception below — because real stored agents (see the module docstring)
    already carry it outside llm and this normalizer runs fresh on every task
    execution, not once at authoring time.
    """

    def test_accepted_on_llm(self) -> None:
        step = normalize_workflow_step(
            {
                "id": "diagnose",
                "type": "llm",
                "name": "Diagnose",
                "success_criteria": "Root cause is identified",
            },
            index=1,
        )
        self.assertEqual(step["success_criteria"], "Root cause is identified")

    def test_accepted_but_inert_on_shell(self) -> None:
        # Not rejected — see _UNRESTRICTED_LEGACY_FIELDS — but the shell
        # executor (agent.shell_step_executor) never reads this key, so it
        # does not change what the step does, only what is recorded on it.
        step = normalize_workflow_step(
            {
                "id": "cycle",
                "type": "shell",
                "name": "Run cycle",
                "script_id": "script_1",
                "success_criteria": "사이클 정상 종료",
            },
            index=1,
        )
        self.assertEqual(step["success_criteria"], "사이클 정상 종료")


class RealWorldAgentShapeStillNormalizesTest(unittest.TestCase):
    """A fixture shaped like the agents already stored in this project's DB.

    Read from ``~/.code-bridge/core/code_bridge.db`` (read-only) while
    building this change: a live, currently-scheduled agent
    (`agent_9f6af9fdeed24da8804ca8eb5d3368cc`) has exactly this two-step
    shape — a `shell` step with `success_criteria` and `tool_hint: null`
    stamped by the pre-scoping normalizer, escalating on failure to an `llm`
    diagnosis step. This is not the DB itself (per the task's read-only
    instruction), but a fixture built to match it, so a regression here would
    mean this project's own agents stop running.
    """

    def _legacy_flow(self) -> list[dict]:
        return [
            {
                "id": "cycle",
                "type": "shell",
                "name": "N960N 사이클",
                "script_id": "script_6655441dfd704e01968c992003925fdf",
                "script_args": ["24c2b6bcef0d7ece", "com.example.app"],
                "description": "스몰뎁 교환 사이클 1회",
                "success_criteria": "사이클 정상 종료",
                "on_failure": {"type": "goto_step", "target_step_id": "diagnose"},
                "tool_hint": None,
                "actions": [],
                "on_success": {"type": "end"},
            },
            {
                "id": "diagnose",
                "type": "llm",
                "name": "실패 진단",
                "description": "직전 사이클 스텝의 exit code와 로그를 읽고 보고한다.",
                "tool_hint": None,
                "success_criteria": "",
                "on_failure": {"type": "ask_user", "resume": "same_step"},
                "actions": [],
                "on_success": {"type": "continue"},
            },
        ]

    def test_the_legacy_shape_normalizes_without_raising(self) -> None:
        steps = normalize_workflow(self._legacy_flow())
        self.assertEqual(steps[0]["type"], "shell")
        self.assertEqual(steps[0]["script_id"], "script_6655441dfd704e01968c992003925fdf")
        self.assertEqual(steps[0]["success_criteria"], "사이클 정상 종료")
        self.assertEqual(steps[1]["type"], "llm")
        self.assertEqual(steps[1]["on_failure"], {"type": "ask_user", "resume": "same_step"})

    def test_normalizing_twice_is_stable(self) -> None:
        # normalize_workflow runs fresh from stored flow_json on every task
        # execution (agent.task_orchestrator._workflow_steps_for_task), so a
        # scheduled agent normalizes the same raw JSON over and over. It must
        # not degrade or start failing on the second pass.
        once = normalize_workflow(self._legacy_flow())
        twice = normalize_workflow(self._legacy_flow())
        self.assertEqual(once, twice)


if __name__ == "__main__":
    unittest.main()
