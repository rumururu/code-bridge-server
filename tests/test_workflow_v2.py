"""Workflow v2 normalization tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent.workflow_v2 import (  # noqa: E402
    WorkflowNormalizationError,
    normalize_failure_policy,
    normalize_workflow,
)


class WorkflowV2NormalizationTest(unittest.TestCase):
    def test_legacy_failure_strings_normalize_to_structured_policies(self) -> None:
        self.assertEqual(
            normalize_failure_policy("ask_user"),
            {"type": "ask_user", "resume": "same_step"},
        )
        self.assertEqual(normalize_failure_policy("abort"), {"type": "abort"})
        self.assertEqual(
            normalize_failure_policy("retry_once"),
            {"type": "retry", "max_attempts": 1, "then": {"type": "abort"}},
        )

    def test_v2_failure_policy_objects_are_normalized(self) -> None:
        self.assertEqual(
            normalize_failure_policy(
                {
                    "type": "retry",
                    "max_retries": "2",
                    "then": {"type": "ask_user", "prompt": "Log in, then continue"},
                }
            ),
            {
                "type": "retry",
                "max_attempts": 2,
                "then": {
                    "type": "ask_user",
                    "resume": "same_step",
                    "prompt": "Log in, then continue",
                },
            },
        )

    def test_legacy_steps_get_ids_default_type_and_structured_failure(self) -> None:
        workflow = normalize_workflow(
            [
                {
                    "name": "Open cafe",
                    "description": "Check login state.",
                    "tool_hint": "playwright",
                    "success_criteria": "Cafe is visible",
                    "on_failure": "ask_user",
                },
                {
                    "id": "submit",
                    "name": "Submit",
                    "on_failure": "retry_once",
                },
            ]
        )

        self.assertEqual(workflow[0]["id"], "step_1")
        self.assertEqual(workflow[0]["type"], "llm")
        self.assertEqual(workflow[0]["on_failure"]["type"], "ask_user")
        self.assertEqual(workflow[1]["id"], "submit")
        self.assertEqual(workflow[1]["on_failure"]["type"], "retry")

    def test_v2_browser_actions_are_normalized(self) -> None:
        workflow = normalize_workflow(
            [
                {
                    "id": "login_check",
                    "type": "browser_action",
                    "name": "Login check",
                    "actions": [
                        {"type": "Navigate", "url": "https://example.test"},
                        {"type": "ASSERT", "kind": "text_visible", "value": "Cafe"},
                    ],
                }
            ]
        )

        self.assertEqual(workflow[0]["type"], "browser_action")
        self.assertEqual(
            [action["type"] for action in workflow[0]["actions"]],
            ["navigate", "assert"],
        )

    def test_goto_failure_policy_validates_target_step(self) -> None:
        workflow = normalize_workflow(
            [
                {
                    "id": "login_check",
                    "name": "Login check",
                    "on_failure": {"type": "goto", "target": "login_check"},
                }
            ]
        )

        self.assertEqual(
            workflow[0]["on_failure"],
            {"type": "goto_step", "target_step_id": "login_check"},
        )

    def test_unknown_goto_target_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            WorkflowNormalizationError,
            "unknown goto_step target: missing",
        ):
            normalize_workflow(
                [
                    {
                        "id": "login_check",
                        "name": "Login check",
                        "on_failure": {"type": "goto_step", "step_id": "missing"},
                    }
                ]
            )

    def test_duplicate_step_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(WorkflowNormalizationError, "duplicate step id"):
            normalize_workflow(
                [
                    {"id": "same", "name": "One"},
                    {"id": "same", "name": "Two"},
                ]
            )

    def test_unknown_step_and_action_types_are_rejected(self) -> None:
        with self.assertRaisesRegex(WorkflowNormalizationError, "unknown step type"):
            # "shell" used to be the example here; it is a real step type now.
            normalize_workflow([{"type": "teleport", "name": "Run"}])

        with self.assertRaisesRegex(WorkflowNormalizationError, "unknown action type"):
            normalize_workflow(
                [
                    {
                        "type": "browser_action",
                        "name": "Browser",
                        "actions": [{"type": "hover"}],
                    }
                ]
            )


if __name__ == "__main__":
    unittest.main()
