"""What the app tells an author about a condition step must be true.

The app's validator accepted six different fields as "the condition" —
`instruction`, `description`, `observation`, `condition`, `condition_text`,
`expression` — and its message named an expression first. The server accepts
exactly one of them.

That mismatch is not a rejected save. `normalize_workflow` raises, and
`_workflow_steps_for_task` catches the exception and returns zero steps, so an
agent built on the app's advice fires on schedule, runs nothing, increments its
fire count, and reports no error. Nothing anywhere says why.

Nothing stored uses `condition` today, so this was latent rather than live —
but the advice was reachable, and this test is what keeps the two sides from
drifting apart again once track H gives the type real fields.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from code_bridge_core.workflow_v2 import (  # noqa: E402
    COMMON_STEP_FIELDS,
    WORKFLOW_STEP_SCHEMA,
    WorkflowNormalizationError,
    normalize_workflow,
)

APP_VALIDATOR = (
    Path(SERVER_DIR).parent / "lib" / "models" / "workflow_step_validation.dart"
)


def _condition_step(**extra):
    return {"id": "c1", "type": "condition", "name": "Decide", **extra}


class WhatTheServerAcceptsTest(unittest.TestCase):
    def test_description_is_the_only_prose_a_condition_carries(self) -> None:
        # Track H (T-H-04) gave the type its one real field: `branches`, the
        # structured routing table. It is deliberately not prose — the reason
        # the app used to offer `expression` was that there was nowhere
        # structured to put a condition, and an expression string would have
        # meant three clients each writing a parser for it
        # (TICKETS_H_RUNNER_BRANCHING.md §1). So `description` is still the
        # only place an author writes a sentence about what the step decides,
        # and it is still a common field rather than a condition-specific one.
        self.assertEqual(WORKFLOW_STEP_SCHEMA["condition"], frozenset({"branches"}))
        self.assertIn("description", COMMON_STEP_FIELDS)

    def test_a_description_normalizes(self) -> None:
        steps = normalize_workflow([_condition_step(description="count > 10")])
        self.assertEqual(steps[0]["description"], "count > 10")

    def test_the_fields_the_app_used_to_advertise_are_refused(self) -> None:
        for field in ("instruction", "observation", "expression", "condition_text"):
            with self.subTest(field=field):
                with self.assertRaises(WorkflowNormalizationError):
                    normalize_workflow([_condition_step(**{field: "x"})])


class TheAppSaysTheSameThingTest(unittest.TestCase):
    """Reads the Dart validator as text — a Flutter test cannot check the
    Python contract, and this is the boundary where the two meet."""

    def setUp(self) -> None:
        if not APP_VALIDATOR.is_file():
            self.skipTest(f"app validator not present at {APP_VALIDATOR}")
        source = APP_VALIDATOR.read_text(encoding="utf-8")
        start = source.index("_validateConditionStep")
        self.body = source[start : source.index("\n  }", start)]

    def test_it_does_not_offer_a_field_the_server_refuses(self) -> None:
        for refused in ("instruction", "observation", "expression", "condition_text"):
            with self.subTest(field=refused):
                self.assertNotIn(
                    f"'{refused}'",
                    self.body,
                    f"the app treats {refused} as a valid condition, but "
                    "normalize_workflow refuses it and the whole workflow "
                    "silently becomes zero steps",
                )
        self.assertNotIn("step.instruction", self.body)
        self.assertNotIn("step.observation", self.body)

    def test_it_asks_for_the_field_the_server_accepts(self) -> None:
        self.assertIn("step.description", self.body)


if __name__ == "__main__":
    unittest.main()
