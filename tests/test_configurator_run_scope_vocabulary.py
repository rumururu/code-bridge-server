"""The prompt never said which `{{...}}` names a run can hold, so one was invented.

Every other authoring vocabulary in `configurator.py` is generated from the
executor — step fields, browser actions, app actions, branch operators — on the
stated grounds that a model cannot use what it was never told exists, and will
otherwise write the shape that reads naturally. The run scope was the one
vocabulary still missing, and the prediction held exactly: asked for a workflow
that branches on a test result, the Configurator wrote
`{{run_tests.output}}` — a name `build_run_scope` never produces.

The save gate refuses that (`workflow_contract` check 5), so it cannot ship.
But the refusal lands after the design is finished, and a model told only "that
name is wrong" cannot reliably infer the right one.

The second half is subtler and the gate does *not* catch it. `step_facts` reads
`exit_code` and `stdout` out of `output["shell"]`, so only shell steps ever
publish them; an llm step publishes `status` alone. The gate's producible set is
every id crossed with every fact — deliberately, since it refuses only names no
run could publish — so a condition on an llm step's `stdout` saves cleanly and
fails on every run. These tests pin both statements to the code that makes them
true.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from code_bridge_core.configurator import (  # noqa: E402
    _run_scope_vocabulary_block,
    build_configurator_system_prompt,
)
from code_bridge_core.run_scope import STEP_FACTS, step_facts  # noqa: E402


class RunScopeVocabularyBlockTests(unittest.TestCase):
    def test_every_fact_the_scope_publishes_is_named(self) -> None:
        """Read from `STEP_FACTS`, not transcribed beside it."""

        block = _run_scope_vocabulary_block()
        for fact in STEP_FACTS:
            self.assertIn(fact, block, f"{fact} is publishable but unnamed")

    def test_the_shape_the_model_actually_wrote_is_called_out(self) -> None:
        self.assertIn("{{step_id.output}}", _run_scope_vocabulary_block())

    def test_the_route_from_a_model_to_a_later_step_is_named(self) -> None:
        """A model can only use what the prompt tells it exists."""

        block = _run_scope_vocabulary_block()

        self.assertIn("text", block)
        self.assertIn("{{steps.<id>.text}}", block)

    def test_the_block_reaches_the_prompt(self) -> None:
        prompt = build_configurator_system_prompt()
        self.assertIn("Referring to earlier steps:", prompt)
        self.assertNotIn("{{RUN_SCOPE_VOCABULARY_BLOCK}}", prompt)


class WhatTheBlockClaimsIsTrueTests(unittest.TestCase):
    """The claim about shell-only facts, checked against the producer."""

    @staticmethod
    def _row(step_id: str, output: dict) -> dict:
        return {
            "run_id": "run_1",
            "status": "completed",
            "input": {"workflow_step_id": step_id},
            "output": output,
        }

    def test_a_shell_step_publishes_stdout_and_exit_code(self) -> None:
        facts = step_facts(
            [self._row("run_tests", {"shell": {"exit_code": 1, "stdout": "TESTS=FAIL"}})],
            run_id="run_1",
        )

        self.assertEqual(facts["steps.run_tests.stdout"], "TESTS=FAIL")
        self.assertEqual(facts["steps.run_tests.exit_code"], "1")

    def test_an_llm_step_publishes_its_answer_as_text(self) -> None:
        """The route from a model to a later step, which did not exist.

        An llm step writes its answer to `output["result"]`. Nothing read it,
        so a workflow that meant "write the post, then type it into the form"
        could not be expressed — the article had to be a literal string in the
        browser action, and every run posted the same one.
        """

        facts = step_facts(
            [self._row("compose", {"result": "오늘의 글 본문"})],
            run_id="run_1",
        )

        self.assertEqual(facts["steps.compose.text"], "오늘의 글 본문")
        # Still not a shell step: those two remain shell-only.
        self.assertNotIn("steps.compose.stdout", facts)
        self.assertNotIn("steps.compose.exit_code", facts)

    def test_a_step_that_said_nothing_publishes_no_text(self) -> None:
        """Absent, not empty — the same rule `exit_code` follows."""

        facts = step_facts(
            [self._row("quiet", {"message": "Workflow step completed."})],
            run_id="run_1",
        )

        self.assertEqual(facts, {"steps.quiet.status": "completed"})


if __name__ == "__main__":
    unittest.main()
