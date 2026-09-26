"""What a later step is allowed to read, and what it must never read.

A `condition` step decides where a run goes by comparing values earlier steps
produced. That makes the *scope* — which values exist, under which names, from
which run — the thing that decides where the run goes, so it gets pinned here
rather than discovered at 3am.

Two rules carry most of the weight:

* **Same run only.** Yesterday's exit code is not tonight's evidence. A leak
  across runs would send a step down a branch nobody looked at and record it
  as a success.
* **A missing value is missing.** Nothing here turns an absent value into
  `""` or `"null"`. The caller decides what an unanswered reference means;
  guessing is how a workflow reports the wrong branch confidently.

Spec: `docs/concept/spec/RUNNER_BRANCHING_SPEC.md` section 3 (T-H-03).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from code_bridge_core import run_scope  # noqa: E402
from source_lint import source_of  # noqa: E402


def _browser_row(row_id, run_id, extracted, **extra):
    return {
        "id": row_id,
        "run_id": run_id,
        "status": "completed",
        "output": {"browser_action": {"extracted": extracted}},
        **extra,
    }


def _shell_row(row_id, run_id, workflow_step_id, shell, status="completed"):
    return {
        "id": row_id,
        "run_id": run_id,
        "status": status,
        "input": {"workflow_step_id": workflow_step_id},
        "output": {"shell": shell},
    }


class BrowserBindingsAreUnchangedTest(unittest.TestCase):
    """The shape `task_orchestrator` has always read, read the same way.

    `_bindings_from_earlier_steps` delegates here now; if any of these drift,
    browser action binding drifts with them.
    """

    def test_an_earlier_extraction_is_bound_by_its_name(self) -> None:
        rows = [_browser_row("s1", "run_1", [{"name": "cafe_id", "value": "31245773"}])]
        self.assertEqual(
            run_scope.browser_bindings(rows, run_id="run_1"),
            {"cafe_id": "31245773"},
        )

    def test_another_runs_extraction_is_not_reused(self) -> None:
        rows = [_browser_row("s1", "run_0", [{"name": "cafe_id", "value": "999"}])]
        self.assertEqual(run_scope.browser_bindings(rows, run_id="run_1"), {})

    def test_the_caller_does_not_read_its_own_row(self) -> None:
        rows = [_browser_row("s2", "run_1", [{"name": "cafe_id", "value": "stale"}])]
        self.assertEqual(
            run_scope.browser_bindings(rows, run_id="run_1", exclude_step_id="s2"),
            {},
        )

    def test_the_freshest_reading_wins(self) -> None:
        rows = [
            _browser_row("s1", "run_1", [{"name": "token", "value": "old"}]),
            _browser_row("s2", "run_1", [{"name": "token", "value": "new"}]),
        ]
        self.assertEqual(
            run_scope.browser_bindings(rows, run_id="run_1"), {"token": "new"}
        )

    def test_an_extraction_with_no_value_binds_nothing(self) -> None:
        """The pattern matched nothing. An empty string here would send the
        next action to a truncated URL instead of stopping."""
        rows = [_browser_row("s1", "run_1", [{"name": "cafe_id", "matched": False}])]
        self.assertEqual(run_scope.browser_bindings(rows, run_id="run_1"), {})

    def test_an_unnamed_extraction_is_a_read_not_a_variable(self) -> None:
        rows = [_browser_row("s1", "run_1", [{"text": "some page text"}])]
        self.assertEqual(run_scope.browser_bindings(rows, run_id="run_1"), {})

    def test_rows_that_are_not_dicts_are_skipped(self) -> None:
        rows = [None, "nonsense", _browser_row("s1", "run_1", [{"name": "a", "value": "1"}])]
        self.assertEqual(run_scope.browser_bindings(rows, run_id="run_1"), {"a": "1"})


class StepFactsTest(unittest.TestCase):
    def test_a_completed_shell_step_publishes_status_exit_code_and_stdout(self) -> None:
        rows = [
            _shell_row(
                "row1",
                "run_1",
                "run_sync",
                {"status": "completed", "exit_code": 0, "stdout": "12 devices\n"},
            )
        ]
        self.assertEqual(
            run_scope.step_facts(rows, run_id="run_1"),
            {
                "steps.run_sync.status": "completed",
                "steps.run_sync.exit_code": "0",
                "steps.run_sync.stdout": "12 devices\n",
            },
        )

    def test_a_failed_shell_steps_exit_code_is_readable(self) -> None:
        """The whole point of the feature, and it comes out of `output`.

        `_fail_step` writes its error dict to the *output* column
        (`update_task_step(id, {"status": "failed", "output": error})`), and
        the shell executor passes `{"shell": result.to_output()}` as that
        error. So the nonzero exit code a `condition` step wants to branch on
        sits in exactly the same place a zero one does. A reader that only
        looked at successful rows would find nothing here, which is the one
        case that matters.
        """
        rows = [
            _shell_row(
                "row1",
                "run_1",
                "run_sync",
                {"status": "failed", "exit_code": 3, "stdout": "", "stderr": "boom"},
                status="failed",
            )
        ]
        self.assertEqual(
            run_scope.step_facts(rows, run_id="run_1"),
            {
                "steps.run_sync.status": "failed",
                "steps.run_sync.exit_code": "3",
                "steps.run_sync.stdout": "",
            },
        )

    def test_an_absent_exit_code_is_absent_not_the_string_null(self) -> None:
        """The script was never found, or timed out. "We do not know" must not
        become a value, or `equals "null"` starts working."""
        rows = [
            _shell_row(
                "row1",
                "run_1",
                "run_sync",
                {"status": "failed", "exit_code": None, "stdout": ""},
                status="failed",
            )
        ]
        facts = run_scope.step_facts(rows, run_id="run_1")
        self.assertNotIn("steps.run_sync.exit_code", facts)
        self.assertEqual(facts["steps.run_sync.status"], "failed")

    def test_empty_stdout_is_published(self) -> None:
        """"It ran and said nothing" is a fact a predicate may want to test —
        unlike a missing exit code, the value exists."""
        rows = [
            _shell_row("row1", "run_1", "poll", {"exit_code": 0, "stdout": ""})
        ]
        self.assertEqual(run_scope.step_facts(rows, run_id="run_1")["steps.poll.stdout"], "")

    def test_facts_are_keyed_by_the_authored_step_id_not_the_row_id(self) -> None:
        """The author writes branch targets in workflow step ids, so a
        predicate must read in the same vocabulary. The row id is a generated
        value nobody ever typed."""
        rows = [_shell_row("row-9f6a", "run_1", "run_sync", {"exit_code": 0, "stdout": "x"})]
        facts = run_scope.step_facts(rows, run_id="run_1")
        self.assertIn("steps.run_sync.exit_code", facts)
        self.assertNotIn("steps.row-9f6a.exit_code", facts)

    def test_a_row_with_no_workflow_step_id_publishes_nothing(self) -> None:
        rows = [
            {"id": "row1", "run_id": "run_1", "status": "completed",
             "output": {"shell": {"exit_code": 0, "stdout": "hi"}}},
            {"id": "row2", "run_id": "run_1", "status": "completed",
             "input": {}, "output": {"shell": {"exit_code": 0, "stdout": "hi"}}},
        ]
        self.assertEqual(run_scope.step_facts(rows, run_id="run_1"), {})

    def test_a_non_shell_step_publishes_only_its_status(self) -> None:
        rows = [
            {"id": "row1", "run_id": "run_1", "status": "completed",
             "input": {"workflow_step_id": "summarize"},
             "output": {"llm": {"text": "done"}}},
        ]
        self.assertEqual(
            run_scope.step_facts(rows, run_id="run_1"),
            {"steps.summarize.status": "completed"},
        )

    def test_a_re_run_step_publishes_its_freshest_attempt(self) -> None:
        """A backward branch can send the run through a step twice."""
        rows = [
            _shell_row("row1", "run_1", "poll", {"exit_code": 0, "stdout": "RUNNING"}),
            _shell_row("row2", "run_1", "poll", {"exit_code": 0, "stdout": "DONE"}),
        ]
        self.assertEqual(
            run_scope.step_facts(rows, run_id="run_1")["steps.poll.stdout"], "DONE"
        )


class ValuesFromAnotherRunNeverLeakInTest(unittest.TestCase):
    """The rule that keeps a branch decision honest. Pinned in both namespaces
    and through the merged scope, because a leak in any one of them is enough
    to route a run on evidence nobody gathered tonight."""

    def _mixed_rows(self):
        return [
            # Yesterday's run: same step ids, different answers.
            _browser_row("old-b", "run_0", [{"name": "stock_count", "value": "500"}]),
            _shell_row("old-s", "run_0", "run_sync", {"exit_code": 0, "stdout": "OK"}),
            # Tonight's run.
            _browser_row("new-b", "run_1", [{"name": "stock_count", "value": "0"}]),
            _shell_row(
                "new-s", "run_1", "run_sync",
                {"exit_code": 3, "stdout": "FAILED"}, status="failed",
            ),
        ]

    def test_browser_bindings_only_see_this_run(self) -> None:
        self.assertEqual(
            run_scope.browser_bindings(self._mixed_rows(), run_id="run_1"),
            {"stock_count": "0"},
        )

    def test_step_facts_only_see_this_run(self) -> None:
        self.assertEqual(
            run_scope.step_facts(self._mixed_rows(), run_id="run_1"),
            {
                "steps.run_sync.status": "failed",
                "steps.run_sync.exit_code": "3",
                "steps.run_sync.stdout": "FAILED",
            },
        )

    def test_the_merged_scope_only_sees_this_run(self) -> None:
        scope = run_scope.build_run_scope(self._mixed_rows(), run_id="run_1")
        self.assertEqual(
            scope,
            {
                "stock_count": "0",
                "steps.run_sync.status": "failed",
                "steps.run_sync.exit_code": "3",
                "steps.run_sync.stdout": "FAILED",
            },
        )
        # Nothing yesterday said survives anywhere in the scope.
        self.assertNotIn("500", scope.values())
        self.assertNotIn("OK", scope.values())

    def test_a_run_with_no_rows_of_its_own_gets_an_empty_scope(self) -> None:
        """Not a partially-filled one built from whatever was lying around."""
        self.assertEqual(
            run_scope.build_run_scope(self._mixed_rows(), run_id="run_2"), {}
        )


class ReservedNamespaceTest(unittest.TestCase):
    def test_a_browser_extract_cannot_claim_a_reserved_name(self) -> None:
        """The `{{name}}` regex allows dots, so the collision is reachable.

        Dropping it rather than letting it be overwritten matters: overwriting
        would make the name resolve in exactly those runs where no row
        published that fact, so the same workflow would answer one way tonight
        and another way tomorrow.
        """
        rows = [
            _browser_row(
                "b1", "run_1", [{"name": "steps.run_sync.exit_code", "value": "0"}]
            ),
        ]
        # The raw browser reader is untouched — browser action binding must not
        # change by a character.
        self.assertEqual(
            run_scope.browser_bindings(rows, run_id="run_1"),
            {"steps.run_sync.exit_code": "0"},
        )
        # The condition scope drops it, and no row published a real fact.
        self.assertEqual(run_scope.build_run_scope(rows, run_id="run_1"), {})

    def test_a_real_fact_beats_a_shadowing_extract(self) -> None:
        rows = [
            _browser_row(
                "b1", "run_1", [{"name": "steps.run_sync.exit_code", "value": "0"}]
            ),
            _shell_row("s1", "run_1", "run_sync", {"exit_code": 3, "stdout": ""}),
        ]
        self.assertEqual(
            run_scope.build_run_scope(rows, run_id="run_1")["steps.run_sync.exit_code"],
            "3",
        )


class ReferenceSyntaxIsSharedTest(unittest.TestCase):
    def test_the_pattern_is_the_adapters_own(self) -> None:
        """Imported, not restated. `code_bridge_core.workflow_contract` already does the
        same; three consumers of one syntax must move together."""
        from agent.browser_action_adapter import _BINDING_REF

        self.assertIs(run_scope._BINDING_REF, _BINDING_REF)

    def test_references_are_returned_in_order_without_duplicates(self) -> None:
        self.assertEqual(
            run_scope.references("{{b}}/{{a}}/{{b}}"), ("b", "a")
        )

    def test_a_string_with_no_reference_has_none(self) -> None:
        self.assertEqual(run_scope.references("plain"), ())
        self.assertEqual(run_scope.references(None), ())
        self.assertEqual(run_scope.references(7), ())


class ResolveTest(unittest.TestCase):
    def test_a_bound_reference_is_substituted(self) -> None:
        result = run_scope.resolve("{{stock_count}}", {"stock_count": "3"})
        self.assertEqual(result.text, "3")
        self.assertEqual(result.missing, ())
        self.assertTrue(result.complete)

    def test_a_reference_embedded_in_a_longer_string_is_substituted(self) -> None:
        result = run_scope.resolve(
            "exit={{steps.a.exit_code}} out={{steps.a.stdout}}",
            {"steps.a.exit_code": "0", "steps.a.stdout": "ok"},
        )
        self.assertEqual(result.text, "exit=0 out=ok")

    def test_an_unbound_reference_is_reported_and_left_written(self) -> None:
        """Not blanked. An empty string would make `is_empty` true and send
        the run down a branch on a value nobody read."""
        result = run_scope.resolve("{{missing}}", {})
        self.assertEqual(result.text, "{{missing}}")
        self.assertEqual(result.missing, ("missing",))
        self.assertFalse(result.complete)

    def test_every_unbound_name_is_reported_once_in_order(self) -> None:
        result = run_scope.resolve("{{b}}{{a}}{{b}}", {})
        self.assertEqual(result.missing, ("b", "a"))

    def test_a_partially_bound_string_reports_only_what_is_missing(self) -> None:
        result = run_scope.resolve("{{a}}-{{b}}", {"a": "1"})
        self.assertEqual(result.text, "1-{{b}}")
        self.assertEqual(result.missing, ("b",))

    def test_substitution_is_one_pass(self) -> None:
        """A value that looks like a reference is a value, not a reference.

        Re-expanding would let a page's contents name a variable — the data
        deciding what the workflow reads."""
        result = run_scope.resolve("{{a}}", {"a": "{{b}}", "b": "deep"})
        self.assertEqual(result.text, "{{b}}")
        self.assertEqual(result.missing, ())

    def test_a_non_string_operand_resolves_to_its_text(self) -> None:
        self.assertEqual(run_scope.resolve(7, {}).text, "7")
        self.assertEqual(run_scope.resolve(None, {}).text, "")


class PurityTest(unittest.TestCase):
    def test_nothing_here_touches_a_store(self) -> None:
        """Store access stays with the caller, so predicate evaluation is a
        pure function of the rows it was handed — same scope, same branch."""
        import inspect

        source = source_of(run_scope)
        for forbidden in ("get_agent_store", "list_task_steps", "sqlite", "requests"):
            self.assertNotIn(forbidden, source)

    def test_the_input_rows_are_not_mutated(self) -> None:
        rows = [
            _browser_row("b1", "run_1", [{"name": "a", "value": "1"}]),
            _shell_row("s1", "run_1", "run_sync", {"exit_code": 0, "stdout": "x"}),
        ]
        import copy

        before = copy.deepcopy(rows)
        run_scope.build_run_scope(rows, run_id="run_1")
        self.assertEqual(rows, before)


class OrchestratorBindsTheWholeRunScopeTest(unittest.TestCase):
    """`_bindings_from_earlier_steps` hands a browser action the run scope.

    It used to hand over the extract names alone, on the stated grounds that
    step facts "belong to condition scope". That was true while facts were
    only `status`/`exit_code`/`stdout` — nothing a browser action would type.
    It stopped being true when `steps.<id>.text` arrived, which exists so a
    browser step can type what an `llm` step wrote: with extracts only, the
    one flow the fact was added for parked on its own reference while the
    value sat in the row above, written and saved and gate-approved.
    """

    def test_extract_names_still_bind(self) -> None:
        from agent.task_orchestrator import _bindings_from_earlier_steps

        rows = [
            _browser_row("s1", "run_1", [{"name": "cafe_id", "value": "31245773"}]),
            _shell_row("s3", "run_1", "run_sync", {"exit_code": 0, "stdout": "x"}),
        ]

        class _Store:
            def list_task_steps(self, task_id):
                return rows

        bindings = _bindings_from_earlier_steps(
            _Store(), run_id="run_1", task_id="task_1", step_id="s2"
        )

        self.assertEqual(bindings["cafe_id"], "31245773")

    def test_a_step_fact_binds_too(self) -> None:
        from agent.task_orchestrator import _bindings_from_earlier_steps

        rows = [
            _shell_row("s3", "run_1", "run_sync", {"exit_code": 0, "stdout": "x"}),
        ]

        class _Store:
            def list_task_steps(self, task_id):
                return rows

        bindings = _bindings_from_earlier_steps(
            _Store(), run_id="run_1", task_id="task_1", step_id="s2"
        )

        self.assertEqual(bindings.get("steps.run_sync.stdout"), "x")

    def test_an_unreadable_store_still_parks_rather_than_crashes(self) -> None:
        from agent.task_orchestrator import _bindings_from_earlier_steps

        class _BrokenStore:
            def list_task_steps(self, task_id):
                raise RuntimeError("db is gone")

        self.assertEqual(
            _bindings_from_earlier_steps(
                _BrokenStore(), run_id="run_1", task_id="task_1", step_id="s2"
            ),
            {},
        )


if __name__ == "__main__":
    unittest.main()
