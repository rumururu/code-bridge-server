"""The predicate evaluator alone: eleven operators, and no quiet fallback.

`code_bridge_core.condition_eval` is the whole of "which arm". Everything it can be
asked is here — each operator true and false, each way a judgement can fail,
the order the table is walked in, and the two properties the spec makes
acceptance criteria: it is pure, and the same input gives the same answer.

The failure tests are the point of the file. A condition step that cannot
judge its predicates must *fail*, not slide into the default arm, because the
run record has to stay able to tell "stock was fine" from "stock was
unreadable" — see RUNNER_BRANCHING_SPEC 4.2.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from code_bridge_core.condition_eval import (  # noqa: E402
    MAX_PATTERN_CHARS,
    MAX_SUBJECT_CHARS,
    OPERATORS,
    BranchDecision,
    ConditionEvaluationError,
    evaluate_branches,
)
from code_bridge_core.workflow_v2 import (  # noqa: E402
    CONDITION_OPERATORS,
    CONDITION_UNARY_OPERATORS,
    WorkflowNormalizationError,
    normalize_workflow,
)
from source_lint import read_module_source  # noqa: E402


def _arm(op, left, right=None, *, target="hit", label="hit"):
    when = {"left": left, "op": op}
    if right is not None:
        when["right"] = right
    return {"label": label, "when": when, "target_step_id": target}


def _default(target="fallthrough", label="rest"):
    return {"label": label, "when": None, "target_step_id": target}


class VocabularyTest(unittest.TestCase):
    """The evaluator and the normalizer must know the same eleven operators."""

    def test_every_declared_operator_has_an_implementation(self):
        # The drift this catches: an operator added to CONDITION_OPERATORS —
        # which is what the schema publishes and the Configurator prompt
        # offers — with nothing here to judge it. Authoring would accept it
        # and the runner would fail the step at 3am.
        self.assertEqual(set(OPERATORS), set(CONDITION_OPERATORS))

    def test_there_are_eleven_of_them(self):
        # Fixed by spec section 2; extension is a spec revision, not a patch.
        self.assertEqual(len(CONDITION_OPERATORS), 11)


class OperatorTest(unittest.TestCase):
    def _decide(self, arms, scope=None):
        return evaluate_branches(arms, scope or {})

    def _matches(self, op, left, right=None, scope=None):
        """True when the single arm built from these operands is taken."""
        decision = self._decide([_arm(op, left, right), _default()], scope)
        return not decision.default

    # -- string operators ---------------------------------------------------

    def test_equals_is_exact(self):
        self.assertTrue(self._matches("equals", "OK", "OK"))
        self.assertFalse(self._matches("equals", "ok", "OK"), "case is not folded")
        self.assertFalse(self._matches("equals", " OK", "OK"), "no runtime trim")

    def test_equals_can_compare_against_the_empty_string(self):
        self.assertTrue(self._matches("equals", "{{v}}", "", scope={"v": ""}))

    def test_not_equals(self):
        self.assertTrue(self._matches("not_equals", "1", "0"))
        self.assertFalse(self._matches("not_equals", "0", "0"))

    def test_contains_and_not_contains(self):
        self.assertTrue(self._matches("contains", "job RUNNING now", "RUNNING"))
        self.assertFalse(self._matches("contains", "job DONE", "RUNNING"))
        self.assertTrue(self._matches("not_contains", "job DONE", "RUNNING"))
        self.assertTrue(
            self._matches("contains", "anything", ""),
            "an empty needle is in every haystack — Python's own meaning",
        )

    # -- numeric operators --------------------------------------------------

    def test_ordered_comparisons(self):
        self.assertTrue(self._matches("gt", "10", "9"))
        self.assertFalse(self._matches("gt", "9", "10"))
        self.assertTrue(self._matches("gte", "9", "9"))
        self.assertTrue(self._matches("lt", "9", "10"))
        self.assertFalse(self._matches("lt", "10", "9"))
        self.assertTrue(self._matches("lte", "9", "9"))

    def test_numeric_comparison_is_numeric_not_lexical(self):
        # The whole reason section 2.2 refuses a string fallback: "10" < "9"
        # is true as text and false as arithmetic, and the wrong one of those
        # gets recorded as a confident success.
        self.assertFalse(self._matches("lt", "10", "9"))
        self.assertTrue(self._matches("lt", "9", "10"))

    def test_decimals_do_not_pick_up_binary_float_error(self):
        self.assertTrue(self._matches("equals", "0.3", "0.3"))
        self.assertTrue(self._matches("lte", "0.30", "0.3"))
        self.assertTrue(self._matches("gte", "0.30", "0.3"))

    def test_numeric_operands_tolerate_surrounding_space(self):
        # The single documented exception to "no runtime trim" (spec 2.2).
        self.assertTrue(self._matches("gt", " 10 ", "9"))

    def test_a_non_numeric_operand_fails_rather_than_comparing_as_text(self):
        with self.assertRaises(ConditionEvaluationError) as caught:
            self._decide([_arm("gt", "many", "9"), _default()])
        self.assertEqual(caught.exception.reason, "not_a_number")
        self.assertEqual(caught.exception.operand, "left")
        self.assertEqual(caught.exception.branch_index, 0)
        self.assertIn("many", caught.exception.message)

    def test_the_right_side_is_checked_too(self):
        with self.assertRaises(ConditionEvaluationError) as caught:
            self._decide([_arm("lt", "9", "lots"), _default()])
        self.assertEqual(caught.exception.reason, "not_a_number")
        self.assertEqual(caught.exception.operand, "right")

    def test_nan_and_infinity_are_refused(self):
        for text in ("NaN", "nan", "sNaN", "Infinity", "-Infinity"):
            with self.subTest(text=text):
                with self.assertRaises(ConditionEvaluationError) as caught:
                    self._decide([_arm("gt", text, "0")])
                self.assertEqual(caught.exception.reason, "not_a_number")

    def test_an_empty_operand_is_not_zero(self):
        with self.assertRaises(ConditionEvaluationError) as caught:
            self._decide([_arm("gt", "{{v}}", "0")], {"v": ""})
        self.assertEqual(caught.exception.reason, "not_a_number")

    # -- matches ------------------------------------------------------------

    def test_matches_is_a_search_not_a_fullmatch(self):
        self.assertTrue(self._matches("matches", "error 42 here", r"\d+"))
        self.assertFalse(self._matches("matches", "error here", r"\d+"))

    def test_matches_can_be_anchored_by_the_author(self):
        self.assertFalse(self._matches("matches", "x42", r"^\d+$"))
        self.assertTrue(self._matches("matches", "42", r"^\d+$"))

    def test_a_reference_in_the_pattern_is_never_substituted(self):
        # The pattern is literal. Splicing a runtime value in would let page
        # content rewrite the predicate — and would break on any "(" the
        # value happened to contain.
        decision = self._decide(
            [_arm("matches", "{{v}}", "{{v}}"), _default()], {"v": "abc"}
        )
        self.assertTrue(decision.default, "'abc' does not contain the text {{v}}")
        self.assertTrue(
            self._matches("matches", "{{v}}", r"\{\{v\}\}", scope={"v": "{{v}}"})
        )

    def test_a_pattern_over_the_cap_fails(self):
        with self.assertRaises(ConditionEvaluationError) as caught:
            self._decide([_arm("matches", "x", "a" * (MAX_PATTERN_CHARS + 1))])
        self.assertEqual(caught.exception.reason, "pattern_too_long")
        self.assertEqual(caught.exception.detail["limit"], MAX_PATTERN_CHARS)

    def test_a_subject_over_the_cap_fails(self):
        with self.assertRaises(ConditionEvaluationError) as caught:
            self._decide(
                [_arm("matches", "{{v}}", "a")], {"v": "b" * (MAX_SUBJECT_CHARS + 1)}
            )
        self.assertEqual(caught.exception.reason, "subject_too_long")

    def test_an_uncompilable_pattern_fails_the_step_rather_than_the_run(self):
        # Normalization refuses this shape, so a row carrying it was written
        # some other way. What must not happen is an unhandled re.error inside
        # the runner's dispatch loop.
        with self.assertRaises(ConditionEvaluationError) as caught:
            self._decide([_arm("matches", "x", "(unclosed")])
        self.assertEqual(caught.exception.reason, "pattern_invalid")

    def test_normalization_refuses_the_two_patterns_the_evaluator_defends(self):
        for pattern in ("(unclosed", "{{stock}}"):
            with self.subTest(pattern=pattern):
                with self.assertRaises(WorkflowNormalizationError):
                    normalize_workflow(
                        [
                            {
                                "id": "c",
                                "type": "condition",
                                "branches": [
                                    _arm("matches", "{{v}}", pattern, target="c")
                                ],
                            }
                        ]
                    )

    # -- unary --------------------------------------------------------------

    def test_is_empty_means_exactly_the_empty_string(self):
        self.assertTrue(self._matches("is_empty", "{{v}}", scope={"v": ""}))
        self.assertFalse(
            self._matches("is_empty", "{{v}}", scope={"v": " "}),
            "a space is a character; spec 2.4",
        )

    def test_is_not_empty(self):
        self.assertTrue(self._matches("is_not_empty", "{{v}}", scope={"v": "x"}))
        self.assertFalse(self._matches("is_not_empty", "{{v}}", scope={"v": ""}))

    def test_an_unbound_reference_is_not_an_empty_string(self):
        # The tempting bug: `{{missing}}` resolving to "" would make is_empty
        # answer true for a value nobody ever produced.
        for op in sorted(CONDITION_UNARY_OPERATORS):
            with self.subTest(op=op):
                with self.assertRaises(ConditionEvaluationError) as caught:
                    self._decide([_arm(op, "{{missing}}")])
                self.assertEqual(caught.exception.reason, "unbound_reference")

    def test_a_unary_arm_records_only_a_left_operand(self):
        decision = self._decide([_arm("is_not_empty", "{{v}}")], {"v": "x"})
        self.assertEqual(decision.resolved, {"left": "x"})


class ReferenceResolutionTest(unittest.TestCase):
    def test_several_references_in_one_operand(self):
        decision = evaluate_branches(
            [_arm("equals", "{{a}}-{{b}}", "1-2")], {"a": "1", "b": "2"}
        )
        self.assertEqual(decision.resolved["left"], "1-2")

    def test_a_reference_on_the_right_resolves(self):
        decision = evaluate_branches(
            [_arm("equals", "{{a}}", "{{b}}")], {"a": "7", "b": "7"}
        )
        self.assertEqual(decision.resolved, {"left": "7", "right": "7"})

    def test_the_unbound_message_names_branch_operand_and_reference(self):
        with self.assertRaises(ConditionEvaluationError) as caught:
            evaluate_branches(
                [
                    _arm("equals", "a", "b", label="첫째"),
                    _arm("equals", "{{stock_count}}", "0", label="소량"),
                ],
                {},
            )
        error = caught.exception
        self.assertEqual(error.reason, "unbound_reference")
        self.assertEqual(error.branch_index, 1)
        self.assertEqual(error.operand, "left")
        self.assertEqual(error.detail["reference"], "stock_count")
        self.assertIn("소량", error.message)
        self.assertIn("left", error.message)
        self.assertIn("stock_count", error.message)
        self.assertEqual(
            [entry["index"] for entry in error.evaluated],
            [0],
            "the arms already judged travel with the failure",
        )


class OrderTest(unittest.TestCase):
    def test_first_match_wins(self):
        # Example B from the spec: lte 0 before lt 10. Reverse them and a
        # sold-out shop is reported as "low stock".
        arms = [
            _arm("lte", "{{stock}}", "0", target="restock", label="품절"),
            _arm("lt", "{{stock}}", "10", target="warn", label="소량"),
            _default("notify_ok", "충분"),
        ]
        self.assertEqual(evaluate_branches(arms, {"stock": "0"}).target_step_id, "restock")
        self.assertEqual(evaluate_branches(arms, {"stock": "3"}).target_step_id, "warn")
        self.assertEqual(
            evaluate_branches(arms, {"stock": "50"}).target_step_id, "notify_ok"
        )

    def test_arms_after_the_match_are_never_judged(self):
        # The second arm would raise not_a_number if it were reached.
        decision = evaluate_branches(
            [_arm("equals", "x", "x"), _arm("gt", "nonsense", "1")], {}
        )
        self.assertEqual(decision.matched_index, 0)

    def test_the_default_arm_records_no_comparison(self):
        decision = evaluate_branches([_arm("equals", "a", "b"), _default()], {})
        self.assertTrue(decision.default)
        self.assertEqual(decision.matched_index, 1)
        self.assertIsNone(decision.op)
        self.assertIsNone(decision.resolved)

    def test_no_match_and_no_default_is_a_failure(self):
        with self.assertRaises(ConditionEvaluationError) as caught:
            evaluate_branches(
                [
                    _arm("equals", "a", "b", label="하나"),
                    _arm("equals", "c", "d", label="둘"),
                ],
                {},
            )
        error = caught.exception
        self.assertEqual(error.reason, "no_branch_matched")
        self.assertEqual(len(error.evaluated), 2)
        self.assertIn("하나", error.message)
        self.assertIn("둘", error.message)

    def test_an_empty_branches_list_is_a_failure(self):
        with self.assertRaises(ConditionEvaluationError) as caught:
            evaluate_branches([], {})
        self.assertEqual(caught.exception.reason, "empty_branches")


class RecordShapeTest(unittest.TestCase):
    """RUNNER_BRANCHING_SPEC 5 — the cursor routes off this, so it is a contract."""

    def test_a_match_records_what_it_compared(self):
        decision = evaluate_branches(
            [
                _arm("lte", "{{stock}}", "0", target="restock", label="품절"),
                _arm("lt", "{{stock}}", "10", target="warn", label="소량"),
            ],
            {"stock": "3"},
        )
        self.assertEqual(
            decision.to_record(),
            {
                "matched_index": 1,
                "matched_label": "소량",
                "target_step_id": "warn",
                "default": False,
                "op": "lt",
                "resolved": {"left": "3", "right": "10"},
                "evaluated": [
                    {
                        "index": 0,
                        "label": "품절",
                        "op": "lte",
                        "resolved": {"left": "3", "right": "0"},
                        "matched": False,
                    }
                ],
            },
        )

    def test_the_matched_arm_is_not_repeated_inside_evaluated(self):
        record = evaluate_branches([_arm("equals", "a", "a")], {}).to_record()
        self.assertEqual(record["evaluated"], [])

    def test_a_failure_record_points_at_the_operand(self):
        with self.assertRaises(ConditionEvaluationError) as caught:
            evaluate_branches([_arm("gt", "{{stock}}", "0", label="많음")], {})
        record = caught.exception.to_record()
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["reason"], "unbound_reference")
        self.assertEqual(record["branch_index"], 0)
        self.assertEqual(record["operand"], "left")
        self.assertEqual(record["detail"]["reference"], "stock")
        self.assertIsInstance(record["message"], str)

    def test_the_record_is_plain_json_types(self):
        import json

        record = evaluate_branches(
            [_arm("lte", "{{stock}}", "0"), _default()], {"stock": "5"}
        ).to_record()
        self.assertEqual(json.loads(json.dumps(record)), record)


class PurityTest(unittest.TestCase):
    """Acceptance criteria of T-H-05: no store, no clock, same answer twice."""

    def test_the_module_imports_nothing_that_can_observe_the_world(self):
        # Read as an import graph rather than as text: a store, a socket or a
        # clock reached from here would make "same scope, same arm" false,
        # and reproducibility is what lets a run record be re-read months
        # later. Every import in the module — top level or inside a function
        # — has to be on this list.
        import ast

        allowed = {
            "__future__",
            "re",
            "dataclasses",
            "decimal",
            "typing",
            "code_bridge_core.run_scope",
            "code_bridge_core.workflow_v2",
        }
        tree = ast.parse(
            read_module_source("code_bridge_core", "condition_eval")
        )
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        self.assertEqual(imported - allowed, set())

    def test_the_same_input_gives_the_same_answer(self):
        arms = [
            _arm("lte", "{{stock}}", "0", target="restock", label="품절"),
            _arm("lt", "{{stock}}", "10", target="warn", label="소량"),
            _default("notify_ok", "충분"),
        ]
        scope = {"stock": "3"}
        first = evaluate_branches(arms, scope).to_record()
        for _ in range(20):
            self.assertEqual(evaluate_branches(arms, scope).to_record(), first)

    def test_evaluation_does_not_mutate_its_arguments(self):
        arms = [_arm("lt", "{{stock}}", "10", label="소량"), _default()]
        scope = {"stock": "3"}
        before_arms = repr(arms)
        before_scope = dict(scope)
        evaluate_branches(arms, scope)
        self.assertEqual(repr(arms), before_arms)
        self.assertEqual(scope, before_scope)

    def test_a_decision_is_frozen(self):
        decision = evaluate_branches([_arm("equals", "a", "a")], {})
        self.assertIsInstance(decision, BranchDecision)
        with self.assertRaises(Exception):
            decision.matched_index = 3  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
