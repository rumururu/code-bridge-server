"""The `branches` field on a condition step: shape, order, and what it refuses.

Before this, `condition` was a step type with no fields (`WORKFLOW_STEP_SCHEMA`
mapped it to an empty set), so the runner wrote "condition step completed
without branching" into the output and fell through to the next step — the type
was a label. `branches` is where the routing table lives.

Two properties are load-bearing enough to pin here rather than leave to the
evaluator (T-H-05) or the graph judge (T-H-12):

* **A bare condition step still normalizes to exactly what it did before.**
  Every condition step anyone has ever stored is bare, so the absence of
  `branches` has to keep meaning "no branching, honour `on_success`" — and it
  has to stay tellable from an empty list, which is a half-authored step.
* **Only one thing decides where a successful condition step goes.** With
  branches present, `on_success` must stay `continue`; two authorities over one
  event leave a run record that cannot say which one routed it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from code_bridge_core.workflow_v2 import (  # noqa: E402
    CONDITION_OPERATORS,
    CONDITION_UNARY_OPERATORS,
    WorkflowNormalizationError,
    is_default_branch,
    normalize_condition_branches,
    normalize_workflow,
    normalize_workflow_step,
)


def _flow(*branches, on_success=None):
    """A three-step flow whose middle step is a condition with `branches`."""

    condition: dict = {
        "id": "decide",
        "type": "condition",
        "name": "Decide",
        "branches": list(branches),
    }
    if on_success is not None:
        condition["on_success"] = on_success
    return [
        {"id": "look", "type": "notify", "name": "Look"},
        condition,
        {"id": "left", "type": "notify", "name": "Left"},
        {"id": "right", "type": "notify", "name": "Right"},
    ]


class TheBareConditionIsUntouchedTest(unittest.TestCase):
    """The compatibility case that matters: every stored condition is bare."""

    def test_a_bare_condition_gets_no_branches_key_at_all(self) -> None:
        step = normalize_workflow_step(
            {"id": "c", "type": "condition", "name": "Decide"}, index=1
        )
        self.assertNotIn("branches", step)

    def test_a_bare_condition_still_honours_on_success(self) -> None:
        # Nothing about branching narrows the pre-branching contract.
        step = normalize_workflow_step(
            {
                "id": "c",
                "type": "condition",
                "name": "Decide",
                "on_success": {"type": "end"},
            },
            index=1,
        )
        self.assertEqual(step["on_success"], {"type": "end"})

    def test_an_explicit_null_reads_as_absent(self) -> None:
        step = normalize_workflow_step(
            {"id": "c", "type": "condition", "name": "Decide", "branches": None},
            index=1,
        )
        self.assertNotIn("branches", step)

    def test_an_empty_list_is_kept_and_is_not_the_same_as_absent(self) -> None:
        # A condition node just dropped on the canvas has an empty list, and
        # the commit gate reports that as `branch.empty`. Folding it into
        # "absent" would hide a half-authored step behind the legacy
        # passthrough, so the two stay distinguishable.
        step = normalize_workflow_step(
            {"id": "c", "type": "condition", "name": "Decide", "branches": []},
            index=1,
        )
        self.assertEqual(step["branches"], [])


class TheShapeTest(unittest.TestCase):
    def test_a_branch_normalizes_to_label_when_and_target(self) -> None:
        steps = normalize_workflow(
            _flow(
                {
                    "label": "plenty",
                    "when": {"left": "{{stock}}", "op": "GT", "right": 0},
                    "target_step_id": "right",
                }
            )
        )
        self.assertEqual(
            steps[1]["branches"],
            [
                {
                    "label": "plenty",
                    # `op` is folded to lower case and `right` stringified, so
                    # the evaluator gets one shape rather than three spellings.
                    "when": {"left": "{{stock}}", "op": "gt", "right": "0"},
                    "target_step_id": "right",
                }
            ],
        )

    def test_a_branch_with_no_when_is_the_default(self) -> None:
        steps = normalize_workflow(_flow({"label": "otherwise", "target_step_id": "left"}))
        branch = steps[1]["branches"][0]
        self.assertIsNone(branch["when"])
        self.assertTrue(is_default_branch(branch))

    def test_the_unary_operators_publish_no_right(self) -> None:
        for operator in sorted(CONDITION_UNARY_OPERATORS):
            with self.subTest(operator=operator):
                branches = normalize_condition_branches(
                    [{"when": {"left": "{{x}}", "op": operator}, "target_step_id": "a"}]
                )
                self.assertEqual(branches[0]["when"], {"left": "{{x}}", "op": operator})

    def test_every_published_operator_is_accepted(self) -> None:
        # The operator tuple and the normalizer cannot disagree: a client
        # drawing a picker from `CONDITION_OPERATORS` must not be able to
        # author something the server then refuses.
        for operator in CONDITION_OPERATORS:
            with self.subTest(operator=operator):
                when = {"left": "{{x}}", "op": operator}
                if operator not in CONDITION_UNARY_OPERATORS:
                    when["right"] = "1"
                branches = normalize_condition_branches(
                    [{"when": when, "target_step_id": "a"}]
                )
                self.assertEqual(branches[0]["when"]["op"], operator)


class TheOrderTest(unittest.TestCase):
    def test_declared_order_is_preserved(self) -> None:
        steps = normalize_workflow(
            _flow(
                {"label": "a", "when": {"left": "{{x}}", "op": "is_empty"}, "target_step_id": "left"},
                {"label": "b", "when": {"left": "{{x}}", "op": "equals", "right": "1"}, "target_step_id": "right"},
                {"label": "c", "when": {"left": "{{x}}", "op": "contains", "right": "z"}, "target_step_id": "left"},
            )
        )
        self.assertEqual([b["label"] for b in steps[1]["branches"]], ["a", "b", "c"])

    def test_the_default_is_moved_last_and_the_rest_keep_their_order(self) -> None:
        # A default matches everything, so anything written after it can never
        # run. Moving it is the only ordering in which all three are reachable.
        steps = normalize_workflow(
            _flow(
                {"label": "a", "when": {"left": "{{x}}", "op": "is_empty"}, "target_step_id": "left"},
                {"label": "fallback", "target_step_id": "left"},
                {"label": "b", "when": {"left": "{{x}}", "op": "equals", "right": "1"}, "target_step_id": "right"},
            )
        )
        self.assertEqual(
            [b["label"] for b in steps[1]["branches"]], ["a", "b", "fallback"]
        )

    def test_two_defaults_are_refused_rather_than_reordered(self) -> None:
        with self.assertRaisesRegex(
            WorkflowNormalizationError, r"at most one default branch"
        ):
            normalize_workflow(
                _flow(
                    {"label": "one", "target_step_id": "left"},
                    {"label": "two", "target_step_id": "right"},
                )
            )


class TheTargetTest(unittest.TestCase):
    def test_a_target_may_name_a_later_step(self) -> None:
        # The reason existence is checked once the whole list is normalized
        # rather than per step: forward branches are the normal case.
        steps = normalize_workflow(_flow({"label": "on", "target_step_id": "right"}))
        self.assertEqual(steps[1]["branches"][0]["target_step_id"], "right")

    def test_a_target_may_name_an_earlier_step(self) -> None:
        # Backward branches are loops, which the transition budget bounds —
        # normalization does not get to call them invalid.
        steps = normalize_workflow(_flow({"label": "back", "target_step_id": "look"}))
        self.assertEqual(steps[1]["branches"][0]["target_step_id"], "look")

    def test_a_target_that_is_not_a_step_is_refused(self) -> None:
        with self.assertRaisesRegex(
            WorkflowNormalizationError, r"branch 1 targets unknown step: ghost"
        ):
            normalize_workflow(_flow({"label": "on", "target_step_id": "ghost"}))

    def test_a_branch_with_no_target_is_refused(self) -> None:
        with self.assertRaisesRegex(
            WorkflowNormalizationError, r"branch 1 requires target_step_id"
        ):
            normalize_condition_branches([{"label": "nowhere"}])

    def test_the_check_lands_at_the_same_moment_as_a_goto_check(self) -> None:
        # Both are refusals from `normalize_workflow`, never from
        # `normalize_workflow_step`. If they split, a workflow would be valid
        # or invalid depending on which entry point asked.
        step = {
            "id": "decide",
            "type": "condition",
            "name": "Decide",
            "branches": [{"label": "on", "target_step_id": "ghost"}],
        }
        normalize_workflow_step(step, index=1)  # shape only — no complaint
        with self.assertRaises(WorkflowNormalizationError):
            normalize_workflow([step])

        goto = {
            "id": "go",
            "type": "notify",
            "name": "Go",
            "on_success": {"type": "goto_step", "target_step_id": "ghost"},
        }
        normalize_workflow_step(goto, index=1)
        with self.assertRaises(WorkflowNormalizationError):
            normalize_workflow([goto])


class OneAuthorityOverSuccessTest(unittest.TestCase):
    def test_branches_plus_a_non_continue_on_success_is_refused(self) -> None:
        for policy in ({"type": "end"}, {"type": "goto_step", "target_step_id": "left"}):
            with self.subTest(policy=policy["type"]):
                with self.assertRaisesRegex(
                    WorkflowNormalizationError, r"on_success must stay 'continue'"
                ):
                    normalize_workflow(
                        _flow(
                            {"label": "on", "target_step_id": "right"},
                            on_success=policy,
                        )
                    )

    def test_branches_with_the_default_continue_are_fine(self) -> None:
        steps = normalize_workflow(_flow({"label": "on", "target_step_id": "right"}))
        self.assertEqual(steps[1]["on_success"], {"type": "continue"})

    def test_an_explicit_continue_is_fine(self) -> None:
        steps = normalize_workflow(
            _flow({"label": "on", "target_step_id": "right"}, on_success="continue")
        )
        self.assertEqual(steps[1]["on_success"], {"type": "continue"})

    def test_an_empty_branch_list_leaves_on_success_alone(self) -> None:
        # Nothing to conflict with: with no branches there is still only one
        # authority, so the pre-branching contract holds.
        steps = normalize_workflow(_flow(on_success={"type": "end"}))
        self.assertEqual(steps[1]["on_success"], {"type": "end"})

    def test_on_failure_is_untouched(self) -> None:
        # A predicate that cannot be judged is a *failure*, a different event,
        # and the existing failure vocabulary keeps all of it.
        steps = normalize_workflow(
            _flow(
                {"label": "on", "target_step_id": "right"},
                on_success={"type": "continue"},
            )
        )
        self.assertEqual(
            steps[1]["on_failure"], {"type": "ask_user", "resume": "same_step"}
        )


class TypoedKeysAreRefusedTest(unittest.TestCase):
    """A dropped key becomes a predicate comparing something else."""

    def test_an_unknown_branch_key_is_refused(self) -> None:
        with self.assertRaisesRegex(
            WorkflowNormalizationError, r"'targetStepId' is not a field of a branch"
        ):
            normalize_condition_branches([{"targetStepId": "left"}])

    def test_an_unknown_predicate_key_is_refused(self) -> None:
        with self.assertRaisesRegex(
            WorkflowNormalizationError,
            r"'value' is not a field of a branch condition",
        ):
            normalize_condition_branches(
                [
                    {
                        "when": {"left": "{{x}}", "op": "equals", "value": "1"},
                        "target_step_id": "left",
                    }
                ]
            )

    def test_an_unknown_operator_is_refused(self) -> None:
        with self.assertRaisesRegex(
            WorkflowNormalizationError, r"unknown condition operator: ~="
        ):
            normalize_condition_branches(
                [{"when": {"left": "{{x}}", "op": "~="}, "target_step_id": "left"}]
            )

    def test_a_unary_operator_given_a_right_is_refused(self) -> None:
        with self.assertRaisesRegex(
            WorkflowNormalizationError, r"'is_empty' compares nothing on the right"
        ):
            normalize_condition_branches(
                [
                    {
                        "when": {"left": "{{x}}", "op": "is_empty", "right": "1"},
                        "target_step_id": "left",
                    }
                ]
            )

    def test_a_binary_operator_with_no_right_is_refused(self) -> None:
        with self.assertRaisesRegex(
            WorkflowNormalizationError, r"'gt' needs a 'right'"
        ):
            normalize_condition_branches(
                [{"when": {"left": "{{x}}", "op": "gt"}, "target_step_id": "left"}]
            )

    def test_a_predicate_with_no_left_is_refused(self) -> None:
        with self.assertRaisesRegex(WorkflowNormalizationError, r"'when.left' is required"):
            normalize_condition_branches(
                [{"when": {"op": "is_empty"}, "target_step_id": "left"}]
            )

    def test_branches_must_be_a_list_of_objects(self) -> None:
        with self.assertRaisesRegex(WorkflowNormalizationError, r"branches must be a list"):
            normalize_condition_branches("plenty")
        with self.assertRaisesRegex(WorkflowNormalizationError, r"branch 1 must be an object"):
            normalize_condition_branches(["plenty"])


class BranchesBelongToConditionStepsOnlyTest(unittest.TestCase):
    def test_another_step_type_carrying_branches_is_refused(self) -> None:
        # Not the legacy carve-out: `branches` is new, so no stored step can
        # have one on the wrong type, and scoping it costs nothing.
        for step_type in ("llm", "shell", "notify", "approval_gate"):
            step = {"id": "s", "type": step_type, "name": "S", "branches": []}
            if step_type == "shell":
                step["script_id"] = "x"
            with self.subTest(step_type=step_type):
                with self.assertRaisesRegex(
                    WorkflowNormalizationError,
                    rf"'branches' is not a field of a {step_type} step",
                ):
                    normalize_workflow_step(step, index=1)


if __name__ == "__main__":
    unittest.main()
