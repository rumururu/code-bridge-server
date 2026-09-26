"""The cursor routes a branching condition off the record, and nothing else.

T-H-06. Evaluation happens in the runner and lands on the row as
``output.condition``; ``advance_on_success`` *reads* that and returns a goto.
The split matters: the cursor's contract is that it touches no store and has
no side effects (its module docstring), and judging a predicate needs the
other rows' outputs, which needs a store. Reading one field off the completed
row needs neither — and it also means the route and the record can never
disagree, because there is one judgement and the route is read out of it.

`tests/test_step_cursor.py` pins the pre-branching behaviour and passes
unmodified; this file only adds what branching introduces.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from code_bridge_core.step_cursor import StepCursor, branch_history  # noqa: E402
from source_lint import read_module_source  # noqa: E402


def _branching_run(condition_output):
    """A run shaped like spec example A: shell → condition → two arms."""

    return [
        {
            "id": "r0",
            "status": "completed",
            "input": {
                "workflow_step_id": "run_sync",
                "workflow_type": "shell",
                "on_success": {"type": "continue"},
            },
        },
        {
            "id": "r1",
            "status": "completed",
            "input": {
                "workflow_step_id": "check_exit",
                "workflow_type": "condition",
                "on_success": {"type": "continue"},
                "branches": [
                    {
                        "label": "실패",
                        "when": {
                            "left": "{{steps.run_sync.exit_code}}",
                            "op": "not_equals",
                            "right": "0",
                        },
                        "target_step_id": "diagnose",
                    },
                    {
                        "label": "성공",
                        "when": {
                            "left": "{{steps.run_sync.exit_code}}",
                            "op": "equals",
                            "right": "0",
                        },
                        "target_step_id": "notify_ok",
                    },
                ],
            },
            "output": condition_output,
        },
        {
            "id": "r2",
            "status": "queued",
            "input": {"workflow_step_id": "diagnose", "workflow_type": "llm"},
        },
        {
            "id": "r3",
            "status": "queued",
            "input": {"workflow_step_id": "notify_ok", "workflow_type": "notify"},
        },
    ]


def _record(target, *, index=0, label="실패", default=False):
    return {
        "condition": {
            "matched_index": index,
            "matched_label": label,
            "target_step_id": target,
            "default": default,
            "op": None if default else "not_equals",
            "resolved": None if default else {"left": "1", "right": "0"},
            "evaluated": [],
        }
    }


class BranchRoutingTest(unittest.TestCase):
    def _route(self, output):
        steps = _branching_run(output)
        return StepCursor(steps).advance_on_success(steps, 1)

    def test_the_recorded_arm_decides_where_the_run_goes(self):
        route = self._route(_record("diagnose"))
        self.assertEqual(route.kind, "goto")
        self.assertEqual(route.target_step_id, "diagnose")
        self.assertEqual(route.next_index, 2)

    def test_the_other_arm_goes_to_the_other_step(self):
        route = self._route(_record("notify_ok", index=1, label="성공"))
        self.assertEqual(route.kind, "goto")
        self.assertEqual(route.next_index, 3)

    def test_a_branch_beats_the_next_step_in_the_list(self):
        # Without the record this row would `advance` to index 2 because its
        # on_success is `continue`. The arm is what decides, not the list.
        route = self._route(_record("notify_ok", index=1, label="성공"))
        self.assertNotEqual(route.next_index, 2)

    def test_a_backward_arm_is_a_jump_not_a_cycle(self):
        # A polling loop is written as an arm that goes back a step (spec
        # example C). It must route, not park as an unorderable graph.
        route = self._route(_record("run_sync", label="아직 실행 중"))
        self.assertEqual(route.kind, "goto")
        self.assertEqual(route.next_index, 0)

    def test_a_default_arm_routes_like_any_other(self):
        route = self._route(_record("notify_ok", index=1, label="충분", default=True))
        self.assertEqual(route.kind, "goto")
        self.assertEqual(route.next_index, 3)

    def test_an_unresolvable_target_aborts_the_way_a_goto_does(self):
        route = self._route(_record("nowhere"))
        self.assertEqual(route.kind, "abort")
        self.assertIn("nowhere", route.error_message or "")


class UnchangedRoutingTest(unittest.TestCase):
    """Everything that is not a recorded branch keeps routing as it did."""

    def test_a_condition_with_no_record_advances(self):
        steps = _branching_run(
            {"result": "condition step completed without branching"}
        )
        route = StepCursor(steps).advance_on_success(steps, 1)
        self.assertEqual(route.kind, "advance")

    def test_a_row_with_no_output_advances(self):
        steps = _branching_run(None)
        route = StepCursor(steps).advance_on_success(steps, 1)
        self.assertEqual(route.kind, "advance")

    def test_a_failed_evaluation_record_does_not_route_a_success(self):
        # `on_failure` owns a failed judgement, and a failed step never
        # reaches advance_on_success. Pinned on the shape rather than on the
        # caller's discipline.
        steps = _branching_run(
            {"condition": {"status": "failed", "reason": "unbound_reference"}}
        )
        route = StepCursor(steps).advance_on_success(steps, 1)
        self.assertEqual(route.kind, "advance")

    def test_a_record_without_a_target_does_not_route(self):
        steps = _branching_run({"condition": {"matched_index": 0}})
        route = StepCursor(steps).advance_on_success(steps, 1)
        self.assertEqual(route.kind, "advance")


class CursorStaysPureTest(unittest.TestCase):
    def test_routing_a_branch_does_not_touch_the_rows(self):
        steps = _branching_run(_record("diagnose"))
        before = repr(steps)
        StepCursor(steps).advance_on_success(steps, 1)
        self.assertEqual(repr(steps), before)

    def test_the_cursor_module_never_imports_a_store(self):
        import ast

        tree = ast.parse(read_module_source("code_bridge_core", "step_cursor"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        self.assertEqual(
            imported,
            {"__future__", "dataclasses", "typing", "agent_flow_core.errors",
             "agent_flow_core.topology",
             # The E1–E6 edge rules, shared with the canvas view and the
             # authoring check. Pure like this module: no store, no kernel.
             "edge_rules"},
            "the cursor answers 'where', it never reads or writes the run",
        )

    def test_it_does_not_evaluate_predicates_itself(self):
        # The row below carries branches whose predicates would match arm 0,
        # but no record. If the cursor ever started judging them it would
        # route to `diagnose`; it must advance instead, because judging needs
        # the run scope and the run scope needs a store.
        steps = _branching_run(None)
        route = StepCursor(steps).advance_on_success(steps, 1)
        self.assertEqual(route.kind, "advance")
        self.assertEqual(route.next_index, 2)


class TransitionBudgetMessageTest(unittest.TestCase):
    """Spec 4.3 / T-H-06 — a loop that hits the budget says which arm looped."""

    def test_the_message_carries_the_last_condition_decisions(self):
        steps = _branching_run(_record("run_sync", label="아직 실행 중"))
        cursor = StepCursor(steps)
        message = cursor.budget_exhausted_message(steps)
        self.assertIn("transition limit exceeded", message)
        self.assertIn("check_exit", message)
        self.assertIn("아직 실행 중", message)
        self.assertIn("run_sync", message)

    def test_a_run_with_no_condition_still_gets_a_message(self):
        steps = _branching_run(None)
        message = StepCursor(steps).budget_exhausted_message(steps)
        self.assertIn("transition limit exceeded", message)
        self.assertNotIn("Last condition", message)

    def test_history_reports_one_line_per_deciding_condition(self):
        steps = _branching_run(_record("diagnose"))
        self.assertEqual(len(branch_history(steps)), 1)


if __name__ == "__main__":
    unittest.main()
