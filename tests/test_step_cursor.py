"""The step cursor alone: pure ordering decisions, no store, no side effects.

These mirror, at cursor level, the branches the runner tests pin end-to-end
(`test_workflow_success_policy.SuccessRoutingTest`,
`test_workflow_continue_policy.ContinueRoutingTest`, the retry/park paths in
`test_workflow_runtime.py`). The cursor is the single answer to "which step
runs next" (T-B-06), and in T-B-07 it becomes the kernel delegation point —
so its contract is pinned here independently of the orchestrator's recording.
"""

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent.step_cursor import StepCursor, workflow_step_index


def _steps(first_input=None, *, first_status="completed"):
    return [
        {"id": "s1", "status": first_status, "input": dict(first_input or {})},
        {"id": "s2", "status": "queued", "input": {"workflow_step_id": "diagnose"}},
        {"id": "s3", "status": "queued", "input": {"workflow_step_id": "notify"}},
    ]


class SuccessRoutingTest(unittest.TestCase):
    def _route(self, on_success):
        steps = _steps({"on_success": on_success} if on_success is not None else {})
        return StepCursor(steps).advance_on_success(steps, 0)

    def test_continue_advances(self):
        route = self._route({"type": "continue"})
        self.assertEqual(route.kind, "advance")
        self.assertEqual(route.next_index, 1)

    def test_missing_policy_advances(self):
        # Every workflow written before on_success existed must keep working.
        route = self._route(None)
        self.assertEqual(route.kind, "advance")
        self.assertEqual(route.next_index, 1)

    def test_non_dict_policy_advances(self):
        steps = _steps({"on_success": "end"})  # un-normalized string form
        route = StepCursor(steps).advance_on_success(steps, 0)
        self.assertEqual(route.kind, "advance")
        self.assertEqual(route.next_index, 1)

    def test_end_stops_here(self):
        route = self._route({"type": "end"})
        self.assertEqual(route.kind, "end")
        self.assertIsNone(route.next_index, "the diagnosis step must not run")

    def test_goto_jumps(self):
        route = self._route({"type": "goto_step", "target_step_id": "diagnose"})
        self.assertEqual(route.kind, "goto")
        self.assertEqual(route.next_index, 1)
        self.assertEqual(route.target_step_id, "diagnose")

    def test_goto_with_a_missing_target_aborts(self):
        route = self._route({"type": "goto_step", "target_step_id": "ghost"})
        self.assertEqual(route.kind, "abort")
        self.assertIsNone(route.next_index)
        self.assertEqual(
            route.error_message, "on_success goto_step target not found: ghost"
        )

    def test_goto_with_a_non_string_target_aborts(self):
        route = self._route({"type": "goto_step", "target_step_id": 3})
        self.assertEqual(route.kind, "abort")


class FailureRoutingTest(unittest.TestCase):
    def _route(self, on_failure, *, retry_state=None):
        step_input = {"workflow_step_id": "phone_a"}
        if on_failure is not None:
            step_input["on_failure"] = on_failure
        if retry_state is not None:
            step_input["retry_state"] = retry_state
        steps = _steps(step_input, first_status="failed")
        return StepCursor(steps).route_on_failure(steps, 0)

    # -- retry ---------------------------------------------------------

    def test_first_retry_requeues_the_same_step(self):
        route = self._route({"type": "retry", "max_attempts": 2})
        self.assertEqual(route.kind, "retry")
        self.assertEqual(route.next_index, 0)
        self.assertEqual(route.attempt, 1)
        self.assertEqual(route.max_attempts, 2)

    def test_retry_counts_attempts_from_the_step_input(self):
        route = self._route(
            {"type": "retry", "max_attempts": 3}, retry_state={"attempts": 1}
        )
        self.assertEqual(route.kind, "retry")
        self.assertEqual(route.attempt, 2)

    def test_retry_exhausted_without_then_aborts_with_the_step_error(self):
        route = self._route(
            {"type": "retry", "max_attempts": 2}, retry_state={"attempts": 2}
        )
        self.assertEqual(route.kind, "abort")
        self.assertIsNone(route.error_message, "the step's own error must be reported")

    def test_retry_exhausted_follows_its_then_chain(self):
        route = self._route(
            {"type": "retry", "max_attempts": 1, "then": {"type": "continue"}},
            retry_state={"attempts": 1},
        )
        self.assertEqual(route.kind, "continue")
        self.assertEqual(route.next_index, 1)

    def test_retry_then_goto_jumps(self):
        route = self._route(
            {
                "type": "retry",
                "max_attempts": 1,
                "then": {"type": "goto_step", "target_step_id": "notify"},
            },
            retry_state={"attempts": 1},
        )
        self.assertEqual(route.kind, "goto")
        self.assertEqual(route.next_index, 2)

    def test_max_retries_spelling_is_accepted(self):
        route = self._route({"type": "retry", "max_retries": 2})
        self.assertEqual(route.kind, "retry")
        self.assertEqual(route.max_attempts, 2)

    # -- continue / goto ----------------------------------------------

    def test_continue_moves_to_the_next_step(self):
        route = self._route({"type": "continue"})
        self.assertEqual(route.kind, "continue")
        self.assertEqual(route.next_index, 1, "the second phone must still run")

    def test_goto_step_jumps(self):
        route = self._route({"type": "goto_step", "target_step_id": "notify"})
        self.assertEqual(route.kind, "goto")
        self.assertEqual(route.next_index, 2)
        self.assertEqual(route.target_step_id, "notify")

    def test_goto_alias_is_accepted(self):
        route = self._route({"type": "goto", "target_step_id": "diagnose"})
        self.assertEqual(route.kind, "goto")
        self.assertEqual(route.next_index, 1)

    def test_goto_without_a_target_aborts_with_a_policy_error(self):
        route = self._route({"type": "goto_step"})
        self.assertEqual(route.kind, "abort")
        self.assertEqual(
            route.error_message, "goto_step failure policy is missing target_step_id."
        )

    def test_goto_with_a_missing_target_aborts_with_a_policy_error(self):
        route = self._route({"type": "goto_step", "target_step_id": "ghost"})
        self.assertEqual(route.kind, "abort")
        self.assertEqual(route.error_message, "goto_step target not found: ghost")

    # -- park / abort --------------------------------------------------

    def test_ask_user_parks_with_its_prompt(self):
        route = self._route({"type": "ask_user", "prompt": "What should I do next?"})
        self.assertEqual(route.kind, "park")
        self.assertEqual(route.park_reason, "ask_user")
        self.assertEqual(route.park_prompt, "What should I do next?")

    def test_manual_handoff_parks(self):
        route = self._route({"type": "manual_handoff"})
        self.assertEqual(route.kind, "park")
        self.assertEqual(route.park_reason, "manual_handoff")
        self.assertIsNone(route.park_prompt)

    def test_non_string_prompt_is_dropped(self):
        route = self._route({"type": "ask_user", "prompt": {"say": "hi"}})
        self.assertEqual(route.kind, "park")
        self.assertIsNone(route.park_prompt)

    def test_missing_policy_aborts(self):
        route = self._route(None)
        self.assertEqual(route.kind, "abort")
        self.assertIsNone(route.error_message)

    def test_explicit_abort_aborts(self):
        route = self._route({"type": "abort"})
        self.assertEqual(route.kind, "abort")


class TransitionBudgetTest(unittest.TestCase):
    def test_budget_is_ten_times_the_step_count(self):
        cursor = StepCursor([{}] * 4)
        self.assertEqual(cursor.max_transitions, 40)

    def test_budget_never_drops_below_ten(self):
        cursor = StepCursor([{}])
        self.assertEqual(cursor.max_transitions, 10)

    def test_begin_transition_trips_only_past_the_budget(self):
        cursor = StepCursor([], max_transitions=3)
        self.assertTrue(cursor.begin_transition())
        self.assertTrue(cursor.begin_transition())
        self.assertTrue(cursor.begin_transition())
        self.assertFalse(cursor.begin_transition(), "the 4th transition is over budget")


class SkipAndIndexTest(unittest.TestCase):
    def test_completed_steps_are_skipped(self):
        self.assertTrue(
            StepCursor.should_skip(
                {"status": "completed", "input": {"workflow_step_id": "a"}}
            )
        )

    def test_rows_without_a_workflow_step_id_are_skipped(self):
        self.assertTrue(StepCursor.should_skip({"status": "queued", "input": {}}))
        self.assertTrue(StepCursor.should_skip({"status": "queued", "input": None}))

    def test_workflow_backed_pending_steps_run(self):
        self.assertFalse(
            StepCursor.should_skip(
                {"status": "queued", "input": {"workflow_step_id": "a"}}
            )
        )

    def test_workflow_step_index_finds_by_workflow_id(self):
        steps = _steps()
        self.assertEqual(workflow_step_index(steps, "notify"), 2)
        self.assertIsNone(workflow_step_index(steps, "ghost"))

    def test_workflow_step_index_ignores_non_dict_inputs(self):
        steps = [{"id": "s1", "input": "oops"}, {"id": "s2", "input": {"workflow_step_id": "a"}}]
        self.assertEqual(workflow_step_index(steps, "a"), 1)


if __name__ == "__main__":
    unittest.main()
