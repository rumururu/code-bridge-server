"""There was no way to watch a workflow run one step at a time.

Every execution was all-or-nothing: press run, wait, and read afterwards what
the rows recorded. Diagnosing one broken browser step in a cafe-posting agent
took six full runs that way — each one re-ran the steps that already worked,
and each was read from stored evidence rather than watched.

`max_steps` bounds how many steps a single call executes. When the budget runs
out the loop stops before the next step and the run is left `paused`, which is
its own status on purpose: `failed` would put a red run in the list for a walk
that went fine, and `waiting_for_user` would claim a person had been asked
something.

Nothing else had to learn about pausing. Re-entering the loop continues where
it stopped, because `StepCursor.should_skip` walks past completed rows — the
property the existing resume path already relies on. So "one more step" is the
same call with the same budget, and "run the rest" is the same call with none.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent.agent_models import AgentRunOnceRequest  # noqa: E402
from code_bridge_core.step_cursor import StepCursor  # noqa: E402
from tests.source_lint import read_module_source  # noqa: E402


def _row(step_id: str, status: str) -> dict:
    return {
        "id": step_id,
        "status": status,
        "input": {"workflow_step_id": step_id},
    }


class TheBudgetIsRequestableTests(unittest.TestCase):
    def test_run_once_accepts_a_step_budget(self) -> None:
        body = AgentRunOnceRequest(max_steps=1)

        self.assertEqual(body.max_steps, 1)

    def test_a_run_with_no_budget_is_unchanged(self) -> None:
        """Every caller that existed before this passes nothing."""

        self.assertIsNone(AgentRunOnceRequest().max_steps)

    def test_zero_is_refused(self) -> None:
        """A budget of nothing would start a run that cannot move."""

        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            AgentRunOnceRequest(max_steps=0)


class ContinuingWorksBecauseFinishedStepsAreSkippedTests(unittest.TestCase):
    """The property the pause leans on, asserted rather than assumed.

    If a completed row were ever *not* skipped, resuming a paused run would
    re-run work that already happened — a second post, a second notification.
    """

    def test_a_completed_row_is_walked_past(self) -> None:
        self.assertTrue(StepCursor.should_skip(_row("done", "completed")))

    def test_a_queued_row_is_work(self) -> None:
        self.assertFalse(StepCursor.should_skip(_row("next", "queued")))


class WhatThePauseWritesTests(unittest.TestCase):
    """Source-shape: standing up a real run to observe a pause is a slow way
    to check three statements the function makes outright."""

    def setUp(self) -> None:
        self.source = read_module_source("agent", "task_orchestrator")

    def test_the_budget_is_spent_on_work_not_on_skipped_rows(self) -> None:
        # Checked after `should_skip`, so a continuation walking over rows an
        # earlier call finished does not spend its budget on them.
        skip = self.source.index("if cursor.should_skip(step):")
        budget = self.source.index("if max_steps is not None and executed >= max_steps:")
        self.assertLess(skip, budget)

    def test_the_run_is_left_paused_not_failed(self) -> None:
        self.assertIn('store.update_run_status(run_id, "paused")', self.source)

    def test_the_next_step_is_named_in_the_event(self) -> None:
        """A pause the reader cannot locate is not much better than a stop."""

        self.assertIn('event_type="task.execution.paused"', self.source)
        self.assertIn('"next_workflow_step_id"', self.source)

    def test_the_pending_step_is_not_written_to(self) -> None:
        """It stays `queued` — which is what it is: the next thing to run."""

        start = self.source.index("def _pause_workflow_execution(")
        body = self.source[start : self.source.index("def _finish_workflow_execution(")]
        self.assertNotIn("update_task_step", body)


class WhatAPausedRunIsNotTests(unittest.TestCase):
    """`paused` is deliberately outside both of the scheduler's sets.

    Not *progressing*: nothing is driving it, so `_close_out_unfinished_run`
    must not mistake it for a loop that died and fail it.

    Not *waiting*: those are runs parked on a person who was asked something,
    and the stall sweep abandons them. A pause is the person themselves,
    mid-look, and it holds no schedule — their 9am firing should not be
    blocked because they stopped a manual run on step two to read a
    screenshot.

    The cost is that a paused run is never swept: it waits until it is resumed
    or discarded. That is the right trade for a pause someone asked for, and
    it is a decision rather than an oversight, which is why it is written down
    here.
    """

    def test_it_is_not_treated_as_a_loop_that_died(self) -> None:
        from agent.task_orchestrator import _UNFINISHED_RUN_STATUSES

        self.assertNotIn("paused", _UNFINISHED_RUN_STATUSES)

    def test_it_does_not_hold_the_schedule(self) -> None:
        from agent.scheduler import _ACTIVE_RUN_STATUSES

        self.assertNotIn("paused", _ACTIVE_RUN_STATUSES)

    def test_the_stall_sweep_leaves_it_alone(self) -> None:
        from agent.scheduler import _WAITING_RUN_STATUSES

        self.assertNotIn("paused", _WAITING_RUN_STATUSES)


class ContinuingAPausedRunTests(unittest.TestCase):
    """A pause has no checkpoint, so the resume door had to learn about it.

    `resume_task_orchestration` reads the task's active checkpoint, because
    everything that stopped before this existed stopped by asking somebody
    something. A stepped run asks nothing, so that door answered 409 —
    correctly, for the question it was asked. Measured against the live server
    before this path existed.
    """

    def setUp(self) -> None:
        self.source = read_module_source("agent", "task_orchestrator")

    def test_the_continue_path_does_not_require_a_checkpoint(self) -> None:
        start = self.source.index("def continue_paused_run(")
        body = self.source[start : self.source.index("def resume_task_orchestration(")]

        self.assertNotIn("get_task_checkpoint", body)

    def test_only_a_paused_run_may_be_continued(self) -> None:
        """A running one would be driven twice; a finished one has nowhere to go."""

        start = self.source.index("def continue_paused_run(")
        body = self.source[start : self.source.index("def resume_task_orchestration(")]

        self.assertIn('run.get("status") != "paused"', body)

    def test_the_row_stops_claiming_it_is_paused_before_work_restarts(self) -> None:
        start = self.source.index("def continue_paused_run(")
        body = self.source[start : self.source.index("def resume_task_orchestration(")]

        self.assertIn('store.update_run_status(run_id, "running")', body)

    def test_the_route_sends_a_paused_run_down_it(self) -> None:
        route = read_module_source("routes", "agents")

        self.assertIn('if run.get("status") == "paused":', route)
        self.assertIn("continue_paused_run(run_id)", route)


if __name__ == "__main__":
    unittest.main()
