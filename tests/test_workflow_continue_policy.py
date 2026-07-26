"""`continue`: one workflow, several independent jobs.

The failure vocabulary was abort / ask_user / retry / goto_step. None of those
expresses "this step failed, do the next one anyway", so anything genuinely
independent — three phones, a cleanup allowed to fail — could not share a
workflow. goto_step looks like it works and does not: it skips whatever sits
between the failed step and the target, so a two-phone workflow silently ran
one phone.

Two things are pinned here. The run moves on, and it still ends up failed —
carrying on is not the same as passing, and a green dot on a night when a
phone never ran is the exact silence the status view exists to break.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import task_orchestrator
from agent.workflow_v2 import normalize_workflow


class ContinueNormalizationTest(unittest.TestCase):
    def test_string_form(self):
        step = normalize_workflow([{"type": "llm", "id": "a", "on_failure": "continue"}])[0]
        self.assertEqual(step["on_failure"], {"type": "continue"})

    def test_object_form(self):
        step = normalize_workflow(
            [{"type": "llm", "id": "a", "on_failure": {"type": "continue"}}]
        )[0]
        self.assertEqual(step["on_failure"], {"type": "continue"})

    def test_skip_is_accepted_as_the_same_thing(self):
        step = normalize_workflow([{"type": "llm", "id": "a", "on_failure": "skip"}])[0]
        self.assertEqual(step["on_failure"], {"type": "continue"})

    def test_retry_can_fall_through_to_continue(self):
        step = normalize_workflow(
            [
                {
                    "type": "llm",
                    "id": "a",
                    "on_failure": {"type": "retry", "max_attempts": 2, "then": {"type": "continue"}},
                }
            ]
        )[0]
        self.assertEqual(step["on_failure"]["then"], {"type": "continue"})


class ContinueRoutingTest(unittest.TestCase):
    def _steps(self):
        return [
            {
                "id": "s1",
                "status": "failed",
                "input": {"workflow_step_id": "phone_a", "on_failure": {"type": "continue"}},
            },
            {"id": "s2", "status": "queued", "input": {"workflow_step_id": "phone_b"}},
        ]

    def test_execution_moves_to_the_next_step(self):
        store = MagicMock()
        with patch.object(task_orchestrator, "get_agent_store", return_value=store), patch.object(
            task_orchestrator, "_finish_workflow_execution"
        ) as finish:
            next_index = task_orchestrator._apply_workflow_failure_policy(
                task_id="task_1",
                run_id="run_1",
                steps=self._steps(),
                failed_index=0,
                error={"message": "phone A cycle failed"},
            )
        self.assertEqual(next_index, 1, "the second phone must still run")
        finish.assert_not_called()

    def test_the_failure_is_recorded_not_swallowed(self):
        store = MagicMock()
        with patch.object(task_orchestrator, "get_agent_store", return_value=store), patch.object(
            task_orchestrator, "_finish_workflow_execution"
        ):
            task_orchestrator._apply_workflow_failure_policy(
                task_id="task_1",
                run_id="run_1",
                steps=self._steps(),
                failed_index=0,
                error={"message": "phone A cycle failed"},
            )
        events = [call.kwargs.get("event_type") for call in store.append_event.call_args_list]
        self.assertIn("task.step.continued_after_failure", events)

    def test_abort_still_stops_everything(self):
        steps = self._steps()
        steps[0]["input"]["on_failure"] = {"type": "abort"}
        store = MagicMock()
        with patch.object(task_orchestrator, "get_agent_store", return_value=store), patch.object(
            task_orchestrator, "_finish_workflow_execution"
        ) as finish:
            next_index = task_orchestrator._apply_workflow_failure_policy(
                task_id="task_1",
                run_id="run_1",
                steps=steps,
                failed_index=0,
                error={"message": "boom"},
            )
        self.assertIsNone(next_index)
        finish.assert_called_once()

    def test_goto_step_still_jumps(self):
        steps = self._steps()
        steps[0]["input"]["on_failure"] = {"type": "goto_step", "target_step_id": "phone_b"}
        store = MagicMock()
        with patch.object(task_orchestrator, "get_agent_store", return_value=store), patch.object(
            task_orchestrator, "_finish_workflow_execution"
        ):
            next_index = task_orchestrator._apply_workflow_failure_policy(
                task_id="task_1",
                run_id="run_1",
                steps=steps,
                failed_index=0,
                error={"message": "boom"},
            )
        self.assertEqual(next_index, 1)


if __name__ == "__main__":
    unittest.main()
