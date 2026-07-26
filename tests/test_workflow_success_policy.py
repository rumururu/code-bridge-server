"""`on_success`: where a workflow goes when a step worked.

A step placed after the work always ran. That is fine for a summary and wrong
for an escalation: a diagnosis step that exists to explain failures also ran on
every clean night, spending a model turn to report that nothing was wrong.
`on_failure: goto_step: diagnose` gives the failure route; without a success
route there was no way to say "and if it worked, we're done".

Defaults to `continue`, which is what every existing workflow already did.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import task_orchestrator
from agent.workflow_v2 import WorkflowNormalizationError, normalize_workflow


class SuccessNormalizationTest(unittest.TestCase):
    def test_default_is_continue(self):
        step = normalize_workflow([{"type": "llm", "id": "a"}])[0]
        self.assertEqual(step["on_success"], {"type": "continue"})

    def test_end_forms(self):
        for raw in ("end", "stop", "done", {"type": "end"}):
            step = normalize_workflow([{"type": "llm", "id": "a", "on_success": raw}])[0]
            self.assertEqual(step["on_success"], {"type": "end"}, raw)

    def test_goto_form(self):
        step = normalize_workflow(
            [
                {"type": "llm", "id": "a", "on_success": "goto_step:report"},
                {"type": "llm", "id": "report"},
            ]
        )[0]
        self.assertEqual(step["on_success"], {"type": "goto_step", "target_step_id": "report"})

    def test_unknown_target_is_rejected(self):
        # Same guard the failure side has: a typo must fail at authoring time.
        with self.assertRaises(WorkflowNormalizationError):
            normalize_workflow([{"type": "llm", "id": "a", "on_success": "goto_step:ghost"}])

    def test_unknown_policy_is_rejected(self):
        with self.assertRaises(WorkflowNormalizationError):
            normalize_workflow([{"type": "llm", "id": "a", "on_success": "teleport"}])

    def test_the_escalation_shape_normalizes(self):
        steps = normalize_workflow(
            [
                {
                    "type": "shell",
                    "id": "cycle",
                    "script_id": "script_1",
                    "on_success": "end",
                    "on_failure": "goto_step:diagnose",
                },
                {"type": "llm", "id": "diagnose"},
            ]
        )
        self.assertEqual(steps[0]["on_success"], {"type": "end"})
        self.assertEqual(steps[0]["on_failure"]["target_step_id"], "diagnose")


class SuccessRoutingTest(unittest.TestCase):
    def _steps(self, on_success):
        return [
            {"id": "s1", "status": "completed", "input": {"on_success": on_success}},
            {"id": "s2", "status": "queued", "input": {"workflow_step_id": "diagnose"}},
        ]

    def _route(self, on_success, *, failed=()):
        store = MagicMock()
        store.list_task_steps.return_value = [
            {"id": step_id, "status": "failed"} for step_id in failed
        ]
        with patch.object(task_orchestrator, "get_agent_store", return_value=store), patch.object(
            task_orchestrator, "_finish_workflow_execution"
        ) as finish:
            index = task_orchestrator._apply_workflow_success_policy(
                task_id="task_1",
                run_id="run_1",
                steps=self._steps(on_success),
                completed_index=0,
            )
        return index, finish

    def test_continue_advances(self):
        index, finish = self._route({"type": "continue"})
        self.assertEqual(index, 1)
        finish.assert_not_called()

    def test_missing_policy_advances(self):
        # Every workflow written before this existed must keep working.
        index, finish = self._route(None)
        self.assertEqual(index, 1)
        finish.assert_not_called()

    def test_end_stops_and_completes(self):
        index, finish = self._route({"type": "end"})
        self.assertIsNone(index, "the diagnosis step must not run")
        self.assertEqual(finish.call_args.kwargs["status"], "completed")

    def test_end_still_reports_failure_when_an_earlier_step_failed(self):
        # Stopping early does not launder a failure that already happened.
        index, finish = self._route({"type": "end"}, failed=["s0"])
        self.assertIsNone(index)
        self.assertEqual(finish.call_args.kwargs["status"], "failed")

    def test_goto_jumps(self):
        index, _ = self._route({"type": "goto_step", "target_step_id": "diagnose"})
        self.assertEqual(index, 1)

    def test_goto_with_a_missing_target_fails_the_run(self):
        index, finish = self._route({"type": "goto_step", "target_step_id": "ghost"})
        self.assertIsNone(index)
        self.assertEqual(finish.call_args.kwargs["status"], "failed")


if __name__ == "__main__":
    unittest.main()
