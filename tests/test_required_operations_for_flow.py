"""The readiness rail's "which step types are ungated" list must not drift.

`policy/required_operations.py` decides which policy operations an agent's
workflow step type will actually make the scheduler consult, by reading two
things: which step types `agent/task_orchestrator.py`'s
`_execute_single_workflow_task_step` dispatches to a dedicated (ungated)
executor, and whether any of those executors ever calls
`evaluate_direct_action_gate`. If either changes without this module being
updated to match, the dashboard's "unattended" readiness chip goes back to
lying — either claiming an operation is required when the runtime never
checks it (the old `OPERATION_FOR_STEP` bug), or, worse, missing a real gate
that got added later and reporting "ready" for a run that will actually
stall.

This test reads the source of both sides with `ast` rather than probing
behaviour with sample inputs, for the same reason
`test_llm_tool_approval_operations.py` does: a newly added branch or a newly
added gate call is caught the moment it is written, with no separate sample
list that itself needs to be kept in sync.
"""

import ast
import inspect
import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import task_orchestrator  # noqa: E402
from chat.chat_stream_service import LLM_TOOL_APPROVAL_OPERATIONS  # noqa: E402
from policy.required_operations import (  # noqa: E402
    UNGATED_WORKFLOW_STEP_TYPES,
    required_operations_for_flow,
    required_operations_for_step_type,
    step_type_operation_catalog,
)
from routes.policies import list_step_operations  # noqa: E402

# The workflow_type dispatch happens inside this one function; every named
# step type it hands off to its own executor branch (rather than falling
# through to the llm turn) must show up in `UNGATED_WORKFLOW_STEP_TYPES`.
_DISPATCHER = task_orchestrator._execute_single_workflow_task_step

# Helpers `_execute_single_workflow_task_step` calls instead of comparing a
# literal directly (currently just the app-action alias check). Resolved by
# name so a rename doesn't silently stop being checked.
_PREDICATE_HELPER_NAMES = ("_is_app_action_workflow_type",)

# Executors a workflow step type can be dispatched to that must stay gate-free
# for `UNGATED_WORKFLOW_STEP_TYPES` to stay accurate. `_wait_for_user_step`
# (used for manual_handoff/mcp_tool/approval_gate) is deliberately not in
# this list: it always pauses for a person regardless of policy, which is a
# different kind of "not ready" than a missing standing rule and is out of
# scope for this module.
_UNGATED_EXECUTORS = (
    "_execute_shell_workflow_step",
    "_execute_browser_action_workflow_step",
    "_execute_app_action_workflow_step",
    "_execute_notify_workflow_step",
)


def _string_constants(node: ast.AST) -> set[str]:
    """String literals directly inside a Constant or a Set of Constants."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.Set):
        return {
            elt.value
            for elt in node.elts
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        }
    return set()


def _literal_return_comparator_strings(func) -> set[str]:
    """String literals a boolean-returning helper compares its argument to."""
    tree = ast.parse(inspect.getsource(func))
    values: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Compare):
            for comparator in node.value.comparators:
                values |= _string_constants(comparator)
    return values


def _dispatched_step_types(dispatcher) -> set[str]:
    """Every literal `workflow_type` value the dispatcher branches on.

    Handles both a direct `workflow_type == "..."` / `workflow_type in
    {...}` comparison and a call out to a named predicate helper (resolved
    against the same module the dispatcher lives in).
    """
    tree = ast.parse(inspect.getsource(dispatcher))
    module = inspect.getmodule(dispatcher)
    values: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = node.test
            if isinstance(test, ast.Compare) and isinstance(test.left, ast.Name):
                if test.left.id == "workflow_type":
                    for comparator in test.comparators:
                        values |= _string_constants(comparator)
            elif isinstance(test, ast.Call) and isinstance(test.func, ast.Name):
                if test.func.id in _PREDICATE_HELPER_NAMES:
                    helper = getattr(module, test.func.id)
                    values |= _literal_return_comparator_strings(helper)
    return values


def _calls_evaluate_direct_action_gate(func) -> bool:
    tree = ast.parse(inspect.getsource(func))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = node.func
            name = target.id if isinstance(target, ast.Name) else getattr(target, "attr", None)
            if name == "evaluate_direct_action_gate":
                return True
    return False


class DispatchedStepTypesMatchUngatedSetTest(unittest.TestCase):
    def test_every_dispatched_step_type_is_in_the_ungated_set(self):
        dispatched = _dispatched_step_types(_DISPATCHER)
        # "condition" completes inline without dispatching to a named
        # executor function, but it is still a literal branch that never
        # reaches the llm turn, so it belongs in the same ungated set.
        self.assertTrue(dispatched, "expected to find at least one dispatched step type")
        self.assertEqual(
            dispatched,
            set(UNGATED_WORKFLOW_STEP_TYPES),
            "task_orchestrator.py's workflow_type dispatch and "
            "policy/required_operations.py's UNGATED_WORKFLOW_STEP_TYPES "
            "have drifted apart",
        )

    def test_ungated_executors_never_consult_a_policy_gate(self):
        for name in _UNGATED_EXECUTORS:
            with self.subTest(executor=name):
                func = getattr(task_orchestrator, name)
                self.assertFalse(
                    _calls_evaluate_direct_action_gate(func),
                    f"{name} now calls evaluate_direct_action_gate — "
                    "policy/required_operations.py must be updated to report "
                    "the operation it gates on, since it is no longer ungated",
                )


class RequiredOperationsForStepTypeTest(unittest.TestCase):
    def test_llm_requires_the_full_tool_approval_set(self):
        self.assertEqual(
            required_operations_for_step_type("llm"),
            LLM_TOOL_APPROVAL_OPERATIONS,
        )

    def test_unrecognised_step_type_is_treated_as_llm(self):
        # task_orchestrator.py's dispatcher falls through to the llm turn for
        # anything it does not name explicitly, so a future/unknown type must
        # be treated the same way here rather than silently requiring nothing.
        self.assertEqual(
            required_operations_for_step_type("some_future_step_type"),
            LLM_TOOL_APPROVAL_OPERATIONS,
        )

    def test_every_ungated_step_type_requires_nothing(self):
        for step_type in UNGATED_WORKFLOW_STEP_TYPES:
            with self.subTest(step_type=step_type):
                self.assertEqual(required_operations_for_step_type(step_type), ())


class RequiredOperationsForFlowTest(unittest.TestCase):
    def test_walkthrough_flow_needs_the_full_llm_set(self):
        # The exact flow from the walkthrough: shell + llm + approval_gate +
        # notify, with only process.terminal granted as a standing rule.
        flow = [
            {"type": "shell"},
            {"type": "llm"},
            {"type": "approval_gate"},
            {"type": "notify"},
        ]
        required = required_operations_for_flow(flow)
        self.assertEqual(set(required), set(LLM_TOOL_APPROVAL_OPERATIONS))
        granted = {"process.terminal"}
        self.assertFalse(
            set(required).issubset(granted),
            "walkthrough agent must NOT read as unattended-ready with only "
            "process.terminal granted",
        )

    def test_flow_with_no_llm_step_needs_nothing(self):
        flow = [{"type": "shell"}, {"type": "notify"}]
        self.assertEqual(required_operations_for_flow(flow), [])

    def test_result_is_deduped_and_order_stable(self):
        flow = [{"type": "llm"}, {"type": "llm"}, {"type": "shell"}]
        required = required_operations_for_flow(flow)
        self.assertEqual(len(required), len(set(required)))
        self.assertEqual(required, list(LLM_TOOL_APPROVAL_OPERATIONS))


class StepTypeOperationCatalogTest(unittest.TestCase):
    def test_catalog_has_an_entry_for_every_ungated_type_and_llm(self):
        catalog = step_type_operation_catalog()
        for step_type in UNGATED_WORKFLOW_STEP_TYPES:
            with self.subTest(step_type=step_type):
                self.assertIn(step_type, catalog)
                self.assertEqual(catalog[step_type], [])
        self.assertIn("llm", catalog)
        self.assertEqual(set(catalog["llm"]), set(LLM_TOOL_APPROVAL_OPERATIONS))


class DashboardStepOperationsRouteTest(unittest.TestCase):
    def test_route_matches_the_module_directly(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from routes import dashboard_agents
        from routes.deps import require_local_access

        app = FastAPI()
        app.include_router(dashboard_agents.router)
        app.dependency_overrides[require_local_access] = lambda: None
        client = TestClient(app)

        response = client.get("/api/dashboard/agent/policies/step-operations")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["operations_by_step_type"],
            list_step_operations()["operations_by_step_type"],
        )
        self.assertEqual(
            response.json()["operations_by_step_type"],
            step_type_operation_catalog(),
        )


if __name__ == "__main__":
    unittest.main()
