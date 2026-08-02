"""The dashboard's "Unattended permissions" operation list must not drift again.

`_approval_operation_for_tool` (chat/chat_stream_service.py) is the *only*
place that decides which policy operation an LLM tool-permission prompt maps
to. The dashboard form that lets a user pre-authorize operations for
unattended runs used to hand-copy a second, independent list of five options
into an HTML `<select>` instead of reading this one. The two lists drifted:
the form offered `file.read` and `browser.control`, which
`_approval_operation_for_tool` can never return, so a standing rule created
for either one looks like a working pre-authorization and is not — no
approval request is ever checked against it. The form also omitted
`git.commit` and, worse, `provider.tool` — the catch-all covering every tool
call that is not a shell command, a file write, or a git operation (Read,
Glob, Grep, WebFetch, MCP tools...). That meant the single most common way an
unattended run stalls — the model wanting to read a file — had no possible
standing rule at all.

If this regresses: either the dashboard can offer an operation that never
matches a real approval request (the original bug), or a new branch gets
added to `_approval_operation_for_tool` without the option list ever finding
out about it, silently reintroducing the same gap.
"""

import ast
import inspect
import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from chat.chat_stream_service import (  # noqa: E402
    LLM_TOOL_APPROVAL_OPERATIONS,
    _approval_operation_for_tool,
)
from routes.policies import list_policy_operations  # noqa: E402


def _literal_return_strings(func) -> set[str]:
    """Every string literal a function's `return` statements can produce.

    Reads the source rather than probing behaviour with sample inputs, so a
    newly added branch is caught the moment it is written — there is no
    sample-input list here that itself needs to be kept in sync.
    """
    source = inspect.getsource(func)
    tree = ast.parse(source)
    values: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                values.add(node.value.value)
    return values


class ApprovalOperationMappingMatchesConstantTest(unittest.TestCase):
    def test_every_literal_return_value_is_in_the_constant(self):
        self.assertEqual(
            _literal_return_strings(_approval_operation_for_tool),
            set(LLM_TOOL_APPROVAL_OPERATIONS),
        )

    def test_the_constant_has_no_stale_entries(self):
        # Belt and suspenders on the previous assertion: an exact set
        # comparison already catches this, but this pins the failure message
        # to "stale" specifically, since a set diff alone reads the same for
        # "added but not covered" and "left behind after removal".
        live = _literal_return_strings(_approval_operation_for_tool)
        stale = set(LLM_TOOL_APPROVAL_OPERATIONS) - live
        self.assertEqual(stale, set(), f"operations no longer producible: {stale}")

    def test_known_tool_names_resolve_to_a_listed_operation(self):
        for tool_name in ("bash", "Edit", "git_commit", "Read", "WebFetch", "some_mcp_tool"):
            with self.subTest(tool_name=tool_name):
                self.assertIn(
                    _approval_operation_for_tool(tool_name), LLM_TOOL_APPROVAL_OPERATIONS
                )


class PolicyOperationsCatalogTest(unittest.TestCase):
    """`list_policy_operations()` is the form's actual data source."""

    def test_every_llm_tool_operation_is_offered_as_llm_tool_call(self):
        catalog = list_policy_operations()["operations"]
        by_value = {entry["value"]: entry for entry in catalog}
        for operation in LLM_TOOL_APPROVAL_OPERATIONS:
            with self.subTest(operation=operation):
                self.assertIn(operation, by_value)
                self.assertEqual(by_value[operation]["surface"], "llm_tool_call")

    def test_provider_tool_is_explicitly_described_as_the_catch_all(self):
        catalog = list_policy_operations()["operations"]
        by_value = {entry["value"]: entry for entry in catalog}
        detail = by_value["provider.tool"]["surface_detail"].lower()
        self.assertIn("any other tool", detail)

    def test_device_control_is_offered_but_labeled_as_a_different_surface(self):
        # device.control is real (routes/projects.py calls
        # evaluate_direct_action_gate(operation="device.control", ...)
        # directly) but it is never what `_approval_operation_for_tool`
        # returns, so it must not be presented as the same kind of thing as
        # the llm_tool_call operations above.
        catalog = list_policy_operations()["operations"]
        by_value = {entry["value"]: entry for entry in catalog}
        self.assertIn("device.control", by_value)
        self.assertNotEqual(by_value["device.control"]["surface"], "llm_tool_call")

    def test_dead_operations_are_not_offered(self):
        # Neither is ever consulted by any approval check anywhere in the
        # codebase (LLM tool path or direct-action gate) — a rule for either
        # one could never have done anything.
        catalog = list_policy_operations()["operations"]
        offered = {entry["value"] for entry in catalog}
        self.assertNotIn("file.read", offered)
        self.assertNotIn("browser.control", offered)


class DashboardPolicyOperationsRouteTest(unittest.TestCase):
    """The dashboard endpoint the HTML form actually fetches from."""

    def test_route_returns_the_same_catalog(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from routes import dashboard_agents
        from routes.deps import require_local_access

        app = FastAPI()
        app.include_router(dashboard_agents.router)
        app.dependency_overrides[require_local_access] = lambda: None
        client = TestClient(app)

        response = client.get("/api/dashboard/agent/policies/operations")
        self.assertEqual(response.status_code, 200)
        values = {entry["value"] for entry in response.json()["operations"]}
        self.assertEqual(values, set(LLM_TOOL_APPROVAL_OPERATIONS) | {"device.control"})


if __name__ == "__main__":
    unittest.main()
