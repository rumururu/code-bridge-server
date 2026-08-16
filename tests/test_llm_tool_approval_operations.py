"""The dashboard's "Unattended permissions" operation list must not drift again.

`_approval_operation_for_tool` (chat/chat_stream_service.py) is the *only*
place that decides which policy operation an LLM tool-permission prompt maps
to. The dashboard form that lets a user pre-authorize operations for
unattended runs used to hand-copy a second, independent list of five options
into an HTML `<select>` instead of reading this one. The two lists drifted:
the form offered `file.read` and `browser.control`, which
`_approval_operation_for_tool` could not then return, so a standing rule
created for either one looked like a working pre-authorization and was not —
no approval request was ever checked against it. The form also omitted
`git.commit` and, worse, `provider.tool` — the catch-all covering every tool
call that is not a shell command, a file write, or a git operation.

`file.read` has since stopped being a phantom: the read-only tools (Read,
Glob, Grep, LS, ...) request approval under it, so it is a real operation
that arrives in the form through `LLM_TOOL_APPROVAL_OPERATIONS` like any
other. `browser.control` is still a phantom and must stay out.

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


class ReadOnlyToolsAreNotTheCatchAllTest(unittest.TestCase):
    """The inversion: asking to read a file, not asking to run a command.

    Read/Glob/Grep used to fall through to `provider.tool`, which is in no
    policy set, so `decide_policy` hit its unknown branch and returned
    `confirm_each`. `Bash` mapped to `process.terminal` — a *named*
    operation, and therefore one a standing allow rule can match, which is
    the natural rule to create the first time an unattended run stalls on a
    shell command. Net effect in the product: it asked permission to read a
    file and did not ask to run a shell command.
    """

    def test_read_only_tools_map_to_file_read(self):
        for tool_name in (
            "Read",
            "read",
            "Glob",
            "Grep",
            "LS",
            "NotebookRead",
            "TodoRead",
            "TodoWrite",
        ):
            with self.subTest(tool_name=tool_name):
                self.assertEqual(_approval_operation_for_tool(tool_name), "file.read")

    def test_web_tools_are_network_not_read(self):
        """A URL never reaches the path guard, so `file.read` would be a blank cheque.

        The safety valve that makes `file.read` acceptable for the local
        tools is `_approval_target_details` handing the path guard something
        to inspect. WebFetch/WebSearch have no path — `decide_policy(
        "file.read", details={})` is a plain `allow` — so filing them as
        read-only would auto-approve every outbound request an unattended
        run makes, and the URL in a prompt-injected fetch is chosen by the
        attacker. `network.external` is CONFIRM_EACH.
        """
        for tool_name in ("WebFetch", "webfetch", "WebSearch", "websearch"):
            with self.subTest(tool_name=tool_name):
                self.assertEqual(
                    _approval_operation_for_tool(tool_name), "network.external"
                )

    def test_mutating_tools_keep_their_own_operations(self):
        # Nothing in the read-only branch may swallow a tool that writes,
        # shells out, or touches the repository.
        for tool_name, expected in (
            ("Bash", "process.terminal"),
            ("Edit", "file.write"),
            ("MultiEdit", "file.write"),
            ("Write", "file.write"),
            ("NotebookEdit", "file.write"),
            ("git_push", "git.commit"),
        ):
            with self.subTest(tool_name=tool_name):
                self.assertEqual(_approval_operation_for_tool(tool_name), expected)

    def test_unknown_tools_still_reach_the_catch_all(self):
        # An MCP tool nobody has classified must not be quietly treated as a
        # read. `provider.tool` -> unknown branch -> confirm_each is the
        # correct, conservative answer for something we cannot characterise.
        for tool_name in ("some_mcp_tool", "linear__create_issue", "", None):
            with self.subTest(tool_name=tool_name):
                self.assertEqual(_approval_operation_for_tool(tool_name), "provider.tool")


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
        # `browser.control` is never consulted by any approval check anywhere
        # in the codebase (LLM tool path or direct-action gate) — a rule for
        # it could never do anything.
        catalog = list_policy_operations()["operations"]
        offered = {entry["value"] for entry in catalog}
        self.assertNotIn("browser.control", offered)

    def test_file_read_is_offered_because_it_is_now_a_real_operation(self):
        # The inverse of the rule above, and the reason the rule is phrased
        # as "must be consultable" rather than as a fixed denylist: an
        # operation belongs in the form exactly when some call site asks for
        # approval under it. `_approval_operation_for_tool` now does.
        self.assertEqual(_approval_operation_for_tool("Read"), "file.read")
        catalog = list_policy_operations()["operations"]
        by_value = {entry["value"]: entry for entry in catalog}
        self.assertIn("file.read", by_value)
        self.assertEqual(by_value["file.read"]["surface"], "llm_tool_call")


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
