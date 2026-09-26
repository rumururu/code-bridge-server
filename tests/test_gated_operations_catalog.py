"""`routes.policies.gated_operations()` must equal the operations the runtime gates.

The dashboard marks a standing rule "no effect" when its operation is not in
`gated_operations()`. That verdict is only as good as the catalog behind it: a
gate call site added with a new operation name that is not also added to
`_DIRECT_ACTION_GATED_OPERATIONS` would make the dashboard call a perfectly
live rule inert — and the reverse, a name left in the catalog after its gate is
removed, would show a dead rule as live. So this test reads every
`evaluate_direct_action_gate(operation=...)` and
`request_approval_for_operation(operation=...)` call in the non-test source
and compares the literal operation names against the catalog, the same way
`test_llm_tool_approval_operations.py` pins the LLM-side constant.
"""

import ast
import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent.device_permissions import OPERATION as DEVICE_PERMISSION_OPERATION  # noqa: E402
from chat.chat_stream_service import LLM_TOOL_APPROVAL_OPERATIONS  # noqa: E402
from routes.policies import (  # noqa: E402
    _DIRECT_ACTION_GATED_OPERATIONS,
    gated_operations,
)

_GATE_FUNCTIONS = {"evaluate_direct_action_gate", "request_approval_for_operation"}
_SCAN_DIRS = ("routes", "agent", "chat", "approvals", "policy", "audit", "code_bridge_core")


def _literal_gate_operations() -> set[str]:
    found: set[str] = set()
    for directory in _SCAN_DIRS:
        for path in (SERVER_DIR / directory).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                if name not in _GATE_FUNCTIONS:
                    continue
                for keyword in node.keywords:
                    if keyword.arg == "operation" and isinstance(keyword.value, ast.Constant):
                        if isinstance(keyword.value.value, str):
                            found.add(keyword.value.value)
    return found


class GatedOperationsCatalogTest(unittest.TestCase):
    def test_direct_action_catalog_matches_gate_call_sites(self):
        self.assertEqual(_literal_gate_operations(), set(_DIRECT_ACTION_GATED_OPERATIONS))

    def test_catalog_is_union_of_every_gating_surface(self):
        self.assertEqual(
            gated_operations(),
            frozenset(
                (*LLM_TOOL_APPROVAL_OPERATIONS, *_DIRECT_ACTION_GATED_OPERATIONS, DEVICE_PERMISSION_OPERATION)
            ),
        )

    def test_feedback_reply_send_is_not_gated(self):
        # The feedback agent files its approvals by inserting rows directly, so
        # a standing rule under this name is exactly the inert case the
        # dashboard has to be able to point at.
        self.assertNotIn("feedback.reply.send", gated_operations())


if __name__ == "__main__":
    unittest.main()
