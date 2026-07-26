"""``approve_rule`` has to leave a standing policy rule behind.

The decision value has always been accepted by the API, but nothing acted on
it: the approval was recorded and the next identical operation prompted again.
That is invisible in interactive use — you just tap approve twice — and fatal
for scheduled runs, where no client is connected to tap anything and the task
parks in ``waiting_for_user`` until someone notices.

These tests pin both halves: the rule is written, and the policy engine then
answers "allowed" for the same operation without creating an approval.
"""

import sys
import tempfile
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from approvals import approval_service, approval_store
from audit import audit_store
from core import database
from policy import policy_store


class StandingRuleTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "standing_rule_test.db"
        self._reset_stores()
        self.addCleanup(self._restore)

        self.service = approval_service
        self.rules = policy_store.get_policy_rule_store()

    @staticmethod
    def _reset_stores() -> None:
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None

    def _restore(self) -> None:
        self._reset_stores()
        database.DB_PATH = self._original_db_path

    def _pending(self, *, project_name: str | None = "demo_project", run_id: str = "run_1"):
        details = {"tool_name": "Bash"}
        if project_name is not None:
            details["project_name"] = project_name
        return self.service.request_approval_for_operation(
            operation="process.terminal",
            run_id=run_id,
            actor={"type": "agent_session"},
            details=details,
        )

    def test_approve_once_leaves_no_rule(self):
        pending = self._pending()
        self.assertTrue(pending["approval_required"])

        self.service.decide_approval(
            pending["approval"]["id"],
            decision="approve_once",
            approver={"type": "user"},
        )
        self.assertEqual(self.rules.list_rules(), [])

        # Same operation still needs approval.
        self.assertTrue(self._pending(run_id="run_2")["approval_required"])

    def test_approve_rule_writes_an_allow_rule_and_stops_prompting(self):
        pending = self._pending()
        result = self.service.decide_approval(
            pending["approval"]["id"],
            decision="approve_rule",
            approver={"type": "user", "id": "mankil"},
        )

        rule = result["rule"]
        self.assertEqual(rule["effect"], "allow")
        self.assertEqual(rule["operation"], "process.terminal")
        self.assertEqual(rule["created_by"], "mankil")

        # The whole point: the next run of the same operation is allowed
        # outright, with no approval row created for anyone to answer.
        second = self._pending(run_id="run_2")
        self.assertFalse(second["approval_required"])
        self.assertTrue(second["allowed"])
        self.assertNotIn("approval", second)

    def test_rule_scope_defaults_to_the_project_not_global(self):
        pending = self._pending(project_name="demo_project")
        result = self.service.decide_approval(
            pending["approval"]["id"], decision="approve_rule"
        )
        self.assertEqual(result["rule"]["scope"], "project:demo_project")

    def test_rule_scope_falls_back_to_the_run_when_there_is_no_project(self):
        # "__global__" is the placeholder the agent runner uses for "no
        # project"; treating it as a project name would write a rule that
        # matches nothing.
        pending = self._pending(project_name="__global__", run_id="run_9")
        result = self.service.decide_approval(
            pending["approval"]["id"], decision="approve_rule"
        )
        self.assertEqual(result["rule"]["scope"], "run:run_9")

    def test_explicit_rule_scope_wins(self):
        pending = self._pending()
        result = self.service.decide_approval(
            pending["approval"]["id"],
            decision="approve_rule",
            rule_scope="global",
        )
        self.assertEqual(result["rule"]["scope"], "global")

    def test_project_rule_does_not_leak_into_another_project(self):
        pending = self._pending(project_name="demo_project")
        self.service.decide_approval(pending["approval"]["id"], decision="approve_rule")

        other = self.service.request_approval_for_operation(
            operation="process.terminal",
            run_id="run_3",
            details={"project_name": "other_project", "tool_name": "Bash"},
        )
        self.assertTrue(other["approval_required"])

    def test_decision_audit_records_the_standing_rule(self):
        pending = self._pending()
        self.service.decide_approval(pending["approval"]["id"], decision="approve_rule")

        events = audit_store.get_audit_store().list_events(limit=10)
        decision_events = [e for e in events if e.get("decision") == "approve_rule"]
        self.assertTrue(decision_events, "approve_rule decision was not audited")
        payload = decision_events[0].get("payload") or {}
        self.assertEqual((payload.get("standing_rule") or {}).get("effect"), "allow")


if __name__ == "__main__":
    unittest.main()
