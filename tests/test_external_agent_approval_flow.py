"""An agent outside the server files approvals through the API, not the DB.

The feedback agent used to INSERT into ``approval_requests`` directly. Its
requests therefore never met the policy engine: a standing "always allow" rule
for ``feedback.reply.send`` could be created from the phone (the decision
route happily wrote one) and would never fire, because nothing ever asked
``decide_policy_with_rules`` about that operation. This pins the replacement
path end to end on the localhost dashboard listener the agent actually uses:
request -> poll -> decide with approve_rule -> next request is allowed by the
rule -> the rules listing reports the rule as consulted and fired.
"""

import sys
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from approvals import approval_store  # noqa: E402
from audit import audit_store  # noqa: E402
from core import database  # noqa: E402
from policy import policy_store  # noqa: E402
from routes import dashboard_agents  # noqa: E402
from routes.deps import require_local_access, verify_api_key  # noqa: E402

DASH = "/api/dashboard/agent"


class ExternalAgentApprovalFlowTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "external_agent_flow.db"
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None
        app = FastAPI()
        app.include_router(dashboard_agents.router)
        app.dependency_overrides[require_local_access] = lambda: None
        app.dependency_overrides[verify_api_key] = lambda: None
        self.client = TestClient(app)

    def tearDown(self):
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def _request(self):
        return self.client.post(
            f"{DASH}/approvals/request",
            json={
                "operation": "feedback.reply.send",
                "actor": {"type": "agent", "id": "agent-fb", "name": "Feedback"},
                "details": {"project_name": "tfkeyboard", "recipient": "a@example.com"},
                "risk_level": "high",
            },
        )

    def test_request_poll_decide_then_rule_takes_over(self):
        first = self._request()
        self.assertEqual(first.status_code, 200)
        self.assertTrue(first.json()["approval_required"])
        approval_id = first.json()["approval"]["id"]

        polled = self.client.get(f"{DASH}/approvals/{approval_id}").json()
        self.assertEqual(polled["approval"]["status"], "pending")
        self.assertIsNone(polled["decision"])

        # Before anyone decides, a rule for this operation would be inert and
        # the listing has to say so — the audit log only has the gate's own
        # `approval_requested` for it once the request above went through, so
        # from here on it counts as consulted.
        decided = self.client.post(
            f"{DASH}/approvals/{approval_id}/decision",
            json={"decision": "approve_rule", "scope": "once"},
        )
        self.assertEqual(decided.status_code, 200)
        rule = decided.json()["rule"]
        self.assertEqual(rule["scope"], "project:tfkeyboard")

        polled = self.client.get(f"{DASH}/approvals/{approval_id}").json()
        self.assertEqual(polled["approval"]["status"], "approved")
        self.assertEqual(polled["decision"]["decision"], "approve_rule")

        second = self._request()
        self.assertEqual(second.status_code, 200)
        self.assertTrue(second.json()["allowed"])
        self.assertFalse(second.json()["approval_required"])
        self.assertEqual(second.json()["policy"]["rule"]["id"], rule["id"])

        listed = {r["id"]: r for r in self.client.get(f"{DASH}/policies/rules").json()["rules"]}
        self.assertTrue(listed[rule["id"]]["consulted"])
        self.assertEqual(listed[rule["id"]]["matched_count"], 1)

    def test_a_runless_request_rings_the_phone_and_a_run_bound_one_does_not(self):
        from unittest.mock import patch

        with patch("agent.task_orchestrator._push_notification_best_effort") as push:
            self._request()
            self.assertEqual(push.call_count, 1)
            kwargs = push.call_args.kwargs
            self.assertEqual(kwargs["data_extra"]["kind"], "approval_request")
            self.assertTrue(kwargs["data_extra"]["approval_id"].startswith("apr_"))
            self.assertEqual(kwargs["title"], "Feedback needs you: needs your approval")

            # A request a run is parked on is rung for by the orchestrator.
            self.client.post(
                f"{DASH}/approvals/request",
                json={"operation": "process.terminal", "run_id": "run_x",
                      "details": {"command": "git status"}},
            )
            self.assertEqual(push.call_count, 1)

    def test_unknown_approval_is_404(self):
        self.assertEqual(self.client.get(f"{DASH}/approvals/apr_nope").status_code, 404)

    def test_edited_reply_is_recorded_only_on_approval(self):
        approval_id = self._request().json()["approval"]["id"]
        result = self.client.post(
            f"{DASH}/approvals/{approval_id}/decision",
            json={"decision": "approve_once", "constraints": {"reply_body": "edited answer"}},
        )
        self.assertEqual(result.status_code, 200)
        polled = self.client.get(f"{DASH}/approvals/{approval_id}").json()
        self.assertEqual(polled["decision"]["constraints"]["reply_body"], "edited answer")

    def test_rule_for_an_operation_the_gate_never_saw_is_inert(self):
        # A rule written without any request ever passing the gate under that
        # name (the old direct-INSERT path) must not read as live.
        created = self.client.post(
            f"{DASH}/policies/rules",
            json={"scope": "project:x", "operation": "never.gated", "effect": "allow"},
        ).json()["rule"]
        listed = {r["id"]: r for r in self.client.get(f"{DASH}/policies/rules").json()["rules"]}
        self.assertFalse(listed[created["id"]]["consulted"])


if __name__ == "__main__":
    unittest.main()
