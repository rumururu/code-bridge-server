import sys
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from approvals import approval_store
from audit import audit_store
from core import database
from policy import policy_store
from routes import approvals, audit
from routes.deps import verify_api_key


class ApprovalAuditRoutesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "code_bridge_safety_test.db"
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None

        app = FastAPI()
        app.include_router(approvals.router)
        app.include_router(audit.router)
        app.dependency_overrides[verify_api_key] = lambda: "test-api-key"
        self.client = TestClient(app)

    def tearDown(self):
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def test_terminal_operation_creates_pending_approval_and_audit(self):
        response = self.client.post(
            "/api/approvals/request",
            json={
                "operation": "process.terminal",
                "run_id": "run_demo",
                "actor": {"client_id": "phone-1"},
                "details": {
                    "command": "npm install",
                    "cwd": "/tmp/demo",
                    "api_key": "secret-value",
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["approval_required"])
        self.assertEqual(payload["policy"]["effect"], "confirm_each")
        approval_id = payload["approval"]["id"]

        pending_response = self.client.get("/api/approvals/pending")
        self.assertEqual(pending_response.status_code, 200)
        self.assertEqual(pending_response.json()["approvals"][0]["id"], approval_id)

        decision_response = self.client.post(
            f"/api/approvals/{approval_id}/decision",
            json={
                "decision": "approve_once",
                "scope": "once",
                "reason": "Install project dependencies",
                "constraints": {"argv_prefix": ["npm", "install"]},
                "approver": {"client_id": "phone-1"},
            },
        )
        self.assertEqual(decision_response.status_code, 200)
        self.assertEqual(decision_response.json()["decision"]["decision"], "approve_once")
        self.assertEqual(decision_response.json()["approval"]["status"], "approved")

        audit_response = self.client.get("/api/audit/events?run_id=run_demo")
        self.assertEqual(audit_response.status_code, 200)
        events = audit_response.json()["events"]
        self.assertEqual(events[0]["decision"], "approve_once")
        self.assertEqual(events[1]["decision"], "approval_requested")
        self.assertEqual(
            events[1]["payload"]["details"]["api_key"],
            "[redacted]",
        )
        self.assertIsNotNone(events[0]["hash_current"])

    def test_allowed_operation_does_not_create_approval(self):
        response = self.client.post(
            "/api/approvals/request",
            json={"operation": "project.read", "details": {"project": "demo"}},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["approval_required"])
        self.assertTrue(payload["allowed"])
        self.assertEqual(payload["policy"]["effect"], "allow")
        self.assertEqual(self.client.get("/api/approvals/pending").json()["approvals"], [])

    def test_forbidden_operation_returns_403(self):
        response = self.client.post(
            "/api/approvals/request",
            json={"operation": "audit.disable"},
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["policy"]["effect"], "forbidden")

    def test_desktop_only_approval_requires_desktop_approver(self):
        response = self.client.post(
            "/api/approvals/request",
            json={
                "operation": "settings.accessible_roots",
                "run_id": "run_demo",
                "details": {"path": "/Users/demo"},
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["approval_required"])
        self.assertTrue(payload["approval"]["desktop_only"])
        self.assertEqual(payload["approval"]["policy"]["effect"], "desktop_only")
        approval_id = payload["approval"]["id"]

        rejected = self.client.post(
            f"/api/approvals/{approval_id}/decision",
            json={
                "decision": "approve_once",
                "approver": {"type": "chat_banner"},
            },
        )
        self.assertEqual(rejected.status_code, 403)
        self.assertIn("Desktop-only", rejected.json()["error"])

        approved = self.client.post(
            f"/api/approvals/{approval_id}/decision",
            json={
                "decision": "approve_once",
                "approver": {"type": "desktop_app"},
            },
        )
        self.assertEqual(approved.status_code, 200)
        self.assertEqual(approved.json()["approval"]["status"], "approved")

    def test_policies_endpoint_returns_default_sets(self):
        response = self.client.get("/api/approvals/policies")

        self.assertEqual(response.status_code, 200)
        self.assertIn("process.terminal", response.json()["default_policy"]["confirm_each"])


if __name__ == "__main__":
    unittest.main()
