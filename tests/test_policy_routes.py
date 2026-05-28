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
from routes import approvals, policies
from routes.deps import verify_api_key


class PolicyRoutesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "code_bridge_policy_test.db"
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None

        app = FastAPI()
        app.include_router(approvals.router)
        app.include_router(policies.router)
        app.dependency_overrides[verify_api_key] = lambda: "test-api-key"
        self.client = TestClient(app)

    def tearDown(self):
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def test_global_allow_rule_bypasses_repeated_confirmation(self):
        create_response = self.client.post(
            "/api/policies/rules",
            json={
                "scope": "global",
                "operation": "process.terminal",
                "effect": "allow",
                "created_by": "desktop-test",
            },
        )

        self.assertEqual(create_response.status_code, 200)
        rule = create_response.json()["rule"]
        self.assertTrue(rule["id"].startswith("pol_"))

        request_response = self.client.post(
            "/api/approvals/request",
            json={
                "operation": "process.terminal",
                "run_id": "run_demo",
                "details": {"command": "npm test"},
            },
        )

        self.assertEqual(request_response.status_code, 200)
        payload = request_response.json()
        self.assertFalse(payload["approval_required"])
        self.assertTrue(payload["allowed"])
        self.assertEqual(payload["policy"]["effect"], "allow")
        self.assertEqual(payload["policy"]["rule"]["id"], rule["id"])
        self.assertEqual(self.client.get("/api/approvals/pending").json()["approvals"], [])

    def test_builtin_forbidden_policy_cannot_be_downgraded_by_rule(self):
        create_response = self.client.post(
            "/api/policies/rules",
            json={
                "scope": "global",
                "operation": "audit.disable",
                "effect": "allow",
            },
        )
        self.assertEqual(create_response.status_code, 200)

        request_response = self.client.post(
            "/api/approvals/request",
            json={"operation": "audit.disable"},
        )

        self.assertEqual(request_response.status_code, 403)
        self.assertEqual(request_response.json()["policy"]["effect"], "forbidden")

    def test_delete_policy_rule(self):
        rule = self.client.post(
            "/api/policies/rules",
            json={
                "scope": "project:demo",
                "operation": "file.write",
                "effect": "confirm_each",
            },
        ).json()["rule"]

        delete_response = self.client.delete(f"/api/policies/rules/{rule['id']}")
        self.assertEqual(delete_response.status_code, 200)
        self.assertEqual(self.client.get("/api/policies/rules").json()["rules"], [])


if __name__ == "__main__":
    unittest.main()
