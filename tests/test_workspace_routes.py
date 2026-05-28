import sys
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from audit import audit_store
from core import database
from routes import workspaces
from routes.deps import verify_api_key
from workspaces import workspace_store


class WorkspaceRoutesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "code_bridge_workspace_test.db"
        workspace_store._workspace_store = None
        audit_store._audit_store = None

        app = FastAPI()
        app.include_router(workspaces.router)
        app.dependency_overrides[verify_api_key] = lambda: "test-api-key"
        self.client = TestClient(app)

    def tearDown(self):
        workspace_store._workspace_store = None
        audit_store._audit_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def test_workspace_lifecycle(self):
        create_response = self.client.post(
            "/api/workspaces",
            json={
                "project_name": "demo",
                "root_path": "/tmp/demo",
                "display_name": "Demo Workspace",
                "permissions": {"roots": ["/tmp/demo"]},
            },
        )

        self.assertEqual(create_response.status_code, 200)
        workspace = create_response.json()["workspace"]
        self.assertTrue(workspace["id"].startswith("wsp_"))
        self.assertEqual(workspace["status"], "active")
        self.assertEqual(workspace["permissions"]["roots"], ["/tmp/demo"])

        list_response = self.client.get("/api/workspaces?project_name=demo")
        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(list_response.json()["workspaces"][0]["id"], workspace["id"])

        update_response = self.client.patch(
            f"/api/workspaces/{workspace['id']}",
            json={"display_name": "Renamed", "permissions": {"roots": ["/tmp/demo/lib"]}},
        )
        self.assertEqual(update_response.status_code, 200)
        self.assertEqual(update_response.json()["workspace"]["display_name"], "Renamed")
        self.assertEqual(update_response.json()["workspace"]["permissions"]["roots"], ["/tmp/demo/lib"])

        archive_response = self.client.post(f"/api/workspaces/{workspace['id']}/archive")
        self.assertEqual(archive_response.status_code, 200)
        self.assertEqual(archive_response.json()["workspace"]["status"], "archived")
        self.assertEqual(self.client.get("/api/workspaces").json()["workspaces"], [])

        archived = self.client.get("/api/workspaces?status=archived").json()["workspaces"]
        self.assertEqual(archived[0]["id"], workspace["id"])

    def test_unknown_workspace_returns_404(self):
        response = self.client.get("/api/workspaces/wsp_missing")

        self.assertEqual(response.status_code, 404)

    def test_workspace_snapshot_returns_shallow_entries_and_audit(self):
        root = Path(self._tmp.name) / "demo"
        root.mkdir()
        (root / "app").mkdir()
        (root / "README.md").write_text("# Demo\n", encoding="utf-8")
        workspace = self.client.post(
            "/api/workspaces",
            json={"project_name": "demo", "root_path": str(root)},
        ).json()["workspace"]

        response = self.client.get(f"/api/workspaces/{workspace['id']}/snapshot")

        self.assertEqual(response.status_code, 200)
        snapshot = response.json()["snapshot"]
        self.assertTrue(snapshot["root_exists"])
        self.assertEqual(snapshot["summary"]["entry_count"], 2)
        self.assertEqual(
            {entry["name"] for entry in snapshot["entries"]},
            {"app", "README.md"},
        )
        audit_events = audit_store.get_audit_store().list_events(project_name="demo")
        self.assertEqual(audit_events[0]["operation"], "workspace.snapshot")


if __name__ == "__main__":
    unittest.main()
