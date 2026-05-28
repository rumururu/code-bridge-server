import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store
from audit import audit_store
from core import database
from policy import policy_store
from preview.preview_route_service import PreviewRouteResult
from routes.deps import verify_api_key
from routes.preview import router as preview_router


class PreviewRoutesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "code_bridge_preview_test.db"
        agent_store._agent_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None

        app = FastAPI()
        app.include_router(preview_router)
        app.dependency_overrides[verify_api_key] = lambda: True
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        agent_store._agent_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def test_create_preview_token_returns_http_exception_detail(self):
        with patch(
            "routes.preview.create_preview_token_for_current_server",
            return_value=PreviewRouteResult(
                success=False,
                status_code=404,
                payload={"error": "No running dev server for project demo"},
            ),
        ):
            response = self.client.post("/api/preview/token", params={"project": "demo"})

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json().get("detail"), "No running dev server for project demo")

    def test_create_preview_token_records_safe_agent_artifact(self):
        run = agent_store.get_agent_store().create_run(project_name="demo", title="Preview")
        with patch(
            "routes.preview.create_preview_token_for_current_server",
            return_value=PreviewRouteResult.ok(
                {
                    "token": "secret-token",
                    "project": "demo",
                    "expires_in_minutes": 15,
                    "preview_url": "/preview/demo/?preview_token=secret-token",
                }
            ),
        ):
            response = self.client.post(
                "/api/preview/token",
                params={"project": "demo", "run_id": run["id"]},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["token"], "secret-token")
        artifacts = agent_store.get_agent_store().list_artifacts(run["id"])
        self.assertEqual(artifacts[0]["kind"], "preview_token")
        result = artifacts[0]["metadata"]["result"]
        self.assertNotIn("token", result)
        self.assertEqual(result["preview_path"], "/preview/demo/")

    def test_preview_proxy_authorization_failure_returns_json_error(self):
        with patch(
            "routes.preview.authorize_project_preview_request_for_current_server",
            return_value=PreviewRouteResult(
                success=False,
                status_code=403,
                payload={"error": "Invalid or expired preview token"},
            ),
        ):
            response = self.client.get("/preview/demo/")

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json().get("error"), "Invalid or expired preview token")

    def test_preview_proxy_success_delegates_to_proxy(self):
        with (
            patch(
                "routes.preview.authorize_project_preview_request_for_current_server",
                return_value=PreviewRouteResult(success=True, status_code=200, payload={}),
            ),
            patch(
                "routes.preview.resolve_project_preview_target_for_current_server",
                return_value=PreviewRouteResult(success=True, status_code=200, payload={"port": 5173}),
            ),
            patch(
                "routes.preview.proxy_preview_request_for_current_server",
                new=AsyncMock(return_value=JSONResponse(content={"proxied": True})),
            ) as proxy_mock,
        ):
            response = self.client.get("/preview/demo/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"proxied": True})
        proxy_mock.assert_awaited_once()

    def test_root_file_proxy_rejects_unknown_filename(self):
        response = self.client.get("/unknown.txt")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json().get("error"), "Not found")


if __name__ == "__main__":
    unittest.main()
