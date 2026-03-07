import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from preview_route_service import PreviewRouteResult
from routes.deps import verify_api_key
from routes.preview import router as preview_router


class PreviewRoutesTest(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(preview_router)
        app.dependency_overrides[verify_api_key] = lambda: True
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

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
