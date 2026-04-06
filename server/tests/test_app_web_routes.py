import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from app_web_service import AppWebFileResolution
from routes.app_web import router as app_web_router


class AppWebRoutesTest(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(app_web_router)
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

    def test_serve_flutter_app_returns_error_payload(self):
        with patch(
            "routes.app_web.resolve_flutter_web_file_for_current_server",
            return_value=AppWebFileResolution(
                success=False,
                status_code=404,
                error_payload={"error": "Flutter web build not found"},
            ),
        ):
            response = self.client.get("/app/")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json().get("error"), "Flutter web build not found")

    def test_serve_flutter_app_returns_html_for_index(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "index.html"
            file_path.write_text('<base href="/">')

            with patch(
                "routes.app_web.resolve_flutter_web_file_for_current_server",
                return_value=AppWebFileResolution(
                    success=True,
                    status_code=200,
                    file_path=file_path,
                ),
            ):
                response = self.client.get("/app/")

        self.assertEqual(response.status_code, 200)
        self.assertIn('<base href="/app/">', response.text)

    def test_serve_flutter_app_returns_file_response(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "main.js"
            file_path.write_text("console.log('ok');")

            with patch(
                "routes.app_web.resolve_flutter_web_file_for_current_server",
                return_value=AppWebFileResolution(
                    success=True,
                    status_code=200,
                    file_path=file_path,
                ),
            ):
                response = self.client.get("/app/main.js")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("content-type"), "application/javascript")


if __name__ == "__main__":
    unittest.main()
