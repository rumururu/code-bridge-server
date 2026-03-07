import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from file_action_service import FileActionResult
from routes.deps import verify_api_key
from routes.files import router as files_router


class FilesRoutesTest(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(files_router)
        app.dependency_overrides[verify_api_key] = lambda: True
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

    def test_list_files_project_not_found_returns_404(self):
        with patch(
            "routes.files.list_project_files_for_current_server",
            return_value=FileActionResult(
                success=False,
                status_code=404,
                payload={"error": "Project demo not found"},
            ),
        ):
            response = self.client.get("/api/projects/demo/files")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json().get("error"), "Project demo not found")

    def test_write_file_content_success_returns_payload(self):
        with patch(
            "routes.files.write_project_file_content_for_current_server",
            return_value=FileActionResult(
                success=True,
                status_code=200,
                payload={"success": True, "path": "lib/main.dart"},
            ),
        ):
            response = self.client.put(
                "/api/projects/demo/files/content",
                json={
                    "path": "lib/main.dart",
                    "content": "void main() {}",
                    "create_dirs": False,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"success": True, "path": "lib/main.dart"})

    def test_delete_file_passes_recursive_flag(self):
        with patch(
            "routes.files.delete_project_path_for_current_server",
            return_value=FileActionResult(
                success=True,
                status_code=200,
                payload={"success": True, "path": "logs"},
            ),
        ) as delete_mock:
            response = self.client.delete(
                "/api/projects/demo/files",
                params={"path": "logs", "recursive": "true"},
            )

        self.assertEqual(response.status_code, 200)
        delete_mock.assert_called_once_with("demo", "logs", recursive=True)

    def test_upload_attachment_success_returns_payload(self):
        with patch(
            "routes.files.upload_project_attachment_for_current_server",
            return_value=FileActionResult(
                success=True,
                status_code=200,
                payload={
                    "success": True,
                    "path": ".codebridge_uploads/20260225-abc-note.txt",
                    "name": "note.txt",
                    "size": 5,
                    "source": "file",
                },
            ),
        ) as upload_mock:
            response = self.client.post(
                "/api/projects/demo/files/upload",
                files={"file": ("note.txt", b"hello", "text/plain")},
                data={"source": "file"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["path"], ".codebridge_uploads/20260225-abc-note.txt")
        upload_mock.assert_called_once()
        call_args = upload_mock.call_args
        self.assertEqual(call_args.args[0], "demo")
        self.assertEqual(call_args.kwargs["filename"], "note.txt")
        self.assertEqual(call_args.kwargs["content"], b"hello")


if __name__ == "__main__":
    unittest.main()
