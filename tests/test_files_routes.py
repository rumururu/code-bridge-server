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

from agent import agent_store
from approvals import approval_store
from approvals.approval_service import decide_approval
from audit import audit_store
from core import database
from files.file_action_service import FileActionResult
from policy import policy_store
from routes.deps import verify_api_key
from routes.files import router as files_router


class FilesRoutesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "code_bridge_files_test.db"
        agent_store._agent_store = None
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None

        app = FastAPI()
        app.include_router(files_router)
        app.dependency_overrides[verify_api_key] = lambda: True
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        agent_store._agent_store = None
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

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
        run = agent_store.get_agent_store().create_run(project_name="demo", title="Write file")
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
                    "run_id": run["id"],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"success": True, "path": "lib/main.dart"})
        artifacts = agent_store.get_agent_store().list_artifacts(run["id"])
        self.assertEqual(artifacts[0]["kind"], "file_write")
        self.assertEqual(artifacts[0]["metadata"]["details"]["path"], "lib/main.dart")

    def test_write_file_content_can_require_approval(self):
        run = agent_store.get_agent_store().create_run(project_name="demo", title="Write file")
        with patch(
            "routes.files.write_project_file_content_for_current_server",
            return_value=FileActionResult.ok({"success": True, "path": "lib/main.dart"}),
        ) as write_mock:
            blocked = self.client.put(
                "/api/projects/demo/files/content?require_approval=true",
                json={
                    "path": "lib/main.dart",
                    "content": "void main() {}",
                    "run_id": run["id"],
                },
            )

            self.assertEqual(blocked.status_code, 409)
            write_mock.assert_not_called()
            approval = approval_store.get_approval_store().list_pending(run_id=run["id"])[0]
            self.assertEqual(approval["operation"], "file.write")

            decide_approval(
                approval["id"],
                decision="approve_once",
                reason="test approval",
            )
            allowed = self.client.put(
                "/api/projects/demo/files/content?require_approval=true",
                json={
                    "path": "lib/main.dart",
                    "content": "void main() {}",
                    "run_id": run["id"],
                    "approval_id": approval["id"],
                },
            )

        self.assertEqual(allowed.status_code, 200)
        write_mock.assert_called_once_with(
            "demo",
            "lib/main.dart",
            "void main() {}",
            create_dirs=False,
        )

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

    def test_rename_file_records_agent_artifact(self):
        run = agent_store.get_agent_store().create_run(project_name="demo", title="Rename file")
        with patch(
            "routes.files.rename_project_path_for_current_server",
            return_value=FileActionResult.ok({"success": True, "path": "lib/app.dart"}),
        ) as rename_mock:
            response = self.client.post(
                "/api/projects/demo/files/rename",
                params={
                    "old_path": "lib/main.dart",
                    "new_path": "lib/app.dart",
                    "run_id": run["id"],
                },
            )

        self.assertEqual(response.status_code, 200)
        rename_mock.assert_called_once_with("demo", "lib/main.dart", "lib/app.dart")
        artifacts = agent_store.get_agent_store().list_artifacts(run["id"])
        self.assertEqual(artifacts[0]["kind"], "file_move")
        self.assertEqual(artifacts[0]["metadata"]["details"]["source"], "lib/main.dart")

    def test_upload_attachment_can_require_approval(self):
        run = agent_store.get_agent_store().create_run(project_name="demo", title="Upload file")
        with patch(
            "routes.files.upload_project_attachment_for_current_server",
            return_value=FileActionResult.ok(
                {
                    "success": True,
                    "path": ".codebridge_uploads/note.txt",
                    "name": "note.txt",
                    "size": 5,
                    "source": "file",
                }
            ),
        ) as upload_mock:
            blocked = self.client.post(
                "/api/projects/demo/files/upload?require_approval=true",
                files={"file": ("note.txt", b"hello", "text/plain")},
                data={"source": "file", "run_id": run["id"]},
            )

            self.assertEqual(blocked.status_code, 409)
            upload_mock.assert_not_called()
            approval = approval_store.get_approval_store().list_pending(run_id=run["id"])[0]
            self.assertEqual(approval["operation"], "file.upload")

            decide_approval(
                approval["id"],
                decision="approve_once",
                reason="test approval",
            )
            allowed = self.client.post(
                "/api/projects/demo/files/upload?require_approval=true",
                files={"file": ("note.txt", b"hello", "text/plain")},
                data={
                    "source": "file",
                    "run_id": run["id"],
                    "approval_id": approval["id"],
                },
            )

        self.assertEqual(allowed.status_code, 200)
        upload_mock.assert_called_once()
        call_args = upload_mock.call_args
        self.assertEqual(call_args.args[0], "demo")
        self.assertEqual(call_args.kwargs["filename"], "note.txt")
        self.assertEqual(call_args.kwargs["content"], b"hello")

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
