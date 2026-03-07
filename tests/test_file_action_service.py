import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import file_action_service


class FileActionServiceTest(unittest.TestCase):
    def test_list_project_files_for_current_server_delegates(self):
        fake_project_manager = MagicMock()
        fake_project_manager.get_project.return_value = {"path": "/tmp/demo"}

        fake_file_manager = MagicMock()
        fake_file_manager.list_directory.return_value = {"entries": [{"name": "src"}]}

        fake_factory = MagicMock(return_value=fake_file_manager)

        result = file_action_service.list_project_files_for_current_server(
            "demo",
            "src",
            manager=fake_project_manager,
            file_manager_factory=fake_factory,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.payload, {"entries": [{"name": "src"}]})
        fake_project_manager.get_project.assert_called_once_with("demo")
        fake_factory.assert_called_once_with("/tmp/demo")
        fake_file_manager.list_directory.assert_called_once_with("src")

    def test_list_project_files_for_current_server_project_not_found(self):
        fake_project_manager = MagicMock()
        fake_project_manager.get_project.return_value = None

        result = file_action_service.list_project_files_for_current_server(
            "missing",
            manager=fake_project_manager,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 404)
        self.assertEqual(result.payload.get("error"), "Project missing not found")

    def test_list_project_files_for_current_server_project_without_path(self):
        fake_project_manager = MagicMock()
        fake_project_manager.get_project.return_value = {"name": "demo"}

        result = file_action_service.list_project_files_for_current_server(
            "demo",
            manager=fake_project_manager,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 400)
        self.assertEqual(result.payload.get("error"), "Project has no path configured")

    def test_delete_project_path_for_current_server_recursive_fallback(self):
        fake_project_manager = MagicMock()
        fake_project_manager.get_project.return_value = {"path": "/tmp/demo"}

        fake_file_manager = MagicMock()
        fake_file_manager.delete_file.return_value = {
            "error": "Not a file (use delete_directory for directories)",
            "code": 400,
        }
        fake_file_manager.delete_directory.return_value = {"success": True, "path": "logs"}

        result = file_action_service.delete_project_path_for_current_server(
            "demo",
            "logs",
            recursive=True,
            manager=fake_project_manager,
            file_manager_factory=lambda _: fake_file_manager,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.payload, {"success": True, "path": "logs"})
        fake_file_manager.delete_file.assert_called_once_with("logs")
        fake_file_manager.delete_directory.assert_called_once_with("logs")

    def test_delete_project_path_for_current_server_non_recursive_keeps_error(self):
        fake_project_manager = MagicMock()
        fake_project_manager.get_project.return_value = {"path": "/tmp/demo"}

        fake_file_manager = MagicMock()
        fake_file_manager.delete_file.return_value = {"error": "Not a file", "code": 400}

        result = file_action_service.delete_project_path_for_current_server(
            "demo",
            "logs",
            recursive=False,
            manager=fake_project_manager,
            file_manager_factory=lambda _: fake_file_manager,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 400)
        self.assertEqual(result.payload.get("error"), "Not a file")
        fake_file_manager.delete_file.assert_called_once_with("logs")
        fake_file_manager.delete_directory.assert_not_called()

    def test_upload_project_attachment_for_current_server_saves_file(self):
        fake_project_manager = MagicMock()
        with tempfile.TemporaryDirectory() as tmp:
            fake_project_manager.get_project.return_value = {"path": tmp}

            result = file_action_service.upload_project_attachment_for_current_server(
                "demo",
                filename="../receipt.png",
                content=b"hello attachment",
                content_type="image/png",
                source="photo",
                manager=fake_project_manager,
            )

            self.assertTrue(result.success)
            self.assertEqual(result.status_code, 200)
            self.assertTrue(result.payload["path"].startswith(".codebridge_uploads/"))
            stored_path = Path(tmp) / result.payload["path"]
            self.assertTrue(stored_path.exists())
            self.assertEqual(stored_path.read_bytes(), b"hello attachment")
            self.assertEqual(result.payload["name"], "receipt.png")

    def test_upload_project_attachment_for_current_server_rejects_large_file(self):
        fake_project_manager = MagicMock()
        with tempfile.TemporaryDirectory() as tmp:
            fake_project_manager.get_project.return_value = {"path": tmp}

            oversized = b"a" * (file_action_service.MAX_ATTACHMENT_UPLOAD_BYTES + 1)
            result = file_action_service.upload_project_attachment_for_current_server(
                "demo",
                filename="large.bin",
                content=oversized,
                manager=fake_project_manager,
            )

            self.assertFalse(result.success)
            self.assertEqual(result.status_code, 413)


if __name__ == "__main__":
    unittest.main()
