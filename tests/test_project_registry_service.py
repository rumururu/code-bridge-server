import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import project_action_service


class ProjectRegistryServiceSyncTest(unittest.TestCase):
    def test_create_project_record_for_current_server_invalid_payload(self):
        fake_db = MagicMock()
        fake_db.get_all.return_value = []

        with patch(
            "project_action_service.collect_existing_project_state",
            return_value=(set(), {}),
        ), patch(
            "project_action_service.prepare_project_payload",
            return_value=(None, "Invalid project", 422),
        ):
            result = project_action_service.create_project_record_for_current_server(
                path_value="/tmp/demo",
                project_db=fake_db,
            )

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 422)
        self.assertEqual(result.payload.get("error"), "Invalid project")

    def test_create_project_record_for_current_server_success(self):
        fake_db = MagicMock()
        fake_db.get_all.return_value = []
        fake_db.create.return_value = {"name": "demo", "path": "/tmp/demo"}

        with patch(
            "project_action_service.collect_existing_project_state",
            return_value=(set(), {}),
        ), patch(
            "project_action_service.prepare_project_payload",
            return_value=({"name": "demo", "path": "/tmp/demo", "type": "flutter"}, None, None),
        ):
            result = project_action_service.create_project_record_for_current_server(
                path_value="/tmp/demo",
                project_db=fake_db,
            )

        self.assertTrue(result.success)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.payload.get("name"), "demo")
        fake_db.create.assert_called_once()

    def test_import_project_records_for_current_server_empty_paths(self):
        result = project_action_service.import_project_records_for_current_server([])

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 400)
        self.assertEqual(result.payload.get("error"), "No project paths provided")

    def test_import_project_records_for_current_server_tracks_created_skipped_failed(self):
        fake_db = MagicMock()
        fake_db.get_all.return_value = []
        fake_db.create.return_value = {"name": "demo", "path": "/tmp/demo"}

        with patch(
            "project_action_service.collect_existing_project_state",
            return_value=(set(), {}),
        ), patch(
            "project_action_service.prepare_project_payload",
            side_effect=[
                ({"name": "demo", "path": "/tmp/demo", "type": "flutter"}, None, None),
                (None, "Path already registered as project", 400),
                (None, "Invalid path", 422),
            ],
        ):
            result = project_action_service.import_project_records_for_current_server(
                ["/tmp/demo", "/tmp/dup", "/tmp/invalid"],
                project_db=fake_db,
            )

        self.assertTrue(result.success)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.payload["summary"], {"created": 1, "skipped": 1, "failed": 1, "requested": 3})

    def test_update_project_record_for_current_server_not_found(self):
        fake_db = MagicMock()
        fake_db.exists.return_value = False

        result = project_action_service.update_project_record_for_current_server(
            "missing",
            {"path": "/tmp/demo"},
            project_db=fake_db,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 404)
        self.assertEqual(result.payload.get("error"), "Project missing not found")


class ProjectRegistryServiceAsyncTest(unittest.IsolatedAsyncioTestCase):
    async def test_delete_project_record_for_current_server_stops_running_server(self):
        fake_db = MagicMock()
        fake_db.exists.return_value = True
        fake_db.delete.return_value = True

        stop_mock = AsyncMock(return_value={"success": True})

        result = await project_action_service.delete_project_record_for_current_server(
            "demo",
            project_db=fake_db,
            is_dev_server_running=lambda _: True,
            stop_dev_server=stop_mock,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.payload, {"status": "deleted", "name": "demo"})
        stop_mock.assert_awaited_once_with("demo")
        fake_db.delete.assert_called_once_with("demo")

    async def test_close_project_session_for_current_server_delegates(self):
        fake_session_manager = MagicMock()
        fake_session_manager.close_session = AsyncMock()

        result = await project_action_service.close_project_session_for_current_server(
            "demo",
            session_manager=fake_session_manager,
        )

        self.assertEqual(result, {"status": "closed", "project": "demo"})
        fake_session_manager.close_session.assert_awaited_once_with("demo")


if __name__ == "__main__":
    unittest.main()
