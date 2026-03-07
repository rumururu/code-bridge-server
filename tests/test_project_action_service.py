import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import project_action_service


class ProjectActionServiceSyncTest(unittest.TestCase):
    def test_list_projects_for_current_server_delegates(self):
        fake_manager = MagicMock()
        fake_manager.get_all_projects.return_value = [{"name": "demo"}]

        result = project_action_service.list_projects_for_current_server(manager=fake_manager)

        self.assertEqual(result, [{"name": "demo"}])
        fake_manager.get_all_projects.assert_called_once_with()

    def test_get_project_for_current_server_delegates(self):
        fake_manager = MagicMock()
        fake_manager.get_project.return_value = {"name": "demo"}

        result = project_action_service.get_project_for_current_server("demo", manager=fake_manager)

        self.assertEqual(result, {"name": "demo"})
        fake_manager.get_project.assert_called_once_with("demo")

    def test_get_project_build_status_for_current_server_delegates(self):
        fake_manager = MagicMock()
        fake_manager.get_build_status.return_value = {"status": "ready"}

        result = project_action_service.get_project_build_status_for_current_server("demo", manager=fake_manager)

        self.assertEqual(result, {"status": "ready"})
        fake_manager.get_build_status.assert_called_once_with("demo")


class ProjectActionServiceAsyncTest(unittest.IsolatedAsyncioTestCase):
    async def test_start_project_dev_server_for_current_server_delegates(self):
        fake_manager = MagicMock()
        fake_manager.start_dev_server = AsyncMock(return_value={"success": True})

        result = await project_action_service.start_project_dev_server_for_current_server(
            "demo",
            manager=fake_manager,
        )

        self.assertEqual(result, {"success": True})
        fake_manager.start_dev_server.assert_awaited_once_with("demo")

    async def test_run_project_on_device_for_current_server_delegates(self):
        fake_manager = MagicMock()
        fake_manager.run_project_on_device = AsyncMock(return_value={"success": True})

        result = await project_action_service.run_project_on_device_for_current_server(
            "demo",
            "emulator-5554",
            manager=fake_manager,
        )

        self.assertEqual(result, {"success": True})
        fake_manager.run_project_on_device.assert_awaited_once_with("demo", "emulator-5554")

    async def test_build_project_flutter_web_for_current_server_delegates(self):
        fake_manager = MagicMock()
        fake_manager.build_flutter_web = AsyncMock(return_value={"success": True})

        result = await project_action_service.build_project_flutter_web_for_current_server(
            "demo",
            manager=fake_manager,
        )

        self.assertEqual(result, {"success": True})
        fake_manager.build_flutter_web.assert_awaited_once_with("demo")
