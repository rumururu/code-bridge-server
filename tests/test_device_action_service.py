import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import device_action_service


class DeviceActionServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_list_connected_devices_for_current_server_delegates(self):
        fake_scrcpy = MagicMock()
        fake_scrcpy.get_devices = AsyncMock(return_value=[{"id": "device-1"}])

        result = await device_action_service.list_connected_devices_for_current_server(
            scrcpy_manager=fake_scrcpy
        )

        self.assertEqual(result, [{"id": "device-1"}])
        fake_scrcpy.get_devices.assert_awaited_once_with()

    def test_get_scrcpy_status_for_current_server_delegates(self):
        fake_scrcpy = MagicMock()
        fake_scrcpy.get_status.return_value = {"running": True}

        result = device_action_service.get_scrcpy_status_for_current_server(scrcpy_manager=fake_scrcpy)

        self.assertEqual(result, {"running": True})
        fake_scrcpy.get_status.assert_called_once_with()

    async def test_start_scrcpy_for_current_server_delegates(self):
        fake_scrcpy = MagicMock()
        fake_scrcpy.start = AsyncMock(return_value={"success": True})

        result = await device_action_service.start_scrcpy_for_current_server(scrcpy_manager=fake_scrcpy)

        self.assertEqual(result, {"success": True})
        fake_scrcpy.start.assert_awaited_once_with()

    async def test_stop_scrcpy_for_current_server_delegates(self):
        fake_scrcpy = MagicMock()
        fake_scrcpy.stop = AsyncMock(return_value={"success": True})

        result = await device_action_service.stop_scrcpy_for_current_server(scrcpy_manager=fake_scrcpy)

        self.assertEqual(result, {"success": True})
        fake_scrcpy.stop.assert_awaited_once_with()
