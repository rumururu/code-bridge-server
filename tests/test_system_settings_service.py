import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import system_settings_service


class SystemSettingsServiceTest(unittest.TestCase):
    def test_get_heartbeat_settings_for_current_server(self):
        with patch("system_settings_service.get_heartbeat_interval", return_value=9):
            result = system_settings_service.get_heartbeat_settings_for_current_server()

        self.assertTrue(result.success)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.payload, {"interval_minutes": 9, "min": 5, "max": 15})

    def test_update_llm_selection_for_current_server_validation_error(self):
        with patch(
            "system_settings_service.set_selected_llm",
            side_effect=ValueError("provider unavailable"),
        ):
            result = system_settings_service.update_llm_selection_for_current_server("openai", "o3")

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 400)
        self.assertEqual(result.payload.get("error"), "provider unavailable")

    def test_update_codex_settings_for_current_server_success(self):
        expected_payload = {
            "sandbox_mode": "workspace-write",
            "sandbox_modes": [{"id": "workspace-write"}],
        }
        with patch(
            "system_settings_service.set_codex_sandbox_mode",
            return_value=expected_payload,
        ):
            result = system_settings_service.update_codex_settings_for_current_server("workspace-write")

        self.assertTrue(result.success)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.payload, expected_payload)


if __name__ == "__main__":
    unittest.main()
