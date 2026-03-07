import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import auth_service


class AuthServiceTest(unittest.TestCase):
    @patch("auth_service._check_allow_ip_login", return_value=False)
    def test_validate_api_key_for_current_server_rejects_missing_key(
        self, mock_ip_login
    ):
        """Without IP login, missing API key should be rejected."""
        config = SimpleNamespace(api_key="")
        result = auth_service.validate_api_key_for_current_server(None, config=config)

        self.assertFalse(result.success)
        self.assertEqual(result.error, "API key required")

    @patch("auth_service._check_allow_ip_login", return_value=False)
    def test_validate_api_key_for_current_server_rejects_invalid_key_when_auth_enabled(
        self, mock_ip_login
    ):
        fake_pairing = MagicMock()
        fake_pairing.validate_api_key.return_value = False
        config = SimpleNamespace(api_key="server-static-key")

        result = auth_service.validate_api_key_for_current_server(
            "bad-key",
            pairing_service=fake_pairing,
            config=config,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, "Invalid API key")

    @patch("auth_service._check_allow_ip_login", return_value=False)
    def test_validate_api_key_for_current_server_accepts_static_configured_key(
        self, mock_ip_login
    ):
        fake_pairing = MagicMock()
        config = SimpleNamespace(api_key="server-static-key")

        result = auth_service.validate_api_key_for_current_server(
            "server-static-key",
            pairing_service=fake_pairing,
            config=config,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.api_key, "server-static-key")
        fake_pairing.validate_api_key.assert_not_called()

    @patch("auth_service._check_allow_ip_login", return_value=False)
    def test_validate_api_key_for_current_server_accepts_valid_pairing_key_when_auth_enabled(
        self, mock_ip_login
    ):
        fake_pairing = MagicMock()
        fake_pairing.validate_api_key.return_value = True
        config = SimpleNamespace(api_key="server-static-key")

        result = auth_service.validate_api_key_for_current_server(
            "good-key",
            pairing_service=fake_pairing,
            config=config,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.api_key, "good-key")

    @patch("auth_service._check_allow_ip_login", return_value=True)
    def test_validate_api_key_for_current_server_allows_anonymous_when_ip_login_enabled(
        self, mock_ip_login
    ):
        """When IP login is enabled, anonymous access is allowed."""
        config = SimpleNamespace(api_key="server-static-key")

        result = auth_service.validate_api_key_for_current_server(None, config=config)

        self.assertTrue(result.success)
        self.assertTrue(result.is_ip_login)
        self.assertEqual(result.api_key, "__ip_login__")


if __name__ == "__main__":
    unittest.main()
