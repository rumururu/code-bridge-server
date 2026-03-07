import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import optional_services


class OptionalServicesTest(unittest.TestCase):
    def test_get_active_tunnel_url_returns_none_without_service(self):
        with patch.object(optional_services, "get_tunnel_service", return_value=None):
            self.assertIsNone(optional_services.get_active_tunnel_url())

    def test_get_active_tunnel_url_returns_none_for_invalid_status_shape(self):
        fake_tunnel = MagicMock()
        fake_tunnel.get_status.return_value = "invalid"

        with patch.object(optional_services, "get_tunnel_service", return_value=fake_tunnel):
            self.assertIsNone(optional_services.get_active_tunnel_url())

    def test_get_active_tunnel_url_returns_none_for_empty_url(self):
        fake_tunnel = MagicMock()
        fake_tunnel.get_status.return_value = {"url": ""}

        with patch.object(optional_services, "get_tunnel_service", return_value=fake_tunnel):
            self.assertIsNone(optional_services.get_active_tunnel_url())

    def test_get_active_tunnel_url_returns_url_when_present(self):
        fake_tunnel = MagicMock()
        fake_tunnel.get_status.return_value = {"url": "https://demo.trycloudflare.com"}

        with patch.object(optional_services, "get_tunnel_service", return_value=fake_tunnel):
            self.assertEqual(optional_services.get_active_tunnel_url(), "https://demo.trycloudflare.com")


if __name__ == "__main__":
    unittest.main()
