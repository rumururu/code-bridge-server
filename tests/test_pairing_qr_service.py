import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import pairing_qr_service


class PairingQrServiceTest(unittest.TestCase):
    def test_build_pairing_qr_payload_for_current_server(self):
        fake_config = SimpleNamespace(port=8080, server_name="Code Bridge")
        fake_pairing_data = SimpleNamespace(
            to_qr_url=lambda: "codebridge://pair?x=1",
            local_url="http://127.0.0.1:8080",
            tunnel_url="https://demo.trycloudflare.com",
        )
        fake_pairing = MagicMock()
        fake_pairing.create_pairing_data.return_value = fake_pairing_data

        payload = pairing_qr_service.build_pairing_qr_payload_for_current_server(
            tunnel_url="https://demo.trycloudflare.com",
            config=fake_config,
            pairing_service=fake_pairing,
        )

        self.assertEqual(payload.qr_url, "codebridge://pair?x=1")
        self.assertEqual(payload.local_url, "http://127.0.0.1:8080")
        self.assertEqual(payload.tunnel_url, "https://demo.trycloudflare.com")
        self.assertEqual(payload.server_name, "Code Bridge")
        self.assertEqual(payload.pair_url, "http://localhost:8080/pair")

    def test_display_pairing_qr_payload_calls_display_fn(self):
        payload = pairing_qr_service.PairingQrPayload(
            qr_url="u",
            local_url="l",
            tunnel_url=None,
            server_name="s",
            pair_url="p",
        )
        display_mock = MagicMock()

        pairing_qr_service.display_pairing_qr_payload(payload, display_fn=display_mock)

        display_mock.assert_called_once_with(
            qr_url="u",
            local_url="l",
            tunnel_url=None,
            server_name="s",
        )

    def test_open_pairing_page_uses_opener(self):
        opener = MagicMock()

        pairing_qr_service.open_pairing_page("http://localhost:8080/pair", opener=opener)

        opener.assert_called_once_with("http://localhost:8080/pair")


if __name__ == "__main__":
    unittest.main()
