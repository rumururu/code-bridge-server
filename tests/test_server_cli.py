import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import server_cli


class ServerCliTest(unittest.TestCase):
    def test_show_pairing_qr_prints_error_when_qrcode_unavailable(self):
        with patch("server_cli.QRCODE_AVAILABLE", False):
            captured = io.StringIO()
            with redirect_stdout(captured):
                server_cli.show_pairing_qr()

        output = captured.getvalue()
        self.assertIn("QR code library not installed", output)

    def test_show_pairing_qr_displays_and_opens_browser(self):
        payload = SimpleNamespace(pair_url="http://localhost:8080/pair")

        with (
            patch("server_cli.QRCODE_AVAILABLE", True),
            patch("server_cli.get_active_tunnel_url", return_value="https://demo.trycloudflare.com"),
            patch("server_cli.build_pairing_qr_payload_for_current_server", return_value=payload),
            patch("server_cli.display_pairing_qr_payload") as display_mock,
            patch("server_cli.open_pairing_page") as open_mock,
        ):
            server_cli.show_pairing_qr(open_browser=True)

        display_mock.assert_called_once_with(payload)
        open_mock.assert_called_once()
        self.assertEqual(open_mock.call_args.args[0], "http://localhost:8080/pair")


if __name__ == "__main__":
    unittest.main()
