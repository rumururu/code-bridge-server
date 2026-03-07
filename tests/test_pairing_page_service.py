import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from pairing import PairingPageContextResult
from pairing_page_service import build_pairing_page_html_for_current_server


class PairingPageServiceTest(unittest.TestCase):
    def test_build_pairing_page_html_for_current_server_success(self):
        context = PairingPageContextResult(
            success=True,
            status_code=200,
            qr_url="codebridge://pair/demo",
            local_url="http://127.0.0.1:8080",
            pair_token="token-1",
            expires_in_seconds=240,
        )

        qr_encoder = MagicMock(return_value="BASE64")
        html_renderer = MagicMock(return_value="<html>ok</html>")

        result = build_pairing_page_html_for_current_server(
            context_result=context,
            qr_encoder=qr_encoder,
            html_renderer=html_renderer,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.content, "<html>ok</html>")
        qr_encoder.assert_called_once_with("codebridge://pair/demo")
        html_renderer.assert_called_once_with(
            qr_base64="BASE64",
            local_url="http://127.0.0.1:8080",
            pair_token="token-1",
            expires_in_seconds=240,
        )

    def test_build_pairing_page_html_for_current_server_invalid_context(self):
        context = PairingPageContextResult(
            success=True,
            status_code=200,
            qr_url=None,
            local_url="http://127.0.0.1:8080",
            pair_token="token-1",
            expires_in_seconds=240,
        )

        result = build_pairing_page_html_for_current_server(context_result=context)

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 500)
        self.assertEqual(result.content, "Failed to build pairing page")


if __name__ == "__main__":
    unittest.main()
