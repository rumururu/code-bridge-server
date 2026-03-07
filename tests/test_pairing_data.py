import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from pairing import (
    CurrentPairingDataResult,
    PairingClientStatus,
    PairingPageContextResult,
    PairingRevokeResult,
    PairingStatus,
    PairingVerifyTokenResult,
    PairTokenStatus,
)
from pairing import PairingData
from pairing import PairingService
from pairing import (
    build_current_pairing_data_result,
    build_current_pairing_page_context_result,
    build_current_pairing_qr_result,
    create_current_pairing_data,
    get_pair_token_status_for_current_server,
    get_pairing_status_for_current_server,
    revoke_paired_client_for_current_server,
)


class PairingDataTest(unittest.TestCase):
    def test_expires_in_seconds_uses_given_timestamp(self):
        pairing_data = PairingData(
            v=1,
            type="codebridge-pair",
            server_id="server-1",
            name="Demo",
            local_url="http://127.0.0.1:8080",
            tunnel_url=None,
            pair_token="token-1",
            expires=1000,
        )

        self.assertEqual(pairing_data.expires_in_seconds(900), 100)
        self.assertEqual(pairing_data.expires_in_seconds(1005), -5)

    def test_to_qr_response_contains_expected_fields(self):
        pairing_data = PairingData(
            v=1,
            type="codebridge-pair",
            server_id="server-1",
            name="Demo",
            local_url="http://127.0.0.1:8080",
            tunnel_url="https://demo.tunnel",
            pair_token="token-1",
            expires=1000,
        )

        response = pairing_data.to_qr_response(now_ts=900)

        self.assertEqual(response["payload"]["pair_token"], "token-1")
        self.assertEqual(response["local_url"], "http://127.0.0.1:8080")
        self.assertEqual(response["tunnel_url"], "https://demo.tunnel")
        self.assertEqual(response["expires_in_seconds"], 100)
        self.assertTrue(response["qr_url"].startswith("codebridge://pair/"))

    def test_create_current_pairing_data_delegates_to_service(self):
        fake_pairing_service = MagicMock()
        expected_pairing_data = PairingData(
            v=1,
            type="codebridge-pair",
            server_id="server-1",
            name="Demo",
            local_url="http://127.0.0.1:8080",
            tunnel_url=None,
            pair_token="token-1",
            expires=1000,
        )
        fake_pairing_service.create_pairing_data.return_value = expected_pairing_data

        result = create_current_pairing_data(
            port=8080,
            server_name="Demo",
            tunnel_url=None,
            pairing_service=fake_pairing_service,
        )

        self.assertEqual(result, expected_pairing_data)
        fake_pairing_service.create_pairing_data.assert_called_once_with(
            port=8080,
            server_name="Demo",
            tunnel_url=None,
        )

    def test_build_current_pairing_data_result_success(self):
        expected_pairing_data = PairingData(
            v=1,
            type="codebridge-pair",
            server_id="server-1",
            name="Demo",
            local_url="http://127.0.0.1:8080",
            tunnel_url="https://demo.tunnel",
            pair_token="token-1",
            expires=1000,
        )

        with (
            patch("pairing.get_config", return_value=SimpleNamespace(port=8080, server_name="Demo")),
            patch("pairing.get_active_tunnel_url", return_value="https://demo.tunnel"),
            patch("pairing.create_current_pairing_data", return_value=expected_pairing_data) as mock_create,
        ):
            result = build_current_pairing_data_result()

        self.assertTrue(result.success)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.pairing_data, expected_pairing_data)
        mock_create.assert_called_once_with(
            port=8080,
            server_name="Demo",
            tunnel_url="https://demo.tunnel",
            pairing_service=None,
        )

    def test_build_current_pairing_data_result_returns_error_on_exception(self):
        with patch("pairing.get_config", side_effect=RuntimeError("broken")):
            result = build_current_pairing_data_result()

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 500)
        self.assertEqual(result.error, "Failed to build pairing data")

    def test_build_current_pairing_qr_result_propagates_pairing_data_error(self):
        with patch(
            "pairing.build_current_pairing_data_result",
            return_value=CurrentPairingDataResult(
                success=False,
                status_code=503,
                error="upstream failed",
            ),
        ):
            result = build_current_pairing_qr_result()

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 503)
        self.assertEqual(result.as_response_fields(), {"error": "upstream failed"})

    def test_build_current_pairing_page_context_result_uses_pairing_data_fields(self):
        fake_pairing_data = MagicMock()
        fake_pairing_data.to_qr_url.return_value = "codebridge://pair/demo"
        fake_pairing_data.local_url = "http://127.0.0.1:8080"
        fake_pairing_data.pair_token = "token-1"
        fake_pairing_data.expires_in_seconds.return_value = 180

        with patch(
            "pairing.build_current_pairing_data_result",
            return_value=CurrentPairingDataResult(
                success=True,
                status_code=200,
                pairing_data=fake_pairing_data,
            ),
        ):
            result = build_current_pairing_page_context_result()

        self.assertTrue(result.success)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.qr_url, "codebridge://pair/demo")
        self.assertEqual(result.local_url, "http://127.0.0.1:8080")
        self.assertEqual(result.pair_token, "token-1")
        self.assertEqual(result.expires_in_seconds, 180)
        fake_pairing_data.expires_in_seconds.assert_called_once_with()

    def test_pairing_page_context_result_to_render_context(self):
        result = PairingPageContextResult(
            success=True,
            status_code=200,
            qr_url="codebridge://pair/demo",
            local_url="http://127.0.0.1:8080",
            pair_token="token-1",
            expires_in_seconds=None,
        )

        self.assertEqual(
            result.to_render_context(),
            ("codebridge://pair/demo", "http://127.0.0.1:8080", "token-1", 0),
        )

    def test_pairing_page_context_result_to_render_context_returns_none_on_invalid_fields(self):
        result = PairingPageContextResult(
            success=True,
            status_code=200,
            qr_url=None,
            local_url="http://127.0.0.1:8080",
            pair_token="token-1",
            expires_in_seconds=10,
        )

        self.assertIsNone(result.to_render_context())

    def test_pairing_page_context_result_to_html_error_for_failed_result(self):
        result = PairingPageContextResult(
            success=False,
            status_code=503,
            error="upstream failed",
        )

        self.assertEqual(result.to_html_error(), ("upstream failed", 503))

    def test_pairing_page_context_result_to_html_error_for_invalid_success_result(self):
        result = PairingPageContextResult(
            success=True,
            status_code=200,
            qr_url=None,
            local_url="http://127.0.0.1:8080",
            pair_token="token-1",
            expires_in_seconds=10,
        )

        self.assertEqual(result.to_html_error(), ("Failed to build pairing page", 500))

    def test_pair_token_status_as_response_fields(self):
        status = PairTokenStatus(exists=False, used=False, expired=True)
        self.assertEqual(
            status.as_response_fields(),
            {"exists": False, "used": False, "expired": True},
        )

    def test_pairing_status_as_response_fields(self):
        status = PairingStatus(
            server_id="server-1",
            active_clients=1,
            pending_tokens=2,
            clients=[
                PairingClientStatus(
                    client_id="c1",
                    device_name="Pixel",
                    paired_at=1000.0,
                    last_used=1001.5,
                )
            ],
        )
        self.assertEqual(
            status.as_response_fields(),
            {
                "server_id": "server-1",
                "active_clients": 1,
                "pending_tokens": 2,
                "clients": [
                    {
                        "client_id": "c1",
                        "device_name": "Pixel",
                        "paired_at": 1000.0,
                        "last_used": 1001.5,
                    }
                ],
            },
        )

    def test_pairing_verify_token_result_as_response_fields(self):
        success_result = PairingVerifyTokenResult(
            success=True,
            status_code=200,
            api_key="k1",
            server_id="server-1",
            client_id="c1",
        )
        error_result = PairingVerifyTokenResult(
            success=False,
            status_code=400,
            error="Token expired",
        )

        self.assertEqual(
            success_result.as_response_fields(),
            {
                "success": True,
                "api_key": "k1",
                "server_id": "server-1",
                "client_id": "c1",
            },
        )
        self.assertEqual(error_result.as_response_fields(), {"error": "Token expired"})

    def test_pairing_revoke_result_as_response_fields(self):
        success_result = PairingRevokeResult(
            success=True,
            status_code=200,
            message="Client c1 revoked",
        )
        error_result = PairingRevokeResult(
            success=False,
            status_code=404,
            error="Client c1 not found",
        )

        self.assertEqual(
            success_result.as_response_fields(),
            {"success": True, "message": "Client c1 revoked"},
        )
        self.assertEqual(error_result.as_response_fields(), {"error": "Client c1 not found"})

    def test_get_pair_token_status_for_current_server_delegates_to_service(self):
        fake_pairing_service = MagicMock()
        fake_pairing_service.get_pair_token_status.return_value = PairTokenStatus(
            exists=True,
            used=False,
            expired=False,
        )

        result = get_pair_token_status_for_current_server(
            "token-1",
            pairing_service=fake_pairing_service,
        )

        self.assertEqual(result.as_response_fields(), {"exists": True, "used": False, "expired": False})
        fake_pairing_service.get_pair_token_status.assert_called_once_with("token-1")

    def test_get_pairing_status_for_current_server_delegates_to_service(self):
        fake_pairing_service = MagicMock()
        fake_pairing_service.get_pairing_status.return_value = PairingStatus(
            server_id="server-1",
            active_clients=2,
            pending_tokens=1,
            clients=[],
        )

        result = get_pairing_status_for_current_server(pairing_service=fake_pairing_service)

        self.assertEqual(
            result.as_response_fields(),
            {
                "server_id": "server-1",
                "active_clients": 2,
                "pending_tokens": 1,
                "clients": [],
            },
        )
        fake_pairing_service.get_pairing_status.assert_called_once_with()

    def test_revoke_paired_client_for_current_server_delegates_to_service(self):
        fake_pairing_service = MagicMock()
        fake_pairing_service.revoke_client.return_value = PairingRevokeResult(
            success=True,
            status_code=200,
            message="Client c1 revoked",
        )

        result = revoke_paired_client_for_current_server(
            "c1",
            pairing_service=fake_pairing_service,
        )

        self.assertEqual(result.as_response_fields(), {"success": True, "message": "Client c1 revoked"})
        fake_pairing_service.revoke_client.assert_called_once_with("c1")

    def test_pairing_service_revoke_client_removes_existing_key(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = PairingService(config_dir=Path(tmp_dir))
            service._api_keys["c1"] = {
                "api_key": "k1",
                "device_name": "Demo Device",
                "paired_at": 1000.0,
                "last_used": 1000.0,
            }

            result = service.revoke_client("c1")

            self.assertTrue(result.success)
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.message, "Client c1 revoked")
            self.assertNotIn("c1", service._api_keys)

    def test_pairing_service_revoke_client_returns_not_found(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = PairingService(config_dir=Path(tmp_dir))

            result = service.revoke_client("missing")

            self.assertFalse(result.success)
            self.assertEqual(result.status_code, 404)
            self.assertEqual(result.error, "Client missing not found")
